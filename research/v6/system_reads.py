"""V6-5 system reads: E-primary, Gate A, no-harm, claim 3, nested selection, the combined vote (docs/V6_RUNSHEET.md 0.2, 0.5, 0.8f).

    python -m research.v6.system_reads --validate-on-v5
    python -m research.v6.system_reads --finalists anchor panderm_b dinov2_b anchor384 --tag v64 \
           --fold0-tags anchor=v62p panderm_b=v63 dinov2_b=v63 anchor384=v63 [--checkpoint swad] [--rerun-reason "..."]

Written and frozen before any V6 run, so the analysis cannot be fitted to the numbers. CPU only. Every read below is on folds 1-4;
fold 0, the selection fold, enters only as selection rows for the nested read and fitting rows for the cross-fitted calibration.

**Systems.** V5 = `composite_tta_mean` in `results/v5/v5_oof_common_matrix.csv`, 0.7005 on these rows. V6 = the uniform soft vote of the
finalists, per image, each pooled over 24 views by V5's entropy weighting (`predict_tta.py`), with the 8-class head collapsed to 7.
Both sides are read before Dirichlet; the Dirichlet read is descriptive. No-TTA reads compare with `composite_mean` (0.6911).

**Estimator.** Paired, single level: resample `group_id` clusters with replacement, 2,000 times, and score both systems on the same
resample; the interval is the percentile 95% interval. V5's confirmation scheme also resampled seed pairs (`results/v5/
confirm_s42_s43_s44.json`), which cannot apply here because V6 has one seed per member; a V6 system is one realisation, which is why
E-primary's pAUC bar is the single-seed `bar_k1`.

**Verdicts, all fixed here.**
  * E-primary: pAUC delta > bar_k1 AND Macro-F1 delta > 0 with interval lower bound > 0. Both must pass, so it is one hypothesis.
  * Gate A: under-40 pAUC delta >= +0.050 with interval lower bound > 0.
  * No-harm: argmax escalation sensitivity and under-40 pAUC must not be significantly worse (interval upper bound >= 0). A system that
    fails is not deployed, whatever E-primary says.
  * Claim 3 (only if PanDerm-B is a finalist): adding it to the supervised finalists, under-40 pAUC gain minus all-age pAUC gain,
    passes if the interval lower bound is > 0. One contrast, so one hypothesis in its family.
  * Nested selection (0.5): replaces uniform only if it beats it on the pooled nested Macro-F1 by more than one bootstrap SD.
  * Combined vote: V5's three TTA models plus the V6 finalists, uniform per model; a declared secondary, no extra training.
  * Deployed system, decided by this rule and not after looking: the combined vote if it passes E-primary and no-harm and beats the V6 vote's
    Macro-F1 by more than one bootstrap SD; otherwise the V6 vote if it passes E-primary and no-harm; otherwise V5 stays deployed.

Before anything is read, `check_runs` confirms that every member and fold used one recipe, so a mistyped queue line cannot mix pools in the headline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml.evaluation.metrics import compute_metrics
from research import testguard
from research.agerule import lambda_rule as lr
from research.v5 import screen_gate as sg
from research.v6 import gate, head8, plan_freeze

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "results" / "v5" / "v5_oof_common_matrix.csv"
MANIFEST = REPO / "ml" / "data" / "manifest_v4.csv"
OUT_DIR = REPO / "results" / "v6"
E_PRIMARY = OUT_DIR / "e_primary.json"
CLAIM3 = OUT_DIR / "claim3.json"
NESTED = OUT_DIR / "nested_selection.json"
VALIDATION = OUT_DIR / "system_reads_validation.json"
RUN_DIR = OUT_DIR / "runs"
LEDGER = REPO / "research" / "experiments.csv"
V6_TTA_DIR = OUT_DIR / "preds_tta"
V5_SYSTEM, V5_NO_TTA, V5_CONTROL = "composite_tta_mean", "composite_mean", "control_mean"
V5_TTA_SEEDS = ("composite_tta_s42", "composite_tta_s43", "composite_tta_s44")
ESCALATING = tuple(head8.CLASSES_7.index(c) for c in ("akiec", "bcc", "mel"))
SUPERVISED = ("anchor", "anchor384", "anchor_h7", "anchor384_h7")
PANDERM = "panderm_b"
N_BOOT, SEED, MCID = 2000, 20261103, 0.050
NESTED_BAGS = 20
BOOT_METRICS = ("macro_f1", "pauc_all", "pauc_u40", "esc_sens")


@dataclass(frozen=True)
class Rows:
    image_id: np.ndarray
    fold: np.ndarray
    y: np.ndarray
    esc: np.ndarray
    u40: np.ndarray
    group: np.ndarray
    archive: np.ndarray

    def where(self, include_fold_0: bool = False) -> np.ndarray:
        return np.arange(len(self.y)) if include_fold_0 else np.where(self.fold != 0)[0]


def load_matrix(models: tuple[str, ...] = (V5_SYSTEM, V5_NO_TTA, V5_CONTROL, *V5_TTA_SEEDS)) -> tuple[Rows, dict[str, np.ndarray]]:
    columns = [f"prob_{m}_{c}" for m in models for c in head8.CLASSES_7]
    frame = pd.read_csv(MATRIX, usecols=["image_id", "group_id", "fold", "true_label", "age_band", "escalation_status", "archive", *columns],
                        low_memory=False, dtype={"image_id": str, "group_id": str})
    rows = Rows(frame["image_id"].to_numpy(), frame["fold"].to_numpy(), frame["true_label"].to_numpy(),
                frame["escalation_status"].to_numpy().astype(bool), (frame["age_band"] == "<40").to_numpy(),
                frame["group_id"].to_numpy(), frame["archive"].to_numpy())
    return rows, {m: frame[[f"prob_{m}_{c}" for c in head8.CLASSES_7]].to_numpy() for m in models}


def load_member(rows: Rows, label: str, tags: dict[int, str], seed: int = 42, checkpoint: str = "", tta: bool = True) -> dict[str, np.ndarray]:
    """{"p7": (N, 7), "p8": (N, 8) or absent}, aligned to `rows`, NaN on folds it has no file for. `checkpoint` is "" or "swad"."""
    suffix = "_".join(s for s in (checkpoint, "tta" if tta else "") if s)
    p7 = np.full((len(rows.y), 7), np.nan)
    p8 = np.full((len(rows.y), 8), np.nan)
    position = {image: i for i, image in enumerate(rows.image_id)}
    for fold, tag in tags.items():
        directory = V6_TTA_DIR if tta else gate.V6_PRED_DIR
        frame = pd.read_csv(gate.pred_path(label, seed, tag, directory, fold=fold, checkpoint=suffix), low_memory=False, dtype={"image_id": str})
        where = np.array([position[i] for i in frame["image_id"]])
        if set(rows.fold[where]) != {fold} or len(where) != int((rows.fold == fold).sum()):
            raise AssertionError(f"{label} fold {fold}: rows do not match the S71 partition")
        p7[where] = frame[[f"p_{c}" for c in head8.CLASSES_7]].to_numpy()
        if f"p8_{head8.CLASSES_8[0]}" in frame.columns:
            p8[where] = frame[[f"p8_{c}" for c in head8.CLASSES_8]].to_numpy()
    return {"p7": p7, "p8": p8}


def vote(systems: list[np.ndarray]) -> np.ndarray:
    return np.mean(systems, axis=0)


# ------------------------------------------------------------------ metrics
def _mf1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p.argmax(1), average="macro", labels=list(range(p.shape[1])), zero_division=0))


def _esc_score(p: np.ndarray) -> np.ndarray:
    return p[:, list(ESCALATING)].sum(1)


def _esc_sens(y: np.ndarray, esc: np.ndarray, p: np.ndarray) -> float:
    called = np.isin(p.argmax(1), ESCALATING)
    return float(called[esc].mean()) if esc.any() else float("nan")


def _pauc(esc: np.ndarray, p: np.ndarray) -> float:
    return float(sg.pauc(esc, _esc_score(p))) if len(np.unique(esc)) == 2 else float("nan")


def fast_endpoints(p: np.ndarray, rows: Rows, idx: np.ndarray) -> dict[str, float]:
    y, esc, young = rows.y[idx], rows.esc[idx], rows.u40[idx]
    return {"macro_f1": _mf1(y, p[idx]), "pauc_all": _pauc(esc, p[idx]), "pauc_u40": _pauc(esc[young], p[idx][young]),
            "esc_sens": _esc_sens(y, esc, p[idx])}


def endpoints(p: np.ndarray, rows: Rows, idx: np.ndarray) -> dict[str, float]:
    """The point estimates, through the repo's metric engine."""
    metrics = compute_metrics(rows.y[idx], p[idx].argmax(1), p[idx])
    out = fast_endpoints(p, rows, idx)
    if abs(out["macro_f1"] - float(metrics["macro_f1"])) > 1e-9:
        raise AssertionError("fast Macro-F1 differs from ml.evaluation.metrics.compute_metrics")
    return {**out, "balanced_accuracy": float(metrics["balanced_accuracy"])}


def clusters_of(rows: Rows, idx: np.ndarray) -> list[np.ndarray]:
    groups = pd.Series(idx).groupby(rows.group[idx]).groups
    return [idx[np.asarray(v)] for v in groups.values()]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rows: Rows, idx: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Delta a - b on `idx`, plus a `samples` array per metric for contrasts. Every resample scores both systems on the same rows."""
    rng = np.random.default_rng(seed)
    clusters = clusters_of(rows, idx)
    samples = {m: np.empty(n_boot) for m in BOOT_METRICS}
    for i in range(n_boot):
        pick = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        first, second = fast_endpoints(a, rows, pick), fast_endpoints(b, rows, pick)
        for m in BOOT_METRICS:
            samples[m][i] = first[m] - second[m]
    pa, pb = fast_endpoints(a, rows, idx), fast_endpoints(b, rows, idx)
    out = {m: {"a": pa[m], "b": pb[m], "delta": pa[m] - pb[m], "ci95": [float(np.nanpercentile(samples[m], 2.5)), float(np.nanpercentile(samples[m], 97.5))],
               "sd": float(np.nanstd(samples[m]))} for m in BOOT_METRICS}
    out["samples"] = samples
    return out


# ------------------------------------------------------------------ verdicts
def e_primary_verdict(read: dict, bar_pauc: float) -> dict:
    pauc_pass = read["pauc_all"]["delta"] > bar_pauc
    macro_pass = read["macro_f1"]["delta"] > 0 and read["macro_f1"]["ci95"][0] > 0
    return {"bar_pauc_k1": bar_pauc, "pauc_pass": bool(pauc_pass), "macro_f1_pass": bool(macro_pass), "passed": bool(pauc_pass and macro_pass)}


def gate_a_verdict(read: dict) -> dict:
    d = read["pauc_u40"]
    return {"mcid": MCID, "point": d["delta"], "ci95": d["ci95"], "passed": bool(d["delta"] >= MCID and d["ci95"][0] > 0)}


def no_harm_verdict(read: dict) -> dict:
    sens, young = read["esc_sens"]["ci95"][1] >= 0, read["pauc_u40"]["ci95"][1] >= 0
    return {"esc_sens_ok": bool(sens), "pauc_u40_ok": bool(young), "passed": bool(sens and young)}


def deployed_system(v6_primary: dict, v6_harm: dict, combined_primary: dict, combined_harm: dict, combined_vs_v6: dict) -> str:
    """Nothing replaces V5 without passing E-primary and no-harm."""
    margin = combined_vs_v6["macro_f1"]["delta"] > combined_vs_v6["macro_f1"]["sd"]
    if combined_primary["passed"] and combined_harm["passed"] and (margin or not v6_primary["passed"]):
        return "combined_vote"
    return "v6_vote" if v6_primary["passed"] and v6_harm["passed"] else "v5_system"


def base_name(label: str) -> str:
    """`panderm_b_h7` -> `panderm_b`: the same member with the 7-class head."""
    return label.removesuffix("_h7")


def run_json_name(label: str, fold: int, seed: int, tag: str) -> str:
    trunk = "_in22k" if label.startswith("anchor") else ""
    return f"{label}_f{fold}_s{seed}{trunk}_{tag}.json"


def check_runs(tags: dict[str, dict[int, str]], seed: int = 42, run_dir: Path | None = None) -> dict:
    """`tags` maps a label to {fold: tag}. Every run must be finished, at its full budget, under one freeze, with one head, pool and sampler."""
    run_dir = run_dir or RUN_DIR
    recipes, freezes, problems = set(), set(), []
    for label, folds in tags.items():
        for fold, tag in folds.items():
            path = run_dir / run_json_name(label, fold, seed, tag)
            if not path.exists():
                problems.append(f"missing {path.name}")
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            added = Path(run["added_train"]).as_posix() if run.get("added_train") else None
            recipes.add((run["head"], added, run["sampler"]))
            freezes.add(run.get("plan_freeze_sha256"))
            if run["smoke"] or run["epochs_run"] != run["epochs"] or run["optimizer_steps"] != run["step_budget"]:
                problems.append(f"{path.name} is not a complete run")
            if (run["image_size"] == 384) != ("384" in label):
                problems.append(f"{path.name} has image size {run['image_size']}")
    if len(recipes) > 1:
        problems.append(f"members or folds used different recipes (head, pool, sampler): {sorted(map(str, recipes))}")
    if len(freezes) > 1:
        problems.append(f"runs span {len(freezes)} plan-freeze hashes")
    if problems:
        raise SystemExit("refusing to read:\n  " + "\n  ".join(problems))
    return {"recipe": dict(zip(("head", "added_train", "sampler"), next(iter(recipes)))), "runs_checked": sum(len(v) for v in tags.values())}


def log_ledger(report: dict, ledger: Path | None = None) -> None:
    """One row, pruning this tag's earlier row, as `train_v5.write_ledger` does."""
    ledger = ledger or LEDGER
    v6, read = report["descriptive"]["v6_endpoints"], report["e_primary"]["read"]
    row = {"timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "session": "v6_5", "method": f"e_primary_{report['tag']}", "split": "folds_1_4_oof", "macro_f1": v6["macro_f1"],
           "balanced_accuracy": v6["balanced_accuracy"], "escalation_sens": v6["esc_sens"],
           "notes": (f"finalists={'+'.join(report['finalists'])} dMacroF1={read['macro_f1']['delta']:+.4f} dpAUC={read['pauc_all']['delta']:+.4f} "
                     f"e_primary={report['e_primary']['passed']} gate_a={report['gate_a']['passed']} deployed={report['deployed_system']}")}
    frame = pd.DataFrame([row])
    if ledger.is_file():
        old = pd.read_csv(ledger, low_memory=False)
        frame = pd.concat([old[~((old["session"] == row["session"]) & (old["method"] == row["method"]))], frame], ignore_index=True)
    frame.to_csv(ledger, index=False)


def claim3_read(base: np.ndarray, with_panderm: np.ndarray, rows: Rows, idx: np.ndarray, n_boot: int = N_BOOT) -> dict:
    read = paired_bootstrap(with_panderm, base, rows, idx, n_boot)
    contrast = read["samples"]["pauc_u40"] - read["samples"]["pauc_all"]
    low, high = float(np.nanpercentile(contrast, 2.5)), float(np.nanpercentile(contrast, 97.5))
    point = read["pauc_u40"]["delta"] - read["pauc_all"]["delta"]
    return {"gain_u40": read["pauc_u40"]["delta"], "gain_all": read["pauc_all"]["delta"], "contrast": point, "ci95": [low, high], "passed": bool(low > 0)}


# ------------------------------------------------------------------ nested selection
def forward_select(systems: list[np.ndarray], y: np.ndarray, esc: np.ndarray, young: np.ndarray, guard: float) -> np.ndarray:
    """Forward selection with replacement on one bag: maximise Macro-F1, keeping under-40 pAUC within `guard` of the uniform vote's."""
    floor = _pauc(esc[young], vote(systems)[young]) - guard
    counts = np.zeros(len(systems))
    total = np.zeros_like(systems[0])
    for _ in range(2 * len(systems)):
        best, best_f1 = None, -1.0
        for m, p in enumerate(systems):
            trial = (total + p) / (counts.sum() + 1)
            if _pauc(esc[young], trial[young]) < floor:
                continue
            f1 = _mf1(y, trial)
            if f1 > best_f1:
                best, best_f1 = m, f1
        if best is None:
            break
        counts[best] += 1
        total = total + systems[best]
    return counts / counts.sum() if counts.sum() else np.full(len(systems), 1 / len(systems))


def nested_selection(names: list[str], systems: list[np.ndarray], rows: Rows, guard: float, bags: int = NESTED_BAGS, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Section 0.5. For each outer fold, weights come only from the other folds, fold 0 included. A bag is a resample of lesion clusters."""
    rng = np.random.default_rng(seed)
    nested = np.full_like(systems[0], np.nan)
    weights_by_fold = {}
    for fold in (1, 2, 3, 4):
        selection = np.where(rows.fold != fold)[0]
        clusters = clusters_of(rows, selection)
        weights = np.zeros(len(systems))
        for _ in range(bags):
            pick = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            weights += forward_select([p[pick] for p in systems], rows.y[pick], rows.esc[pick], rows.u40[pick], guard)
        weights /= bags
        held = rows.fold == fold
        nested[held] = sum(w * p[held] for w, p in zip(weights, systems))
        weights_by_fold[fold] = {n: float(w) for n, w in zip(names, weights)}
    idx = rows.where()
    read = paired_bootstrap(nested, vote(systems), rows, idx, n_boot, seed)
    read_out = {k: v for k, v in read.items() if k != "samples"}
    return {"weights_by_fold": weights_by_fold, "read_vs_uniform": read_out, "replaces_uniform": bool(read["macro_f1"]["delta"] > read["macro_f1"]["sd"]),
            "guard_pauc_u40": guard, "bags": bags}


# ------------------------------------------------------------------ descriptive
def dirichlet_read(p: np.ndarray, rows: Rows, idx: np.ndarray) -> dict:
    """Cross-fitted Dirichlet over all five folds, read on `idx`, as `research.v5.refit_dirichlet` does. Descriptive."""
    calibrated = lr.crossfit_calibration(p, rows.y, rows.fold)
    return {k: fast_endpoints(calibrated, rows, idx)[k] for k in BOOT_METRICS}


def balanced_accuracy_8(p8: np.ndarray) -> float:
    """8-class balanced accuracy on the 5-fold OOF, the read beside Gessert's 71.7. Descriptive; needs an 8-class head."""
    manifest = pd.read_csv(MANIFEST, usecols=["image_id", "class_index_8"], dtype={"image_id": str}).set_index("image_id")["class_index_8"]
    rows, _ = load_matrix((V5_SYSTEM,))
    y8 = manifest.loc[rows.image_id].to_numpy()
    return float(np.mean([(p8[y8 == c].argmax(1) == c).mean() for c in np.unique(y8)]))


def per_archive_macro_f1(p: np.ndarray, rows: Rows, idx: np.ndarray) -> dict[str, float]:
    return {str(a): _mf1(rows.y[idx][rows.archive[idx] == a], p[idx][rows.archive[idx] == a]) for a in np.unique(rows.archive[idx])}


def clean(read: dict) -> dict:
    return {k: v for k, v in read.items() if k != "samples"}


# ------------------------------------------------------------------ runs
def validate_on_v5(n_boot: int = N_BOOT) -> dict:
    """Checks the machinery on numbers already known, with no V6 file: a system against itself is exactly zero, TTA against no TTA
    reproduces the matrix's own figures, and E-primary refuses a null."""
    rows, models = load_matrix()
    idx = rows.where()
    pair_sd = gate.load_pair_sd(check_lock=False)
    null = paired_bootstrap(models[V5_SYSTEM], models[V5_SYSTEM], rows, idx, n_boot)
    tta = paired_bootstrap(models[V5_SYSTEM], models[V5_NO_TTA], rows, idx, n_boot)
    report = {"what": "machinery check on V5 predictions only; no V6 file read", "rows": len(idx), "groups": int(len(np.unique(rows.group[idx]))),
              "v5_endpoints": endpoints(models[V5_SYSTEM], rows, idx), "null_delta": {m: null[m]["delta"] for m in BOOT_METRICS},
              "null_max_abs_sample": float(max(np.nanmax(np.abs(null["samples"][m])) for m in BOOT_METRICS)),
              "null_e_primary": e_primary_verdict(null, gate.bar(pair_sd["pauc_all"], 1)),
              "tta_minus_no_tta": clean(tta), "tta_e_primary": e_primary_verdict(tta, gate.bar(pair_sd["pauc_all"], 1)),
              "null_gate_a": gate_a_verdict(null), "null_no_harm": no_harm_verdict(null)}
    return report


def run_reads(args: argparse.Namespace) -> int:
    parsed = dict(pair.split("=") for pair in args.fold0_tags)
    if set(parsed) != set(args.finalists):
        raise SystemExit("--fold0-tags must name every finalist")
    if E_PRIMARY.exists() and not args.rerun_reason:
        raise SystemExit(f"{E_PRIMARY.name} exists: the read happens once. Pass --rerun-reason to repeat it, which is recorded.")
    freeze_sha = plan_freeze.require()
    recipe = check_runs({name: {0: parsed[name], **{f: args.tag for f in (1, 2, 3, 4)}} for name in args.finalists})
    rows, models = load_matrix()
    idx = rows.where()
    pair_sd = gate.load_pair_sd()
    bar_k1 = gate.bar(pair_sd["pauc_all"], 1)
    members_tta, members_plain = {}, {}
    for name in args.finalists:
        tags = {0: parsed[name], **{f: args.tag for f in (1, 2, 3, 4)}}
        members_tta[name] = load_member(rows, name, tags, checkpoint=args.checkpoint, tta=True)
        members_plain[name] = load_member(rows, name, tags, checkpoint=args.checkpoint, tta=False)
    v6_tta = vote([members_tta[n]["p7"] for n in args.finalists])
    v6_plain = vote([members_plain[n]["p7"] for n in args.finalists])
    combined = vote([*(members_tta[n]["p7"] for n in args.finalists), *(models[m] for m in V5_TTA_SEEDS)])

    read = paired_bootstrap(v6_tta, models[V5_SYSTEM], rows, idx)
    combined_read = paired_bootstrap(combined, models[V5_SYSTEM], rows, idx)
    combined_vs_v6 = paired_bootstrap(combined, v6_tta, rows, idx)
    plain_read = paired_bootstrap(v6_plain, models[V5_NO_TTA], rows, idx)
    control_read = paired_bootstrap(v6_plain, models[V5_CONTROL], rows, idx)
    primary, harm = e_primary_verdict(read, bar_k1), no_harm_verdict(read)
    combined_primary, combined_harm = e_primary_verdict(combined_read, bar_k1), no_harm_verdict(combined_read)
    report = {
        "executed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rerun_reason": args.rerun_reason or None, "plan_freeze_part0_sha256": freeze_sha,
        "finalists": args.finalists, "recipe_check": recipe, "tag": args.tag, "checkpoint": args.checkpoint or "last", "rows": len(idx), "fold_0_pooled": False, "test_read": False, "reserved_read": False,
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "scheme": "group_id clusters, paired, single level, percentile 95%"},
        "e_primary": {"read": clean(read), **primary}, "gate_a": gate_a_verdict(read), "no_harm": harm,
        "without_tta": {"read": clean(plain_read), **e_primary_verdict(plain_read, bar_k1)},
        "continuity_vs_in1k_control_no_tta": clean(control_read),
        "combined_vote": {"read": clean(combined_read), **combined_primary, "no_harm": combined_harm, "vs_v6_vote": clean(combined_vs_v6)},
        "deployed_system": deployed_system(primary, harm, combined_primary, combined_harm, combined_vs_v6),
        "descriptive": {"v6_endpoints": endpoints(v6_tta, rows, idx), "v5_endpoints": endpoints(models[V5_SYSTEM], rows, idx), "combined_endpoints": endpoints(combined, rows, idx),
                        "per_member_tta": {n: fast_endpoints(members_tta[n]["p7"], rows, idx) for n in args.finalists},
                        "per_archive_macro_f1": {"v6": per_archive_macro_f1(v6_tta, rows, idx), "v5": per_archive_macro_f1(models[V5_SYSTEM], rows, idx)},
                        "with_dirichlet": {"v6": dirichlet_read(v6_tta, rows, idx), "v5": dirichlet_read(models[V5_SYSTEM], rows, idx), "combined": dirichlet_read(combined, rows, idx)}},
    }
    p8 = [members_tta[n]["p8"] for n in args.finalists]
    if all(not np.isnan(p).any() for p in p8):
        report["descriptive"]["balanced_accuracy_8_all_folds_vs_gessert_71_7"] = balanced_accuracy_8(vote(p8))
    E_PRIMARY.write_text(json.dumps(report, indent=2), encoding="utf-8")
    panderm = next((n for n in args.finalists if base_name(n) == PANDERM), None)
    if panderm:
        base = [n for n in args.finalists if n in SUPERVISED]
        result = claim3_read(vote([members_tta[n]["p7"] for n in base]), vote([*(members_tta[n]["p7"] for n in base), members_tta[panderm]["p7"]]), rows, idx)
        CLAIM3.write_text(json.dumps({"base": base, "added": panderm, **result, "test_read": False}, indent=2), encoding="utf-8")
    else:
        CLAIM3.write_text(json.dumps({"tested": False, "reason": "PanDerm-B is not a finalist"}, indent=2), encoding="utf-8")
    nested = nested_selection(args.finalists, [members_tta[n]["p7"] for n in args.finalists], rows, gate.bar(pair_sd["pauc_u40"], 1))
    NESTED.write_text(json.dumps(nested, indent=2), encoding="utf-8")
    log_ledger(report)
    print(f"E-primary {'PASS' if primary['passed'] else 'FAIL'}: dMacro-F1 {read['macro_f1']['delta']:+.4f} {read['macro_f1']['ci95']}, dpAUC {read['pauc_all']['delta']:+.4f} (bar {bar_k1:.4f})")
    print(f"Gate A {'PASS' if report['gate_a']['passed'] else 'FAIL'}: {read['pauc_u40']['delta']:+.4f} {read['pauc_u40']['ci95']}; no-harm {'ok' if harm['passed'] else 'FAILED'}; deployed: {report['deployed_system']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate-on-v5", action="store_true")
    parser.add_argument("--finalists", nargs="+", default=[])
    parser.add_argument("--tag", default="")
    parser.add_argument("--fold0-tags", nargs="+", default=[], metavar="MEMBER=TAG")
    parser.add_argument("--checkpoint", choices=("", "swad"), default="", help="swad if V6-1 adopted averaging")
    parser.add_argument("--rerun-reason", default="")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V6 system reads: S71 OOF rows only")
    if args.validate_on_v5:
        report = validate_on_v5()
        VALIDATION.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("v5_endpoints", "null_delta", "null_max_abs_sample", "null_e_primary")}, indent=1))
        return 0
    if not args.finalists or not args.tag:
        raise SystemExit("pass --finalists and --tag, or --validate-on-v5")
    return run_reads(args)


if __name__ == "__main__":
    raise SystemExit(main())
