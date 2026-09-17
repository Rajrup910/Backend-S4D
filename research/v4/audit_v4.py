"""V4 audit, made repeatable: integrity checks plus independent recomputation of the headlines.

The 2026-09-17 audit (CHANGELOG, "V4 audit") was run by hand. This module re-runs it so that any
later session -- or the 11:00 routine -- can prove nothing has drifted:

    integrity    HAM test receipt still at 2 executions; every V4 reserved receipt has one
                 completed execution per stage and no rerun reason; plan hashes and receipt item
                 hashes match the files on disk; manifest_v4 splits share no group, lesion or
                 duplicate cluster; ham_val/ham_test equal split_v1; the six published checkpoints
                 are byte-identical; `frozen_params --selftest` passes; the ledger holds no
                 conflicting duplicate rows.
    headlines    recomputed with sklearn (`f1_score`, `roc_auc_score(max_fpr=0.2)`), not with the
                 runners' own metric code: S54 Gate B, S56 primary, S57b/S65 A1 under-40, S58 H0/H1,
                 S64 under-40 AUC, S65 primary, S66 A0/P0 on BCN, S67 stage 1.
    colour       the fixed `shades_of_grey` is the identity on a neutral image.

**Nothing new is computed from the reserved cohort or the test split.** The recomputations
re-derive quantities already recorded in `results/`, from the frozen prediction and feature files
those sessions wrote; no receipt is opened. `--check` writes nothing.

`--emit` writes the two audit-derived artefacts that the V4 write-up cites (Hard Rule 4), both from
development data only:

    results/v4/audit/under40_case_mix.json   escalating class mix and melanoma-only AUC per band (OOF)
    results/v4/audit/colour_check.json       brightness / clipping of old vs fixed shades_of_grey
                                             on 60 training images

`--rescore` (off by default, CPU, imports torch, minutes) re-scores one banked V4 checkpoint on V4
val and compares it with the run's training log.

    $py -m research.v4.audit_v4 --check
    $py -m research.v4.audit_v4 --emit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RESULTS = REPO_ROOT / "results"
V4 = RESULTS / "v4"
AUDIT_DIR = V4 / "audit"
LEDGER = REPO_ROOT / "research" / "experiments.csv"
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
SPLIT_V1 = REPO_ROOT / "ml" / "configs" / "splits" / "split_v1.csv"
TOL = 1e-9

RECEIPTS = {  # receipt -> plan file (None when the receipt pins a different artefact)
    "s54/reserved_receipt.json": "s54_plan.json",
    "s56/reserved_receipt.json": "s56_plan.json",
    "s57a/reserved_audit_receipt.json": None,
    "s57b/reserved_receipt.json": "s57b_plan.json",
    "s58/reserved_receipt.json": "s58_plan.json",
    "s65/reserved_receipt.json": "s65_plan.json",
    "s66/reserved_receipt.json": "s66_plan.json",
    "s59/reserved_receipt.json": "s59_plan.json",
}
PLAN_PREFIXES = {
    "backbone_probe_plan.json": "09d5795ef02eca14",
    "recipe_ladder_plan.json": "eff9d79f36f351ff",
    "s67/stage1_plan.json": "20405144a9884445",
    "s53r_plan.json": "2ef2e91bf5db6df5",
    "s59_plan.json": "e7ebc3f59402f0e9",
    "analysis_plan_v4.json": "bd320319e1d9fff6",
}
RESERVED_READS = 8  # receipted reserved reads across V4 (S51's probe predates receipts)
ESC = ["p_akiec", "p_bcc", "p_mel"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append(("PASS" if ok else "FAIL", name, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)

    def warn(self, name: str, detail: str) -> None:
        self.rows.append(("WARN", name, detail))
        print(f"  [WARN] {name} -- {detail}", flush=True)

    def guarded(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # a check that cannot run is a failed check, never a silent skip
            self.check(name, False, f"could not run: {type(exc).__name__}: {exc}")

    @property
    def failed(self) -> int:
        return sum(r[0] == "FAIL" for r in self.rows)


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def pauc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s, max_fpr=0.2))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, labels=range(7), average="macro", zero_division=0))


# ============================================================================ integrity
def check_receipts(rep: Report) -> None:
    test = json.loads((RESULTS / "test_pass_receipt.json").read_text(encoding="utf-8"))
    rep.check("HAM test receipt still at 2 executions", test.get("n_executions") == 2,
              f"n_executions={test.get('n_executions')}")
    for rel, plan in RECEIPTS.items():
        path = V4 / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        execs = data["executions"]
        stages = [e.get("stage", "single") for e in execs]
        ok = (all(e["status"] == "completed" for e in execs) and len(stages) == len(set(stages))
              and not any(e.get("rerun_reason") for e in execs))
        rep.check(f"receipt {rel}: one completed execution per stage, no rerun reason", ok,
                  f"{len(execs)} execution(s), stages {stages}")
        if plan is not None:
            rep.check(f"receipt {rel}: plan hash matches {plan}",
                      data["plan_sha256"] == sha256(V4 / plan))
        bad = []
        for e in execs:
            for key, value in (e.get("items") or {}).items():
                item, digest = ((value["path"], value["sha256"]) if isinstance(value, dict)
                                else (key, value))
                if not (REPO_ROOT / item).is_file() or sha256(REPO_ROOT / item) != digest:
                    bad.append(item)
        rep.check(f"receipt {rel}: item hashes match", not bad, ", ".join(bad[:3]))
    for rel, prefix in PLAN_PREFIXES.items():
        path = V4 / rel
        rep.check(f"plan {rel} hash {prefix}", path.is_file() and sha256(path).startswith(prefix),
                  sha256(path)[:16] if path.is_file() else "missing")
    found = sorted(p.relative_to(V4).as_posix() for p in V4.glob("*/reserved*receipt.json"))
    rep.check(f"{RESERVED_READS} receipted reserved reads, all listed in RECEIPTS",
              len(found) == RESERVED_READS and set(found) == set(RECEIPTS), ", ".join(found))


def check_splits(rep: Report) -> None:
    m = pd.read_csv(MANIFEST, low_memory=False)
    for col in ("group_id", "lesion_id", "dup_cluster"):
        sets = {s: set(m.loc[(m.split == s) & m[col].notna(), col].astype(str))
                for s in m.split.unique()}
        overlaps = [(a, b, len(sets[a] & sets[b])) for a in sets for b in sets
                    if a < b and sets[a] & sets[b]]
        rep.check(f"manifest_v4: no {col} shared between splits", not overlaps, str(overlaps[:3]))
    rep.check("reserved holds no HAM image", not (m.loc[m.split == "reserved", "archive"] == "ham").any())
    v1 = pd.read_csv(SPLIT_V1)
    for v4_split, v1_split in (("ham_val", "val"), ("ham_test", "test")):
        same = (set(m.loc[m.split == v4_split, "image_id"].astype(str))
                == set(v1.loc[v1.split == v1_split, "image_id"].astype(str)))
        rep.check(f"{v4_split} identical to split_v1 {v1_split}", same)


def check_frozen(rep: Report) -> None:
    for name, argv in (("six published checkpoints byte-identical",
                        ["-m", "research.v2.frozen_checkpoints", "--check"]),
                       ("frozen_params --selftest", ["-m", "research.external.frozen_params",
                                                     "--selftest"])):
        proc = subprocess.run([PY, *argv], cwd=REPO_ROOT, capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        rep.check(name, proc.returncode == 0, tail[:120])


def check_ledger(rep: Report) -> None:
    ledger = pd.read_csv(LEDGER, low_memory=False)
    v4 = ledger[ledger["session"].astype(str).str.startswith("v4_")]
    value_cols = [c for c in v4.columns if c not in ("timestamp", "notes")]
    dup = v4[v4.duplicated(["session", "method", "split"], keep=False)]
    conflicting = [key for key, grp in dup.groupby(["session", "method", "split"])
                   if len(grp[value_cols].astype(str).drop_duplicates()) > 1]
    identical = dup.groupby(["session", "method", "split"]).ngroups - len(conflicting)
    rep.check("ledger: no conflicting duplicate V4 rows", not conflicting, str(conflicting[:3]))
    if identical:
        rep.warn("ledger: identical duplicate V4 rows", f"{identical} group(s)")


def check_colour(rep: Report) -> None:
    from PIL import Image

    from research.v4.colour import shades_of_grey

    grey = Image.fromarray(np.full((16, 16, 3), 150, dtype=np.uint8))
    delta = np.abs(np.asarray(shades_of_grey(grey), dtype=float) - 150).max()
    rep.check("shades_of_grey is the identity on a neutral image (sqrt(3) bug fixed)", delta <= 1,
              f"max |delta| {delta:.1f}")


# ============================================================================ headlines
def reserved_v1() -> tuple[pd.DataFrame, np.ndarray]:
    from research.v4 import s54_gate as g

    panel = g.build_panel("reserved")
    probs, covered, _ = g.load_v1_probs(panel, smoke=False)
    assert covered.all()
    return panel, probs


def lambda_pred(probs: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """The frozen 3-band rule, through its sole loader (project rule: never type a lambda)."""
    from research.external import frozen_params as fp

    return fp.apply_age_rule(probs, bands=bands)


def check_s54(rep: Report) -> None:
    gate = json.loads((V4 / "s54/s54_gate.json").read_text(encoding="utf-8"))
    recorded = {(c["checkpoint"], c["endpoint"]): c["delta"] for c in gate["contrasts"]
                if c["seed"] == "mean" and c["contrast"].startswith("B")}
    for kind in ("last", "best"):
        dm, dp = [], []
        for seed in (42, 43, 44):
            frames = [pd.read_csv(V4 / f"s54/predictions/{arm}_pooled_s{seed}_{kind}.csv")
                      .sort_values("image_id").reset_index(drop=True) for arm in ("R1+R4", "R0")]
            vals = []
            for d in frames:
                u = d[d.age_band == "<40"]
                vals.append((macro_f1(d.y_true, d[[c for c in d if c.startswith("p_")]].to_numpy().argmax(1)),
                             pauc(u.y_esc, u[ESC].sum(axis=1))))
            dm.append(vals[0][0] - vals[1][0])
            dp.append(vals[0][1] - vals[1][1])
        rep.check(f"S54 Gate B {kind}: Macro-F1 {np.mean(dm):+.4f}, <40 pAUC {np.mean(dp):+.4f}",
                  close(np.mean(dm), recorded[(kind, "macro_f1")])
                  and close(np.mean(dp), recorded[(kind, "pauc")]))


def check_s56(rep: Report, panel: pd.DataFrame, probs: np.ndarray) -> None:
    pols = json.loads((V4 / "s56/policies_oof.json").read_text(encoding="utf-8"))
    report = json.loads((V4 / "s56/s56_report.json").read_text(encoding="utf-8"))
    u = 1 - probs.max(1)
    bands = panel.age_band.to_numpy()
    y = panel.y_esc.to_numpy()
    pred_esc = np.isin(probs.argmax(1), [0, 1, 4])
    m = bands == "<40"

    def sens(p: dict) -> float:
        tau = np.minimum(np.array([p["tau_band"].get(b, np.inf) for b in bands]), p["tau_global"])
        return float((pred_esc | (u >= tau))[m & y].mean())

    delta = sens(pols["band_0.2"]) - sens(pols["global_0.2"])
    rep.check(f"S56 primary <40 system sensitivity band-global {delta:+.4f}",
              close(delta, report["primary"]["delta"]))
    statuses = pd.Series([r["status"] for r in report["floor_transfer"]]).value_counts().to_dict()
    rep.check(f"S56 floor transfer counts {statuses}", statuses.get("met", 0) == 0,
              "documentation must quote these counts")


def check_a1(rep: Report, panel: pd.DataFrame, probs: np.ndarray) -> None:
    bands = panel.age_band.to_numpy()
    y = panel.y_esc.to_numpy()
    pe = np.isin(lambda_pred(probs, bands), [0, 1, 4])
    m = bands == "<40"
    sens, ref = float(pe[m & y].mean()), float(pe[m].mean())
    a = pd.read_csv(V4 / "s57b/ablation_reserved.csv").set_index("arm").loc["A1"]
    rep.check(f"S57b A1 <40 sensitivity {sens:.4f} referral {ref:.4f}",
              close(sens, a["sensitivity|<40"], 1e-6) and close(ref, a["referral_rate|<40"], 1e-6))
    s65 = pd.read_csv(V4 / "s65/frontier_reserved.csv")
    lam = s65[(s65.arm == "LAM") & (s65.band == "<40")].iloc[0]
    rep.check("S65 LAM arm equals S57b A1 under 40", close(lam.system_sens, sens, 1e-6))

    bcn = (panel.archive == "bcn20000").to_numpy()
    arms = pd.read_csv(V4 / "s66/reserved_arms.csv")
    arms = arms[arms.centre == "bcn20000"].set_index("arm")
    a0 = np.isin(probs.argmax(1), [0, 1, 4])
    for name, p in (("A0", a0), ("P0", pe)):
        mm = bcn & m
        s = float(p[mm & y].mean())
        rep.check(f"S66 BCN {name} <40 sensitivity {s:.4f}", close(s, arms.loc[name, "sensitivity|<40"], 1e-6))


def check_s65(rep: Report) -> None:
    rpt = json.loads((V4 / "s65/s65_report.json").read_text(encoding="utf-8"))
    f = pd.read_csv(V4 / "s65/frontier_reserved.csv")

    def at(arm: str, budget: float, band: str) -> float:
        return float(f[(f.arm == arm) & (f.budget == budget) & (f.band == band)].system_sens.iloc[0])

    delta = at("COMB", 0.2, "<40") - at("S56", 0.2, "<40")
    rep.check(f"S65 primary COMB-S56 <40 {delta:+.4f} (frontier vs report)",
              close(delta, rpt["primary"]["delta"]))
    s56 = pd.read_csv(V4 / "s56/frontier_reserved.csv")
    ref = s56[(s56.role == "primary") & (s56.arm == "band") & (s56.budget == 0.2) & (s56.band == "<40")]
    rep.check("S65's S56 arm reproduces S56 under 40", close(at("S56", 0.2, "<40"), ref.system_sens.iloc[0]))


def check_s58(rep: Report, panel: pd.DataFrame) -> None:
    from research.v4.s58_front_end import LinearHead

    state = np.load(V4 / "s58/fit_state.npz")
    store = np.load(REPO_ROOT / "research/v4/features/convnext_tiny_s58_reserved.npz")
    idx = pd.Index(store["image_ids"].astype(str)).get_indexer(panel.image_id)
    assert (idx >= 0).all()
    x = store["features"][idx].astype(np.float64)
    marg = pd.read_csv(V4 / "s58/stage3_marginals.csv")
    marg = marg[marg.subset == "all"].set_index("model")
    u = (panel.age_band == "<40").to_numpy()
    for head, model in (("ham", "H0_ham_head"), ("pooled", "H1_pooled_head")):
        p = LinearHead.from_arrays(state, head).predict_proba(x)
        mf, pa = macro_f1(panel.y7, p.argmax(1)), pauc(panel.y_esc[u], p[u][:, [0, 1, 4]].sum(1))
        rep.check(f"S58 {model}: Macro-F1 {mf:.4f}, <40 pAUC {pa:.4f}",
                  close(mf, marg.loc[model, "macro_f1"], 1e-6) and close(pa, marg.loc[model, "pauc_u40"], 1e-6))


def check_s67(rep: Report) -> None:
    probes = json.loads((V4 / "s67/probes.json").read_text(encoding="utf-8"))
    d = pd.read_csv(V4 / "s67/stage1_scores.csv")
    u = d[d.age_band == "<40"]
    for arm, rec in probes["marginals"].items():
        value = pauc(u.y_esc, u[arm])
        rep.check(f"S67 {arm} <40 pAUC {value:.4f}", close(value, rec["u40_pauc"]))


def check_s59(rep: Report, panel: pd.DataFrame, probs: np.ndarray) -> None:
    """The deployed stack (S56 band_0.2 on argmax) re-applied to the frozen V1 probabilities."""
    report = json.loads((V4 / "s59/s59_report.json").read_text(encoding="utf-8"))
    pol = json.loads((V4 / "s56/policies_oof.json").read_text(encoding="utf-8"))["band_0.2"]
    rep.check("S59 deployed stack is S56@0.2", report["stack"] == "S56"
              and report["contract"]["deployed_arm"] == "S56@0.2")
    bands = panel.age_band.to_numpy()
    y = panel.y_esc.to_numpy()
    tau = np.minimum(np.array([pol["tau_band"].get(b, np.inf) for b in bands]), pol["tau_global"])
    referred = (1 - probs.max(1)) >= tau
    caught = np.isin(probs.argmax(1), [0, 1, 4]) | referred
    terms = {t["term"]: t for t in report["contract"]["terms"]}
    for band in ("<40", "40-59", "60+"):
        value = float(caught[(bands == band) & y].mean())
        rep.check(f"S59 system sensitivity {band} {value:.4f}",
                  close(value, terms[f"sensitivity[{band}]"]["value"], 1e-9))
    ref = float(referred.mean())
    rep.check(f"S59 referral rate {ref:.4f}", close(ref, terms["referral_rate[ALL]"]["value"], 1e-9))
    # `contract.coverage` is the share decided without referral, NOT a joint bootstrap pass rate.
    rep.check(f"S59 contract.coverage {report['contract']['coverage']:.4f} == 1 - referral rate",
              close(report["contract"]["coverage"], 1 - ref, 1e-9))
    rep.check("S59 joint verdict CONTRACT_FAILS with every term NOT_MET",
              report["contract"]["joint"] == "CONTRACT_FAILS"
              and all(t["status"] == "NOT_MET" for t in terms.values()))
    v1 = float(np.isin(probs.argmax(1), [0, 1, 4])[y].mean())
    c = pd.read_csv(V4 / "s59/contrasts_reserved.csv")
    rec = c[(c.label == "deployed_vs_V1") & (c.band == "ALL") & (c.stat == "system_sens")].iloc[0]
    rep.check(f"S59 deployed - V1 all-ages system sensitivity {float(caught[y].mean()) - v1:+.4f}",
              close(float(caught[y].mean()) - v1, rec.delta, 1e-9))


def check_s53r(rep: Report) -> None:
    """Per-rung final-epoch deltas re-derived from the banked run JSONs, paired by seed."""
    report = json.loads((V4 / "s53r/s53r_report.json").read_text(encoding="utf-8"))

    def final(rung: str, seed: int) -> float:
        tagged = V4 / f"recipe_runs/{rung}_ham_only_s{seed}_rerun.json"
        path = tagged if tagged.is_file() else V4 / f"recipe_runs/{rung}_ham_only_s{seed}.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["epochs_run"] == len(run["history"]) and run["history"][-1]["epoch"] == run["epochs_run"]
        return float(run["final_val_macro_f1"])

    control = [final("R0", s) for s in (42, 43, 44)]
    rep.check(f"S53r control seed SD {np.std(control, ddof=1):.4f}",
              close(np.std(control, ddof=1), report["control"]["seed_sd"], 1e-9))
    for rung, rec in report["verdicts"].items():
        deltas = [final(rung, s) - c for s, c in zip((42, 43, 44), control)]
        rep.check(f"S53r {rung} mean final delta {np.mean(deltas):+.4f} ({rec['reading']})",
                  close(np.mean(deltas), rec["final_macro_f1"]["mean"], 1e-9))


def check_s64(rep: Report) -> None:
    from sklearn.metrics import roc_auc_score

    from research.v4.lambda_age import dev_panel

    ceiling = json.loads((V4 / "s64/ceiling.json").read_text(encoding="utf-8"))
    panel = dev_panel()
    probs = panel.probs
    esc = [0, 1, 4]
    ben = [c for c in range(7) if c not in esc]
    d = probs[:, esc].max(1) - probs[:, ben].max(1)
    y = np.isin(panel.y, esc)
    for band, rec in ceiling["bands"].items():
        m = np.asarray(panel.bands).astype(str) == band
        value = float(roc_auc_score(y[m], d[m]))
        rep.check(f"S64 {band} AUC(d) {value:.4f}", close(value, rec["auc_d"], 1e-9))


# ============================================================================ emit
def emit(rep: Report) -> None:
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    from research.v4 import colour
    from research.v4.lambda_age import dev_panel
    from research.v4.recipe import IMAGE_DIR

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    panel = dev_panel()
    probs, y7 = panel.probs, np.asarray(panel.y)
    bands = np.asarray(panel.bands).astype(str)
    codes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    esc = [0, 1, 4]
    mass = probs[:, esc].sum(1)
    y_esc = np.isin(y7, esc)
    out = {"source": "HAM OOF, 6-CNN 24-view TTA soft-vote, S5 cross-fitted Dirichlet "
                     "(research.v4.lambda_age.dev_panel); score = escalation mass",
           "reserved_read": False, "test_read": False, "bands": {}}
    for band in ("<40", "40-59", "60+"):
        m = bands == band
        mel = m & ((y7 == 4) | ~y_esc)
        mel_nv = m & np.isin(y7, [4, 5])
        out["bands"][band] = {
            "n": int(m.sum()),
            "escalating_mix": {codes[c]: int(((y7 == c) & m).sum()) for c in esc},
            "auc_all_escalating_vs_benign": float(roc_auc_score(y_esc[m], mass[m])),
            "auc_melanoma_vs_benign": float(roc_auc_score((y7 == 4)[mel], mass[mel])),
            "auc_melanoma_vs_nevus": float(roc_auc_score((y7 == 4)[mel_nv], mass[mel_nv]))}
    path = AUDIT_DIR / "under40_case_mix.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    rep.check(f"emitted {path.relative_to(REPO_ROOT).as_posix()}", True,
              ", ".join(f"{b} mel-vs-benign {v['auc_melanoma_vs_benign']:.3f}" for b, v in out["bands"].items()))

    def buggy(image: Image.Image) -> Image.Image:
        """The pre-fix gain, ||e|| / e_c, kept here only to measure what R2 and S49 trained on."""
        a = np.asarray(image.convert("RGB"), dtype=np.float64)
        e = np.power(np.mean(np.power(a, colour.MINKOWSKI_P), axis=(0, 1)), 1.0 / colour.MINKOWSKI_P)
        e = np.where(e < 1e-6, 1e-6, e)
        return Image.fromarray(np.clip(a * (np.linalg.norm(e) / e), 0, 255).astype(np.uint8))

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    ids = manifest[manifest.split == "train"].sample(60, random_state=0).image_id
    rows = []
    for image_id in ids:
        with Image.open(IMAGE_DIR / f"{image_id}.jpg") as image:
            image = image.convert("RGB")
            arrays = {"input": np.asarray(image, dtype=float),
                      "old_buggy": np.asarray(buggy(image), dtype=float),
                      "fixed": np.asarray(colour.shades_of_grey(image), dtype=float)}
        rows.append({f"{k}_{s}": (v.mean() if s == "mean" else (v >= 254.5).any(axis=2).mean())
                     for k, v in arrays.items() for s in ("mean", "clipped")})
    frame = pd.DataFrame(rows)
    summary = {"images": "60 manifest_v4 train images, random_state=0",
               "clipped": "fraction of pixels with any channel >= 255",
               "mean": {c: float(frame[c].mean()) for c in frame},
               "median": {c: float(frame[c].median()) for c in frame}}
    path = AUDIT_DIR / "colour_check.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rep.check(f"emitted {path.relative_to(REPO_ROOT).as_posix()}", True,
              f"clipped median old {summary['median']['old_buggy_clipped']:.3f} "
              f"fixed {summary['median']['fixed_clipped']:.3f}")


    # S49's "conventional rule" table (either hash within 6) had no file behind it.
    from research.v4 import dedupe

    frame = dedupe.load_hashes()
    frame = frame[((frame["dhash"] != 0) | (frame["phash"] != 0)).to_numpy()]
    rules = {}
    for radius in (3, 6):
        d_pairs = dedupe._pairs_within(frame["dhash"].to_numpy(), radius)
        p_pairs = dedupe._pairs_within(frame["phash"].to_numpy(), radius)
        either = d_pairs | p_pairs
        chance = dedupe.chance_rate(frame, radius)
        union = dedupe._Union(len(frame))
        for a, b in either:
            union.union(a, b)
        _, sizes = np.unique([union.find(i) for i in range(len(frame))], return_counts=True)
        rules[f"either_within_{radius}"] = {
            "pairs_dhash": len(d_pairs), "pairs_phash": len(p_pairs), "pairs_either": len(either),
            "chance_rate_dhash": chance["rate_dhash"], "chance_rate_phash": chance["rate_phash"],
            "expected_dhash": chance["expected_dhash"], "expected_phash": chance["expected_phash"],
            "expected_either": chance["expected_either"],
            "enrichment_either_over_chance": len(either) / chance["expected_either"],
            "largest_component": int(sizes.max()), "largest_component_share": float(sizes.max() / len(frame))}
    path = AUDIT_DIR / "dedupe_conventional_rule.json"
    path.write_text(json.dumps({"source": "research.v4.dedupe cache (chance_rate seed 0), readable images",
                                "n_images": int(len(frame)), "rules": rules}, indent=2), encoding="utf-8")
    six = rules["either_within_6"]
    rep.check(f"emitted {path.relative_to(REPO_ROOT).as_posix()}", True,
              f"r6 either: {six['pairs_dhash']} dHash pairs, component {six['largest_component']}, "
              f"enrichment {six['enrichment_either_over_chance']:.2f}x")


def rescore(rep: Report) -> None:
    proc = subprocess.run([PY, "-m", "scratchpad.audit_v4.rescore_val", "control", "44", "best"],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("AUDIT")), proc.stderr[-200:])
    rep.check("CPU re-score of R0_pooled_s44_best matches its training log",
              proc.returncode == 0 and "diff +0.00e+00" in line, line)


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="integrity + headline recomputation")
    parser.add_argument("--emit", action="store_true", help="write results/v4/audit/*.json")
    parser.add_argument("--rescore", action="store_true", help="CPU checkpoint re-score (slow)")
    args = parser.parse_args(argv)
    if not (args.check or args.emit or args.rescore):
        args.check = True
    rep = Report()
    if args.check:
        print("integrity")
        for name, fn in (("receipts", lambda: check_receipts(rep)), ("splits", lambda: check_splits(rep)),
                         ("frozen", lambda: check_frozen(rep)), ("ledger", lambda: check_ledger(rep)),
                         ("colour", lambda: check_colour(rep))):
            rep.guarded(name, fn)
        print("headlines")
        rep.guarded("S54", lambda: check_s54(rep))
        panel, probs = reserved_v1()
        rep.guarded("S56", lambda: check_s56(rep, panel, probs))
        rep.guarded("S57b/S65/S66", lambda: check_a1(rep, panel, probs))
        rep.guarded("S65", lambda: check_s65(rep))
        rep.guarded("S58", lambda: check_s58(rep, panel))
        rep.guarded("S67", lambda: check_s67(rep))
        rep.guarded("S64", lambda: check_s64(rep))
        rep.guarded("S59", lambda: check_s59(rep, panel, probs))
        rep.guarded("S53r", lambda: check_s53r(rep))
    if args.emit:
        print("emit")
        rep.guarded("emit", lambda: emit(rep))
    if args.rescore:
        rep.guarded("rescore", lambda: rescore(rep))
    passed = sum(r[0] == "PASS" for r in rep.rows)
    print(f"\naudit_v4: {passed} passed, {rep.failed} failed, "
          f"{sum(r[0] == 'WARN' for r in rep.rows)} warning(s)")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
