"""V6-6, the one receipted HIBA read (docs/V6_RUNSHEET.md 0.4; §A9.1 C1, C2, C6; template `research.v5.s84_milk10k`).

    python -m research.v6.hiba_read --dry-run     # preconditions only: no image, no label, no model, no receipt
    python -m research.v6.hiba_read --smoke       # the analysis path on synthetic data -> results/v6/hiba/smoke/, no receipt
    python -m research.v6.hiba_read --prepare     # label-free: cohort, image download, hash links. No model, no label, no receipt
    python -m research.v6.hiba_read --execute     # THE read. Refuses a second execution (receipt n_executions <= 1)
    python -m research.v6.hiba_read --execute --resume   # only after a crash before any label was read

NOT RUN by its author. `results/v6/hiba_receipt.json` holds `n_executions`; it is incremented BEFORE any image is scored, is never reset, and there is no rerun
flag: a repeat read is the owner's decision outside this script. `--resume` continues the same execution, and only while `label_read_count` is 0. Labels are
fetched in exactly one place (`read_labels`), which first increments `label_read_count`. The receipt pins the hash of every frozen input.

Stages:
  prepare  CPU and network, NO labels. Dermoscopic rows, one image per lesion, dropped if linked by id or hash to any training source. The metadata cache also holds a
           diagnosis column, and it is never loaded here (`COHORT_COLUMNS`). Images go under data/external/hiba/ (git-ignored).
  infer    GPU, NO labels. Every model, 24-view TTA at its own input size: V6 finalists (averaged weights, 5 folds each), V5 (15 composites), V1 (six HAM-only CNNs).
           One file per model under results/v6/hiba/inference/, so a crash loses at most one model.
  analyse  CPU. Every decision is fixed from probabilities and age bands first. Then the labels are read once.

Systems: V6 vote and combined vote (`research.v6.refit_system`), V5 (`research.v5.refit_*`) and V1 (frozen stack), each with its own Dirichlet map and S56 band policy.
Budget lock (C2): each system's OOF-fitted per-band thresholds keep their relative structure and move by ONE common scalar on the uncertainty scale, solved so referral on
the analysed lesions equals `BUDGET`. Label-free and in closed form, and the same for every system, so all comparisons are at matched referral.

Read, deployed system against V5 and V1:
  primary      the mechanism endpoint: melanoma vs histopathology-confirmed nevus pAUC@FPR 0.20, all ages (§A2, revision 2 Oct); patient-clustered bootstrap, because HIBA has
               about two images per patient.
  all-age      escalation pAUC@0.20 and 7-class Macro-F1 (full coverage), same bootstrap. Macro-F1 non-inferiority margin 0.02, as S84.
  C6           the S48 endpoint in its declared form: under-40 escalation sensitivity at matched referral, lesion unit, MCID 0.10, exact one-sided McNemar. Descriptive: about 20 lesions.
  S70          under-40 system sensitivity >= 0.855 at the locked budget, Clopper-Pearson. Referral holds by construction, so sensitivity is the real test. Descriptive.
  C1           false reassurance rate by band from the conformal output field, Clopper-Pearson, V6 systems only (V5 and V1 carry no refit of this layer).
Nothing here fits on HIBA except the one label-free scalar above.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.v5 import s84_milk10k as s84

REPO = Path(__file__).resolve().parents[2]
HIBA_DIR = REPO / "results" / "v6" / "hiba"
RECEIPT = REPO / "results" / "v6" / "hiba_receipt.json"
COHORT = HIBA_DIR / "cohort.csv"
INFERENCE = HIBA_DIR / "inference"
REPORT = HIBA_DIR / "hiba_report.json"
DECISIONS = HIBA_DIR / "decisions.csv"
IMAGES = REPO / "data" / "external" / "hiba"
E_PRIMARY = REPO / "results" / "v6" / "e_primary.json"
REFIT = REPO / "results" / "v6" / "refit"
PIXEL_COPIES = REPO / "results" / "v6" / "screens" / "leak_audit_hiba_pixels.json"
CHECKPOINTS = REPO / "ml" / "checkpoints"
FINALISTS = {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}
TAG_FOLDS_1_4 = "v64"
COLLECTIONS = ("175", "176", "251")
COHORT_COLUMNS = ["isic_id", "lesion_id", "patient_id", "age", "image_type"]  # never a diagnosis column
BUDGET = 0.20  # S56's primary budget and S84's, inside the S70 contract's 0.25; declared here, before any read
SEED, N_BOOT = 20261105, 2000
ALPHA, S70_SENS, C6_MCID, F1_MARGIN = 0.05, 0.855, 0.10, 0.02
HASH_RADIUS = 3
CLASS_7, ESC_IDX = s84.CLASS_7, s84.ESC_IDX
SYSTEMS = ("v6_vote", "combined_vote", "v5", "v1")


# =================================================================== the single-read receipt
def frozen_inputs() -> dict[str, Path]:
    inputs = {"e_primary": E_PRIMARY, "reader": Path(__file__), "cohort": COHORT, "v5_dirichlet": s84.DIRICHLET_STATE, "v5_s56": s84.S56_POLICIES_V5, "v1_s56": s84.S56_POLICIES_V1}
    for system in ("v6_vote", "combined_vote"):
        for name in ("dirichlet_state.json", "s56_policies_v6.json", "conformal_state.json"):
            inputs[f"{system}_{name}"] = REFIT / system / name
    if PIXEL_COPIES.is_file():
        inputs["pixel_copies"] = PIXEL_COPIES
    return inputs


def input_hashes() -> dict[str, str]:
    missing = [n for n, p in frozen_inputs().items() if not p.is_file()]
    if missing:
        raise SystemExit(f"cannot start: missing frozen inputs {missing}")
    return {n: s84.sha256(p) for n, p in frozen_inputs().items()}


def load_receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.is_file() else {"cohort": "HIBA (ISIC collections 175, 176, 251), V6-6", "n_executions": 0, "label_read_count": 0, "status": "none", "inputs": {}, "executions": []}


def write_receipt(receipt: dict) -> None:
    if receipt["n_executions"] > 1:
        raise AssertionError("the HIBA receipt must never record more than one execution")
    s84.write_json(RECEIPT, receipt)


def begin_execution(resume: bool) -> dict:
    receipt, hashes = load_receipt(), input_hashes()
    if resume:
        if receipt["n_executions"] != 1 or receipt["status"] != "started":
            raise SystemExit("--resume needs the one execution to have started and not completed")
        if receipt["label_read_count"] != 0:
            raise SystemExit("a label was already read: this execution cannot be resumed or repeated; a second read is the owner's decision, outside this script")
        if receipt["inputs"] != hashes:
            raise SystemExit("a frozen input changed after the read started; refusing")
        return receipt
    if receipt["n_executions"] >= 1:
        raise SystemExit(f"HIBA already executed (n_executions = {receipt['n_executions']}, status {receipt['status']}); this script has no rerun flag")
    receipt.update(n_executions=1, status="started", inputs=hashes)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    receipt["executions"].append({"execution": 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head})
    write_receipt(receipt)
    return receipt


# =================================================================== decisions (label-free)
def lock_budget(policy, probs: np.ndarray, bands: np.ndarray, budget: float = BUDGET) -> tuple[np.ndarray, float]:
    """C2. One common scalar on the uncertainty scale, in closed form. Returns (referred, scalar)."""
    from research.v4 import s56_abstention as s56

    margin = s56.uncertainty(policy.score, probs, ESC_IDX) - policy.thresholds(np.asarray(bands).astype(str))
    scalar = float(np.quantile(margin, 1 - budget))
    return margin >= scalar, scalar


def v6_layers(system: str):
    from research.v5 import refit_dirichlet as rd
    from research.v5 import refit_s56 as rs

    folder = REFIT / system
    blob = json.loads((folder / "s56_policies_v6.json").read_text(encoding="utf-8"))
    return rd.load_state(folder / "dirichlet_state.json"), rs.policy_from_dict(blob["policies"][f"band_{blob['primary_budget']}"])


def decide(raw: dict[str, np.ndarray], bands: np.ndarray) -> dict[str, dict]:
    """raw: uncalibrated (N, 7) per system. Calibrate with each system's own map, then lock the budget. No label exists here."""
    from research.external import frozen_params as fp
    from research.v5 import refit_dirichlet as rd
    from research.v5 import refit_s56 as rs

    out = {}
    for system, probs in raw.items():
        if system == "v1":
            calibrated = fp.calibrate(probs)
            policy = rs.policy_from_dict(json.loads(s84.S56_POLICIES_V1.read_text(encoding="utf-8"))[f"band_{s84.V1_BUDGET}"])
        elif system == "v5":
            calibrated = rd.calibrate(rd.load_state(s84.DIRICHLET_STATE), probs)
            policy = rs.policy_from_dict(json.loads(s84.S56_POLICIES_V5.read_text(encoding="utf-8"))["policies"][f"band_{s84.V1_BUDGET}"])
        else:
            state, policy = v6_layers(system)
            calibrated = rd.calibrate(state, probs)
        referred, scalar = lock_budget(policy, calibrated, bands)
        out[system] = {"probs": calibrated, "referred": referred, "budget_scalar": scalar, "referral": float(referred.mean())}
    return out


def conformal_sets(system: str, probs: np.ndarray, bands: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """C1: the V6 system's stored thresholds on its RAPS scores. An output field: it refers nobody."""
    from research.conformal import hierarchical
    from research.conformal import scores as cs
    from research.v6 import refit_system as rf

    blob = json.loads((REFIT / system / "conformal_state.json").read_text(encoding="utf-8"))["states"][str(alpha)]
    matrix = cs.aps_scores(probs, np.random.default_rng(blob["seed"]), penalty=blob["penalty"], k_reg=blob["k_reg"])
    groups = hierarchical.class_groups(len(CLASS_7), ESC_IDX)
    index = {b: i for i, b in enumerate(rf.BAND_ORDER)}
    thresholds = np.empty((len(rf.BAND_ORDER), len(CLASS_7)))
    for band, i in index.items():
        for c, group in enumerate(groups):
            q = blob["cell_quantiles"][f"{band}|{hierarchical.GROUP_LABELS[int(group)]}"]
            thresholds[i, c] = q if np.isfinite(q) else blob["fallback_quantiles"][c]
    return matrix <= thresholds[np.array([index.get(b, index["unknown"]) for b in np.asarray(bands).astype(str)])]


# =================================================================== analysis (pure: no file, no receipt)
def cluster_bootstrap(metric, patients: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        try:
            out[b] = metric(pick)
        except ValueError:
            continue
    return out


def _delta_read(metric_of, a: np.ndarray, b: np.ndarray, patients: np.ndarray, every: np.ndarray, n_boot: int) -> dict:
    delta = metric_of(a, every) - metric_of(b, every)
    samples = cluster_bootstrap(lambda pick: metric_of(a, pick) - metric_of(b, pick), patients, n_boot)
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {"delta": float(delta), "ci95": [float(low), float(high)], "valid_resamples": int(np.isfinite(samples).sum())}


def analyse_core(data: pd.DataFrame, systems: dict[str, dict], deployed: str, n_boot: int = N_BOOT) -> dict:
    """data: one row per analysed lesion: `class_7` (None = out of scope), `histo` (bool), `age_band`, `patient_id`, row-aligned with each system's probs and referred."""
    from research.v5.d0_brainstorm_diagnostics import pauc

    inscope = data["class_7"].notna().to_numpy()
    y7 = np.array([CLASS_7.index(c) if isinstance(c, str) else -1 for c in data["class_7"]])
    y_esc = np.isin(y7, ESC_IDX)
    bands = data["age_band"].astype(str).to_numpy()
    patients = data["patient_id"].astype(str).to_numpy()
    u40 = (bands == "<40") & inscope
    mel, nv = CLASS_7.index("mel"), CLASS_7.index("nv")
    mechanism_rows = inscope & ((y7 == mel) | ((y7 == nv) & data["histo"].to_numpy().astype(bool)))
    esc_mass = {k: v["probs"][:, ESC_IDX].sum(1) for k, v in systems.items()}
    pred = {k: v["probs"].argmax(1) for k, v in systems.items()}
    caught = {k: np.isin(pred[k], ESC_IDX) | np.asarray(systems[k]["referred"], bool) for k in systems}
    every = np.arange(len(data))

    def mech(scores: np.ndarray, pick: np.ndarray) -> float:
        kept = pick[mechanism_rows[pick]]
        labels = y7[kept] == mel
        if labels.all() or not labels.any():
            raise ValueError("one class only")
        return float(pauc(labels, scores[kept]))

    def all_age(scores: np.ndarray, pick: np.ndarray) -> float:
        kept = pick[inscope[pick]]
        labels = y_esc[kept]
        if labels.all() or not labels.any():
            raise ValueError("one class only")
        return float(pauc(labels, scores[kept]))

    def f1(predicted: np.ndarray, pick: np.ndarray) -> float:
        kept = pick[inscope[pick]]
        return s84.macro_f1(y7[kept], predicted[kept])

    rows = {k: {"mechanism_pauc": mech(esc_mass[k], every), "pauc_all": all_age(esc_mass[k], every), "macro_f1": f1(pred[k], every), "referral": float(np.mean(systems[k]["referred"])),
                "budget_scalar": systems[k].get("budget_scalar"), "u40": s84.sens_block(caught[k], y_esc, u40), "all_ages": s84.sens_block(caught[k], y_esc, inscope)} for k in systems}
    comparisons = {}
    for other in [k for k in ("v5", "v1", "v6_vote", "combined_vote") if k != deployed and k in systems]:
        pos = u40 & y_esc
        n10, n01 = int((caught[deployed][pos] & ~caught[other][pos]).sum()), int((~caught[deployed][pos] & caught[other][pos]).sum())
        macro = _delta_read(f1, pred[deployed], pred[other], patients, every, n_boot)
        comparisons[other] = {"mechanism_pauc": _delta_read(mech, esc_mass[deployed], esc_mass[other], patients, every, n_boot), "pauc_all": _delta_read(all_age, esc_mass[deployed], esc_mass[other], patients, every, n_boot),
                              "macro_f1": {**macro, "margin": F1_MARGIN, "non_inferior": bool(macro["ci95"][0] > -F1_MARGIN)},
                              "c6_u40_matched_referral": {"n_lesions": int(pos.sum()), "deployed_only": n10, "other_only": n01, "delta_sensitivity": (n10 - n01) / max(int(pos.sum()), 1), "mcid": C6_MCID,
                                                          "meets_mcid": bool((n10 - n01) / max(int(pos.sum()), 1) >= C6_MCID), "mcnemar_exact_one_sided_p": s84.exact_mcnemar_p(n10, n01), "alpha": ALPHA, "descriptive": True}}
    frr = {}
    for system, sets in {k: systems[k]["sets"] for k in systems if "sets" in systems[k]}.items():
        has_esc = sets[:, ESC_IDX].any(axis=1)
        frr[system] = {}
        for band in ("<40", "40-59", "60+", "unknown"):
            pos = (bands == band) & inscope & y_esc
            if pos.any():
                missed = int((~has_esc[pos]).sum())
                low, high = s84.clopper_pearson(missed, int(pos.sum()))
                frr[system][band] = {"missed": missed, "n_escalating": int(pos.sum()), "rate": missed / int(pos.sum()), "ci95_clopper_pearson": [low, high]}
    contract = rows[deployed]["u40"]
    return {"deployed": deployed, "budget": BUDGET, "lesions": len(data), "in_scope": int(inscope.sum()), "mechanism_rows": int(mechanism_rows.sum()), "under40_escalating_lesions": int((u40 & y_esc).sum()), "systems": rows, "deployed_vs": comparisons,
            "s70": {"target": S70_SENS, "sensitivity": contract["sensitivity"], "ci95": contract["ci95_clopper_pearson"], "point_estimate_meets": bool(contract["sensitivity"] is not None and contract["sensitivity"] >= S70_SENS), "descriptive": True},
            "c1_false_reassurance": frr, "n_boot": n_boot, "rng_seed": SEED}


# =================================================================== prepare (no labels, no model)
def linked_ids() -> tuple[set[str], set[str], set[str]]:
    from research.v6 import added_pool

    guarded = added_pool.guarded_ids()
    added = pd.read_csv(REPO / "results/v6/added_pool/candidates.csv", low_memory=False, dtype=str, usecols=["isic_id", "lesion_id", "patient_id"])
    kept = pd.read_csv(REPO / "results/v6/added_pool/added_train.csv", low_memory=False, dtype=str, usecols=["image_id"])
    added = added[added["isic_id"].isin(set(kept["image_id"]))]
    images = guarded["manifest_images"] | guarded["extras_images"] | set(kept["image_id"])
    lesions = guarded["manifest_lesions"] | set(added["lesion_id"].dropna())
    patients = guarded["manifest_patients"] | set(added["patient_id"].dropna())
    return images, lesions - {""}, patients - {""}


def hash_copies(ids: list[str]) -> set[str]:
    """dHash AND pHash within radius 3 of the corpus, or of a kept added row. The audit's own hashes."""
    from research.v4 import dedupe

    def popcount(x: np.ndarray) -> np.ndarray:
        return np.array([bin(int(v)).count("1") for v in x])

    hiba = pd.read_csv(REPO / "results/v6/added_pool/hiba_hashes.csv", dtype={"image_id": str, "dhash": str, "phash": str}).set_index("image_id")
    added = pd.read_csv(REPO / "results/v6/added_pool/hashes.csv", dtype={"image_id": str, "dhash": str, "phash": str}, usecols=["image_id", "dhash", "phash"])
    kept = set(pd.read_csv(REPO / "results/v6/added_pool/added_train.csv", low_memory=False, dtype=str, usecols=["image_id"])["image_id"])
    added = added[added["image_id"].isin(kept)]
    corpus = dedupe.load_hashes()
    d_ref = np.concatenate([corpus["dhash"].to_numpy(dtype=np.uint64), added["dhash"].astype("uint64").to_numpy()])
    p_ref = np.concatenate([corpus["phash"].to_numpy(dtype=np.uint64), added["phash"].astype("uint64").to_numpy()])
    copies = set()
    for image in ids:
        if image not in hiba.index:
            raise SystemExit(f"{image} has no hash: the audit's hiba_hashes.csv does not cover the cohort")
        d, p = np.uint64(hiba.at[image, "dhash"]), np.uint64(hiba.at[image, "phash"])
        near = popcount(d_ref ^ d) <= HASH_RADIUS
        if near.any() and (popcount(p_ref[near] ^ p) <= HASH_RADIUS).any():
            copies.add(image)
    return copies


def build_cohort() -> pd.DataFrame:
    """Label-free. Reads only `COHORT_COLUMNS` from the cache, which also holds a diagnosis column."""
    from research.v6 import data_audit
    from research.v6.added_pool import age_band

    frames = [pd.read_csv(data_audit.CACHE / f"hiba_{c}.csv", usecols=COHORT_COLUMNS, dtype=str).assign(collection=c) for c in COLLECTIONS]
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("isic_id")
    meta = meta[meta["image_type"].astype(str).str.lower() == "dermoscopic"].copy()
    meta["lesion_key"] = meta["lesion_id"].fillna("").where(meta["lesion_id"].fillna("") != "", meta["isic_id"])
    meta = meta.sort_values("isic_id").drop_duplicates("lesion_key").reset_index(drop=True)
    images, lesions, patients = linked_ids()
    linked = meta["isic_id"].isin(images) | meta["lesion_id"].isin(lesions) | meta["patient_id"].isin(patients)
    copies = hash_copies(meta["isic_id"].tolist())
    meta["excluded"] = np.where(linked, "id_link", np.where(meta["isic_id"].isin(copies), "hash_copy", ""))
    if PIXEL_COPIES.is_file():
        flagged = set(json.loads(PIXEL_COPIES.read_text(encoding="utf-8"))["copies"])
        meta.loc[meta["isic_id"].isin(flagged) & (meta["excluded"] == ""), "excluded"] = "pixel_copy"
    meta["age_band"] = [age_band(float(a)) if str(a).strip() not in ("", "nan") else "unknown" for a in meta["age"]]
    meta["patient_id"] = meta["patient_id"].fillna("").where(meta["patient_id"].fillna("") != "", meta["isic_id"])
    return meta[["isic_id", "lesion_key", "patient_id", "age_band", "collection", "excluded"]]


def download(ids: list[str], workers: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from research.v6.added_pool import FULL_URL, fetch

    IMAGES.mkdir(parents=True, exist_ok=True)

    def one(image: str) -> None:
        path = IMAGES / f"{image}.jpg"
        if not path.exists():
            part = path.with_suffix(".part")
            part.write_bytes(fetch(FULL_URL.format(image)))
            part.replace(path)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, ids))


def prepare() -> int:
    from research import testguard

    testguard.block_test_reads("V6-6 prepare: HIBA cohort without labels")
    if not (load_receipt()["n_executions"] == 0):
        raise SystemExit("the read has started; the cohort cannot change")
    cohort = build_cohort()
    HIBA_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(COHORT, index=False)
    kept = cohort[cohort["excluded"] == ""]
    download(kept["isic_id"].tolist())
    print(f"cohort: {len(cohort):,} lesions, {len(kept):,} analysed; excluded {cohort['excluded'][cohort['excluded'] != ''].value_counts().to_dict()} -> {s84.rel(COHORT)}")
    print(f"under 40: {int((kept['age_band'] == '<40').sum()):,} lesions, labels unknown. No label and no model was read.")
    return 0


# =================================================================== infer (GPU, no labels)
def v6_checkpoints() -> list[Path]:
    out = []
    for label, tag in FINALISTS.items():
        member = "anchor" if label.startswith("anchor") else label.removesuffix("_h7")
        for fold in range(5):
            run = f"{label}_f{fold}_s42{'_in22k' if label.startswith('anchor') else ''}_{tag if fold == 0 else TAG_FOLDS_1_4}"
            out.append(CHECKPOINTS / f"{member}-v6_{run}_swad.pt")
    return out


class HibaViews:
    """Module level, so it pickles under Windows spawn. No label exists at this stage."""

    def __init__(self, ids: list[str], image_size: int, mean: list[float], std: list[float]) -> None:
        self.ids, self.image_size, self.mean, self.std, self._views = ids, image_size, mean, std, None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        from research.v6 import tta

        if self._views is None:
            self._views = tta.build_views(self.image_size, self.mean, self.std)
        with Image.open(IMAGES / f"{self.ids[index]}.jpg") as image:
            image = image.convert("RGB")
        return torch.stack([v.transform(image) for v in self._views]), 0


def stage_infer(device_name: str, workers: int) -> int:
    import torch
    from torch.utils.data import DataLoader

    from ml.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD
    from ml.training.common import build_model_from_checkpoint
    from research.ensembling.data import ARCHS
    from research.v5.tta_v5 import load_composite, tta_pass
    from research.v6 import predict_tta as pt

    device = torch.device(device_name)
    cohort = pd.read_csv(COHORT, dtype=str)
    ids = cohort.loc[cohort["excluded"].fillna("") == "", "isic_id"].tolist()
    INFERENCE.mkdir(parents=True, exist_ok=True)
    jobs = [("v6", p) for p in v6_checkpoints()] + [("v5", p) for p in s84.composite_checkpoints_available()] + [("v1", CHECKPOINTS / f"{a}_best.HAM-only.pt") for a in ARCHS]
    for family, path in jobs:
        out = INFERENCE / f"{family}__{path.stem}.npy"
        if out.exists():
            print(f"skip {out.name}", flush=True)
            continue
        if family == "v6":
            model, payload = pt.load_model(path, device)
            size, (mean, std) = payload["image_size"], pt.statistics_of(payload)
        elif family == "v5":
            model, payload = load_composite(path, device)
            size, mean, std = payload["image_size"], list(IMAGENET_MEAN), list(IMAGENET_STD)
        else:
            model, payload = build_model_from_checkpoint(path, device)
            size, mean, std = payload.get("image_size", 224), list(IMAGENET_MEAN), list(IMAGENET_STD)
        loader = DataLoader(HibaViews(ids, size, mean, std), batch_size=pt.BATCH_SIZE.get(size, 8), shuffle=False, num_workers=workers)
        pooled = tta_pass({"m": model}, loader, device, path.stem)["pooled"]["m"]
        np.save(out, pt.to_seven(pooled))
        del model
        torch.cuda.empty_cache()
        print(f"wrote {out.name}", flush=True)
    return 0


def member_vote(paths: list[Path]) -> np.ndarray:
    return np.mean([np.load(p) for p in paths], axis=0)


def system_probabilities() -> dict[str, np.ndarray]:
    """Uniform per model, as in `research.v6.system_reads`. Each OOF row came from one fold model of each member; here a member is the mean of its five fold models."""
    members = [member_vote([INFERENCE / f"v6__{p.stem}.npy" for p in v6_checkpoints() if f"_{label}_f" in p.stem]) for label in FINALISTS]
    v5_paths = [INFERENCE / f"v5__{p.stem}.npy" for p in s84.composite_checkpoints_available()]
    seeds = [member_vote([p for p in v5_paths if f"_s{seed}_" in p.name]) for seed in (42, 43, 44)]
    from research.ensembling.data import ARCHS
    from research.ensembling.methods import soft_vote_arithmetic

    v1 = soft_vote_arithmetic(np.stack([np.load(INFERENCE / f"v1__{a}_best.HAM-only.npy") for a in ARCHS], axis=1))
    return {"v6_vote": np.mean(members, axis=0), "combined_vote": np.mean([*members, *seeds], axis=0), "v5": np.mean(seeds, axis=0), "v1": v1}


# =================================================================== analyse (the one label read)
def read_labels(cohort: pd.DataFrame, receipt: dict | None) -> pd.DataFrame:
    """THE label read. Increments the receipt's label_read_count first. Labels come fresh from the archive, and only in this function."""
    from research.v5 import young_data

    if receipt is not None:
        receipt["label_read_count"] += 1
        receipt["executions"][-1]["labels_read_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_receipt(receipt)
    rows = []
    for collection in COLLECTIONS:
        page = young_data._get({"collections": collection, "limit": 100})
        while True:
            for item in page["results"]:
                clinical = item["metadata"].get("clinical", {})
                rows.append({"isic_id": item["isic_id"], "confirm": clinical.get("diagnosis_confirm_type", ""), **{f"diagnosis_{i}": clinical.get(f"diagnosis_{i}", "") for i in (1, 2, 3, 4)}})
            if not page.get("next"):
                break
            page = young_data._get(url=page["next"])
    labels = pd.DataFrame(rows).drop_duplicates("isic_id")
    merged = cohort.merge(labels, on="isic_id", how="left")
    merged["class_7"] = merged.apply(s84.map_class, axis=1)
    merged["histo"] = merged["confirm"].fillna("").astype(str).str.lower().str.contains("histopathology")
    return merged


def stage_analyse(receipt: dict | None) -> int:
    cohort = pd.read_csv(COHORT, dtype=str)
    cohort = cohort[cohort["excluded"].fillna("") == ""].reset_index(drop=True)
    deployed = json.loads(E_PRIMARY.read_text(encoding="utf-8"))["deployed_system"]
    bands = cohort["age_band"].astype(str).to_numpy()
    # Every decision is fixed here, from probabilities and age bands alone, before any label is read.
    systems = decide(system_probabilities(), bands)
    for system in ("v6_vote", "combined_vote"):
        systems[system]["sets"] = conformal_sets(system, systems[system]["probs"], bands)
    data = read_labels(cohort, receipt)
    report = analyse_core(data, systems, deployed)
    report.update(cohort={"lesions": len(cohort), "patients": int(cohort["patient_id"].nunique()), "by_collection": cohort["collection"].value_counts().to_dict()},
                  pixel_copies_excluded=bool(PIXEL_COPIES.is_file()), v5_models=[p.stem for p in s84.composite_checkpoints_available()], label="external, new population and device")
    s84.write_json(REPORT, report)
    pd.DataFrame({"isic_id": data["isic_id"], **{f"{k}_pred": systems[k]["probs"].argmax(1) for k in systems}, **{f"{k}_referred": systems[k]["referred"] for k in systems}}).to_csv(DECISIONS, index=False)
    print(json.dumps({k: report[k] for k in ("deployed", "lesions", "under40_escalating_lesions", "deployed_vs", "s70")}, indent=2, default=s84._plain))
    return 0


# =================================================================== orchestration
def execute(resume: bool, device: str, workers: int) -> int:
    from research import testguard

    testguard.block_test_reads("V6-6: HIBA only; HAM test, reserved and development labels untouched")
    deployed = json.loads(E_PRIMARY.read_text(encoding="utf-8"))["deployed_system"] if E_PRIMARY.is_file() else None
    if deployed is None or deployed == "v5_system":
        raise SystemExit(f"e_primary.json deploys {deployed}: V6-6 reads a system that passed V6-5, so HIBA stays unread")
    if dry_run() != 0:
        raise SystemExit("preconditions fail; see above. Nothing was started.")
    receipt = begin_execution(resume)
    code = subprocess.run([sys.executable, "-m", "research.v6.hiba_read", "--stage-infer", "--device", device, "--num-workers", str(workers)], cwd=REPO).returncode
    if code != 0:
        raise SystemExit("inference failed before any label was read; fix it and run with --resume")
    stage_analyse(receipt)
    receipt["status"] = "completed"
    receipt["executions"][-1].update(completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), outputs={s84.rel(p): s84.sha256(p) for p in (REPORT, DECISIONS, COHORT)})
    write_receipt(receipt)
    return 0


def dry_run() -> int:
    ok = True
    for name, path in frozen_inputs().items():
        present = path.is_file()
        ok &= present
        print(f"{'ok     ' if present else 'MISSING'} {name}")
    missing = [p.name for p in v6_checkpoints() if not p.is_file()]
    print(f"V6 fold models: {20 - len(missing)} of 20" + (f"; missing e.g. {missing[0]}" if missing else ""))
    v5 = s84.composite_checkpoints_available()
    print(f"V5 composites: {len(v5)} of 15")
    if COHORT.is_file():
        cohort = pd.read_csv(COHORT, dtype=str)
        ids = cohort.loc[cohort["excluded"].fillna("") == "", "isic_id"].tolist()
        absent = [i for i in ids if not (IMAGES / f"{i}.jpg").is_file()]
        print(f"cohort: {len(ids):,} analysed, {len(absent):,} images not downloaded")
        ok &= not absent
    receipt = load_receipt()
    print(f"receipt: n_executions = {receipt['n_executions']}, labels read {receipt['label_read_count']}")
    print("no label read, no model loaded")
    return 0 if ok and not missing and len(v5) == 15 else 1


def smoke() -> int:
    rng = np.random.default_rng(0)
    n = 600
    classes = rng.choice([*CLASS_7, None], n, p=[.05, .06, .12, .01, .15, .55, .03, .03])
    data = pd.DataFrame({"class_7": classes, "histo": rng.random(n) < 0.6, "age_band": rng.choice(["<40", "40-59", "60+", "unknown"], n), "patient_id": rng.integers(0, 300, n).astype(str)})

    def fake() -> dict:
        probs = rng.dirichlet(np.ones(7), n)
        return {"probs": probs, "referred": rng.random(n) < 0.2, "sets": rng.random((n, 7)) < 0.5}

    report = analyse_core(data, {s: fake() for s in SYSTEMS}, "v6_vote", n_boot=20)
    s84.write_json(HIBA_DIR / "smoke" / "hiba_smoke_report.json", report)
    print("smoke ok:", sorted(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--dry-run", "--smoke", "--prepare", "--execute", "--resume", "--stage-infer"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.stage_infer:
        if load_receipt()["status"] != "started":
            parser.error("--stage-infer is internal: no HIBA image is scored outside a started --execute")
        return stage_infer(args.device, args.num_workers)
    if args.dry_run:
        return dry_run()
    if args.smoke:
        return smoke()
    if args.prepare:
        return prepare()
    if args.execute:
        return execute(args.resume, args.device, args.num_workers)
    parser.error("give one of --dry-run, --smoke, --prepare, --execute")


if __name__ == "__main__":
    raise SystemExit(main())
