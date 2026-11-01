"""S84 -- the one-shot MILK10k external read (runsheet section 9; deviation D5 for the power method).

    python -m research.v5.s84_milk10k --power            # label-free of MILK10k: power from OOF folds 1-4 (D5 row set)
    python -m research.v5.s84_milk10k --dry-run          # preconditions only; no label, no model, no receipt
    python -m research.v5.s84_milk10k --smoke            # the analysis path on synthetic data -> s84/smoke/, no receipt
    python -m research.v5.s84_milk10k --execute          # THE read. Refuses a second execution (receipt n_executions <= 1)
    python -m research.v5.s84_milk10k --execute --resume # only after a crash that happened before any label was read

NOT RUN by its author. Single read (section 9): `results/v5/s84_receipt.json` holds `n_executions`; it is incremented
BEFORE any image is scored, is never reset, and `--execute` refuses when it is already >= 1. There is no rerun flag:
a repeat read is the owner's decision outside this script. `--resume` continues the SAME execution (n_executions
unchanged) and only while `label_read_count == 0`, i.e. the crash happened in the GPU / cohort stages, which read no
label. The labels are read in exactly one place (`read_labels`, analysis stage), which first increments
`label_read_count`. The receipt pins the sha256 of every frozen input; a changed input refuses the run.

Stages (each its own process so the GPU stage never imports the GPU-hiding S75 module):
  cohort   CPU, NO labels. One dermoscopic image per lesion, S75's exclusion (ISIC id, or dHash AND pHash within 3 of any
           of the 25,331 ISIC-2019 images). Writes s84/cohort.csv (isic_id, lesion_id, age_band, attribution, zip name).
  infer    GPU, NO labels. V5: every available composite `_last.pt` of the 15 (3 seeds x 5 folds), 24-view TTA, mean of
           the model probabilities. V1: the six frozen HAM-only CNNs, 24-view TTA, uniform soft-vote. -> s84/inference.npz
  analyse  CPU. Reads the labels once. Applies the decision layers, runs section 9's analysis, writes s84/s84_report.json.

Section 9 items implemented (A01 A9):
  systems        V5 = composite 3x5 ensemble + TTA + Dirichlet refit on the seed-averaged V5 OOF + S56 refit on that OOF
                 (research.v5.refit_dirichlet / refit_s56, applied here unchanged); V1 = frozen six-CNN stack, frozen
                 Dirichlet (research.external.frozen_params), deployed S56 band policy @ 0.20 (results/v4/s56/policies_oof.json).
  primary        paired difference in <40 escalation sensitivity (caught = argmax escalates OR referred) at MATCHED
                 referral: V5's S56 budget is bisected, label-free, until its referral on the in-scope lesions equals V1's;
                 lesion unit; exact one-sided McNemar on the discordant lesions, alpha 0.05.
  key secondary  the S70 contract: V5 <40 system sensitivity >= 0.855 AND referral <= 0.25, Clopper-Pearson.
  global         7-class Macro-F1 V5 vs V1 (full coverage), non-inferiority margin 0.02, bootstrap CI (2,000 resamples,
                 seed 20261004; one image per lesion, so the lesion is the resampling unit).
  descriptive    escalation AUROC and pAUC@0.20 overall and by age band, per-institution breakdown (`attribution`),
                 out-of-scope stratum referral rate of each system, retained-set Macro-F1.
  power          computed BEFORE the read by `--power` (D5 method, 2,000 bootstrap resamples over the 37 under-40 lesions),
                 required by `--execute`, reported whatever it is; "underpowered" is stated when < 0.80.
  single read    the receipt guard above.
  allocation     MILK10k is V5 confirmation only; nothing here fits anything on it (every parameter is loaded frozen).
Interpretation choices to check against section 9 (flagged for the Opus review): (a) matched referral is measured over ALL eligible lesions (label-free; decisions are fixed before the labels are read); the
S70 contract is scored at V5 own deployed band-0.20 policy, not at the matched point; (b) global Macro-F1 is
full-coverage argmax of each system's calibrated probabilities (abstention changes no class prediction); (c) V5 uses the
mean over all available checkpoints of the 15, the set recorded in the report.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

REPO_ROOT = Path(__file__).resolve().parents[2]
V5 = REPO_ROOT / "results" / "v5"
S84_DIR = V5 / "s84"
RECEIPT = V5 / "s84_receipt.json"
POWER_PATH = S84_DIR / "power.json"
COHORT_PATH = S84_DIR / "cohort.csv"
INFERENCE_PATH = S84_DIR / "inference.npz"
REPORT_PATH = S84_DIR / "s84_report.json"
MILK_DIR = REPO_ROOT / "data" / "external" / "milk10k"
MILK_ZIP, MILK_META = MILK_DIR / "milk10k.zip", MILK_DIR / "milk10k.csv"
S75_JSON = V5 / "s75_milk10k_eligibility.json"
POWER_NOTE = V5 / "s84_power_method.md"
DIRICHLET_STATE = V5 / "refit" / "dirichlet_state.json"
S56_POLICIES_V5 = V5 / "refit" / "s56_policies_v5.json"
S56_POLICIES_V1 = REPO_ROOT / "results" / "v4" / "s56" / "policies_oof.json"
OOF_MATRIX = V5 / "v5_oof_common_matrix.csv"
V1_FILES = V5 / "preds_v1"
CKPT_DIR = REPO_ROOT / "ml" / "checkpoints"

ALPHA = 0.05
SEED = 20261004
N_BOOT = 2000
V1_BUDGET = 0.2
S70_SENS, S70_REFERRAL = 0.855, 0.25
F1_MARGIN = 0.02
S70_RELATIVE_MCID = 0.05  # results/v4/s70/s70_decisions.json D2_contract.relative_term
MATCH_TOLERANCE = 0.01
REQUIRED_DIRICHLET_MODEL = "composite_tta_mean"
ESCALATING = {"mel", "bcc", "akiec"}
CLASS_7 = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ESC_IDX = [CLASS_7.index(c) for c in ("akiec", "bcc", "mel")]
BANDS = ("<40", "40-59", "60+", "unknown")
EXTRA_RULES = (("diagnosis_3", "Keratoacanthoma", "akiec"), ("diagnosis_4", "Keratoacanthoma", "akiec"))
D5_ROWS, D5_LESIONS, D5_U40_LESIONS, D5_U40_IMAGES = 6672, 2838, 37, 104
D5_EXCLUDED_IMAGE = "ISIC_0010852"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def write_json(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2, default=_plain), encoding="utf-8")
    tmp.replace(path)


# =================================================================== the single-read receipt
def frozen_inputs() -> dict[str, Path]:
    """Every input whose hash the receipt pins. The power file is among them: power is computed before the read."""
    return {"power_method": POWER_NOTE, "power_result": POWER_PATH, "s75_eligibility": S75_JSON,
            "dirichlet_state": DIRICHLET_STATE, "s56_policies_v5": S56_POLICIES_V5,
            "s56_policies_v1": S56_POLICIES_V1, "oof_matrix": OOF_MATRIX, "s84_script": Path(__file__)}


def input_hashes() -> dict[str, str]:
    missing = [n for n, p in frozen_inputs().items() if not p.is_file()]
    if missing:
        raise SystemExit(f"cannot start S84: missing frozen inputs {missing} (run --power / the refits first)")
    return {name: sha256(p) for name, p in frozen_inputs().items()}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_receipt(receipt: dict) -> None:
    assert receipt["n_executions"] <= 1, "the S84 receipt must never record more than one execution"
    write_json(RECEIPT, receipt)


def load_receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.is_file() else {
        "cohort": "MILK10k dermoscopic, S75-eligible (S84, runsheet section 9)", "n_executions": 0,
        "label_read_count": 0, "status": "none", "inputs": {}, "executions": []}


def begin_execution(resume: bool) -> dict:
    """Increment n_executions BEFORE any scoring; refuse a second execution. `resume` continues the same one."""
    receipt = load_receipt()
    hashes = input_hashes()
    if resume:
        if receipt["n_executions"] != 1 or receipt["status"] != "started":
            raise SystemExit("--resume needs the one execution to have started and not completed")
        if receipt["label_read_count"] != 0:
            raise SystemExit("a label was already read in this execution: it cannot be resumed or repeated; "
                             "a second read is the owner's decision, outside this script")
        if receipt["inputs"] != hashes:
            raise SystemExit("a frozen input changed after S84 started; refusing")
        receipt["executions"][-1].setdefault("resumed_at", []).append(_now())
        _write_receipt(receipt)
        return receipt
    if receipt["n_executions"] >= 1:
        raise SystemExit(f"S84 already executed (n_executions = {receipt['n_executions']}, status "
                         f"{receipt['status']}): section 9 allows one read; this script has no rerun flag")
    receipt.update(n_executions=receipt["n_executions"] + 1, status="started", inputs=hashes)
    receipt["executions"].append({"execution": 1, "started_at": _now(), "git_head": _git_head()})
    _write_receipt(receipt)
    return receipt


def mark_label_read(receipt: dict) -> None:
    receipt["label_read_count"] += 1
    receipt["executions"][-1]["labels_read_at"] = _now()
    _write_receipt(receipt)


def complete_execution(receipt: dict, outputs: list[Path]) -> None:
    receipt["status"] = "completed"
    receipt["executions"][-1].update(completed_at=_now(), outputs={rel(p): sha256(p) for p in outputs})
    _write_receipt(receipt)


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


# =================================================================== statistics
def exact_mcnemar_p(n10: int, n01: int) -> float:
    """One-sided exact McNemar: P[Bin(d, 0.5) >= n10] with d = n10 + n01 (H1: system 1 catches more)."""
    d = n10 + n01
    return 1.0 if d == 0 else float(binom.sf(n10 - 1, d, 0.5))


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar_critical(d: int, alpha: float = ALPHA) -> int:
    """Smallest c with P[Bin(d, 0.5) >= c] <= alpha; d + 1 when no c exists (never rejects)."""
    for c in range(0, d + 2):
        if binom.sf(c - 1, d, 0.5) <= alpha:
            return c
    return d + 1


def mcnemar_power(n: int, p10: float, p01: float, alpha: float = ALPHA) -> float:
    """Unconditional power of the exact one-sided conditional McNemar test on n paired lesions (D5 section 5)."""
    pd_ = p10 + p01
    if n <= 0 or pd_ <= 0:
        return 0.0
    pi = p10 / pd_
    total = 0.0
    for d in range(1, n + 1):
        c = mcnemar_critical(d, alpha)
        total += binom.pmf(d, n, pd_) * (binom.sf(c - 1, d, pi) if c <= d else 0.0)
    return float(total)


def match_budget(rate_of_budget, target: float, lo: float = 0.0, hi: float = 1.0, iters: int = 24,
                 tolerance: float = MATCH_TOLERANCE) -> tuple[float, float]:
    """Label-free bisection on the S56 budget R so that the system's referral rate equals `target` (rate is
    non-decreasing in R; R may reach 1.0). Returns (R, achieved rate) for the grid point closest to the target and
    RAISES if even that point is more than `tolerance` away: a silently unmatched read would void the primary."""
    best = (lo, rate_of_budget(lo))
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = rate_of_budget(mid)
        if abs(rate - target) < abs(best[1] - target) - 1e-12:
            best = (mid, rate)
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    top = rate_of_budget(1.0)
    if abs(top - target) < abs(best[1] - target):
        best = (1.0, top)
    if abs(best[1] - target) > tolerance:
        raise SystemExit(f"cannot match referral: target {target:.4f}, closest achievable {best[1]:.4f} at R = "
                         f"{best[0]:.3f} (gap > {tolerance})")
    return best


def require_tta_state() -> None:
    """The V5 Dirichlet map must come from the 24-view TTA V5 OOF (section 9: TTA is Stage 7). refit_dirichlet falls
    back to single view when the matrix has no TTA columns; that would apply a single-view map to TTA probabilities.
    Checked in --dry-run and before the receipt starts, never after: a failure after the start burns the one read."""
    if not DIRICHLET_STATE.is_file():
        raise SystemExit(f"{rel(DIRICHLET_STATE)} missing: run research.v5.refit_dirichlet")
    state = json.loads(DIRICHLET_STATE.read_text(encoding="utf-8"))
    if state["model_id"] != REQUIRED_DIRICHLET_MODEL:
        raise SystemExit(f"Dirichlet state was fitted on {state['model_id']!r}, not {REQUIRED_DIRICHLET_MODEL!r}: "
                         f"rerun oof_table (TTA columns) then refit_dirichlet and refit_s56")
    if state["matrix_sha256"] != sha256(OOF_MATRIX):
        raise SystemExit("Dirichlet state does not match the current OOF matrix: rerun the refits")
    policies = json.loads(S56_POLICIES_V5.read_text(encoding="utf-8")) if S56_POLICIES_V5.is_file() else None
    if policies is None or policies["matrix_sha256"] != sha256(OOF_MATRIX):
        raise SystemExit("S56 policies are missing or were fitted on a different OOF matrix: rerun refit_s56")


def v5_deployed_referred(probs: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """V5's own deployed S56 policy (s56_policies_v5.json["band_0.2"], pinned in the receipt) on calibrated probs."""
    from research.v5 import refit_s56 as rs

    blob = json.loads(S56_POLICIES_V5.read_text(encoding="utf-8"))["policies"][f"band_{V1_BUDGET}"]
    return rs.refer(rs.policy_from_dict(blob), probs, bands, ESC_IDX)


# =================================================================== decision layers
def esc_mass(probs: np.ndarray) -> np.ndarray:
    return probs[:, ESC_IDX].sum(1)


def v1_decide(member_probs: np.ndarray, bands: np.ndarray) -> dict:
    """member_probs: (N, 6, 7) in `research.ensembling.data.ARCHS` order. Frozen stack, nothing fitted."""
    from research.ensembling.methods import soft_vote_arithmetic
    from research.external import frozen_params as fp
    from research.v5 import refit_s56 as rs

    probs = fp.calibrate(soft_vote_arithmetic(member_probs))
    policy = rs.policy_from_dict(json.loads(S56_POLICIES_V1.read_text(encoding="utf-8"))[f"band_{V1_BUDGET}"])
    return {"probs": probs, "referred": rs.refer(policy, probs, bands, fp.escalating_indices()), "policy": policy.as_dict()}


@functools.lru_cache(maxsize=1)
def _v5_panel():
    from research.external import frozen_params as fp
    from research.v5 import refit_s56 as rs

    return rs.build_panel(pd.read_csv(OOF_MATRIX), pd.read_csv(rs.CROSSFIT), fp.escalating_indices())


def v5_policy_at(budget: float):
    """The V5 S56 band policy at an arbitrary budget, refitted on the stored V5 OOF panel (same code path as the
    stored s56_policies_v5.json; `fit_policy` is deterministic)."""
    from research.v5 import refit_s56 as rs

    return rs.fit_policy(_v5_panel(), budget, "band")


def v5_decide(member_probs: np.ndarray, bands: np.ndarray, target_rate: float, match_mask: np.ndarray,
              weights: np.ndarray | None = None, policy_factory=v5_policy_at) -> dict:
    """member_probs: (N, 7) seed/fold-averaged V5 probabilities. Calibrate with the V5 Dirichlet refit, then bisect
    the S56 budget so the referral rate over `match_mask` rows (lesion-weighted when `weights` given) equals
    `target_rate`. Label-free."""
    from research.v5 import refit_dirichlet as rd
    from research.v5 import refit_s56 as rs

    probs = rd.calibrate(rd.load_state(DIRICHLET_STATE), member_probs)
    w = np.ones(len(probs)) if weights is None else weights

    def referred_at(budget: float) -> np.ndarray:
        return rs.refer(policy_factory(budget), probs, bands, ESC_IDX)

    def rate(budget: float) -> float:
        r = referred_at(budget)[match_mask]
        return float((r * w[match_mask]).sum() / w[match_mask].sum())

    budget, achieved = match_budget(rate, target_rate)
    return {"probs": probs, "referred": referred_at(budget), "budget": budget, "matched_rate": achieved}


def attach_deployed(v5: dict, bands: np.ndarray) -> dict:
    v5["deployed_referred"] = v5_deployed_referred(v5["probs"], bands)
    return v5


# =================================================================== power (D5; MILK10k-free)
def d5_rows() -> pd.DataFrame:
    """Non-HAM rows of OOF folds 1-4 minus any row touching a HAM group / dup cluster / lesion (s84_power_method.md s3)."""
    f = pd.read_csv(REPO_ROOT / "results" / "v4" / "kfold" / "fold_assignments.csv")
    ham = f["archive"].eq("ham")
    touch = (f["group_id"].isin(f.loc[ham, "group_id"]) | f["dup_cluster"].isin(f.loc[ham, "dup_cluster"])
             | f["effective_lesion_id"].isin(f.loc[ham, "effective_lesion_id"]))
    rows = f[f["fold"].isin([1, 2, 3, 4]) & ~ham & ~touch].reset_index(drop=True)
    u40 = rows[(rows["age_band"] == "<40") & rows["class_7"].isin(ESCALATING)]
    assert len(rows) == D5_ROWS and rows["effective_lesion_id"].nunique() == D5_LESIONS, (len(rows), rows["effective_lesion_id"].nunique())
    assert set(rows["archive"]) == {"bcn20000", "mskcc"}
    assert u40["effective_lesion_id"].nunique() == D5_U40_LESIONS and len(u40) == D5_U40_IMAGES
    assert D5_EXCLUDED_IMAGE not in set(rows["image_id"])
    return rows


def lesion_weights(rows: pd.DataFrame) -> np.ndarray:
    """Each lesion weight 1, split equally across its images."""
    return 1.0 / rows.groupby("effective_lesion_id")["image_id"].transform("size").to_numpy()


def lesion_caught(rows: pd.DataFrame, caught: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-lesion fraction of images caught, for the lesions in `mask`."""
    frame = pd.DataFrame({"les": rows["effective_lesion_id"].to_numpy()[mask], "c": caught[mask].astype(float)})
    return frame.groupby("les")["c"].mean().sort_index().to_numpy()


def power_from_lesions(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator | None = None,
                       n_boot: int = N_BOOT) -> dict:
    """a, b: per-lesion V5 / V1 caught indicators (may be fractional). Plug-in power plus a lesion bootstrap."""
    p10, p01 = float(np.mean(a * (1 - b))), float(np.mean((1 - a) * b))
    out = {"p10": p10, "p01": p01, "discordance": p10 + p01, "n_lesions_estimation": int(len(a)), "N": int(n),
           "power_plugin": mcnemar_power(n, p10, p01), "discordant_lesions_V5_only": float((a * (1 - b)).sum()),
           "discordant_lesions_V1_only": float(((1 - a) * b).sum())}
    if rng is not None:
        draws = []
        for _ in range(n_boot):
            i = rng.integers(0, len(a), len(a))
            draws.append(mcnemar_power(n, float(np.mean(a[i] * (1 - b[i]))), float(np.mean((1 - a[i]) * b[i]))))
        out.update(power_bootstrap_median=float(np.median(draws)), power_bootstrap_p10=float(np.percentile(draws, 10)),
                   power_bootstrap_p90=float(np.percentile(draws, 90)), n_boot=n_boot, rng_seed=SEED)
    out["estimable"] = (p10 + p01) > 0
    out["underpowered"] = out["power_plugin"] < 0.80
    if not out["estimable"]:
        out["statement"] = "NOT ESTIMABLE: 0 discordant lesions on the plug-in; power reported as 0, underpowered"
    return out


def run_power() -> int:
    """Needs the V5 refits and Q9's V1 predictions on the D5 rows. Reads development labels only."""
    from research import testguard
    from research.ensembling.data import ARCHS
    from research.external import frozen_params as fp
    from research.v5 import refit_s56 as rs

    testguard.block_test_reads("S84 power: development OOF folds 1-4 only; no MILK10k label")
    rows = d5_rows()
    members = []
    for arch in ARCHS:
        path = V1_FILES / f"{arch}_f1-4_nonham_tta.csv"
        if not path.is_file():
            raise SystemExit(f"{rel(path)} missing: run Q9's V1 inference first")
        frame = pd.read_csv(path).set_index("image_id")
        members.append(frame.loc[rows["image_id"], [f"p_{c}" for c in CLASS_7]].to_numpy())
    bands = rows["age_band"].astype(str).to_numpy()
    v1 = v1_decide(np.stack(members, axis=1), bands)
    matrix = pd.read_csv(OOF_MATRIX).set_index("image_id")
    from research.v5.refit_dirichlet import default_model

    model_id = default_model(matrix.columns)
    if DIRICHLET_STATE.is_file() and json.loads(DIRICHLET_STATE.read_text(encoding="utf-8"))["model_id"] != model_id:
        raise SystemExit("the Dirichlet state and the OOF matrix disagree on the V5 model (single view vs TTA): rerun the refits")
    v5_probs = matrix.loc[rows["image_id"], [f"prob_{model_id}_{c}" for c in CLASS_7]].to_numpy()
    w = lesion_weights(rows)
    all_rows = np.ones(len(rows), bool)
    target = float((v1["referred"] * w).sum() / w.sum())
    v5 = v5_decide(v5_probs, bands, target, all_rows, w)
    y_esc = rows["class_7"].isin(ESCALATING).to_numpy()
    mask = (rows["age_band"] == "<40").to_numpy() & y_esc
    caught1 = np.isin(v1["probs"].argmax(1), ESC_IDX) | v1["referred"]
    caught5 = np.isin(v5["probs"].argmax(1), ESC_IDX) | v5["referred"]
    a, b = lesion_caught(rows, caught5, mask), lesion_caught(rows, caught1, mask)
    n = json.loads(S75_JSON.read_text(encoding="utf-8"))["under40_escalating_lesions"]
    result = power_from_lesions(a, b, n, np.random.default_rng(SEED))
    result.update(method="results/v5/s84_power_method.md (D5)", v5_model=model_id,
                  v5_view="24-view TTA" if "tta" in model_id else "single view (no TTA outputs at power time)",
                  matched_budget_R=v5["budget"], v1_referral_lesion_weighted=target,
                  v5_referral_lesion_weighted=v5["matched_rate"], rows=len(rows),
                  lesions=int(rows["effective_lesion_id"].nunique()),
                  archive_split_u40_lesions=rows.loc[mask].groupby("archive")["effective_lesion_id"].nunique().to_dict(),
                  residual_bias_note="V5 Dirichlet and S56 refits include these rows: V5 slightly in-sample, power slightly optimistic")
    write_json(POWER_PATH, result)
    if not result["estimable"]:
        print(result["statement"])
    print(f"power {result['power_plugin']:.3f} (bootstrap median {result['power_bootstrap_median']:.3f}, "
          f"p10-p90 {result['power_bootstrap_p10']:.3f}-{result['power_bootstrap_p90']:.3f}); N={n}; "
          f"{'UNDERPOWERED' if result['underpowered'] else 'adequately powered'}")
    return 0


# =================================================================== stage: cohort (no labels)
def age_band_of(age) -> str:
    if pd.isna(age):
        return "unknown"
    return "<40" if float(age) < 40 else ("40-59" if float(age) < 60 else "60+")


def stage_cohort() -> int:
    """Exclusion exactly as S75 (shares its helpers), with no diagnosis column touched."""
    from research.v5 import s75_milk10k as s75  # CPU stage only: this import hides the GPU
    from research.v4 import dedupe

    meta = pd.read_csv(MILK_META, low_memory=False, usecols=["isic_id", "lesion_id", "age_approx", "attribution", "image_type"])
    derm = meta[meta["image_type"].astype(str).str.lower() == "dermoscopic"].drop_duplicates("lesion_id").reset_index(drop=True)
    hashes = dedupe.load_hashes()
    isic = hashes[hashes["archive"] == "isic2019"]
    assert len(isic) == 25_331
    d_ref, p_ref = isic["dhash"].to_numpy(dtype=np.uint64), isic["phash"].to_numpy(dtype=np.uint64)
    isic_ids = set(isic["image_id"].astype(str))
    excluded, names_out = [], []
    with zipfile.ZipFile(MILK_ZIP) as z:
        names = {n.rsplit("/", 1)[-1].rsplit(".", 1)[0]: n for n in z.namelist() if n.lower().endswith(".jpg")}
        for iid in derm["isic_id"].astype(str):
            names_out.append(names.get(iid, ""))
            dh, ph = s75.hash_bytes(z.read(names[iid])) if iid in names else (0, 0)
            hit = bool(((s75.popcount(d_ref ^ np.uint64(dh)) <= s75.RADIUS) & (s75.popcount(p_ref ^ np.uint64(ph)) <= s75.RADIUS)).any()) if (dh or ph) else False
            excluded.append(iid in isic_ids or hit)
    derm["zip_name"], derm["excluded"] = names_out, excluded
    derm["age_band"] = derm["age_approx"].map(age_band_of)
    eligible = derm[~derm["excluded"]].reset_index(drop=True)
    expected = json.loads(S75_JSON.read_text(encoding="utf-8"))["eligible_lesions"]
    if len(eligible) != expected:
        raise SystemExit(f"eligible lesions {len(eligible)} != S75's {expected}: the cohort is not the registered one")
    S84_DIR.mkdir(parents=True, exist_ok=True)
    eligible[["isic_id", "lesion_id", "age_band", "attribution", "zip_name"]].to_csv(COHORT_PATH, index=False)
    print(f"cohort: {len(eligible):,} eligible lesions -> {rel(COHORT_PATH)}")
    return 0


# =================================================================== stage: infer (GPU, no labels)
def stage_infer(device_name: str, batch_size: int, workers: int) -> int:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import io

    from ml.training.common import build_model_from_checkpoint, resolve_device
    from research.ensembling.data import ARCHS
    from research.tta.transforms import build_tta_views
    from research.v5.tta_v5 import load_composite, tta_pass

    class MilkViews(Dataset):
        def __init__(self, names: list[str]) -> None:
            self.names, self._zip, self._views = names, None, None

        def __len__(self) -> int:
            return len(self.names)

        def __getitem__(self, i: int):
            if self._zip is None:
                self._zip, self._views = zipfile.ZipFile(MILK_ZIP), build_tta_views(224)
            image = Image.open(io.BytesIO(self._zip.read(self.names[i]))).convert("RGB")
            return torch.stack([v.transform(image) for v in self._views]), 0  # no label exists at this stage

    device = resolve_device(device_name)
    cohort = pd.read_csv(COHORT_PATH)
    models, v5_names = {}, []
    for path in composite_checkpoints_available():
        model, payload = load_composite(path, device)
        if tuple(payload["class_codes"]) != tuple(CLASS_7) or payload["image_size"] != 224:
            raise SystemExit(f"{path.name}: class order / image size differ from the registered ones")
        models[f"v5:{payload['run_id']}"] = model
        v5_names.append(payload["run_id"])
    if not v5_names:
        raise SystemExit("no composite checkpoints found")
    for arch in ARCHS:
        model, payload = build_model_from_checkpoint(CKPT_DIR / f"{arch}_best.HAM-only.pt", device)
        if payload.get("image_size", 224) != 224 or tuple(payload["class_codes"]) != tuple(CLASS_7):
            raise SystemExit(f"{arch}: image size / class order differ from the registered ones (as v1_infer asserts)")
        models[f"v1:{arch}"] = model
    loader = DataLoader(MilkViews(cohort["zip_name"].tolist()), batch_size=batch_size, shuffle=False, num_workers=workers)
    result = tta_pass(models, loader, device, "S84 TTA", views_per_forward=48)
    S84_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INFERENCE_PATH.with_name("inference.tmp.npz")
    np.savez(tmp, isic_id=cohort["isic_id"].astype(str).to_numpy(), v5_names=np.array(v5_names),
             v5=np.stack([result["pooled"][f"v5:{n}"] for n in v5_names]),
             v1_names=np.array(ARCHS), v1=np.stack([result["pooled"][f"v1:{a}"] for a in ARCHS], axis=1))
    tmp.replace(INFERENCE_PATH)
    print(f"{len(v5_names)} V5 models + {len(ARCHS)} V1 models on {len(cohort):,} lesions -> {rel(INFERENCE_PATH)}")
    return 0


def composite_checkpoints_available() -> list[Path]:
    """The 15 composite `_last.pt` that exist. Section 9: 'if s43/s44 are incomplete at S84 time, the completed
    seeds, stated'. Only whole seeds count, so a half-finished seed never enters the average."""
    out = []
    for seed in (42, 43, 44):
        paths = [CKPT_DIR / f"convnext_tiny-v5_composite_f{f}_s{seed}_in22k_{'v5stack' if f == 0 else 'v5conf'}_last.pt"
                 for f in range(5)]
        if all(p.is_file() for p in paths):
            out.extend(paths)
    return out


# =================================================================== stage: analyse (the one label read)
def map_class(row: pd.Series) -> str | None:
    from research.v5.young_data import CLASS_RULES

    for field, needle, code in CLASS_RULES + EXTRA_RULES:
        if needle.lower() in str(row.get(field, "") or "").lower():
            return code
    return None


def read_labels(cohort: pd.DataFrame, receipt: dict | None) -> pd.DataFrame:
    """THE label read. Increments the receipt's label_read_count first."""
    if receipt is not None:
        mark_label_read(receipt)
    meta = pd.read_csv(MILK_META, low_memory=False)
    meta = meta[meta["image_type"].astype(str).str.lower() == "dermoscopic"].drop_duplicates("lesion_id")
    merged = cohort.merge(meta[["isic_id", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]], on="isic_id", how="left")
    merged["class_7"] = merged.apply(map_class, axis=1)
    return merged


def sens_block(caught: np.ndarray, y_esc: np.ndarray, mask: np.ndarray) -> dict:
    pos = mask & y_esc
    k, n = int(caught[pos].sum()), int(pos.sum())
    lo, hi = clopper_pearson(k, n)
    return {"caught": k, "n_escalating": n, "sensitivity": (k / n) if n else None, "ci95_clopper_pearson": [lo, hi]}


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, average="macro", labels=range(7), zero_division=0))


def analyse_core(data: pd.DataFrame, v5: dict, v1: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """data: one row per eligible lesion with `class_7` (None = out of scope), `age_band`, `attribution`.
    v5 / v1: {"probs": calibrated (N,7), "referred": bool (N,)}. Pure function: no file, no receipt."""
    from sklearn.metrics import roc_auc_score
    from research.v5.d0_brainstorm_diagnostics import pauc

    inscope = data["class_7"].notna().to_numpy()
    y7 = np.array([CLASS_7.index(c) if isinstance(c, str) else -1 for c in data["class_7"]])
    y_esc = np.isin(y7, ESC_IDX)
    bands = data["age_band"].astype(str).to_numpy()
    pred = {"V5": v5["probs"].argmax(1), "V1": v1["probs"].argmax(1)}
    referred = {"V5": np.asarray(v5["referred"], bool), "V1": np.asarray(v1["referred"], bool)}
    caught = {k: np.isin(pred[k], ESC_IDX) | referred[k] for k in pred}
    u40 = (bands == "<40") & inscope

    # primary: paired, lesion unit, exact one-sided McNemar on discordant <40 escalating lesions, at MATCHED referral
    pos = u40 & y_esc
    n10 = int((caught["V5"][pos] & ~caught["V1"][pos]).sum())
    n01 = int((~caught["V5"][pos] & caught["V1"][pos]).sum())
    p = exact_mcnemar_p(n10, n01)
    primary = {"n_escalating_u40_lesions": int(pos.sum()), "V5": sens_block(caught["V5"], y_esc, u40),
               "V1": sens_block(caught["V1"], y_esc, u40), "discordant_V5_only": n10, "discordant_V1_only": n01,
               "delta_sensitivity": (n10 - n01) / max(int(pos.sum()), 1),
               "mcnemar_exact_one_sided_p": p, "alpha": ALPHA, "reject_V5_better": bool(p <= ALPHA and n10 > n01),
               "referral_all_eligible_matched": {"V5": float(referred["V5"].mean()), "V1": float(referred["V1"].mean())},
               "referral_matched": bool(abs(referred["V5"].mean() - referred["V1"].mean()) <= MATCH_TOLERANCE)}
    # key secondary: the S70 contract, scored at V5's OWN deployed policy (band 0.20), not at the matched point
    deployed = np.asarray(v5["deployed_referred"], bool)
    caught_dep = np.isin(pred["V5"], ESC_IDX) | deployed
    dep_u40 = sens_block(caught_dep, y_esc, u40)
    lo = dep_u40["ci95_clopper_pearson"][0]
    dep_referral = float(deployed.mean())
    secondary = {"contract": f"<40 system sensitivity >= {S70_SENS} and referral <= {S70_REFERRAL}, V5 deployed S56 band 0.20",
                 "v5_deployed_u40": dep_u40, "v5_deployed_referral_all_eligible": dep_referral,
                 "v5_deployed_referral_u40": float(deployed[u40].mean()) if u40.any() else None,
                 "point_estimate_meets": bool(dep_u40["sensitivity"] is not None and dep_u40["sensitivity"] >= S70_SENS
                                              and dep_referral <= S70_REFERRAL),
                 "lower_bound_meets_sensitivity": bool(lo >= S70_SENS) if lo == lo else False,
                 "relative_term": {"quantity": "V5 minus V1 <40 system sensitivity at matched referral (S70 D2)",
                                   "mcid": S70_RELATIVE_MCID, "delta": primary["delta_sensitivity"],
                                   "meets_mcid": bool(primary["delta_sensitivity"] >= S70_RELATIVE_MCID)}}
    # global: Macro-F1 non-inferiority, bootstrap over lesions
    f1 = {k: macro_f1(y7[inscope], pred[k][inscope]) for k in pred}
    rng = np.random.default_rng(seed)
    idx_all = np.flatnonzero(inscope)
    deltas = []
    for _ in range(n_boot):
        i = rng.choice(idx_all, len(idx_all))
        deltas.append(macro_f1(y7[i], pred["V5"][i]) - macro_f1(y7[i], pred["V1"][i]))
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    global_ = {"macro_f1_V5": f1["V5"], "macro_f1_V1": f1["V1"], "delta": f1["V5"] - f1["V1"], "ci95_bootstrap": ci,
               "margin": F1_MARGIN, "non_inferior": bool(ci[0] > -F1_MARGIN), "n_boot": n_boot, "rng_seed": seed,
               "retained_macro_f1": {k: macro_f1(y7[inscope & ~referred[k]], pred[k][inscope & ~referred[k]]) for k in pred}}
    # descriptive
    def auc_block(mask: np.ndarray) -> dict:
        out = {}
        for k, blob in (("V5", v5), ("V1", v1)):
            s, y = esc_mass(blob["probs"])[mask], y_esc[mask]
            ok = 0 < y.sum() < len(y)
            out[k] = {"auroc": float(roc_auc_score(y, s)) if ok else None, "pauc_0.20": float(pauc(y, s)) if ok else None,
                      "n": int(mask.sum()), "n_escalating": int(y.sum())}
        return out
    descriptive = {"by_age_band": {"ALL": auc_block(inscope), **{b: auc_block(inscope & (bands == b)) for b in BANDS if (inscope & (bands == b)).any()}},
                   "by_institution": {}, "out_of_scope": {"n": int((~inscope).sum()),
                                                          "referral_V5": float(referred["V5"][~inscope].mean()) if (~inscope).any() else None,
                                                          "referral_V1": float(referred["V1"][~inscope].mean()) if (~inscope).any() else None}}
    for inst, grp in data[inscope].groupby("attribution"):
        m = np.zeros(len(data), bool)
        m[grp.index.to_numpy()] = True
        descriptive["by_institution"][str(inst)] = {"n": int(m.sum()), "u40_V5": sens_block(caught["V5"], y_esc, m & u40),
                                                    "u40_V1": sens_block(caught["V1"], y_esc, m & u40)}
    return {"primary": primary, "key_secondary": secondary, "global": global_, "descriptive": descriptive}


def stage_analyse(receipt: dict | None) -> int:
    from research.ensembling.data import ARCHS
    from research.v5 import refit_s56 as rs

    cohort = pd.read_csv(COHORT_PATH)
    inference = np.load(INFERENCE_PATH, allow_pickle=False)
    assert inference["isic_id"].astype(str).tolist() == cohort["isic_id"].astype(str).tolist(), "cohort / inference mismatch"
    assert list(inference["v1_names"]) == list(ARCHS)
    # BOTH systems' decisions are fixed here, from probabilities, age bands and the frozen policies alone, before the
    # labels are read, so they provably cannot depend on them. Referral is matched over ALL eligible lesions.
    bands = cohort["age_band"].astype(str).to_numpy()
    v1 = v1_decide(inference["v1"], bands)
    target = float(v1["referred"].mean())
    v5 = attach_deployed(v5_decide(inference["v5"].mean(axis=0), bands, target, np.ones(len(cohort), bool)), bands)
    data = read_labels(cohort, receipt)  # the one label read
    s75 = json.loads(S75_JSON.read_text(encoding="utf-8"))
    inscope = data["class_7"].notna().to_numpy()
    u40_esc = int((inscope & (data["age_band"] == "<40") & data["class_7"].isin(ESCALATING)).sum())
    if int(inscope.sum()) != s75["in_scope_lesions"] or u40_esc != s75["under40_escalating_lesions"]:
        raise SystemExit(f"label mapping disagrees with S75: in-scope {int(inscope.sum())} vs {s75['in_scope_lesions']}, "
                         f"<40 escalating {u40_esc} vs {s75['under40_escalating_lesions']}")
    report = analyse_core(data, v5, v1)
    report.update(power=json.loads(POWER_PATH.read_text(encoding="utf-8")),
                  v5_models=[str(n) for n in inference["v5_names"]], v5_n_models=int(len(inference["v5_names"])),
                  v5_budget_matched=v5["budget"], v5_referral_matched=v5["matched_rate"], v1_referral_target=target,
                  v1_policy=v1["policy"], cohort_label="cohort-external, partly same-institution",
                  in_scope_lesions=int(inscope.sum()), out_of_scope_lesions=int((~inscope).sum()),
                  underpowered_statement=("UNDERPOWERED: planned power "
                                          f"{report['power']['power_plugin']:.2f} < 0.80" if report["power"]["underpowered"] else "power >= 0.80"))
    write_json(REPORT_PATH, report)
    pd.DataFrame({"isic_id": data["isic_id"], "V5_pred": v5["probs"].argmax(1), "V5_referred": v5["referred"],
                  "V1_pred": v1["probs"].argmax(1), "V1_referred": v1["referred"]}).to_csv(S84_DIR / "decisions.csv", index=False)
    print(json.dumps({k: report[k] for k in ("primary", "key_secondary", "global")}, indent=2, default=_plain))
    return 0


# =================================================================== orchestration
def dry_run() -> int:
    ok = True
    for name, path in frozen_inputs().items():
        present = path.is_file()
        ok &= present
        print(f"{'ok     ' if present else 'MISSING'} {name}: {rel(path)}")
    available = composite_checkpoints_available()
    print(f"composite checkpoints (whole seeds): {len(available)} of 15")
    try:
        require_tta_state()
        print("ok      V5 Dirichlet / S56 state is the TTA model and matches the OOF matrix")
    except SystemExit as error:
        ok = False
        print(f"NOT READY {error}")
    print(f"receipt: n_executions = {load_receipt()['n_executions']} (a value >= 1 blocks --execute)")
    print("no label read, no model loaded")
    return 0 if ok and len(available) >= 5 and load_receipt()["n_executions"] == 0 else 1


def execute(resume: bool, device: str, batch_size: int, workers: int) -> int:
    from research import testguard

    testguard.block_test_reads("S84: MILK10k only; HAM test, reserved and development labels untouched")
    require_tta_state()  # before the receipt: a failure after the start would burn the single execution
    receipt = begin_execution(resume)  # n_executions incremented here, before any scoring
    env = {**os.environ}
    for stage in ("cohort", "infer"):
        out = COHORT_PATH if stage == "cohort" else INFERENCE_PATH
        if out.is_file():
            print(f"stage {stage}: {rel(out)} exists, skipping")
            continue
        args = [sys.executable, "-m", "research.v5.s84_milk10k", "--stage", stage, "--device", device,
                "--batch-size", str(batch_size), "--num-workers", str(workers)]
        if subprocess.run(args, cwd=REPO_ROOT, env=env).returncode != 0:
            raise SystemExit(f"stage {stage} failed before any label was read; fix and run with --resume")
    stage_analyse(receipt)
    complete_execution(receipt, [REPORT_PATH, S84_DIR / "decisions.csv", COHORT_PATH, INFERENCE_PATH])
    return 0


def smoke() -> int:
    """The analysis path on synthetic data; nothing is read, nothing is receipted."""
    rng = np.random.default_rng(0)
    n = 600
    classes = rng.choice(list(CLASS_7) + [None], n, p=[.05, .06, .12, .01, .1, .6, .03, .03])
    data = pd.DataFrame({"class_7": classes, "age_band": rng.choice(["<40", "40-59", "60+", "unknown"], n),
                         "attribution": rng.choice(["Vienna", "MSKCC", "Other"], n)})

    def fake():
        p = rng.dirichlet(np.ones(7), n)
        return {"probs": p, "referred": rng.random(n) < 0.2, "deployed_referred": rng.random(n) < 0.2}
    report = analyse_core(data, fake(), fake(), n_boot=50)
    write_json(S84_DIR / "smoke" / "s84_smoke_report.json", report)
    print("smoke ok:", sorted(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["cohort", "infer"], help="internal: run by --execute")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.stage and load_receipt()["status"] != "started":
        parser.error("--stage is internal: it runs only inside a started --execute (no MILK10k image is scored outside the receipt)")
    if args.stage == "cohort":
        return stage_cohort()
    if args.stage == "infer":
        return stage_infer(args.device, args.batch_size, args.num_workers)
    if args.power:
        return run_power()
    if args.dry_run:
        return dry_run()
    if args.smoke:
        return smoke()
    if args.execute:
        return execute(args.resume, args.device, args.batch_size, args.num_workers)
    parser.error("give one of --power, --dry-run, --smoke, --execute")


if __name__ == "__main__":
    raise SystemExit(main())
