"""S60 -- the inference contract: decision logic behind the product's API, model-free.

V4 runbook Sec.5, S60: "the response payload carries the whole decision, not just a label
... a prediction that cannot state its own guarantee is not deployable." This module is
that guarantee, kept separate from the FastAPI wiring in `s60_api.py` and from the model
so it can be unit-tested against known frozen numbers without a GPU, a model, or ONNX.
(`torch` is still imported, CPU-only, by `research.calibration.methods` via `frozen_params`.)
The ONNX export is deferred: S54 and S53r kept the V1 six-CNN ensemble as the base, and
exporting six backbones plus 24-view TTA is a packaging job, not a research one.

S59 froze the deployed stack as **S56 alone** (neither the lambda layer nor the conformal
layer beat S56 at matched workload) and measured its contract on the reserved cohort:
every term NOT_MET, joint CONTRACT_FAILS. `load_contract_info` refuses to start if S59's
report names a different stack, and every response carries that external verdict.

**What is deployed vs. what is informational.** Only two policies are frozen and adopted
end-to-end: the six-CNN uniform soft-vote V1 ensemble with the deployed global Dirichlet map
(`frozen_params.calibrate`), and S56's per-band CRC-floor abstention on the `msp` score at the
primary budget R=0.20 (`results/v4/s56/policies_oof.json:band_0.2`). Two more frozen artifacts
are surfaced for transparency but are **not** stacked into the decision:

* the per-band lambda(age) rule (S5, reproduced via `frozen_params.apply_age_rule`) -- shown
  as `lambda_rule` for context, never used to pick the returned class, because S65 measured
  that stacking it under S55's per-band calibration undoes it and S59 (the module that would
  compose it properly) is still a draft pending the S53r base-model readout;
* the bipartite RAPS(alpha=0.05) conformal set (S6/S9) -- a safety net that can flag
  "no escalating class in the set" independently of the abstention call, per
  `research/conformal/hierarchical.py`'s own logic, reproduced here with the k_reg/penalty
  the session6 log recorded (k_reg=1, lambda=0.05) since the fit_state does not persist them.

**What is out of scope for a stubbed model.** OOD/Mahalanobis distance needs the ConvNeXt
backbone features (`research/selective/features/*.npz`), age estimation needs the same
features and is separately documented as unreliable under 40 (S57a: +19.5y bias), and
Grad-CAM needs the model and the image. All three fields are present in the contract but
`None` with a stated reason until the model is wired in -- a caller must not be able to
mistake "not computed tonight" for "computed and clean".

**Read-once discipline.** Nothing here reads HAM test or the reserved cohort; every number
loaded is a previously frozen artifact (`results/v4/s56/*.json`, `research/agerule/...`,
`research/conformal/results_hierarchical_oof/fit_state.json`, `research/multical/...`).
A live request is not a "read" in the project's sense -- it scores one new case against
fixed, already-registered thresholds -- so it does not touch any receipt.

    $py -m research.v4.s60_contract --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.paths import load_class_mapping, resolve
from research.external import frozen_params as fp

# ---------------------------------------------------------------------- frozen artifact paths
S56_POLICIES = "results/v4/s56/policies_oof.json"
S56_PLAN = "results/v4/s56_plan.json"
S56_FRONTIER_OOF = "results/v4/s56/frontier_oof_insample.csv"
HIERARCHICAL_FIT = "research/conformal/results_hierarchical_oof/fit_state.json"
S59_PLAN = "results/v4/s59_plan.json"
S59_REPORT = "results/v4/s59/s59_report.json"
PRIMARY_BUDGET = 0.20
RAPS_K_REG = 1          # session6_logs/S6_hierarchical.log: "alpha=0.05 ... k_reg=1, lambda=0.05"
RAPS_PENALTY = 0.05
NNB_PREVALENCES = (0.01, 0.03, 0.05)

FITZPATRICK_NOTICE = (
    "Fitzpatrick I-IV spread is 0.102 and non-monotonic (I 0.349, II 0.287, III 0.247, "
    "IV 0.278; n=59 at IV). Types V/VI are suppressed for insufficient sample (n=8/1). "
    "This system has NO EVIDENCE for darker skin tones (Fitzpatrick V/VI) and must not be "
    "presented as validated for them (S8b, S19)."
)
UNDER40_NOTICE = (
    "Under-40 escalation ranking is a measured ceiling, not a fixable decision-rule gap "
    "(S51 backbone probe, S64 ceiling analysis, S67 ranking probes all failed to close it; "
    "under-40 AUC ~0.88-0.90 vs ~0.93-0.95 in older bands). Abstention (S56) reallocates "
    "referral budget toward this band; it does not create ranking signal that is not there."
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ClassInfo:
    codes: tuple[str, ...]
    escalating_idx: tuple[int, ...]

    @property
    def escalating_codes(self) -> tuple[str, ...]:
        return tuple(self.codes[i] for i in self.escalating_idx)

    def is_escalating(self, idx: int) -> bool:
        return idx in self.escalating_idx


def load_class_info() -> ClassInfo:
    mapping = load_class_mapping()
    codes = mapping.codes
    esc = tuple(c.index for c in mapping.classes if c.malignancy in ("malignant", "premalignant"))
    return ClassInfo(codes=codes, escalating_idx=esc)


@dataclass(frozen=True)
class AbstentionPolicy:
    score: str
    budget: float
    floor: float
    tau_band: dict[str, float]
    tau_global: float

    def threshold_for(self, band: str) -> float:
        return min(self.tau_band.get(band, float("inf")), self.tau_global)


def load_abstention_policy(budget: float = PRIMARY_BUDGET) -> AbstentionPolicy:
    policies = _load_json(S56_POLICIES)
    key = f"band_{budget}"
    if key not in policies:
        raise KeyError(f"{key!r} not in {S56_POLICIES}; available: {sorted(policies)}")
    p = policies[key]
    return AbstentionPolicy(
        score=p["score"], budget=p["budget"], floor=p["floor"],
        tau_band=dict(p["tau_band"]), tau_global=p["tau_global"],
    )


@dataclass(frozen=True)
class BipartiteConformal:
    """One (band x escalating/benign) RAPS quantile grid, S6/S9's safety net."""

    class_info: ClassInfo
    band_order: tuple[str, ...]
    cell_quantiles: dict[str, float]   # "<40|benign" -> quantile

    def quantile_row(self, band: str) -> np.ndarray:
        """(C,) quantile per class for this band, benign/escalating looked up per class."""
        lookup_band = band if band in self.band_order else "unknown"
        out = np.empty(len(self.class_info.codes), dtype=np.float64)
        for i in range(len(out)):
            group = "escalating" if self.class_info.is_escalating(i) else "benign"
            out[i] = self.cell_quantiles[f"{lookup_band}|{group}"]
        return out

    def prediction_set(self, calibrated_probs: np.ndarray, band: str) -> list[str]:
        scores = raps_scores_single(calibrated_probs, self.class_info.escalating_idx,
                                     k_reg=RAPS_K_REG, penalty=RAPS_PENALTY)
        q = self.quantile_row(band)
        members = np.where(scores <= q)[0]
        return [self.class_info.codes[i] for i in members]


def raps_scores_single(probs: np.ndarray, escalating_idx: tuple[int, ...],
                        *, k_reg: int, penalty: float) -> np.ndarray:
    """Deterministic (non-randomized) RAPS score for every candidate class of one row.

    Non-randomized (`u=1`) is deliberately conservative: real coverage is >= nominal and
    sets may be a touch larger than the exact randomized construction the frozen quantiles
    were calibrated with. Chosen because a live API endpoint should be reproducible for the
    same input, and the gap is second-order next to the k_reg/penalty choice itself.
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    ranks = np.argsort(order)                 # 0-indexed rank of each class
    cumulative_including = cumulative[ranks]
    scores = cumulative_including - probs + probs   # u=1 -> cumulative_including exactly
    if penalty > 0.0:
        scores = scores + penalty * np.maximum(0, (ranks + 1) - k_reg)
    return scores


def load_bipartite_conformal() -> BipartiteConformal:
    state = _load_json(HIERARCHICAL_FIT)
    cell = state["states"]["RAPS_bipartite_a05"]["cell_quantiles"]
    quantiles = {k: (float("inf") if v == "Infinity" else float(v)) for k, v in cell.items()}
    return BipartiteConformal(
        class_info=load_class_info(),
        band_order=tuple(state["band_order"]),
        cell_quantiles=quantiles,
    )


@dataclass(frozen=True)
class ContractInfo:
    """Static policy metadata returned by `/contract` and embedded in every `/predict`."""

    plan_sha256: str
    primary_budget: float
    floor_oof: float
    abstention_score: str
    decision_rule: str
    route: str
    fairness_notice: str
    under40_notice: str
    known_limitations: list[str]
    band_operating_point: dict[str, dict[str, float]]  # band -> {referral_rate, system_sens, nnb_pi*, retained_macro_f1}
    s59_plan_sha256: str
    s59_stack: str
    external_contract: dict[str, Any]   # S59's reserved-cohort measurement of this contract


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_band_operating_point(budget: float = PRIMARY_BUDGET) -> dict[str, dict[str, float]]:
    """Per-band OOF operating characteristics at the deployed budget (band arm).

    Read with the standard-library csv module, not pandas, so this module has no heavy
    dependency beyond numpy -- the FastAPI process should start fast and cold.
    """
    import csv
    path = resolve(S56_FRONTIER_OOF)
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != "band" or float(row["budget"]) != budget:
                continue
            band = row["band"]
            if band == "ALL":
                band = "all_ages"
            out[band] = {
                "referral_rate": float(row["referral_rate"]),
                "system_sens": float(row["system_sens"]),
                "retained_macro_f1": float(row["retained_macro_f1"]),
                "nnb_pi0.01": float(row["nnb_pi0.01"]),
                "nnb_pi0.03": float(row["nnb_pi0.03"]),
                "nnb_pi0.05": float(row["nnb_pi0.05"]),
            }
    return out


def load_external_contract() -> tuple[str, str, dict[str, Any]]:
    """S59's frozen verdict: which stack was adopted, and how its contract held on reserved."""
    report = _load_json(S59_REPORT)
    c = report["contract"]
    terms = {t["term"]: {"value": t["value"], "target": t["target"], "direction": t["direction"],
                         "ci": [t.get("ci_lo"), t.get("ci_hi")], "status": t["status"]}
             for t in c["terms"]}
    return report["stack"], report["plan_sha256"], {
        "panel": c["panel"], "deployed_arm": c["deployed_arm"], "joint": c["joint"],
        # S59's `coverage` is 1 - referral rate (the share decided without referral), not a
        # joint bootstrap pass rate.
        "selective_coverage": c["coverage"], "terms": terms, "source": S59_REPORT,
    }


def load_contract_info(budget: float = PRIMARY_BUDGET) -> ContractInfo:
    policy = load_abstention_policy(budget)
    stack, s59_sha, external = load_external_contract()
    if _sha256_file(resolve(S59_PLAN)) != s59_sha:
        raise RuntimeError(f"{S59_PLAN} does not match the plan hash in {S59_REPORT}")
    if stack != "S56" or external["deployed_arm"] != f"S56@{budget}":
        raise RuntimeError(f"S59 adopted {external['deployed_arm']!r}; this service implements "
                           f"S56@{budget} only")
    return ContractInfo(
        plan_sha256=_sha256_file(resolve(S56_PLAN)),
        primary_budget=policy.budget,
        floor_oof=policy.floor,
        abstention_score=policy.score,
        decision_rule=(
            "argmax over the deployed global-Dirichlet-calibrated V1 soft-vote ensemble; "
            "the per-band lambda(age) rule is computed and reported for transparency but is "
            "NOT stacked into this decision (S65/S66: REJECT-cost when composed with abstention)"
        ),
        route="ham_ensemble_v1 (fixed single route; S58's admissibility gate and domain "
              "router both failed their own gates and are not deployed)",
        fairness_notice=FITZPATRICK_NOTICE,
        under40_notice=UNDER40_NOTICE,
        known_limitations=[
            "OOF-fitted operating characteristics (referral rate, system sensitivity, NNB) "
            "are in-distribution figures. On the reserved external cohort (BCN/MSKCC) S59 "
            "measured this exact contract end-to-end and every term failed (see "
            "external_contract) -- treat band_operating_point as nominal for HAM-like "
            "dermoscopy only, not a transferable guarantee.",
            "Deployed 40-59 lambda violates its own 0.85 specificity floor on reserved "
            "(0.748 [0.718, 0.775]) -- S57a.",
            UNDER40_NOTICE,
        ],
        band_operating_point=load_band_operating_point(budget),
        s59_plan_sha256=s59_sha,
        s59_stack=stack,
        external_contract=external,
    )


# ------------------------------------------------------------------------------- decision
@dataclass(frozen=True)
class AgeInfo:
    value: float | None
    source: str    # "recorded" | "unknown"
    band: str


@dataclass(frozen=True)
class Decision:
    predicted_class: str
    predicted_index: int
    probs_calibrated: list[float]
    age: AgeInfo
    lambda_used: float
    lambda_rule_class: str
    abstain: bool
    abstention_threshold: float
    abstention_score_value: float
    conformal_set: list[str]
    conformal_set_covers_escalation: bool
    false_reassurance_risk: bool
    ood_score: float | None
    ood_available: bool
    gradcam: None
    nnb: dict[str, float] | None
    contract: ContractInfo

    def to_dict(self) -> dict[str, Any]:
        d = {
            "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index,
            "probs_calibrated": self.probs_calibrated,
            "age": {"value": self.age.value, "source": self.age.source, "band": self.age.band},
            "lambda_used": self.lambda_used,
            "lambda_rule_class": self.lambda_rule_class,
            "abstain": self.abstain,
            "abstention_threshold": self.abstention_threshold,
            "abstention_score_value": self.abstention_score_value,
            "conformal_set": self.conformal_set,
            "conformal_set_covers_escalation": self.conformal_set_covers_escalation,
            "false_reassurance_risk": self.false_reassurance_risk,
            "ood": {"score": self.ood_score, "available": self.ood_available,
                    "reason": None if self.ood_available else
                    "Mahalanobis distance needs backbone features; model is stubbed tonight"},
            "gradcam": {"overlay": self.gradcam, "reason":
                        "needs the model and the source image; not available in the stub"},
            "nnb": self.nnb,
            "contract": {
                "plan_sha256": self.contract.plan_sha256,
                "primary_budget": self.contract.primary_budget,
                "floor_oof": self.contract.floor_oof,
                "abstention_score": self.contract.abstention_score,
                "decision_rule": self.contract.decision_rule,
                "route": self.contract.route,
                "fairness_notice": self.contract.fairness_notice,
                "under40_notice": self.contract.under40_notice,
                "known_limitations": self.contract.known_limitations,
                "s59_plan_sha256": self.contract.s59_plan_sha256,
                "s59_stack": self.contract.s59_stack,
                "external_contract": {
                    "joint": self.contract.external_contract["joint"],
                    "selective_coverage": self.contract.external_contract["selective_coverage"],
                    "source": self.contract.external_contract["source"],
                },
            },
        }
        return d


class DecisionEngine:
    """Holds every frozen artifact once; scores requests with no per-request file I/O."""

    def __init__(self) -> None:
        self.class_info = load_class_info()
        self.lambda_by_band = fp.load_lambda_by_band()
        self.abstention = load_abstention_policy()
        self.conformal = load_bipartite_conformal()
        self.contract = load_contract_info()
        self._band_op = self.contract.band_operating_point

    def decide(self, probs: list[float], *, age: float | None, calibrated: bool) -> Decision:
        raw = np.asarray(probs, dtype=np.float64)
        if raw.shape != (len(self.class_info.codes),):
            raise ValueError(f"expected {len(self.class_info.codes)} class probabilities, "
                              f"got shape {raw.shape}")
        if raw.min() < 0 or not np.isfinite(raw).all():
            raise ValueError("probabilities must be finite and non-negative")
        total = raw.sum()
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"probabilities must sum to ~1.0, got {total:.4f}")
        raw = raw / total

        cal = raw if calibrated else fp.calibrate(raw[None, :])[0]
        cal = np.clip(cal, 0.0, 1.0)
        cal = cal / cal.sum()

        band = fp.age_bands(np.array([age if age is not None else np.nan]))[0]
        age_info = AgeInfo(value=age, source="recorded" if age is not None else "unknown",
                            band=band)

        pred_idx = int(np.argmax(cal))
        pred_class = self.class_info.codes[pred_idx]

        lam = self.lambda_by_band.get(band, self.lambda_by_band["pooled"])
        lam_idx = int(fp.apply_age_rule(cal[None, :], bands=np.array([band]),
                                         lam=self.lambda_by_band)[0])
        lam_class = self.class_info.codes[lam_idx]

        msp = 1.0 - float(cal.max())
        threshold = self.abstention.threshold_for(band)
        abstain = bool(msp >= threshold)

        conformal_set = self.conformal.prediction_set(cal, band)
        covers = any(c in self.class_info.escalating_codes for c in conformal_set)
        false_reassurance_risk = (not abstain) and (pred_class not in self.class_info.escalating_codes) and (not covers)

        band_key = band if band in self._band_op else ("unknown" if "unknown" in self._band_op else None)
        nnb = None
        if band_key is not None:
            row = self._band_op[band_key]
            nnb = {f"pi_{p}": row[f"nnb_pi{p}"] for p in NNB_PREVALENCES}

        return Decision(
            predicted_class=pred_class,
            predicted_index=pred_idx,
            probs_calibrated=[round(float(x), 6) for x in cal],
            age=age_info,
            lambda_used=float(lam),
            lambda_rule_class=lam_class,
            abstain=abstain,
            abstention_threshold=float(threshold),
            abstention_score_value=float(msp),
            conformal_set=conformal_set,
            conformal_set_covers_escalation=covers,
            false_reassurance_risk=false_reassurance_risk,
            ood_score=None,
            ood_available=False,
            gradcam=None,
            nnb=nnb,
            contract=self.contract,
        )


# ---------------------------------------------------------------------------- self-test
def _selftest() -> None:
    engine = DecisionEngine()
    ci = engine.class_info
    assert ci.codes == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"), ci.codes
    assert ci.escalating_codes == ("akiec", "bcc", "mel"), ci.escalating_codes

    # policies_oof.json:band_0.2 reproduced exactly
    assert abs(engine.abstention.tau_band["<40"] - 0.07148993334109954) < 1e-9
    assert abs(engine.abstention.tau_global - 0.37250286262959076) < 1e-9
    assert engine.abstention.floor == 0.855

    # a clean, confident nv call in an older patient: no referral, no lambda flip
    probs = [0.01, 0.01, 0.02, 0.01, 0.02, 0.90, 0.03]
    d = engine.decide(probs, age=65, calibrated=True)
    assert d.predicted_class == "nv", d.predicted_class
    assert d.age.band == "60+"
    assert not d.abstain, "confident nv call should not be referred"
    assert "nv" in d.conformal_set

    # an unconfident mixed call in a <40 patient with real escalation mass should refer
    probs2 = [0.12, 0.10, 0.15, 0.05, 0.18, 0.30, 0.10]
    d2 = engine.decide(probs2, age=25, calibrated=True)
    assert d2.age.band == "<40"
    assert d2.abstain, "high-entropy <40 call should clear the band-specific floor"

    # unknown age falls back to tau_global and the unknown-band conformal cell
    d3 = engine.decide(probs, age=None, calibrated=True)
    assert d3.age.band == "unknown"
    assert d3.age.source == "unknown"

    # a confident false-negative shape: argmax benign, low uncertainty, no escalating class
    # in the conformal set -> false_reassurance_risk should be able to fire
    probs4 = [0.02, 0.02, 0.03, 0.02, 0.03, 0.85, 0.03]
    d4 = engine.decide(probs4, age=70, calibrated=True)
    assert not d4.abstain
    # sanity: a benign call that IS abstained can never be a false-reassurance risk
    assert not (d4.abstain and d4.false_reassurance_risk)

    contract = engine.contract
    assert contract.floor_oof == 0.855
    assert "all_ages" in contract.band_operating_point
    assert contract.band_operating_point["<40"]["referral_rate"] > 0.3
    assert len(contract.plan_sha256) == 64

    print("s60_contract --selftest: OK (11 checks)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
