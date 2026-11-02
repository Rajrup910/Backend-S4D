"""Every number shown in the README and its GIFs, read from results/ (Hard Rule 4). Nothing here is hand-entered."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _json(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def _csv(path: str) -> list[dict]:
    with (REPO / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sgn(x: float, d: int = 4) -> str:
    """+0.0351 / −0.0121, with a true minus sign."""
    return f"{'+' if x >= 0 else '−'}{abs(x):.{d}f}"


def ci(lo: float, hi: float, d: int = 4) -> str:
    return f"[{sgn(lo, d)}, {sgn(hi, d)}]"


def num(x: float, d: int = 4) -> str:
    return f"{'−' if x < 0 else ''}{abs(x):.{d}f}"


def load() -> dict:
    f: dict = {}

    # ------------------------------------------------------------------ V1
    profile = {r["class_code"]: r for r in _csv("results/review2/ham10000_class_profile.csv")}
    classes = [k for k in profile if not k.startswith("_")]
    f["ham_images"] = sum(int(profile[c]["images"]) for c in classes)
    f["ham_lesions"] = sum(int(profile[c]["lesions"]) for c in classes)
    f["nv_share"] = float(profile["nv"]["image_share"])
    f["test_images"] = sum(int(profile[c]["test_images"]) for c in classes)

    rungs = {r["rung"]: r for r in _csv("results/ablation_table.csv")}
    f["rungs"] = {k.split("_")[0] if k.startswith("A") else k: {
        "f1": float(r["macro_f1"]), "lo": float(r["macro_f1_ci_low"]), "hi": float(r["macro_f1_ci_high"]),
        "ba": float(r["balanced_accuracy"]), "sens": float(r["escalation_sensitivity"]),
        "missed": int(r["missed_serious"]), "n": int(r["n"]), "cov": float(r["coverage"])} for k, r in rungs.items()}

    mc = {(x["a"], x["b"]): x["mcnemar"] for x in _json("results/mcnemar_delong.json")}
    ens = mc[("A5_soft_vote_6cnn", "A2_convnext_tiny")]
    f["ens_chi2"], f["ens_p"], f["ens_only"], f["cnn_only"] = ens["statistic"], ens["p_value"], ens["only_a_correct"], ens["only_b_correct"]

    members = {r["arch"]: r for r in _csv("results/review2/v1_member_metrics.csv")}
    f["maxvit_test"], f["maxvit_val"] = float(members["maxvit_tiny"]["macro_f1"]), float(members["maxvit_tiny"]["val_macro_f1"])
    f["convnext_val"] = float(members["convnext_tiny"]["val_macro_f1"])

    esc = _json("results/review2/escalation_mcnemar_a6_a7.json")
    f["a6a7_lost"], f["a6a7_gained"], f["a6a7_p"], f["serious_total"] = esc["flagged_a6_only"], esc["flagged_a7_only"], esc["p_exact_binomial"], esc["n_serious"]

    prior = {(r["split"], r["age_band"]): float(r["escalating_share"]) for r in _csv("results/age_band_prior.csv")}
    f["prior_u40"], f["prior_40"], f["prior_60"] = prior[("train", "<40")], prior[("train", "40-59")], prior[("train", "60+")]

    report = (REPO / "research/selective/results/session4_report.md").read_text(encoding="utf-8")
    band = {}
    for label in ("<40", "40-59", "60+"):
        m = re.search(rf"^\| {re.escape(label)} \| (\d+) \| (0\.\d+) \| (\d+) \| (\d+) \| ([\d.]+)% \| ([\d.]+)% \|", report, re.M)
        band[label] = {"n": int(m[1]), "sens": float(m[2]), "missed": int(m[3]), "referred": int(m[4]), "referred_pct": float(m[5])}
        band[label]["caught"] = band[label]["n"] - band[label]["missed"]
    f["band"] = band

    oof = {(r["split"], r["band"]): r for r in _csv("research/stats/results_oof/age_gap_intervals.csv")}
    f["oof_u40_sens"], f["oof_u40_caught"], f["oof_u40_n"] = float(oof[("oof", "<40")]["sensitivity"]), int(oof[("oof", "<40")]["n_caught"]), int(oof[("oof", "<40")]["n_escalating"])

    f["receipt_n"] = _json("results/test_pass_receipt.json")["n_executions"]
    power = _json("results/v4/power_audit.json")
    f["u40_test_images"] = power["ham_under40_counts"]["test"]["escalating_images"]
    f["u40_test_lesions"] = power["ham_under40_counts"]["test"]["escalating_lesions"]
    f["power_single"], f["power_paired"] = power["min_n_for_half_width"]["0.10"], power["min_n_paired_mcnemar"]["loss_ratio_0.5"]
    f["u40_oof_lesions"] = power["ham_under40_counts"]["train"]["escalating_lesions"]

    # ------------------------------------------------------------------ V2
    boot = _json("results/v2/bootstrap_intervals.json")
    f["v2_c"] = {r["cohort"]: {"C": r["C"], "lo": r["ci_lo"], "hi": r["ci_hi"], "n": r["n_escalating"]} for r in boot["F1"]["rows"]}
    f["v2_mcid"] = boot["mcid"]
    verdict2 = _json("results/v2/final_verdict.json")
    f["v2_verdicts"] = {k: v["verdict"] for k, v in verdict2["per_hypothesis"].items()}
    h4 = next(r for r in _csv("results/v2/frr_by_group.csv") if r["alpha"] == "0.05" and r["group"] == "all_escalating")
    f["v2_frr"], f["v2_frr_lo"], f["v2_frr_hi"], f["v2_cov"], f["v2_one_minus"] = float(h4["frr"]), float(h4["cp_lo"]), float(h4["cp_hi"]), float(h4["marginal_coverage"]), float(h4["one_minus_marginal_coverage"])
    safety = next(r for r in _csv("results/v2/conformal_subgroup_safety.csv") if r["alpha"] == "0.1")
    f["v2_frr_m_k"], f["v2_frr_b_k"], f["v2_frr_n"], f["v2_frr_holm"] = int(safety["frr_mondrian_k"]), int(safety["frr_bipartite_k"]), int(safety["frr_mondrian_n"]), float(safety["p_holm"])

    # ------------------------------------------------------------------ V3
    verdict3 = _json("results/v3/final_verdict.json")
    f["v3_counts"] = verdict3["verdict_counts"]
    f["v3_best"] = verdict3["headline_ceiling"]["best_v3_val_macro_f1"]
    hyp = {h["id"]: h for h in verdict3["hypotheses"]}
    f["v3_hyp"] = {k: v["verdict"] for k, v in hyp.items()}
    f["v3_h3_auc"], f["v3_h3_ci"] = hyp["H3"]["evidence"]["age_band_auc"], hyp["H3"]["evidence"]["age_band_ci"]
    f["v3_h4"], f["v3_h4_ci"] = hyp["H4"]["evidence"]["delta"], hyp["H4"]["evidence"]["ci"]
    f["v3_h2_all"] = hyp["H2"]["evidence"]["delta_head_by_band"]["all"]
    f["v3_h5_excl"] = hyp["H5"]["evidence"]["zero_shot_gains_excluding_zero"]
    oos = _json("results/v3/oos_probe_report.json")["stages"]
    u40 = next(b for b in oos["stage1"]["out_of_sample"]["bands"] if b["band"] == "<40")
    ins = next(b for b in oos["stage0"]["in_sample"]["bands"] if b["band"] == "<40")
    f["v3_in"], f["v3_out"], f["v3_post"] = ins["np_pauc"], u40["np_pauc"], u40["s_pauc"]
    s44 = _json("results/v3/s44_control_and_multiplicity.json")["control_reproduction"]
    f["v3_seed_spread"], f["v3_cond_spread"] = s44["same_data_spread"]["esc_sens_under40"], s44["between_condition_spread"]["esc_sens_under40"]
    f["v3_seed_a"], f["v3_seed_b"], f["v3_seed_n"] = s44["published"]["under40_caught"], s44["v3_retrain"]["under40_caught"], s44["published"]["under40_n"]
    d2 = _json("results/v3/d2_strong_evaluation.json")
    arms = {a["arm"]: a for a in d2["arms"]}
    f["v3_f1_before"], f["v3_f1_after"] = arms["all_three"]["macro_f1_val"], arms["d2_strong"]["macro_f1_val"]
    f["v3_f1_delta"], f["v3_f1_lo"], f["v3_f1_hi"] = d2["primary"]["delta"], d2["primary"]["ci_lo"], d2["primary"]["ci_hi"]
    f["v3_auc_after"] = d2["under40_escalation_auc"]["per_arm"]["d2_strong"]["pooled"]["auc"]
    f["v3_auc_before"] = d2["under40_escalation_auc"]["per_arm"]["all_three"]["pooled"]["auc"]
    f["v3_auc_delta"] = d2["under40_escalation_auc"]["pooled_delta_d2_minus_baseline"]

    # ------------------------------------------------------------------ V4
    gate = _json("results/v4/s54/s54_gate.json")
    last = gate["checkpoints"]["last"]["gate_b_decision"]
    f["v4_cohort"] = gate["cohort"]
    f["v4_b_f1"], f["v4_b_f1_lo"], f["v4_b_f1_hi"] = last["macro_f1"]["delta"], last["macro_f1"]["ci_lo"], last["macro_f1"]["ci_hi"]
    f["v4_b_p"], f["v4_b_p_lo"], f["v4_b_p_hi"] = last["under40_pauc"]["delta"], last["under40_pauc"]["ci_lo"], last["under40_pauc"]["ci_hi"]
    f["v4_outcome"] = last["outcome"]
    margin = [r for r in _csv("results/v4/s54/s54_marginals.csv") if r["checkpoint"] == "last"]
    f["v4_v1_f1"] = float(next(r for r in margin if r["model"] == "v1_deployed")["macro_f1"])
    ctrl = [float(r["macro_f1"]) for r in margin if r["model"].startswith("control_")]
    f["v4_ctrl_f1"], f["v4_ctrl_min"], f["v4_ctrl_max"] = sum(ctrl) / len(ctrl), min(ctrl), max(ctrl)
    gate_a = next(r for r in _csv("results/v4/s54/s54_contrasts.csv") if r["contrast"].startswith("A:") and r["endpoint"] == "macro_f1" and r["seed"] == "mean" and r["checkpoint"] == "last")
    f["v4_a_f1"], f["v4_a_lo"], f["v4_a_hi"] = float(gate_a["delta"]), float(gate_a["ci_lo"]), float(gate_a["ci_hi"])
    probe = {(r["contrast"]): r for r in _csv("results/v4/backbone_probe_deltas.csv") if r["pooling"] == "primary" and r["band"] == "<40"}
    pan = next(v for k, v in probe.items() if k.startswith("panderm_vitb16_minus_convnext"))
    dino = next(v for k, v in probe.items() if k.startswith("dinov2_vitb14_minus_convnext"))
    f["v4_pan"], f["v4_pan_lo"], f["v4_pan_hi"] = float(pan["delta_pauc"]), float(pan["ci_lo"]), float(pan["ci_hi"])
    f["v4_dino"] = float(dino["delta_pauc"])
    s59 = _json("results/v4/s59/s59_report.json")["contract"]
    terms = {t["term"]: t for t in s59["terms"]}
    f["v4_terms"], f["v4_terms_met"] = len(terms), sum(t["status"] != "NOT_MET" for t in terms.values())
    f["v4_s59"] = {k: terms[f"sensitivity[{k}]"]["value"] for k in ("<40", "40-59", "60+")}
    f["v4_floor"] = terms["sensitivity[<40]"]["target"]
    f["v4_referral"], f["v4_referral_lo"], f["v4_referral_hi"], f["v4_referral_cap"] = terms["referral_rate[ALL]"]["value"], terms["referral_rate[ALL]"]["ci_lo"], terms["referral_rate[ALL]"]["ci_hi"], terms["referral_rate[ALL]"]["target"]
    f["v4_ret_f1"], f["v4_ret_target"] = terms["retained_macro_f1[ALL]"]["value"], terms["retained_macro_f1[ALL]"]["target"]
    s56 = next(r for r in _csv("results/v4/s56/contrasts_reserved.csv") if r["budget"] == "0.2" and r["contrast"] == "band-global" and r["band"] == "<40" and r["stat"] == "system_sens")
    f["v4_s56"], f["v4_s56_lo"], f["v4_s56_hi"] = float(s56["delta"]), float(s56["ci_lo"]), float(s56["ci_hi"])
    front = [r for r in _csv("results/v4/s56/frontier_reserved.csv") if r["budget"] == "0.2" and r["band"] == "<40" and r["score"] == "msp"]
    f["v4_ref_global"] = float(next(r for r in front if r["arm"] == "global")["referral_rate"])
    f["v4_ref_band"] = float(next(r for r in front if r["arm"] == "band")["referral_rate"])
    ceil = _json("results/v4/s64/ceiling.json")
    f["v4_ceiling_raw"] = ceil

    # ------------------------------------------------------------------ V5
    conf = _json("results/v5/confirm_s42_s43_s44.json")
    gates = conf["gates"]
    f["v5_rows"], f["v5_lesions"], f["v5_u40_lesions"], f["v5_archive"] = conf["n_rows"], conf["n_lesions"], conf["n_u40_escalating_lesions"], conf["u40_escalating_lesions_by_archive"]
    f["v5_seeds"] = conf["seeds"]
    f["v5_nboot"] = conf["bootstrap"]["n_boot"] if isinstance(conf["bootstrap"], dict) and "n_boot" in conf["bootstrap"] else None
    f["v5_gates"] = gates
    f["v5_lock_sha"] = re.search(r"sha256 ([0-9a-f]{8})", conf["lock"])[1]
    q11 = _json("results/v5/q11_decomposition.json")
    f["v5_q11_raw"] = q11
    f["v5_primary_raw"] = conf["primary"]
    matrix = _json("results/v5/v5_oof_common_matrix.json")
    f["v5_oof_rows"], f["v5_oof_lesions"] = matrix["rows"], matrix["lesions"]
    f["v5_oof_f1"] = matrix["descriptive_macro_f1"]

    s84 = _json("results/v5/s84/s84_report.json")
    f["s84"] = s84
    receipt = _json("results/v5/s84_receipt.json")
    f["s84_receipt"] = {"n": receipt["n_executions"], "labels": receipt["label_read_count"], "status": receipt["status"]}
    s75 = _json("results/v5/s75_milk10k_eligibility.json")
    f["s75_eligible"], f["s75_u40"] = s75["eligible_lesions"], s75["under40_escalating_lesions"]
    links = _json("results/v5/s84_lesion_links.json")
    f["s84_links_raw"] = links
    return f


if __name__ == "__main__":
    data = load()
    for key in sorted(data):
        if not key.endswith("_raw") and key not in ("s84",):
            print(key, data[key])
