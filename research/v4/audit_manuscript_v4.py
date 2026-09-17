"""Audit `paper/v4/manuscript_v4.tex` against the files its numbers come from (Hard Rule 4).

Two checks:

    claims     every registered literal occurs in the manuscript and equals its source value at
               the printed precision (magnitude; signs are written out in the text)
    closure    every decimal literal in the manuscript body is either a registered claim or a
               declared design constant -- an unknown number fails the audit

Nothing is recomputed from predictions; values are read from the frozen reports.

    $py -m research.v4.audit_manuscript_v4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ml.paths import REPO_ROOT

TEX = REPO_ROOT / "paper" / "v4" / "manuscript_v4.tex"
V4 = REPO_ROOT / "results" / "v4"

#: design constants fixed in plans before any read (not results)
CONSTANTS = {"0.20", "0.05", "0.10", "0.03", "0.5", "0.80", "0.855", "0.250", "1.5"}


def js(rel: str) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / rel)


def row(frame: pd.DataFrame, col: str, **where: Any) -> float:
    m = pd.Series(True, index=frame.index)
    for k, v in where.items():
        m &= frame[k] == v
    sel = frame.loc[m, col]
    assert len(sel) == 1, (col, where, len(sel))
    return float(sel.iloc[0])


def claims() -> list[tuple[str, float, str]]:
    """(printed literal, source value, source) -- the literal's decimals set the precision."""
    out: list[tuple[str, float, str]] = []

    def add(literals: str, values: list[float] | float, source: str) -> None:
        values = values if isinstance(values, list) else [values]
        for lit, val in zip(literals.split(), values):
            out.append((lit, float(val), source))

    pa = js("results/v4/power_audit.json")
    hist = pa["historical_audit"][0]
    add("0.313 0.221 0.166 0.636 0.409",
        [pa["ham_under40_counts"]["test"]["half_width_at_p50_lesion_level"], hist["half_width_lesion"],
         hist["half_width"], pa["seed_floor"]["seed_a"], pa["seed_floor"]["seed_b"]], "power_audit.json")

    dd = js("results/v4/audit/dedupe_conventional_rule.json")["rules"]["either_within_6"]
    ds = js("results/v4/duplicate_stats.json")
    add("1.08 1005", [dd["enrichment_either_over_chance"], ds["enrichment_both_over_chance"]],
        "audit/dedupe_conventional_rule.json, duplicate_stats.json")

    ce = js("results/v4/s64/ceiling.json")
    b = ce["bands"]
    add("0.878 0.811 0.942", [b["<40"]["auc_d"], *b["<40"]["auc_d_ci"]], "s64/ceiling.json")
    add("0.948 0.929 0.963", [b["40-59"]["auc_d"], *b["40-59"]["auc_d_ci"]], "s64/ceiling.json")
    add("0.929 0.916 0.941", [b["60+"]["auc_d"], *b["60+"]["auc_d_ci"]], "s64/ceiling.json")
    add("0.259 0.168", [b["<40"]["referral_for_sens"]["0.80"]["constant"],
                        b["<40"]["referral_for_sens"]["0.80"]["oracle_bins"]], "s64/ceiling.json")
    e = ce["ensemble_under40"]
    add("0.0037 0.0421 0.0516 0.0072", [e["delta_raw_vs_best_single"], *e["delta_raw_vs_best_single_ci"],
                                        e["delta_cal_vs_raw"]], "s64/ceiling.json")

    cm = js("results/v4/audit/under40_case_mix.json")["bands"]
    add("0.889 0.873 0.953 0.940 0.933 0.909",
        [cm["<40"]["auc_all_escalating_vs_benign"], cm["<40"]["auc_melanoma_vs_benign"],
         cm["40-59"]["auc_all_escalating_vs_benign"], cm["40-59"]["auc_melanoma_vs_benign"],
         cm["60+"]["auc_all_escalating_vs_benign"], cm["60+"]["auc_melanoma_vs_benign"]],
        "audit/under40_case_mix.json")
    add("0.044 0.036", [cm["60+"]["auc_all_escalating_vs_benign"] - cm["<40"]["auc_all_escalating_vs_benign"],
                        cm["60+"]["auc_melanoma_vs_benign"] - cm["<40"]["auc_melanoma_vs_benign"]],
        "audit/under40_case_mix.json")

    bp = csv("results/v4/backbone_probe_deltas.csv")
    bp = bp[(bp.pooling == "primary") & (bp.band == "<40")]
    for contrast, lits in (("panderm_vitb16_minus_convnext_tiny", "0.036 0.083 0.013"),
                           ("dinov2_vitb14_minus_convnext_tiny", "0.073 0.118 0.029")):
        r = bp[bp.contrast == contrast].iloc[0]
        add(lits, [r.delta_pauc, r.ci_lo, r.ci_hi], "backbone_probe_deltas.csv")

    bm = csv("results/v4/backbone_probe_marginals.csv")
    bm = bm[(bm.pooling == "primary") & (bm.band == "<40")]
    add("0.07", float(((bm.rho_ci_hi - bm.rho_ci_lo) / 2).mean()), "backbone_probe_marginals.csv")

    sc = csv("results/v4/s54/s54_contrasts.csv")
    sc = sc[sc.seed == "mean"]
    for chk, contrast, ep, lits in (
            ("last", "B", "pauc", "0.009 0.020 0.037"), ("best", "B", "pauc", "0.011 0.037 0.016"),
            ("last", "B", "macro_f1", "0.012 0.032 0.010"), ("best", "B", "macro_f1", "0.032 0.004 0.056"),
            ("last", "A", "macro_f1", "0.180 0.130 0.225"), ("last", "A", "pauc", "0.004 0.052 0.062")):
        r = sc[(sc.checkpoint == chk) & sc.contrast.str.startswith(contrast) & (sc.endpoint == ep)].iloc[0]
        add(lits, [r.delta, r.ci_lo, r.ci_hi], "s54/s54_contrasts.csv")
    sm = csv("results/v4/s54/s54_marginals.csv")
    v4m = sm[~sm.model.str.startswith("v1")]
    add("0.720 0.750 0.7335", [v4m.pauc.min(), v4m.pauc.max(),
                               row(sm, "pauc", model="v1_deployed", checkpoint="last")], "s54/s54_marginals.csv")

    s58 = csv("results/v4/s58/stage3_contrasts.csv")
    for contrast, lits in (("H1-H0", "0.020 0.012 0.054"), ("H2-H0", "0.007 0.027 0.040")):
        r = s58[(s58.contrast == contrast) & (s58.endpoint == "pauc_<40")].iloc[0]
        add(lits, [r.delta, r.ci_lo, r.ci_hi], "s58/stage3_contrasts.csv")
    r = s58[(s58.contrast == "H2-H1") & (s58.endpoint == "macro_f1")].iloc[0]
    add("0.008 0.024 0.007", [r.delta, r.ci_lo, r.ci_hi], "s58/stage3_contrasts.csv")
    mg = csv("results/v4/s58/stage3_marginals.csv")
    mg = mg[mg.subset == "all"]
    h0, h1 = row(mg, "macro_f1", model="H0_ham_head"), row(mg, "macro_f1", model="H1_pooled_head")
    ctrl = mg[mg.model.str.startswith("REF_V4_pooled_control")].macro_f1.mean()
    add("0.419 0.534 63 0.711", [h0, h1, 100 * (h1 - h0) / (ctrl - h0),
                                 row(mg, "pauc_u40", model="H0_ham_head")], "s58/stage3_marginals.csv")
    s58r = js("results/v4/s58/s58_report.json")
    add("0.378 0.932", [s58r["gate"]["pad_reject"], s58r["router"]["accuracy"]], "s58/s58_report.json")

    pr = js("results/v4/s67/probes.json")["primary"]
    for arm, lits in (("A_specialist", "0.039 0.108 0.034"), ("B_reweight", "0.000 0.013 0.012"),
                      ("C_metadata", "0.007 0.002 0.017")):
        add(lits, [pr[arm]["delta"], pr[arm]["ci_lo"], pr[arm]["ci_hi"]], "s67/probes.json")

    c56 = csv("results/v4/s56/contrasts_reserved.csv")
    c56 = c56[(c56.budget == 0.2) & (c56.contrast == "band-global")]
    get = lambda band, stat: c56[(c56.band == band) & (c56.stat == stat)].iloc[0]  # noqa: E731
    r = get("<40", "system_sens")
    add("0.154 0.104 0.216", [r.delta, r.ci_lo, r.ci_hi], "s56/contrasts_reserved.csv")
    r = get("60+", "system_sens")
    add("0.100 0.082 0.118", [r.delta, r.ci_hi, r.ci_lo], "s56/contrasts_reserved.csv")
    add("0.040 0.017", [get("ALL", "system_sens").delta, get("ALL", "referral_rate").delta],
        "s56/contrasts_reserved.csv")
    f56 = csv("results/v4/s56/frontier_reserved.csv")
    f56 = f56[(f56.role == "primary") & (f56.budget == 0.2)]
    add("0.232 0.454 0.386", [row(f56, "referral_rate", arm="global", band="<40"),
                              row(f56, "referral_rate", arm="band", band="<40"),
                              row(f56, "referral_rate", arm="band", band="ALL")], "s56/frontier_reserved.csv")
    ft = pd.Series([x["status"] for x in js("results/v4/s56/s56_report.json")["floor_transfer"]]).value_counts()
    add("15 12 3", [ft.sum(), ft["violated"], ft["below_point_within_ci"]], "s56/s56_report.json")

    lv = js("results/v4/lambda_verdict.json")["members"]["A3"]
    g2 = lv["gate2"]["reserved"]["<40"]
    add("0.208 0.127 0.296 1.92 0.960 0.857",
        [lv["gate1"]["delta"], *lv["gate1"]["ci"], lv["gate4"]["under40_referral_ratio"], g2["A1"], g2["candidate"]],
        "lambda_verdict.json")
    la = csv("results/v4/lambda_ablation.csv")
    add("0.145 0.278", [row(la, "referral_rate|<40", split="reserved", arm="A1"),
                        row(la, "referral_rate|<40", split="reserved", arm="A3")], "lambda_ablation.csv")

    g66 = js("results/v4/s66/s66_report.json")["gates"]
    add("0.216 0.125 0.313 1.90 0.160 0.303",
        [g66["1_primary"]["delta"], *g66["1_primary"]["ci"], g66["4_referral"]["ratio"],
         g66["4_referral"]["P0"], g66["4_referral"]["P2"]], "s66/s66_report.json")
    sp = g66["2_specificity"]["per_band"]["<40"]
    add("0.965 0.859", [sp["P0"], sp["P2"]], "s66/s66_report.json")
    ra = csv("results/v4/s66/reserved_arms.csv")
    ra = ra[ra.centre == "bcn20000"].set_index("arm")
    # extra referrals per extra catch on BCN under 40: extra flagged rows over extra caught images
    extra_catch = ra.loc["P0", "missed_serious|<40"] - ra.loc["P2", "missed_serious|<40"]
    extra_ref = g66["4_referral"]["delta"]["delta"] * _bcn_u40_rows()
    add("1.95", extra_ref / extra_catch, "s66/reserved_arms.csv + s66_report.json")

    c65 = csv("results/v4/s65/contrasts_reserved.csv")
    r = c65[c65.label == "primary"].iloc[0]
    add("0.082 0.042 0.131", [r.delta, r.ci_lo, r.ci_hi], "s65/contrasts_reserved.csv")
    r = c65[(c65.label == "workload_matched") & (c65.band == "<40")].iloc[0]
    add("0.004 0.014 0.025", [r.delta, r.ci_lo, r.ci_hi], "s65/contrasts_reserved.csv")
    f65 = csv("results/v4/s65/frontier_reserved.csv")
    add("0.608 0.558 0.409 0.262",
        [row(f65, "flag_rate", arm="COMB", band="<40", budget=0.2),
         row(f65, "flag_rate", arm="S56_matched", band="<40"),
         row(f65, "system_sens", arm="LAM", band="<40"),
         row(f65, "system_sens", arm="COMB_none", band="<40")], "s65/frontier_reserved.csv")

    gs = js("research/multical/results_oof/gap_summary.json")
    bc = csv("research/multical/results_oof/band_calibration_multical.csv")
    add("0.068 0.008 0.015 0.021",
        [gs["dirichlet_global"]["signed_gap_spread"], gs["dirichlet_per_band"]["signed_gap_spread"],
         gs["dirichlet_global"]["ece_gap"], gs["dirichlet_per_band"]["ece_gap"]], "multical/gap_summary.json")
    add("0.0276 0.0095 0.0301", [row(bc, "ece", source="dirichlet_global", group="<40"),
                                 row(bc, "ece", source="dirichlet_per_band", group="<40"),
                                 row(bc, "ece", source="dirichlet_per_band", group="60+")],
        "multical/band_calibration_multical.csv")

    cr = js("results/v4/corpus_report.json")
    sp, dup = cr["splits"], js("results/v4/audit/dedupe_conventional_rule.json")["rules"]["either_within_6"]
    add("25,331 13,909 13,931 15,294 2,270 4,733 1,992 279 104 47 151 10,498",
        [cr["manifest"]["manifest_rows"], cr["grouping"]["n_groups"], cr["grouping"]["n_effective_lesions"],
         sp["images_per_split"]["train"], sp["images_per_split"]["val"], sp["images_per_split"]["reserved"],
         sp["groups_per_split"]["reserved"], sp["under40_escalating_by_split"]["reserved"]["images"],
         sp["under40_escalating_by_split"]["reserved"]["lesions"],
         sp["allocation_per_stratum"]["<40|True"]["train"], sp["allocation_per_stratum"]["<40|True"]["groups"],
         dup["largest_component"]], "corpus_report.json, audit/dedupe_conventional_rule.json")
    mix = js("results/v4/audit/under40_case_mix.json")["bands"]["<40"]["escalating_mix"]
    add("51 64", [mix["mel"], sum(mix.values())], "audit/under40_case_mix.json")
    add("8", len(list(V4.glob("*/reserved*receipt.json"))), "results/v4/*/reserved*receipt.json")

    s53 = js("results/v4/s53r/s53r_report.json")
    r1 = s53["verdicts"]["R1"]["final_macro_f1"]
    add("0.023 0.0297 0.0097 0.0466", [s53["control"]["seed_sd"], r1["mean"], r1["min"], r1["max"]],
        "s53r/s53r_report.json")

    c59 = csv("results/v4/s59/contrasts_reserved.csv")
    g = lambda label, band, stat="system_sens": c59[(c59.label == label) & (c59.band == band) & (c59.stat == stat)].iloc[0]  # noqa: E731
    for label, band, lits in (("lambda_layer_workload_matched", "ALL", "0.018 0.025 0.011"),
                              ("conformal_layer_workload_matched", "ALL", "0.069 0.086 0.052"),
                              ("deployed_vs_V1", "ALL", "0.301 0.273 0.328"),
                              ("deployed_vs_V1", "<40", "0.423 0.310 0.539")):
        r = g(label, band)
        add(lits, [r.delta, r.ci_lo, r.ci_hi], "s59/contrasts_reserved.csv")
    terms = {t["term"]: t for t in js("results/v4/s59/s59_report.json")["contract"]["terms"]}
    for term, lits in (("sensitivity[<40]", "0.763 0.655 0.853"), ("sensitivity[40-59]", "0.704 0.631 0.771"),
                       ("sensitivity[60+]", "0.674 0.638 0.713"), ("retained_macro_f1[ALL]", "0.465 0.401 0.516"),
                       ("referral_rate[ALL]", "0.386 0.363 0.408")):
        t = terms[term]
        add(lits, [t["value"], t["ci_lo"], t["ci_hi"]], "s59/s59_report.json")
    add("0.888 13.6 7.9", [terms["retained_macro_f1[ALL]"]["target"], terms["nnb_pi0.03[ALL]"]["value"],
                           terms["nnb_pi0.03[ALL]"]["target"]], "s59/s59_report.json")
    return out


def _bcn_u40_rows() -> int:
    from research.v4 import s54_gate

    panel = s54_gate.build_panel("reserved")
    return int(((panel.archive == "bcn20000") & (panel.age_band == "<40")).sum())


def body_text() -> str:
    tex = TEX.read_text(encoding="utf-8")
    tex = tex.split(r"\begin{thebibliography}")[0]
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    return tex


def matches(literal: str, value: float) -> bool:
    if "." not in literal:
        return round(abs(value)) == int(literal.replace(",", ""))
    decimals = len(literal.split(".")[1])
    return f"{abs(value):.{decimals}f}" == literal


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    text = body_text()
    failed = 0
    registered = set()
    for literal, value, source in claims():
        registered.add(literal)
        present = re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d])", text) is not None
        ok = present and matches(literal, value)
        failed += not ok
        if not ok:
            print(f"  [FAIL] {literal} vs {value:.6g} ({source}){'' if present else ' -- not in manuscript'}")
    n_claims = len(claims())
    unknown = sorted({m for m in re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", text)}
                     - registered - CONSTANTS)
    for lit in unknown:
        print(f"  [FAIL] unregistered decimal in manuscript: {lit}")
    failed += len(unknown)
    print(f"audit_manuscript_v4: {n_claims} claims + closure, {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
