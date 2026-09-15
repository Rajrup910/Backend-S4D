"""Session 5, Part C: assert every headline number in the manuscript matches results/.

Hard rule 4 says nothing in the paper is hand-entered. LaTeX cannot enforce that, so this
script does: it re-reads results/ablation_table.csv, results/mcnemar_delong.json and
results/age_band_prior.csv, and checks both the numeric claims and the literal strings that
carry them in paper/manuscript.tex. Run it after any re-run of run_part_a.py -- if a metric
moves, this fails and names the sentence that needs editing.

Session 11 folded in the Session 10 verifiers, so the script now also covers everything the
manuscript revision added: the two new ladder rungs, the age-conditional rule, NNB, FRR and
bipartite conformal coverage, per-band calibration, intersectional cells, lesion-interior
attribution, the external PAD-UFES-20 evaluation, and the frozen-plan/receipt discipline. Some
of those are checked as whole reconstructed table rows rather than as loose substrings, which
is stricter and is what caught two rounding errors during S10.

A number of load-bearing *directional* claims are asserted outright rather than compared to a
literal -- that errors score higher on lesion-interior attribution, that under-40 conformal
coverage improves marginal < class-conditional < bipartite, that the post-calibration residuals
disagree in sign across age bands. If any of those flips, a paragraph is wrong, not a digit.

Exit code 1 on any mismatch, so it can gate a build. Structural validation of the .tex itself
(citations, labels, refs, environments) is a separate script, research/ablation/validate_structure.py.

Usage:
    python -m research.ablation.audit_manuscript
"""

from __future__ import annotations

import argparse
import csv
import json
import re

from ml.paths import resolve

DEFAULT_TARGET = "paper/manuscript.tex"

# Parsed here, at module import time, because SRC and every check below is built from it. The
# rest of the file stays a flat sequence of module-level assertions -- this is the minimal
# change that lets `python -m research.ablation.audit_manuscript --target ...` point the whole
# script at a different document without restructuring it into functions.
_arg_parser = argparse.ArgumentParser(add_help=True)
_arg_parser.add_argument("--target", default=DEFAULT_TARGET,
                         help=f"manuscript .tex to audit (default: {DEFAULT_TARGET})")
TARGET = _arg_parser.parse_args().target

#: True for any manuscript other than the published one -- e.g. paper/manuscript_edited.tex,
#: the downsized paper that deliberately drops the PAD-UFES-20 track, the CLAIM/TRIPOD
#: checklist appendices, the Grad-CAM figure and the case atlas. Checks tied to that dropped
#: content are recorded with `skip()` rather than run, so their absence is never confused with
#: a silent failure.
EDITED = TARGET != DEFAULT_TARGET

SRC = resolve(TARGET).read_text(encoding="utf-8")

ladder = {}
with open(resolve("results/ablation_table.csv"), encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        ladder[row["rung"]] = row

comparisons = json.load(open(resolve("results/mcnemar_delong.json"), encoding="utf-8"))
by_pair = {(c["a"], c["b"]): c for c in comparisons}

failures = []
skipped = []
checks = 0


def claim(label, value, expected, tol=5e-5):
    """Assert a number quoted in the manuscript matches the artifact."""
    global checks
    checks += 1
    if abs(float(value) - float(expected)) > tol:
        failures.append(f"{label}: manuscript {value} vs artifact {expected}")


def quoted(text):
    """Confirm a literal string appears in the manuscript."""
    global checks
    checks += 1
    if text not in SRC:
        failures.append(f"string not found in manuscript: {text!r}")


def skip(label, reason):
    """Record a check deliberately not run because `--target` dropped its content.

    Used only for content the edited manuscript is known to have removed on purpose (the
    PAD-UFES-20 track, the CLAIM/TRIPOD checklist appendices, the Grad-CAM figure). A check
    that just silently passed over missing content would be indistinguishable from one that
    silently failed to catch a regression -- this makes the omission a named, counted line in
    the report instead.
    """
    global checks
    checks += 1
    skipped.append(f"{label}: {reason}")


# --- Ladder Macro-F1 values quoted in prose -------------------------------------------
for rung, quoted_value in [
    ("A1_resnet50", 0.7058), ("A2_convnext_tiny", 0.7459), ("A3_swinv2_tiny", 0.7273),
    ("A4_gated_fusion", 0.7411), ("A5_soft_vote_6cnn", 0.7718),
    ("A6_soft_vote_6cnn_tta", 0.7859), ("A7_tta_dirichlet", 0.8047),
    ("B_margin_abstain10", 0.8577), ("B_margin_abstain20", 0.8957),
]:
    claim(f"{rung} macro_f1", quoted_value, round(float(ladder[rung]["macro_f1"]), 4))
    quoted(f"${quoted_value}$")

# --- Confidence intervals quoted in prose ---------------------------------------------
for rung, lo, hi in [
    ("A1_resnet50", 0.6470, 0.7501),
    ("A2_convnext_tiny", 0.6928, 0.7806),
    ("A5_soft_vote_6cnn", 0.7201, 0.8118),
]:
    claim(f"{rung} ci_low", lo, round(float(ladder[rung]["macro_f1_ci_low"]), 4))
    claim(f"{rung} ci_high", hi, round(float(ladder[rung]["macro_f1_ci_high"]), 4))
    quoted(f"[{lo:.4f}, {hi:.4f}]")

# --- Escalation sensitivity / missed serious for the A6 -> A7 regression ---------------
claim("A6 esc sens", 0.7862, round(float(ladder["A6_soft_vote_6cnn_tta"]["escalation_sensitivity"]), 4))
claim("A7 esc sens", 0.7310, round(float(ladder["A7_tta_dirichlet"]["escalation_sensitivity"]), 4))
claim("A6 missed", 62, int(ladder["A6_soft_vote_6cnn_tta"]["missed_serious"]))
claim("A7 missed", 78, int(ladder["A7_tta_dirichlet"]["missed_serious"]))
claim("B10 missed", 54, int(ladder["B_margin_abstain10"]["missed_serious"]))
claim("B20 missed", 36, int(ladder["B_margin_abstain20"]["missed_serious"]))

# --- Coverage values -------------------------------------------------------------------
for rung, cov in [("B_margin_abstain05", 0.954), ("B_margin_abstain10", 0.892),
                  ("B_margin_abstain15", 0.836), ("B_margin_abstain20", 0.779)]:
    claim(f"{rung} coverage", cov, round(float(ladder[rung]["coverage"]), 3), tol=6e-4)
quoted("$0.954/0.892/0.836/0.779$")

# deferral percentages quoted in abstract/results
claim("abstain10 deferral pct", 10.8, round((1 - float(ladder["B_margin_abstain10"]["coverage"])) * 100, 1))
claim("abstain20 deferral pct", 22.1, round((1 - float(ladder["B_margin_abstain20"]["coverage"])) * 100, 1))
quoted("$10.8\\%$")
quoted("$22.1\\%$")

# --- Paired statistics -----------------------------------------------------------------
ens = by_pair[("A5_soft_vote_6cnn", "A2_convnext_tiny")]
claim("ensembling McNemar chi2", 17.20, round(ens["mcnemar"]["statistic"], 2), tol=6e-3)
claim("ensembling only_a_correct", 85, ens["mcnemar"]["only_a_correct"])
claim("ensembling only_b_correct", 38, ens["mcnemar"]["only_b_correct"])
claim("ensembling diff", 0.026, round(ens["macro_f1_diff"]["point_estimate"], 3), tol=6e-4)
quoted("3.4\\times10^{-5}")

bkl = [d for d in ens["delong_per_class"] if d["class_code"] == "bkl"][0]
claim("bkl AUC ensemble", 0.9572, round(bkl["auc_a"], 4))
claim("bkl AUC convnext", 0.9195, round(bkl["auc_b"], 4))
claim("bkl z", 3.61, round(bkl["z"], 2), tol=6e-3)
claim("bkl Holm p", 0.013, round(bkl["p_value_holm"], 3), tol=6e-4)
assert bkl["significant_holm"], "bkl should be the surviving Holm-significant test"

# exactly one Holm-significant test overall, as the manuscript claims
n_sig = sum(1 for c in comparisons for d in c["delong_per_class"] if d["significant_holm"])
claim("Holm-significant count", 1, n_sig)
n_tests = sum(len(c["delong_per_class"]) for c in comparisons)
claim("DeLong family size", 42, n_tests)
quoted("$42$ DeLong tests")

for pair, p in [(("A2_convnext_tiny", "A1_resnet50"), 0.42),
                (("A3_swinv2_tiny", "A2_convnext_tiny"), 0.38),
                (("A4_gated_fusion", "A2_convnext_tiny"), 0.17),
                (("A6_soft_vote_6cnn_tta", "A5_soft_vote_6cnn"), 0.10),
                (("A7_tta_dirichlet", "A6_soft_vote_6cnn_tta"), 0.66)]:
    claim(f"McNemar p {pair[0]}", p, round(by_pair[pair]["mcnemar"]["p_value"], 2), tol=6e-3)

# --- Calibration direction ------------------------------------------------------------
# The manuscript claims the uncalibrated ensemble is UNDER-confident, contradicting the usual
# single-network result. That claim is load-bearing (it justifies Dirichlet over temperature
# scaling) and it was wrong in an earlier draft, so it is asserted here rather than trusted.
from research.ablation.loader import load_predictions  # noqa: E402
from research.ensembling.data import load_split_matrix  # noqa: E402

_matrix = load_split_matrix("test", predictions_dir="research/predictions")
_ens = _matrix.probs.mean(axis=1)
_conf = _ens.max(axis=1).mean()
_acc = (_ens.argmax(axis=1) == _matrix.y_true).mean()
claim("ensemble mean confidence", 0.7048, round(float(_conf), 4))
claim("ensemble accuracy", 0.8609, round(float(_acc), 4))
claim("ensemble signed gap", -0.156, round(float(_conf - _acc), 3), tol=6e-4)
assert _conf < _acc, "ensemble must be UNDER-confident for the Sec. IV-B argument to hold"
quoted("$0.7048$")
quoted("$0.8609$")
quoted("$-0.156$")

_single = load_predictions("research/predictions/convnext_tiny_test.csv", "convnext_tiny")
_sconf = _single.probs.max(axis=1).mean()
_sacc = (_single.y_pred == _single.y_true).mean()
claim("convnext confidence", 0.7606, round(float(_sconf), 4))
claim("convnext accuracy", 0.8296, round(float(_sacc), 4))
assert _sconf < _sacc, "single backbone is also under-confident, as the manuscript states"

# --- Figure 5 plots APS, so its caption must quote the APS worst class, not LAC's ---------
quoted("worst on \\textsc{mel} at $0.796$")

# --- Age-band prior --------------------------------------------------------------------
with open(resolve("results/age_band_prior.csv"), encoding="utf-8") as fh:
    prior = {(r["split"], r["age_band"]): r for r in csv.DictReader(fh)}
claim("train <40 escalating share", 0.049, round(float(prior[("train", "<40")]["escalating_share"]), 3), tol=6e-4)
claim("train 60+ escalating share", 0.355, round(float(prior[("train", "60+")]["escalating_share"]), 3), tol=6e-4)
quoted("$4.9\\%$")
quoted("$35.5\\%$")

# --- Split sizes -----------------------------------------------------------------------
for text in ["6{,}981", "1{,}532", "1{,}502", "7{,}470", "10{,}015", "5{,}229",
             "1{,}120", "1{,}121", "1{,}004", "1{,}340", "1{,}170"]:
    quoted(text)

# =======================================================================================
#  Session 10 additions: every number the manuscript revision introduced.
#
#  Folded in from the S10 staging scripts (research/ablation/verify_s10_{numbers,tables}.py,
#  now deleted). Two classes of check:
#
#    present()  a literal, formatted from a results/ artifact, appears somewhere in the
#               manuscript or one of its \input tables.
#    row()      a whole table row reconstructs cell-for-cell from the artifact. This is the
#               stricter one and it is what caught two FRR rounding errors in S10 that
#               substring presence alone did not.
#
#  Nothing here reads the test split; every source is a frozen file under results/.
# =======================================================================================

import pandas as pd  # noqa: E402


def _input_paths(tex: str) -> list[str]:
    """Every `\\input{...}` path in `tex`, relative to paper/, resolved recursively.

    Was a hardcoded 5-name tuple naming the tables `manuscript.tex` inputs. That list is
    specific to one document; `--target` needs the equivalent set for whatever manuscript is
    passed (e.g. `paper/tables_edited/*.tex` for the edited paper), so this discovers it the
    same way `build_overleaf_bundle.dependencies()` does, one level further: it also follows
    `\\input`s nested inside the tables themselves (`appendix_checklists.tex` pulls in
    `appendix_table_tripod_ai.tex` this way).
    """
    seen: list[str] = []
    queue = list(re.findall(r"\\input\{([^}]+)\}", tex))
    while queue:
        rel = queue.pop(0)
        rel = rel if rel.endswith(".tex") else rel + ".tex"
        if rel in seen:
            continue
        seen.append(rel)
        target = resolve(f"paper/{rel}")
        if target.is_file():
            queue += re.findall(r"\\input\{([^}]+)\}", target.read_text(encoding="utf-8"))
    return seen


_TABLES = "".join(
    resolve(f"paper/{rel}").read_text(encoding="utf-8")
    for rel in _input_paths(SRC) if resolve(f"paper/{rel}").is_file()
)
FULL = SRC + _TABLES
NORM = " ".join(FULL.split())


def present(label, literal, source):
    """A literal derived from an artifact must appear in the manuscript or its tables."""
    global checks
    checks += 1
    if literal not in FULL:
        failures.append(f"{label}: {literal!r} absent (source: {source})")


def row(label, cells, source):
    """A whole table row must reconstruct from the artifact, ignoring column padding."""
    global checks
    checks += 1
    want = " ".join((" & ".join(cells) + r" \\").split())
    if want not in NORM:
        failures.append(f"{label}: row absent (source: {source}) -- want: {want}")


def _f(x, nd=3):
    return ("%%.%df" % nd) % float(x)


def _csv(path):
    return pd.read_csv(resolve(path))


def _json(path):
    return json.load(open(resolve(path), encoding="utf-8"))


S9 = "results/session9/"

# --- New ladder rungs A7-oof and A8 ----------------------------------------------------
_lad = _csv(S9 + "ladder.csv").set_index("rung")
for _rung in ("A7_oof", "A8"):
    present(f"{_rung} macro_f1", _f(_lad.loc[_rung, "macro_f1"], 4), S9 + "ladder.csv")
    present(f"{_rung} ci_low", _f(_lad.loc[_rung, "macro_f1_ci_low"], 4), S9 + "ladder.csv")
    present(f"{_rung} ci_high", _f(_lad.loc[_rung, "macro_f1_ci_high"], 4), S9 + "ladder.csv")
present("A8 missed serious", str(int(_lad.loc["A8", "missed_serious"])), S9 + "ladder.csv")
# A7-val inside the S9 pass must still reproduce the published rung, or the frozen matrices moved
claim("A7_val reproduction", float(_lad.loc["A7_val", "macro_f1"]),
      float(ladder["A7_tta_dirichlet"]["macro_f1"]))

_cmp = _json(S9 + "new_rung_comparisons.json")
for _c, _ph in zip(_cmp["comparisons"], _cmp["holm"]["p_holm"]):
    _n = _c["comparison"]
    present(f"delta {_n}", _f(abs(_c["point_estimate"]), 4), S9 + "new_rung_comparisons.json")
    present(f"delta {_n} ci_low", _f(abs(_c["ci_low"]), 4), S9 + "new_rung_comparisons.json")
    present(f"delta {_n} ci_high", _f(abs(_c["ci_high"]), 4), S9 + "new_rung_comparisons.json")
    present(f"delta {_n} holm", _f(_ph, 3), S9 + "new_rung_comparisons.json")
    # both rungs are negative; the Discussion says so in words
    assert _c["point_estimate"] < 0, f"{_n} is no longer negative -- rewrite Sec. IV-B"

# --- Age gap and the escalation-mass AUC -----------------------------------------------
_gap = _csv(S9 + "age_gap_test.csv").set_index("band")
for _b in ("<40", "40-59", "60+"):
    present(f"agegap sens {_b}", _f(_gap.loc[_b, "sensitivity"], 3), S9 + "age_gap_test.csv")
    present(f"agegap AUC {_b}", _f(_gap.loc[_b, "escalation_mass_auc"], 3),
            S9 + "age_gap_test.csv")
# The softened mechanism claim is load-bearing: on test the <40 band must rank WORST, else
# Sec. IV-F's "part decision rule, part lost information" reading needs rewriting again.
assert (_gap.loc["<40", "escalation_mass_auc"]
        < min(_gap.loc["40-59", "escalation_mass_auc"],
              _gap.loc["60+", "escalation_mass_auc"])), \
    "under-40 no longer has the worst within-band AUC on test -- revisit Sec. IV-F"

_conf = _json(S9 + "age_gap_confirmatory_test.json")
for _k in ("difference", "ci_lo", "ci_hi"):
    present(f"confirmatory {_k}", _f(abs(_conf[_k]), 3), S9 + "age_gap_confirmatory_test.json")
assert _conf["holm"]["significant_holm"][0], "confirmatory age comparison must stay significant"

# --- The age-conditional rule -----------------------------------------------------------
_ar = _csv(S9 + "agerule_test.csv")
_ars = _json(S9 + "agerule_summary_test.json")
for _b in ("<40", "40-59", "60+", "ALL"):
    for _r in ("argmax", "lambda_rule"):
        _row = _ar[(_ar["band"] == _b) & (_ar["rule"] == _r)].iloc[0]
        present(f"rule sens {_b} {_r}", _f(_row["escalation_sensitivity"], 3),
                S9 + "agerule_test.csv")
        present(f"rule referral {_b} {_r}", _f(_row["referral_rate"], 3),
                S9 + "agerule_test.csv")
for _b in ("<40", "40-59", "60+"):
    present(f"lambda {_b}", _f(_ars["lambda_by_band"][_b], 2), S9 + "agerule_summary_test.json")
present("rule macro_f1", _f(_ars["macro_f1_lambda_rule"], 4), S9 + "agerule_summary_test.json")
present("argmax macro_f1", _f(_ars["macro_f1_argmax"], 4), S9 + "agerule_summary_test.json")
present("rule macro_f1 delta", _f(abs(_ars["macro_f1_delta"]), 3),
        S9 + "agerule_summary_test.json")
present("rule missed", str(_ars["missed_serious_lambda_rule"]), S9 + "agerule_summary_test.json")

# --- Number Needed to Biopsy at the stated reference prevalence -------------------------
_nnb = _csv(S9 + "nnb_test.csv")
for _c in ("<40", "40-59", "60+", "ALL"):
    for _r in ("argmax", "lambda_rule"):
        _row = _nnb[(_nnb["cohort"] == _c) & (_nnb["rule"] == _r)].iloc[0]
        present(f"NNB {_c} {_r}", _f(_row["nnb_reference"], 1), S9 + "nnb_test.csv")
_all = _nnb[_nnb["cohort"] == "ALL"]
for _col in ("nnb_at_0.01", "nnb_at_0.05"):
    for _r in ("argmax", "lambda_rule"):
        present(f"{_col} {_r}", _f(_all[_all["rule"] == _r][_col].iloc[0], 1),
                S9 + "nnb_test.csv")

# tab:agerule reconstructed row for row -- sensitivity, referral and NNB share a row, so a
# per-value check would not catch a cell landing in the wrong band.
# The edited manuscript's "merged age blind spot + rule recovery" table
# (paper/tables_edited/table_agegap_edited.tex) carries sensitivity by band and split in a
# different layout (val/OOF/test columns, not per-rule rows) and does not carry NNB or
# referral rate at all -- those are reported in prose (Sec. IV-D) instead of reproducing
# tab:agerule's row shape as a fifth table.
_TEXBAND = {"<40": "$<40$", "40-59": "$40$--$59$", "60+": "$60+$", "ALL": "All"}
if EDITED:
    for _b in ("<40", "40-59", "60+", "ALL"):
        for _r in ("argmax", "lambda_rule"):
            skip(f"tab:agerule row {_b} {_r}",
                 "the edited manuscript reports sensitivity via table_agegap_edited.tex and "
                 "NNB/referral in prose, not as a reproduced tab:agerule row (outside its "
                 "four-table budget)")
else:
    for _b in ("<40", "40-59", "60+", "ALL"):
        for _r, _texrule in (("argmax", r"$\arg\max$"), ("lambda_rule", r"$+\lambda$")):
            _a = _ar[(_ar["band"] == _b) & (_ar["rule"] == _r)].iloc[0]
            _nb = _nnb[(_nnb["cohort"] == _b) & (_nnb["rule"] == _r)].iloc[0]
            _cell = "%s [%s, %s]" % (_f(_a["escalation_sensitivity"], 3),
                                     _f(_a["sens_ci_lo"], 3), _f(_a["sens_ci_hi"], 3))
            if (_b, _r) == ("ALL", "lambda_rule"):
                _cell = r"\textbf{%s}" % _cell
            row(f"agerule {_b} {_r}",
                [_texrule, _cell, _f(_a["referral_rate"], 3), _f(_nb["nnb_reference"], 1)],
                S9 + "agerule_test.csv + nnb_test.csv")


# --- Orthogonality of the rule and the abstention gate ----------------------------------
_ort = _csv(S9 + "orthogonality_test.csv").set_index("band")
for _b in ("<40", "40-59", "60+", "ALL"):
    _r = _ort.loc[_b]
    present(f"ortho missed {_b}", str(int(_r["n_missed_by_argmax"])),
            S9 + "orthogonality_test.csv")
    present(f"ortho deferred {_b}", str(int(_r["n_missed_referred"])),
            S9 + "orthogonality_test.csv")
    present(f"ortho lambda {_b}", str(int(_r["n_missed_caught_by_lambda"])),
            S9 + "orthogonality_test.csv")
    present(f"ortho jaccard {_b}", _f(_r["jaccard_overlap"], 2),
            S9 + "orthogonality_test.csv")
    present(f"rescue rate {_b}", "%.1f" % (100 * float(_r["miss_rescue_rate"])),
            S9 + "orthogonality_test.csv")
for _b in ("<40", "60+"):
    present(f"band abstention {_b}", "%.1f" % (100 * float(_ort.loc[_b, "band_abstention_rate"])),
            S9 + "orthogonality_test.csv")
# The paper reports this as a NEGATIVE result. If the overlap ever stops being total, the
# Limitations paragraph and Sec. IV-G both become wrong in the friendly direction.
assert _ort.loc["<40", "jaccard_overlap"] == 1.0, \
    "under-40 orthogonality is no longer total -- Sec. IV-G and Limitations both overstate it"

# tab:ortho reconstructed row for row.
# The edited manuscript condenses this table into prose (the same four numbers per band,
# without a float) rather than reproducing tab:ortho -- it is not one of its four permitted
# tables. The individual present() checks above already require every one of those numbers to
# appear in the edited manuscript; only the exact tabular row structure is skipped here.
if EDITED:
    for _b in ("<40", "40-59", "60+", "ALL"):
        skip(f"tab:ortho row {_b}",
             "the edited manuscript reports the orthogonality numbers in prose, not as a "
             "reproduced float (outside its four-table budget)")
else:
    for _tex, _b, _bold in (("$<40$", "<40", True), ("$40$--$59$", "40-59", False),
                            ("$60+$", "60+", False), ("All", "ALL", False)):
        _r = _ort.loc[_b]
        _j = _f(_r["jaccard_overlap"], 2)
        row(f"ortho row {_b}",
            [_tex, str(int(_r["n_missed_by_argmax"])), str(int(_r["n_missed_referred"])),
             str(int(_r["n_missed_caught_by_lambda"])),
             (r"\textbf{%s}" % _j) if _bold else _j],
            S9 + "orthogonality_test.csv")


# --- Conformal: coverage, the under-40 column, FRR --------------------------------------
_cf = _csv(S9 + "conformal_test.csv")
_CAL = {"marginal": "marginal", "class_conditional": "class-cond.", "bipartite": "bipartite"}
_BOLD_U40 = {("LAC", "bipartite", 0.10), ("LAC", "bipartite", 0.05)}
_BOLD_FRR = {("RAPS", "bipartite", 0.05)}
_CONFORMAL_ROWS = [("LAC", "marginal", 0.10), ("LAC", "class_conditional", 0.10),
                   ("LAC", "bipartite", 0.10), ("LAC", "bipartite", 0.05),
                   ("RAPS", "marginal", 0.10), ("RAPS", "class_conditional", 0.10),
                   ("RAPS", "bipartite", 0.10), ("RAPS", "marginal", 0.05),
                   ("RAPS", "bipartite", 0.05)]
# The edited manuscript's conformal table (paper/tables_edited/conformal_coverage_edited.tex)
# is a condensed 3-calibrator x 2-alpha grid without the <40-restricted column tab:bipartite
# carries -- that breakdown is reported in prose instead (Sec. IV-B), which the present()-style
# checks elsewhere in this block already require. Only the exact 7-cell tab:bipartite row
# reconstruction is skipped for the edited target.
if EDITED:
    for _m, _c, _a in _CONFORMAL_ROWS:
        skip(f"conformal {_m}/{_c}/a{_a:.2f} row",
             "the edited manuscript's conformal table omits the <40-restricted column and "
             "reports it in prose instead (outside its four-table budget)")
else:
    for _m, _c, _a in _CONFORMAL_ROWS:
        _r = _cf[(_cf["method"] == _m) & (_cf["calibrator"] == _c)
                 & (_cf["alpha"].round(3) == _a)].iloc[0]
        _u40 = _f(_r["under40_escalating_coverage"], 3)
        if (_m, _c, _a) in _BOLD_U40:
            _u40 = r"\textbf{%s}" % _u40
        _frr = "%s [%s, %s]" % (_f(_r["frr"], 3), _f(_r["frr_ci_lo"], 3), _f(_r["frr_ci_hi"], 3))
        if (_m, _c, _a) in _BOLD_FRR:
            _frr = r"\textbf{%s}" % _frr
        row(f"conformal {_m}/{_c}/a{_a:.2f}",
            ["%.2f" % _a, _CAL[_c], _f(_r["marginal_coverage"], 3),
             _f(_r["escalating_coverage"], 3), _u40, _frr, _f(_r["mean_set_size"], 2)],
            S9 + "conformal_test.csv")

# The paper's sharpest new claim: class-conditional calibration does NOT repair the under-40
# subgroup and bipartite does. Asserted, not trusted.
def _u40cov(m, c, a):
    return float(_cf[(_cf["method"] == m) & (_cf["calibrator"] == c)
                     & (_cf["alpha"].round(3) == a)].iloc[0]["under40_escalating_coverage"])


for _m in ("LAC", "RAPS"):
    assert _u40cov(_m, "marginal", 0.10) < _u40cov(_m, "class_conditional", 0.10) \
        < _u40cov(_m, "bipartite", 0.10), \
        f"{_m}: marginal < class-conditional < bipartite no longer holds for under-40 coverage"

_bounds = _csv(S9 + "frr_bounds_test.csv")
_rb = _bounds[(_bounds["method"] == "RAPS") & (_bounds["calibrator"] == "bipartite")]
for _b in (0.05, 0.02):
    present(f"FRR bound {_b:.2f} set size", _f(_rb[_rb["frr_bound"] == _b]["mean_set_size"].iloc[0], 2),
            S9 + "frr_bounds_test.csv")
# "no alpha in the grid bounds FRR below 0.01" is a claim about the whole sweep, not one row
assert not _bounds[_bounds["frr_bound"] == 0.01]["achieved"].any(), \
    "some alpha now bounds FRR below 0.01 -- Sec. IV-E claims none does"
quoted("bounds FRR below")

# --- Per-band calibration ----------------------------------------------------------------
_bc = _csv(S9 + "band_calibration_test.csv")


def _bcell(src, grp, col, nd=3):
    _r = _bc[(_bc["source"] == src) & (_bc["group"] == grp)].iloc[0]
    _v = float(_r[col])
    if col == "signed_gap":
        return "$%s%s$" % ("+" if _v >= 0 else "-", _f(abs(_v), nd))
    return "$%s$" % _f(_v, nd)


# tab:bandcal reconstructed row for row. The edited manuscript is not required to carry the
# per-band calibration table (not one of its four permitted floats); the sign-disagreement
# finding it exists to support is reported in prose from the same source instead.
if EDITED:
    for _grp in ("<40", "40-59", "60+", "ALL"):
        skip(f"tab:bandcal row {_grp}",
             "the edited manuscript does not reproduce the per-band calibration table "
             "(outside its four-table budget)")
else:
    for _tex, _grp in (("$<40$", "<40"), ("$40$--$59$", "40-59"), ("$60+$", "60+"), ("All", "ALL")):
        _n = int(_bc[(_bc["source"] == "dirichlet") & (_bc["group"] == _grp)].iloc[0]["n"])
        row(f"bandcal {_grp}",
            [_tex, str(_n), _bcell("uncalibrated", _grp, "signed_gap"),
             _bcell("uncalibrated", _grp, "ece"), _bcell("dirichlet", _grp, "signed_gap"),
             _bcell("dirichlet", _grp, "ece")],
            S9 + "band_calibration_test.csv")

_gaps = _json(S9 + "band_calibration_gaps_test.json")
present("ece gap uncalibrated", _f(_gaps["test/uncalibrated"]["ece_gap"], 3),
        S9 + "band_calibration_gaps_test.json")
present("ece gap dirichlet", _f(_gaps["test/dirichlet"]["ece_gap"], 3),
        S9 + "band_calibration_gaps_test.json")

# The load-bearing calibration claim is the SIGN DISAGREEMENT after a single global map, not
# the ranking (which S9 showed does not replicate). Assert the sign disagreement directly.
_dir = _bc[(_bc["source"] == "dirichlet") & (_bc["group"].isin(["<40", "40-59", "60+"]))]
assert _dir["signed_gap"].min() < 0 < _dir["signed_gap"].max(), \
    "post-Dirichlet residuals no longer disagree in sign across bands -- Sec. IV-B overstates"

_oofbc = _csv("research/stats/results_oof/band_calibration.csv")
for _split, _src, _grp in (("oof", "uncalibrated", "<40"), ("oof", "uncalibrated", "40-59"),
                           ("oof", "uncalibrated", "60+"), ("oof", "dirichlet", "<40"),
                           ("oof", "dirichlet", "60+"), ("val", "dirichlet", "60+")):
    _r = _oofbc[(_oofbc["split"] == _split) & (_oofbc["source"] == _src)
                & (_oofbc["group"] == _grp)].iloc[0]
    present(f"{_split}/{_src}/{_grp} signed gap", _f(abs(_r["signed_gap"]), 3),
            "research/stats/results_oof/band_calibration.csv")
present("oof <40 accuracy",
        _f(_oofbc[(_oofbc["split"] == "oof") & (_oofbc["source"] == "uncalibrated")
                  & (_oofbc["group"] == "<40")].iloc[0]["accuracy"], 3),
        "research/stats/results_oof/band_calibration.csv")

# --- Per-class F1: the corrected Limitations attribution ---------------------------------
_pcf = _csv(S9 + "per_class_f1_test.csv").set_index("class_code")
for _c in ("akiec", "mel", "df", "vasc"):
    present(f"per-class F1 {_c}", _f(_pcf.loc[_c, "f1"], 3), S9 + "per_class_f1_test.csv")
for _c in ("df", "vasc"):
    present(f"per-class CI width {_c}", _f(_pcf.loc[_c, "ci_width"], 3),
            S9 + "per_class_f1_test.csv")
# Limitations now says the LEVEL drag is akiec/mel and the VARIANCE is df/vasc. Both halves
# of that correction are asserted, because the earlier draft had it backwards.
assert _pcf.loc[["akiec", "mel"], "f1"].max() < _pcf.loc[["df", "vasc"], "f1"].min(), \
    "akiec/mel are no longer the lowest per-class F1 -- Limitations attribution is stale"
assert _pcf.loc[["df", "vasc"], "ci_width"].min() > _pcf.loc[["akiec", "mel"], "ci_width"].max(), \
    "df/vasc no longer have the widest intervals -- Limitations attribution is stale"

# --- Intersectional cells ----------------------------------------------------------------
_int = _csv(S9 + "intersectional_test.csv").set_index("group")
for _cell in ("40-59 x female", "40-59 x male", "60+ x female", "60+ x male", "<40 x female"):
    _r = _int.loc[_cell]
    for _col in ("escalation_sensitivity", "sens_ci_lo", "sens_ci_hi"):
        present(f"intersectional {_cell} {_col}", _f(_r[_col], 3),
                S9 + "intersectional_test.csv")
_idis = _json(S9 + "intersectional_disparities_test.json")
for _k in ("equalized_odds_tpr_gap", "demographic_parity_gap", "referral_burden_gap"):
    present(f"intersectional {_k}", _f(_idis[_k], 3),
            S9 + "intersectional_disparities_test.json")
claim("intersectional usable cells", 5, int(_idis["n_usable_cells"]))
# The manuscript says the <40 male cell is suppressed and declines to quote its sensitivity.
# Sec. IV-H says this cell "fails the gate by one case" and declines to quote a sensitivity.
# A bare substring test for its value is useless (0.000 appears inside CIs), so check the
# substantive claim instead: it is flagged suppressed, and it is short by exactly one positive.
from research.selective.fairness import MIN_POSITIVES  # noqa: E402

_u40m = _int.loc["<40 x male"]
assert str(_u40m["suppressed"]).lower() == "true", "<40 x male is no longer suppressed"
claim("<40 x male escalating count", MIN_POSITIVES - 1, int(_u40m["n_escalating"]))
claim("fairness positives gate", 10, MIN_POSITIVES)
quoted("so it fails the gate by one case")

# --- Grad-CAM attribution against the Tschandl masks --------------------------------------
# The edited manuscript drops the Grad-CAM figure and its explainability subsection entirely,
# so none of these literals are expected to appear in it.
if EDITED:
    _att = _json(S9 + "attribution_summary_test.json")
    for _k in ("interior_fraction_mean", "interior_fraction_ci_lo", "interior_fraction_ci_hi",
               "lesion_area_fraction_mean", "concentration_ratio_mean"):
        skip(f"attribution {_k}", "Grad-CAM figure and explainability subsection dropped")
    for _g in ("correct", "incorrect", "true=nv"):
        skip(f"attribution {_g}", "Grad-CAM figure and explainability subsection dropped")
    skip("attribution missing masks", "Grad-CAM figure and explainability subsection dropped")
    skip("attribution errors-score-higher (directional)",
         "Grad-CAM figure and explainability subsection dropped")
else:
    _att = _json(S9 + "attribution_summary_test.json")
    for _k, _nd in (("interior_fraction_mean", 3), ("interior_fraction_ci_lo", 3),
                    ("interior_fraction_ci_hi", 3), ("lesion_area_fraction_mean", 3),
                    ("concentration_ratio_mean", 2)):
        present(f"attribution {_k}", _f(_att[_k], _nd), S9 + "attribution_summary_test.json")
    for _g in ("correct", "incorrect", "true=nv"):
        _b = [x for x in _att["breakdown"] if x["group"] == _g][0]
        present(f"attribution {_g}", _f(_b["mean"], 3), S9 + "attribution_summary_test.json")
    claim("attribution missing masks", 0, int(_att["n_missing_masks"]))
    # The paper explains the predicted-class artefact by pointing at errors scoring HIGHER.
    _c_mean = [x for x in _att["breakdown"] if x["group"] == "correct"][0]["mean"]
    _i_mean = [x for x in _att["breakdown"] if x["group"] == "incorrect"][0]["mean"]
    assert _i_mean > _c_mean, "errors no longer score higher -- Sec. IV-I's caveat needs rewriting"

# --- Selective classification at the OOF-fitted gate ---------------------------------------
_sel = _csv(S9 + "selective_test.csv").set_index("target_abstention")
_s10 = _sel.loc["10%"]
present("sel 10% coverage", "%.1f" % (100 * float(_s10["coverage"])), S9 + "selective_test.csv")
present("sel 10% macro_f1", _f(_s10["macro_f1"], 4), S9 + "selective_test.csv")
present("sel 10% missed", str(int(_s10["missed_serious"])), S9 + "selective_test.csv")

# --- External evaluation on PAD-UFES-20 ----------------------------------------------------
# The edited manuscript drops the entire PAD-UFES-20 smartphone track, so none of the S8b /
# Fitzpatrick / Mahalanobis-on-PAD literals below are expected in it.
if EDITED:
    for _m in ("pad_ensemble_softvote", "pad_member_convnext_small", "pad_member_efficientnet_b3"):
        skip(f"PAD macro_f1 {_m}", "PAD-UFES-20 smartphone track dropped")
    for _arm in ("dirichlet", "dirichlet_plus_age_rule"):
        for _q in ("macro_f1", "esc sens", "mel recall", "missed"):
            skip(f"PAD {_q} {_arm} (deployed map)", "PAD-UFES-20 smartphone track dropped")
    skip("PAD rule agreement / repoint sanity checks", "PAD-UFES-20 smartphone track dropped")
    skip("PAD ensemble-worse-than-best-member (directional)",
         "PAD-UFES-20 smartphone track dropped")
    for _g in ("I", "II", "III", "IV", "unknown"):
        skip(f"fitzpatrick sens/n {_g}", "PAD-UFES-20 smartphone track dropped")
    skip("fitzpatrick pooled tpr gap", "PAD-UFES-20 smartphone track dropped")
    skip("fitzpatrick I-IV spread", "PAD-UFES-20 smartphone track dropped")
    for _q in ("auroc", "id median", "shift median", "separation"):
        skip(f"mahalanobis {_q}", "PAD-UFES-20 smartphone track dropped (shift cohort was PAD)")
else:
    # The per-member Macro-F1 values still come from the S8b sweep, the only run that scored the
    # six checkpoints individually. Everything the calibrator touches now comes from
    # `pad_age_rule_deployed.json` instead: S8b applied the *calibration* OOF Dirichlet fit rather
    # than the deployed one, and S16 put the two in the same paper, at which point the third
    # decimal stopped being a curiosity (missed serious 1,264 against 1,268).
    _X = "research/xdomain/results/"
    _pad = _csv(_X + "ensemble_on_pad.csv").set_index("method")
    for _m in ("pad_ensemble_softvote", "pad_member_convnext_small", "pad_member_efficientnet_b3"):
        present(f"PAD macro_f1 {_m}", _f(_pad.loc[_m, "macro_f1"], 3), _X + "ensemble_on_pad.csv")

    _PADRULE = "results/external/pad_age_rule_deployed.json"
    _padr = _json(_PADRULE)
    assert _padr["rule_implementations_agree"], \
        "frozen_params and run_session8b implement different age rules on PAD"
    assert (_padr["superseded_s8b"]["dirichlet"]["missed_serious"]
            != _padr["deployed"]["dirichlet"]["missed_serious"]), \
        "the two Dirichlet maps no longer differ on PAD -- the S16 repoint is moot, simplify it"
    for _arm in ("dirichlet", "dirichlet_plus_age_rule"):
        _row = _padr["deployed"][_arm]
        present(f"PAD macro_f1 {_arm} (deployed map)", _f(_row["macro_f1"], 3), _PADRULE)
        present(f"PAD esc sens {_arm} (deployed map)", _f(_row["escalation_sens"], 3), _PADRULE)
        present(f"PAD mel recall {_arm} (deployed map)", _f(_row["mel_recall"], 3), _PADRULE)
        present(f"PAD missed {_arm} (deployed map)",
                "{:,}".format(int(_row["missed_serious"])).replace(",", "{,}"), _PADRULE)
    # the frozen rule must still trade referrals for sensitivity in the same direction as on HAM
    assert (_padr["deployed"]["dirichlet_plus_age_rule"]["escalation_sens"]
            > _padr["deployed"]["dirichlet"]["escalation_sens"]), \
        "the frozen rule no longer raises PAD escalation sensitivity -- Sec. IV-J is wrong"

    # "the ensemble is worse than its own best member" is the paper's cross-domain claim
    _members = [i for i in _pad.index if i.startswith("pad_member_")]
    assert _pad.loc["pad_ensemble_softvote", "macro_f1"] < _pad.loc[_members, "macro_f1"].max(), \
        "the PAD soft-vote no longer trails its best member -- Sec. IV-J overstates"

    _fitz = _csv(_X + "fitzpatrick_slice.csv")
    _fitz = _fitz[_fitz["group"] != "ALL"].set_index("group")
    for _g in ("I", "II", "III", "IV"):
        present(f"fitzpatrick sens {_g}", _f(_fitz.loc[_g, "escalation_sensitivity"], 3),
                _X + "fitzpatrick_slice.csv")
        present(f"fitzpatrick n {_g}", str(int(_fitz.loc[_g, "n"])), _X + "fitzpatrick_slice.csv")
    present("fitzpatrick unknown sens", _f(_fitz.loc["unknown", "escalation_sensitivity"], 3),
            _X + "fitzpatrick_slice.csv")
    present("fitzpatrick unknown n", str(int(_fitz.loc["unknown", "n"])), _X + "fitzpatrick_slice.csv")
    present("fitzpatrick pooled tpr gap", _f(_json(_X + "fitzpatrick_gaps.json")["equalized_odds_tpr_gap"], 3),
            _X + "fitzpatrick_gaps.json")
    _iiv = _fitz.loc[["I", "II", "III", "IV"], "escalation_sensitivity"]
    present("fitzpatrick I-IV spread", _f(_iiv.max() - _iiv.min(), 3),
            _X + "fitzpatrick_slice.csv (derived)")
    # The manuscript says the I-IV pattern is NON-monotonic; that is the whole point of the
    # paragraph, so it is asserted rather than described.
    assert not (_iiv.is_monotonic_increasing or _iiv.is_monotonic_decreasing), \
        "the Fitzpatrick I-IV spread is now monotonic -- Sec. IV-J's reading changes"

    _mah = _json(_X + "mahalanobis_shift.json")
    present("mahalanobis auroc", _f(_mah["auroc_id_vs_shift"], 3), _X + "mahalanobis_shift.json")
    present("mahalanobis id median", "%d" % round(_mah["median_score_id_val"]),
            _X + "mahalanobis_shift.json")
    present("mahalanobis shift median",
            "{:,}".format(round(_mah["median_score_shift_pad"])).replace(",", "{,}"),
            _X + "mahalanobis_shift.json")
    present("mahalanobis separation", "%d" % round(_mah["separation_ratio_median"]),
            _X + "mahalanobis_shift.json")

# --- OOF-vs-val diagnostics quoted in Methods and Sec. IV-E --------------------------------
_ovv = _csv("results/oof_vs_val_comparison.csv")


def _ovv_get(module, quantity, column):
    return _ovv[(_ovv["module"] == module)
                & (_ovv["quantity"] == quantity)].iloc[0][column]


present("conformal calibration rows (oof)",
        "{:,}".format(int(float(_ovv_get("conformal", "n_calibration_rows", "oof_fitted"))))
        .replace(",", "{,}"), "results/oof_vs_val_comparison.csv")
present("conformal min class count (oof)",
        str(int(float(_ovv_get("conformal", "min_class_calibration_count_a05", "oof_fitted")))),
        "results/oof_vs_val_comparison.csv")
claim("degenerate cells a05 (val)",
      6, int(float(_ovv_get("conformal", "degenerate_class_conditional_cells_a05", "val_fitted"))))
claim("degenerate cells a05 (oof)",
      0, int(float(_ovv_get("conformal", "degenerate_class_conditional_cells_a05", "oof_fitted"))))
# Methods states plainly that the OOF arm forfeits the exact guarantee.
assert str(_ovv_get("conformal", "exact_finite_sample_guarantee", "oof_fitted")) in ("False", "false"), \
    "the OOF conformal arm now claims an exact guarantee -- Methods says it does not"
# and that OOF fitting flips the selected uncertainty score
claim_val = str(_ovv_get("selective", "policy_score", "val_fitted"))
claim_oof = str(_ovv_get("selective", "policy_score", "oof_fitted"))
assert (claim_val, claim_oof) == ("margin", "msp"), \
    f"the val/OOF score selection is now ({claim_val}, {claim_oof}) -- Sec. IV-D says (margin, msp)"

# --- Fold yields quoted in Methods ---------------------------------------------------------
_folds = resolve("ml/results/oof_fold_report.md").read_text(encoding="utf-8")
for _tok, _what in (("71", "df OOF rows"), ("99", "vasc OOF rows"), ("64", "escalating <40")):
    checks += 1
    if _tok not in _folds:
        failures.append(f"fold report missing {_what} ({_tok})")
    present(f"Methods quotes {_what}", f"${_tok}$", "ml/results/oof_fold_report.md")

# --- Holm-corrected McNemar, quoted in the abstract and Discussion --------------------------
_fam = _json("results/comparison_families.json")
_holm = _fam["ladder_mcnemar_adjusted"]
_idx = _holm["members"].index("A5_soft_vote_6cnn vs A2_convnext_tiny")
claim("ensembling McNemar Holm p", 2.0e-4, round(_holm["p_holm"][_idx], 6), tol=6e-6)
quoted("2.0\\times10^{-4}")
claim("McNemar family size", 6, _holm["n_tests"])
claim("McNemar Holm survivors", 1, _holm["n_significant"])
claim("declared comparison families", 7, _fam["n_families"])
claim("confirmatory families", 4, _fam["n_confirmatory"])
claim("exploratory families", 3, _fam["n_exploratory"])
quoted("Seven families")

# --- The frozen analysis plan and the single test pass ---------------------------------------
_plan = _json("results/analysis_plan.json")
claim("pre-registered quantities", 19, int(_plan["n_quantities"]))
quoted("$19$-item")
claim("reference prevalence", 0.03, float(_plan["constants"]["reference_prevalence"]))
_receipt = _json("results/test_pass_receipt.json")
_emitted = sum(int(e["n_quantities"]) for e in _receipt["executions"])
claim("quantities emitted by the test pass", 19, _emitted)
for _e in _receipt["executions"]:
    assert _e["plan_sha256"] == _plan["self_sha256"], \
        "a test-pass execution carries a different plan hash than the frozen plan"
    assert not _e.get("rerun_reason"), \
        "the test split was re-read with a stated reason -- the manuscript claims one pass"

# --- CLAIM 2024 checklist, and the supplementary that renders it ---------------------------
# The edited manuscript drops the CLAIM/TRIPOD checklist appendices altogether.
if EDITED:
    for _q in ("items met", "items partial", "items not met", "items N/A", "total",
               "rows present"):
        skip(f"CLAIM {_q}", "CLAIM/TRIPOD checklist appendices dropped")
    skip("CLAIM summary sentence quoted", "CLAIM/TRIPOD checklist appendices dropped")
    skip("paper/supplementary.tex existence",
         "the edited manuscript's supplement, if any, is that session's own concern")
else:
    # The manuscript quotes the checklist outcome, so the checklist is the artifact and the
    # manuscript is the claim -- exactly the direction hard rule 4 wants.
    _claim = resolve("results/CLAIM_checklist.md").read_text(encoding="utf-8")
    _counts = dict(re.findall(r"^\| (Met|Partial|Not met|N/A) \| (\d+) \|$", _claim, flags=re.M))
    claim("CLAIM items met", 33, int(_counts["Met"]))
    claim("CLAIM items partial", 7, int(_counts["Partial"]))
    claim("CLAIM items not met", 2, int(_counts["Not met"]))
    claim("CLAIM items N/A", 2, int(_counts["N/A"]))
    claim("CLAIM total", 44, sum(int(v) for v in _counts.values()))
    claim("CLAIM rows present", 44, len(re.findall(r"^\| \d+ \|", _claim, flags=re.M)))
    quoted("$33$ items met, $7$ partial, $2$ not")
    # The supplement is generated from that Markdown; if it is missing the bundle ships without it.
    assert resolve("paper/supplementary.tex").is_file(), \
        "paper/supplementary.tex is missing -- run research.ablation.build_supplementary"


# --- S16: the three-centre external battery (E1) ----------------------------------------
# Every literal below is read from the frozen S13/S14 reports, and the directional asserts
# fire when a *paragraph* of Sec. IV-K becomes wrong rather than a digit -- the same style
# S11 introduced. Both pre-registered claims failed, and the write-up depends on them having
# failed in the specific ways recorded here.
_E1 = _json("results/external/age_rule_transfer_report.json")
_dose = {r["cohort"]: r for r in _E1["dose_response"]["rows"]}

present("E1 AUC spread", _f(_E1["dose_response"]["claim_a_auc_spread"], 3),
        "age_rule_transfer_report.json")
assert not _E1["verdict"]["claim_a_flat_across_centres"], \
    "Claim A now holds -- Sec. IV-K says it fails"
assert _E1["verdict"]["claim_b_ordering_reversed"], \
    "the Claim B ordering is no longer reversed -- Sec. IV-K and the abstract say it is"
assert not _E1["verdict"]["mix_control_changes_conclusion"], \
    "the melanoma-only mix control now changes the conclusion"
assert _E1["dose_response"]["claim_a_within_cohort"]["mskcc"]["worst_band"] != "<40", \
    "under-40 is worst within MSKCC again -- Sec. IV-K claims it is the best-ranked band"
assert _E1["dose_response"]["claim_a_within_cohort"]["ham_oof"]["under40_is_worst"], \
    "under-40 is no longer the worst band on HAM -- the internal result moved"

for _c in ("ham_oof", "bcn20000", "mskcc"):
    _row = _dose[_c]
    present(f"E1 {_c} under40 AUC", _f(_row["under40_escalation_mass_auc"]), "E1 report")
    present(f"E1 {_c} under40 argmax sens", _f(_row["under40_argmax_sensitivity"]), "E1 report")
    present(f"E1 {_c} melanoma-only sens",
            _f(_row["under40_melanoma_only_sensitivity"]), "E1 report")
    _lam = [t for t in _E1["transfer"]
            if t["cohort"] == _c and t["band"] == "<40" and t["rule"] == "frozen_lambda"][0]
    present(f"E1 {_c} lambda sens", _f(_lam["point"]), "E1 report")
    present(f"E1 {_c} lambda referral cost",
            "%+.3f" % _lam["delta_referral_rate"], "E1 report")
    # the frozen rule must still help, at a cost, in every centre
    assert _lam["delta_sensitivity"] > 0 and _lam["delta_referral_rate"] > 0, \
        f"the frozen lambda no longer trades referrals for sensitivity in {_c}"

# The descriptive lambda sweep is quoted in three places (Sec. IV-K, the Discussion and
# Limitations) as the argument that the rule's *magnitude* does not transport.
_sweep = _csv("results/external/e1_lambda_sweep_descriptive.csv")
_sweep = _sweep[_sweep["band"] == "<40"]
_opt = {c: g.loc[g["expected_cost"].idxmin(), "lambda"]
        for c, g in _sweep.groupby("cohort")}
for _c, _lam_opt in sorted(_opt.items()):
    present(f"E1 {_c} under40 lambda optimum", _f(_lam_opt, 2),
            "e1_lambda_sweep_descriptive.csv")
assert abs(_opt["ham_oof"] - 0.26) < abs(_opt["bcn20000"] - 0.26),     "the source-cohort optimum is no longer the closest to the frozen lambda"

_conf1 = _E1["confirmatory"][_E1["confirmatory"]["primary_cohort"]]
assert _conf1["only_argmax_caught"] == 0, \
    "argmax now catches a case the rule misses -- the one-sidedness argument in Sec. IV-K breaks"
present("E1 confirmatory rescues", str(int(_conf1["only_rule_caught"])), "E1 report")

# all-ages Macro-F1 of the two cohorts, quoted in the design-limitation paragraph
_ham_all = [r for r in _E1["claim_b"] if r["cohort"] == "ham_oof" and r["band"] == "ALL"][0]
_bcn_all = [r for r in _E1["claim_b"] if r["cohort"] == "bcn20000" and r["band"] == "ALL"][0]
present("E1 HAM all-ages macro_f1", _f(_ham_all["macro_f1"]), "E1 report")
present("E1 BCN all-ages macro_f1", _f(_bcn_all["macro_f1"]), "E1 report")
assert _bcn_all["macro_f1"] < _ham_all["macro_f1"], \
    "BCN no longer transfers worse than HAM -- the entanglement limitation is void"

# --- S16: decision curves (E6) ----------------------------------------------------------
_E6 = _json("results/external/decision_curve_report.json")
_all10 = [p for p in _E6["panels"]["HAM10000 OOF (N=6,981)"]["reference_points"]
          if abs(p["p_t"] - 0.10) < 1e-9][0]
_u4010 = [p for p in _E6["panels"][r"HAM10000 OOF, age $<$40"]["reference_points"]
          if abs(p["p_t"] - 0.10) < 1e-9][0]
present("E6 all-ages delta at 0.10", "+%.4f" % _all10["delta_nb"], "decision_curve_report.json")
present("E6 all-ages CI at 0.10",
        "[+%.4f, +%.4f]" % (_all10["ci_low"], _all10["ci_high"]), "decision_curve_report.json")
present("E6 under40 delta at 0.10", "+%.4f" % _u4010["delta_nb"], "decision_curve_report.json")
assert _all10["ci_low"] > 0, "the all-ages net-benefit gain no longer excludes zero"
assert _u4010["ci_low"] < 0 < _u4010["ci_high"], \
    "the under-40 net-benefit difference is no longer null -- Sec. IV-K overstates"
_u4020 = [p for p in _E6["panels"][r"HAM10000 OOF, age $<$40"]["reference_points"]
          if abs(p["p_t"] - 0.20) < 1e-9][0]
assert _u4020["delta_nb"] < 0, "the under-40 curve no longer turns negative by p_t = 0.20"
assert _E6["test_read"] is False, "the decision-curve report now declares a test read"

# --- S16: prior-shift decomposition on PAD (E2) -----------------------------------------
# E2 is a PAD-only workstream (prior-shift decoupling has no meaning outside PAD's inverted
# class prior); the edited manuscript drops it along with the rest of the PAD track.
if EDITED:
    for _q in ("oracle macro_f1", "EM macro_f1", "EM iterations", "EM df prior",
               "oracle bcc prior", "source nv prior", "McNemar chi2", "only-raw-correct",
               "only-EM-correct"):
        skip(f"E2 {_q}", "PAD-UFES-20 smartphone track dropped (E2 is PAD-only)")
    skip("E2 directional checks (oracle>raw, EM<raw, confirmatory direction)",
         "PAD-UFES-20 smartphone track dropped (E2 is PAD-only)")
else:
    _E2 = _json("results/external/pad_prior_decoupling_report.json")
    _variants = {v["variant"]: v for v in _E2["results"]}
    _raw = _variants["Raw Ensemble (Soft-Vote)"]
    _oracle = _variants["Oracle Prior Correction"]
    _em = _variants["Deployable EM Prior (Saerens et al.)"]
    present("E2 oracle macro_f1", _f(_oracle["macro_f1"]), "pad_prior_decoupling_report.json")
    present("E2 EM macro_f1", _f(_em["macro_f1"]), "pad_prior_decoupling_report.json")
    present("E2 EM iterations", str(int(_E2["em_iterations"])), "pad_prior_decoupling_report.json")
    present("E2 EM df prior", _f(_E2["em_estimated_prior_pi_t"]["df"]),
            "pad_prior_decoupling_report.json")
    present("E2 oracle bcc prior", _f(_E2["oracle_target_prior_pi_t"]["bcc"]),
            "pad_prior_decoupling_report.json")
    present("E2 source nv prior", _f(_E2["implicit_source_prior_pi_s"]["nv"]),
            "pad_prior_decoupling_report.json")
    assert _oracle["macro_f1"] > _raw["macro_f1"], \
        "the oracle prior no longer beats raw -- the prior-shift half of the decomposition is void"
    assert _em["macro_f1"] < _raw["macro_f1"], \
        "label-free EM no longer hurts -- Sec. IV-J reports it as a negative result"
    _e2c = _E2["confirmatory"]
    assert _e2c["only_raw_correct"] > _e2c["only_em_correct"], \
        "the E2 confirmatory member is no longer significant in the wrong direction"
    present("E2 McNemar chi2", "%.2f" % _e2c["mcnemar_statistic"], "pad_prior_decoupling_report.json")
    present("E2 only-raw-correct", str(int(_e2c["only_raw_correct"])),
            "pad_prior_decoupling_report.json")
    present("E2 only-EM-correct", str(int(_e2c["only_em_correct"])),
            "pad_prior_decoupling_report.json")

# --- S16: triage and the safety nets under shift (E3, E4, E5) ---------------------------
# E3's HAM row survives in the edited paper (the row-filtered triage panel keeps HAM/BCN/MSKCC);
# only its PAD row and the PAD-vs-HAM comparison are dropped. E4 (shift = PAD) and E5
# (Fitzpatrick, PAD-only) are PAD end to end and drop wholesale.
_E3 = {r["cohort"]: r for r in _json("results/external/clinical_triage_report.json")}
_ham_cal = _E3["HAM10000 OOF (Calibrated Ensemble, N=6,981)"]
present("E3 HAM NNB", "%.1f" % _ham_cal["nnb_pi_03"], "clinical_triage_report.json")
present("E3 HAM missed T1", str(int(_ham_cal["missed_tier1_as_tier3"])),
        "clinical_triage_report.json")
if EDITED:
    skip("E3 PAD NNB", "PAD-UFES-20 smartphone track dropped")
    skip("E3 PAD point-FRR", "PAD-UFES-20 smartphone track dropped")
    skip("E3 PAD-costs-more-than-HAM (directional)", "PAD-UFES-20 smartphone track dropped")
    for _label in ("ID", "shift"):
        for _q in ("set size", "singleton", "serious coverage"):
            skip(f"E4 APS {_label} {_q}", "PAD-UFES-20 smartphone track dropped (E4 shift cohort is PAD)")
    skip("E4 widening/degradation-under-shift (directional)",
         "PAD-UFES-20 smartphone track dropped (E4 shift cohort is PAD)")
    skip("E5 tier-1 sens by Fitzpatrick group", "PAD-UFES-20 smartphone track dropped (E5 is PAD-only)")
    skip("E5 I-IV spread", "PAD-UFES-20 smartphone track dropped (E5 is PAD-only)")
    skip("E5 non-monotonicity / unlabelled-stratum checks",
         "PAD-UFES-20 smartphone track dropped (E5 is PAD-only)")
else:
    _pad_cal = _E3["PAD-UFES-20 (Calibrated Ensemble)"]
    present("E3 PAD NNB", "%.1f" % _pad_cal["nnb_pi_03"], "clinical_triage_report.json")
    present("E3 PAD point-FRR", _f(_pad_cal["point_frr"]), "clinical_triage_report.json")
    assert _pad_cal["nnb_pi_03"] > _ham_cal["nnb_pi_03"], \
        "PAD no longer costs more biopsies per malignancy than HAM"

    _E4 = _json("results/external/conformal_shift_audit.json")
    _aps = {r["cohort"]: r for r in _E4["conformal_audit"]
            if r["method"] == "APS" and r["mondrian"] and abs(r["alpha"] - 0.05) < 1e-9}
    _id, _shift = _aps["HAM OOF tuning half (L0: ID)"], _aps["PAD-UFES-20 (L2: Shift)"]
    for _label, _rec in (("ID", _id), ("shift", _shift)):
        present(f"E4 APS {_label} set size", "%.2f" % _rec["mean_set_size"],
                "conformal_shift_audit.json")
        present(f"E4 APS {_label} singleton", _f(_rec["singleton_rate"], 3),
                "conformal_shift_audit.json")
        present(f"E4 APS {_label} serious coverage", _f(_rec["serious_coverage"], 3),
                "conformal_shift_audit.json")
    assert _shift["mean_set_size"] > _id["mean_set_size"], \
        "conformal sets no longer widen under shift -- the safety-net argument in Sec. IV-J is void"
    assert _shift["serious_coverage"] < _id["serious_coverage"], \
        "serious-class coverage no longer degrades under shift -- Sec. V overstates the caveat"

    _E5 = [r for r in _json("results/external/fitzpatrick_slices.json")
           if r["powered"] and int(r["n_tier1"]) > 0]
    _sens5 = [r["tier1_sensitivity"] for r in _E5]
    for _rec in _E5:
        present(f"E5 tier-1 sens {_rec['group']}", _f(_rec["tier1_sensitivity"]),
                "fitzpatrick_slices.json")
    present("E5 I-IV spread", _f(max(_sens5) - min(_sens5)), "fitzpatrick_slices.json")
    assert _sens5 != sorted(_sens5) and _sens5 != sorted(_sens5, reverse=True), \
        "Fitzpatrick I-IV Tier-1 sensitivity is now monotonic -- the fairness paragraph is wrong"
    assert all(int(r["n_tier1"]) == 0 for r in _json("results/external/fitzpatrick_slices.json")
               if r["group"] == "Unknown"), \
        "the unlabelled stratum now holds Tier-1 lesions -- its rates are no longer undefined"


# --- S19 regression guards: defects that no numeric comparison can catch -----------------------
# Every check below fires on a *paragraph* or a *code path* becoming wrong, not on a digit
# drifting. Each one corresponds to a defect that was actually present in the manuscript or the
# pipeline on 2026-09-06 and that all 348 preceding checks passed straight over.

def absent(label, literal, why):
    """A literal that must NOT reappear in the manuscript."""
    global checks
    checks += 1
    if literal in FULL:
        failures.append(f"{label}: {literal!r} is back -- {why}")


_ABSTRACT = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", SRC, re.S).group(1)

# 1. The abstract quoted 0.831 -- the all-ages escalation sensitivity -- two sentences after the
#    under-40 failure, so it read as the fix for that band. Under-40 reaches only 0.238.
if "0.831" in _ABSTRACT:
    checks += 1
    if "0.238" not in _ABSTRACT:
        failures.append(
            "abstract quotes the all-ages 0.831 without the under-40 0.238 it does NOT fix; "
            "results/session9/agerule_test.csv puts under-40 at 0.238 [0.082, 0.472]")
    checks += 1
    if "mitigated" not in _ABSTRACT:
        failures.append(
            "abstract no longer says the under-40 blind spot is mitigated rather than closed; "
            "Sec. results-agerule says 'we do not present this as a solved problem'")

# 2. Net benefit at a FIXED operating point still declines in p_t, because the false-positive
#    term carries the odds weight p_t/(1-p_t). The caption claimed the curves were flat.
absent("DCA caption flatness", "therefore flat in $p_t$",
       "a fixed operating point holds TP and FP fixed but its net benefit still falls in p_t")
_nb = _E6["panels"]["HAM10000 OOF (N=6,981)"]["reference_points"]
_by_pt = {r["p_t"]: r for r in _nb}
checks += 1
if not _by_pt[0.10]["net_benefit_argmax"] < _by_pt[0.05]["net_benefit_argmax"]:
    failures.append("argmax net benefit no longer declines in p_t -- the DCA caption fix is void")
checks += 1
if not _by_pt[0.10]["net_benefit_lambda_rule"] < _by_pt[0.05]["net_benefit_lambda_rule"]:
    failures.append("lambda-rule net benefit no longer declines in p_t -- DCA caption fix is void")

# 3. The Grad-CAM caption argued from "the attribution is on the lesion" for the ERROR row --
#    the predicted-class artefact the body text explicitly forbids reading that way.
absent("Grad-CAM caption artefact", "the attribution is on the lesion and the model is confident",
       "errors score higher on lesion-interior fraction only because the map is taken w.r.t. the "
       "predicted class; the body says so and the caption must not argue the other way")

# 4. S8b must score PAD through the DEPLOYED Dirichlet map. It read Session 2's calibration map
#    until 2026-09-06, which silently moved every calibrated and age-rule PAD figure.
_S8B_SRC = resolve("research/xdomain/run_session8b.py").read_text(encoding="utf-8")
# Strip the module docstring: it *names* the wrong map in order to record the bug, so a naive
# substring test over the whole file can never fail. Only executable code is checked.
_S8B_CODE = _S8B_SRC.split('"""', 2)[-1] if _S8B_SRC.lstrip().startswith('"""') else _S8B_SRC
checks += 1
if "research/calibration/results_oof/fit_state.json" in _S8B_CODE:
    failures.append(
        "run_session8b.py again references Session 2's calibration map in code; it must load "
        "the deployed research/selective/results_oof/fit_state.json via "
        "research.external.frozen_params (the two differ by max|dW| = 0.18)")
checks += 1
if "from research.external import frozen_params" not in _S8B_CODE:
    failures.append("run_session8b.py no longer imports frozen_params -- it is the only "
                    "permitted loader for a frozen transfer parameter (S12)")

# 5. The conformal caption said marginal calibration under-covers "the escalating classes".
#    akiec sits at target; only mel and bcc are below it.
absent("conformal caption overstatement", "while under-covering the escalating classes",
       "akiec is at target under marginal calibration, so only two of the three are under-covered")

# 6. (manuscript_edited.tex only) Quoting the all-ages age-rule sensitivity 0.831 anywhere in
#    the abstract without the under-40 figure 0.238 in the *same sentence* is the exact defect
#    S19 had to fix once already (guard 1, above, only checks the two appear somewhere in the
#    same abstract -- this is the stricter version the task asked for, scoped to the edited
#    paper specifically rather than to every non-default target).
if TARGET.endswith("manuscript_edited.tex"):
    _sentences = re.split(r"(?<=[.!?])\s+", " ".join(_ABSTRACT.split()))
    _with_0831 = [s for s in _sentences if "0.831" in s]
    checks += 1
    if not _with_0831:
        failures.append("manuscript_edited.tex abstract: 0.831 not found at all -- if the "
                        "all-ages age-rule sensitivity is no longer quoted this guard is moot "
                        "and should be removed, not left silently passing")
    else:
        checks += 1
        if any("0.238" not in s for s in _with_0831):
            failures.append("manuscript_edited.tex abstract: 0.831 appears in a sentence "
                            "without the under-40 figure 0.238 alongside it")
        checks += 1
        if len(_with_0831) > 1:
            failures.append("manuscript_edited.tex abstract: 0.831 appears outside the "
                            "sentence that also carries 0.238 -- quote it once, with the caveat")


# --- Bibliography grew as the Related Work rewrite requires ------------------------------------
if EDITED:
    skip("bibliography size", "the edited manuscript drops whole sections and cites fewer "
                              "works; its own reference count is not audited here")
else:
    claim("bibliography size", 49, SRC.count("\\bibitem{"))

# --- S48 regression guards: the unit of analysis on the under-40 cell ---------------------------
# The manuscript argued at length that resampling images "would treat correlated views as
# independent evidence and produce intervals that are too narrow", and then described its own
# headline count -- 3 of 21 -- as "escalating lesions". They are images, on ten lesions. S48
# measured the cell from ml/configs/splits/split_v1.csv joined to the HAM metadata; the counts
# live in results/v4/power_audit.json and are asserted against it here rather than restated.

_POWER = _json("results/v4/power_audit.json")
_CELL = _POWER["ham_under40_counts"]

absent("under-40 cell called lesions", "of $21$ escalating lesions caught",
       "the 21 are images sitting on 10 lesions (results/v4/power_audit.json, "
       "ham_under40_counts.test); calling them lesions asserts 21 independent units")

for _split, _label in (("test", "test"), ("val", "validation"), ("train", "out-of-fold")):
    present(f"under-40 {_label} image count",
            "$%d$" % _CELL[_split]["escalating_images"], "results/v4/power_audit.json")
present("under-40 test lesion count",
        "$21$ test images sit on", "results/v4/power_audit.json")

# The three lesion counts must all appear, and the test/val cells must be the same size -- if a
# future split changes that, the Limitations sentence "the 22 validation images on 10" is wrong.
checks += 1
if _CELL["test"]["escalating_lesions"] != _CELL["val"]["escalating_lesions"]:
    failures.append(
        "Limitations pairs the test and validation under-40 cells at 10 lesions each, but "
        f"power_audit.json now has {_CELL['test']['escalating_lesions']} and "
        f"{_CELL['val']['escalating_lesions']}")

checks += 1
if not all(_CELL[s]["reconciles_with_split_file"] for s in ("test", "val", "train")):
    failures.append(
        "power_audit.json no longer reconciles its lesion counts against age_band_prior.csv, so "
        "the manuscript's per-split image/lesion pairs are not computed on the same cell")

# Directional: the under-40 cell must stay clustered. If a future corpus gives it ~1 image per
# lesion the whole Power limitation paragraph is obsolete and must be rewritten, not kept.
checks += 1
if _CELL["test"]["images_per_lesion"] < 1.5:
    failures.append(
        "the under-40 test cell is no longer meaningfully clustered "
        f"({_CELL['test']['images_per_lesion']:.2f} images/lesion); the Limitations claim that "
        "image-level intervals are 'correspondingly optimistic' no longer holds")

# The paper must not claim it recomputed the under-40 sensitivity on a lesion denominator -- that
# would be a test quantity outside the frozen plan, and the receipt would have to show it.
checks += 1
if _POWER["test_read"]:
    failures.append("results/v4/power_audit.json reports a test read; S48 declared none")

n_passed = checks - len(failures) - len(skipped)
print(f"target: {TARGET}")
print(f"{checks} checks run -- {n_passed} passed, {len(skipped)} skipped, {len(failures)} failed")

if skipped:
    print(f"\n{len(skipped)} SKIPPED (content deliberately dropped from this target):")
    for s in skipped:
        print("  -", s)

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  -", f)
else:
    print("\nAll manuscript numbers match results/ artifacts.")
raise SystemExit(1 if failures else 0)
