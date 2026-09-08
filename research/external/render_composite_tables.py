r"""Session 15: render the two composite external floats from the frozen JSON reports.

Seven external tables were generated across S8b-S14, one per analysis script, each with its
own caption, notes block and page-break gap. Seven floats is roughly two and a half pages of
an IEEE two-column paper spent on furniture rather than findings, and the manuscript has a
page budget. This module collapses six of them into two `table*` floats with `\multicolumn`
panel headers (the seventh, the TRIPOD+AI table, is reporting apparatus and moves to the
supplementary document instead):

    "External validity battery"  ->  paper/tables/external_table_validity_battery.tex
      (A) three-centre dose-response (E1)      results/external/age_rule_transfer_report.json
      (B) PAD-UFES-20 prior decoupling (E2)    results/external/pad_prior_decoupling_report.json
      (C) three-tier clinical triage (E3)      results/external/clinical_triage_report.json
      (D) decision-curve net benefit (E6)      results/external/decision_curve_report.json

    "Safety nets under shift"    ->  paper/tables/external_table_safety_nets.tex
      (A) Mahalanobis + conformal widening     results/external/conformal_shift_audit.json
      (B) Fitzpatrick I-IV slices              results/external/fitzpatrick_slices.json

Every cell is read from those artifacts; nothing is typed (hard rule 4). Row labels are
mapped through explicit dictionaries so that an unrecognised variant raises rather than being
silently renamed. This is a renderer, not an analysis: it computes no new quantity, reads no
split and writes no ledger row -- the same standing as `run_part_a.py --table-only`.

Two rendering corrections against the superseded per-script tables, both stated in the
captions:

  * The Fitzpatrick "Unlabelled" stratum contains **no Tier-1 lesions at all**, so its Tier-1
    sensitivity, point-FRR and set-FRR are undefined. `eval_fitzpatrick_fairness.py` set them
    to 0.0 in its no-positives branch and then suppressed only the hardcoded groups V and VI,
    so the old table printed a fabricated `0.000 [0.00, 0.00]` for quantities that do not
    exist. Suppression here is driven by the data (`n_tier1`, `powered`), not by a name.
  * Panel (A) of the E1 table is transposed to one row per centre ordered by ascending prior
    skew -- the axis the dose-response hypothesis is actually about -- instead of three
    stacked sub-panels that make the ordering hard to read off.

Session (tooling): a `--exclude-pad` mode renders a second, PAD-free copy of these tables into
`paper/tables_edited/`, for the downsized `paper/manuscript_edited.tex` that drops the whole
PAD-UFES-20 smartphone track. Three things had to be handled row- or panel-level rather than by
skipping a file:

  * The safety-nets table is ENTIRELY PAD (shift detection uses PAD as the shifted cohort;
    Fitzpatrick is a PAD-only slice) -- under `--exclude-pad` it is not emitted at all, and
    nothing in `paper/tables_edited/` references it.
  * The validity battery's panel (B) (PAD prior decoupling, `panel_b_pad_prior`) is dropped
    wholesale and the remaining panels re-letter: (A) dose-response, (B) triage, (C) decision
    curve.
  * Panel (C) triage (`panel_c_triage`) is filtered by row, not dropped by panel: it reads
    `clinical_triage_report.json`'s `cohort` field and keeps every row whose cohort does not
    start with "PAD-UFES-20". Today that source file's non-PAD rows are HAM10000 only; the
    filter is written against the cohort label rather than a hardcoded HAM/PAD split so a
    future update that adds more non-PAD cohorts to that report is picked up for free.
  * `validity_caption()` quotes `pad_prior_decoupling_report.json` and describes panel (B) by
    name -- both regenerated for the reduced panel set, not hand-edited.

Usage:
    python -m research.external.render_composite_tables [--check]
    python -m research.external.render_composite_tables --exclude-pad [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import json

from ml.paths import REPO_ROOT, resolve
from research.selective.fairness import MIN_GROUP_SIZE, MIN_POSITIVES

EXTERNAL = "results/external"

VALIDITY_TABLE = "paper/tables/external_table_validity_battery.tex"
SAFETY_TABLE = "paper/tables/external_table_safety_nets.tex"

#: --exclude-pad targets. The safety-nets table has no PAD-free counterpart at all (see
#: module docstring), so it is simply absent from this map.
VALIDITY_TABLE_EDITED = "paper/tables_edited/external_table_validity_battery.tex"

#: Cohort keys in the E1 report, and how they are named in the paper.
COHORT_LABEL = {
    "bcn20000": "BCN-20000",
    "mskcc": "MSKCC",
    "ham_oof": "HAM10000 (source)",
}

#: E2 variant strings -> short row labels. A KeyError here is the intended behaviour:
#: it means the analysis changed and the table must be re-read, not silently re-labelled.
PAD_VARIANT_LABEL = {
    "Raw Ensemble (Soft-Vote)": "Raw soft-vote",
    "Raw Dirichlet Calibrated": "Dirichlet (deployed)",
    "Deployable EM Prior (Saerens et al.)": "EM prior (Saerens)",
    "Oracle Prior Correction": "Oracle prior",
    "Ordering: Dirichlet -> EM Prior": r"Dirichlet $\rightarrow$ EM",
}

#: E3 cohort strings -> short row labels.
TRIAGE_LABEL = {
    "HAM10000 OOF (Raw Ensemble, N=6,981)": "HAM10000 OOF, raw",
    "HAM10000 OOF (Calibrated Ensemble, N=6,981)": "HAM10000 OOF, calibrated",
    "PAD-UFES-20 (Raw Ensemble)": "PAD-UFES-20, raw",
    "PAD-UFES-20 (Calibrated Ensemble)": "PAD-UFES-20, calibrated",
    "PAD-UFES-20 (Oracle Prior Triage)": "PAD-UFES-20, oracle prior",
}

#: Conformal audit cohort strings -> which half of the paired row they fill.
ID_COHORT = "HAM OOF tuning half (L0: ID)"
SHIFT_COHORT = "PAD-UFES-20 (L2: Shift)"

P_THRESHOLDS = ("0.05", "0.10", "0.15", "0.20")

GENERATED_BY = (
    "% generated by research/external/render_composite_tables.py -- do not hand-edit\n"
    "% supersedes external_table_{bcn_age_replication,pad_prior_shift,cross_cohort_triage,\n"
    "%   decision_curve,conformal_shift,fitzpatrick_slices}.tex, which stay on disk as the\n"
    "%   per-analysis outputs of their own scripts but are no longer part of the manuscript.\n"
)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def _json(name: str) -> Any:
    return json.loads(resolve(f"{EXTERNAL}/{name}").read_text(encoding="utf-8"))


def _f(x: float, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


def _ci(interval: list[float], nd: int = 3) -> str:
    return f"[{float(interval[0]):.{nd}f}, {float(interval[1]):.{nd}f}]"


def _signed(x: float, nd: int = 3) -> str:
    return f"{'+' if float(x) >= 0 else ''}{float(x):.{nd}f}"


def _int(x: float) -> str:
    """An integer with a LaTeX-safe thousands separator, applied to the number only."""
    return f"{int(x):,}".replace(",", r"{,}")


def _sci(p: float, nd: int = 2) -> str:
    """A p-value as LaTeX maths, not as Python's `1.49e-08`."""
    mantissa, exponent = f"{float(p):.{nd}e}".split("e")
    return r"%s \times 10^{%d}" % (mantissa, int(exponent))


def _pick(records: list[dict], **where: Any) -> dict:
    """The single record matching every key/value pair; raises if that is not unique."""
    hits = [r for r in records if all(r.get(k) == v for k, v in where.items())]
    if len(hits) != 1:
        raise KeyError(f"expected exactly 1 record for {where}, found {len(hits)}")
    return hits[0]


def _consumed(cells: list[str]) -> int:
    """How many table columns a list of cells occupies, counting \\multicolumn spans."""
    total = 0
    for cell in cells:
        if r"\multicolumn{" in cell:
            total += int(cell.split(r"\multicolumn{")[1].split("}")[0])
        else:
            total += 1
    return total


def _row(cells: list[str], width: int) -> str:
    """One tabular row, padded with empty cells to the table's column count."""
    used = _consumed(cells)
    if used > width:
        raise ValueError(f"row occupies {used} columns, table is {width} wide: {cells}")
    return " & ".join(cells + [""] * (width - used)) + r" \\"


def _span(text: str, width: int) -> str:
    return r"\multicolumn{%d}{@{}l}{%s} \\" % (width, text)


def _banner(text: str, width: int) -> str:
    """A full-width section header that wraps within the table's linewidth."""
    return r"\multicolumn{%d}{@{}p{\linewidth}@{}}{%s} \\" % (width, text)


# ---------------------------------------------------------------------------------------
# Table: external validity battery  (9 columns)
# ---------------------------------------------------------------------------------------

VALIDITY_COLS = 9


def panel_a_dose_response() -> list[str]:
    """(A) E1: one row per centre, ordered by ascending under-40 prior skew."""
    e1 = _json("age_rule_transfer_report.json")
    by_cohort = {row["cohort"]: row for row in e1["dose_response"]["rows"]}

    lines = [
        _banner(r"\textbf{(A) Three-centre dose--response of the under-40 escalation failure "
                r"(E1).} Centres ordered by ascending prior skew. No parameter is fitted on "
                r"BCN-20000 or MSKCC.", VALIDITY_COLS),
        r"\addlinespace[0.2em]",
        r"Cohort & Skew & \multicolumn{3}{c}{Escalation-mass AUC by age band} & "
        r"\multicolumn{4}{c}{Under-40 escalation sensitivity} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l{0.4em}){6-9}",
        r" & $60{+}/{<}40$ & $<$40 [95\% CI] & 40--59 & 60+ & argmax [95\% CI] & "
        r"mel.\ only & frozen $\lambda$ & $\Delta$ referral \\",
        r"\midrule",
    ]
    for cohort in e1["dose_response"]["ordered_by_skew_ascending"]:
        row = by_cohort[cohort]
        lam = _pick(e1["transfer"], cohort=cohort, band="<40", rule="frozen_lambda")
        arg = _pick(e1["claim_b"], cohort=cohort, band="<40", rule="argmax")
        band_auc = _pick(e1["claim_a"], cohort=cohort, band="40-59")
        # the two reports must agree on the same quantity before either is printed
        assert abs(arg["point"] - row["under40_argmax_sensitivity"]) < 1e-12
        lines.append(_row([
            COHORT_LABEL[cohort],
            _f(row["skew_ratio"], 2) + r"$\times$",
            f"{_f(row['under40_escalation_mass_auc'])} {_ci(row['under40_auc_ci'])}",
            _f(band_auc["auc"]),
            _f(row["over60_escalation_mass_auc"]),
            f"{_f(row['under40_argmax_sensitivity'])} {_ci(row['under40_sensitivity_ci'])}",
            _f(row["under40_melanoma_only_sensitivity"]),
            r"\textbf{%s}" % _f(lam["point"]),
            _signed(lam["delta_referral_rate"]),
        ], VALIDITY_COLS))
    return lines


def panel_b_pad_prior() -> list[str]:
    """(B) E2: how much of the PAD collapse is prior shift rather than optical shift."""
    e2 = _json("pad_prior_decoupling_report.json")
    lines = [
        r"\midrule",
        _banner(r"\textbf{(B) PAD-UFES-20 prior decoupling (E2).} All $N=2{,}106$ images, no "
                r"weights retrained. \textsc{oracle} uses the true target prior and is a "
                r"ceiling, not a deployable system.", VALIDITY_COLS),
        r"\addlinespace[0.2em]",
        _row(["Variant", "Deployable", "Macro-F1", r"Bal.\ acc.", "Accuracy",
              r"Esc.\ sens.", r"Mel.\ recall", "Missed serious"], VALIDITY_COLS),
        r"\midrule",
    ]
    for row in e2["results"]:
        # "Yes", "Yes (Deployable)" -> deployable; "No (Ceiling)" -> the oracle ceiling.
        deployable = (r"yes" if str(row["deployable"]).strip().lower().startswith("yes")
                      else r"\textsc{oracle}")
        lines.append(_row([
            PAD_VARIANT_LABEL[row["variant"]],
            deployable,
            _f(row["macro_f1"], 4),
            _f(row["balanced_accuracy"], 4),
            _f(row["accuracy"], 4),
            _f(row["escalation_sens"], 4),
            _f(row["mel_recall"], 4),
            _int(row["missed_serious"]),
        ], VALIDITY_COLS))
    return lines


def _is_pad_cohort(cohort: str) -> bool:
    return cohort.startswith("PAD-UFES-20")


def panel_c_triage(panel_letter: str = "C", exclude_pad: bool = False) -> list[str]:
    """(C) E3: the same system read as a three-tier clinical action.

    Under `exclude_pad`, rows are filtered by cohort label rather than the panel being
    dropped: `clinical_triage_report.json` mixes HAM10000 and PAD-UFES-20 cohorts in one flat
    list, and only the PAD ones are out of scope for the edited manuscript.
    """
    rows = _json("clinical_triage_report.json")
    if exclude_pad:
        rows = [r for r in rows if not _is_pad_cohort(r["cohort"])]
        if not rows:
            raise ValueError("panel_c_triage: excluding PAD leaves no rows to render -- "
                             "clinical_triage_report.json may now be PAD-only")
    lines = [
        r"\midrule",
        _banner(r"\textbf{(%s) Three-tier clinical actionability (E3).} Tier~1 urgent biopsy "
                r"(\textsc{mel}, \textsc{bcc}, \textsc{scc}); Tier~2 consult (\textsc{akiec}); "
                r"Tier~3 discharge." % panel_letter, VALIDITY_COLS),
        r"\addlinespace[0.2em]",
        r"Cohort / configuration & Tier acc. & T1 sens. & T1 spec. & point-FRR & "
        r"Missed T1 & \multicolumn{3}{c}{NNB at reference $\pi$} \\",
        r"\cmidrule(lr){7-9}",
        r" & & & & & & $\pi{=}0.01$ & $\pi{=}0.03$ & $\pi{=}0.05$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(_row([
            TRIAGE_LABEL[row["cohort"]],
            _f(row["tier_accuracy"], 4),
            _f(row["tier1_sensitivity"], 4),
            _f(row["tier1_specificity"], 4),
            _f(row["point_frr"], 4),
            str(int(row["missed_tier1_as_tier3"])),
            _f(row["nnb_pi_01"], 2),
            _f(row["nnb_pi_03"], 2),
            _f(row["nnb_pi_05"], 2),
        ], VALIDITY_COLS))
    return lines


def panel_d_decision_curve(panel_letter: str = "D") -> list[str]:
    """(D) E6: net benefit of the frozen rule against argmax and biopsy-all."""
    e6 = _json("decision_curve_report.json")
    anchor = e6["ham_test_anchor_from_s9"]
    panels = [
        ("All ages", e6["panels"]["HAM10000 OOF (N=6,981)"], anchor["ALL"]),
        (r"Age $<$40", e6["panels"][r"HAM10000 OOF, age $<$40"], anchor["<40"]),
    ]
    header_d = " & ".join([
        r"$p_t$", "Biopsy all", "Argmax", r"$\lambda$-rule",
        "Risk model", r"$\Delta$", r"95\% CI",
        r"\multicolumn{2}{c}{Test $\Delta^{\dagger}$}",
    ]) + r" \\"

    lines = [
        r"\midrule",
        _banner(r"\textbf{(%s) Decision-curve net benefit of the frozen age rule (E6).} "
                r"$\mathrm{NB} = \mathrm{TP}/N - (\mathrm{FP}/N)\,p_t/(1-p_t)$; "
                r"$\Delta$ is $\lambda$-rule minus argmax under a lesion-grouped paired "
                r"bootstrap, bold where the interval excludes zero." % panel_letter,
                VALIDITY_COLS),
        r"\addlinespace[0.2em]",
        header_d,
        r"\midrule",
    ]
    for title, panel, anchor_band in panels:
        lines.append(_span(r"\textit{%s}" % title, VALIDITY_COLS))
        points = {f"{p['p_t']:.2f}": p for p in panel["reference_points"]}
        for p_t in P_THRESHOLDS:
            point = points[p_t]
            delta_test = (anchor_band["lambda_rule"]["net_benefit"][p_t]
                          - anchor_band["argmax"]["net_benefit"][p_t])
            delta = _signed(point["delta_nb"], 4)
            excludes_zero = point["ci_low"] > 0 or point["ci_high"] < 0
            lines.append(_row([
                p_t,
                _f(point["net_benefit_treat_all"], 4),
                _f(point["net_benefit_argmax"], 4),
                _f(point["net_benefit_lambda_rule"], 4),
                _f(point["net_benefit_risk_model"], 4),
                (r"\textbf{%s}" % delta) if excludes_zero else delta,
                f"[{_signed(point['ci_low'], 4)}, {_signed(point['ci_high'], 4)}]",
                r"\multicolumn{2}{c}{%s}" % _signed(delta_test, 4),
            ], VALIDITY_COLS))
    return lines


def validity_caption(exclude_pad: bool = False) -> str:
    """One caption carrying every note the four superseded tables kept in parbox blocks.

    Under `exclude_pad` panel (B) (PAD prior decoupling) is dropped and the remaining panels
    re-letter to (A) dose-response, (B) triage, (C) decision curve; the sentence describing
    panel (B)'s confirmatory member is regenerated away rather than deleted by hand, since it
    quotes `pad_prior_decoupling_report.json` and would otherwise silently go stale.
    """
    e1 = _json("age_rule_transfer_report.json")
    conf = e1["confirmatory"][e1["confirmatory"]["primary_cohort"]]
    n_panels = "three" if exclude_pad else "four"
    triage_letter, decision_letter = ("B", "C") if exclude_pad else ("C", "D")

    text = (
        f"External validity battery. All {n_panels} panels score the deployed system "
        r"unmodified -- uniform six-CNN soft-vote, 24-view TTA, frozen HAM10000-OOF Dirichlet "
        r"map (\texttt{selective/results\_oof/fit\_state.json}) and frozen per-band $\lambda$ "
        r"(" + ", ".join(r"%s: $%s$" % (label, e1["lambda_by_band"][band])
                         for band, label in ((r"<40", r"$<$40"), ("40-59", "40--59"),
                                             ("60+", "60+"))) + r") "
        r"-- so no parameter in any row is fitted on the cohort it is evaluated on. "
        r"Point estimates are image-level; intervals resample lesions, except proportions "
        r"with fewer than 30 events, which take the exact Clopper--Pearson interval. "
        r"Panel~(A): the pre-registered contingency fired --- Claim~A (a cohort-invariant "
        r"escalation-mass AUC) fails, with a spread of "
        f"{_f(e1['dose_response']['claim_a_auc_spread'])}"
        r" across centres, and the Claim~B ordering is reversed rather than non-monotonic, "
        r"so the three point estimates are reported with intervals and no trend is fitted. "
        r"The confirmatory member \texttt{E1\_under40\_sens\_frozen\_lambda\_vs\_argmax} "
        f"(BCN-20000, exact McNemar over {int(conf['discordant'])} discordant cases) gives "
        f"$p = {_sci(conf['p_value_exact'])}$"
        r", Holm upper bound over the five-member family "
        f"$= {_sci(conf['p_holm_upper_bound'])}$"
        r"; because $\lambda \ge 0$ can only add escalating predictions the test is "
        r"one-sided by construction and certifies that cases were rescued, not that the "
        r"rule is net-beneficial --- that trade is the last two columns. MSKCC's label "
        r"space holds only \textsc{nv}, \textsc{mel} and \textsc{bkl}, so seven-class "
        r"Macro-F1 is undefined there. "
    )
    if not exclude_pad:
        e2 = _json("pad_prior_decoupling_report.json")
        e2c = e2["confirmatory"]
        text += (
            r"Panel~(B) confirmatory member "
            r"\texttt{E2\_em\_macro\_f1\_vs\_raw} is significant in the "
            f"\\emph{{wrong}} direction ($\\chi^2 = {e2c['mcnemar_statistic']:.2f}$, "
            f"$p = {_sci(e2c['p_value'])}$, "
            f"{int(e2c['only_raw_correct'])} cases correct only without the correction against "
            f"{int(e2c['only_em_correct'])} only with it). "
        )
    text += (
        f"Panel~({triage_letter}) point-FRR is the fraction of malignant Tier-1 lesions "
        r"predicted Tier~3; NNB is the number needed to biopsy at reference prevalence $\pi$. "
        f"$^{{\\dagger}}$Panel~({decision_letter})'s test column is reconstructed from the "
        r"frozen Session~9 artifacts (\texttt{results/session9/agerule\_test.csv}: TP, FP and "
        r"$N$ only) and involves no new read of the test split."
    )
    return text


def build_validity_table(exclude_pad: bool = False) -> str:
    body: list[str] = []
    body += panel_a_dose_response()
    triage_letter, decision_letter = ("B", "C") if exclude_pad else ("C", "D")
    if not exclude_pad:
        body += panel_b_pad_prior()
    body += panel_c_triage(panel_letter=triage_letter, exclude_pad=exclude_pad)
    body += panel_d_decision_curve(panel_letter=decision_letter)
    return "\n".join([
        GENERATED_BY.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{%s}" % validity_caption(exclude_pad=exclude_pad),
        r"\label{tab:external_battery}",
        # The caption ends with the dagger-footnote sentence for the last panel; without a gap
        # it and the table's \toprule can run together. \smallskip separates them.
        r"\smallskip",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}l" + "c" * (VALIDITY_COLS - 1) + r"@{}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


# ---------------------------------------------------------------------------------------
# Table: safety nets under shift  (9 columns)
# ---------------------------------------------------------------------------------------

SAFETY_COLS = 9


def panel_a_conformal() -> list[str]:
    """(A) Does the shift get *detected*, and do the conformal sets widen when it happens?"""
    audit = _json("conformal_shift_audit.json")
    maha = audit["mahalanobis_shift"]
    records = audit["conformal_audit"]

    methods: list[tuple[str, str, float, bool]] = []
    for record in records:
        key = (record["method"], record["alpha"], record["mondrian"])
        label = "%s %s $\\alpha=%.2f$" % (
            record["method"],
            "Mondrian" if record["mondrian"] else "marginal",
            record["alpha"],
        )
        if key not in [(m[1], m[2], m[3]) for m in methods]:
            methods.append((label, record["method"], record["alpha"], record["mondrian"]))

    lines = [
        _span(r"\textbf{(A) Shift detection and conformal set widening.} "
              r"In-distribution reference is the HAM10000-OOF tuning half, held out from "
              r"the quantile estimate; the shifted cohort is PAD-UFES-20.", SAFETY_COLS),
        r"\addlinespace[0.2em]",
        _span(r"\emph{Mahalanobis detector} (fit on HAM10000 train features, no refit): "
              r"median score "
              f"{_int(round(maha['median_score_L0_val']))} in distribution against "
              f"{_int(round(maha['median_score_L2_pad']))} under shift "
              f"({maha['separation_ratio_val_vs_pad']:.1f}$\\times$ separation), "
              f"AUROC {_f(maha['auroc_val_vs_pad'], 4)}.",
              SAFETY_COLS),
        r"\addlinespace[0.3em]",
        r"Conformal method & \multicolumn{4}{c}{In distribution (HAM10000 OOF)} & "
        r"\multicolumn{4}{c}{Under shift (PAD-UFES-20)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l{0.4em}){6-9}",
        r" & Marginal & Serious & set-FRR & Mean $|\mathcal{C}|$ & "
        r"Marginal & Serious & set-FRR & Mean $|\mathcal{C}|$ \\",
        r"\midrule",
    ]
    for label, method, alpha, mondrian in methods:
        cells = [label]
        for cohort in (ID_COHORT, SHIFT_COHORT):
            record = _pick(records, cohort=cohort, method=method,
                           alpha=alpha, mondrian=mondrian)
            cells += [
                _f(record["marginal_coverage"], 4),
                _f(record["serious_coverage"], 4),
                _f(record["set_frr"], 4),
                _f(record["mean_set_size"], 2),
            ]
        lines.append(_row(cells, SAFETY_COLS))
    return lines


def panel_b_fitzpatrick() -> list[str]:
    """(B) Fitzpatrick strata, with suppression driven by the counts rather than by name."""
    slices = _json("fitzpatrick_slices.json")
    lines = [
        r"\midrule",
        _span(r"\textbf{(B) Fitzpatrick skin-tone strata (PAD-UFES-20).} "
              r"Pre-registered reporting gates: $N \ge 30$ and $N_{\text{Tier 1}} \ge 10$. "
              r"Intervals are exact Clopper--Pearson; conformal sets are Mondrian "
              r"$\alpha=0.05$.", SAFETY_COLS),
        r"\addlinespace[0.2em]",
        _row(["Stratum", r"$N$", r"$N_{\text{Tier 1}}$", r"Tier-1 sens.\ [95\% CI]",
              "point-FRR", r"set-FRR [95\% CI]", r"Mean $|\mathcal{C}|$"], SAFETY_COLS),
        r"\midrule",
    ]
    for record in slices:
        group = record["group"]
        name = "Unlabelled / missing" if group == "Unknown" else f"Type {group}"
        n_tier1 = int(record["n_tier1"])
        head = [name, _int(record["n"]), str(n_tier1)]
        if not record["powered"]:
            # Which gate failed is itself the finding: types V/VI are too small to say
            # anything about, while the unlabelled stratum is large and simply holds no
            # Tier-1 lesion, so its Tier-1 rates are undefined rather than merely noisy.
            reasons = []
            if int(record["n"]) < MIN_GROUP_SIZE:
                reasons.append(r"$N < %d$" % MIN_GROUP_SIZE)
            if n_tier1 == 0:
                reasons.append("no Tier-1 lesion, so Tier-1 rates are undefined")
            elif n_tier1 < MIN_POSITIVES:
                reasons.append(r"$N_{\text{Tier 1}} < %d$" % MIN_POSITIVES)
            note = r"\emph{suppressed: %s}" % "; ".join(reasons)
            lines.append(_row(head + [r"\multicolumn{3}{c}{%s}" % note,
                                      _f(record["mean_set_size"], 2)], SAFETY_COLS))
        else:
            lines.append(_row(head + [
                f"{_f(record['tier1_sensitivity'])} "
                f"{_ci(record['tier1_sensitivity_ci'])}",
                _f(record["point_frr"]),
                f"{_f(record['set_frr'])} {_ci(record['set_frr_ci'])}",
                _f(record["mean_set_size"], 2),
            ], SAFETY_COLS))
    return lines


def safety_caption() -> str:
    audit = _json("conformal_shift_audit.json")
    slices = _json("fitzpatrick_slices.json")
    powered = [s for s in slices if s["powered"] and int(s["n_tier1"]) > 0]
    sens = [s["tier1_sensitivity"] for s in powered]
    labelled = sum(int(s["n"]) for s in slices if s["group"] != "Unknown")
    total = sum(int(s["n"]) for s in slices)
    dark = [s for s in slices if s["group"] in ("V", "VI")]
    return (
        r"Safety nets under distribution shift. Panel~(A): the shift is detectable and the "
        r"conformal sets widen under it, converting silent point errors into visible "
        r"uncertainty --- but coverage itself degrades sharply, and the guarantee does not "
        r"survive the move. "
        + audit["exchangeability_caveat"].replace("&", r"\&") + " "
        r"Panel~(B): across the "
        f"{len(powered)} adequately powered strata the Tier-1 sensitivity spread is "
        f"{_f(max(sens) - min(sens))} and it is \\emph{{not}} monotonic in Fitzpatrick type "
        + ", ".join(f"({s['group']}~{_f(s['tier1_sensitivity'])}" if i == 0 else
                    f"{s['group']}~{_f(s['tier1_sensitivity'])}"
                    for i, s in enumerate(powered)) +
        r"), so these data do not reproduce the reported darker-skin penalty. "
        f"Only {_int(labelled)} of the {_int(total)} lesions carry a Fitzpatrick label at "
        r"all; the two darkest strata are suppressed at "
        + " and ".join(r"$n=%d$" % int(s["n"]) for s in dark) +
        r", so this cohort cannot speak to darker skin. The unlabelled remainder holds no "
        r"Tier-1 lesion, so a pooled labelled-versus-unlabelled gap is a data-completeness "
        r"artefact and is not reported here as a skin-tone effect."
    )


def build_safety_table() -> str:
    body = panel_a_conformal() + panel_b_fitzpatrick()
    return "\n".join([
        GENERATED_BY.rstrip("\n"),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{%s}" % safety_caption(),
        r"\label{tab:external_safety}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "c" * (SAFETY_COLS - 1) + r"@{}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


# ---------------------------------------------------------------------------------------

def _column_audit(tex: str, width: int, name: str) -> list[str]:
    """Every data row must have exactly `width` cells; LaTeX would fail loudly, we cannot."""
    problems = []
    for line_no, line in enumerate(tex.splitlines(), 1):
        stripped = line.strip()
        if not stripped.endswith(r"\\") or stripped.startswith("%"):
            continue
        content = stripped[:-2]
        cells = content.split("&")
        spanned = 0
        for cell in cells:
            if r"\multicolumn{" in cell:
                spanned += int(cell.split(r"\multicolumn{")[1].split("}")[0]) - 1
        if len(cells) + spanned != width:
            problems.append(f"{name}:{line_no}: {len(cells) + spanned} columns "
                            f"(expected {width}): {stripped[:90]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="render in memory and report drift without writing the files")
    parser.add_argument("--exclude-pad", action="store_true",
                        help="render the PAD-free variants into paper/tables_edited/ instead "
                             "(no safety-nets table is emitted -- see module docstring)")
    args = parser.parse_args(argv)

    if args.exclude_pad:
        outputs = {
            VALIDITY_TABLE_EDITED: (build_validity_table(exclude_pad=True), VALIDITY_COLS),
        }
    else:
        outputs = {
            VALIDITY_TABLE: (build_validity_table(), VALIDITY_COLS),
            SAFETY_TABLE: (build_safety_table(), SAFETY_COLS),
        }

    problems: list[str] = []
    for path, (tex, width) in outputs.items():
        problems += _column_audit(tex, width, Path(path).name)
    if problems:
        print("column-count audit FAILED:")
        for problem in problems:
            print("  -", problem)
        return 1

    for path, (tex, _) in outputs.items():
        target = resolve(path)
        if args.check:
            current = target.read_text(encoding="utf-8") if target.is_file() else ""
            status = "up to date" if current == tex else "DRIFTED"
            print(f"{status}: {path}")
            if current != tex:
                return 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(tex, encoding="utf-8")
        rows = sum(1 for line in tex.splitlines() if line.strip().endswith(r"\\"))
        print(f"Wrote {target.relative_to(REPO_ROOT)} ({rows} rows, {len(tex):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
