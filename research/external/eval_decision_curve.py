"""Workstream E6: decision curve analysis for the escalation rule (rebuilt in S12).

**What was wrong with the first version.** It read `research/predictions/{arch}_test.csv`
-- the plain 1-view matrix, on the *test* split, outside the pre-registered S9 pass -- and
thresholded `S_esc` against a hardcoded scalar that appears nowhere in `results/`, which is
not the frozen rule. Three separate integrity failures in one figure. All three are
fixed here: the panel is the OOF+TTA+Dirichlet matrix, the rule is loaded from
`research.external.frozen_params`, and the in-distribution *test* column is reconstructed
from the frozen S9 artifacts rather than by re-reading test.

**What was wrong with it as DCA.** Vickers' decision curve varies the decision with the
threshold probability: at `p_t` you act iff the model's risk exceeds `p_t`, so the curve
traces a *model*, and its envelope against treat-all/treat-none is the thing being read.
The old code plotted one fixed rule as a flat-decision curve at every `p_t`, which is a
legitimate object (the net benefit of a deployed operating point) but is not a decision
curve and cannot be compared to treat-all the way the caption claimed. Both are now drawn
and labelled as what they are:

  * **risk model** -- biopsy iff `S_esc >= p_t`. The escalation mass swept as a risk score.
    This is the Vickers curve.
  * **fixed operating points** -- argmax triage, and the frozen band-conditional lambda
    rule. Horizontal decisions, re-scored at each `p_t`'s exchange rate.

A fixed rule can only touch the risk-model curve at the `p_t` where its referral rate
matches; elsewhere it is below by construction. Reporting delta-NB against argmax (a rule
the clinic would otherwise deploy) is the honest comparison, and it is the one that gets
confidence intervals: lesion-grouped, paired, from `research.ablation.bootstrap`.

Finally the whole analysis is repeated **within the under-40 band**, because that is where
the paper's claim lives and an aggregate curve is dominated by the two older bands.

    $py -m research.external.eval_decision_curve [--n-boot 1000] [--no-pad]

Outputs: `paper/figures/external_figure_decision_curve.png`,
`paper/tables/external_table_decision_curve.tex`,
`results/external/decision_curve_report.json`.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, resolve
from research.ablation.bootstrap import lesion_resample_indices
from research.experiment_log import log_experiment
from research.external import frozen_params as fp

SESSION = "session_post_s11"
REF_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
CURVE_THRESHOLDS = np.linspace(0.01, 0.35, 100)
S9_AGERULE_TEST = "results/session9/agerule_test.csv"
PAD_PRED_DIR = "research/predictions_pad"
PAD_MANIFEST = "ml/data/manifest_pad.csv"
#: PAD carries squamous cell carcinoma, which HAM does not; it is escalating regardless of
#: the 7-class head being unable to name it. Matches `run_session8b`'s convention.
PAD_ESCALATING_CODES = ("mel", "bcc", "scc", "akiec")


# ------------------------------------------------------------------------ net benefit
def net_benefit(y_esc: np.ndarray, act: np.ndarray, p_t: float) -> float:
    """NB = TP/N - (FP/N) * p_t/(1-p_t) -- Vickers & Elkin (2006).

    `act` is the biopsy decision, so this scores a *rule*, not a probability. The exchange
    rate p_t/(1-p_t) is the number of unnecessary biopsies a clinician would trade for one
    missed malignancy at that threshold.
    """
    n = len(y_esc)
    tp = int((act & y_esc).sum())
    fp = int((act & ~y_esc).sum())
    return tp / n - (fp / n) * (p_t / (1.0 - p_t))


def net_benefit_curve(y_esc: np.ndarray, act: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.array([net_benefit(y_esc, act, float(p)) for p in thresholds])


def risk_model_curve(y_esc: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """The Vickers curve: the decision itself moves with `p_t`."""
    return np.array([net_benefit(y_esc, score >= p, float(p)) for p in thresholds])


def treat_all_curve(y_esc: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = float(y_esc.mean())
    return prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))


# ----------------------------------------------------------------------------- panels
class Panel:
    """A cohort's escalation labels, decisions and lesion grouping, ready for DCA."""

    def __init__(self, name, y_esc, act_argmax, act_rule, s_esc, lesion_ids):
        self.name = name
        self.y_esc = y_esc
        self.act_argmax = act_argmax
        self.act_rule = act_rule
        self.s_esc = s_esc
        self.lesion_ids = lesion_ids

    def __len__(self) -> int:
        return len(self.y_esc)

    def subset(self, mask: np.ndarray, name: str) -> "Panel":
        return Panel(name, self.y_esc[mask], self.act_argmax[mask], self.act_rule[mask],
                     self.s_esc[mask], self.lesion_ids[mask])


def ham_panels() -> tuple[Panel, Panel]:
    """The in-distribution panel (OOF+TTA+Dirichlet) and its under-40 sub-panel."""
    ham = fp.load_ham_oof_panel()
    esc = fp.escalating_indices()
    y_esc = np.isin(ham.y_true, esc)
    act_argmax = np.isin(ham.probs.argmax(axis=1), esc)
    act_rule = np.isin(fp.apply_age_rule(ham.probs, bands=ham.bands), esc)
    full = Panel(f"HAM10000 OOF (N={len(ham):,})", y_esc, act_argmax, act_rule,
                 fp.escalation_mass(ham.probs), ham.lesion_ids)
    under40 = full.subset(ham.bands == "<40", "HAM10000 OOF, age $<$40")
    return full, under40


def pad_panel() -> Panel | None:
    """PAD-UFES-20 under cross-modality shift, or None if its matrices are absent."""
    codes = load_class_mapping().codes
    frames = {}
    for arch in fp.ARCHS:
        path = resolve(PAD_PRED_DIR) / f"{arch}.csv"
        if not path.is_file():
            return None
        frames[arch] = pd.read_csv(path).set_index("image_id").sort_index()
    ids = frames[fp.ARCHS[0]].index
    probs_raw = np.mean([frames[a][[f"p_{c}" for c in codes]].to_numpy() for a in fp.ARCHS], axis=0)
    probs = fp.calibrate(probs_raw)

    manifest = pd.read_csv(resolve(PAD_MANIFEST)).set_index("image_id").loc[ids]
    esc = fp.escalating_indices()
    # PAD's own label set, not the 7-class head's: `scc` is escalating and unnameable here.
    y_esc = manifest["class_code"].isin(PAD_ESCALATING_CODES).to_numpy()
    act_argmax = np.isin(probs.argmax(axis=1), esc)
    act_rule = np.isin(fp.apply_age_rule(probs, manifest["age"].to_numpy(dtype=float)), esc)
    lesion_ids = manifest["lesion_id"].fillna(pd.Series(ids, index=manifest.index)).astype(str).to_numpy()
    return Panel(f"PAD-UFES-20 (N={len(y_esc):,})", y_esc, act_argmax, act_rule,
                 fp.escalation_mass(probs), lesion_ids)


# ------------------------------------------------------------------------- inference
def delta_nb_ci(panel: Panel, p_t: float, n_boot: int, seed: int = 42) -> dict:
    """Paired lesion-grouped bootstrap CI for NB(lambda rule) - NB(argmax) at one `p_t`.

    Paired: both rules are scored on the *same* resample each draw, so the interval is on
    the difference and not on the difference of two independent intervals.
    """
    point = net_benefit(panel.y_esc, panel.act_rule, p_t) - net_benefit(
        panel.y_esc, panel.act_argmax, p_t)
    draws = np.empty(n_boot)
    for i, idx in enumerate(lesion_resample_indices(panel.lesion_ids, n_boot, seed)):
        y_b = panel.y_esc[idx]
        draws[i] = net_benefit(y_b, panel.act_rule[idx], p_t) - net_benefit(
            y_b, panel.act_argmax[idx], p_t)
    return {
        "delta_nb": float(point),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_value_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
        "n_boot": n_boot,
    }


def s9_test_anchor(path: str = S9_AGERULE_TEST) -> dict[str, dict[str, dict]]:
    """Net benefit on HAM test, reconstructed from frozen S9 artifacts -- **no test read**.

    NB needs only TP, FP and N, all three of which `results/session9/agerule_test.csv`
    already records per band and per rule from the single pre-registered pass. The same
    precedent S10 set when `_render_table_only` read `age_gap_test.csv` to fill Table IV's
    test column. Nothing here opens a prediction matrix.
    """
    frame = pd.read_csv(resolve(path))
    out: dict[str, dict[str, dict]] = {}
    for _, row in frame.iterrows():
        n = int(row["n"])
        n_esc = int(row["n_escalating"])
        tp = int(row["n_caught"])
        # referral_rate * n is the predicted-positive count; FP is what is left after TP.
        predicted_positive = int(round(float(row["referral_rate"]) * n))
        fp_count = predicted_positive - tp
        out.setdefault(str(row["band"]), {})[str(row["rule"])] = {
            "n": n, "n_escalating": n_esc, "tp": tp, "fp": fp_count,
            "prevalence": n_esc / n if n else float("nan"),
            "net_benefit": {f"{p:.2f}": tp / n - (fp_count / n) * (p / (1 - p))
                            for p in REF_THRESHOLDS},
        }
    return out


# ------------------------------------------------------------------------------ output
def _plot(panels: list[Panel], out_png) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        ax.plot(CURVE_THRESHOLDS, treat_all_curve(panel.y_esc, CURVE_THRESHOLDS),
                ls=":", color="#7f7f7f", lw=1.5, label="Biopsy all")
        ax.axhline(0, ls="-.", color="#bcbd22", lw=1.2, label="Biopsy none")
        ax.plot(CURVE_THRESHOLDS, risk_model_curve(panel.y_esc, panel.s_esc, CURVE_THRESHOLDS),
                color="#2ca02c", lw=2.2, label=r"Risk model: $S_{\mathrm{esc}} \geq p_t$")
        ax.plot(CURVE_THRESHOLDS, net_benefit_curve(panel.y_esc, panel.act_argmax, CURVE_THRESHOLDS),
                ls="--", color="#1f77b4", lw=2.0, label="Argmax triage (fixed)")
        ax.plot(CURVE_THRESHOLDS, net_benefit_curve(panel.y_esc, panel.act_rule, CURVE_THRESHOLDS),
                color="#d62728", lw=2.4,
                label=r"Frozen $\lambda$-rule, per band (fixed)")
        ax.set_title(panel.name, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Threshold probability $p_t$", fontsize=10)
        ax.set_xlim(0.01, 0.35)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Net benefit", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _tex(report: dict, full: Panel, under40: Panel, out_tex) -> None:
    anchor = report["ham_test_anchor_from_s9"]["ALL"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Decision curve analysis of the frozen age-conditional escalation rule. "
        r"Net benefit $=\mathrm{TP}/N-(\mathrm{FP}/N)\,p_t/(1-p_t)$. The HAM10000 columns "
        r"are the out-of-fold panel (24-view TTA, frozen Dirichlet map, $N=6{,}981$); "
        r"$\Delta$ is the frozen $\lambda$-rule minus argmax triage with a lesion-grouped "
        r"paired bootstrap interval. The test column is reconstructed from the frozen "
        r"Session~9 artifacts (TP, FP and $N$ only) and involves no new read of the test "
        r"split.}",
        r"\label{tab:decision_curve}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"$p_t$ & Biopsy all & Argmax & $\lambda$-rule & $\Delta$ [95\% CI] & Test $\Delta$\textsuperscript{a} \\",
        r"\midrule",
    ]
    for panel, heading, band in ((full, "All ages", "ALL"), (under40, r"Age $<$40", "<40")):
        rows = report["panels"][panel.name]["reference_points"]
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{heading}}}}} \\")
        for row in rows:
            p = row["p_t"]
            key = f"{p:.2f}"
            test = report["ham_test_anchor_from_s9"][band]
            d_test = (test["lambda_rule"]["net_benefit"][key]
                      - test["argmax"]["net_benefit"][key])
            # Bold only where the paired interval excludes zero. Bolding every lambda-rule
            # cell would assert a win in the under-40 band, which is exactly the row where
            # the rule does *not* separate from argmax.
            nb_rule = f"{row['net_benefit_lambda_rule']:.4f}"
            if row["ci_low"] > 0 or row["ci_high"] < 0:
                nb_rule = rf"\textbf{{{nb_rule}}}"
            lines.append(
                rf"{p:.2f} & {row['net_benefit_treat_all']:.4f} & "
                rf"{row['net_benefit_argmax']:.4f} & {nb_rule} & "
                rf"{row['delta_nb']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] & "
                rf"{d_test:+.4f} \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\parbox{\linewidth}{\raggedright \textsuperscript{a}Derived from frozen Session~9 "
        r"artifacts (\texttt{results/session9/agerule\_test.csv}); no new test read. "
        rf"Test prevalence {anchor['argmax']['prevalence']:.3f} against "
        rf"{report['panels'][full.name]['prevalence']:.3f} out of fold.}}",
        r"\end{table}",
        "",
    ]
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pad", action="store_true", help="skip the PAD-UFES-20 panel")
    args = parser.parse_args(argv)

    print("=== E6: decision curve analysis (S12 rebuild) ===")
    full, under40 = ham_panels()
    panels = [full, under40]
    if not args.no_pad:
        pad = pad_panel()
        if pad is None:
            print("PAD matrices absent -- skipping the shift panel.")
        else:
            panels.append(pad)

    report: dict = {
        "session": SESSION,
        "workstream": "E6_decision_curves",
        "status": "post_hoc_secondary; declared in analysis_plan_post_s11_v2.json",
        "ham_panel": f"{fp.OOF_PREDICTIONS_DIR}/{{arch}}_{fp.OOF_SPLIT}.csv",
        "calibration": fp.DIRICHLET_STATE,
        "lambda_by_band": fp.load_lambda_by_band(),
        "rule": "argmax_c ( p_c + lambda_band * 1[c escalates] )",
        "test_read": False,
        "panels": {},
        "ham_test_anchor_from_s9": s9_test_anchor(),
    }

    for panel in panels:
        rows = []
        for p_t in REF_THRESHOLDS:
            ci = delta_nb_ci(panel, p_t, args.n_boot, args.seed)
            rows.append({
                "p_t": p_t,
                "net_benefit_treat_all": float(treat_all_curve(panel.y_esc, np.array([p_t]))[0]),
                "net_benefit_argmax": net_benefit(panel.y_esc, panel.act_argmax, p_t),
                "net_benefit_lambda_rule": net_benefit(panel.y_esc, panel.act_rule, p_t),
                "net_benefit_risk_model": net_benefit(panel.y_esc, panel.s_esc >= p_t, p_t),
                **ci,
            })
        report["panels"][panel.name] = {
            "n": len(panel),
            "n_escalating": int(panel.y_esc.sum()),
            "prevalence": float(panel.y_esc.mean()),
            "referral_rate_argmax": float(panel.act_argmax.mean()),
            "referral_rate_lambda_rule": float(panel.act_rule.mean()),
            "reference_points": rows,
        }
        print(f"\n{panel.name}: prevalence {panel.y_esc.mean():.3f}, "
              f"referral {panel.act_argmax.mean():.3f} -> {panel.act_rule.mean():.3f}")
        print(pd.DataFrame(rows)[["p_t", "net_benefit_argmax", "net_benefit_lambda_rule",
                                  "net_benefit_risk_model", "delta_nb", "ci_low", "ci_high",
                                  "p_value_two_sided"]].to_string(index=False))

    fig_dir = resolve("paper/figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = resolve("paper/tables"); tab_dir.mkdir(parents=True, exist_ok=True)
    res_dir = resolve("results/external"); res_dir.mkdir(parents=True, exist_ok=True)

    _plot(panels, fig_dir / "external_figure_decision_curve.png")
    _tex(report, full, under40, tab_dir / "external_table_decision_curve.tex")
    (res_dir / "decision_curve_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    for panel in panels:
        block = report["panels"][panel.name]
        row = next(r for r in block["reference_points"] if r["p_t"] == 0.10)
        log_experiment({
            "session": SESSION,
            "method": f"E6_dca_lambda_vs_argmax[{panel.name}]",
            "split": "oof" if panel.name.startswith("HAM") else "pad",
            "escalation_sens": round(float((panel.act_rule & panel.y_esc).sum()
                                           / max(int(panel.y_esc.sum()), 1)), 4),
            "missed_serious": int((panel.y_esc & ~panel.act_rule).sum()),
            "p_value_vs_baseline": round(row["p_value_two_sided"], 4),
            "notes": (f"delta net benefit at p_t=0.10 = {row['delta_nb']:+.4f} "
                      f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], "
                      f"{args.n_boot}x lesion-grouped paired bootstrap; frozen band lambdas; "
                      f"no test read (test column from results/session9/agerule_test.csv)"),
        })

    print(f"\nWrote {fig_dir / 'external_figure_decision_curve.png'}")
    print(f"Wrote {tab_dir / 'external_table_decision_curve.tex'}")
    print(f"Wrote {res_dir / 'decision_curve_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
