"""Review-2 figures and tables for V5, built only from result files (no model is run, nothing hand-entered).

    python -m research.v5.review2_figures [--confirm results/v5/confirm_s42_s43.json]

Writes results/v5/figures/{fig_confirmation_forest,fig_u40_by_archive,fig_screens}.{png,svg} and
results/v5/review2_tables.md. Sources: results/v5/screens/*.json, results/v5/precheck_verdicts.json,
the confirmation JSON passed with --confirm (default: the newest of confirm_s42*.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research.v4.recipe import REPO_ROOT  # noqa: E402

V5 = REPO_ROOT / "results" / "v5"
SCREENS = V5 / "screens"
FIG = V5 / "figures"
# Reference palette (dataviz skill, light mode): one series hue + neutral inks.
SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

# arm, gate file, comparator label, falsifier outcome (from the screen/falsifier files and the Q5 read)
SCREEN_ROWS = [
    ("IN-22k trunk", "gate_control_in22k_vs_control", "in1k control", "n/a (trunk)", True),
    ("twostep", "gate_twostep_in22k_vs_control", "control", "PASS", True),
    ("m4", "gate_m4_in22k_vs_twostep", "twostep", "PASS", True),
    ("youngdata", "gate_youngdata_in22k_vs_control", "control", "PASS (within-band)", True),
    ("look (fixed)", "gate_look_in22k_vs_control_v5fix", "control", "not reached", False),
    ("geometry (fixed)", "gate_geometry_in22k_vs_control_v5fix", "control", "FAIL (rotation keeps 75%)", False),
    ("clues", "gate_clues_in22k_vs_twostep", "twostep", "not reached", False),
    ("gem", "gate_gem_in22k_vs_clues", "clues", "not reached", False),
    ("memory", "gate_memory_in22k_vs_control", "control", "FAIL (prototypes not young-specific)", False),
    ("zoom", "gate_zoom_in22k_vs_clues", "clues", "FAIL (random crop wins 3/3)", False),
]
NOT_RUN = [("structure", "Q5 pre-check FAIL: 0 young structure tokens"),
           ("m5", "Q4 pre-check FAIL: 416 mask overlaps < 1,000"),
           ("m7", "LOAO FAIL: mean present-class Macro-F1 −0.033")]


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def forest(conf: dict) -> None:
    d = conf["primary"]["delta"]
    seeds = conf["seeds"]
    rows = [("All-age escalation pAUC (Gate D)", "pauc_all", None),
            ("Under-40 escalation pAUC (Gate A)", "pauc_u40", 0.050),
            ("pAUC, histopathology rows", "pauc_histo", None),
            ("Macro-F1 (Gate B, floor −0.010)", "macro_f1", -0.010),
            ("Balanced accuracy", "balanced_accuracy", None),
            ("Sensitivity @ spec 0.80, 40–59 (Gate C)", "sens_at_spec80_40-59", -0.030),
            ("Sensitivity @ spec 0.80, 60+ (Gate C)", "sens_at_spec80_60+", -0.030),
            ("Under-40, class-standardised pAUC (B2)", "b2_std_pauc_u40", None)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    style(ax, f"Composite − in1k control, held-out folds 1–4 (seeds {', '.join(map(str, seeds))})")
    for i, (label, key, bar) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = d[key]["ci95"]
        ax.plot([lo, hi], [y, y], color=SERIES, linewidth=2, solid_capstyle="round")
        ax.plot(d[key]["point"], y, "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        for r in conf["primary"]["per_seed"]:
            ax.plot(r["delta"][key], y + 0.22, "o", color=INK2, markersize=3.5, alpha=0.8)
        if bar is not None:
            ax.plot([bar, bar], [y - 0.32, y + 0.32], color=INK, linewidth=1.2, linestyle=(0, (2, 2)))
        ax.text(max(hi, d[key]["point"]) + 0.004, y, f"{d[key]['point']:+.3f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK, fontsize=9)
    ax.set_xlabel("Δ (composite − control), 95% hierarchical bootstrap CI;  small grey dots = per seed;  "
                  "dashed = pre-declared gate threshold", color=INK2, fontsize=8)
    save(fig, "fig_confirmation_forest")


def archives(conf: dict) -> None:
    d = conf["primary"]["delta"]
    n = conf["u40_escalating_lesions_by_archive"]
    rows = [(f"BCN20000 (n = {n['bcn20000']} lesions)", "b4_pauc_u40_bcn20000"),
            (f"HAM10000 (n = {n['ham']})", "b4_pauc_u40_ham"),
            (f"MSKCC (n = {n['mskcc']})", "b4_pauc_u40_mskcc"),
            ("Pooled, archive fixed effects", "b4_pooled_fe"),
            ("All under-40 rows", "pauc_u40")]
    ctrl = conf["primary"]["per_seed"][0]["b"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), facecolor=SURFACE)
    style(ax, "Under-40 escalation pAUC gain by archive (B4)")
    for i, (label, key) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = d[key]["ci95"]
        ax.plot([lo, hi], [y, y], color=SERIES, linewidth=2, solid_capstyle="round")
        ax.plot(d[key]["point"], y, "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        extra = f"   control {ctrl[key]:.2f}" if key in ctrl else ""
        ax.text(hi + 0.01, y, f"{d[key]['point']:+.3f}{extra}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK, fontsize=9)
    ax.set_xlabel("Δ under-40 pAUC@0.20 (composite − control), 95% CI; 'control' = control pAUC, seed "
                  f"{conf['seeds'][0]}", color=INK2, fontsize=8)
    save(fig, "fig_u40_by_archive")


def screens() -> list[dict]:
    out = []
    for arm, f, comp, fals, kept in SCREEN_ROWS:
        g = json.loads((SCREENS / f"{f}.json").read_text(encoding="utf-8"))
        out.append({"arm": arm, "comparator": comp, "d_all": g["mean_delta"]["pauc_all"],
                    "bar": g["thresholds"]["pauc_all"], "d_histo": g["mean_delta"]["pauc_histo"],
                    "d_u40": g["mean_delta"].get("pauc_u40", float("nan")),
                    "d_f1": g["mean_delta"]["macro_f1"], "gate": bool(g["gate_pass_before_falsifier"]),
                    "falsifier": fals, "kept": kept})
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURFACE)
    style(ax, "V5 fold-0 screens: Δ all-age escalation pAUC vs comparator (3 seeds)")
    for i, r in enumerate(out):
        y = len(out) - 1 - i
        filled = r["kept"]
        ax.plot([0, r["d_all"]], [y, y], color=GRID if not filled else SERIES, linewidth=2)
        ax.plot(r["d_all"], y, "o", markersize=8, color=SERIES if filled else SURFACE,
                markeredgecolor=SERIES if filled else INK2, markeredgewidth=1.8)
        verdict = "kept" if filled else ("falsified" if r["falsifier"].startswith("FAIL") else "gate fail")
        ax.text(max(r["d_all"], out[1]["bar"]) + 0.0006, y, f"{r['d_all']:+.4f}  {verdict}", va="center",
                fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.axvline(out[1]["bar"], color=INK, linewidth=1.2, linestyle=(0, (2, 2)))
    ax.text(out[1]["bar"] + 0.0002, -0.75, "screen bar (80th pct of null)", fontsize=8, color=INK2, va="center")
    ax.set_ylim(-1.1, len(out) - 0.5)
    ax.set_yticks(range(len(out)))
    ax.set_yticklabels([f"{r['arm']}  vs {r['comparator']}" for r in out][::-1], color=INK, fontsize=9)
    ax.set_xlabel("Δ all-age pAUC@0.20 (filled = kept in the locked composite or trunk; hollow = excluded). A gate pass on "
                  "either all-age or histopathology pAUC counts (memory passed on histo).",
                  color=INK2, fontsize=8)
    save(fig, "fig_screens")
    return out


def tables(scr: list[dict], conf: dict, conf_name: str) -> None:
    d, g = conf["primary"]["delta"], conf["gates"]
    lines = ["# V5 Review-2 tables (generated by `research/v5/review2_figures.py`; do not hand-edit)", "",
             "## Screens (fold 0, 3 seeds; sources `results/v5/screens/*.json`)", "",
             "| Arm | vs | Δ pAUC all (bar) | Δ pAUC histo | Δ <40 pAUC | Δ Macro-F1 | Gate | Falsifier |",
             "|---|---|---|---|---|---|---|---|"]
    for r in scr:
        lines.append(f"| {r['arm']} | {r['comparator']} | {r['d_all']:+.4f} ({r['bar']:.4f}) | {r['d_histo']:+.4f} | "
                     f"{r['d_u40']:+.4f} | {r['d_f1']:+.4f} | {'pass' if r['gate'] else 'fail'} | {r['falsifier']} |")
    for arm, why in NOT_RUN:
        lines.append(f"| {arm} | — | not run | | | | — | {why} |")
    lines += ["", f"## Confirmation (folds 1–4, seeds {conf['seeds']}; source `results/v5/{conf_name}`)", "",
              "| Endpoint | Δ composite − control | 95% CI | Gate |", "|---|---|---|---|"]
    gate_of = {"pauc_all": f"D: {g['D']['status'] if g['D']['pass'] is None else ('pass' if g['D']['pass'] else 'FAIL')}",
               "pauc_u40": f"A: {'pass' if g['A']['pass'] else 'FAIL'} (needs ≥ +0.050 and CI > 0)",
               "macro_f1": f"B: {'pass' if g['B']['pass'] else 'FAIL'}",
               "sens_at_spec80_40-59": f"C: {'pass' if g['C']['pass'] else 'FAIL'}",
               "sens_at_spec80_60+": f"C: {'pass' if g['C']['pass'] else 'FAIL'}"}
    for key, label in [("pauc_all", "all-age escalation pAUC"), ("pauc_u40", "under-40 escalation pAUC"),
                       ("pauc_histo", "pAUC histopathology rows"), ("macro_f1", "Macro-F1"),
                       ("balanced_accuracy", "balanced accuracy"), ("esc_sens_argmax", "escalation sensitivity (argmax)"),
                       ("sens_at_spec80_40-59", "sensitivity @ spec 0.80, 40–59"),
                       ("sens_at_spec80_60+", "sensitivity @ spec 0.80, 60+"),
                       ("b2_melnv_auc_u40", "B2 MEL-vs-NV AUC, under 40"),
                       ("b2_std_pauc_u40", "B2 class-standardised under-40 pAUC"),
                       ("b4_pooled_fe", "B4 archive-fixed-effects under-40 pAUC")]:
        lo, hi = d[key]["ci95"]
        lines.append(f"| {label} | {d[key]['point']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {gate_of.get(key, 'descriptive')} |")
    if g["D"]["pass"] is None and g["D"]["n_positive"] == 0 and len(conf["seeds"]) == 2:
        lines.append("")
        lines.append("Gate D is arithmetically settled as FAIL: both available seeds are ≤ 0, so ≥ 2 of 3 positive "
                     "seeds is impossible whatever seed 44 shows.")
    lines += ["", "## Absolute values on the same rows (folds 1–4, 12,235 images, last epoch, single models)", "",
              "| System | Seed | Macro-F1 | Balanced acc. | All-age pAUC | Under-40 pAUC | Escalation sens. |",
              "|---|---|---|---|---|---|---|"]
    for r in conf["primary"]["per_seed"]:
        for name, m in (("in1k control (V4 recipe)", r["b"]), ("V5 locked composite", r["a"])):
            lines.append(f"| {name} | {r['seed']} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} | "
                         f"{m['pauc_all']:.4f} | {m['pauc_u40']:.4f} | {m['esc_sens_argmax']:.4f} |")
    q11_p = V5 / "q11_decomposition.json"
    if q11_p.is_file():
        q11 = json.loads(q11_p.read_text(encoding="utf-8"))
        lines += ["", f"## Q11 Trunk vs Module Decomposition (folds 1–4, seeds {q11['seeds']}; source `results/v5/q11_decomposition.json`)", "",
                  "| Contrast | Under-40 pAUC [95% CI] | p | Macro-F1 [95% CI] | p | All-age pAUC [95% CI] | p |",
                  "|---|---|---|---|---|---|---|"]
        m_u = q11["module_effect"]["delta"]["pauc_u40"]
        m_f1 = q11["module_effect"]["delta"]["macro_f1"]
        m_a = q11["module_effect"]["delta"]["pauc_all"]
        lines.append(f"| Module effect: composite − Q11 control (same IN-22k trunk) | {m_u['point']:+.4f} [{m_u['ci95'][0]:+.4f}, {m_u['ci95'][1]:+.4f}] | {m_u['p_two_sided']:.3f} | "
                     f"{m_f1['point']:+.4f} [{m_f1['ci95'][0]:+.4f}, {m_f1['ci95'][1]:+.4f}] | {m_f1['p_two_sided']:.3f} | "
                     f"{m_a['point']:+.4f} [{m_a['ci95'][0]:+.4f}, {m_a['ci95'][1]:+.4f}] | {m_a['p_two_sided']:.3f} |")

        t_u = q11["trunk_effect"]["delta"]["pauc_u40"]
        t_f1 = q11["trunk_effect"]["delta"]["macro_f1"]
        t_a = q11["trunk_effect"]["delta"]["pauc_all"]
        lines.append(f"| Trunk effect: Q11 control − in1k control (trunk swap only) | {t_u['point']:+.4f} [{t_u['ci95'][0]:+.4f}, {t_u['ci95'][1]:+.4f}] | {t_u['p_two_sided']:.3f} | "
                     f"{t_f1['point']:+.4f} [{t_f1['ci95'][0]:+.4f}, {t_f1['ci95'][1]:+.4f}] | {t_f1['p_two_sided']:.3f} | "
                     f"{t_a['point']:+.4f} [{t_a['ci95'][0]:+.4f}, {t_a['ci95'][1]:+.4f}] | {t_a['p_two_sided']:.3f} |")

        c_u = conf["primary"]["delta"]["pauc_u40"]
        c_f1 = conf["primary"]["delta"]["macro_f1"]
        c_a = conf["primary"]["delta"]["pauc_all"]
        lines.append(f"| Total confirmation: composite − in1k control (primary) | {c_u['point']:+.4f} [{c_u['ci95'][0]:+.4f}, {c_u['ci95'][1]:+.4f}] | {c_u['p_two_sided']:.3f} | "
                     f"{c_f1['point']:+.4f} [{c_f1['ci95'][0]:+.4f}, {c_f1['ci95'][1]:+.4f}] | {c_f1['p_two_sided']:.3f} | "
                     f"{c_a['point']:+.4f} [{c_a['ci95'][0]:+.4f}, {c_a['ci95'][1]:+.4f}] | {c_a['p_two_sided']:.3f} |")

    (V5 / "review2_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--confirm", default=None)
    args = ap.parse_args(argv)
    path = Path(args.confirm) if args.confirm else max(V5.glob("confirm_s42*.json"), key=lambda p: len(p.stem))
    conf = json.loads(path.read_text(encoding="utf-8"))
    forest(conf)
    archives(conf)
    scr = screens()
    tables(scr, conf, path.name)
    print(f"wrote 3 figures (png+svg) to {FIG.relative_to(REPO_ROOT)} and results/v5/review2_tables.md from {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
