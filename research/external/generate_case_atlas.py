"""Workstream E7: the clinical safety-net case atlas (rebuilt in S12).

**Why it was rebuilt.** Panel A's entire story was an escalation mass clearing a
hardcoded threshold that does not exist. The frozen rule is
`argmax_c ( p_c + lambda_band * 1[c escalates] )` with lambda = 0.26 in the under-40 band,
and the constant the old caption quoted appears nowhere in `results/`. The
old script also selected its exemplars from `research/predictions/{arch}_test.csv`: the
plain 1-view matrix, on the test split, outside the pre-registered pass. A figure whose
caption states a decision that the deployed system does not make is worse than no figure,
because a reader cannot tell it apart from one that does.

Exemplars are now drawn from the OOF+TTA+Dirichlet panel and selected by *running the rule*
-- `argmax(p) is benign` and `apply_age_rule(p) is escalating` -- rather than by comparing a
score to a number. Two consequences are deliberate:

  * **No threshold is tuned to keep a panel.** `_first_or_none` returns None when no case
    satisfies the real criterion; the panel then says so on the figure and in the JSON.
    The previous version silently widened Panel A's filter (dropping the age constraint)
    when the strict query came back empty, which is the same failure in a smaller form.
  * **Panels are picked deterministically**, by a stated ordering (largest rule margin,
    largest top-2 overlap, ...), not by `.iloc[0]` on an arbitrary row order.

    $py -m research.external.generate_case_atlas

Outputs: `paper/figures/external_figure_case_atlas.png`,
`results/external/case_atlas_report.json`.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ml.paths import load_class_mapping, resolve
from research.external import frozen_params as fp

SESSION = "session_post_s11"
PAD_MANIFEST = "ml/data/manifest_pad.csv"
HAM_IMAGE_DIRS = ("data/ham10000/HAM10000_images_part_1", "data/ham10000/HAM10000_images_part_2")
PAD_IMAGE_DIRS = ("data/pad_ufes_20/imgs_part_1", "data/pad_ufes_20/imgs_part_2",
                  "data/pad_ufes_20/imgs_part_3")


def _ham_image(image_id: str):
    for directory in HAM_IMAGE_DIRS:
        path = resolve(directory) / f"{image_id}.jpg"
        if path.is_file():
            return path
    return None


def _pad_image(name: str):
    for directory in PAD_IMAGE_DIRS:
        path = resolve(directory) / str(name)
        if path.is_file():
            return path
    return None


def _first_or_none(frame: pd.DataFrame, by: str, ascending: bool = False):
    """Deterministic pick, or None. **Never** relaxes the query to manufacture a case."""
    if frame.empty:
        return None
    return frame.sort_values([by, "image_id"], ascending=[ascending, True]).iloc[0]


def build_frame() -> pd.DataFrame:
    """The OOF panel as a per-image frame, with both decisions already applied."""
    codes = list(load_class_mapping().codes)
    panel = fp.load_ham_oof_panel()
    esc = fp.escalating_indices()
    lam = fp.load_lambda_by_band()

    argmax_idx = panel.probs.argmax(axis=1)
    rule_idx = fp.apply_age_rule(panel.probs, bands=panel.bands)
    bonus = np.zeros(panel.probs.shape[1])
    bonus[esc] = 1.0
    lam_row = np.array([lam.get(str(b), lam["pooled"]) for b in panel.bands])
    adjusted = panel.probs + lam_row[:, None] * bonus

    frame = pd.DataFrame(panel.probs, columns=[f"p_{c}" for c in codes])
    frame["image_id"] = panel.image_ids
    frame["age"] = panel.ages
    frame["band"] = panel.bands
    frame["lambda_band"] = lam_row
    frame["true_code"] = [codes[i] for i in panel.y_true]
    frame["argmax_code"] = [codes[i] for i in argmax_idx]
    frame["rule_code"] = [codes[i] for i in rule_idx]
    frame["true_escalating"] = np.isin(panel.y_true, esc)
    frame["argmax_escalating"] = np.isin(argmax_idx, esc)
    frame["rule_escalating"] = np.isin(rule_idx, esc)
    frame["s_esc"] = fp.escalation_mass(panel.probs)
    # How far the bonus carried the case past the benign winner: the quantity the rule
    # actually decides on, and the honest replacement for the fabricated threshold test.
    frame["rule_margin"] = adjusted.max(axis=1) - adjusted[np.arange(len(frame)), argmax_idx]
    top2 = np.sort(panel.probs, axis=1)[:, -2:]
    frame["top2_gap"] = top2[:, 1] - top2[:, 0]
    return frame


def select_cases(frame: pd.DataFrame) -> list[dict]:
    """The four panels, each defined by running the deployed rule -- not by a threshold."""
    rescued = frame[frame["true_escalating"] & ~frame["argmax_escalating"]
                    & frame["rule_escalating"]]

    case_a = _first_or_none(rescued[(rescued["band"] == "<40") & (rescued["true_code"] == "mel")],
                            "rule_margin")
    fallback_a = None
    if case_a is None:
        # Report the shortfall rather than dropping the age constraint, which is what the
        # previous version did when the strict query came back empty.
        fallback_a = ("no under-40 melanoma in the OOF panel is missed by argmax and "
                      "recovered by the frozen lambda=0.26 rule")
        case_a = _first_or_none(rescued[rescued["band"] == "<40"], "rule_margin")
        if case_a is not None:
            fallback_a += f"; nearest true case is a {case_a['true_code']} in the same band"

    case_b = _first_or_none(frame[(frame["true_escalating"]) & (frame["top2_gap"] < 0.05)],
                            "top2_gap", ascending=True)
    case_d = _first_or_none(frame[(frame["true_code"] == "bkl")
                                  & (frame["argmax_code"] == "bkl")
                                  & (frame["s_esc"] < 0.05)], "s_esc", ascending=True)

    pad = pd.read_csv(resolve(PAD_MANIFEST))
    pad_row = pad.sort_values("image_id").iloc[0] if len(pad) else None

    return [
        {
            "panel": "A", "source": "ham",
            "title": r"A. Under-40 escalation recovered by the frozen $\lambda$-rule",
            "row": case_a, "note": fallback_a,
        },
        {
            "panel": "B", "source": "ham",
            "title": "B. Ambiguous lesion: top-2 classes within 0.05",
            "row": case_b, "note": None,
        },
        {
            "panel": "C", "source": "pad",
            "title": "C. Cross-modality smartphone capture (PAD-UFES-20)",
            "row": pad_row, "note": None,
        },
        {
            "panel": "D", "source": "ham",
            "title": "D. Benign keratosis, discharged with low escalation mass",
            "row": case_d, "note": None,
        },
    ]


def _caption(case: dict) -> str:
    row = case["row"]
    if row is None:
        return "No case in the panel satisfies this criterion."
    if case["source"] == "pad":
        return (f"Cohort: PAD-UFES-20  |  Fitzpatrick {row.get('fitzpatrick', 'n/a')}\n"
                f"Diagnosis: {str(row['class_code']).upper()}  |  clinical photograph, "
                f"not dermoscopy")
    lam = row["lambda_band"]
    return (f"True {row['true_code'].upper()}  |  argmax {row['argmax_code'].upper()}  |  "
            f"$\\lambda$-rule {row['rule_code'].upper()}\n"
            f"age {row['age']:.0f} (band {row['band']}), $\\lambda$={lam:.2f}, "
            f"rule margin {row['rule_margin']:+.3f}, $S_{{esc}}$={row['s_esc']:.2f}")


def _plot(cases: list[dict], codes: list[str], out_png) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, case in zip(axes.flatten(), cases):
        ax.set_title(case["title"], fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        row = case["row"]
        if row is None:
            ax.text(0.5, 0.5, "criterion not satisfied\nby any case in the panel",
                    ha="center", va="center", fontsize=11, style="italic", color="#b00020")
            continue

        colour = "#d62728" if case["panel"] == "A" else (
            "#ff7f0e" if case["panel"] == "B" else
            "#9467bd" if case["panel"] == "C" else "#2ca02c")
        path = (_ham_image(str(row["image_id"])) if case["source"] == "ham"
                else _pad_image(row["path"].split("/")[-1] if "path" in row else row["image_id"]))
        img_ax = ax.inset_axes([0.02, 0.14, 0.42, 0.76])
        if path is not None:
            img_ax.imshow(Image.open(path).convert("RGB"))
        else:
            img_ax.text(0.5, 0.5, "image file not found", ha="center", va="center", fontsize=9)
        img_ax.set_xticks([]); img_ax.set_yticks([])
        for spine in img_ax.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(2.5)

        bar_ax = ax.inset_axes([0.52, 0.20, 0.44, 0.64])
        if case["source"] == "ham":
            probs = [row[f"p_{c}"] for c in codes]
            colours = ["#d62728" if c in ("mel", "bcc", "akiec") else "#1f77b4" for c in codes]
            bar_ax.barh(np.arange(len(codes)), probs, color=colours, height=0.6)
            bar_ax.set_yticks(np.arange(len(codes)))
            bar_ax.set_yticklabels([c.upper() for c in codes], fontsize=8)
            bar_ax.set_xlim(0, 1.0)
            bar_ax.set_xlabel("Calibrated probability", fontsize=8)
            bar_ax.invert_yaxis()
            bar_ax.grid(axis="x", ls="--", alpha=0.5)
            bar_ax.axvline(row["lambda_band"], color="#d62728", ls="--", lw=1.2)
            bar_ax.text(row["lambda_band"], -0.7, rf"$\lambda$={row['lambda_band']:.2f}",
                        fontsize=7.5, color="#d62728", ha="center")
        else:
            bar_ax.axis("off")
            bar_ax.text(0.0, 0.75, "Modality: smartphone", fontsize=9, fontweight="bold")
            bar_ax.text(0.0, 0.55, f"Fitzpatrick: {row.get('fitzpatrick', 'n/a')}", fontsize=9)
            bar_ax.text(0.0, 0.35, "Out-of-distribution by Mahalanobis", fontsize=9,
                        color="#9467bd", fontweight="bold")
            bar_ax.text(0.0, 0.15, "AUROC 0.913 vs HAM val (S8b)", fontsize=8, color="#555555")

        text = _caption(case)
        if case["note"]:
            text += f"\n[{case['note']}]"
        ax.text(0.02, 0.005, text, fontsize=8.5, style="italic", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa",
                          edgecolor="#ced4da", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)

    print("=== E7: clinical safety-net case atlas (S12 rebuild) ===")
    codes = list(load_class_mapping().codes)
    frame = build_frame()
    cases = select_cases(frame)

    fig_dir = resolve("paper/figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = resolve("results/external"); res_dir.mkdir(parents=True, exist_ok=True)
    _plot(cases, codes, fig_dir / "external_figure_case_atlas.png")

    rescued = frame[frame["true_escalating"] & ~frame["argmax_escalating"] & frame["rule_escalating"]]
    report = {
        "session": SESSION,
        "workstream": "E7_case_atlas",
        "status": "post_hoc_secondary; declared in analysis_plan_post_s11_v2.json",
        "panel": f"{fp.OOF_PREDICTIONS_DIR}/{{arch}}_{fp.OOF_SPLIT}.csv",
        "calibration": fp.DIRICHLET_STATE,
        "lambda_by_band": fp.load_lambda_by_band(),
        "rule": "argmax_c ( p_c + lambda_band * 1[c escalates] )",
        "test_read": False,
        "selection": "deterministic; criteria evaluated by running the rule, no tuned threshold",
        "rescued_by_rule": {
            "total": int(len(rescued)),
            "by_band": rescued["band"].value_counts().to_dict(),
            "under40_melanoma": int(((rescued["band"] == "<40")
                                     & (rescued["true_code"] == "mel")).sum()),
        },
        "cases": [],
    }
    for case in cases:
        row = case["row"]
        entry = {"panel": case["panel"], "title": case["title"], "source": case["source"],
                 "note": case["note"], "found": row is not None}
        if row is not None:
            entry["image_id"] = str(row["image_id"])
            if case["source"] == "ham":
                entry.update({
                    "true_code": str(row["true_code"]),
                    "argmax_code": str(row["argmax_code"]),
                    "rule_code": str(row["rule_code"]),
                    "age": float(row["age"]), "band": str(row["band"]),
                    "lambda_band": float(row["lambda_band"]),
                    "rule_margin": float(row["rule_margin"]),
                    "s_esc": float(row["s_esc"]), "top2_gap": float(row["top2_gap"]),
                })
            else:
                entry.update({"class_code": str(row["class_code"]),
                              "fitzpatrick": str(row.get("fitzpatrick", "n/a"))})
        report["cases"].append(entry)
        print(f"  Panel {case['panel']}: "
              f"{'found ' + str(entry.get('image_id')) if entry['found'] else 'NOT FOUND'}"
              f"{' -- ' + case['note'] if case['note'] else ''}")

    (res_dir / "case_atlas_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRule rescues {report['rescued_by_rule']['total']} argmax misses "
          f"({report['rescued_by_rule']['under40_melanoma']} under-40 melanomas)")
    print(f"Wrote {fig_dir / 'external_figure_case_atlas.png'}")
    print(f"Wrote {res_dir / 'case_atlas_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
