"""Check README.md against the repository. Exits 1 on any failure.

  python scripts/readme_media/audit_readme.py

1. Every decimal number in the README prose and tables must match a value in results/ (rounded to the printed precision).
2. Every key claim is matched exactly against a string formatted from results/ (facts.py).
3. Every relative link and image exists and is not git-ignored.
4. Every GIF is horizontal, at least 850 px wide, at least 20 fps and under 20 MB.
5. No AI attribution line.
6. The project is Scan4Disease (S4D), with no version number in the name, in the README and in the GIF scenes.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from facts import REPO, ci, load, sgn

README = REPO / "README.md"
ALLOW = {"3.12", "2.11", "12.8", "0.80"}  # Python, PyTorch, CUDA, the V3 ceiling and spec 0.80 as labels
# plan thresholds printed as labels, each checked against its source in the exact-claim list below
SOURCES = [
    "results/ablation_table.csv", "results/mcnemar_delong.json", "results/review2/v1_member_metrics.csv", "results/age_band_prior.csv",
    "research/selective/results/session4_report.md", "research/stats/results_oof/age_gap_intervals.csv",
    "results/reports/research__calibration__results__session2_report.md", "results/review2/ham10000_class_profile.csv",
    "results/v2/bootstrap_intervals.json", "results/v2/final_verdict.json", "results/v2/frr_by_group.csv", "results/v2/conformal_subgroup_safety.csv",
    "results/v3/final_verdict.json", "results/v3/oos_probe_report.json", "results/v3/s44_control_and_multiplicity.json", "results/v3/d2_strong_evaluation.json",
    "results/v4/power_audit.json", "results/v4/s54/s54_gate.json", "results/v4/s54/s54_contrasts.csv", "results/v4/s54/s54_marginals.csv", "results/v4/backbone_probe_deltas.csv",
    "results/v4/s59/s59_report.json", "results/v4/s65/contrasts_reserved.csv", "results/v4/s56/contrasts_reserved.csv", "results/v4/s56/frontier_reserved.csv", "results/v4/s64/ceiling.json",
    "results/v5/confirm_s42_s43_s44.json", "results/v5/q11_decomposition.json", "results/v5/review2_tables.md", "results/v5/same_footing.md", "results/v5/s84/s84_report.json", "results/v5/v5_oof_common_matrix.json",
]


def numbers_from(paths) -> list[float]:
    values: list[float] = []
    for rel in paths:
        text = (REPO / rel).read_text(encoding="utf-8")
        values += [float(m) for m in re.findall(r"(?<![\w.])-?\d+\.\d+(?:[eE]-?\d+)?", text)]
    return values


def prose(text: str) -> str:
    """README without code blocks, HTML tags, URLs and link targets."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\]\([^)]+\)", "]", text)
    text = re.sub(r"`[^`]*`", " ", text)
    return text


def claims(f) -> list[str]:
    r, b = f["rungs"], f["band"]
    g = f["v5_gates"]
    s = f["s84"]
    pr, sec, gl = s["primary"], s["key_secondary"], s["global"]
    d6 = s["d6_lesion_link_exclusion"]
    allb, u40 = s["descriptive"]["by_age_band"]["ALL"], s["descriptive"]["by_age_band"]["<40"]
    dep = sec["v5_deployed_u40"]
    p, mod, tr = f["v5_primary"], f["v5_module"], f["v5_trunk"]
    c = f["v2_c"]
    tta = f["v5_oof_f1"]
    out = [
        f"**{f['nv_share'] * 100:.1f}%** of HAM10000's {f['ham_images']:,} images", f"{f['ham_images']:,} images on {f['ham_lesions']:,} lesions",
        f"**{b['<40']['sens']:.3f}** ({b['<40']['caught']} of {b['<40']['n']} images, from only {f['u40_test_lesions']} lesions".replace("** (", "** ("),
        f"χ² {f['ens_chi2']:.2f}", f"({f['ens_only']} vs {f['cnn_only']} discordant cases)",
        f"{f['calib_raw']['test ECE']:.4f} → {f['calib_dirichlet']['test ECE']:.4f}", f"{r['A6']['f1']:.4f} → {r['A7']['f1']:.4f}", f"{r['A6']['missed']} → {r['A7']['missed']}",
        f"{b['<40']['referred_pct']:.1f}% of them", f"{f['maxvit_test']:.4f} on test but {f['maxvit_val']:.4f} on val",
        f"**{f['prior_u40'] * 100:.1f}%** | {f['prior_40'] * 100:.1f}% | {f['prior_60'] * 100:.1f}%",
        f"{b['40-59']['sens']:.3f} ({b['40-59']['caught']} / {b['40-59']['n']})", f"{b['60+']['sens']:.3f} ({b['60+']['caught']} / {b['60+']['n']})",
        f"{b['40-59']['referred_pct']:.1f}% ({b['40-59']['referred']} / {b['40-59']['missed']})", f"{b['60+']['referred_pct']:.1f}% ({b['60+']['referred']} / {b['60+']['missed']})",
        f"{f['oof_u40_sens']:.3f} ({f['oof_u40_caught']} / {f['oof_u40_n']})",
        f"{f['v2_cov'] * 100:.2f}%", f"{f['v2_frr'] * 100:.2f}%", f"{f['v2_one_minus'] * 100:.2f}% ceiling", f"{f['v2_frr_m_k']}/{f['v2_frr_n']} to {f['v2_frr_b_k']}/{f['v2_frr_n']}", f"Holm p = {f['v2_frr_holm']:.3f}",
        f"{sgn(c['ham_oof']['C'])} | {ci(c['ham_oof']['lo'], c['ham_oof']['hi'])}", f"{sgn(c['bcn20000']['C'])} | {ci(c['bcn20000']['lo'], c['bcn20000']['hi'])}",
        f"{f['v3_in']:.4f} to {f['v3_out']:.4f}", f"({f['v3_post']:.4f})", f"{f['v3_seed_a']}/{f['v3_seed_n']} and {f['v3_seed_b']}/{f['v3_seed_n']}",
        f"{f['v3_f1_before']:.4f} to {f['v3_f1_after']:.4f}", f"{f['v3_auc_before']:.4f} to {f['v3_auc_after']:.4f}",
        f"overall {sgn(f['v3_h2_all'])}", f"age-band AUC {f['v3_h3_auc']:.4f} [{f['v3_h3_ci'][0]:.4f}, {f['v3_h3_ci'][1]:.4f}]", f"{sgn(f['v3_h4'])} {ci(*f['v3_h4_ci'])}", f"under-40 AUC {sgn(f['v3_auc_delta'])}", f"Best condition {f['v3_best']:.4f}",
        f"{f['v4_cohort']['images']:,} images now holds {f['v4_cohort']['under40_escalating_lesions']} under-40", f"from {f['v4_v1_f1']:.3f} (V1) to {f['v4_ctrl_f1']:.3f}",
        f"under-40 AUC {f['v4_auc']['<40']:.3f} against {f['v4_auc']['40-59']:.3f} and {f['v4_auc']['60+']:.3f}", f"meets {f['v4_terms_met']} of {f['v4_terms']} terms",
        f"Macro-F1 {sgn(f['v4_a_f1'])} {ci(f['v4_a_lo'], f['v4_a_hi'])}", f"outcome {f['v4_outcome']}", f"Macro-F1 {sgn(f['v4_b_f1'])} {ci(f['v4_b_f1_lo'], f['v4_b_f1_hi'])}", f"under-40 pAUC {sgn(f['v4_b_p'])} {ci(f['v4_b_p_lo'], f['v4_b_p_hi'])}",
        f"{sgn(f['v4_pan'])} {ci(f['v4_pan_lo'], f['v4_pan_hi'])}", f"{f['v4_s59']['<40']:.4f} / {f['v4_s59']['40-59']:.4f} / {f['v4_s59']['60+']:.4f} vs a {f['v4_floor']} floor", f"referral {f['v4_referral']:.3f} vs a {f['v4_referral_cap']:.2f} cap",
        f"{sgn(f['v4_s56'], 3)} {ci(f['v4_s56_lo'], f['v4_s56_hi'], 3)}", f"{f['v4_ref_global']:.3f} → {f['v4_ref_band']:.3f}",
        f"{sgn(g['A']['point'])} {ci(*g['A']['ci95'])}, p = {p['pauc_u40']['p_two_sided']:.3f}", f"{f['v5_rows']:,} images, {f['v5_lesions']:,} lesions, {f['v5_u40_lesions']} under-40 escalating lesions, {f['v5_nboot']:,} hierarchical",
        f"{sgn(mod['pauc_u40']['point'])} {ci(*mod['pauc_u40']['ci95'])}, p = {mod['pauc_u40']['p_two_sided']:.3f}", f"{sgn(tr['pauc_u40']['point'])}, p = {tr['pauc_u40']['p_two_sided']:.3f}",
        f"{sgn(p['macro_f1']['point'])} | {ci(*p['macro_f1']['ci95'])} | {p['macro_f1']['p_two_sided']:.3f}", f"{sgn(g['D']['mean'])} | {ci(*p['pauc_all']['ci95'])} | {p['pauc_all']['p_two_sided']:.3f}",
        f"{sgn(g['C']['40-59']['point'])} (40–59), {sgn(g['C']['60+']['point'])} (60+)", f"{ci(*g['C']['40-59']['ci95'])}, {ci(*g['C']['60+']['ci95'])}", f"{g['C']['40-59']['p_two_sided']:.3f}, {g['C']['60+']['p_two_sided']:.3f}",
        f"{sgn(mod['macro_f1']['point'])} {ci(*mod['macro_f1']['ci95'])} | {mod['macro_f1']['p_two_sided']:.3f}", f"{sgn(tr['macro_f1']['point'])} {ci(*tr['macro_f1']['ci95'])} | {tr['macro_f1']['p_two_sided']:.3f}", f"{sgn(tr['pauc_u40']['point'])} {ci(*tr['pauc_u40']['ci95'])}",
        f"Macro-F1 {gl['macro_f1_V5']:.3f} vs V1's {gl['macro_f1_V1']:.3f}, Δ {sgn(gl['delta'], 3)} {ci(*gl['ci95_bootstrap'], 3)}", f"−{gl['margin']} margin",
        f"{pr['V5']['caught']}/{pr['V5']['n_escalating']} vs {pr['V1']['caught']}/{pr['V1']['n_escalating']} at a matched {pr['referral_all_eligible_matched']['V5']:.2f} referral, McNemar p = {pr['mcnemar_exact_one_sided_p']:.4f}",
        f"only {pr['n_escalating_u40_lesions']} of the planned {f['s75_u40']} lesions", f"sensitivity {dep['sensitivity']:.3f} clears the floor (lower bound {dep['ci95_clopper_pearson'][0]:.3f})", f"referral is {sec['v5_deployed_referral_all_eligible']:.3f} against a 0.25 cap",
        f"{allb['V5']['n_escalating'] / s['in_scope_lesions'] * 100:.1f}% escalating, against {f['v5_dev_prevalence'] * 100:.1f}% in the development pool", f"without the {f['s84_linked_total']} linked lesions",
        f"{f['s75_eligible']:,} eligible, {d6['excluded_lesions']} excluded, {d6['analysed_lesions']:,} analysed, {s['in_scope_lesions']:,} in scope",
        f"{pr['V5']['sensitivity']:.3f} ({pr['V5']['caught']} / {pr['V5']['n_escalating']}) | {pr['V1']['sensitivity']:.3f} ({pr['V1']['caught']} / {pr['V1']['n_escalating']})",
        f"{allb['V5']['auroc']:.3f} | {allb['V1']['auroc']:.3f}", f"{u40['V5']['pauc_0.20']:.3f} | {u40['V1']['pauc_0.20']:.3f}",
        f"{tta['composite_mean']['macro_f1_folds_1_4']:.4f} → {tta['composite_tta_mean']['macro_f1_folds_1_4']:.4f}",
        f"**{f['receipt_n']} executions**", f"{f['s84_receipt']['n']} execution and {f['s84_receipt']['labels']} label read",
        f"{r['A2']['f1']:.4f} → {r['A5']['f1']:.4f}",
    ]
    for key, label in (("A1", "ResNet-50"), ("A2", "**ConvNeXt-Tiny**"), ("A3", "SwinV2-Tiny"), ("A4", "Gated metadata fusion"), ("A5", "**6-CNN uniform soft vote**"), ("A6", "+ 24-view TTA")):
        x = r[key]
        out.append(f"[{x['lo']:.3f}, {x['hi']:.3f}]")
        out.append(f"{x['f1']:.4f}")
        out.append(f"{x['ba']:.4f}")
        out.append(f"{x['sens']:.4f}")
    out.append(f"**{r['A7']['f1']:.4f}** [{r['A7']['lo']:.3f}, {r['A7']['hi']:.3f}] | {r['A7']['ba']:.4f} | {r['A7']['sens']:.4f} | {r['A7']['missed']}")
    return out


def main() -> int:
    text = README.read_text(encoding="utf-8")
    fails: list[str] = []
    f = load()

    # 2. exact claims
    checked = claims(f)
    for claim in checked:
        if claim not in text:
            fails.append(f"claim not found verbatim: {claim}")

    # 1. every decimal in the prose
    known = numbers_from(SOURCES)
    body = prose(text)
    tokens = re.findall(r"(?<![\w.])\d+\.\d+(?![\w.]*\.\w)", body)
    unknown = []
    for tok in tokens:
        if tok in ALLOW:
            continue
        v, places = float(tok), len(tok.split(".")[1])
        tol = 0.5 * 10 ** (-places) + 1e-12
        if not any(abs(abs(k) - v) <= tol or abs(abs(k) * 100 - v) <= tol for k in known):
            unknown.append(tok)
    for tok in sorted(set(unknown)):
        fails.append(f"decimal with no source value: {tok}")

    # 3. links
    targets = re.findall(r'src="([^"]+)"', text) + re.findall(r"\]\(([^)]+)\)", text) + re.findall(r"`((?:results|research|docs|paper|ml|scripts)/[\w./\-]+?\.\w+)`", text)
    links = 0
    for target in targets:
        if target.startswith(("http", "#")):
            continue
        links += 1
        path = REPO / target
        if not path.exists():
            fails.append(f"missing file: {target}")
            continue
        ignored = subprocess.run(["git", "check-ignore", "-q", target], cwd=REPO).returncode == 0
        if ignored:
            fails.append(f"git-ignored, so broken on GitHub: {target}")

    # anchors
    anchors = set(re.findall(r'<a id="([\w-]+)"', text))
    for head in re.findall(r"^#{2,3} (.+)$", text, re.M):
        anchors.add(re.sub(r"[^\w\- ]", "", head.lower()).strip().replace(" ", "-"))
    for ref in re.findall(r'(?:href="|\]\()#([\w-]+)', text):
        if ref not in anchors:
            fails.append(f"broken anchor: #{ref}")

    # 4. GIFs
    gifs = re.findall(r'src="(docs/assets/readme/[\w.]+\.gif)"', text)
    total = 0
    for gif in gifs:
        path = REPO / gif
        if not path.exists():
            continue
        with Image.open(path) as im:
            w, h, n = im.width, im.height, getattr(im, "n_frames", 1)
            delays = []
            for i in range(n):
                im.seek(i)
                delays.append(im.info.get("duration", 0))
        size = path.stat().st_size / 1e6
        total += size
        if w < 850 or w <= h or n < 10 or size > 20 or min(delays) > 50 or min(delays) < 20:
            fails.append(f"{gif}: {w}x{h}, {n} frames, shortest delay {min(delays)} ms, {size:.1f} MB")

    # 5. attribution
    if re.search(r"Co-Authored-By|Generated with|Cl[a]ude|Anthr[o]pic|ChatGPT|AI-generated", text, re.I):
        fails.append("AI attribution found in README")

    # 6. project name
    wrong_name = re.compile(r"(scan\s*4\s*disease|s4d)[\s·:_-]*(v\s*)?\d(\.\d+)?\b", re.I)  # \b lets a citation key's year (s4d_2026) through
    for label, source in (("README", text), ("scenes.py", (Path(__file__).parent / "scenes.py").read_text(encoding="utf-8"))):
        if wrong_name.search(source):
            fails.append(f"{label}: the project name carries a version number: {wrong_name.search(source)[0]}")

    print(f"{len(checked)} exact claims, {len(tokens)} decimals, {links} file links, {len(gifs)} GIFs ({total:.1f} MB)")
    for line in fails:
        print("FAIL", line)
    print("README audit:", "FAILED" if fails else "all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
