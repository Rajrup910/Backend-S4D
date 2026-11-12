"""V7 decision instruments: the added-score test, the paired cluster bootstrap, the promotion rule and build B5's transform."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from research.v5 import chromophore as ch
from research.v6 import system_reads as sr
from research.v7 import darkness, promotion, stats


# ------------------------------------------------------------------ added-score test
def _cohort(signal: float, n_groups: int = 400, per_group: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    latent = rng.normal(size=n_groups)[groups]
    extra = rng.normal(size=n_groups)[groups]
    # one label per lesion, as in the data: every image of a lesion carries the same diagnosis
    y = (rng.random(n_groups)[groups] < 1 / (1 + np.exp(-(latent + signal * extra - 0.5)))).astype(int)
    ensemble = 1 / (1 + np.exp(-(latent - 0.5)))
    return y, ensemble, extra + 0.1 * rng.normal(size=len(groups)), groups


def test_a_score_carrying_new_signal_is_detected() -> None:
    result = stats.added_score_test(*_cohort(signal=1.0))
    assert result["adds_signal"] and result["beta_per_sd"] > 0 and result["wald_p_cluster"] < 1e-4


def test_a_score_with_no_new_signal_is_not_detected_at_the_nominal_rate() -> None:
    hits = sum(stats.added_score_test(*_cohort(signal=0.0, seed=s))["adds_signal"] for s in range(60))
    assert hits <= 8, f"{hits}/60 false positives at alpha 0.05"


def test_clustering_widens_the_interval_when_images_of_a_lesion_repeat() -> None:
    result = stats.added_score_test(*_cohort(signal=0.5, per_group=6))
    assert result["se_cluster_robust"] > result["se_naive"]


def test_the_added_score_test_refuses_degenerate_input() -> None:
    y, e, s, g = _cohort(signal=0.5)
    with pytest.raises(ValueError):
        stats.added_score_test(np.zeros_like(y), e, s, g)
    with pytest.raises(ValueError):
        stats.added_score_test(y, e, np.ones_like(s), g)


def test_the_paired_bootstrap_resamples_whole_groups_and_is_seeded() -> None:
    groups = np.repeat(np.arange(50), 4)
    values = np.repeat(np.arange(50, dtype=float), 4)
    a = stats.paired_cluster_bootstrap(lambda i: values[i].mean(), lambda i: 0.0, groups, n_boot=200)
    b = stats.paired_cluster_bootstrap(lambda i: values[i].mean(), lambda i: 0.0, groups, n_boot=200)
    assert a == b and a["n_clusters"] == 50 and a["ci95"][0] < 24.5 < a["ci95"][1]


# ------------------------------------------------------------------ promotion
def _rows(n_groups: int = 600, seed: int = 1) -> tuple[sr.Rows, np.ndarray]:
    rng = np.random.default_rng(seed)
    per = 2
    groups = np.repeat(np.arange(n_groups), per)
    y = np.repeat(rng.choice(7, n_groups, p=[0.06, 0.1, 0.12, 0.02, 0.15, 0.53, 0.02]), per)
    young = np.repeat(rng.random(n_groups) < 0.3, per)
    rows = sr.Rows(image_id=np.array([f"i{k}" for k in range(len(y))]), fold=np.repeat(rng.integers(1, 5, n_groups), per),
                   y=y, esc=np.isin(y, (0, 1, 4)), u40=young, group=groups, archive=np.repeat(rng.choice(["ham", "bcn20000"], n_groups), per))
    return rows, rng.random((len(y), 7))


def _probs(y: np.ndarray, noise: np.ndarray, strength: float) -> np.ndarray:
    p = noise.copy()
    p[np.arange(len(y)), y] += strength
    return p / p.sum(1, keepdims=True)


def test_a_clearly_better_candidate_passes_p1_and_p2_and_a_worse_one_fails() -> None:
    rows, noise = _rows()
    where = rows.where()
    reference, better, worse = _probs(rows.y, noise, 0.6), _probs(rows.y, noise, 1.5), _probs(rows.y, noise, 0.2)
    good = promotion.p1_p2(better, reference, rows, where, bar=0.008, n_boot=200)
    bad = promotion.p1_p2(worse, reference, rows, where, bar=0.008, n_boot=200)
    assert good["p1"]["passed"] and good["p2"]["passed"]
    assert not bad["p1"]["passed"] and not bad["p2"]["passed"]


def test_xauc_keeps_young_positives_and_older_negatives_only_and_never_decides() -> None:
    rows, noise = _rows()
    where = rows.where()
    reference = _probs(rows.y, noise, 0.6)
    young_boost = _probs(rows.y, noise, 0.6)
    young_boost[rows.u40 & rows.esc] = _probs(rows.y, noise, 1.5)[rows.u40 & rows.esc]  # only young positives improve
    out = promotion.xauc_u40(young_boost, reference, rows, where, n_boot=200)
    kept = where[(rows.esc[where] & rows.u40[where]) | (~rows.esc[where] & ~rows.u40[where])]
    assert out["rows"] == len(kept) and out["positives"] == int((rows.esc & rows.u40)[kept].sum())
    assert out["xauc_young_pos_older_neg"]["delta"] > 0 and out["decisive"] is False


def test_under40_gain_claims_use_the_wider_interval_and_p2_keeps_95() -> None:
    rows, noise = _rows()
    where = rows.where()
    reference, better = _probs(rows.y, noise, 0.6), _probs(rows.y, noise, 1.5)
    out = promotion.p1_p2(better, reference, rows, where, bar=0.008, n_boot=300)
    gate = out["secondary_gate_a2"]
    assert gate["ci_gain"][0] <= gate["ci95"][0] and gate["ci_gain"][1] >= gate["ci95"][1]  # wider on both sides
    assert gate["passed"] == bool(gate["point"] >= promotion.MCID_SECONDARY and gate["ci_gain"][0] > 0)
    assert out["read"]["pauc_u40"]["ci95"] == gate["ci95"]  # P2 still reads the 95% interval


def test_p3_is_non_inferiority_with_a_0_02_margin() -> None:
    rng = np.random.default_rng(3)
    groups = np.arange(800)
    y = rng.choice(3, 800)
    esc = (y == 2).astype(int)
    base = np.full((800, 7), 0.01)
    base[np.arange(800), np.array([5, 2, 4])[y]] = 1.0
    candidate = base.copy()
    same = promotion.p3_holdout(np.array([5, 2, 4])[y], esc, groups, candidate, base, n_boot=200)
    assert same["passed"]
    broken = base.copy()
    broken[:, 4] = 0.0  # melanoma never called: escalation ranking and Macro-F1 collapse
    assert not promotion.p3_holdout(np.array([5, 2, 4])[y], esc, groups, broken, base, n_boot=200)["passed"]


def test_s1_counts_only_classes_with_enough_lesion_groups() -> None:
    # class 4: 60 groups (counted), class 6: 50 groups (counted, boundary), class 3: 49 groups (reported only)
    y = np.r_[np.full(120, 4), np.full(50, 6), np.full(49, 3)]
    groups = np.r_[np.repeat(np.arange(60), 2), np.arange(100, 150), np.arange(200, 249)]
    counted, present = promotion.classes_counted(y, groups, 50)
    assert counted == [4, 6]
    assert present == {3: 49, 4: 60, 6: 50}
    assert promotion.classes_counted(y, groups, 0)[0] == [3, 4, 6]  # S1 off: every present class counts


def test_the_verdict_requires_every_row_set_and_a_cross_site_read() -> None:
    ok = {"p1": {"passed": True}, "p2": {"passed": True}}
    full = {s: ok for s in promotion.ROW_SETS}
    sites = {a: {"passed": True} for a in promotion.P3_HOLDOUTS}
    assert promotion.verdict(full, sites)["promoted"] and promotion.verdict(full, sites)["status"] == "PROMOTED"
    assert not promotion.verdict(full, None)["promoted"]
    assert not promotion.verdict({"all_rows": ok}, sites)["promoted"]
    failing = {**full, "without_426_rows": {"p1": {"passed": False}, "p2": {"passed": True}}}
    assert promotion.verdict(failing, sites)["decision"] == "V6 stays"


def test_the_verdict_fails_closed_on_an_empty_partial_or_undeclared_read() -> None:
    ok = {"p1": {"passed": True}, "p2": {"passed": True}}
    full = {s: ok for s in promotion.ROW_SETS}
    sites = {a: {"passed": True} for a in promotion.P3_HOLDOUTS}
    empty = promotion.verdict(full, {})                                   # all([]) is True: an empty read must not promote
    assert not empty["promoted"] and empty["status"] == "NOT_ESTIMABLE"
    one = promotion.verdict(full, {"bcn20000": {"passed": True}})         # one passing archive is not three
    assert not one["promoted"] and "P3 not read on ['ham', 'mskcc']" in one["reasons"]
    assert not promotion.verdict({}, sites)["promoted"]                  # no in-domain read at all
    assert not promotion.verdict({**full, "a_new_row_set": ok}, sites)["promoted"]
    assert not promotion.verdict(full, {**sites, "ham": {"passed": False}})["promoted"]
    assert promotion.verdict(full, {"bcn20000": {"passed": True}}, holdouts=("bcn20000",))["promoted"]   # decision 16 (c), if chosen before signing


def test_p3_fails_closed_when_a_decisive_read_is_not_estimable() -> None:
    rng = np.random.default_rng(4)
    groups = np.arange(600)
    y = np.array([5, 2, 4])[rng.choice(3, 600)]
    esc = (y == 4).astype(int)
    probs = np.full((600, 7), 0.01)
    probs[np.arange(600), y] = 1.0
    assert promotion.p3_holdout(y, esc, groups, probs, probs, n_boot=100)["passed"]
    # no class reaches the support floor: Macro-F1 cannot be read, so the archive fails instead of passing on pAUC alone
    starved = promotion.p3_holdout(y, esc, groups, probs, probs, n_boot=100, min_groups=10**6)
    assert not starved["passed"] and starved["not_estimable"] == ["macro_f1_present_classes"]
    # one class only: escalation pAUC cannot be read
    nv = np.full(600, 5)
    single = promotion.p3_holdout(nv, np.zeros(600, int), groups, probs, probs, n_boot=100)
    assert not single["passed"] and "escalation_pauc" in single["not_estimable"]


def test_p2_vetoes_a_candidate_that_lowers_every_young_score() -> None:
    """The external review's case (13 Nov 2026): shrink the escalation mass of every under-40 row by one monotone map. Ranks
    inside the band do not change, so the under-40 pAUC delta is exactly 0 and the old two-endpoint veto cannot see it; the
    young cancers now rank below older benign lesions, so the xAUC falls and P2 must fail on it."""
    rows, noise = _rows()
    where = rows.where()
    reference = _probs(rows.y, noise, 1.0)
    shifted = reference.copy()
    young = rows.u40
    esc_cols = list(promotion.ESC)
    shifted[np.ix_(young, esc_cols)] *= 0.05
    shifted[young] /= shifted[young].sum(1, keepdims=True)
    out = promotion.p1_p2(shifted, reference, rows, where, bar=0.008, n_boot=200)
    assert abs(out["read"]["pauc_u40"]["delta"]) < 1e-9          # invisible to the within-band read
    assert out["read"]["xauc_u40"]["ci95"][1] < 0                # visible to the cross-band read
    assert not out["p2"]["xauc_u40_ok"] and not out["p2"]["passed"]


def test_p2_fails_closed_without_the_xauc_read() -> None:
    with pytest.raises(SystemExit, match="xauc_u40"):
        promotion.p2_verdict({"pauc_u40": {"ci95": [0.0, 0.1]}, "esc_sens": {"ci95": [0.0, 0.1]}}, margins={})


def test_p2_is_the_veto_by_default_and_margins_only_tighten_it() -> None:
    read = {"pauc_u40": {"ci95": [-0.06, 0.01]}, "esc_sens": {"ci95": [-0.015, 0.005]}, "xauc_u40": {"ci95": [-0.03, 0.02]}}
    assert not promotion.p2_verdict({**read, "xauc_u40": {"ci95": [-0.05, -0.01]}}, margins={})["passed"]  # xAUC harm vetoes
    assert promotion.P2_NONINFERIORITY_MARGINS is None or set(promotion.P2_NONINFERIORITY_MARGINS) == set(promotion.P2_ENDPOINTS)
    veto = promotion.p2_verdict(read, margins={})
    assert veto["passed"] and veto["non_inferiority"] is None and "not a non-inferiority claim" in veto["form"]
    tight = promotion.p2_verdict(read, margins={"pauc_u40": 0.05, "esc_sens": 0.02})
    assert not tight["passed"] and not tight["non_inferiority"]["pauc_u40"]["ok"] and tight["non_inferiority"]["esc_sens"]["ok"]
    assert promotion.p2_verdict(read, margins={"pauc_u40": 0.08, "esc_sens": 0.02})["passed"]
    harmed = {"pauc_u40": {"ci95": [-0.09, -0.01]}, "esc_sens": {"ci95": [-0.01, 0.01]}, "xauc_u40": {"ci95": [-0.01, 0.01]}}
    assert not promotion.p2_verdict(harmed, margins={"pauc_u40": 0.5, "esc_sens": 0.5})["passed"]      # a margin never rescues a veto failure
    with pytest.raises(SystemExit, match="margins"):
        promotion.p2_verdict(read, margins={"pauc_u40": 0.05})


def test_holm_adjusts_within_the_declared_family_and_counts_members_without_a_p_value() -> None:
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adjusted["a"] == pytest.approx(0.04) and adjusted["c"] == pytest.approx(0.09) and adjusted["b"] == pytest.approx(0.09)
    assert np.isnan(adjusted["d"])
    assert stats.holm({"only": 0.2}) == {"only": 0.2}
    assert promotion.boot_p(np.array([0.1, 0.2, 0.3, -0.1])) == pytest.approx(0.8) and np.isnan(promotion.boot_p(np.array([np.nan])))


# ------------------------------------------------------------------ build B5
def _basis() -> ch.ChromophoreBasis:
    def plane(v):
        v = np.asarray(v, float) - np.mean(v)
        return (v / np.linalg.norm(v)).tolist()
    return ch.ChromophoreBasis(v_mel=plane(ch.REF_MEL), v_hb=plane(ch.REF_HB), t_mel=0.3)


def _image() -> torch.Tensor:
    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w]
    lesion = np.hypot((yy - h / 2) / 25, (xx - w / 2) / 32) < 1
    a = np.empty((h, w, 3), np.float32)
    a[:] = (0.85, 0.65, 0.55)
    a[lesion] = (0.45, 0.28, 0.20)
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0), lesion


def _lesion_excess_melanin(srgb: torch.Tensor, lesion: np.ndarray) -> float:
    """The dark-lesion read's measure (`b1_hard_core.mean_melanin`): melanin above skin, inside the lesion."""
    maps = ch.compute_maps(srgb, _basis())
    excess = (maps.c_mel - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)[0, 0].numpy()
    return float(excess[lesion].mean())


def test_lesion_melanin_scaling_moves_the_measured_melanin_by_k_and_leaves_skin_alone() -> None:
    srgb, lesion = _image()
    base = _lesion_excess_melanin(srgb, lesion)
    assert base > 0.05
    for k in (1.8, 0.55):
        out = darkness.scale_lesion_melanin(srgb, _basis(), k)
        ratio = _lesion_excess_melanin(out, lesion) / base
        assert abs(ratio - k) < 0.15 * k, (k, ratio)
        corner = (slice(0, 10), slice(0, 10))
        assert torch.abs(out[0, :, corner[0], corner[1]] - srgb[0, :, corner[0], corner[1]]).max() < 0.02


def test_the_transform_returns_an_image_of_the_same_size_and_k_one_is_identity() -> None:
    srgb, _ = _image()
    same = darkness.scale_lesion_melanin(srgb, _basis(), 1.0)
    assert torch.allclose(same, srgb, atol=0.01)
    pil = Image.fromarray((srgb[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    out = darkness.LesionMelaninScale(_basis(), p=1.0, seed=0)(pil)
    assert out.size == pil.size and out.mode == "RGB"


def test_a_missing_basis_refuses_rather_than_silently_skipping(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(darkness, "BASIS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        darkness.load_basis(fold=3)


# ------------------------------------------------------------------ sealed external read
def _cohort_files(tmp_path, n: int = 120):
    import pandas as pd
    rng = np.random.default_rng(5)
    classes = list(external.CLASSES)
    labels = rng.choice(classes, n, p=[0.05, 0.1, 0.1, 0.05, 0.2, 0.45, 0.05])
    cohort = pd.DataFrame({"image_id": [f"EXT_{i:05d}" for i in range(n)], "group_id": [f"L{i // 2}" for i in range(n)],
                           "age_band": rng.choice(["<40", "40-59", "60+"], n)})
    probs = rng.random((n, 7))
    probs[np.arange(n), [classes.index(c) for c in labels]] += 1.5
    probs /= probs.sum(1, keepdims=True)
    preds = cohort[["image_id"]].assign(**{f"p_{c}": probs[:, i] for i, c in enumerate(classes)})
    paths = {k: tmp_path / f"{k}.csv" for k in ("cohort", "labels", "preds")}
    cohort.to_csv(paths["cohort"], index=False)
    pd.DataFrame({"image_id": cohort["image_id"], "label": labels}).to_csv(paths["labels"], index=False)
    preds.to_csv(paths["preds"], index=False)
    return paths


from research.v7 import external_read as external  # noqa: E402


def _audit(name: str, paths, copies=()) -> None:
    import json
    seal = json.loads((external.ROOT / name / "seal.json").read_text(encoding="utf-8"))
    (external.ROOT / name / "pixel_audit.json").write_text(json.dumps(
        {"cohort_sha256": external.sha(paths["cohort"]), "images": seal["rows"], "copies": list(copies), "labels_read": 0}), encoding="utf-8")


def test_the_external_read_runs_in_order_once_and_refuses_a_second_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(external, "ROOT", tmp_path / "ext")
    paths = _cohort_files(tmp_path)
    with pytest.raises(SystemExit, match="seal the cohort first"):
        external.predict("c1", paths["preds"], gate0=lambda: "sig")
    external.seal("c1", paths["cohort"], paths["labels"], master=tmp_path / "absent.csv")
    with pytest.raises(SystemExit, match="no pixel audit"):
        external.predict("c1", paths["preds"], gate0=lambda: "sig")
    _audit("c1", paths)
    external.predict("c1", paths["preds"], gate0=lambda: "sig")
    with pytest.raises(SystemExit, match="owner-approved"):
        external.read("c1", owner_approved=False)
    receipt = external.read("c1", owner_approved=True, n_boot=100)
    assert receipt["label_read_count"] == 1 and receipt["result"]["all_ages"]["macro_f1_present_classes"] > 0.5
    with pytest.raises(SystemExit, match="read once"):
        external.read("c1", owner_approved=True)


def test_the_external_read_refuses_changed_labels_and_cohorts_already_in_the_master(tmp_path, monkeypatch) -> None:
    import pandas as pd
    monkeypatch.setattr(external, "ROOT", tmp_path / "ext")
    paths = _cohort_files(tmp_path)
    master = tmp_path / "master.csv"
    pd.DataFrame({"archive_id": ["EXT_00003"], "sha256": ["x"]}).to_csv(master, index=False)
    with pytest.raises(SystemExit, match="already in the V7 master"):
        external.seal("c2", paths["cohort"], paths["labels"], master=master)
    external.seal("c3", paths["cohort"], paths["labels"], master=tmp_path / "absent.csv")
    _audit("c3", paths)
    external.predict("c3", paths["preds"], gate0=lambda: "sig")
    paths["labels"].write_text(paths["labels"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="label file changed"):
        external.read("c3", owner_approved=True)


def test_sealing_ignores_the_cohorts_own_rows_in_the_master_but_not_other_sources(tmp_path, monkeypatch) -> None:
    import pandas as pd
    monkeypatch.setattr(external, "ROOT", tmp_path / "ext")
    paths = _cohort_files(tmp_path)
    master = tmp_path / "master.csv"
    pd.DataFrame({"archive_id": ["EXT_00001", "ISIC_1"], "sha256": ["a", "b"], "frozen_split": ["midas", "train"]}).to_csv(master, index=False)
    assert external.seal("own", paths["cohort"], paths["labels"], master=master, own_split="midas")["rows"] == 120
    pd.DataFrame({"archive_id": ["EXT_00001"], "sha256": ["a"], "frozen_split": ["train"]}).to_csv(master, index=False)
    with pytest.raises(SystemExit, match="already in the V7 master"):
        external.seal("other", paths["cohort"], paths["labels"], master=master, own_split="midas")


def test_the_paired_external_read_freezes_both_systems_and_reports_v7_minus_v6(tmp_path, monkeypatch) -> None:
    import pandas as pd
    monkeypatch.setattr(external, "ROOT", tmp_path / "ext")
    paths = _cohort_files(tmp_path)
    worse = pd.read_csv(paths["preds"], dtype={"image_id": str})
    probs = worse[external.PROBS].to_numpy()
    worse[external.PROBS] = 0.5 * probs + 0.5 / 7  # same ranking, flatter: a weaker V6 stand-in
    worse[external.PROBS] = worse[external.PROBS].to_numpy()[::-1]  # and wrong rows, so it is worse
    worse.to_csv(tmp_path / "v6.csv", index=False)
    external.seal("pair", paths["cohort"], paths["labels"], master=tmp_path / "absent.csv")
    _audit("pair", paths, copies=["EXT_00000", "EXT_00001"])
    with pytest.raises(SystemExit, match="same file"):
        external.predict("pair", paths["preds"], baseline=paths["preds"], gate0=lambda: "sig")
    assert not (external.ROOT / "pair" / "predictions.json").exists()
    record = external.predict("pair", paths["preds"], baseline=tmp_path / "v6.csv", gate0=lambda: "sig")
    assert record["form"] == "paired" and record["baseline_sha256"] != record["predictions_sha256"]
    result = external.read("pair", owner_approved=True, n_boot=100)["result"]
    assert result["form"] == "paired" and result["pixel_copies_excluded"] == 2 and result["all_ages"]["images"] == 118
    delta = result["all_ages"]["v7_minus_v6"]["macro_f1_present_classes"]
    assert delta["delta"] > 0 and delta["ci95"][0] <= delta["delta"] <= delta["ci95"][1]
    lo, hi = result["all_ages"]["escalation_sensitivity_ci95_exact"]
    assert lo <= result["all_ages"]["escalation_sensitivity"] <= hi
    assert "v7_minus_v6" in result["under40"] and result["under40"]["images"] < 118


def test_the_external_read_refuses_a_changed_cohort_copy_baseline_or_pixel_audit(tmp_path, monkeypatch) -> None:
    import pandas as pd
    monkeypatch.setattr(external, "ROOT", tmp_path / "ext")
    paths = _cohort_files(tmp_path)
    shuffled = pd.read_csv(paths["preds"], dtype={"image_id": str}).sample(frac=1, random_state=1)
    shuffled.to_csv(tmp_path / "v6.csv", index=False)
    for name, target, match in (("k1", "cohort.csv", "cohort changed"), ("k2", "baseline.csv", "baseline predictions changed"),
                                ("k3", "pixel_audit.json", "pixel audit changed")):
        external.seal(name, paths["cohort"], paths["labels"], master=tmp_path / "absent.csv")
        _audit(name, paths)
        external.predict(name, paths["preds"], baseline=tmp_path / "v6.csv", gate0=lambda: "sig")
        f = external.ROOT / name / target
        f.write_text(f.read_text(encoding="utf-8").replace("}", ', "x": 1}') if target.endswith(".json") else f.read_text(encoding="utf-8") + "\n",
                     encoding="utf-8")
        with pytest.raises(SystemExit, match=match):
            external.read(name, owner_approved=True, n_boot=50)
        assert not (external.ROOT / name / "receipt.json").exists()


def test_the_pixel_audit_must_name_this_cohort_and_cover_every_image(tmp_path, monkeypatch) -> None:
    import json
    monkeypatch.setattr(external, "ROOT", tmp_path / "ext")
    paths = _cohort_files(tmp_path)
    external.seal("a", paths["cohort"], paths["labels"], master=tmp_path / "absent.csv")
    audit = external.ROOT / "a" / "pixel_audit.json"
    for bad, match in (({"cohort_sha256": "other", "images": 120, "copies": [], "labels_read": 0}, "different cohort"),
                       ({"cohort_sha256": external.sha(paths["cohort"]), "images": 119, "copies": [], "labels_read": 0}, "every sealed"),
                       ({"cohort_sha256": external.sha(paths["cohort"]), "images": 120, "copies": [], "labels_read": 3}, "every sealed")):
        audit.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit, match=match):
            external.predict("a", paths["preds"], gate0=lambda: "sig")


def test_u40_lesion_sensitivity_is_lesion_level_matched_specificity_and_never_decides() -> None:
    rows, noise = _rows()
    where = rows.where()
    reference = _probs(rows.y, noise, 0.6)
    same = promotion.u40_lesion_sensitivity(reference, reference, rows, where, n_boot=100)
    assert same["delta"] == 0 and same["decisive"] is False and same["specificity"] == 0.8
    assert same["lesion_groups"] == len(np.unique(rows.group[where][rows.u40[where]]))   # lesions, not images
    better = reference.copy()
    young_pos = rows.u40 & rows.esc
    better[young_pos] = _probs(rows.y, noise, 1.5)[young_pos]
    up = promotion.u40_lesion_sensitivity(better, reference, rows, where, n_boot=100)
    assert up["delta"] > 0 and up["ci95"][0] <= up["delta"] <= up["ci95"][1]
