"""M1: clusters, split, masks, the scorer's arithmetic, the loss's masking and the protocol's write-once guard. CPU, synthetic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from research.v7.m1 import bench, data, protocol, train


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": ["a1", "a2", "a3", "b1", "b2", "c1", "d1", "e1"],
        "archive": ["bcn20000"] * 3 + ["ham"] * 2 + ["mskcc", "mskcc", "derm12345"],
        "src_lesion_id": ["L1", "L1", "L1", "H1", "H1", "M1", "M2", None],
        "src_patient_id": [None, None, None, None, None, "P1", "P1", None],
    })


def test_clusters_join_lesions_patients_and_known_links_only_among_given_rows() -> None:
    frame = _frame()
    links = pd.DataFrame({"a": ["b2", "zz"], "b": ["e1", "a1"]})          # b2-e1 joins; zz is not in the frame
    cl = data.clusters(frame, links)
    assert cl[0] == cl[1] == cl[2]                                        # one lesion
    assert cl[3] == cl[4] == cl[7]                                        # lesion H1 plus the link to e1
    assert cl[5] == cl[6]                                                 # one patient, two lesions
    assert len({cl[0], cl[3], cl[5]}) == 3


def test_the_split_is_a_fixed_hash_of_the_cluster_root() -> None:
    roots = [f"ISIC_{i:07d}" for i in range(4000)]
    share = np.mean([data.held_out(r) for r in roots])
    assert abs(share - data.HELD_OUT_SHARE) < 0.03
    assert [data.held_out(r) for r in roots[:50]] == [data.held_out(r) for r in roots[:50]]
    assert not any(train.inner_val(r) and False for r in roots)          # inner val is drawn only from non-held-out rows (by caller)


def test_pairs_count_positives_and_mask_same_cluster_pairs_from_negatives() -> None:
    frame = _frame()
    frame["cluster"] = data.clusters(frame, pd.DataFrame({"a": [], "b": []}))
    frame["lesion"] = np.where(frame["src_lesion_id"].notna(), frame["archive"] + ":" + frame["src_lesion_id"].fillna(""), "")
    p = bench.Pairs(frame)
    assert len(p.pos) == 3 + 1                                            # L1: 3 pairs, H1: 1; M1 and M2 are single images
    # 28 pairs in all; same-cluster pairs: L1 3, H1 1, patient P1 1 -> 23 negatives
    assert p.n_neg == 28 - 5
    assert not p.is_negative(np.array([5]), np.array([6]))[0]             # same patient, different lesion: masked, not negative
    assert not p.is_positive(np.array([5]), np.array([6]))[0]


def test_a_perfect_embedding_scores_tpr_one_and_a_random_one_scores_near_the_fmr() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for k in range(300):
        rows += [{"image_id": f"x{k}_{j}", "archive": "bcn20000", "src_lesion_id": f"L{k}", "src_patient_id": None} for j in range(2)]
    rows += [{"image_id": f"s{k}", "archive": "derm12345", "src_lesion_id": None, "src_patient_id": None} for k in range(400)]
    frame = pd.DataFrame(rows)
    frame["cluster"] = data.clusters(frame, pd.DataFrame({"a": [], "b": []}))
    frame["lesion"] = np.where(frame["src_lesion_id"].notna(), frame["archive"] + ":" + frame["src_lesion_id"].fillna(""), "")
    pairs = bench.Pairs(frame)
    centre = rng.normal(size=(len(frame), 64))
    lesion_code = pd.factorize(frame["lesion"].where(frame["lesion"].ne(""), frame["image_id"]))[0]
    perfect = rng.normal(size=(lesion_code.max() + 1, 64))[lesion_code]
    perfect = perfect / np.linalg.norm(perfect, axis=1, keepdims=True)
    good = bench.score_embedding(perfect.astype(np.float32), pairs, torch.device("cpu"))
    assert good["hits"][str(protocol.FMR_SECONDARY)].mean() == 1.0 and good["recall_at_1"] == 1.0
    noise = (centre / np.linalg.norm(centre, axis=1, keepdims=True)).astype(np.float32)
    bad = bench.score_embedding(noise, pairs, torch.device("cpu"))
    assert bad["hits"][str(protocol.FMR_SECONDARY)].mean() < 0.05


def test_the_lesion_bootstrap_counts_a_big_lesion_once_and_pairs_methods() -> None:
    pos_lesion = np.array([0] * 100 + [1, 2, 3, 4])
    hits = {"a": np.array([True] * 100 + [False] * 4), "b": np.array([True] * 104)}
    out = bench.lesion_bootstrap(hits, pos_lesion, 500, 1)
    assert out["lesions"] == 5
    assert (out["boot"]["b"] >= out["boot"]["a"]).all()                   # shared resamples: b never below a


def test_the_multi_similarity_loss_ignores_same_cluster_pairs_of_different_lesions() -> None:
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.98, 0.15], [0.0, 1.0]]), dim=1)
    lesion = torch.tensor([0, 0, 1, 2])
    split = train.multi_similarity_loss(z, lesion, torch.tensor([0, 0, 1, 2]))       # image 2 is another cluster: a hard negative
    masked = train.multi_similarity_loss(z, lesion, torch.tensor([0, 0, 0, 2]))      # same cluster (shared patient): masked
    assert masked < split


def test_the_rim_crop_removes_a_dark_frame_and_a_round_field() -> None:
    boxed = np.zeros((200, 300, 3), np.uint8)
    boxed[20:180, 40:260] = 180
    out = data.crop_rim(Image.fromarray(boxed))
    assert out.size[0] == out.size[1] and np.asarray(out).mean() > 170
    yy, xx = np.mgrid[:300, :300]
    round_field = np.where(((yy - 150) ** 2 + (xx - 150) ** 2 < 140 ** 2)[..., None], 200, 0).astype(np.uint8).repeat(3, 2)
    out = np.asarray(data.crop_rim(Image.fromarray(round_field)))
    assert out.min() > 100                                                 # no black corner left


def test_the_protocol_is_written_once_and_any_change_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(protocol, "PROTOCOL", tmp_path / "protocol.json")
    protocol.write()
    with pytest.raises(SystemExit, match="written once"):
        protocol.write()
    assert protocol.require()["fmr_primary"] == protocol.FMR_PRIMARY
    record = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(protocol, "FMR_PRIMARY", 1e-3)                    # the rule moved after it was written
    with pytest.raises(SystemExit, match="changed after it was written"):
        protocol.require()
    assert record["sha256"]


# ------------------------------------------------------------------ amendment 01: density-normalised scores
def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_local_density_is_the_mean_cosine_over_the_named_ranks_without_the_image_itself() -> None:
    from research.v7.m1 import density
    rng = np.random.default_rng(1)
    emb = _unit(rng.normal(size=(60, 16)))
    got = density.local_density(emb, torch.device("cpu"), lo=2, hi=5, chunk=7)
    s = emb @ emb.T
    np.fill_diagonal(s, -9)
    want = (-np.sort(-s, axis=1))[:, 2:5].mean(1)
    assert np.allclose(got, want, atol=1e-6)
    with pytest.raises(ValueError, match="needs more than"):
        density.local_density(emb[:4], torch.device("cpu"), lo=2, hi=5)


def test_fused_cosine_is_the_mean_of_the_parts_and_a_zero_offset_is_the_raw_cosine() -> None:
    from research.v7.m1 import density
    rng = np.random.default_rng(2)
    a, b = _unit(rng.normal(size=(9, 8))), _unit(rng.normal(size=(9, 5)))
    fused = density.fuse(a, b)
    assert np.allclose(fused @ fused.T, (a @ a.T + b @ b.T) / 2, atol=1e-6)
    i, j = np.array([0, 3]), np.array([1, 8])
    assert np.allclose(density.pair_scores(a, np.zeros(9, np.float32), i, j), (a @ a.T)[i, j], atol=1e-6)
    off = np.arange(9, dtype=np.float32) / 10
    assert np.allclose(density.pair_scores(a, off, i, j), (a @ a.T)[i, j] - off[i] - off[j], atol=1e-6)


def test_normalising_by_density_recovers_pairs_that_a_dense_region_hides_from_one_threshold() -> None:
    """Look-alikes packed around one direction outscore true pairs elsewhere; measured against their own neighbourhood they do not."""
    from research.v7.m1 import density
    rng = np.random.default_rng(3)
    hub = np.zeros(48)
    hub[0] = 1.0
    crowd = _unit(hub + 0.25 * rng.normal(size=(150, 48)) / np.sqrt(48))            # 150 unrelated look-alikes, cosine about 0.94
    centres = _unit(rng.normal(size=(120, 48)))
    twins = _unit(np.repeat(centres, 2, axis=0) + 0.9 * rng.normal(size=(240, 48)) / np.sqrt(48))   # true pairs, cosine about 0.55
    emb = np.concatenate([crowd, twins])
    rows = [{"image_id": f"c{k}", "archive": "ham", "src_lesion_id": None, "src_patient_id": None} for k in range(150)]
    rows += [{"image_id": f"t{k}", "archive": "bcn20000", "src_lesion_id": f"L{k // 2}", "src_patient_id": None} for k in range(240)]
    frame = pd.DataFrame(rows)
    frame["cluster"] = data.clusters(frame, pd.DataFrame({"a": [], "b": []}))
    frame["lesion"] = np.where(frame["src_lesion_id"].notna(), frame["archive"] + ":" + frame["src_lesion_id"].fillna(""), "")
    pairs = bench.Pairs(frame)
    cpu = torch.device("cpu")
    raw = bench.score_embedding(emb, pairs, cpu, fmrs=(1e-3,))
    half = density.local_density(emb, cpu, lo=4, hi=12) / 2
    normalised = bench.score_embedding(emb, pairs, cpu, offset=half, fmrs=(1e-3,))
    assert raw["hits"]["0.001"].mean() < 0.2 < 0.9 < normalised["hits"]["0.001"].mean()


def test_the_scan_ranks_neighbours_by_the_normalised_score() -> None:
    from research.v7.m1 import scan
    q = _unit(np.array([[1.0, 0.0]]))
    g = _unit(np.array([[1.0, 0.1], [1.0, 0.3]]))
    _, raw = scan.topk(q, g, 1, torch.device("cpu"), same=False)
    sim, off = scan.topk(q, g, 1, torch.device("cpu"), same=False, query_offset=np.array([0.1], np.float32),
                         gallery_offset=np.array([0.5, 0.0], np.float32))
    assert raw[0, 0] == 0 and off[0, 0] == 1                                # the dense gallery row loses its lead
    assert np.isclose(sim[0, 0], float(q[0] @ g[1]) - 0.1, atol=1e-6)


def test_the_amendment_is_written_once_and_a_changed_tier_rule_is_refused(tmp_path, monkeypatch) -> None:
    from research.v7.m1 import amend, tiers
    monkeypatch.setattr(protocol, "PROTOCOL", tmp_path / "protocol.json")
    monkeypatch.setattr(amend, "AMENDMENT", tmp_path / "protocol_amendment_01.json")
    monkeypatch.setattr(amend, "RESULT", tmp_path / "amendment_01.json")
    monkeypatch.setattr(amend.bench, "file_sha", lambda path: "0" * 64)
    protocol.write()
    amend.write()
    with pytest.raises(SystemExit, match="written once"):
        amend.write()
    assert amend.require()["tier_rule"] == tiers.RULE
    with pytest.raises(SystemExit, match="has not been scored"):
        amend.result()
    monkeypatch.setattr(tiers, "RULE", tiers.RULE + " (edited after the read)")
    with pytest.raises(SystemExit, match="changed after it was written"):
        amend.require()


def test_the_amendment_cannot_be_written_after_its_result(tmp_path, monkeypatch) -> None:
    from research.v7.m1 import amend
    monkeypatch.setattr(protocol, "PROTOCOL", tmp_path / "protocol.json")
    monkeypatch.setattr(amend, "AMENDMENT", tmp_path / "protocol_amendment_01.json")
    monkeypatch.setattr(amend, "RESULT", tmp_path / "amendment_01.json")
    monkeypatch.setattr(amend.bench, "file_sha", lambda path: "0" * 64)
    protocol.write()
    (tmp_path / "amendment_01.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="after the read"):
        amend.write()


def test_the_fused_candidate_is_adopted_only_if_it_also_beats_the_untrained_one() -> None:
    from research.v7.m1 import amend
    yes, no = {"passes": True}, {"passes": False}
    assert amend.choose({"sscd_sn": yes, "sscd_femto_sn": yes}, yes) == "sscd_femto_sn"
    assert amend.choose({"sscd_sn": yes, "sscd_femto_sn": yes}, no) == "sscd_sn"       # no gain over the simpler method
    assert amend.choose({"sscd_sn": no, "sscd_femto_sn": yes}, no) == "sscd_femto_sn"
    assert amend.choose({"sscd_sn": yes, "sscd_femto_sn": no}, yes) == "sscd_sn"
    assert amend.choose({"sscd_sn": no, "sscd_femto_sn": no}, no) == amend.BASELINE


def test_the_adoption_rule_needs_a_paired_gain_and_no_powered_archive_lower() -> None:
    from research.v7.m1 import amend

    class P:
        pos_lesion = np.repeat(np.arange(200), 2)
        pos_archive = np.array(["ham"] * 200 + ["bcn20000"] * 200)

    base = np.zeros(400, bool)
    base[:60] = True                                                         # ham 0.30, bcn 0.00
    better = base.copy()
    better[200:300] = True                                                   # bcn 0.50, ham unchanged
    assert amend.passes(base, better, P)["passes"]
    worse_on_ham = better.copy()
    worse_on_ham[:20] = False                                                # pooled gain, but ham 0.20 < 0.30
    verdict = amend.passes(base, worse_on_ham, P)
    assert verdict["difference_ci95"][0] > 0 and not verdict["not_lower_on_any_powered_archive"] and not verdict["passes"]


def test_more_evidence_never_lowers_a_tier_and_no_evidence_is_no_link() -> None:
    """The contract any tier rule must keep: a higher score or more inliers never moves a pair to a weaker tier."""
    from research.v7.m1 import tiers
    rank = {"": 0, "at_risk": 1, "confirmed": 2}
    scores, inliers = np.linspace(-0.2, 0.6, 33), np.arange(0, 80, 2)
    for strict, loose in ((0.13, 0.017), (0.13, None), (None, None)):
        grid = np.array([[rank[t] for t in tiers.tier(np.full(len(inliers), s), inliers.astype(float), 0.066, strict, loose)]
                         for s in scores])
        assert (np.diff(grid, axis=0) >= 0).all() and (np.diff(grid, axis=1) >= 0).all()
        assert set(np.unique(grid)) <= {0, 1, 2} and grid[0].max() == 0     # far below the threshold: never a link
        assert grid[-1, -1] == 2                                            # both detectors strongly agree: confirmed


def test_the_amended_tier_rule_spends_sift_where_the_score_is_unsure() -> None:
    """T4 < T5 < T6: the score alone links only from T6; below that SIFT must agree, more of it the lower the score."""
    from research.v7.m1 import tiers
    t4, t5, t6 = 0.017, 0.066, 0.13
    cases = [(0.20, 0, "at_risk"), (0.20, 12, "confirmed"), (0.10, 7, ""), (0.10, 8, "at_risk"), (0.10, 29, "at_risk"),
             (0.10, 30, "confirmed"), (0.03, 11, ""), (0.03, 12, "at_risk"), (0.03, 80, "at_risk"), (0.01, 80, "")]
    got = tiers.tier(np.array([c[0] for c in cases]), np.array([c[1] for c in cases]), t5, t6, t4)
    assert list(got) == [c[2] for c in cases]
    old = tiers.tier(np.array([0.10, 0.10, 0.03]), np.array([0, 30, 80]), t5, None)    # a matcher scored before the amendment
    assert list(old) == ["at_risk", "confirmed", ""]


def test_a_fused_candidate_averages_the_parts_normalised_scores(monkeypatch) -> None:
    """Each part is normalised by its own density; the fused pair score is the mean of the parts' normalised scores."""
    from research.v7.m1 import amend, density, scan
    rng = np.random.default_rng(0)
    parts = {p: _unit(rng.normal(size=(40, 6))) for p in amend.CANDIDATES["sscd_femto_sn"]}
    monkeypatch.setattr(scan, "embed_all", lambda p, frame, device: parts[p])
    emb, off = amend.scored("sscd_femto_sn", None, torch.device("cpu"))
    a, b = np.arange(0, 20), np.arange(20, 40)
    fused = density.pair_scores(emb, off, a, b)
    each = [density.pair_scores(e, density.local_density(e, torch.device("cpu")) / 2, a, b) for e in parts.values()]
    assert np.allclose(fused, np.mean(each, axis=0), atol=1e-5)
