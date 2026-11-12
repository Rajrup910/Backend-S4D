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
