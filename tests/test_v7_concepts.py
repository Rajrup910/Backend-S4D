"""Build B2 (concept teacher, image-only student): targets travel with the image, the trunk never sees them, the criterion
adds the concept term only for packed batches, and the teacher's rows obey the data gate."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from research.v6 import train_v6 as v6
from research.v7 import builds, concept_student as cs, train_v7


class _FakeArm(nn.Module):
    """Stands in for V5's ArmNet: returns a dict with `logits` and the pooled feature `z`."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Conv2d(3, 768, 1)
        self.head = nn.Linear(768, 7)
        self.seen_channels: int | None = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_channels = x.shape[1]
        z = self.base(x).mean((2, 3))
        return {"logits": self.head(z), "z": z}


def _packed_batch(n: int = 4) -> tuple[torch.Tensor, np.ndarray]:
    targets = np.random.default_rng(0).random((n, cs.N_CONCEPTS)).astype(np.float32)
    x = torch.stack([cs.pack(torch.rand(3, 16, 16), t) for t in targets])
    return x, targets


def test_pack_puts_the_flag_and_targets_in_the_fourth_plane_only() -> None:
    image = torch.rand(3, 16, 16)
    packed = cs.pack(image, np.arange(1, 6, dtype=np.float32) / 10)
    assert packed.shape == (4, 16, 16)
    assert torch.equal(packed[:3], image)
    assert packed[3, 0, 0] == 1 and torch.allclose(packed[3, 0, 1:6], torch.arange(1, 6) / 10)
    assert packed[3].sum() == pytest.approx(1 + 1.5)


def test_the_trunk_never_sees_the_target_plane_and_the_state_holds_this_batch() -> None:
    model = cs.attach_concept_head(_FakeArm())
    x, targets = _packed_batch()
    out = model(x)
    assert model.seen_channels == 3
    assert out["logits"].shape == (4, 7)
    assert cs.STATE.logits.shape == (4, cs.N_CONCEPTS)
    assert np.allclose(cs.STATE.targets.numpy(), targets)
    cs.STATE.logits = cs.STATE.targets = None


def test_evaluation_inputs_pass_through_without_touching_the_state() -> None:
    model = cs.attach_concept_head(_FakeArm())
    cs.STATE.logits = cs.STATE.targets = None
    model(torch.rand(2, 3, 16, 16))
    assert cs.STATE.logits is None


def test_the_criterion_adds_the_weighted_concept_bce_and_clears_the_state() -> None:
    model = cs.attach_concept_head(_FakeArm())
    x, _ = _packed_batch()
    labels = torch.tensor([0, 1, 4, 5])
    logits = model(x)["logits"]
    expected_concept = nn.functional.binary_cross_entropy_with_logits(cs.STATE.logits, cs.STATE.targets).item()
    loss = cs.CrossEntropyPlusConcepts(nn.CrossEntropyLoss())(logits, labels)
    expected = nn.CrossEntropyLoss()(logits, labels).item() + cs.CONCEPT_WEIGHT * expected_concept
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert cs.STATE.logits is None
    loss.backward()
    assert model.concept_head.weight.grad is not None


def test_the_criterion_refuses_a_batch_without_concept_logits() -> None:
    cs.STATE.logits = cs.STATE.targets = None
    with pytest.raises(AssertionError, match="no concept logits"):
        cs.CrossEntropyPlusConcepts(nn.CrossEntropyLoss())(torch.rand(2, 7), torch.tensor([0, 1]))


def test_the_dataset_refuses_rows_without_a_target(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import torchvision.transforms as T
    from PIL import Image

    image = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16), (120, 80, 60)).save(image)
    monkeypatch.setattr(cs, "load_fold_targets", lambda fold: {"a": np.zeros(cs.N_CONCEPTS, np.float32)})
    dataset = cs.concept_dataset(v6.PoolDataset, 0)
    ok = dataset(pd.DataFrame({"image_id": ["a"], "image_path": [str(image)], "y": [3]}), T.ToTensor(), "y")
    x, y = ok[0]
    assert x.shape[0] == 4 and y == 3
    with pytest.raises(SystemExit, match="no concept target"):
        dataset(pd.DataFrame({"image_id": ["a", "b"], "image_path": [str(image)] * 2, "y": [3, 1]}), T.ToTensor(), "y")
    with pytest.raises(SystemExit, match="different class"):
        cs.concept_dataset(torch.utils.data.Dataset, 0)


def test_the_dataset_can_be_sent_to_a_windows_loader_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows workers receive the dataset by pickle. The first two B2 GPU smokes failed here: a class defined inside a
    function, then a V6 PoolDataset instance pickled while the trainer had `v6.PoolDataset` patched (reproduced below)."""
    import pickle

    monkeypatch.setattr(cs, "load_fold_targets", lambda fold: {"a": np.ones(cs.N_CONCEPTS, np.float32)})
    frame = pd.DataFrame({"image_id": ["a"], "image_path": ["unused.jpg"], "y": [3]})
    with train_v7.configure(builds.BUILDS["B2"], _args(), RECEIPT):   # the seam is patched, as in a real run
        dataset = v6.PoolDataset(frame, None, "y")
        copy = pickle.loads(pickle.dumps(dataset))
    assert len(copy) == 1 and copy.labels == [3] and np.array_equal(copy.concept_targets[0], np.ones(cs.N_CONCEPTS, np.float32))


def test_strip_concept_head_leaves_a_plain_anchor_state_dict() -> None:
    model = cs.attach_concept_head(_FakeArm())
    plain = cs.strip_concept_head(model.state_dict())
    assert set(plain) == set(_FakeArm().state_dict())


def _args(**changes) -> argparse.Namespace:
    base = dict(build="B2", loao_holdout="", member="anchor", fold=0, seed=42, head=7, image_size=224, batch_size=32, grad_accum=1,
                num_workers=2, added_train="x.csv", sampler="young", run_tag="v70", device="cpu", deterministic=False,
                log_steps=False, smoke=False)
    return argparse.Namespace(**{**base, **changes})


RECEIPT = {"report": {"removal_rows": {"global": [], "by_fold": {str(f): [] for f in range(5)},
                                       "by_holdout": {"ham": [], "bcn20000": [], "mskcc": []}}}}


def test_b2_patches_dataset_anchor_and_loss_then_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "load_fold_targets", lambda fold: {})
    before = {name: getattr(v6, name) for name in train_v7.SEAMS}
    with train_v7.configure(builds.BUILDS["B2"], _args(), RECEIPT):
        assert v6.nn.CrossEntropyLoss().__class__ is cs.CrossEntropyPlusConcepts
        # a constructor with PoolDataset's signature, not a subclass: a subclass made inside a function cannot be pickled
        assert v6.PoolDataset is not before["PoolDataset"] and callable(v6.PoolDataset)
        assert v6.build_arm is not before["build_arm"]
    assert {name: getattr(v6, name) for name in train_v7.SEAMS} == before


def test_b2_refuses_a_loao_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "load_fold_targets", lambda fold: {})
    with pytest.raises(SystemExit, match="per-fold teachers"):
        with train_v7.configure(builds.BUILDS["B2"], _args(loao_holdout="ham"), RECEIPT):
            pass


def test_teacher_rows_exclude_the_fold_and_the_gates_removals(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Fold 1's teacher trains on masked rows of other folds and retired splits, never on fold 1, ham_test, or a row the
    gate removes for fold 1 (d0 is linked to fold-1 row d1)."""
    master = pd.DataFrame({
        "image_id": ["d0", "d1", "d2", "r0", "t0"], "frozen_split": ["train", "train", "train", "reserved", "ham_test"],
        "fold": [0, 1, 2, None, None], "archive": ["mskcc"] * 5, "source": ["isic2019_train"] * 5,
        "master_group": ["d0", "d1", "d2", "r0", "t0"], "path": [f"{i}.jpg" for i in ["d0", "d1", "d2", "r0", "t0"]]})
    links = pd.DataFrame({"a": ["d0"], "b": ["d1"], "tier": ["at_risk"], "evidence": ["hash_at_risk"]})
    targets = pd.DataFrame({"image_id": ["d0", "d1", "d2", "r0", "t0"], **{f"{a}_present": [1] * 5 for a in cs.ATTRIBUTES}})
    master.to_csv(tmp_path / "m.csv", index=False)
    links.to_csv(tmp_path / "l.csv", index=False)
    targets.to_csv(tmp_path / "t.csv", index=False)
    monkeypatch.setattr(cs, "MASTER", tmp_path / "m.csv")
    monkeypatch.setattr(cs, "LINKS", tmp_path / "l.csv")
    monkeypatch.setattr(cs, "TARGETS_V7", tmp_path / "t.csv")
    rows, masked = cs.fold_rows(1)
    assert set(masked["image_id"]) == {"d2", "r0"}
    assert set(rows["image_id"]) == {"d2", "r0"}


def test_a_refusal_while_configuring_restores_every_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2 refuses a LOAO run partway through installing seams; nothing may stay patched (found 12 Nov)."""
    monkeypatch.setattr(cs, "load_fold_targets", lambda fold: {})
    before = {name: getattr(train_v7.v6, name) for name in train_v7.SEAMS}
    with pytest.raises(SystemExit, match="per-fold teachers"):
        with train_v7.configure(builds.BUILDS["B2"], _args(loao_holdout="ham"), RECEIPT):
            pass
    assert {name: getattr(train_v7.v6, name) for name in train_v7.SEAMS} == before
