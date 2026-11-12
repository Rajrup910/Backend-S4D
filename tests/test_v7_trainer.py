"""V7 trainer seams, build registry, pAUC loss and Gate 0. No GPU, no images: each seam is checked in place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchvision.transforms as T

from research.v4.recipe import Recipe
from research.v6 import members
from research.v6 import train_v6 as v6
from research.v7 import builds, darkness, losses, plan_freeze, train_v7


def _args(**changes) -> argparse.Namespace:
    base = dict(build="B3", loao_holdout="", member="anchor", fold=2, seed=42, head=7, image_size=384, batch_size=16, grad_accum=2,
                num_workers=2, added_train="x.csv", sampler="young", run_tag="v70", device="cpu", deterministic=False,
                log_steps=False, smoke=False)
    return argparse.Namespace(**{**base, **changes})


RECEIPT = {"report": {"removal_rows": {"global": ["G1"], "by_fold": {str(f): [f"F{f}"] for f in range(5)},
                                       "by_holdout": {"ham": ["H"], "bcn20000": ["B"], "mskcc": ["M"]}}}}


# ------------------------------------------------------------------ registry
def test_every_build_declares_hypothesis_endpoint_drop_rule_cost_basis_and_status() -> None:
    for build in builds.BUILDS.values():
        assert build.hypothesis and build.endpoint and build.drop_rule and build.cost, build.key
        assert build.cost_basis in ("MEASURED", "EXTRAPOLATED", "UNMEASURED"), build.key
        assert build.status in ("DONE", "READY", "NEEDS_CODE", "NEEDS_BENCHMARK", "NEEDS_ARTEFACT", "CONDITIONAL", "OWNER_DECISION"), build.key


def test_the_owner_order_b0_to_b6_is_kept() -> None:
    assert [k for k in builds.BUILDS if k.startswith("B")] == [f"B{i}" for i in range(7)]


def test_measured_costs_match_the_v6_run_logs() -> None:
    repo = Path(__file__).resolve().parents[1]
    for member, (minutes, vram) in builds.MEASURED.items():
        logs = [json.loads(p.read_text()) for p in sorted((repo / "results/v6/runs").glob(f"{member}_f[1-4]_s42*_v64.json"))]
        assert len(logs) == 4
        assert abs(sum(x["train_time_seconds"] for x in logs) / 4 / 60 - minutes) < 0.1
        assert abs(max(x["vram_peak_gb"] for x in logs) - vram) < 0.01


def test_loao_minutes_scale_with_the_pool() -> None:
    assert round(builds.loao_minutes("anchor_h7", "bcn20000"), 1) == 41.4
    total = sum(builds.loao_minutes(m, h) for m in builds.MEASURED for h in builds.LOAO_POOL_ROWS)
    assert 735 < total < 745


# ------------------------------------------------------------------ pAUC loss
def test_pauc_loss_is_zero_when_positives_clear_the_hard_negatives_by_the_margin() -> None:
    loss = losses.OneWayPAUCLoss(beta=0.2, margin=0.6)
    score = torch.tensor([3.0, 2.8, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    positive = torch.tensor([True, True] + [False] * 10)
    assert loss(score, positive).item() == 0.0
    score[0] = 0.5
    assert loss(score, positive).item() > 0.0


def test_pauc_loss_looks_only_at_the_top_negatives() -> None:
    loss = losses.OneWayPAUCLoss(beta=0.2, margin=0.6)
    positive = torch.tensor([True] + [False] * 10)
    a = torch.tensor([2.0, 1.9, 1.8] + [-5.0] * 8)
    b = torch.tensor([2.0, 1.9, 1.8] + [-1.0] * 8)  # only easy negatives moved
    assert torch.isclose(loss(a, positive), loss(b, positive))


def test_a_batch_without_both_classes_contributes_zero_and_keeps_the_graph() -> None:
    score = torch.zeros(4, requires_grad=True)
    out = losses.OneWayPAUCLoss()(score, torch.zeros(4, dtype=torch.bool))
    out.backward()
    assert out.item() == 0.0 and score.grad is not None


def test_cross_entropy_plus_pauc_has_ces_signature_and_adds_the_term() -> None:
    torch.manual_seed(0)
    logits, labels = torch.randn(16, 7, requires_grad=True), torch.tensor([0, 1, 4, 5] * 4)
    ce = torch.nn.CrossEntropyLoss()
    total = losses.CrossEntropyPlusPAUC(ce)(logits, labels)
    assert total.item() >= ce(logits, labels).item()
    total.backward()
    assert logits.grad is not None


# ------------------------------------------------------------------ seams
def test_configure_patches_every_seam_and_restores_them(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {name: getattr(v6, name) for name in train_v7.SEAMS}
    with train_v7.configure(builds.BUILDS["B3"], _args(), RECEIPT):
        assert v6.nn.CrossEntropyLoss().__class__ is losses.CrossEntropyPlusPAUC
        assert v6.RUN_DIR == train_v7.V7_DIR / "runs"
        assert v6.run_id_for("anchor", 2, 42, "v70", 384, 7).startswith("B3_")
    assert {name: getattr(v6, name) for name in train_v7.SEAMS} == before


def test_seams_are_restored_even_when_the_run_fails() -> None:
    before = v6.fold_frames
    with pytest.raises(RuntimeError):
        with train_v7.configure(builds.BUILDS["B3"], _args(), RECEIPT):
            raise RuntimeError("boom")
    assert v6.fold_frames is before


def test_the_fold_split_drops_the_guards_global_and_fold_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"image_id": ["G1", "F2", "F3", "keep"]})
    monkeypatch.setattr(v6, "fold_frames", lambda manifest, fold: (frame.copy(), frame.head(0)))
    with train_v7.configure(builds.BUILDS["B3"], _args(fold=2), RECEIPT) as record:
        train, _ = v6.fold_frames(None, 2)
    assert list(train["image_id"]) == ["F3", "keep"] and record["removed_training_rows"] == 2


def test_the_loao_split_uses_the_holdout_removals(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"image_id": ["B", "G1", "keep"]})
    monkeypatch.setattr(train_v7.train_v5, "loao_frames", lambda manifest, holdout: (frame.copy(), frame.head(0)))
    with train_v7.configure(builds.BUILDS["L0"], _args(build="L0", loao_holdout="bcn20000", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        train, _ = v6.fold_frames(None, 0)
    assert list(train["image_id"]) == ["keep"]


def test_the_darkness_step_sits_right_after_the_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(darkness, "load_basis", lambda fold=None, holdout=None: "basis")
    monkeypatch.setattr(darkness, "LesionMelaninScale", lambda basis, seed=None: ("melanin", basis))
    with train_v7.configure(builds.BUILDS["B5"], _args(build="B5", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        steps = v6.build_train_transform(Recipe(image_size=224)).transforms
    crop = [i for i, s in enumerate(steps) if isinstance(s, T.RandomResizedCrop)][0]
    assert steps[crop + 1] == ("melanin", "basis")


def test_learning_rate_builds_override_the_member_spec_and_the_anchor_weight_decay() -> None:
    with train_v7.configure(builds.BUILDS["R3"], _args(build="R3", member="panderm_b", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        spec = members.require_exact(members.ROSTER["panderm_b"])
    assert spec.layer_decay == 0.80 and members.ROSTER["panderm_b"].layer_decay == 0.65
    with train_v7.configure(builds.BUILDS["R3b"], _args(build="R3b", member="panderm_b", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        spec = members.require_exact(members.ROSTER["panderm_b"])
    assert spec.backbone_lr == spec.head_lr == 5e-4 and spec.layer_decay == 0.65
    assert members.ROSTER["panderm_b"].backbone_lr == 5e-4 * members.BATCH_SIZE / 128  # V6's spec untouched
    with train_v7.configure(builds.BUILDS["R1"], _args(build="R1", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        assert v6.WEIGHT_DECAY == 0.05
    assert v6.WEIGHT_DECAY == 1e-4


@pytest.mark.parametrize("changes, message", [
    ({"member": "dinov2_b"}, "declared for member"), ({"image_size": 224, "batch_size": 32, "grad_accum": 1}, "runs at 384"),
    ({"batch_size": 32, "grad_accum": 1}, "batch 16"),
    ({"run_tag": ""}, "needs --run-tag")])
def test_check_args_refuses_runs_the_build_did_not_declare(changes: dict, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        train_v7.check_args_v7(builds.BUILDS["B3"])(_args(**changes))


def test_the_trainer_refuses_without_a_data_gate_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_v7, "RECEIPT", tmp_path / "none.json")
    with pytest.raises(SystemExit, match="no data-gate receipt"):
        train_v7.main(["--build", "B3", "--member", "anchor", "--image-size", "384", "--batch-size", "16", "--grad-accum", "2", "--smoke"])


def test_unbenchmarked_builds_refuse_a_run_of_record(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    monkeypatch.setitem(builds.BUILDS, "B1", dataclasses.replace(builds.BUILDS["B1"], status="NEEDS_BENCHMARK"))
    with pytest.raises(SystemExit, match="NEEDS_BENCHMARK"):
        train_v7.main(["--build", "B1", "--member", "dinov2_b", "--image-size", "392", "--batch-size", "16", "--grad-accum", "2", "--run-tag", "x"])


@pytest.mark.parametrize("member, size, ok", [("dinov2_b", 392, True), ("panderm_b", 384, True), ("dinov2_b", 384, False),
                                              ("panderm_b", 392, False), ("anchor", 384, False)])
def test_b1_runs_each_member_at_its_own_size_and_batch(member: str, size: int, ok: bool) -> None:
    args = _args(build="B1", member=member, image_size=size)
    check = train_v7.check_args_v7(builds.BUILDS["B1"])
    if ok:
        check(args)
        with pytest.raises(SystemExit, match="batch 16"):
            check(_args(build="B1", member=member, image_size=size, batch_size=32, grad_accum=1))
    else:
        with pytest.raises(SystemExit):
            check(args)


def test_b1_sets_each_members_image_size() -> None:
    for member, size in (("dinov2_b", 392), ("panderm_b", 384)):
        with train_v7.configure(builds.BUILDS["B1"], _args(build="B1", member=member, image_size=size), RECEIPT):
            assert members.require_exact(members.ROSTER[member]).image_size == size


# ------------------------------------------------------------------ Gate 0
def test_gate_0_refuses_unsigned_and_changed_plans(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = tmp_path / "plan.md"
    doc.write_text("rule A", encoding="utf-8")
    monkeypatch.setattr(plan_freeze, "REPO", tmp_path)
    monkeypatch.setattr(plan_freeze, "FROZEN", ("plan.md",))
    freeze = tmp_path / "freeze.json"
    with pytest.raises(SystemExit, match="unsigned"):
        plan_freeze.require(freeze)
    plan_freeze.sign(freeze, owner="test owner", check_decisions=False)   # the decision table has its own test (test_v7_freeze.py)
    assert plan_freeze.require(freeze)
    doc.write_text("rule B", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after signing"):
        plan_freeze.require(freeze)
    with pytest.raises(SystemExit, match="owner-approved"):
        plan_freeze.main(["--sign"])


# ------------------------------------------------------------------ V7 pool loader (retired splits)
def _manifest_pool(tmp_path: Path):
    manifest = pd.DataFrame({"image_id": ["D1", "R1", "R2", "T1"], "split": ["train", "reserved", "ham_val", "ham_test"],
                             "group_id": ["gd", "gr1", "gr2", "gt"], "class_index_7": [5, 4, 5, 1], "class_index_8": [5, 4, 5, 1]})
    pool = pd.DataFrame({"image_id": ["D1"]})
    cols = dict(effective_lesion_id="e", age_band="<40", archive="bcn20000", escalating_7=False, image_path="x.jpg")
    rows = pd.DataFrame([{"image_id": "A1", "group_id": "ga", "class_index_7": 5, "class_index_8": 5, "pool_source": "v6_added", **cols},
                         {"image_id": "R1", "group_id": "gr1", "class_index_7": 4, "class_index_8": 4, "pool_source": "retired_split", **cols},
                         {"image_id": "R2", "group_id": "gr2", "class_index_7": 5, "class_index_8": 5, "pool_source": "retired_split", **cols}])
    return manifest, pool, rows


def test_the_pool_loader_sends_outside_rows_through_v6_and_checks_retired_rows(tmp_path: Path) -> None:
    manifest, pool, rows = _manifest_pool(tmp_path)
    path = tmp_path / "pool.csv"
    rows.to_csv(path, index=False)
    seen = {}

    def v6_loader(p, m, pl):
        seen["ids"] = list(pd.read_csv(p)["image_id"])
        return pd.read_csv(p)

    out = train_v7.load_pool_v7(v6_loader, path, manifest, pool)
    assert seen["ids"] == ["A1"] and list(out["image_id"]) == ["A1", "R1", "R2"]


@pytest.mark.parametrize("change, message", [
    (lambda r: r.assign(class_index_7=[5, 5, 5]), "differs from the manifest"),
    (lambda r: r.assign(image_id=["A1", "D1", "R2"]), "not a manifest row of a retired split"),
    (lambda r: r.assign(group_id=["ga", "gd", "gr2"]), "shares a lesion group"),
])
def test_the_pool_loader_refuses_tampered_or_overlapping_retired_rows(tmp_path: Path, change, message: str) -> None:
    manifest, pool, rows = _manifest_pool(tmp_path)
    path = tmp_path / "pool.csv"
    change(rows).to_csv(path, index=False)
    with pytest.raises(AssertionError, match=message):
        train_v7.load_pool_v7(lambda p, m, pl: pd.read_csv(p), path, manifest, pool)


# ------------------------------------------------------------------ second-chance candidates (C1-C3)
from research.v7 import members_v7  # noqa: E402


def test_candidates_join_the_roster_only_during_a_run_and_v6_is_restored() -> None:
    before_roster, before_build = members.ROSTER, members.build_member
    args = _args(build="C2", member="maxvit_t", image_size=224, batch_size=32, grad_accum=1)
    with train_v7.configure(builds.BUILDS["C2"], args, RECEIPT):
        assert set(members_v7.CANDIDATES) <= set(members.ROSTER)
        assert members.build_member is members_v7.build_member
    assert members.ROSTER is before_roster and members.build_member is before_build
    assert not set(members_v7.CANDIDATES) & set(members.ROSTER)


@pytest.mark.parametrize("name", sorted(members_v7.CANDIDATES))
def test_each_candidate_builds_on_cpu_with_a_7_class_head_and_its_own_normalisation(name: str) -> None:
    spec = members_v7.CANDIDATES[name]
    model = members_v7.build_member(spec, num_classes=7, pretrained=False)
    assert model.get_classifier().out_features == 7
    x = torch.zeros(1, 3, spec.image_size, spec.image_size)
    assert model.eval()(x).shape == (1, 7)
    if name in members_v7.GRAD_CHECKPOINTING:
        assert getattr(model, "grad_checkpointing", False)


def test_the_v7_builder_hands_v6_members_to_v6s_own_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(members_v7, "_V6_BUILD", lambda spec, **kw: seen.append(spec.name) or "v6-built")
    assert members_v7.build_member(members.ROSTER["dinov2_b"], num_classes=7, pretrained=False) == "v6-built" and seen == ["dinov2_b"]


def test_456_px_is_only_for_members_declared_with_gradient_checkpointing() -> None:
    check = train_v7.check_args_v7(builds.BUILDS["C1"])
    check(_args(build="C1", member="effnet_b5_ns", image_size=456, batch_size=32, grad_accum=1))
    with pytest.raises(SystemExit, match="declared for member"):
        check(_args(build="C1", member="maxvit_t", image_size=456, batch_size=32, grad_accum=1))


# ------------------------------------------------------------------ decision 7: hold V6's step budget
# These patch and call `train_v7.v6`, the module object the seams live on (another test reloads `train_v6`).
def _pool(train_ids: list[str], extra_ids: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_v7.v6, "fold_frames", lambda manifest, fold: (pd.DataFrame({"image_id": train_ids}), pd.DataFrame({"image_id": []})))
    monkeypatch.setattr(train_v7.v6, "load_extra_train", lambda *a, **k: pd.DataFrame({"image_id": extra_ids}))


def test_the_budget_is_v6s_pool_before_the_guards_removals(monkeypatch: pytest.MonkeyPatch) -> None:
    train = [f"t{i}" for i in range(14000)] + ["G1", "F2"]          # G1: a global removal; F2: fold 2's removal
    _pool(train, [f"y{i}" for i in range(311)], monkeypatch)
    args = _args(build="R1", fold=2, image_size=224, batch_size=32, grad_accum=1)
    assert train_v7.STEP_BUDGET == "hold_v6"
    with train_v7.configure(builds.BUILDS["R1"], args, RECEIPT) as record:
        kept, _ = train_v7.v6.fold_frames(None, 2)
        extra = train_v7.v6.load_extra_train("young", kept, None)
        reduced = train_v7.v6.optimiser_steps_per_epoch(len(kept) + len(extra), 32, 1)
        schedule = train_v7.v6.Schedule(reduced, 0.0)
    before = len(train) + 311
    assert record["pool_rows_before_removal"] == before and record["pool_rows_kept"] == before - 2
    assert schedule.budget == train_v7.v6.EPOCHS_DEFAULT * (before // 32)                      # V6's budget, not the reduced pool's
    assert record["tick_steps_held"] == before // 32 and record["tick_steps_reduced"] == (before - 2) // 32


def test_the_budget_seam_refuses_a_tick_it_did_not_expect(monkeypatch: pytest.MonkeyPatch) -> None:
    _pool([f"t{i}" for i in range(3200)], [], monkeypatch)
    with train_v7.configure(builds.BUILDS["R1"], _args(build="R1", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        train_v7.v6.fold_frames(None, 2)
        train_v7.v6.load_extra_train("young", None, None)
        with pytest.raises(SystemExit, match="refusing to guess"):
            train_v7.v6.Schedule(7, 0.0)


def test_gate_pool_restores_the_reduced_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _pool([f"t{i}" for i in range(3200)] + ["G1"] * 64, [], monkeypatch)
    monkeypatch.setattr(train_v7, "STEP_BUDGET", "gate_pool")
    with train_v7.configure(builds.BUILDS["R1"], _args(build="R1", image_size=224, batch_size=32, grad_accum=1), RECEIPT):
        kept, _ = train_v7.v6.fold_frames(None, 2)
        assert train_v7.v6.Schedule(len(kept) // 32, 0.0).budget == train_v7.v6.EPOCHS_DEFAULT * 100
