"""CPU tests for `research/v6/train_v6.py`, its plan freeze and the gate's V6 paths. Synthetic data, no results read, no weights."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from research.v5 import train_v5 as train_v6_ledger
from research.v5.m7 import SwadAverager
from research.v6 import gate, head8, members, plan_freeze, train_v6

# ---------------------------------------------------------------- schedule


def _v5_lrs(tick_steps: int) -> list[float]:
    """The LR of every step as `train_v5.run` produces it: two optimisers and per-epoch CosineAnnealingLR."""
    param = [nn.Parameter(torch.zeros(1))]
    lrs: list[float] = []
    for first, last, lr in ((0, 3, 1e-3), (3, 30, 1e-4)):
        optimizer = torch.optim.AdamW(param, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30 - first)
        for _ in range(first, last):
            lrs += [optimizer.param_groups[0]["lr"]] * tick_steps
            optimizer.step()
            scheduler.step()
    return lrs


def test_the_anchor_schedule_is_v5s_per_epoch_cosine_on_the_v5_pool() -> None:
    schedule = train_v6.Schedule(tick_steps=4)
    ours = [(1e-3 if schedule.stage(s) == "head" else 1e-4) * schedule.factor(s) for s in range(schedule.budget)]
    assert ours == pytest.approx(_v5_lrs(4), rel=1e-12)


def test_head_stage_is_one_tenth_of_the_budget_whatever_the_pool() -> None:
    schedule = train_v6.Schedule(tick_steps=447)
    assert schedule.budget == 13410 and schedule.head_steps == 1341
    assert schedule.stage(1340) == "head" and schedule.stage(1341) == "finetune"


def test_warmup_rises_linearly_to_one_then_the_cosine_falls() -> None:
    schedule = train_v6.Schedule(tick_steps=10, warmup_fraction=0.2)
    assert schedule.warmup_steps == 54
    start = schedule.head_steps
    assert schedule.factor(start) == pytest.approx(1 / 54)
    assert schedule.factor(start + 53) == pytest.approx(1.0)
    assert schedule.factor(start + 54) == pytest.approx(1.0)
    after = [schedule.factor(s) for s in range(start + 54, schedule.budget)]
    assert all(a >= b for a, b in zip(after, after[1:])) and 0 < after[-1] < 0.02


def test_epochs_shrink_to_hold_steps_constant() -> None:
    per_v5_epoch = train_v6.optimiser_steps_per_epoch(14313, 32, 1)
    assert per_v5_epoch == 447
    assert train_v6.optimiser_steps_per_epoch(14313, 16, 2) == 447  # the 384 px member has the same budget
    budget = train_v6.Schedule(per_v5_epoch).budget
    assert train_v6.epochs_for(budget, per_v5_epoch) == 30
    assert train_v6.epochs_for(budget, 2 * per_v5_epoch) == 15
    assert train_v6.epochs_for(budget, 1000) == 14  # 13.41: the last epoch is cut at the budget


# ---------------------------------------------------------------- the loop


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.pos_embed = nn.Parameter(torch.ones(1), requires_grad=False)
        self.head = nn.Linear(4, 8)

    def get_classifier(self) -> nn.Module:
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _batches(n: int, batch: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.randn(batch, 4), torch.randint(0, 8, (batch,))) for _ in range(n)]


def _progress(model: nn.Module, schedule: train_v6.Schedule, spec: members.MemberSpec) -> train_v6.Progress:
    fixed = train_v6.freeze_map(model)
    return train_v6.Progress(schedule, lambda stage: train_v6.stage_optimizer(model, spec, stage, fixed))


DINO = members.ROSTER["dinov2_b"]


@pytest.mark.parametrize("per_epoch", [10, 25, 37])
def test_the_run_stops_exactly_at_the_budget_and_changes_stage_mid_epoch(per_epoch: int) -> None:
    model, schedule = _Net(), train_v6.Schedule(tick_steps=10)
    progress = _progress(model, schedule, DINO)
    device, criterion = torch.device("cpu"), nn.CrossEntropyLoss()
    for _ in range(train_v6.epochs_for(schedule.budget, per_epoch)):
        train_v6.train_epoch(model, _batches(per_epoch), criterion, progress, device, None, "t")
    assert progress.steps == schedule.budget == 300
    assert progress.stage == "finetune"


def test_accumulation_counts_optimiser_steps_and_matches_the_budget() -> None:
    model, schedule = _Net(), train_v6.Schedule(tick_steps=5)
    progress = _progress(model, schedule, DINO)
    for _ in range(30):
        train_v6.train_epoch(model, _batches(11), nn.CrossEntropyLoss(), progress, torch.device("cpu"), None, "t", accum=2)
    assert progress.steps == schedule.budget == 150  # 11 micro-batches = 5 steps; the odd one is dropped, as in V5


def test_head_stage_moves_only_the_head_and_a_builder_freeze_survives_both_stages() -> None:
    model = _Net()
    fixed = train_v6.freeze_map(model)
    train_v6.stage_optimizer(model, DINO, "head", fixed)
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["head.weight", "head.bias"]
    optimizer = train_v6.stage_optimizer(model, DINO, "finetune", fixed)
    assert not model.pos_embed.requires_grad and model.backbone.weight.requires_grad
    assert [g["lr"] for g in optimizer.param_groups] == [DINO.backbone_lr, DINO.head_lr]


def test_real_panderm_architecture_keeps_its_position_embedding_frozen_through_the_stages() -> None:
    spec = members.ROSTER["panderm_b"]
    model = members.build_member(spec, num_classes=8, pretrained=False)
    fixed = train_v6.freeze_map(model)
    train_v6.stage_optimizer(model, spec, "head", fixed)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {"head.weight", "head.bias"}
    optimizer = train_v6.stage_optimizer(model, spec, "finetune", fixed)
    assert not model.pos_embed.requires_grad
    assert all(p is not model.pos_embed for g in optimizer.param_groups for p in g["params"])
    assert max(g["lr"] for g in optimizer.param_groups) == pytest.approx(spec.backbone_lr)
    assert all("base_lr" in g for g in optimizer.param_groups)


def test_a_non_finite_loss_voids_the_run() -> None:
    model = _Net()
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    progress = _progress(model, train_v6.Schedule(tick_steps=4), DINO)
    with pytest.raises(train_v6.VoidRun):
        train_v6.train_epoch(model, _batches(2), nn.CrossEntropyLoss(), progress, torch.device("cpu"), None, "t")


def test_evaluate_reads_the_8_class_head_as_a_7_class_simplex() -> None:
    model = _Net()
    loader = [(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))]
    out = train_v6.evaluate(model, loader, torch.device("cpu"), head=8)
    assert out["probabilities"].shape == (4, 7) and out["probabilities_8"].shape == (4, 8)
    assert out["probabilities"].sum(1) == pytest.approx(np.ones(4), abs=1e-5)
    assert out["probabilities"][:, 0] == pytest.approx(out["probabilities_8"][:, [0, 6]].sum(1))


def test_the_average_is_the_mean_of_the_window_and_leaves_last_alone() -> None:
    model = _Net()
    averager = train_v6.averaging.EpochAverager(3)
    assert isinstance(averager, SwadAverager) and (averager.start, averager.end) == (2, 3)
    values = []
    for epoch in range(1, 4):
        with torch.no_grad():
            model.backbone.weight.fill_(float(epoch))
        values.append(float(epoch))
        averager.update(model, epoch)
    assert averager.n == 2
    assert averager.state_dict()["backbone.weight"][0, 0].item() == pytest.approx(2.5)
    assert model.backbone.weight[0, 0].item() == 3.0


# ---------------------------------------------------------------- rows


def _pool(n_young: int, n_other: int) -> pd.DataFrame:
    return pd.DataFrame({"age_band": ["<40"] * n_young + ["40-59"] * (n_other // 2) + ["unknown"] * (n_other - n_other // 2)})


def test_rows_with_no_age_are_not_young() -> None:
    assert train_v6.young_mask(_pool(2, 4)).tolist() == [True, True, False, False, False, False]


def test_young_sampler_holds_the_share_against_dilution() -> None:
    frame = _pool(2000, 18000)
    sampler = train_v6.young_sampler(frame, 0.276, torch.Generator().manual_seed(1))
    draws = np.fromiter(iter(sampler), dtype=np.int64)
    assert len(draws) == len(frame)
    assert train_v6.young_mask(frame)[draws].mean() == pytest.approx(0.276, abs=0.012)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame({"image_id": ["m1", "m2"], "group_id": ["g1", "g2"], "effective_lesion_id": ["l1", "l2"]})


def _added(**changes) -> pd.DataFrame:
    frame = pd.DataFrame({"image_id": ["a1", "a2"], "group_id": ["ga1", "ga2"], "effective_lesion_id": ["la1", "la2"],
                          "age_band": ["<40", "unknown"], "archive": ["x", "x"], "class_index_7": [0, 5],
                          "class_index_8": [6, 5], "escalating_7": [True, False], "image_path": ["p/a1.jpg", "p/a2.jpg"]})
    for column, values in changes.items():
        frame[column] = values
    return frame


def _load(tmp_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    path = tmp_path / "added.csv"
    frame.to_csv(path, index=False)
    return train_v6.load_added_train(path, _manifest(), pd.DataFrame({"image_id": ["m1", "y1"]}))


def test_clean_added_rows_load(tmp_path: Path) -> None:
    assert len(_load(tmp_path, _added())) == 2


@pytest.mark.parametrize("changes", [{"image_id": ["m1", "a2"]}, {"group_id": ["g2", "ga2"]},
                                     {"effective_lesion_id": ["l1", "la2"]}, {"image_id": ["y1", "a2"]},
                                     {"class_index_8": [0, 6]}])
def test_added_rows_cannot_touch_the_manifest_the_pool_or_the_labels(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(AssertionError):
        _load(tmp_path, _added(**changes))


@pytest.mark.parametrize("changes", [{"image_id": ["a1", "a1"]}, {"effective_lesion_id": ["la1", None]}])
def test_added_rows_refuse_repeats_and_empty_cells(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(SystemExit):
        _load(tmp_path, _added(**changes))


def test_added_rows_need_every_column(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _load(tmp_path, _added().drop(columns="image_path"))


def test_every_8_class_index_collapses_as_the_manifest_says() -> None:
    labels = np.arange(8)
    assert head8.collapse_labels(labels).tolist() == [0, 1, 2, 3, 4, 5, 0, 6]


# ---------------------------------------------------------------- arguments and ids


def _args(**values) -> train_v6.argparse.Namespace:
    base = dict(member="anchor", image_size=224, batch_size=32, grad_accum=1, sampler="plain", added_train="", smoke=False, run_tag="t")
    return train_v6.argparse.Namespace(**{**base, **values})


@pytest.mark.parametrize("bad", [dict(member="dinov2_b", image_size=384, batch_size=16, grad_accum=2),
                                 dict(image_size=384), dict(batch_size=16, grad_accum=2),
                                 dict(sampler="young"), dict(run_tag="")])
def test_a_run_outside_the_recipe_is_refused(bad: dict) -> None:
    with pytest.raises(SystemExit):
        train_v6.check_args(_args(**bad))


def test_the_recipe_forms_are_accepted() -> None:
    train_v6.check_args(_args())
    train_v6.check_args(_args(image_size=384, batch_size=16, grad_accum=2))
    train_v6.check_args(_args(member="panderm_b", added_train="x.csv", sampler="young"))
    train_v6.check_args(_args(run_tag="", smoke=True))


def test_run_ids_are_the_ones_the_runner_and_the_gate_build() -> None:
    assert train_v6.run_id_for("anchor", 0, 42, "v61t") == "anchor_f0_s42_in22k_v61t"
    assert train_v6.run_id_for("panderm_b", 0, 42, "v63") == "panderm_b_f0_s42_v63"
    assert train_v6.run_id_for("anchor", 0, 42, "v63", 384) == "anchor384_f0_s42_in22k_v63"
    assert train_v6.run_id_for("anchor", 0, 42, "v63", 224, 7) == "anchor_h7_f0_s42_in22k_v63"
    forms = {train_v6.run_id_for("anchor", 0, 42, "t", size, head) for size in (224, 384) for head in (7, 8)}
    assert len(forms) == 4  # no two forms can share a run id, so the runner can never skip one as "complete"
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "run_v6_queue.ps1").read_text(encoding="utf-8")
    assert re.search(r'research\.v6\.train_v6"\) \{\s+\$member = Get-Arg \$t "--member"', runner)
    assert 'results\\v6\\runs\\{0}.json' in runner and '$member -eq "anchor"' in runner


def test_only_dinov2_takes_the_anchors_head_dropout() -> None:
    assert members.ROSTER["dinov2_b"].head_dropout == 0.3
    assert members.ROSTER["panderm_b"].head_dropout == members.ROSTER["siglip2_b"].head_dropout == 0.0
    model = members.build_member(members.ROSTER["dinov2_b"], num_classes=8, pretrained=False)
    assert model.head_drop.p == 0.3


def test_the_summary_has_every_field_the_ledger_writer_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / "experiments.csv"
    pd.DataFrame([{"session": "other", "method": "x", "macro_f1": 0.5}]).to_csv(ledger, index=False)
    monkeypatch.setattr(train_v6_ledger, "LEDGER_PATH", ledger)
    history = [{"val_macro_f1": 0.7, "val_accuracy": 0.8, "val_balanced_accuracy": 0.6, "val_escalation_sensitivity": 0.75, "val_missed_serious": 3}]
    train_v6.write_ledger({"session": "v6_t", "run_id": "anchor_f0_s42_in22k_t", "split": "f0", "history": history, "arm": "anchor", "seed": 42,
                           "batch_size": 32, "grad_accum": 1, "effective_batch": 32, "num_workers": 2, "image_size": 224, "trunk": "in22k",
                           "extra_train_rows": 2078, "epochs_run": 30, "train_time_seconds": 60.0})
    rows = pd.read_csv(ledger)
    assert rows["method"].tolist() == ["x", "anchor_f0_s42_in22k_t"] and rows["macro_f1"].tolist() == [0.5, 0.7]


# ---------------------------------------------------------------- plan freeze


def test_the_freeze_covers_everything_v6_imports() -> None:
    paths = set(plan_freeze.frozen_paths())
    assert {"research/v5/tta_v5.py", "research/tta/transforms.py", "ml/preprocessing/transforms.py", "research/stats/families.py", "results/v5/queue_V61.txt"} <= paths
    assert all((plan_freeze.REPO / p).exists() for p in paths)


def test_no_freeze_file_blocks_a_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="owner has not approved"):
        plan_freeze.require(tmp_path / "none.json")


def test_the_freeze_hash_catches_a_changed_file_and_part0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_freeze.lock_stage0, "problems", lambda: [])
    record = plan_freeze.build_record("test")
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert plan_freeze.require(path) == record["part0_sha256"]
    tampered = json.loads(json.dumps(record))
    tampered["files"]["research/v6/train_v6.py"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert plan_freeze.problems(path) == ["changed: research/v6/train_v6.py"]
    tampered = json.loads(json.dumps(record))
    tampered["part0_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert plan_freeze.problems(path) == ["changed: Part 0 of docs/V6_RUNSHEET.md"]


def test_a_second_approval_needs_an_amendment_and_keeps_the_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_freeze, "RUN_DIR", tmp_path / "no_runs")
    path = tmp_path / "freeze.json"
    with pytest.raises(SystemExit, match="no freeze to amend"):
        plan_freeze.approve("why", path)
    plan_freeze.approve("", path)
    with pytest.raises(SystemExit, match="amendment"):
        plan_freeze.approve("", path)
    plan_freeze.approve("a reason", path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["note"] == "a reason" and len(record["amendments"]) == 1


def test_a_first_freeze_cannot_follow_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "anchor_f0_s42_in22k_v61t.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(plan_freeze, "RUN_DIR", runs)
    with pytest.raises(SystemExit, match="could not precede"):
        plan_freeze.approve("", tmp_path / "freeze.json")


# ---------------------------------------------------------------- gate


def _score(**values) -> dict:
    return {e: values.get(e, 0.8) for e in gate.DELTAS}


def test_averaging_is_adopted_only_with_a_pauc_gain_and_macro_f1_kept() -> None:
    last = {42: _score()}
    assert gate.averaging_read({42: _score(pauc_all=0.801)}, last)["averaging_is_the_read_out"]
    assert gate.averaging_read({42: _score(pauc_histo=0.801)}, last)["averaging_is_the_read_out"]
    assert not gate.averaging_read({42: _score()}, last)["averaging_is_the_read_out"]
    assert not gate.averaging_read({42: _score(pauc_all=0.801, macro_f1=0.789)}, last)["averaging_is_the_read_out"]


def test_pred_path_finds_anchor_member_and_average_files(tmp_path: Path) -> None:
    for name in ("anchor_f0_s42_in22k_v61t.csv", "anchor_f0_s42_in22k_v61t_swad.csv", "panderm_b_f0_s42_v63.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert gate.pred_path("anchor", 42, "v61t", tmp_path).name == "anchor_f0_s42_in22k_v61t.csv"
    assert gate.pred_path("anchor", 42, "v61t", tmp_path, checkpoint="swad").name == "anchor_f0_s42_in22k_v61t_swad.csv"
    assert gate.pred_path("panderm_b", 42, "v63", tmp_path).name == "panderm_b_f0_s42_v63.csv"
    with pytest.raises(FileNotFoundError):
        gate.pred_path("dinov2_b", 42, "v63", tmp_path)
