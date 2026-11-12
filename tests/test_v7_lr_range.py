"""R4, the learning-rate range check (freeze blocker F8), and the trainer's refusals that depend on it.

CPU only, synthetic model and batches. Nothing here trains a V7 model or reads a prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from research.v7 import builds, lr_range, train_v7

SIGNED = lambda: {"sig"}  # noqa: E731


def _args(**changes) -> argparse.Namespace:
    base = dict(build="B3", loao_holdout="", member="anchor", fold=0, seed=42, head=7, image_size=384, batch_size=16, grad_accum=2,
                num_workers=2, added_train="x.csv", sampler="young", run_tag="v70", device="cpu", deterministic=False,
                log_steps=False, smoke=False)
    return argparse.Namespace(**{**base, **changes})


# ------------------------------------------------------------------ the rule, on synthetic curves
def _curve(kind: str, n: int = 150) -> list[float]:
    if kind == "down":
        return [2.0 * 0.99 ** i + 0.3 for i in range(n)]
    if kind == "flat":
        return [1.5 + 0.001 * (i % 3) + 0.0005 * i for i in range(n)]
    if kind == "up":
        return [1.0 * 0.98 ** i + 0.4 if i < 60 else 0.7 + 0.2 * (i - 60) for i in range(n)]
    if kind == "creep":                               # red-team 3d: 0.52 -> 0.93, below twice the best, still not stable
        return [0.52 + 0.41 * i / (n - 1) for i in range(n)]
    return [1.0] * 40 + [float("nan")]


def _arm(kind: str, blocks: dict | None = None) -> dict:
    return {**lr_range.arm_summary(_curve(kind), [1.0] * 10, 150), "block_movement": blocks or {}}


def test_an_arm_is_classified_from_its_own_curve() -> None:
    assert lr_range.arm_summary(_curve("down"), [0.5, 0.7], 150)["status"] == "DECREASING"
    assert lr_range.arm_summary(_curve("flat"), [], 150)["status"] == "FLAT"
    assert lr_range.arm_summary(_curve("up"), [], 150)["status"] == "DIVERGED"
    assert lr_range.arm_summary(_curve("creep"), [], 150)["status"] == "RISING"
    bad = lr_range.arm_summary(_curve("nan"), [], 150)
    assert bad["status"] == "NON_FINITE" and bad["first_non_finite_step"] == 40 and not bad["finite"]
    assert lr_range.arm_summary(_curve("down", 80), [], 150)["status"] == "INCOMPLETE"
    assert lr_range.arm_summary([], [], 150)["status"] == "NO_STEPS"


@pytest.mark.parametrize(("quarter", "one", "four", "expected"), [
    ("down", "down", "down", "PASS"), ("down", "flat", "down", "PASS"), ("down", "down", "up", "PASS_AT_EDGE"),
    ("down", "down", "nan", "PASS_AT_EDGE"), ("down", "up", "up", "FAIL"), ("down", "nan", "nan", "FAIL"), ("nan", "down", "down", "PASS"),
    ("down", "creep", "down", "FAIL"), ("down", "down", "creep", "PASS_AT_EDGE")])
def test_the_verdict_reads_only_the_declared_rate_and_its_fourfold(quarter: str, one: str, four: str, expected: str) -> None:
    out = lr_range.verdict({0.25: _arm(quarter), 1.0: _arm(one), 4.0: _arm(four)})
    assert out["verdict"] == expected and (out["edge"] is True) == (expected == "PASS_AT_EDGE")


def test_a_missing_or_unfinished_arm_fails_and_still_lower_blocks_are_reported_not_decided() -> None:
    assert lr_range.verdict({1.0: _arm("down")})["verdict"] == "FAIL"
    short = {**_arm("down"), "status": "INCOMPLETE"}
    assert lr_range.verdict({0.25: _arm("down"), 1.0: short, 4.0: _arm("down")})["verdict"] == "FAIL"
    still = {str(i): (0.001 if i < 3 else 0.05) for i in range(12)}
    out = lr_range.verdict({0.25: _arm("down"), 1.0: _arm("down", still), 4.0: _arm("down")})
    assert out["verdict"] == "PASS" and out["lower_blocks_still"] is True and out["loss_fell_at_declared_rate"] is True
    moving = {str(i): 0.03 + 0.002 * i for i in range(12)}
    assert lr_range.verdict({0.25: _arm("down"), 1.0: _arm("down", moving), 4.0: _arm("down")})["lower_blocks_still"] is False


# ------------------------------------------------------------------ the measurement, on a small CPU model
class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        for block in self.blocks:
            x = torch.tanh(block(x))
        return self.head(x)


def _setup(lr: float, n_batches: int = 2000):
    torch.manual_seed(0)
    model = Tiny()
    teacher = torch.randn(16, 3)

    def batches():
        g = torch.Generator().manual_seed(1)
        for _ in range(n_batches):
            x = torch.randn(32, 16, generator=g)
            yield x, (x @ teacher).argmax(1)

    def make_optimizer(stage: str):
        if stage == "head":
            groups = [{"params": list(model.head.parameters()), "lr": 1e-2}]
        else:
            groups = [{"params": list(model.blocks.parameters()), "lr": lr, "lr_scale": 0.5}, {"params": list(model.head.parameters()), "lr": lr}]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        for group in optimizer.param_groups:
            group["base_lr"] = group["lr"]
        return optimizer
    return model, batches(), make_optimizer


def test_a_stable_rate_passes_and_the_model_is_left_at_the_common_start() -> None:
    model, batches, make_optimizer = _setup(lr=1e-3)
    out = lr_range.range_test(model, batches, nn.CrossEntropyLoss(), make_optimizer, torch.device("cpu"), head_steps=20, arm_steps=60, ramp_steps=10)
    assert out["verdict"] == "PASS" and out["head_stage"]["steps_run"] == 20
    assert [out["arms"][m]["steps_run"] for m in lr_range.MULTIPLIERS] == [60, 60, 60]
    moved = [out["arms"][m]["overall"] for m in lr_range.MULTIPLIERS]
    assert moved[0] < moved[1] < moved[2]                                   # a larger rate moves the weights further
    assert sorted(out["arms"][1.0]["block_movement"]) == ["0", "1", "2", "3"]
    assert len(out["arms"][1.0]["group_movement"]) == 2 and out["optimizer_groups"][0]["lr_scale"] == 0.5
    assert out["settings"]["multipliers"] == [0.25, 1.0, 4.0] and out["vram_peak_gb"] is None
    after = lr_range.movement(model, {n: p.detach().clone() for n, p in model.named_parameters()})
    assert after["overall"] == 0.0


def test_an_absurd_rate_fails() -> None:
    model, batches, make_optimizer = _setup(lr=30.0)
    out = lr_range.range_test(model, batches, nn.CrossEntropyLoss(), make_optimizer, torch.device("cpu"), head_steps=10, arm_steps=60, ramp_steps=5)
    assert out["verdict"] == "FAIL" and out["arms"][1.0]["status"] in ("DIVERGED", "NON_FINITE")


def test_running_out_of_batches_is_a_failure_not_a_pass() -> None:
    model, batches, make_optimizer = _setup(lr=1e-3, n_batches=70)
    out = lr_range.range_test(model, batches, nn.CrossEntropyLoss(), make_optimizer, torch.device("cpu"), head_steps=10, arm_steps=60, ramp_steps=5)
    assert out["verdict"] == "FAIL" and out["arms"][1.0]["stopped_early"]


# ------------------------------------------------------------------ which runs need a receipt
def test_only_a_fold_0_screen_at_v6s_own_rates_runs_without_a_receipt() -> None:
    inherited = sorted(k for k, b in builds.BUILDS.items() if b.lr_inherited)
    assert inherited == ["B2", "B3", "B5", "L0", "R1", "X2"]            # every changed or new rate is absent: R2, R3, R3b, B1, C1-C3
    assert not lr_range.needs_receipt(builds.BUILDS["B3"], _args())
    assert lr_range.needs_receipt(builds.BUILDS["B3"], _args(fold=1))
    assert lr_range.needs_receipt(builds.BUILDS["B3"], _args(loao_holdout="ham"))
    for key in ("R2", "R3", "R3b", "B1", "C1", "C2", "C3"):
        assert lr_range.needs_receipt(builds.BUILDS[key], _args(build=key)), key


def _receipt(tmp_path: Path, build: str, args: argparse.Namespace, **changes) -> Path:
    config = lr_range.lr_config(builds.BUILDS[build], args)
    record = {"build": build, "member": args.member, "image_size": args.image_size, "config": config, "config_sha256": lr_range.config_sha(config),
              "verdict": "PASS", "reason": "stable at x1 and x4", "plan_freeze_sha256": "sig", "rehearsal": False, **changes}
    path = lr_range.receipt_path(config, tmp_path)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_run_that_needs_a_receipt_is_refused_without_a_passing_one(tmp_path: Path) -> None:
    r3b, args = builds.BUILDS["R3b"], _args(build="R3b", member="panderm_b", image_size=224, batch_size=32, grad_accum=1)
    with pytest.raises(SystemExit, match="no learning-rate range check"):
        lr_range.require(r3b, args, SIGNED, tmp_path)
    path = _receipt(tmp_path, "R3b", args, verdict="FAIL", reason="at the declared rate the arm is DIVERGED")
    with pytest.raises(SystemExit, match="failed its learning-rate range check"):
        lr_range.require(r3b, args, SIGNED, tmp_path)
    _receipt(tmp_path, "R3b", args, plan_freeze_sha256="made-before-signing")
    with pytest.raises(SystemExit, match="not made under a recorded Gate 0"):
        lr_range.require(r3b, args, SIGNED, tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "plan_freeze_sha256": "sig", "config": {**record["config"], "backbone_lr": 1e-5}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="different learning-rate configuration"):
        lr_range.require(r3b, args, SIGNED, tmp_path)
    _receipt(tmp_path, "R3b", args, verdict="PASS_AT_EDGE")
    assert lr_range.require(r3b, args, SIGNED, tmp_path)["verdict"] == "PASS_AT_EDGE"
    assert lr_range.require(builds.BUILDS["B3"], _args(), SIGNED, tmp_path) is None            # exempt: nothing to show


def test_a_receipt_binds_to_the_learning_rate_configuration() -> None:
    r3 = lr_range.lr_config(builds.BUILDS["R3"], _args(build="R3", member="panderm_b", image_size=224, batch_size=32, grad_accum=1))
    r3b = lr_range.lr_config(builds.BUILDS["R3b"], _args(build="R3b", member="panderm_b", image_size=224, batch_size=32, grad_accum=1))
    assert r3["layer_decay"] == 0.80 and r3b["layer_decay"] == 0.65 and r3b["backbone_lr"] == r3b["head_lr"] == 5e-4
    assert r3["backbone_lr"] != r3b["backbone_lr"] and lr_range.config_sha(r3) != lr_range.config_sha(r3b)
    r1 = lr_range.lr_config(builds.BUILDS["R1"], _args(build="R1", image_size=224, batch_size=32, grad_accum=1))
    assert r1["weight_decay"] == 0.05 and r1["effective_batch"] == 32


# ------------------------------------------------------------------ the trainer's refusals
@pytest.mark.parametrize(("key", "member", "message"), [
    ("R4", "anchor", "CPU build"), ("X1", "anchor", "is NEEDS_CODE"), ("X2", "anchor", "is NEEDS_CODE"), ("B0", "anchor", "CPU build")])
def test_the_trainer_refuses_builds_without_code_even_for_a_smoke(key: str, member: str, message: str) -> None:
    for smoke in (True, False):
        with pytest.raises(SystemExit, match=message):
            train_v7.refuse_unready(builds.BUILDS[key], _args(build=key, member=member, smoke=smoke), SIGNED)


def test_a_status_that_is_not_ready_cannot_run_of_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import dataclasses
    for status in ("NEEDS_BENCHMARK", "NEEDS_ARTEFACT", "OWNER_DECISION", "CONDITIONAL", "BLOCKED"):
        build = dataclasses.replace(builds.BUILDS["B3"], status=status)
        with pytest.raises(SystemExit, match=f"is {status}"):
            train_v7.refuse_unready(build, _args(), SIGNED, tmp_path, tmp_path)
    for status in ("NEEDS_BENCHMARK", "NEEDS_ARTEFACT", "OWNER_DECISION"):
        train_v7.refuse_unready(dataclasses.replace(builds.BUILDS["B3"], status=status), _args(smoke=True), SIGNED, tmp_path, tmp_path)
    assert builds.BUILDS["R4"].status == "NEEDS_BENCHMARK" and builds.BUILDS["L0"].status == "OWNER_DECISION"
    assert not [k for k, b in builds.BUILDS.items() if b.status == "READY" and b.member != "cpu" and not b.evidence], "a READY build cites no evidence"


def test_fold_0_at_inherited_rates_runs_and_everything_else_needs_r4(tmp_path: Path) -> None:
    train_v7.refuse_unready(builds.BUILDS["B3"], _args(), SIGNED, tmp_path, tmp_path)                     # fold 0, V6's rate
    with pytest.raises(SystemExit, match="no learning-rate range check"):
        train_v7.refuse_unready(builds.BUILDS["B3"], _args(fold=1), SIGNED, tmp_path, tmp_path)           # folds 1-4
    for key, member, size, batch, accum in (("C1", "effnet_b5_ns", 456, 32, 1), ("C2", "maxvit_t", 224, 32, 1), ("C3", "eva02_s", 224, 32, 1),
                                            ("R2", "dinov2_b", 224, 32, 1), ("R3b", "panderm_b", 224, 32, 1), ("B1", "dinov2_b", 392, 16, 2)):
        with pytest.raises(SystemExit, match="no learning-rate range check"):
            train_v7.refuse_unready(builds.BUILDS[key], _args(build=key, member=member, image_size=size, batch_size=batch, grad_accum=accum),
                                    SIGNED, tmp_path, tmp_path)
    _receipt(tmp_path, "B3", _args(fold=1))
    train_v7.refuse_unready(builds.BUILDS["B3"], _args(fold=1), SIGNED, tmp_path, tmp_path)


def test_a_candidate_members_loao_run_waits_for_a_frozen_candidate(tmp_path: Path) -> None:
    args = _args(loao_holdout="ham")
    _receipt(tmp_path, "B3", args)
    with pytest.raises(SystemExit, match="waits until the candidate is frozen"):
        train_v7.refuse_unready(builds.BUILDS["B3"], args, SIGNED, tmp_path, tmp_path)
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "candidate.json").write_text(json.dumps({"name": "c1", "members": {"m": {"1": {"v7_build": "R1"}}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="waits until the candidate is frozen"):
        train_v7.refuse_unready(builds.BUILDS["B3"], args, SIGNED, tmp_path, tmp_path)                    # frozen, but without B3
    (tmp_path / "c1" / "candidate.json").write_text(json.dumps({"name": "c1", "members": {"m": {"1": {"v7_build": "B3"}}}}), encoding="utf-8")
    train_v7.refuse_unready(builds.BUILDS["B3"], args, SIGNED, tmp_path, tmp_path)
    with pytest.raises(SystemExit, match="not declared for leave-one-archive-out"):
        train_v7.check_args_v7(builds.BUILDS["B2"])(_args(build="B2", image_size=224, batch_size=32, grad_accum=1, loao_holdout="ham"))


def test_l0_can_run_as_the_four_member_v6_vote_and_nothing_else() -> None:
    check = train_v7.check_args_v7(builds.BUILDS["L0"])
    for member, size, batch, accum in (("anchor", 224, 32, 1), ("anchor", 384, 16, 2), ("dinov2_b", 224, 32, 1), ("panderm_b", 224, 32, 1)):
        check(_args(build="L0", member=member, image_size=size, batch_size=batch, grad_accum=accum, loao_holdout="bcn20000"))
    with pytest.raises(SystemExit, match="runs at 224"):
        check(_args(build="L0", member="dinov2_b", image_size=392, batch_size=16, grad_accum=2, loao_holdout="bcn20000"))
    with pytest.raises(SystemExit, match="declared for member"):
        check(_args(build="L0", member="maxvit_t", image_size=224, batch_size=32, grad_accum=1, loao_holdout="bcn20000"))
    assert builds.sizes_of(builds.BUILDS["B1"]) == {"dinov2_b": (392,), "panderm_b": (384,)}
    assert builds.sizes_of(builds.BUILDS["B3"]) == {"anchor": (384,)}


def test_the_range_check_command_lists_receipts_and_refuses_builds_without_a_trainer(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _receipt(tmp_path, "B3", _args(fold=1))
    assert lr_range.status(tmp_path)[0]["verdict"] == "PASS"
    with pytest.raises(SystemExit, match="pass --build and --member"):
        lr_range.main([])
    for key in ("R4", "X1"):
        with pytest.raises(SystemExit, match="nothing to range-check"):
            lr_range.main(["--build", key, "--member", "anchor"])


def test_a_rehearsal_record_licenses_nothing_even_if_copied_into_the_receipt_folder(tmp_path: Path) -> None:
    r3b, args = builds.BUILDS["R3b"], _args(build="R3b", member="panderm_b", image_size=224, batch_size=32, grad_accum=1)
    _receipt(tmp_path, "R3b", args, rehearsal=True)
    with pytest.raises(SystemExit, match="is a rehearsal"):
        lr_range.require(r3b, args, SIGNED, tmp_path)
    path = _receipt(tmp_path, "R3b", args)
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["rehearsal"]                                   # a record without the flag is not trusted either
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="is a rehearsal"):
        lr_range.require(r3b, args, SIGNED, tmp_path)
    assert lr_range.REHEARSAL_DIR != lr_range.OUT_DIR


def test_the_rehearsal_folder_is_not_a_v7_output_so_it_cannot_block_gate0() -> None:
    import fnmatch
    from research.v7 import plan_freeze
    name = str(lr_range.REHEARSAL_DIR.relative_to(lr_range.REPO)).replace("\\", "/") + "/R2_dinov2_b_224_abc.json"
    assert not any(fnmatch.fnmatch(name, g) for g in plan_freeze.OUTPUT_GLOBS)
    real = str(lr_range.OUT_DIR.relative_to(lr_range.REPO)).replace("\\", "/") + "/R2_dinov2_b_224_abc.json"
    assert any(fnmatch.fnmatch(real, g) for g in plan_freeze.OUTPUT_GLOBS)
