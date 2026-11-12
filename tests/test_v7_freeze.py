"""V7 Gate 0 and its amendment record (plan 13.6, freeze blocker F3). Synthetic files only; never signs the real plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.v7 import plan_freeze

OK = lambda: {"command": "stub", "returncode": 0, "tail": ["3 passed"]}  # noqa: E731
FAIL = lambda: {"command": "stub", "returncode": 1, "tail": ["1 failed"]}  # noqa: E731
PLAN_OPEN = "# plan\n### 13.5 Owner decisions\n| # | Decision | Recommended | Owner's choice |\n|:-:|---|---|---|\n| 1 | rule | 10.4 | as recommended |\n| 2 | scope | vote | |\n| 3 | unit | (a) | AWAITING_OWNER |\n\n### 13.6 next\n| 9 | not a decision | x | |\n"
PLAN_DONE = PLAN_OPEN.replace("| vote | |", "| vote | (a), 12 Nov |").replace("AWAITING_OWNER", "as recommended")


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A tiny repository: one protected rule file, one amendable build file, and the plan with its decision table."""
    for name, text in (("rule.py", "MARGIN = 0.02"), ("builds.py", "BUILDS = ['B1']"), ("plan.md", PLAN_DONE)):
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(plan_freeze, "REPO", tmp_path)
    monkeypatch.setattr(plan_freeze, "PROTECTED", ("rule.py",))
    monkeypatch.setattr(plan_freeze, "AMENDABLE", ("builds.py", "plan.md"))
    monkeypatch.setattr(plan_freeze, "FROZEN", ("rule.py", "builds.py", "plan.md"))
    monkeypatch.setattr(plan_freeze, "PLAN", "plan.md")
    return tmp_path


def _freeze(repo: Path) -> Path:
    return repo / "results" / "freeze.json"


def test_signing_refuses_while_an_owner_decision_is_blank_or_awaiting(repo: Path) -> None:
    assert plan_freeze.open_decisions(PLAN_OPEN) == ["2", "3"]          # the row after 13.5's table is not a decision
    assert plan_freeze.open_decisions(PLAN_DONE) == []
    (repo / "plan.md").write_text(PLAN_OPEN, encoding="utf-8")
    with pytest.raises(SystemExit, match=r"owner decisions \['2', '3'\]"):
        plan_freeze.sign(_freeze(repo), owner="Owner")
    assert not _freeze(repo).exists()


def test_signing_needs_a_named_owner_and_happens_once(repo: Path) -> None:
    with pytest.raises(SystemExit, match="--owner NAME"):
        plan_freeze.sign(_freeze(repo), owner=" ")
    first = plan_freeze.sign(_freeze(repo), owner="Owner")
    assert first["kind"] == "GATE0" and plan_freeze.require(_freeze(repo)) == first["record_sha256"]
    with pytest.raises(SystemExit, match="already signed"):
        plan_freeze.sign(_freeze(repo), owner="Owner")


def test_a_changed_file_without_an_amendment_refuses_and_an_amendment_restores(repo: Path) -> None:
    gate0 = plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "builds.py").write_text("BUILDS = ['B1', 'B8']", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after signing"):
        plan_freeze.require(_freeze(repo))
    record = plan_freeze.amend(_freeze(repo), owner="Owner", reason="B8 code written", affects="B8", tests=OK)
    assert record["kind"] == "AMENDMENT" and record["id"] == "A001" and list(record["changed"]) == ["builds.py"]
    assert record["changed"]["builds.py"]["from"] == gate0["files"]["builds.py"] != record["changed"]["builds.py"]["to"]
    assert record["previous_record_sha256"] == gate0["record_sha256"] and record["tests"]["returncode"] == 0
    assert plan_freeze.require(_freeze(repo)) == record["record_sha256"]


def test_the_original_signature_and_every_amendment_are_kept(repo: Path) -> None:
    gate0 = plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "builds.py").write_text("v2", encoding="utf-8")
    one = plan_freeze.amend(_freeze(repo), owner="Owner", reason="r1", affects="B8", tests=OK)
    (repo / "builds.py").write_text("v3", encoding="utf-8")
    two = plan_freeze.amend(_freeze(repo), owner="Owner", reason="r2", affects="B9", tests=OK)
    stored = json.loads(_freeze(repo).read_text(encoding="utf-8"))["records"]
    assert [r["id"] for r in stored] == ["G0", "A001", "A002"] and stored[0] == gate0
    assert plan_freeze.valid_signatures(_freeze(repo)) == {gate0["record_sha256"], one["record_sha256"], two["record_sha256"]}
    assert [h["reason"] for h in plan_freeze.history(_freeze(repo))] == [None, "r1", "r2"]


def test_an_amendment_cannot_change_a_protected_rule_file(repo: Path) -> None:
    plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "rule.py").write_text("MARGIN = 0.05", encoding="utf-8")
    with pytest.raises(SystemExit, match="may not change P1-P3"):
        plan_freeze.amend(_freeze(repo), owner="Owner", reason="wider margin", affects="P3", tests=OK)
    with pytest.raises(SystemExit, match="changed after signing"):
        plan_freeze.require(_freeze(repo))                              # still refused: nothing was recorded


def test_a_protocol_change_is_named_as_one_and_is_impossible_after_a_read(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "builds.py").write_text("v2", encoding="utf-8")
    with pytest.raises(SystemExit, match="no protected file changed"):
        plan_freeze.amend(_freeze(repo), owner="Owner", reason="x", affects="B8", protocol_change=True, tests=OK)
    (repo / "rule.py").write_text("MARGIN = 0.05", encoding="utf-8")
    monkeypatch.setattr(plan_freeze, "reads_of_record", lambda: ["results/v7/preds/B3_anchor384_h7_f1_s42_v70.csv"])
    with pytest.raises(SystemExit, match="cannot change after results"):
        plan_freeze.amend(_freeze(repo), owner="Owner", reason="x", affects="P3", protocol_change=True, tests=OK)
    monkeypatch.setattr(plan_freeze, "reads_of_record", lambda: [])
    record = plan_freeze.amend(_freeze(repo), owner="Owner", reason="margin fixed before any read", affects="P3", protocol_change=True, tests=OK)
    assert record["kind"] == "PROTOCOL_CHANGE" and set(record["changed"]) == {"builds.py", "rule.py"}


def test_an_amendment_needs_owner_reason_scope_a_real_change_and_passing_tests(repo: Path) -> None:
    plan_freeze.sign(_freeze(repo), owner="Owner")
    with pytest.raises(SystemExit, match="nothing to amend"):
        plan_freeze.amend(_freeze(repo), owner="Owner", reason="r", affects="B8", tests=OK)
    (repo / "builds.py").write_text("v2", encoding="utf-8")
    for kwargs, needs in (({"owner": "", "reason": "r", "affects": "B8"}, "--owner"), ({"owner": "O", "reason": "", "affects": "B8"}, "--reason"),
                          ({"owner": "O", "reason": "r", "affects": ""}, "--affects")):
        with pytest.raises(SystemExit, match=needs):
            plan_freeze.amend(_freeze(repo), tests=OK, **kwargs)
    with pytest.raises(SystemExit, match="tests do not pass"):
        plan_freeze.amend(_freeze(repo), owner="Owner", reason="r", affects="B8", tests=FAIL)
    assert len(plan_freeze.load(_freeze(repo))) == 1                    # nothing appended by any refusal


@pytest.mark.parametrize("damage", ["edit_owner", "drop_first", "reorder", "not_json", "old_schema", "empty"])
def test_a_damaged_or_rewritten_record_refuses(repo: Path, damage: str) -> None:
    plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "builds.py").write_text("v2", encoding="utf-8")
    plan_freeze.amend(_freeze(repo), owner="Owner", reason="r", affects="B8", tests=OK)
    data = json.loads(_freeze(repo).read_text(encoding="utf-8"))
    if damage == "edit_owner":
        data["records"][0]["owner"] = "Someone else"
    elif damage == "drop_first":
        data["records"] = data["records"][1:]
    elif damage == "reorder":
        data["records"] = data["records"][::-1]
    elif damage == "old_schema":
        data = {"signed": "2026-11-10T00:00:00", "files": data["records"][-1]["files"], "history": []}
    elif damage == "empty":
        data["records"] = []
    _freeze(repo).write_text("{not json" if damage == "not_json" else json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="inconsistent|not a valid signature record"):
        plan_freeze.require(_freeze(repo))


def test_a_missing_frozen_file_refuses_rather_than_hashing_nothing(repo: Path) -> None:
    plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "builds.py").unlink()
    with pytest.raises(SystemExit, match="missing"):
        plan_freeze.require(_freeze(repo))


def test_the_protected_list_is_the_one_signed_not_the_one_on_disk(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red-team 1a: shortening PROTECTED (or FROZEN) in the code after signing must not let a rule change through."""
    plan_freeze.sign(_freeze(repo), owner="Owner")
    (repo / "rule.py").write_text("MARGIN = 0.05", encoding="utf-8")
    monkeypatch.setattr(plan_freeze, "PROTECTED", ())
    monkeypatch.setattr(plan_freeze, "FROZEN", ("builds.py", "plan.md"))
    with pytest.raises(SystemExit, match="changed after signing: \\['rule.py'\\]"):
        plan_freeze.require(_freeze(repo))                              # a file dropped from FROZEN stays hashed
    with pytest.raises(SystemExit, match="may not change P1-P3"):
        plan_freeze.amend(_freeze(repo), owner="Owner", reason="wider margin", affects="P3", tests=OK)


def test_a_second_gate_0_is_refused_once_v7_has_produced_anything(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red-team 1b: deleting the record and signing again would orphan every run made under the first signature."""
    run = repo / "results" / "v7" / "runs" / "B3_anchor384_h7_f0_s42_v70.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="outputs already exist"):
        plan_freeze.sign(_freeze(repo), owner="Owner")


@pytest.mark.parametrize("cell", ["TBD", "-", "?", "AWAITING_OWNER (see 14)", "**AWAITING_OWNER**", "x", "pending"])
def test_a_placeholder_is_not_a_written_choice(cell: str) -> None:
    table = PLAN_DONE.replace("| vote | (a), 12 Nov |", f"| vote | {cell} |")
    assert plan_freeze.open_decisions(table) == ["2"]


def test_the_tests_are_hashed_and_a_deleted_test_file_fails_the_test_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert set(plan_freeze.TESTS) <= set(plan_freeze.AMENDABLE)
    monkeypatch.setattr(plan_freeze, "REPO", tmp_path)
    result = plan_freeze.run_tests()
    assert result["returncode"] == 1 and "missing" in result["tail"][0]


def test_reads_of_record_cover_every_read_the_code_writes() -> None:
    globs = " ".join(plan_freeze.READ_GLOBS)
    for written in ("results/v7/promotion/*/read*.json", "results/v7/external/*/receipt.json", "results/v7/external/*/predictions.json"):
        assert written in globs


def test_the_command_line_never_signs_or_amends_without_owner_approval(repo: Path) -> None:
    for argv in (["--sign"], ["--amend", "--reason", "r", "--affects", "B8", "--owner", "O"], ["--sign", "--owner", "O"]):
        with pytest.raises(SystemExit, match="owner-approved"):
            plan_freeze.main(argv)


def test_the_real_freeze_list_names_existing_files_and_protects_the_rule() -> None:
    assert not [name for name in plan_freeze.FROZEN if not (plan_freeze.REPO / name).is_file()]
    for rule_file in ("research/v7/promotion.py", "research/v7/promotion_run.py", "research/v7/stats.py", "research/v6/system_reads.py",
                      "research/v6/head8.py", "research/v5/train_v5.py", "research/v7/dark_read.py", "research/v7/data_roles.py",
                      "research/v7/external_read.py", "research/data/overlap_guard.py", "research/v7/plan_freeze.py",
                      "results/v6/leak_sensitivity_rows.csv", "results/v7/v6_reference_manifest.json", "results/v6/anchor_noise_floor.json"):
        assert rule_file in plan_freeze.PROTECTED
    assert not set(plan_freeze.PROTECTED) & set(plan_freeze.AMENDABLE)
