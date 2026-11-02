"""CPU tests for the stage-0 lock (research/v6/lock_stage0.py). No torch, no results read."""

from __future__ import annotations

from pathlib import Path

from research.v6 import lock_stage0 as lock


def test_line_endings_do_not_change_the_hash(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "a.csv", tmp_path / "b.csv"
    lf.write_bytes(b"x,y\n1,2\n")
    crlf.write_bytes(b"x,y\r\n1,2\r\n")
    assert lock.normalised_sha256(lf) == lock.normalised_sha256(crlf)


def test_binary_files_are_hashed_as_they_are(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "a.bin", tmp_path / "b.bin"
    lf.write_bytes(b"x\n")
    crlf.write_bytes(b"x\r\n")
    assert lock.normalised_sha256(lf) != lock.normalised_sha256(crlf)


def test_compare_catches_a_change_and_a_missing_file(tmp_path: Path) -> None:
    kept, edited, gone = tmp_path / "kept.py", tmp_path / "edited.py", tmp_path / "gone.py"
    for path in (kept, edited, gone):
        path.write_text("a = 1\n", encoding="utf-8")
    files = {p.name: lock.normalised_sha256(p) for p in (kept, edited, gone)}
    assert lock.compare(files, tmp_path) == []
    edited.write_text("a = 2\n", encoding="utf-8")
    gone.unlink()
    assert sorted(lock.compare(files, tmp_path)) == ["changed: edited.py", "missing: gone.py"]


def test_import_closure_follows_imports_and_ignores_outside_packages(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "a.py").write_text("import json\nfrom research import b\n", encoding="utf-8")
    (tmp_path / "research" / "b.py").write_text("from research.c import thing\n", encoding="utf-8")
    (tmp_path / "research" / "c.py").write_text("thing = 1\n", encoding="utf-8")
    (tmp_path / "research" / "unused.py").write_text("", encoding="utf-8")
    assert lock.import_closure(("research.a",), tmp_path) == ["research/a.py", "research/b.py", "research/c.py"]


def test_new_seed_outputs_are_detected(tmp_path: Path) -> None:
    preds = tmp_path / "results" / "v5" / "preds"
    preds.mkdir(parents=True)
    (preds / "youngdata_f0_s42_in22k_v5scr.csv").write_text("", encoding="utf-8")
    assert lock.new_seed_outputs(tmp_path) == []
    (preds / "youngdata_f0_s45_in22k_v6nf.csv").write_text("", encoding="utf-8")
    assert lock.new_seed_outputs(tmp_path) == ["results/v5/preds/youngdata_f0_s45_in22k_v6nf.csv"]


def test_the_lock_covers_the_trainer_the_rule_and_the_data() -> None:
    paths = lock.locked_paths()
    for needed in ("research/v5/train_v5.py", "research/v6/anchor_noise_floor.py", "research/v5/screen_gate.py",
                   "research/v4/s71_kfold.py", "results/v5/queue_V6N0.txt", "results/v5/young_data/extra_train.csv",
                   "results/v5/preds/youngdata_f0_s42_in22k_v5scr.csv"):
        assert needed in paths
