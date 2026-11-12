"""Freeze audit (V7 plan section 14): the change ledger. Hashes every file the audit wrote or edited and records scope checks.

    python scripts/v7_freeze_audit_manifest.py

Writes `results/v7/freeze_audit/manifest.json`. Reads file bytes and Git metadata only; opens no prediction and no label.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "results" / "v7" / "freeze_audit"
WRITTEN = ("research/v7/plan_freeze.py", "research/v7/promotion.py", "research/v7/stats.py", "research/v7/promotion_run.py",
           "research/v7/lr_range.py", "research/v7/builds.py", "research/v7/train_v7.py", "research/v7/data_roles.py",
           "scripts/audit_v7_final_verification.py", "scripts/v7_p2_coverage.py", "scripts/v7_freeze_audit_manifest.py",
           "tests/test_v7_freeze.py", "tests/test_v7_promotion_run.py", "tests/test_v7_lr_range.py", "tests/test_v7_decisions.py",
           "tests/test_v7_trainer.py", "tests/test_v7_data_gate.py", "results/v7/p2_coverage.json", "results/v7/p2_coverage_trials.npz",
           "results/v7/data_roles_v7.json", "docs/v7_final_revision/V7_FINAL_MASTER_RESEARCH_PLAN.md", "docs/LEAK_DISCLOSURE.md",
           "docs/V6_GLOBAL_BENCHMARK_REVIEW.md", "README.md", "CHANGELOG.md", "docs/v7_final_revision/README.md")
NEW = ("research/v7/promotion_run.py", "research/v7/lr_range.py", "scripts/v7_p2_coverage.py", "scripts/v7_freeze_audit_manifest.py",
       "tests/test_v7_freeze.py", "tests/test_v7_promotion_run.py", "tests/test_v7_lr_range.py", "results/v7/p2_coverage.json",
       "results/v7/p2_coverage_trials.npz")
MUST_NOT_EXIST = ("results/v7/plan_freeze.json", "results/v7/preds", "results/v7/preds_tta", "results/v7/runs", "ml/checkpoints/v7",
                  "results/v7/promotion", "results/v7/lr_range")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    banners = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "docs" / "v7_final_revision").glob("V7_*.md")
                     if "Precedence (11 Nov 2026)" in p.read_text(encoding="utf-8"))
    start = {}
    for line in (AUDIT / "baseline" / "sha256_start.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            start[name.lstrip("*").strip()] = digest
    files = {}
    for name in sorted({*WRITTEN, *banners}):
        path = REPO / name
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                       "sha256_at_session_start": start.get(name), "created_this_session": name in NEW}
    exists = {name: (REPO / name).exists() for name in MUST_NOT_EXIST}
    manifest = {
        "what": "V7 freeze audit, 11 Nov 2026: every file the audit wrote or edited, with its SHA-256 from the bytes on disk",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "git_staged": git("diff", "--cached", "--name-only") or None, "python": platform.python_version(), "files": files,
        "scope": {"must_not_exist": exists, "violations": [n for n, e in exists.items() if e],
                  "statement": "no training, no GPU job, no V7 prediction, no ham_test / HIBA-outcome / MIDAS-label read, no Gate 0 signature, "
                               "no commit, no push"},
        "baseline": "results/v7/freeze_audit/baseline/ (git status, HEAD and SHA-256 at session start)",
        "raw_outputs": sorted(p.relative_to(REPO).as_posix() for p in (AUDIT / "raw").glob("*")),
        "workstream_reports": sorted(p.relative_to(REPO).as_posix() for p in (AUDIT / "workstreams").glob("*.md")),
    }
    (AUDIT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(files)} files hashed; HEAD {manifest['git_head'][:7]}; staged: {manifest['git_staged']}; scope violations: {manifest['scope']['violations']}")
    return 1 if manifest["scope"]["violations"] or manifest["git_staged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
