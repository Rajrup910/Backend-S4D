"""S54 -- the read-once guard for the reserved cohort, and the host checks every S54 stage runs first.

The reserved split (4,733 BCN-20000 + MSKCC images, 104 under-40 escalating lesions) is the only
cohort V4 has not selected on. S52 declared that the under-40 endpoint is read from it **once**, at
S54. S9's `research/session9/receipt.py` made the same promise about the HAM test split enforceable
rather than aspirational; this module is the same mechanism pointed at a different split.

What "once" means here
----------------------

S54 touches the reserved rows in three **stages** of one plan, like S9's `tables` + `attribution`:

    v1_topup   score the 146 `scc` images with the six frozen V1 checkpoints
    infer      score the reserved cohort with the six V4 checkpoints
    gate       read the frozen predictions and emit the outcome

Each stage is recorded in `results/v4/s54/reserved_receipt.json`, an **append-only** file:

* a stage that has **completed** refuses to run again unless `--rerun-reason` is given, and the
  reason is kept permanently beside the new execution;
* a stage that **started but did not complete** (a crash, a power cut, Windows error 1455) is
  resumed with `--resume`. Resuming is not a rerun: items already written are skipped by hash, so a
  resumed execution scores each checkpoint at most once;
* the receipt pins the plan sha256. If the plan file changes after any reserved read, every stage
  refuses. A plan edited after the data was seen is not a pre-registration.

Host checks, from the 2026-09-16 failures
-----------------------------------------

Each check lives in this script, not in a prompt (see the unattended-runs lesson in CHANGELOG §S53):

* **one copy at a time** -- an OS file lock, released by the kernel when the process dies, so a
  crashed run never leaves a stale lock behind;
* **no training on the GPU** -- refuses while any `research.v4.train_v4`, `run_block3.ps1` or
  `run_morning.ps1` process exists. S54 runs *after* Block 3, never beside it;
* **Block 3 banked** -- every required arm x seed has its end-of-run JSON (written only after the
  last epoch), so a `_last.pt` still being overwritten epoch by epoch is never scored;
* **disk >= 5 GB** and **system commit headroom** -- a full disk pins the pagefile and surfaces as
  error 1455; a torch import is a large commit charge, and a second one next to a training run can
  kill that run.

This module imports no torch, so the gate's statistics can be checked without a GPU process.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
S54_DIR = REPO_ROOT / "results" / "v4" / "s54"
SMOKE_DIR = S54_DIR / "smoke"
RECEIPT_PATH = S54_DIR / "reserved_receipt.json"
LOCK_PATH = S54_DIR / ".s54.lock"
RUN_DIR = REPO_ROOT / "results" / "v4" / "recipe_runs"

STAGES = ("v1_topup", "infer", "gate")
MIN_FREE_DISK_GB = 5.0
#: The headroom is read *inside* the scoring process, after its own torch import has already been
#: charged (measured 2026-09-16: 5.73 GB bare -> 3.13 after importing s54_topup_v1 -> 2.88 after
#: CUDA init). What must still fit is the model, the batches and the CUDA context, plus every
#: dataloader worker -- each a spawned process that repeats the full import, measured at 2.66 GB.
#: A fixed 4 GB floor let two workers (~5.3 GB) start against ~3 GB and invited error 1455.
TORCH_PROCESS_REMAINING_GB = 1.5
WORKER_COMMIT_GB = 2.7


def torch_commit_gb(num_workers: int) -> float:
    """Headroom a torch stage needs at preflight, given how many workers it will spawn."""
    return TORCH_PROCESS_REMAINING_GB + max(0, num_workers) * WORKER_COMMIT_GB


#: The CI-only helpers still import torch transitively (`research.external.frozen_params` ->
#: `research.calibration.methods`), but spawn no workers.
MIN_COMMIT_HEADROOM_GB_GATE = 2.5
TRAINING_MARKERS = ("research.v4.train_v4", "run_block3.ps1", "run_morning.ps1")

_LOCK_HANDLE = None


class ReservedReadRefused(RuntimeError):
    """A guard declined to let this stage read the reserved cohort."""


# --------------------------------------------------------------------- hashing
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------- host checks
def acquire_lock() -> None:
    """Exclusive, non-blocking, released by the OS on process exit -- never stale."""
    global _LOCK_HANDLE
    if _LOCK_HANDLE is not None:
        return
    S54_DIR.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ReservedReadRefused(
            f"another S54 process holds {LOCK_PATH.relative_to(REPO_ROOT)}; one copy at a time"
        ) from error
    _LOCK_HANDLE = handle


def free_disk_gb() -> float:
    return shutil.disk_usage(REPO_ROOT).free / 1e9


def require_disk(minimum_gb: float = MIN_FREE_DISK_GB) -> None:
    free = free_disk_gb()
    if free < minimum_gb:
        raise ReservedReadRefused(
            f"{free:.2f} GB free on the repo drive, need {minimum_gb:.1f} GB. A full disk pins the "
            f"pagefile and fails later as Windows error 1455. Free space first.")


def commit_headroom_gb() -> float | None:
    """System commit limit minus commit charge, in GB. None off Windows."""
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPageFile / 1e9


def require_commit(minimum_gb: float) -> None:
    headroom = commit_headroom_gb()
    if headroom is not None and headroom < minimum_gb:
        raise ReservedReadRefused(
            f"system commit headroom {headroom:.2f} GB < {minimum_gb:.1f} GB. Starting here risks "
            f"error 1455 in this process or in any training run beside it. Each dataloader worker "
            f"needs ~{WORKER_COMMIT_GB} GB: use --num-workers 0, or close other programs.")


def running_training() -> list[str]:
    """Command lines of any process that is (or is about to be) training on the GPU."""
    if os.name == "nt":
        command = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process | "
                   "Where-Object { $_.CommandLine } | ForEach-Object { $_.CommandLine }"]
    else:
        command = ["ps", "-eo", "args"]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ReservedReadRefused(f"could not list processes to check for training: {error}")
    # The listing command itself names no marker, so it cannot match itself.
    return [line.strip()[:160] for line in out.splitlines()
            if any(marker in line for marker in TRAINING_MARKERS)]


def require_no_training() -> None:
    hits = running_training()
    if hits:
        listing = "\n  ".join(hits[:6])
        raise ReservedReadRefused(
            "training (or a queued training script) is running -- S54 runs after Block 3, never "
            f"beside it:\n  {listing}")


def run_summary_path(run_id: str) -> Path:
    return RUN_DIR / f"{run_id}.json"


def require_banked(run_ids: list[str]) -> dict[str, dict]:
    """Each run's end-of-run JSON, which train_v4 writes only after the final epoch."""
    summaries, missing = {}, []
    for run_id in run_ids:
        path = run_summary_path(run_id)
        if not path.is_file():
            missing.append(run_id)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("smoke"):
            raise ReservedReadRefused(f"{path.name} is a smoke summary, not a banked run")
        summaries[run_id] = summary
    if missing:
        raise ReservedReadRefused(
            f"not banked yet (no end-of-run JSON in results/v4/recipe_runs/): {missing}. "
            f"Wait for Block 3 to finish.")
    return summaries


def preflight(*, commit_gb: float, training_check: bool = True) -> None:
    acquire_lock()
    require_disk()
    require_commit(commit_gb)
    if training_check:
        require_no_training()


# --------------------------------------------------------------------- receipt
class ReservedReceipt:
    """Append-only record of every reserved-cohort read S54 makes."""

    def __init__(self, plan_sha256: str, path: Path = RECEIPT_PATH) -> None:
        if not plan_sha256:
            raise ReservedReadRefused("the S54 plan is not frozen; run s54_gate --freeze-plan")
        self.path = path
        self.plan_sha256 = plan_sha256
        self.data = self._load()
        self.execution: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"cohort": "manifest_v4 split=reserved", "plan_sha256": self.plan_sha256,
                    "created_at": now(), "executions": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("plan_sha256") != self.plan_sha256:
            raise ReservedReadRefused(
                f"the plan changed after the reserved cohort was read: receipt pins "
                f"{data.get('plan_sha256')}, plan on disk is {self.plan_sha256}. Restore the plan.")
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, indent=2)
        os.replace(tmp, self.path)

    def executions(self, stage: str) -> list[dict[str, Any]]:
        return [e for e in self.data["executions"] if e["stage"] == stage]

    def latest_completed(self, stage: str) -> dict[str, Any] | None:
        done = [e for e in self.executions(stage) if e["status"] == "completed"]
        return done[-1] if done else None

    def open_stage(self, stage: str, rerun_reason: str | None, resume: bool) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        history = self.executions(stage)
        open_runs = [e for e in history if e["status"] == "started"]
        completed = [e for e in history if e["status"] == "completed"]

        if open_runs:
            if not resume:
                raise ReservedReadRefused(
                    f"stage {stage!r} has an unfinished execution from {open_runs[-1]['started_at']}."
                    f" Pass --resume to finish it (already-scored items are skipped; not a rerun).")
            self.execution = open_runs[-1]
            self.execution.setdefault("resumed_at", []).append(now())
            self._save()
            return self.execution
        if completed and not rerun_reason:
            raise ReservedReadRefused(
                f"stage {stage!r} already completed on {completed[-1]['completed_at']}. The reserved "
                f"cohort is read once. A repeat needs --rerun-reason, which the receipt keeps.")
        if resume and not rerun_reason and not completed:
            print(f"[receipt] --resume given but no unfinished {stage!r} execution; starting fresh")

        self.execution = {"stage": stage, "execution": len(history) + 1, "status": "started",
                          "started_at": now(), "completed_at": None,
                          "rerun_reason": rerun_reason, "git_head": git_head(),
                          "argv": sys.argv[1:], "items": {}}
        self.data["executions"].append(self.execution)
        self._save()
        return self.execution

    def items(self) -> dict[str, dict[str, Any]]:
        return self.execution["items"] if self.execution else {}

    def record_item(self, key: str, item: dict[str, Any]) -> None:
        self.execution["items"][key] = {**item, "recorded_at": now()}
        self._save()

    def complete(self, summary: dict[str, Any]) -> None:
        self.execution["status"] = "completed"
        self.execution["completed_at"] = now()
        self.execution["summary"] = summary
        self.data["n_executions"] = {s: len([e for e in self.executions(s)
                                             if e["status"] == "completed"]) for s in STAGES}
        self._save()


def verify_items(items: dict[str, dict[str, Any]], path_key: str, sha_key: str) -> None:
    """Every artefact a completed stage wrote still has the bytes it had when it was written."""
    for key, item in items.items():
        path = REPO_ROOT / item[path_key]
        if not path.is_file():
            raise ReservedReadRefused(f"{key}: {item[path_key]} is missing")
        if sha256_file(path) != item[sha_key]:
            raise ReservedReadRefused(f"{key}: {item[path_key]} changed after it was frozen")
