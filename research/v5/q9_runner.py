"""Resumable runner for results/v5/queue_Q9.txt (TTA benchmark, 15 composite TTA scorings, V1 inference).

`scripts/run_v5_queue.ps1` only dispatches `train_v5` / `train_v4` lines, and Q9's remaining work is
inference, so this is its counterpart for lines with a `tta ` or `v1 ` prefix. Same contract: one job per
line, `#` comments, a finished job is skipped, an interrupted block resumes by running the same command.

    python -m research.v5.q9_runner --dry-run     # print each job and whether it is already done
    python -m research.v5.q9_runner               # run what is not done, one at a time

  tta <args of research.v5.tta_v5>   done when results/v5/preds_tta/<run_id>_tta.csv exists
                                     (`--benchmark`: when results/v5/benchmarks/tta_benchmark.json exists)
  v1  <args of research.v5.v1_infer> done when all six results/v5/preds_v1 files exist
                                     (`--benchmark`: when results/v5/benchmarks/v1_infer_benchmark.json exists)

Rules enforced (runsheet 11.1): C1 one job at a time (a lock file; also refuses if a train_v4/train_v5
process is alive), C2 C: >= 10 GB free before each job, C7 `results/v5/PAUSE` stops the queue between jobs.
Commit headroom (C2, >= 6 GB) is the owner's check with the status command in section 11.1. A failed job is
retried once with one worker, recorded, and the queue moves on. The writers are atomic (`.tmp` then rename),
so an interrupted job never leaves a file that looks finished.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V5 = REPO_ROOT / "results" / "v5"
MODULES = {"tta": "research.v5.tta_v5", "v1": "research.v5.v1_infer"}
MIN_DISK_GB = 10.0
V1_ARCHS = ("convnext_small", "convnext_tiny", "densenet121", "efficientnet_b0", "efficientnet_b3",
            "resnet50")


def parse_queue(path: Path) -> list[tuple[str, list[str]]]:
    jobs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        kind, *tokens = line.split()
        if kind not in MODULES:
            raise SystemExit(f"unknown job kind {kind!r} in {path.name}: {line}")
        jobs.append((kind, tokens))
    return jobs


def arg_of(tokens: list[str], name: str, default: str | None = None) -> str | None:
    if name in tokens and tokens.index(name) + 1 < len(tokens):
        return tokens[tokens.index(name) + 1]
    return default


def is_done(kind: str, tokens: list[str]) -> bool:
    if "--benchmark" in tokens:
        return (V5 / "benchmarks" / ("tta_benchmark.json" if kind == "tta" else "v1_infer_benchmark.json")).is_file()
    if kind == "tta":
        stem = Path(arg_of(tokens, "--checkpoint")).stem  # convnext_tiny-v5_<run_id>_last
        run_id = stem.removeprefix("convnext_tiny-v5_").removesuffix("_last")
        return (V5 / "preds_tta" / f"{run_id}_tta.csv").is_file()
    folds = [int(x) for x in tokens[tokens.index("--folds") + 1:] if x.isdigit()] if "--folds" in tokens else [1, 2, 3, 4]
    rows = arg_of(tokens, "--rows", "nonham")
    return all((V5 / "preds_v1" / f"{a}_f{folds[0]}-{folds[-1]}_{rows}_tta.csv").is_file() for a in V1_ARCHS)


def training_alive() -> bool:
    if os.name != "nt":
        return False
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -match 'research\\.v[45]\\.train_v[45]' } | "
                          "Measure-Object | Select-Object -ExpandProperty Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    log_dir = V5 / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "queue_Q9.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, default=V5 / "queue_Q9.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    jobs = parse_queue(args.queue)
    if args.dry_run:
        for kind, tokens in jobs:
            print(f"{'done ' if is_done(kind, tokens) else 'todo '} {kind} {' '.join(tokens)}")
        return 0
    lock = V5 / "Q9.lock"
    if lock.exists():
        raise SystemExit(f"{lock} exists: another Q9 runner is active (delete it if that run died)")
    if training_alive():
        raise SystemExit("a train_v4/train_v5 process is alive: C1 allows one GPU job at a time")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        for kind, tokens in jobs:
            if is_done(kind, tokens):
                log(f"skip (done): {kind} {' '.join(tokens)}")
                continue
            if (V5 / "PAUSE").exists():
                log("results/v5/PAUSE present: stopping between jobs (C7)")
                return 0
            free = shutil.disk_usage(REPO_ROOT).free / 1e9
            if free < MIN_DISK_GB:
                log(f"C2: C: has {free:.1f} GB free (< {MIN_DISK_GB}); stopping")
                return 1
            command = [sys.executable, "-m", MODULES[kind], *tokens]
            log("start: " + " ".join(command[2:]))
            code = subprocess.run(command, cwd=REPO_ROOT).returncode
            if code != 0:
                log(f"FAILED (exit {code}); retrying once with one worker")
                code = subprocess.run([*command, "--num-workers", "1"], cwd=REPO_ROOT).returncode
            log(f"{'done' if code == 0 else 'FAILED AGAIN (exit %d), moving on' % code}: {kind} {' '.join(tokens)}")
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
