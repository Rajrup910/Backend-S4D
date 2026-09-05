"""Sequential, unattended Session-3 queue: TTA eval -> GPU smoke tests -> full training runs.

Designed to run as a single long-lived background process on a single 8GB GPU (no
concurrent training jobs). Each step's stdout/stderr goes to its own log file under
research/session3_logs/; a step that fails is recorded and the queue moves on rather than
aborting everything that follows -- the point of an unattended run is to come back to as
many finished results as possible, not to lose the whole queue to one bad step.

Queue:
  1. TTA vs. non-TTA comparison (research.tta.evaluate_tta) -- CPU-bound, reads already-
     extracted CSVs from the Session-2 background job.
  2. GPU smoke tests: 1-epoch runs of swinv2_tiny, maxvit_tiny, and the fusion model, to
     catch a crash in minutes rather than discovering it an hour into an unattended run.
  3. Full training, one model at a time, each followed immediately by its single-read
     test evaluation:
       - swinv2_tiny   (Phase 3: transformer paradigm)
       - maxvit_tiny   (Phase 3: transformer paradigm)
       - convnext_tiny + LDAM-DRW loss   (Phase 3: margin loss ablation)
       - convnext_tiny + ASL loss        (Phase 3: margin loss ablation)
       - gated fusion (vision + tabular metadata)

Usage:
    python -m research.run_session3_queue
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from ml.paths import REPO_ROOT, resolve

PYTHON = sys.executable
LOG_DIR = resolve("research/session3_logs")


def run_step(name: str, args: list[str], timeout: int | None = None) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    print(f"\n{'=' * 80}\n[{time.strftime('%H:%M:%S')}] START: {name}\n  cmd: {' '.join(args)}\n{'=' * 80}")

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            result = subprocess.run(
                args, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout
            )
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            log_file.write(f"\n\nTIMED OUT after {timeout}s\n")
            ok = False
        except Exception as exc:  # noqa: BLE001 - queue must survive an unexpected launch failure
            log_file.write(f"\n\nFAILED TO LAUNCH: {exc}\n")
            ok = False

    elapsed = time.time() - started
    status = "OK" if ok else "FAILED"
    print(f"[{time.strftime('%H:%M:%S')}] {status}: {name} ({elapsed / 60:.1f} min). Log: {log_path.relative_to(REPO_ROOT)}")
    return ok


def main() -> int:
    results: dict[str, bool] = {}

    results["tta_evaluate"] = run_step("01_tta_evaluate", [PYTHON, "-m", "research.tta.evaluate_tta"], timeout=600)

    # --- GPU smoke tests (fast fail before committing hours) ---------------------------------
    results["smoke_swinv2_tiny"] = run_step(
        "02_smoke_swinv2_tiny",
        [PYTHON, "-m", "ml.training.train", "--arch", "swinv2_tiny", "--epochs", "1", "--run-name", "smoke_swinv2_tiny"],
        timeout=1800,
    )
    results["smoke_maxvit_tiny"] = run_step(
        "03_smoke_maxvit_tiny",
        [PYTHON, "-m", "ml.training.train", "--arch", "maxvit_tiny", "--epochs", "1", "--run-name", "smoke_maxvit_tiny"],
        timeout=1800,
    )
    results["smoke_fusion"] = run_step(
        "04_smoke_fusion", [PYTHON, "-m", "research.fusion.train_fusion", "--epochs", "1"], timeout=1800
    )

    # --- Full training runs, each followed by its single-read test evaluation ----------------
    if results["smoke_swinv2_tiny"]:
        if run_step("05_train_swinv2_tiny",
                     [PYTHON, "-m", "ml.training.train", "--arch", "swinv2_tiny", "--epochs", "20",
                      "--run-name", "swinv2_tiny_phase3"], timeout=5 * 3600):
            run_step("06_eval_swinv2_tiny",
                      [PYTHON, "-m", "ml.evaluation.evaluate", "--checkpoint", "ml/checkpoints/swinv2_tiny_best.pt",
                       "--split", "test"], timeout=600)
    else:
        print("Skipping full swinv2_tiny run: smoke test failed.")

    if results["smoke_maxvit_tiny"]:
        if run_step("07_train_maxvit_tiny",
                     [PYTHON, "-m", "ml.training.train", "--arch", "maxvit_tiny", "--epochs", "20",
                      "--run-name", "maxvit_tiny_phase3"], timeout=5 * 3600):
            run_step("08_eval_maxvit_tiny",
                      [PYTHON, "-m", "ml.evaluation.evaluate", "--checkpoint", "ml/checkpoints/maxvit_tiny_best.pt",
                       "--split", "test"], timeout=600)
    else:
        print("Skipping full maxvit_tiny run: smoke test failed.")

    if run_step("09_train_convnext_tiny_ldam_drw",
                [PYTHON, "-m", "ml.training.train", "--arch", "convnext_tiny", "--loss", "ldam_drw",
                 "--epochs", "30", "--checkpoint-tag", "ldam_drw", "--run-name", "convnext_tiny_ldam_drw"],
                timeout=5 * 3600):
        run_step("10_eval_convnext_tiny_ldam_drw",
                  [PYTHON, "-m", "ml.evaluation.evaluate",
                   "--checkpoint", "ml/checkpoints/convnext_tiny-ldam_drw_best.pt",
                   "--split", "test", "--out-dir", "ml/results/convnext_tiny_ldam_drw"], timeout=600)

    if run_step("11_train_convnext_tiny_asl",
                [PYTHON, "-m", "ml.training.train", "--arch", "convnext_tiny", "--loss", "asl",
                 "--epochs", "30", "--checkpoint-tag", "asl", "--run-name", "convnext_tiny_asl"],
                timeout=5 * 3600):
        run_step("12_eval_convnext_tiny_asl",
                  [PYTHON, "-m", "ml.evaluation.evaluate",
                   "--checkpoint", "ml/checkpoints/convnext_tiny-asl_best.pt",
                   "--split", "test", "--out-dir", "ml/results/convnext_tiny_asl"], timeout=600)

    if results["smoke_fusion"]:
        if run_step("13_train_fusion", [PYTHON, "-m", "research.fusion.train_fusion", "--epochs", "20"], timeout=5 * 3600):
            run_step("14_eval_fusion", [PYTHON, "-m", "research.fusion.evaluate_fusion"], timeout=600)
    else:
        print("Skipping full fusion run: smoke test failed.")

    print(f"\n{'=' * 80}\nSession 3 queue finished. Step results:\n{'=' * 80}")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
