"""M1's unattended night: every step after the first candidate's training, with decisions by rule and safe stops.

    python -m research.v7.m1.chain                # waits for the running ConvNeXt-Femto training, then runs the rest
    python -m research.v7.m1.chain --dry-run      # prints the plan and the checks; runs nothing

Steps: (1) wait for Femto's summary; (2) train MobileNetV4-Conv-Small; (3) score each finished candidate (`bench candidate`);
(4) choose the matcher by the protocol's adoption rule: an adopted candidate (the higher primary TPR if both), otherwise the best
untrained baseline; (5) scan the master table with it (`scan`: links for the gate and the MIDAS pixel audit); (6) LightGlue as
a verifier, alone on the GPU; (7) `verify`: known copies, unseen re-photographs, a review sheet. Nothing here edits the gate, the master table or any frozen file.
Stops before a step: the PAUSE file (results/v7/m1/PAUSE); C: under 5 GB; commit headroom under 3 GB; a failed earlier step.
A step that fails is not retried. Log: results/v7/m1/logs/chain.log.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.v7.m1 import data

PY = sys.executable
LOG = data.OUT / "logs" / "chain.log"
PAUSE = data.OUT / "PAUSE"
FEMTO = "convnext_femto.d1_in1k"
MOBILE = "mobilenetv4_conv_small.e2400_r224_in1k"
MIN_DISK_GB, MIN_COMMIT_GB = 5.0, 3.0


def summary_path(backbone: str) -> Path:
    return data.OUT / f"m1_{backbone.split('.')[0]}_ms_s42.json"


def checkpoint(backbone: str) -> str:
    return f"ml/checkpoints/m1/m1_{backbone.split('.')[0]}_ms_s42.pt"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def commit_free_gb() -> float:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"], capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip()) / 1024 / 1024
    except Exception:          # unreadable counts as unsafe
        return 0.0


def checks() -> str:
    if PAUSE.exists():
        return "PAUSE file present"
    free = shutil.disk_usage("C:\\").free / 1e9
    if free < MIN_DISK_GB:
        return f"C: has {free:.1f} GB free, under {MIN_DISK_GB}"
    commit = commit_free_gb()
    if commit < MIN_COMMIT_GB:
        return f"commit headroom {commit:.1f} GB, under {MIN_COMMIT_GB}"
    return ""


def step(name: str, args: list[str], dry: bool) -> bool:
    if dry:
        log(f"[dry] {name}: python {' '.join(args)}")
        return True
    problem = checks()
    if problem:
        log(f"STOP before {name}: {problem}")
        return False
    log(f"start {name}")
    t = time.time()
    with open(data.OUT / "logs" / f"{name}.log", "w", encoding="utf-8") as out:
        code = subprocess.run([PY, "-u", *args], cwd=data.REPO, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"{name}: exit {code} after {(time.time() - t) / 60:.0f} min")
    return code == 0


def choose(candidates: list[str]) -> str:
    """The protocol's rule: an adopted candidate, the higher primary TPR if more than one; else the best untrained baseline."""
    adopted = []
    for backbone in candidates:
        path = data.OUT / f"candidate_m1_{backbone.split('.')[0]}_ms_s42.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["adopted"]:
                adopted.append((record["candidate_tpr"], checkpoint(backbone)))
    if adopted:
        return max(adopted)[1]
    return json.loads((data.OUT / "baselines.json").read_text(encoding="utf-8"))["best_untrained_baseline"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=6.0, help="give up waiting for Femto after this long")
    args = parser.parse_args(argv)
    dry = args.dry_run
    log(f"=== M1 chain {'(dry run)' if dry else ''} ===")
    if not dry:
        deadline = time.time() + args.wait_hours * 3600
        while not summary_path(FEMTO).is_file():
            if time.time() > deadline:
                log("STOP: the Femto training summary never appeared")
                return 1
            time.sleep(60)
        log(f"Femto finished: {summary_path(FEMTO).name}")
    if not (data.OUT / "baselines.json").is_file() and not dry:
        log("STOP: baselines.json is missing; run `bench baselines` first")
        return 1
    trained = [FEMTO]
    if step("train_mobilenetv4", ["-m", "research.v7.m1.train", "--backbone", MOBILE, "--epochs", "30", "--num-workers", "3"], dry):
        trained.append(MOBILE)
    scored = [b for b in trained if step(f"candidate_{b.split('.')[0]}", ["-m", "research.v7.m1.bench", "candidate", "--checkpoint", checkpoint(b)], dry)]
    method = choose(scored) if not dry else "<chosen by rule>"
    log(f"matcher chosen by the adoption rule: {method}")
    if not step("scan", ["-m", "research.v7.m1.scan", "--method", method], dry):
        return 1
    step("verify", ["-m", "research.v7.m1.verify"], dry)
    step("lightglue", ["-m", "research.v7.m1.bench", "baselines", "--lightglue"], dry)
    log("=== M1 chain finished: read results/v7/m1/ ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
