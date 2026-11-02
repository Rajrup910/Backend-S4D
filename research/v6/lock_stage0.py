"""Stage-0 lock: hashes everything the V6-0 noise-floor run depends on (docs/V6_RUNSHEET.md section 0.8d).

    python -m research.v6.lock_stage0 --create    # once, before any v6nf output exists
    python -m research.v6.lock_stage0 --verify    # exit 0 if nothing changed, 1 with the list otherwise

This is NOT the V6 plan freeze (results/v6/v6_plan_freeze.json, section 0.8 item 11): it covers only tonight's run and the
rule that turns its seeds into bars. Standard library only, so it never costs a torch import. Text files are hashed with
line endings normalised, so a git checkout cannot fake a change.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "results" / "v6" / "v6_stage0_lock.json"
RUNSHEET = REPO / "docs" / "V6_RUNSHEET.md"
SEEDS_BANKED, SEEDS_NEW = (42, 43, 44), (45, 46, 47)
ARMS = ("youngdata", "control")
TEXT_SUFFIXES = {".py", ".ps1", ".txt", ".csv", ".json", ".md"}
PACKAGES = ("torch", "timm", "torchvision", "numpy", "pandas", "scikit-learn")
#: The queue, its runner and the benchmark tools.
RUN_FILES = ("results/v5/queue_V6N0.txt", "scripts/run_v5_queue.ps1", "scripts/run_v6_benchmarks.ps1",
             "scripts/gpu_benchmark.py")
#: Partition, labels and the train-only rows.
DATA_FILES = ("ml/data/manifest_v4.csv", "results/v5/young_data/extra_train.csv",
              "results/v4/kfold/fold_assignments.csv", "ml/configs/class_mapping.json")
#: Modules whose import closure is the trainer and the bar computation.
ROOT_MODULES = ("research.v5.train_v5", "research.v5.screen_gate", "research.v6.anchor_noise_floor")
RULES = {
    "seeds": {"banked_tag_v5scr": list(SEEDS_BANKED), "new_tag_v6nf": list(SEEDS_NEW)},
    "screen_bars_come_from": "the six youngdata (anchor recipe) seeds; the six control seeds are reported for comparison only",
    "pair_sd": "root mean square of all pairwise seed differences, fold 0",
    "bar": "z80 * pair_sd / sqrt(k), z80 = 0.8416212335729143",
    "k": {"bar_k1": "V6-3 single-seed gate and the E-primary all-age pAUC", "bar_k3": "three-seed screens (V6-1, V6-2)"},
    "gate_a": "under-40 pAUC delta over the V5 system: point >= +0.050 and 95% CI lower bound > 0",
    "e_primary": "all-age pAUC delta above bar_k1 and Macro-F1 delta > 0 with 95% CI lower bound > 0, folds 1-4, TTA on both sides",
    "void_run": "any non-finite training loss voids a run; rerun once under the same seed, never scored (section A0.9)",
}
NOT_LOCKED = ["V6-1 onward", "member builder and member recipes", "exact DINOv2 and SigLIP 2 values", "class maps",
              "data downloads and image-hash audit", "the rest of Part 0, until the plan freeze"]


def normalised_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def import_closure(roots: tuple[str, ...], repo: Path = REPO) -> list[str]:
    """Repo files reachable by import from `roots`, by parsing, so it needs no torch."""
    seen: set[str] = set()

    def walk(module: str) -> None:
        path = repo / (module.replace(".", "/") + ".py")
        if module in seen or not path.exists():
            return
        seen.add(module)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("research", "ml"):
                walk(node.module)
                for alias in node.names:
                    walk(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in ("research", "ml"):
                        walk(alias.name)

    for root in roots:
        walk(root)
    return sorted(m.replace(".", "/") + ".py" for m in seen)


def banked_predictions() -> list[str]:
    return [f"results/v5/preds/{arm}_f0_s{seed}_in22k_v5scr.csv" for arm in ARMS for seed in SEEDS_BANKED]


def locked_paths() -> list[str]:
    return sorted({*RUN_FILES, *DATA_FILES, *import_closure(ROOT_MODULES), *banked_predictions()})


def new_seed_outputs(repo: Path = REPO) -> list[str]:
    found = []
    for folder in ("results/v5/preds", "results/v5/runs"):
        directory = repo / folder
        if directory.exists():
            found += [f"{folder}/{p.name}" for p in directory.glob("*_v6nf*")]
    return found


def compare(files: dict[str, str], repo: Path = REPO) -> list[str]:
    """One line per locked file that is missing or changed; empty means intact."""
    problems = []
    for name, expected in files.items():
        path = repo / name
        if not path.exists():
            problems.append(f"missing: {name}")
        elif normalised_sha256(path) != expected:
            problems.append(f"changed: {name}")
    return problems


def versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in PACKAGES}


def part0_sha256() -> str:
    text = RUNSHEET.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.search(r"^# Part 0 .*?(?=^# Part A )", text, flags=re.M | re.S)
    if not match:
        raise SystemExit("Part 0 not found in the runsheet")
    return hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()


def create() -> int:
    if LOCK.exists():
        raise SystemExit(f"{LOCK.name} already exists; a lock is never overwritten. Declare a dated amendment instead.")
    outputs = new_seed_outputs()
    if outputs:
        raise SystemExit(f"refusing: {len(outputs)} v6nf outputs already exist, so this could not precede the seeds")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    lock = {"stage": "V6-0 benchmarks and noise-floor seeds", "status": "LOCKED",
            "is_the_v6_plan_freeze": False,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head_at_creation": head,
            "v6nf_outputs_at_creation": 0, "rules": RULES, "not_locked": NOT_LOCKED,
            "part0_snapshot_sha256": part0_sha256(),
            "part0_snapshot_note": "a record of the plan at this moment; Part 0 may change by dated revision until the plan freeze, so --verify does not check it",
            "environment": versions(), "files": {name: normalised_sha256(REPO / name) for name in locked_paths()}}
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"locked {len(lock['files'])} files -> {LOCK.relative_to(REPO).as_posix()}")
    return 0


def problems() -> list[str]:
    if not LOCK.exists():
        return [f"no lock at {LOCK.name}"]
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    found = compare(lock["files"])
    current = versions()
    found += [f"package {k}: locked {v}, now {current[k]}" for k, v in lock["environment"].items() if current[k] != v]
    return found


def require() -> None:
    """Called by the bar script: no bars from a changed recipe or rule."""
    found = problems()
    if found:
        raise SystemExit("stage-0 lock broken:\n  " + "\n  ".join(found))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.create:
        return create()
    found = problems()
    for line in found:
        print(line)
    if found:
        print(f"LOCK BROKEN: {len(found)} problem(s). Do not launch or read.")
        return 1
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    print(f"lock intact: {len(lock['files'])} files, environment unchanged; v6nf outputs now: {len(new_seed_outputs())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
