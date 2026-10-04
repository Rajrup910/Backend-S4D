"""E1 -- adopt and hash the V5 runsheet, Amendment 02 and the DRE (docs/V5_RUNSHEET.md section 1).

Owner action. Two modes:

    python -m research.v5.adopt_e1 --verify
        Read-only. Re-verifies every sha256 already in results/v5/v5_plan_freeze.json
        (frozen_artifacts, synced_external_locations, amendments[*]) and prints the three hashes the
        adoption would record. Writes nothing.

    python -m research.v5.adopt_e1 --adopt --decisions <file.json> --by "owner (Rajrup910)"
        Appends amendments[1] with the three paths and sha256, the adoption timestamp,
        adopted_before_first_v5_gpu_run, and the per-item decisions. Refuses if amendments[1]
        already exists, if any existing hash fails, or if the decisions file is missing items.
        The decisions file maps item id -> "ACCEPTED" | "REJECTED" (optionally with a note), e.g.
        {"all": "ACCEPTED"} to accept every item, or {"M1": "ACCEPTED", "DRE-8": "REJECTED", ...}.

The CHANGELOG entry (step 4) is left to the owner, as for amendments[0].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE = REPO_ROOT / "results" / "v5" / "v5_plan_freeze.json"
ADOPTED = ("docs/V5_RUNSHEET.md", "docs/v5_design/V5_PLAN_AMENDMENT_02.md",
           "docs/v5_design/V5_DERM_REASONING_ENGINE.md")
#: The code and data that *implement* the frozen gates (audit AU17-AU36): the arm registry, the
#: screen gate, the histopathology definition and its D5 data, and the trunk loader. Hashed with
#: the documents so a gate cannot drift after the freeze without an Amendment 03.
GATE_FILES = ("research/v5/arms.py", "research/v5/screen_gate.py", "research/v5/confirmation.py",
              "research/v5/trunks.py", "results/v5/diagnostics/d5_acquisition.csv")
#: Items the owner accepts or rejects (runsheet section 1 step 1): Amendment 02 modules and
#: protocol changes, the DRE modules, and the runsheet's own decisions.
ITEMS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "GeM", "G-to-V6",
         "DRE-0", "DRE-1", "DRE-2", "DRE-3", "DRE-4", "DRE-5", "DRE-6", "DRE-7", "DRE-8", "DRE-10",
         "RS-screen-gate", "RS-composite-rule", "RS-param-cards", "RS-schedule",
         "RS-MILK10k-S84", "RS-concurrency",
         # 30 Sep pre-hash audit (docs/v5_design/V5_PLAN_AUDIT.md section 1b)
         "AU17-trunk-screen", "AU18-kseed-null-80pct", "AU19-pauc-histo-coprimary",
         "AU20-384-rescue", "AU21-continuous-queue", "AU22-dependency-order",
         "AU23-15-model-system", "AU24-youngdata-arm", "AU25-calibration-report",
         "AU26-trunk-attribution", "AU27-histo-definition", "AU28-disk", "AU29-s01-bootstrap",
         "AU30-gate-d-ranking", "AU31-gate-a-reading", "AU32-noise-aware-retention",
         "AU33-youngdata-within-band", "AU34-dino-prior", "AU35-unimplemented-modules",
         "AU36-in22k-primary")
IST = timezone(timedelta(hours=5, minutes=30))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_existing(freeze: dict) -> list[str]:
    failures = []
    for entry in freeze.get("frozen_artifacts", []):
        p = REPO_ROOT / entry["relative_path"]
        if not p.is_file() or sha256(p) != entry["sha256"].upper():
            failures.append(entry["relative_path"])
    for entry in freeze.get("synced_external_locations", []):
        p = Path(entry["external_path"])
        if p.is_file() and sha256(p) != entry["sha256"].upper():
            failures.append(entry["external_path"])
    for i, amendment in enumerate(freeze.get("amendments", [])):
        entries = amendment.get("files") or [amendment]
        for entry in entries:
            rel = entry.get("relative_path")
            if rel and sha256(REPO_ROOT / rel) != entry["sha256"].upper():
                failures.append(f"amendments[{i}] {rel}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--adopt", action="store_true")
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--by", default="owner (Rajrup910)")
    parser.add_argument("--open-choices", type=Path,
                        help="JSON of the owner's resolution of Amendment 01's open choices")
    args = parser.parse_args(argv)

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    failures = verify_existing(freeze)
    n_old = (len(freeze.get("frozen_artifacts", [])) + len(freeze.get("amendments", [])))
    print(f"existing hashes: {n_old} recorded in-repo; failures: {failures or 'none'}")
    new = {rel: sha256(REPO_ROOT / rel) for rel in ADOPTED + GATE_FILES}
    for rel, digest in new.items():
        print(f"  {digest}  {rel}")
    if args.verify:
        return 1 if failures else 0

    if failures:
        print("REFUSED: an existing hash no longer matches", file=sys.stderr)
        return 2
    if len(freeze.get("amendments", [])) >= 2:
        print("REFUSED: amendments[1] already exists", file=sys.stderr)
        return 2
    if not args.decisions or not args.decisions.is_file():
        print("REFUSED: --decisions <file.json> is required (owner's per-item decisions)",
              file=sys.stderr)
        return 2
    raw = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = ({item: raw["all"] for item in ITEMS} if "all" in raw else raw)
    missing = [i for i in ITEMS if i not in decisions]
    bad = {k: v for k, v in decisions.items()
           if str(v).split(":")[0].strip().upper() not in ("ACCEPTED", "REJECTED")}
    if missing or bad:
        print(f"REFUSED: missing decisions {missing}; invalid {bad}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(REPO_ROOT))
    from research.v5.arms import registry_sha256

    open_choices = (json.loads(args.open_choices.read_text(encoding="utf-8"))
                    if args.open_choices else {})
    freeze.setdefault("amendments", []).append({
        "index": 1,
        "files": [{"relative_path": rel, "sha256": digest} for rel, digest in new.items()],
        "adopted_timestamp": datetime.now(IST).isoformat(timespec="seconds"),
        "adopted_by": args.by,
        "adopted_before_first_v5_gpu_run": {
            "amendment_02_items": True,
            "note": "S01 (V5-S01-384-CONTROL) is registered by the master plan and Amendment 01 "
                    "and is not governed by this amendment's items"},
        "precedence": "docs/V5_RUNSHEET.md -> Amendment 02 -> DRE -> Amendment 01 -> frozen record",
        "item_decisions": decisions,
        "arm_registry_sha256": registry_sha256(),
        "amendment_01_open_choices_resolved": open_choices,
        "description": "Amendment 02 (biology-first mechanisms M1-M7, GeM control, Group-DRO to "
                       "V6), the Dermatologist Reasoning Engine (DRE-0..10) and the single V5 "
                       "runsheet (arms, pass rules, parameter cards, schedule, MILK10k S84), "
                       "with the 30 Sep pre-hash audit AU17-AU36 (IN-22k/DINOv3 trunk screen, "
                       "k-seed-mean screen null at the 80th percentile, pAUC_histo co-primary, "
                       "histopathology definition from D5, youngdata arm, 15-model system), "
                       "adopted together before any CPU pre-check (audit AU5). The gate code and "
                       "D5 data are hashed with the documents.",
    })
    FREEZE.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    after = verify_existing(json.loads(FREEZE.read_text(encoding="utf-8")))
    total = n_old + len(new)
    print(f"appended amendments[1]; re-verified {total} hashes; failures: {after or 'none'}")
    print("Next (runsheet section 1 step 4): add the CHANGELOG entry.")
    return 1 if after else 0


if __name__ == "__main__":
    raise SystemExit(main())
