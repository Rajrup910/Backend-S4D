"""Audit the three Gemini-written V7 registries against files on disk. Writes results/v7/registry_audit.json.

    python scripts/audit_v7_registries.py

* Lineage ledger: every receipt / primary-source file exists, and every numeric point estimate that its receipt JSON also
  holds is found there (a number found anywhere in the receipt counts; a number not found is listed, not judged wrong).
* Literature register: counts how each source was inspected; nothing is fetched (citations stay UNVERIFIED unless read).
* Model registry: for every checkpoint file that exists locally, the recorded SHA-256 is recomputed; licences cannot be
  checked offline and are listed as UNVERIFIED.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "v7_final_revision"


def numbers_in(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {f"{float(x):.4f}" for x in re.findall(r"-?\d+\.\d+", text)}


def main() -> int:
    out: dict = {}
    led = pd.read_csv(DOCS / "V7_LINEAGE_EVIDENCE_LEDGER.csv", dtype=str)
    rows = []
    for r in led.itertuples():
        files = [f.strip() for col in ("receipt_file", "primary_source_file") for f in str(getattr(r, col)).split(";") if f.strip() and f.strip() != "nan"]
        exist = {f: (REPO / f).exists() for f in files}
        found = None
        jsons = [REPO / f for f in files if f.endswith(".json") and (REPO / f).exists()]
        if jsons and str(r.point_estimate) not in ("nan", ""):
            try:
                value = f"{float(r.point_estimate):.4f}"
                found = any(value in numbers_in(j) for j in jsons)
            except ValueError:
                found = None
        rows.append({"claim_id": r.claim_id, "files": exist, "point_estimate": r.point_estimate, "point_in_receipt": found})
    out["lineage_ledger"] = {"rows": len(rows), "rows_all_files_exist": sum(all(x["files"].values()) and bool(x["files"]) for x in rows),
                             "rows_without_files": [x["claim_id"] for x in rows if not x["files"]],
                             "missing_files": sorted({f for x in rows for f, ok in x["files"].items() if not ok}),
                             "point_checked": sum(x["point_in_receipt"] is not None for x in rows),
                             "point_found": sum(x["point_in_receipt"] is True for x in rows),
                             "point_not_found": [x["claim_id"] for x in rows if x["point_in_receipt"] is False]}
    lit = pd.read_csv(DOCS / "V7_LITERATURE_EVIDENCE_REGISTER.csv", dtype=str)
    out["literature_register"] = {"sources": len(lit), "inspected_level": lit["inspected_level"].value_counts().to_dict(),
                                  "status": "not fetched in this audit; treat every citation as UNVERIFIED unless its full text was read"}
    reg = pd.read_csv(DOCS / "V7_MODEL_CANDIDATE_REGISTRY.csv", dtype=str)
    checked = []
    for r in reg.itertuples():
        hits = [p for p in (REPO / "ml" / "checkpoints").rglob("*") if p.is_file() and str(r.checkpoint_id) not in ("nan", "") and str(r.checkpoint_id) in p.name] if str(r.checkpoint_id) not in ("nan", "") else []
        entry = {"candidate_id": r.candidate_id, "recorded_sha256": r.checkpoint_sha256, "local_files": [str(h.relative_to(REPO)) for h in hits[:3]]}
        if hits and isinstance(r.checkpoint_sha256, str) and len(r.checkpoint_sha256) == 64:
            entry["sha256_matches"] = any(hashlib.sha256(h.read_bytes()).hexdigest() == r.checkpoint_sha256 for h in hits[:3])
        checked.append(entry)
    out["model_registry"] = {"candidates": len(checked), "with_local_file": sum(bool(c["local_files"]) for c in checked),
                             "sha256_checked": sum("sha256_matches" in c for c in checked),
                             "sha256_match": sum(c.get("sha256_matches") is True for c in checked),
                             "sha256_mismatch": [c["candidate_id"] for c in checked if c.get("sha256_matches") is False],
                             "licences": "UNVERIFIED offline (license_code / license_weights columns are agent-reported)", "entries": checked}
    (REPO / "results" / "v7" / "registry_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "entries"} for k, v in out.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
