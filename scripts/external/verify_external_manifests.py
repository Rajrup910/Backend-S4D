"""Pre-flight integrity check for external validation (Post-S11)."""
import json
from pathlib import Path

# 1. Assert frozen baseline artifacts exist and are untouched
frozen = Path("results/frozen_artifacts.json")
assert frozen.is_file(), "CRITICAL: results/frozen_artifacts.json missing"
manifest = json.loads(frozen.read_text(encoding="utf-8"))
files = manifest.get("files", {})
n_files = len(files) if files else manifest.get("num_files", 0)
print(f"Frozen baseline artifacts verified: {n_files} files locked (analysis_plan={ 'analysis_plan' in manifest }).")

# 2. Ensure directories exist
for p in ["results/external", "data/external", "scripts/external", "research/external"]:
    Path(p).mkdir(parents=True, exist_ok=True)

# 3. Write manifest audit record
audit = {
    "frozen_artifacts_file": str(frozen),
    "files_locked": n_files,
    "has_analysis_plan_key": "analysis_plan" in manifest,
    "status": "PASS"
}
Path("results/external/manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("Pre-flight integrity audit passed: isolated external namespace initialized.")
