"""Before the V6-3 chain spends a night: every file its runs need exists and every weight loads offline. Reads no image and writes nothing.

    python scripts/check_v63_inputs.py

Exits 1 and names what is wrong. Run by `scripts/run_v6_3_overnight.ps1`, with HF_HUB_OFFLINE=1, so it sees what the runs will see.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import timm  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from research.v5.trunks import timm_tag  # noqa: E402
from research.v6 import members  # noqa: E402
from research.v6.train_v6 import TRUNK  # noqa: E402

ANCHOR = "anchor_h7_f0_s42_in22k_v62y"
NEEDED = ("results/v6/added_pool/added_train.csv", f"results/v6/runs/{ANCHOR}.json", f"results/v6/preds/{ANCHOR}.csv", f"results/v6/preds/{ANCHOR}_swad.csv",
          f"ml/checkpoints/anchor-v6_{ANCHOR}_swad.pt", "results/v6/screens/decision_v61.json", "results/v6/screens/decision_v62.json")


def main() -> int:
    problems = [f"missing {name}" for name in NEEDED if not (REPO / name).is_file()]
    for name, spec in members.ROSTER.items():
        try:
            members.require_exact(spec)
            path = members.PANDERM_PATH if name == "panderm_b" else Path(hf_hub_download(f"timm/{spec.timm_name}", "model.safetensors"))
            members.check_pin(path, spec)
            print(f"ok {name}: weights match their pinned hash")
        except Exception as error:  # noqa: BLE001
            problems.append(f"{name}: {error}")
    tag = timm_tag(TRUNK, 384)
    try:
        timm.create_model(tag, pretrained=True, num_classes=0)
        print(f"ok anchor384: {tag} loads")
    except Exception as error:  # noqa: BLE001
        problems.append(f"anchor384: {error}")
    for problem in problems:
        print(f"PROBLEM {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
