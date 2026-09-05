"""End-to-end demo: pick a real HAM10000 test-split image, POST it through the backend
in-process, and print the classification, triage decision and LLM explanation.

This is the "python demo.py photo.jpg -> full advice out" script from the project plan --
useful for a quick sanity check, a screen recording, or Review 0/1 without needing curl or
Postman.

Usage:
    .venv/Scripts/python.exe scripts/sample_predict.py
    .venv/Scripts/python.exe scripts/sample_predict.py --image path/to/photo.jpg
    .venv/Scripts/python.exe scripts/sample_predict.py --language hi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def pick_sample_image() -> Path:
    """Grab a real image from the test split so this isn't synthetic noise."""
    import pandas as pd

    from ml.paths import resolve

    splits = pd.read_csv(resolve("ml/configs/splits/split_v1.csv"))
    manifest = resolve("ml/data/manifest.csv")
    if not manifest.is_file():
        raise SystemExit("No manifest found. Run: python -m ml.preprocessing.prepare_dataset")

    manifest_df = pd.read_csv(manifest)
    test_ids = splits[splits["split"] == "test"]["image_id"]
    row = manifest_df[manifest_df["image_id"].isin(test_ids)].sample(1, random_state=None).iloc[0]
    return resolve(row["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="Path to an image. Defaults to a random test-split image.")
    parser.add_argument("--language", default="en", choices=["en", "hi"])
    parser.add_argument("--checkpoint", default="", help="Override MODEL_CHECKPOINT for this run.")
    parser.add_argument("--arch", default="", help="Override MODEL_ARCH for this run.")
    args = parser.parse_args()

    import os

    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint
    if args.arch:
        os.environ["MODEL_ARCH"] = args.arch

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = pick_sample_image()

    print(f"Image: {image_path}")

    from fastapi.testclient import TestClient

    from backend.app.config import get_settings

    get_settings.cache_clear()
    from backend.app.main import create_app

    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        print(f"Model: {health['model_arch']} loaded={health['model_loaded']} "
              f"stub={health['stub_mode']} device={health['device']}")
        print(f"LLM available: {health['llm_available']}")
        print()

        response = client.post(
            "/predict",
            files={"image": (image_path.name, image_path.read_bytes(), "image/jpeg")},
            data={"language": args.language},
        )

        if response.status_code != 200:
            print(f"FAILED: {response.status_code}")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            return 1

        body = response.json()

        print("=" * 60)
        print(f"Predicted:   {body['predicted_name']} ({body['predicted_class']})")
        print(f"Confidence:  {body['confidence']:.1%}")
        print(f"Stub mode:   {body['stub']}")
        print(f"Inference:   {body['inference_ms']} ms")
        print()
        print("Top classes:")
        for p in body["probabilities"][:3]:
            print(f"  {p['code']:<6} {p['name']:<35} {p['probability']:.1%}  ({p['malignancy']})")
        print()
        print(f"Triage:      {body['triage']['label']}  [{body['triage']['category']}]")
        for reason in body["triage"]["reasons"]:
            print(f"  - {reason}")
        print()
        print(f"Grad-CAM:    {body['gradcam_url']}  ({body['gradcam_focus']})")
        print()
        if body["explanation_available"]:
            print("Explanation:")
            print(body["explanation"])
        else:
            print("Explanation: unavailable (Ollama not reachable)")
        print()
        print("Disclaimer:")
        print(body["disclaimer"])
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
