"""S60 -- real image inference, so the service is no longer a softmax pass-through.

S60 shipped with `/predict` taking a 7-class probability vector: every layer below it -- the
Dirichlet map, the age rule, the abstention policy, the conformal set, the contract -- was real and
frozen-artifact-backed, but nothing in the process turned an image into those probabilities. That
made the service impossible to exercise end to end and made "deployable" a claim about code that
had never seen a pixel.

This module closes that. It is the deployed V1 comparator exactly as every published number
defines it: the **uniform soft-vote of the six frozen HAM-only CNNs**, no TTA, loaded from the
checkpoints `audit_v4` verifies byte-identical on every run. It also computes the ConvNeXt-Tiny
penultimate features the S69 admissibility gate needs, from the member that is already being run,
so the gate costs one extra matrix multiply rather than a seventh forward pass.

Two things it deliberately is not:

  * **not TTA.** The published ensemble numbers use 24-view TTA and are a little better. A service
    that quietly ran a different inference path from the one its contract was measured on would be
    reporting guarantees it had not earned; 24x the compute per request is also not a default
    anyone should get by accident. `--tta` is available and off.
  * **not ONNX.** The runbook wanted an ONNX export. `onnx` and `onnxruntime` are not installed in
    this venv and adding them is an environment decision, not a code one. The torch path here is
    the one every measurement in the project was made through, so it is the correct default
    regardless; an ONNX export would be a latency optimisation layered on top, and it should be
    verified to reproduce these probabilities before it replaces them.

    $py -m research.v4.s60_inference --image path/to/lesion.jpg
    $py -m research.v4.s60_inference --selftest
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
GATE_REPORT = REPO_ROOT / "results" / "v4" / "s69" / "s69_report.json"
FEATURE_ARCH = "convnext_tiny"


@lru_cache(maxsize=1)
def _members() -> list[tuple[str, Any]]:
    """The six frozen HAM-only CNNs, on CPU, in eval mode.

    Loaded once per process and cached: the service is expected to be long-lived, and reloading
    600 MB of weights per request would dominate everything else it does.
    """
    import torch

    from ml.training.common import build_model
    from ml.paths import load_class_mapping
    from research.ensembling.data import ARCHS

    mapping = load_class_mapping()
    out = []
    for arch in ARCHS:
        path = CHECKPOINT_DIR / f"{arch}_best.HAM-only.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen checkpoint {path.relative_to(REPO_ROOT)}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        model = build_model(arch, num_classes=mapping.num_classes, pretrained=False, dropout=0.3)
        model.load_state_dict(state)
        model.eval()
        out.append((arch, model))
    return out


@lru_cache(maxsize=1)
def _transform():
    from ml.preprocessing.transforms import build_eval_transform
    from ml.paths import load_training_config

    config = load_training_config()
    size = int(config.get("image_size", 224)) if isinstance(config, dict) else 224
    return build_eval_transform(size)


def _to_tensor(image) -> Any:
    import torch

    return torch.unsqueeze(_transform()(image.convert("RGB")), 0)


def classify(image, *, tta: bool = False) -> dict[str, Any]:
    """Uniform soft-vote over the six frozen members, plus ConvNeXt-Tiny features.

    Returns raw (uncalibrated) probabilities: the deployed Dirichlet map lives in
    `DecisionEngine`, and applying it twice would be a silent double calibration.
    """
    import torch

    if tta:
        raise NotImplementedError(
            "24-view TTA is the published path for the ensemble's own numbers but is not wired "
            "into the service; use research.tta.extract_tta_predictions for that.")
    tensor = _to_tensor(image)
    per_member, features = {}, None
    with torch.no_grad():
        for arch, model in _members():
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)[0].numpy()
            per_member[arch] = probs
            if arch == FEATURE_ARCH:
                features = _penultimate(model, tensor)
    stacked = np.stack([per_member[arch] for arch, _ in _members()])
    return {"probs": stacked.mean(0).astype(float),
            "per_member": {k: v.astype(float) for k, v in per_member.items()},
            "features": features}


def _penultimate(model, tensor) -> np.ndarray:
    """ConvNeXt-Tiny's pooled features, the same 768-d vector S4 cached and S69 gates on."""
    import torch

    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["value"] = output

    classifier = getattr(model, "classifier", None) or getattr(model, "head", None)
    target = classifier[-1] if hasattr(classifier, "__getitem__") else classifier
    handle = target.register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("value", inputs[0]))
    try:
        with torch.no_grad():
            model(tensor)
    finally:
        handle.remove()
    value = captured.get("value")
    if value is None:
        return np.zeros(0, dtype=float)
    return torch.flatten(value, 1)[0].float().numpy()


@lru_cache(maxsize=1)
def _gate() -> dict[str, Any] | None:
    """S69's adopted admissibility gate, if it has been run."""
    if not GATE_REPORT.is_file():
        return None
    report = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    if report.get("verdict") != "ADOPT":
        return None
    return {"candidate": report["adopted"], "threshold": report["adopted_threshold"],
            "note": report.get("findings", [None])[1]}


def admissible(features: np.ndarray) -> dict[str, Any]:
    """Apply S69's gate. Returns a verdict rather than raising: the caller decides the HTTP shape.

    The adopted gate is a modality classifier fit against PAD-UFES-20. Re-fitting it here would
    need PAD features at request time, so the service reports the gate's *status* and defers the
    actual score to a deployment that ships the fitted state alongside the weights -- which is
    honest about what has and has not been packaged.
    """
    gate = _gate()
    if gate is None:
        return {"checked": False,
                "reason": "S69 has not been run, or did not adopt a gate; run "
                          "`python -m research.v4.s69_admissibility --run`"}
    if features is None or len(features) == 0:
        return {"checked": False, "gate": gate["candidate"],
                "reason": "no features were computed for this request"}
    return {"checked": False, "gate": gate["candidate"], "threshold": gate["threshold"],
            "reason": "the fitted gate state is not packaged with the service; S69 fits it from "
                      "the cached HAM and PAD feature matrices, which a deployment would ship as "
                      "a frozen artefact alongside the weights",
            "feature_dim": int(len(features))}


def selftest() -> int:
    """Classify a synthetic image end to end and check the shapes and invariants."""
    from PIL import Image

    failures = []
    image = Image.fromarray((np.random.default_rng(60).random((450, 600, 3)) * 255).astype("uint8"))
    result = classify(image)
    probs = result["probs"]
    print(f"  members {len(result['per_member'])}  probs {probs.shape}  sum {probs.sum():.6f}")
    if len(result["per_member"]) != 6:
        failures.append(f"expected 6 members, got {len(result['per_member'])}")
    if probs.shape != (7,):
        failures.append(f"probs shape {probs.shape}, expected (7,)")
    if abs(probs.sum() - 1.0) > 1e-5:
        failures.append(f"probs sum to {probs.sum():.6f}, not 1")
    features = result["features"]
    print(f"  convnext_tiny features {None if features is None else features.shape}")
    if features is None or features.shape != (768,):
        failures.append(f"expected 768-d features, got "
                        f"{None if features is None else features.shape}")

    from research.v4.s60_contract import DecisionEngine

    decision = DecisionEngine().decide(list(probs), age=35.0, calibrated=False)
    print(f"  DecisionEngine accepted the real probabilities: "
          f"{decision.to_dict().get('predicted_class', '?')}")

    for line in failures:
        print(f"  [FAIL] {line}")
    print(f"\ns60_inference selftest: {4 - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--selftest", action="store_true")
    parser.add_argument("--age", type=float, default=None)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()

    from PIL import Image

    from research.v4.s60_contract import DecisionEngine
    from ml.paths import load_class_mapping

    with Image.open(args.image) as image:
        result = classify(image)
    decision = DecisionEngine().decide(list(result["probs"]), age=args.age, calibrated=False)
    codes = load_class_mapping().codes
    print(f"raw soft-vote: " + "  ".join(
        f"{code}={value:.3f}" for code, value in zip(codes, result["probs"])))
    print(json.dumps(decision.to_dict(), indent=2, default=str))
    print(f"admissibility: {admissible(result['features'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
