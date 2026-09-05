"""Environment self-check for Scan4Disease.

Run this after setup and any time something behaves oddly:

    .venv\\Scripts\\python.exe scripts\\verify_env.py

It answers the questions that actually cause lost days on this project:
  - Is this a Python version PyTorch supports?
  - Does torch see the GPU, and can it *actually run a kernel* on it?
    (`torch.cuda.is_available()` returning True is not sufficient on Blackwell
    cards -- a wheel without sm_120 kernels reports True and then fails at the
    first real operation.)
  - Are the ML/backend libraries importable?
  - Is Ollama up, and is the configured model pulled?

Exit code is 0 if nothing critical failed, 1 otherwise.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
WANTED_LLM = "qwen3"

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
_failures: list[str] = []
_warnings: list[str] = []


def report(status: str, message: str) -> None:
    print(f"{status} {message}")
    if status == FAIL:
        _failures.append(message)
    elif status == WARN:
        _warnings.append(message)


def check_python() -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) == (3, 12) or (major, minor) == (3, 13):
        report(OK, f"Python {version} ({platform.machine()})")
    elif (major, minor) >= (3, 14):
        report(FAIL, f"Python {version} - PyTorch publishes no wheels for 3.14+. Use 3.12.")
    else:
        report(WARN, f"Python {version} - older than recommended (3.12).")
    print(f"       interpreter: {sys.executable}")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        report(FAIL, "torch not installed. Run scripts/setup_env.ps1.")
        return

    report(OK, f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        report(
            WARN,
            "CUDA not available - training will fall back to CPU (very slow). "
            "If you have an NVIDIA GPU, you likely installed the CPU wheel.",
        )
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    report(OK, f"CUDA {torch.version.cuda} | {name} | sm_{cap[0]}{cap[1]} | {total_gb:.1f} GB")

    # The decisive test: does a real kernel launch succeed on this device?
    arch_list = torch.cuda.get_arch_list()
    try:
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        if y != y:  # NaN
            raise RuntimeError("matmul produced NaN")
        report(OK, f"GPU kernel launch verified (arch list: {', '.join(arch_list)})")
    except Exception as exc:  # noqa: BLE001 - we want the raw message here
        report(
            FAIL,
            f"GPU is visible but cannot run kernels: {exc}\n"
            f"       This wheel was built for {arch_list}, your card is sm_{cap[0]}{cap[1]}.\n"
            f"       Reinstall from a newer CUDA index, e.g.\n"
            f"       pip install --force-reinstall torch torchvision "
            f"--index-url https://download.pytorch.org/whl/cu128",
        )


def check_imports() -> None:
    packages = {
        "torchvision": "vision models + transforms",
        "timm": "pretrained backbones",
        "numpy": "arrays",
        "pandas": "metadata handling",
        "sklearn": "metrics + splits",
        "matplotlib": "plots",
        "PIL": "image IO",
        "cv2": "image quality gate",
        "yaml": "training config",
        "fastapi": "backend",
        "pydantic_settings": "backend config",
        "httpx": "LLM client",
    }
    missing = []
    for module, purpose in packages.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(f"{module} ({purpose})")
    if missing:
        report(FAIL, "missing packages: " + ", ".join(missing))
    else:
        report(OK, f"all {len(packages)} required libraries import cleanly")


def check_ollama() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            models = [m["name"] for m in json.load(response).get("models", [])]
    except (urllib.error.URLError, OSError, TimeoutError):
        report(
            WARN,
            f"Ollama not reachable at {OLLAMA_URL} - the LLM explanation layer will be "
            "disabled. Install from ollama.com, then: ollama pull qwen3:8b",
        )
        return

    if any(WANTED_LLM in m for m in models):
        report(OK, f"Ollama up, models: {', '.join(models)}")
    else:
        report(
            WARN,
            f"Ollama up but no '{WANTED_LLM}' model found (have: {', '.join(models) or 'none'}). "
            f"Run: ollama pull qwen3:8b",
        )


def main() -> int:
    print("=" * 68)
    print("Scan4Disease - environment check")
    print("=" * 68)
    check_python()
    check_torch()
    check_imports()
    check_ollama()
    print("=" * 68)

    if _failures:
        print(f"{len(_failures)} critical problem(s). Fix these before training.")
        return 1
    if _warnings:
        print(f"No blockers. {len(_warnings)} warning(s) - fine for ML work, "
              f"resolve before the LLM phase.")
        return 0
    print("Everything green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
