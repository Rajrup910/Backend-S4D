"""Shared harness for the V5 CPU pre-checks (docs/V5_RUNSHEET.md section 4).

Import this FIRST in every pre-check entry point: it hides the GPU before torch is imported.

Guards, all hard errors:
  * ADOPTION  -- the runsheet's pass rules must be hashed before any check reads data (audit AU5,
                 break-nothing item 1). `require_adoption()` refuses unless results/v5/v5_plan_freeze.json
                 holds `amendments[1]` and the sha256 recorded there for every hashed file still
                 matches the file on disk (so no rule was edited after the hash).
  * GPU (C1)  -- CUDA_VISIBLE_DEVICES is forced to "-1" here. The runsheet's PowerShell line
                 `$env:CUDA_VISIBLE_DEVICES=""` deletes the variable in Windows PowerShell rather than
                 emptying it, which leaves the GPU visible; "-1" hides it.
  * PRIORITY / THREADS (C4) -- BelowNormal process priority, torch threads from OMP_NUM_THREADS.
  * MEMORY (C3) -- refuses to start with commit headroom < 3 GB.
  * ROWS (C8) -- only manifest `split == "train"` (the 15,294 development rows) is ever loaded.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # before any torch import (C1)

import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from research import testguard  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
IMAGE_DIR = REPO_ROOT / "data" / "external" / "isic2019_images" / "ISIC_2019_Training_Input"
HAM_MASK_DIR = (REPO_ROOT / "data" / "ham10000" / "HAM10000_segmentations_lesion_tschandl"
                / "HAM10000_segmentations_lesion_tschandl")
HAM_METADATA = REPO_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv"
FREEZE_FILE = REPO_ROOT / "results" / "v5" / "v5_plan_freeze.json"
DIAG_DIR = REPO_ROOT / "results" / "v5" / "diagnostics"
FOLD_ARTEFACT_DIR = REPO_ROOT / "results" / "v5" / "chromophore"
RUNSHEET = "docs/V5_RUNSHEET.md"
MIN_HEADROOM_GB = 3.0
#: Pre-checks read each image once, whole (no crop), with the short side resized to this [impl].
PRECHECK_SHORT_SIDE = 224
DEV_SPLIT = "train"
N_BOOT = 2000
SEED = 20260930


class NotAdopted(SystemExit):
    pass


# ------------------------------------------------------------------ guards
def require_adoption() -> dict[str, Any]:
    """Refuse unless the V5 runsheet is adopted and every hash in the freeze file still matches."""
    if not FREEZE_FILE.is_file():
        raise NotAdopted(f"{FREEZE_FILE} not found")
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    amendments = freeze.get("amendments", [])
    if len(amendments) < 2:
        raise NotAdopted(
            "REFUSED: docs/V5_RUNSHEET.md is not yet adopted and hashed (E1). The pre-check pass "
            "rules decide which arms run, so they must be frozen before any check reads data "
            "(runsheet section 1 step 5, audit AU5). Ask the owner to adopt, then hash.")
    amendment = amendments[1]
    files = amendment.get("files") or amendment.get("hashed_files") or []
    paths = {entry.get("relative_path"): entry.get("sha256") for entry in files}
    if RUNSHEET not in paths:
        raise NotAdopted(f"REFUSED: amendments[1] does not hash {RUNSHEET}")
    for rel, digest in paths.items():
        actual = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest().upper()
        if actual != str(digest).upper():
            raise NotAdopted(f"REFUSED: {rel} changed after it was hashed ({actual[:12]} != "
                             f"{str(digest)[:12]}). Changes after the hash go in an Amendment 03.")
    return {"amendment_index": 1, "hashed_files": paths,
            "adopted_timestamp": amendment.get("adopted_timestamp")}


def commit_headroom_gb() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    if sys.platform != "win32":
        return float("inf")
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPageFile / 1024 ** 3  # commit limit minus committed bytes


def set_below_normal_priority() -> None:
    if sys.platform == "win32":
        below_normal = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, below_normal)


def start(name: str, skip_adoption: bool = False) -> dict[str, Any]:
    """Arm every guard. `skip_adoption` exists only for the synthetic unit tests."""
    testguard.block_test_reads(f"V5 pre-check {name}: development rows only")
    if torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA is visible to a CPU pre-check (C1)")
    adoption = {"skipped": True} if skip_adoption else require_adoption()
    headroom = commit_headroom_gb()
    if headroom < MIN_HEADROOM_GB:
        raise SystemExit(f"REFUSED: commit headroom {headroom:.1f} GB < {MIN_HEADROOM_GB} GB (C3)")
    set_below_normal_priority()
    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(max(1, min(threads, 8)))
    print(f"[{name}] guards armed: GPU hidden, priority BelowNormal, threads {torch.get_num_threads()}, "
          f"headroom {headroom:.1f} GB, adoption {'skipped (test)' if skip_adoption else 'verified'}")
    return {"adoption": adoption, "headroom_gb_at_start": round(headroom, 2),
            "threads": torch.get_num_threads(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ------------------------------------------------------------------ rows
def development_rows() -> pd.DataFrame:
    """The 15,294 pooled development rows; nothing else is ever returned (C8)."""
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    rows = manifest[manifest["split"] == DEV_SPLIT].reset_index(drop=True)
    assert set(rows["split"]) == {DEV_SPLIT}
    ham = pd.read_csv(HAM_METADATA, usecols=["image_id", "dx_type"])
    rows = rows.merge(ham, on="image_id", how="left")
    return rows


def fold_train_rows(rows: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Rows of `rows` outside `fold` in the frozen S71 partition."""
    from research.v4.s71_kfold import load_assignments

    assignments = load_assignments()
    held = set(assignments.loc[assignments["fold"] == fold, "image_id"].astype(str))
    return rows[~rows["image_id"].astype(str).isin(held)].reset_index(drop=True)


# ------------------------------------------------------------------ images
def load_image(image_id: str, short_side: int = PRECHECK_SHORT_SIDE) -> torch.Tensor:
    """(1,3,H,W) sRGB in [0, 1], whole image, short side resized (antialiased bilinear)."""
    with Image.open(IMAGE_DIR / f"{image_id}.jpg") as image:
        image = image.convert("RGB")
        w, h = image.size
        scale = short_side / min(w, h)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.Resampling.BILINEAR, reducing_gap=None)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_ham_mask(image_id: str, size_hw: tuple[int, int]) -> torch.Tensor | None:
    """(1,1,H,W) float in [0,1] expert mask, area-resampled to size_hw; None if absent."""
    path = HAM_MASK_DIR / f"{image_id}_segmentation.png"
    if not path.is_file():
        return None
    with Image.open(path) as mask:
        arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).view(1, 1, *arr.shape)
    return F.interpolate(t, size=size_hw, mode="area")


def stream(ids: list[str], every: int = 500, label: str = "") -> Iterator[tuple[int, str, torch.Tensor]]:
    """One image at a time (C3: never hold the dataset in memory), with progress."""
    started = time.time()
    for i, image_id in enumerate(ids):
        yield i, image_id, load_image(image_id)
        if every and (i + 1) % every == 0:
            rate = (i + 1) / max(time.time() - started, 1e-6)
            print(f"  {label} {i + 1:,}/{len(ids):,}  {rate:.1f} img/s  "
                  f"eta {(len(ids) - i - 1) / max(rate, 1e-6) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ statistics
def auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s, sample_weight=w))


def grouped_bootstrap_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT,
                          seed: int = SEED) -> dict[str, float]:
    """AUC of s for y = 1 (declared direction: higher in the positive group), with a lesion-grouped
    percentile bootstrap (lesion resample counts become image weights). NaN scores are dropped."""
    y, s, groups = np.asarray(y), np.asarray(s, dtype=float), np.asarray(groups)
    keep = np.isfinite(s)
    y, s, groups = y[keep], s[keep], groups[keep]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    out = {"n_pos_images": n_pos, "n_neg_images": n_neg,
           "n_pos_lesions": int(pd.Series(groups[y == 1]).nunique()),
           "n_neg_lesions": int(pd.Series(groups[y == 0]).nunique())}
    if n_pos == 0 or n_neg == 0:
        return {**out, "auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot_valid": 0}
    codes, uniq = pd.factorize(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
        m = w > 0
        if len(np.unique(y[m])) < 2:
            continue
        vals.append(auc(y[m], s[m], w[m]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return {**out, "auc": auc(y, s), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot_valid": len(vals)}


# ------------------------------------------------------------------ output
def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc}"

    return {"head": run("rev-parse", "HEAD"), "dirty_files": len(run("status", "--porcelain").splitlines())}


def write_report(name: str, payload: dict[str, Any], context: dict[str, Any],
                 declarations: dict[str, Any] | None = None) -> Path:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = DIAG_DIR / name
    body = {"check": name, "context": context, "git": git_state(),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reads": "development rows only (manifest split == train); no test, reserved or "
                     "external labels", **({"implementation_declarations": declarations}
                                            if declarations else {}), **payload}
    out.write_text(json.dumps(body, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


Metric = Callable[[np.ndarray, np.ndarray], float]
