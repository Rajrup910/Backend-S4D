"""Download PAD-UFES-20 and the HAM10000 lesion segmentation masks, with hash verification.

Network-bound, not GPU-bound -- safe to run in parallel with the S2 K-fold OOF training
queue (Workstream B is independent of Workstream A).

**PAD-UFES-20** (~3.59 GB total): 3 image zips + metadata.csv, from Mendeley Data
(DOI 10.17632/zr7vgbcyr2.1, dataset id `zr7vgbcyr2`, its only published version). The file
list, sizes, SHA256 hashes and download URLs are fetched live from Mendeley's public
dataset API (`https://data.mendeley.com/public-api/datasets/<id>`) rather than
hardcoded, so a future re-upload that changes internal file ids cannot silently break
this script or (worse) go unverified. Extracted into `data/pad_ufes_20/`, then
`ml.preprocessing.prepare_pad_ufes` is run to build `ml/data/manifest_pad.csv`.

**HAM10000 lesion segmentation masks** (~10.3 MB): `HAM10000_segmentations_lesion_tschandl.zip`
from Harvard Dataverse, same DOI as HAM10000 itself (`10.7910/DVN/DBW86T`). The file id and
MD5 are likewise resolved live via Dataverse's public dataset API
(`https://dataverse.harvard.edu/api/datasets/:persistentId/`), not hardcoded. Extracted into
`data/ham10000/HAM10000_segmentations_lesion_tschandl/`, alongside the existing
`HAM10000_images_part_{1,2}/` and `HAM10000_metadata.csv`. Nothing in this repo consumes
these masks yet -- they exist for the lesion-interior Grad-CAM attribution metric in
Workstream D, which needs them to compute anything beyond the current image-frame heuristic.

Both manifests were live-verified while writing this script (2026-09-04): PAD-UFES-20 totals
3,592,712,972 bytes across 4 files, matching the plan's "~3.5 GB" estimate; the segmentation
zip is 10,808,743 bytes with MD5 6e8d252e09cfdb0189199f15985a5b84.

Usage:
    python -m scripts.download_pad_ufes                # both, then rebuild manifest_pad.csv
    python -m scripts.download_pad_ufes --skip-masks    # PAD only
    python -m scripts.download_pad_ufes --skip-pad      # masks only
    python -m scripts.download_pad_ufes --no-prepare    # download+verify+extract only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from ml.paths import REPO_ROOT, resolve


def _rel(p: Path) -> str:
    """Repo-relative path for display, or the path itself if it lives outside the repo
    (e.g. a real download target is always under REPO_ROOT via resolve(), but this keeps
    the print helpers safe if ever called on an arbitrary path, such as in a test)."""
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)

MENDELEY_DATASET_ID = "zr7vgbcyr2"
MENDELEY_API = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"

DATAVERSE_DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE_API = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
DATAVERSE_ACCESS = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
SEGMENTATION_FILENAME = "HAM10000_segmentations_lesion_tschandl.zip"

CHUNK = 1 << 20  # 1 MiB


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-skin-lesion/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path, expected_size: int, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-skin-lesion/1.0"})
    print(f"  downloading {label} ({expected_size / 1e6:.1f} MB) -> {_rel(dest)}")
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        downloaded = 0
        last_pct = -1
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            pct = int(downloaded * 100 / expected_size) if expected_size else 0
            if pct != last_pct and pct % 10 == 0:
                print(f"    {pct}%  ({downloaded / 1e6:.0f} / {expected_size / 1e6:.0f} MB)")
                last_pct = pct
    actual = dest.stat().st_size
    if expected_size and actual != expected_size:
        raise IOError(f"{label}: downloaded {actual} bytes, expected {expected_size}")


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected_hash: str, algo: str, label: str) -> None:
    actual = _hash_file(path, algo)
    if actual.lower() != expected_hash.lower():
        path.unlink()
        raise IOError(
            f"{label}: {algo.upper()} mismatch (expected {expected_hash}, got {actual}) -- "
            f"file deleted, re-run to retry"
        )
    print(f"  {algo.upper()} verified: {label}")


def _fetch_and_verify(url: str, dest: Path, expected_size: int, expected_hash: str, algo: str, label: str) -> None:
    if dest.is_file() and dest.stat().st_size == expected_size:
        try:
            _verify(dest, expected_hash, algo, f"{label} (already present)")
            return
        except IOError:
            pass  # hash mismatch on an existing file: fall through and re-download
    _download(url, dest, expected_size, label)
    _verify(dest, expected_hash, algo, label)


def fetch_mendeley_manifest() -> list[dict]:
    print(f"Querying Mendeley dataset API: {MENDELEY_API}")
    data = _get_json(MENDELEY_API)
    files = []
    for f in data["files"]:
        cd = f["content_details"]
        files.append({
            "filename": f["filename"],
            "size": cd["size"],
            "sha256": cd["sha256_hash"],
            "download_url": cd["download_url"],
        })
    total = sum(f["size"] for f in files)
    print(f"  dataset version {data.get('version')}, {len(files)} files, {total / 1e9:.2f} GB total")
    return files


def fetch_dataverse_segmentation_file() -> dict:
    print(f"Querying Harvard Dataverse dataset API for {DATAVERSE_DOI}")
    data = _get_json(DATAVERSE_API)
    for entry in data["data"]["latestVersion"]["files"]:
        df = entry["dataFile"]
        if df["filename"] == SEGMENTATION_FILENAME:
            checksum = df["checksum"]
            return {
                "filename": df["filename"],
                "id": df["id"],
                "size": df["filesize"],
                "hash": checksum["value"],
                "hash_algo": checksum["type"].lower(),
                "download_url": DATAVERSE_ACCESS.format(file_id=df["id"]),
            }
    raise SystemExit(f"{SEGMENTATION_FILENAME!r} not found in Dataverse dataset {DATAVERSE_DOI}")


def download_pad(pad_root: Path, cache_dir: Path) -> None:
    files = fetch_mendeley_manifest()
    print(f"\nPAD-UFES-20 -> {_rel(pad_root)}")
    for f in files:
        dest = cache_dir / f["filename"]
        _fetch_and_verify(f["download_url"], dest, f["size"], f["sha256"], "sha256", f["filename"])

    pad_root.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = cache_dir / f["filename"]
        if dest.suffix == ".zip":
            print(f"  extracting {f['filename']} -> {_rel(pad_root)}/")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(pad_root)
        else:
            target = pad_root / f["filename"]
            target.write_bytes(dest.read_bytes())
            print(f"  copied {f['filename']} -> {_rel(target)}")


def download_segmentation_masks(ham_root: Path, cache_dir: Path) -> None:
    f = fetch_dataverse_segmentation_file()
    print(f"\nHAM10000 segmentation masks -> {_rel(ham_root)}")
    dest = cache_dir / f["filename"]
    _fetch_and_verify(f["download_url"], dest, f["size"], f["hash"], f["hash_algo"], f["filename"])

    out_dir = ham_root / "HAM10000_segmentations_lesion_tschandl"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting -> {_rel(out_dir)}/")
    with zipfile.ZipFile(dest) as zf:
        zf.extractall(out_dir)
    mask_count = sum(1 for p in out_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".tif", ".tiff"})
    print(f"  {mask_count} mask file(s) extracted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pad-root", default="data/pad_ufes_20")
    parser.add_argument("--ham-root", default="data/ham10000")
    parser.add_argument("--cache-dir", default="data/_downloads",
                         help="where raw zips/csv land before extraction (kept for re-verification, not auto-deleted)")
    parser.add_argument("--skip-pad", action="store_true")
    parser.add_argument("--skip-masks", action="store_true")
    parser.add_argument("--no-prepare", action="store_true",
                         help="skip running ml.preprocessing.prepare_pad_ufes after PAD download")
    args = parser.parse_args(argv)

    cache_dir = resolve(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pad:
        download_pad(resolve(args.pad_root), cache_dir)
    if not args.skip_masks:
        download_segmentation_masks(resolve(args.ham_root), cache_dir)

    if not args.skip_pad and not args.no_prepare:
        print("\nRunning ml.preprocessing.prepare_pad_ufes ...")
        from ml.preprocessing.prepare_pad_ufes import main as prepare_main
        rc = prepare_main(["--pad-root", args.pad_root])
        if rc != 0:
            print("prepare_pad_ufes reported an error; PAD download/extraction itself "
                  "succeeded, but the manifest was not (re)built.", file=sys.stderr)
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
