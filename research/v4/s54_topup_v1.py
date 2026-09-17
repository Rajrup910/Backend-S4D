"""S54 -- top up the frozen V1 comparator with the 146 reserved `scc` images. GPU, run by hand.

S13 scored BCN-20000 and MSKCC with the six frozen HAM-only CNNs but excluded `scc` by
pre-registration (the HAM mapping has no `scc` slot). The V4 manifest folds `scc` into `akiec`
(`class_index_7 = 0`, escalating), so 146 of the 4,733 reserved images have no V1 prediction and
Gate A would otherwise be read on 103 under-40 escalating lesions instead of the declared 104.

Nothing is re-implemented. Scoring is S13's own path -- `extract_external_predictions`'
`_cohort_config` + `_attach_metadata` around `research.tta.extract_tta_predictions.extract_one`
(24-view TTA, S13's scales/batch) -- and assembly is S14's `assemble_external_ensemble`
(`load_member_matrix` + `assemble`: uniform soft-vote, deployed HAM-OOF Dirichlet map). Outputs go
to `results/v4/s54/v1_topup/`, never over the S13 files that `post_s11_artifacts.json` hashes.

    $py -m research.v4.s54_topup_v1 --dry-run     # list what would be scored; no labels, no model
    $py -m research.v4.s54_topup_v1 --smoke       # V4 val scc + a reproduction check vs S13
    $py -m research.v4.s54_topup_v1               # the reserved top-up (receipt stage v1_topup)

`--smoke` scores V4 **val**'s 59 `scc` images (so the smoke gate has full V1 coverage) and, into a
separate `repro/` folder, a few non-`scc` val images S13 already scored. Those must reproduce the
frozen S13 probabilities -- per architecture and after assembly -- or the smoke fails: that is the
proof that the top-up is the same comparator, not a lookalike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, load_training_config
from ml.training.common import resolve_device
from research.ensembling.data import ARCHS
from research.external.assemble_external_ensemble import (
    ENSEMBLE_TEMPLATE,
    assemble,
    load_member_matrix,
)
from research.external.extract_external_predictions import _attach_metadata, _cohort_config
from research.external.manifests import IMAGES_DIR
from research.tta.extract_tta_predictions import extract_one
from research.v4 import s54_gate as gate
from research.v4 import s54_guard as guard

#: S13's settings -- `extract_external_predictions.parse_args` defaults. The smoke's reproduction
#: check is what proves these are the ones S13 used.
S13_SCALES = (0.9, 1.0, 1.1)
S13_BATCH = 4
REPRO_ROWS = 8
#: Max |dp| against S13, by the device doing the re-scoring. S13 ran on CUDA, where cuDNN convs use
#: TF32 by default; a CPU fp32 re-run differs at the 1e-3 level for that reason alone (measured in
#: the S54 build: 3.1e-04 .. 2.7e-03 across the six members, argmax identical on every row). A real
#: pipeline difference -- wrong scales, transform or checkpoint -- moves probabilities by >1e-2.
#: The argmax must agree on every row on either device.
REPRO_TOL = {"cuda": 1e-3, "cpu": 5e-3}
CHECKPOINTS = guard.REPO_ROOT / "ml" / "checkpoints"


def select(split: str, repro: bool) -> pd.DataFrame:
    manifest = pd.read_csv(gate.MANIFEST, low_memory=False)
    rows = manifest[manifest["split"] == split]
    if repro:
        return rows[rows["class_8"] != "scc"].sort_values("image_id").head(REPRO_ROWS)
    return rows[rows["class_8"] == "scc"].sort_values("image_id")


def write_inputs(rows: pd.DataFrame, folder: Path, cohort: str) -> tuple[Path, Path]:
    """An S13-schema adapted manifest + all-`test` split, for LesionDataset."""
    codes = load_class_mapping().codes
    manifest = pd.DataFrame({
        "image_id": rows["image_id"].astype(str),
        "lesion_id": rows["lesion_id"],
        "class_code": [codes[i] for i in rows["class_index_7"]],
        "class_index": rows["class_index_7"].astype(int),
        "path": [f"{IMAGES_DIR}/{i}.jpg" for i in rows["image_id"]],
        "age_approx": rows["age_approx"],
        "sex": rows["sex"],
    })
    split = manifest[["image_id", "lesion_id", "class_code", "class_index"]].assign(split="test")
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / f"manifest_{cohort}.csv"
    split_path = folder / f"split_{cohort}.csv"
    manifest.to_csv(manifest_path, index=False)
    split.to_csv(split_path, index=False)
    return manifest_path, split_path


def score_rows(rows: pd.DataFrame, folder: Path, device, workers: int) -> list[Path]:
    """Six architectures x 24 views per cohort, then the deployed assembly. Returns written files."""
    config = load_training_config()
    written = []
    for cohort, part in rows.groupby("archive"):
        manifest_path, split_path = write_inputs(part, folder / "inputs", cohort)
        cfg = _cohort_config(config, manifest_path)
        for arch in ARCHS:
            name = f"{arch}_{cohort}.csv"
            extract_one(arch=arch, split="test", checkpoints_dir=CHECKPOINTS, out_dir=folder,
                        device=device, config=cfg, scales=S13_SCALES, batch_size=S13_BATCH,
                        num_workers=workers, limit=None, splits_file=split_path, out_name=name)
            _attach_metadata(folder / name, manifest_path)
            written.append(folder / name)
        frame = assemble(load_member_matrix(cohort, tuple(ARCHS), gate.rel(folder)))
        problems = validate(frame, part)
        if problems:
            raise SystemExit(f"[{cohort}] top-up assembly failed: {problems}")
        out = folder / ENSEMBLE_TEMPLATE.format(cohort=cohort)
        frame.to_csv(out, index=False)
        written.append(out)
        print(f"[{cohort}] {len(frame)} images assembled -> {gate.rel(out)}")
    return written


def validate(frame: pd.DataFrame, rows: pd.DataFrame) -> list[str]:
    codes = load_class_mapping().codes
    problems = []
    if sorted(frame["image_id"].astype(str)) != sorted(rows["image_id"].astype(str)):
        problems.append("image set differs from the selected rows")
    for prefix in ("p_", "p_raw_"):
        block = frame[[f"{prefix}{c}" for c in codes]].to_numpy()
        if not np.isfinite(block).all() or np.abs(block.sum(1) - 1).max() > 1e-6:
            problems.append(f"{prefix} rows are not finite probability vectors")
    truth = rows.set_index("image_id")["class_index_7"].reindex(frame["image_id"]).to_numpy()
    if not np.array_equal(frame["true_index"].to_numpy(), truth):
        problems.append("true_index disagrees with manifest_v4 class_index_7")
    return problems


def reproduce(folder: Path, device_type: str) -> list[str]:
    """Every re-scored non-scc val image must match S13's frozen probabilities."""
    codes = [f"p_{c}" for c in load_class_mapping().codes]
    tolerance = REPRO_TOL.get(device_type, REPRO_TOL["cuda"])
    failures = []
    for mine in sorted(folder.glob("*.csv")):
        frozen = gate.V1_FROZEN_DIR / mine.name
        a = pd.read_csv(mine).set_index("image_id")
        b = pd.read_csv(frozen).set_index("image_id").loc[a.index]
        pa, pb = a[codes].to_numpy(), b[codes].to_numpy()
        drift = float(np.abs(pa - pb).max())
        same = bool((pa.argmax(1) == pb.argmax(1)).all())
        print(f"  repro {mine.name:<36} max |dp| vs S13 = {drift:.2e}  argmax identical={same}")
        if drift > tolerance or not same:
            failures.append(f"{mine.name}: {drift:.2e} (tol {tolerance:g}, {device_type}), "
                            f"argmax identical={same}")
    return failures


def run(args: argparse.Namespace) -> int:
    split = "val" if args.smoke else "reserved"
    rows = select(split, repro=False)
    print(f"{split}: {len(rows)} scc images across {rows['archive'].value_counts().to_dict()}")
    if args.dry_run:
        # Metadata only: image ids and archive. No label, no image, no model.
        frozen = set()
        for cohort in gate.V1_COHORTS:
            path = gate.V1_FROZEN_DIR / f"ensemble_dirichlet_{cohort}.csv"
            frozen |= set(pd.read_csv(path, usecols=["image_id"])["image_id"].astype(str))
        ids = pd.read_csv(gate.MANIFEST, usecols=["image_id", "split"], low_memory=False)
        ids = ids[ids["split"] == split]["image_id"].astype(str)
        print(f"frozen V1 covers {int(ids.isin(frozen).sum())} of {len(ids)} {split} images; "
              f"top-up adds {len(rows)}; overlap {int(rows['image_id'].isin(frozen).sum())}")
        return 0

    digest = gate.require_plan()
    device = resolve_device(args.device)
    guard.preflight(commit_gb=guard.torch_commit_gb(args.num_workers),
                    training_check=not (args.smoke and device.type == "cpu"))
    folder = gate.topup_dir(args.smoke)

    if args.smoke:
        score_rows(rows, folder, device, args.num_workers)
        score_rows(select("val", repro=True), folder / "repro", device, args.num_workers)
        failures = reproduce(folder / "repro", device.type)
        if failures:
            print("SMOKE FAILED: the top-up does not reproduce S13\n  " + "\n  ".join(failures))
            return 1
        print("SMOKE OK: top-up reproduces S13 within tolerance; no receipt, no ledger row")
        return 0

    if len(rows) != gate.EXPECTED_COHORT["scc_topup_images"]:
        raise SystemExit(f"{len(rows)} reserved scc images, plan expects "
                         f"{gate.EXPECTED_COHORT['scc_topup_images']}")
    receipt = guard.ReservedReceipt(digest)
    receipt.open_stage("v1_topup", args.rerun_reason, args.resume)
    for path in score_rows(rows, folder, device, args.num_workers):
        receipt.record_item(path.name, {"path": gate.rel(path), "sha256": guard.sha256_file(path)})
    receipt.complete({"images": len(rows)})
    print(f"v1_topup complete: {len(rows)} images, hashes in {gate.rel(guard.RECEIPT_PATH)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="an interrupted top-up is re-scored whole (146 images, minutes)")
    parser.add_argument("--rerun-reason", default=None)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
