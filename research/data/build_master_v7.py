"""Build the V7 master table and link table, run the overlap guard with planted copies, and write the receipt.

    python -m research.data.build_master_v7 [--workers 8] [--plants 48]

V7 data plan, build steps 2-4. Writes new files only: `ml/data/v7/master_v7.csv`, `ml/data/v7/links_v7.csv`,
`results/v7/data_gate_receipt.json`, `results/v7/data_gate_report.json`. No V6 file is edited, no frozen split is
recomputed. Labels are never loaded: the manifest is read with identifier columns only, so the master table carries no
diagnosis for any row (V7 loaders join labels from the manifest for training rows only). Pixels of evaluation and spent
rows (val, reserved, HAM val, HAM test, HIBA) are read to fingerprint them, which the plan requires; their labels are not.

Deviation from the plan, declared: the plan calls the master "a superset of manifest_v4.csv with existing columns
unchanged". It is a superset of the manifest's ROWS and identifier columns; label columns stay in the manifest, so that a
table every V7 entry point reads holds no test or reserved label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.data import fingerprints as fp  # noqa: E402
from research.data import grouping, overlap_guard, plants  # noqa: E402
from research.data.ids import normalise_id  # noqa: E402

MANIFEST = REPO / "ml" / "data" / "manifest_v4.csv"
ADDED = REPO / "results" / "v6" / "added_pool" / "added_train.csv"
EXTRAS = REPO / "results" / "v5" / "young_data" / "extra_train.csv"
HIBA = REPO / "results" / "v6" / "hiba" / "cohort.csv"
IMAGE_DIR = REPO / "data" / "external" / "isic2019_images" / "ISIC_2019_Training_Input"
EXTRA_DIR = REPO / "data" / "external" / "isic_young_v5"
HIBA_DIR = REPO / "data" / "external" / "hiba"
MIDAS_DIR = REPO / "data" / "external" / "midas"
MIDAS_FILES = MIDAS_DIR / "files.csv"
PATIENT_LINKS = REPO / "results" / "leak_audit" / "patient_links_rows.csv"
AUDIT_PAIRS = {"copies_bar40": (REPO / "results" / "leak_audit" / "pairs" / "copies_bar40.csv", "name_a", "name_b", "confirmed"),
               "sift_eval_hits": (REPO / "results" / "leak_audit" / "pairs" / "eval_hits.csv", "a", "b", "at_risk"),
               # M1 (12 Nov 2026): the adopted same-lesion matcher's links, one tier per row (`research/v7/m1/scan.py`). REQUIRED:
               # an absent file is a refusal, never a silently dropped detector (F12's lesson).
               "m1": (REPO / "results" / "v7" / "m1" / "m1_links.csv", "a", "b", None)}
REQUIRED_AUDIT = ("m1",)
OUT_DIR = REPO / "ml" / "data" / "v7"
MASTER = OUT_DIR / "master_v7.csv"
LINKS = OUT_DIR / "links_v7.csv"
RECEIPT = REPO / "results" / "v7" / "data_gate_receipt.json"
REPORT = REPO / "results" / "v7" / "data_gate_report.json"
PLANT_DIR = OUT_DIR / "plants_disposable"
PLANT_SEED = 7
MAX_PATIENT_CLIQUE = 60


def _strip(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip():
        return None
    text = str(value).strip()
    for prefix in ("__v6__", "__young__", "__singleton__"):
        if text.startswith(prefix):
            return None if prefix == "__singleton__" else text[len(prefix):]
    return text


def load_rows() -> pd.DataFrame:
    """Every row V7 could touch, identifier columns only."""
    from research.v4.s71_kfold import load_assignments

    cols = ["image_id", "archive", "lesion_id", "effective_lesion_id", "group_id", "split"]
    manifest = pd.read_csv(MANIFEST, usecols=cols, dtype=str, low_memory=False)
    folds = load_assignments()[["image_id", "fold"]].astype({"image_id": str})
    manifest = manifest.merge(folds, on="image_id", how="left")
    manifest["source"] = "isic2019_train"
    manifest["path"] = [str(IMAGE_DIR / f"{i}.jpg") for i in manifest["image_id"]]
    added = pd.read_csv(ADDED, usecols=["image_id", "archive", "lesion_id", "effective_lesion_id", "group_id", "image_path"], dtype=str)
    added = added.assign(source=added["archive"], split="train_added", fold=np.nan,
                         path=[str(REPO / p) for p in added["image_path"]]).drop(columns="image_path")
    extras = pd.read_csv(EXTRAS, usecols=["image_id", "archive", "lesion_id", "effective_lesion_id", "group_id"], dtype=str)
    extras = extras.assign(source="isic_young", split="train_extra", fold=np.nan,
                           path=[str(EXTRA_DIR / f"{i}.jpg") for i in extras["image_id"]])
    parts = [manifest, added, extras]
    if HIBA.is_file():
        hiba = pd.read_csv(HIBA, usecols=["isic_id", "lesion_key", "patient_id", "excluded"], dtype=str)
        # Kept rows have an EMPTY `excluded` cell. (Bug fixed 10 Nov: `.astype(str)` on missing cells did not give "nan" here,
        # so every HIBA row was silently dropped from the first gate runs. The per-source count below now stops a silent loss.)
        hiba = hiba[hiba["excluded"].isna() | hiba["excluded"].astype("string").str.strip().eq("")]
        hiba = pd.DataFrame({"image_id": "HIBA:" + hiba["isic_id"], "archive": "hiba", "lesion_id": hiba["lesion_key"],
                             "effective_lesion_id": hiba["lesion_key"], "group_id": "hiba:" + hiba["lesion_key"].fillna(hiba["isic_id"]),
                             "split": "train_hiba", "fold": np.nan, "source": "hiba", "patient_id": hiba["patient_id"],
                             "path": [str(HIBA_DIR / f"{i}.jpg") for i in hiba["isic_id"]]})
        parts.append(hiba)
    if MIDAS_FILES.is_file():
        # Sealed external cohort. Jupyter checkpoint copies (paths with a folder) are not data. Lesion ids come later from the
        # clinical table; until then each file is its own group, and identical files are joined by SHA-256 in the link step.
        files = pd.read_csv(MIDAS_FILES, dtype=str)
        files = files[~files["file_name"].str.contains("/")]
        # The id keeps the extension: three pairs of DIFFERENT files share a stem (.jpg vs .jpeg), and the normalised-id link
        # then joins each pair into one lesion group, so a pair is never counted as two independent cases.
        name = files["file_name"]
        patient = pd.Series(np.nan, index=name.index, dtype=object)
        clinical = MIDAS_DIR / "clinical.csv"
        if clinical.is_file():
            # Record id only (metadata). A record is patient-level (up to 42 images), and the data plan forbids patient ids
            # merging groups, so records become AT-RISK patient links here; the external read clusters by record itself.
            record = pd.read_csv(clinical, usecols=["midas_file_name", "midas_record_id"], dtype=str).drop_duplicates("midas_file_name")
            by_file = dict(zip(record["midas_file_name"], "midas_record:" + record["midas_record_id"]))
            patient = pd.Series([by_file.get(n, np.nan) for n in name], index=name.index, dtype=object)
        parts.append(pd.DataFrame({"image_id": "MIDAS:" + name, "archive": "midas", "lesion_id": np.nan, "effective_lesion_id": np.nan,
                                   "group_id": "midas:" + name, "split": "midas", "fold": np.nan, "source": "midas", "patient_id": patient,
                                   "path": [str(MIDAS_DIR / "images" / n) for n in files["file_name"]]}))
    for frame in parts:
        if frame.empty:
            raise AssertionError("a declared source loaded zero rows: a silent loss, not an empty source")
    rows = pd.concat(parts, ignore_index=True)
    rows["archive_id"] = [normalise_id(i.split(":", 1)[-1]) for i in rows["image_id"]]
    rows["src_lesion_id"] = [_strip(v) for v in rows["lesion_id"]]
    if "patient_id" not in rows:
        rows["patient_id"] = np.nan
    linked = pd.read_csv(PATIENT_LINKS, usecols=["image_id", "ap"], dtype=str) if PATIENT_LINKS.is_file() else pd.DataFrame(columns=["image_id", "ap"])
    patient = dict(zip(linked["image_id"], linked["ap"]))
    rows["src_patient_id"] = [patient.get(i, _strip(p)) for i, p in zip(rows["image_id"], rows["patient_id"])]
    keyed = patient_keyed(rows["group_id"])
    rows.loc[keyed & rows["src_patient_id"].isna(), "src_patient_id"] = rows.loc[keyed, "group_id"].map(_strip)
    rows["id_provenance"] = np.where(rows["source"] == "isic2019_train", "manifest_v4", rows["source"])
    if rows["image_id"].duplicated().any():
        raise AssertionError(f"{int(rows['image_id'].duplicated().sum())} image ids repeat across sources")
    return rows.drop(columns=["patient_id", "lesion_id"]).rename(columns={"split": "frozen_split"})


def _fingerprint(job: tuple[str, str]) -> tuple[str, fp.Fingerprint | None, str | None]:
    image_id, path = job
    try:
        return image_id, fp.fingerprint_file(image_id, path), None
    except Exception as error:  # noqa: BLE001 -- reported per row and counted by the guard's fill check
        return image_id, None, f"{type(error).__name__}: {error}"


def fingerprint_all(jobs: list[tuple[str, str]], workers: int) -> tuple[dict[str, fp.Fingerprint], dict[str, str]]:
    prints, errors = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (image_id, result, error) in enumerate(pool.map(_fingerprint, jobs, chunksize=64), 1):
            if result is None:
                errors[image_id] = error
            else:
                prints[image_id] = result
            if n % 5000 == 0:
                print(f"  fingerprinted {n:,}/{len(jobs):,}", flush=True)
    return prints, errors


def _pairs_within(keys: pd.Series, ids: pd.Series, tier: str, evidence: str, clique: bool) -> list[tuple[str, str, str, str]]:
    out = []
    frame = pd.DataFrame({"k": keys, "i": ids}).dropna()
    for _, members in frame.groupby("k")["i"]:
        m = sorted(set(members))
        if len(m) < 2:
            continue
        if clique and len(m) <= MAX_PATIENT_CLIQUE:
            out.extend((a, b, tier, evidence) for x, a in enumerate(m) for b in m[x + 1:])
        else:
            out.extend((m[0], b, tier, evidence) for b in m[1:])
    return out


def patient_keyed(group_ids: pd.Series) -> pd.Series:
    """True where an existing group id is a patient id (`__v6__IP_...`): V6 grouped DERM12345, which has no lesion ids, by
    patient. The data plan forbids patient ids from merging groups, so these become at-risk patient links instead."""
    return group_ids.astype(str).map(lambda g: (_strip(g) or "").startswith("IP_"))


def identifier_links(rows: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Tier 0 by identifier: same normalised id, same source lesion id or same existing lesion group (confirmed); same
    patient id, including V6's patient-keyed groups (at risk only, never merging groups)."""
    by_patient = patient_keyed(rows["group_id"])
    patients = rows["src_patient_id"].where(rows["src_patient_id"].notna(), rows["group_id"].where(by_patient).map(
        lambda g: _strip(g) if isinstance(g, str) else None))
    links = _pairs_within(rows["archive_id"], rows["image_id"], "confirmed", "normalised_id", clique=False)
    links += _pairs_within(rows["src_lesion_id"], rows["image_id"], "confirmed", "lesion_id", clique=False)
    links += _pairs_within(rows["group_id"].where(~by_patient), rows["image_id"], "confirmed", "existing_group", clique=False)
    links += _pairs_within(patients, rows["image_id"], "at_risk", "patient_id", clique=True)
    return links


def pixel_links(prints: dict[str, fp.Fingerprint]) -> list[tuple[str, str, str, str]]:
    ordered = [prints[k] for k in sorted(prints)]
    found = fp.match(ordered, ordered)
    seen, out = set(), []
    for hit in found:
        key = tuple(sorted((hit["query"], hit["reference"])))
        if key in seen:
            continue
        seen.add(key)
        # Only identical bytes confirm on pixels alone. pHash and dHash are both low-frequency global hashes and fail together:
        # a review by eye of the one oversized component (10 Nov, 34 rows) found small centred lesions in vignetted frames
        # matched at distance <= 4 across seven different HAM lesions. The data plan's rule applies: a hash match is a candidate,
        # confirmed only when a DIFFERENT detector (keypoints or band-pass correlation, `copies_bar40`) agrees.
        tier = "confirmed" if hit["evidence"] == "sha256" else "at_risk"
        out.append((key[0], key[1], tier, hit["evidence"]))
    return out


def audit_links(known: set[str]) -> tuple[list[tuple[str, str, str, str]], dict]:
    out, dropped = [], {}
    for name, (path, col_a, col_b, tier) in AUDIT_PAIRS.items():
        if not path.is_file():
            if name in REQUIRED_AUDIT:
                raise SystemExit(f"audit pairs {name} are required and {path} is absent: run the M1 scan first")
            dropped[name] = "file absent"
            continue
        columns = [col_a, col_b] + (["tier", "evidence"] if tier is None else [])
        pairs = pd.read_csv(path, usecols=columns, dtype=str)
        if tier is None and not pairs["tier"].isin(("confirmed", "at_risk")).all():
            raise SystemExit(f"audit pairs {name}: a tier outside confirmed / at_risk")
        inside = pairs[pairs[col_a].isin(known) & pairs[col_b].isin(known)]
        dropped[name] = {"pairs": int(len(pairs)), "kept": int(len(inside))}
        tiers = inside["tier"] if tier is None else pd.Series(tier, index=inside.index)
        evidence = inside["evidence"] if tier is None else pd.Series(name, index=inside.index)
        out.extend((a, b, t, e) for a, b, t, e in zip(inside[col_a], inside[col_b], tiers, evidence) if a != b)
    return out, dropped


def plant_control(rows: pd.DataFrame, prints: dict[str, fp.Fingerprint], n: int, workers: int) -> dict:
    """Plant n real development images (n/3 per archive) in eight families and require every plant to be found by the same
    link functions the real build uses. Plants never enter the master table."""
    rng = np.random.default_rng(PLANT_SEED)
    dev = rows[(rows["frozen_split"] == "train") & ~rows["image_id"].str.endswith("_downsampled") & rows["image_id"].isin(prints)]
    picks = pd.concat([g.iloc[rng.choice(len(g), size=min(len(g), n // 3), replace=False)] for _, g in dev.groupby("archive")])
    planted: list[plants.Plant] = []
    for row in picks.itertuples():
        planted += plants.make_plants(row.image_id, row.path, row.src_lesion_id, row.src_patient_id, PLANT_DIR)
    pixel_plants = [p for p in planted if p.path]
    plant_prints, errors = fingerprint_all([(p.image_id, p.path) for p in pixel_plants], workers)
    if errors:
        raise AssertionError(f"planted images failed to fingerprint: {errors}")
    found: set[tuple[str, str]] = set()
    for hit in fp.match([plant_prints[p.image_id] for p in pixel_plants], [prints[i] for i in picks["image_id"]]):
        found.add((hit["query"], hit["reference"]))
    plant_rows = pd.DataFrame({"image_id": [p.image_id for p in planted],
                               "archive_id": [normalise_id(p.image_id) for p in planted],
                               "src_lesion_id": [p.lesion_id for p in planted], "src_patient_id": [p.patient_id for p in planted],
                               "group_id": [f"plant:{p.image_id}" for p in planted]})
    both = pd.concat([picks[["image_id", "archive_id", "src_lesion_id", "src_patient_id", "group_id"]], plant_rows], ignore_index=True)
    # A plant's lesion link only counts when the source row has a lesion id to share; otherwise the plant carries a fresh one
    # and must be linked through the planted source id instead, so give the source that id for this control only.
    for p in planted:
        if p.family == "lesion_link" and p.lesion_id.startswith("PLANTLESION_"):
            both.loc[both["image_id"] == p.source_id, "src_lesion_id"] = p.lesion_id
        if p.family == "patient_link" and p.patient_id.startswith("PLANTPATIENT_"):
            both.loc[both["image_id"] == p.source_id, "src_patient_id"] = p.patient_id
    for a, b, _, _ in identifier_links(both):
        found.add((a, b))
    report = plants.plant_recall(planted, found)
    report["sources"] = int(len(picks))
    report["by_archive"] = {k: int(v) for k, v in picks["archive"].value_counts().items()}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the V7 master and link tables and run the data gate.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plants", type=int, default=48, help="real development images to plant (split evenly by archive)")
    parser.add_argument("--limit", type=int, default=grouping.PERCOLATION_LIMIT)
    args = parser.parse_args(argv)
    started = time.time()
    try:
        os.nice(10)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    rows = load_rows()
    present = rows["path"].map(os.path.isfile)
    print(f"{len(rows):,} rows; {int(present.sum()):,} image files present; by split {rows['frozen_split'].value_counts().to_dict()}", flush=True)
    prints, errors = fingerprint_all(list(zip(rows.loc[present, "image_id"], rows.loc[present, "path"])), args.workers)
    t_print = time.time() - started
    print(f"fingerprints: {len(prints):,} in {t_print / 60:.1f} min; {len(errors)} failed", flush=True)
    rows["sha256"] = rows["image_id"].map(lambda i: prints[i].sha256 if i in prints else None)
    rows["phash"] = rows["image_id"].map(lambda i: fp.hex64(prints[i].phash) if i in prints else None)
    rows["dhash"] = rows["image_id"].map(lambda i: fp.hex64(prints[i].dhash) if i in prints else None)

    links = identifier_links(rows)
    t0 = time.time()
    links += pixel_links(prints)
    t_match = time.time() - t0
    extra, audit = audit_links(set(rows["image_id"]))
    links += extra
    link_frame = pd.DataFrame(links, columns=["a", "b", "tier", "evidence"]).drop_duplicates()
    # one row per pair, keeping the strongest tier
    link_frame["pair"] = [tuple(sorted(p)) for p in zip(link_frame["a"], link_frame["b"])]
    link_frame = link_frame.sort_values("tier", key=lambda s: s.map({"confirmed": 0, "at_risk": 1})).drop_duplicates("pair")
    link_frame = link_frame.drop(columns="pair").reset_index(drop=True)

    confirmed = link_frame[link_frame["tier"] == "confirmed"]
    groups = grouping.components(rows["image_id"], zip(confirmed["a"], confirmed["b"]))
    rows["master_group"] = rows["image_id"].map(groups)
    method = confirmed.groupby("a")["evidence"].first().to_dict()
    rows["group_method"] = rows["image_id"].map(lambda i: method.get(i, "singleton_or_member"))
    rows["group_confidence"] = "confirmed"
    at_risk = link_frame[link_frame["tier"] == "at_risk"]
    counts = pd.concat([at_risk["a"], at_risk["b"]]).value_counts()
    rows["at_risk_links"] = rows["image_id"].map(counts).fillna(0).astype(int)
    rows["status"] = "kept"
    rows["status_reason"] = ""
    rows.loc[~present, ["status", "status_reason"]] = ["no_image_file", "image file absent on disk"]

    plant_report = plant_control(rows, prints, args.plants, args.workers)
    declared = int(len(rows))
    report = overlap_guard.check(rows, link_frame, declared, plant_report, args.limit)
    drops = overlap_guard.removals(rows, link_frame)
    rows.loc[rows["image_id"].isin(drops["global"]), ["status", "status_reason"]] = ["excluded_training_side", "linked to an evaluation or spent row"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "archive_id", "archive", "source", "src_lesion_id", "src_patient_id", "id_provenance", "group_id",
               "effective_lesion_id", "fold", "frozen_split", "path", "sha256", "phash", "dhash", "master_group", "group_method",
               "group_confidence", "at_risk_links", "status", "status_reason"]
    rows[columns].to_csv(MASTER, index=False)
    link_frame.to_csv(LINKS, index=False)
    report = overlap_guard.check(pd.read_csv(MASTER, dtype={"fold": "Int64"}, low_memory=False), pd.read_csv(LINKS, dtype=str),
                                 declared, plant_report, args.limit)
    summary = {k: v for k, v in report.items() if k not in ("removal_rows", "review_groups")}
    summary.update({"fingerprint_failures": len(errors), "fingerprint_minutes": round(t_print / 60, 2),
                    "match_minutes": round(t_match / 60, 2), "total_minutes": round((time.time() - started) / 60, 2),
                    "workers": args.workers, "audit_pairs": audit, "rows_by_split": rows["frozen_split"].value_counts().to_dict(),
                    "links_by_evidence": link_frame.groupby(["tier", "evidence"]).size().rename("n").reset_index().to_dict("records"),
                    "radii": {"confirmed_both_hashes_le": fp.CONFIRMED_RADIUS, "at_risk_both_hashes_le": fp.AT_RISK_RADIUS,
                              "source": "results/v7/hash_radius_probe.json", "status": "provisional; fixed at plan step 5"}})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    overlap_guard.write_receipt(RECEIPT, MASTER, LINKS, report)
    print(json.dumps({k: summary[k] for k in ("passed", "failures", "rows_compared", "links", "components", "removals", "total_minutes")},
                     indent=1, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
