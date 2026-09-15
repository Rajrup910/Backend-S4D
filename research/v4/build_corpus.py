"""S49 -- the V4 corpus: taxonomy, merge, dedupe, grouping, and the reserved cohort.

The data-side work, and the highest-risk step in V4: every number the workstream produces is
downstream of this file being right.

**Taxonomy.** V4 moves to the 8-class ISIC-2019 label space (MEL, NV, BCC, AK, BKL, DF, VASC,
SCC). It keeps the 628 SCC images that V3's pre-registered `scc` exclusion had to discard, it
makes the escalation set clinically stateable (MEL / BCC / SCC / AK), and it is the space the
external archives actually ship in. Every row also carries a **7-class collapse** (AK + SCC ->
`akiec`) so that any V4 number can be reported beside its V1-V3 counterpart. `ml/configs/
class_mapping.json` anticipated exactly this and its `planned_extension` block is the authority
for SCC being a separate class rather than folded into `akiec`: HAM's `akiec` is actinic keratosis
and *in situ* carcinoma, and invasive SCC is a different entity.

**Three leakage controls, all required.**

1. *ISIC-2019 contains HAM10000 in full* -- 10,015 of 25,331 rows join exactly on `image_id`. The
   3,034 HAM validation and test images keep their `split_v1.csv` assignment byte-identically and
   are excluded from every V4 training split. This is what keeps V1-V3 comparisons legal.
2. *`lesion_id` is null for 2,084 rows.* Nulls become **singleton** groups, never one giant null
   group, reusing S14's `effective_lesion_id` pattern rather than reinventing it.
3. *Near-duplicates across archives* -- `research/v4/dedupe.py`, never run in this project before.

**The grouping key is the union of (2) and (3), not either alone.** Two images can share a
duplicate cluster while carrying different `lesion_id`s -- that is precisely the case
`lesion_id` grouping cannot catch -- so `group_id` is the connected component of both relations
together, computed by union-find. Splitting on `effective_lesion_id` while merely *reporting* the
duplicate clusters would leave the leak in place.

A group that contains any HAM val or test image is **assigned wholesale to that split**. A
non-HAM image that is a near-duplicate of a HAM test image is a test image, whatever its id says.

**Reserved evaluation cohort.** Carved from the non-HAM pool only, grouped, stratified on age band
x escalation, and sized against the S48 power target of **104-131 under-40 escalating lesions**
(`results/v4/power_audit.json`; the runbook's original ">= 150" is unreachable -- the whole pool
holds 151 and reserving 150 leaves one to train on). This gives V4 a primary endpoint that does
not need the HAM test split at all.

    $py -m research.v4.build_corpus
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.paths import resolve
from ml.preprocessing.split_dataset import assert_no_leakage
from research.experiment_log import log_experiment
from research.v4.dedupe import CLUSTER_RADIUS, cluster

SESSION = "v4_s49"
METHOD_PREFIX = "S49_corpus"   # narrow: prune must not reach archive_probe rows
SEED = 20260915

SOURCES = {
    "isic_gt": "data/external/ISIC_2019_Training_GroundTruth.csv",
    "isic_meta": "data/external/ISIC_2019_Training_Metadata.csv",
    "ham_meta": "data/ham10000/HAM10000_metadata.csv",
    "split_v1": "ml/configs/splits/split_v1.csv",
    "nonham_manifest": "data/external/manifest_isic2019_nonham.csv",
    "power_audit": "results/v4/power_audit.json",
}

MANIFEST_OUT = "ml/data/manifest_v4.csv"
SPLIT_DIR = "ml/configs/splits/v4"
REPORT_OUT = "results/v4/corpus_report.json"
IMAGE_DIR = "data/external/isic2019_images/ISIC_2019_Training_Input"

#: 8-class ISIC-2019 taxonomy, in a fixed index order.
CLASSES_8 = ("akiec_ak", "bcc", "bkl", "df", "mel", "nv", "scc", "vasc")
#: The 7-class collapse that keeps every V4 number comparable to V1-V3.
COLLAPSE_7 = {"akiec_ak": "akiec", "scc": "akiec", "bcc": "bcc", "bkl": "bkl",
              "df": "df", "mel": "mel", "nv": "nv", "vasc": "vasc"}
CLASSES_7 = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

#: ISIC column -> our 8-class code.
ISIC_TO_CODE = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec_ak",
                "BKL": "bkl", "DF": "df", "VASC": "vasc", "SCC": "scc"}

ESCALATING_8 = ("mel", "bcc", "akiec_ak", "scc")
ESCALATING_7 = ("mel", "bcc", "akiec")

#: Share of each non-HAM stratum held out for evaluation...
RESERVED_FRACTION = 0.30
V4_VAL_FRACTION = 0.15

#: ...except the one stratum that is allocated by COUNT rather than by share. A flat 30% puts
#: only 45 under-40 escalating lesions in the reserved cohort, and S48 showed the primary
#: endpoint needs 104 for its interval to be narrower than the declared MCID. This stratum is
#: therefore filled to the S48 requirement first and the remainder goes to training -- which is
#: the trade S48 named: the endpoint is only estimable if the cases that carry it are held out,
#: and they are the same cases the model would most like to train on. The reserved count is read
#: from results/v4/power_audit.json rather than written here.
SCARCE_STRATUM = "<40|True"


def age_band(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return "60+"


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------

class _Union:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_manifest() -> tuple[pd.DataFrame, dict]:
    truth = pd.read_csv(resolve(SOURCES["isic_gt"]))
    meta = pd.read_csv(resolve(SOURCES["isic_meta"]))
    ham = pd.read_csv(resolve(SOURCES["ham_meta"]))
    splits = pd.read_csv(resolve(SOURCES["split_v1"]))
    nonham = pd.read_csv(resolve(SOURCES["nonham_manifest"]))[["image", "source"]]

    label_cols = [c for c in truth.columns if c != "image"]
    winner = truth[label_cols].idxmax(axis=1)
    frame = pd.DataFrame({"image_id": truth["image"], "isic_label": winner})
    n_unk = int((frame["isic_label"] == "UNK").sum())
    frame = frame[frame["isic_label"] != "UNK"].copy()
    frame["class_8"] = frame["isic_label"].map(ISIC_TO_CODE)
    frame["class_7"] = frame["class_8"].map(COLLAPSE_7)
    frame["class_index_8"] = frame["class_8"].map({c: i for i, c in enumerate(CLASSES_8)})
    frame["class_index_7"] = frame["class_7"].map({c: i for i, c in enumerate(CLASSES_7)})
    frame["escalating_8"] = frame["class_8"].isin(ESCALATING_8)
    frame["escalating_7"] = frame["class_7"].isin(ESCALATING_7)

    frame = frame.merge(meta, left_on="image_id", right_on="image", how="left").drop(
        columns=["image"])
    frame["age_band"] = frame["age_approx"].map(age_band)

    # Archive. HAM is the 10,015-row intersection; the rest is labelled by S13's manifest.
    frame["in_ham"] = frame["image_id"].isin(set(ham["image_id"]))
    source = nonham.set_index("image")["source"]
    frame["archive"] = np.where(frame["in_ham"], "ham", frame["image_id"].map(source))

    # Control 2: null lesion ids become singletons, never one shared null group.
    raw = frame["lesion_id"].fillna("").astype(str).str.strip()
    blank = raw == ""
    frame["effective_lesion_id"] = raw.where(~blank, "__singleton__" + frame["image_id"])
    frame["lesion_id_was_null"] = blank

    # HAM's own lesion ids are authoritative where ISIC's metadata is missing them.
    ham_lesion = ham.set_index("image_id")["lesion_id"]
    from_ham = frame["image_id"].map(ham_lesion)
    frame["effective_lesion_id"] = np.where(
        from_ham.notna(), from_ham, frame["effective_lesion_id"])

    stats = {
        "isic_rows": int(len(truth)),
        "dropped_unk": n_unk,
        "manifest_rows": int(len(frame)),
        "in_ham": int(frame["in_ham"].sum()),
        "non_ham": int((~frame["in_ham"]).sum()),
        "null_lesion_ids": int(blank.sum()),
        "archives": {k: int(v) for k, v in frame["archive"].value_counts().items()},
        "class_8": {k: int(v) for k, v in frame["class_8"].value_counts().items()},
        "class_7": {k: int(v) for k, v in frame["class_7"].value_counts().items()},
    }
    return frame, stats


def add_groups(frame: pd.DataFrame, radius: int = CLUSTER_RADIUS) -> tuple[pd.DataFrame, dict]:
    """`group_id` = connected component of (same effective lesion) OR (same duplicate cluster)."""
    clusters = cluster(radius=radius)
    dup_stats = clusters.attrs["stats"]
    isic = clusters[clusters["archive"] == "isic2019"].set_index("image_id")["dup_cluster"]
    frame = frame.assign(dup_cluster=frame["image_id"].map(isic))

    union = _Union(frame["effective_lesion_id"].unique().tolist())
    # Join every pair of lesion ids that share a duplicate cluster.
    multi = frame.groupby("dup_cluster")["effective_lesion_id"].unique()
    merged_by_dupes = 0
    for lesions in multi:
        if len(lesions) > 1:
            for other in lesions[1:]:
                if union.find(lesions[0]) != union.find(other):
                    merged_by_dupes += 1
                union.union(lesions[0], other)

    frame["group_id"] = frame["effective_lesion_id"].map(union.find)
    stats = {
        "duplicate_detection": dup_stats,
        "n_effective_lesions": int(frame["effective_lesion_id"].nunique()),
        "n_groups": int(frame["group_id"].nunique()),
        "lesion_merges_caused_by_duplicates": int(merged_by_dupes),
        "note": (
            "group_id is the union of lesion identity and perceptual duplication. The difference "
            "between n_effective_lesions and n_groups is exactly the leak that lesion_id "
            "grouping alone would have missed."
        ),
    }
    return frame, stats


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------

def assign_splits(frame: pd.DataFrame, under40_target: int,
                  seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """ham_test / ham_val inherited; reserved / val / train carved from the non-HAM pool.

    Groups, not images, are assigned. A group holding any HAM val or test image is assigned
    wholesale to that split, so a non-HAM near-duplicate of a HAM test image is treated as a test
    image regardless of what its own metadata says.
    """
    v1 = pd.read_csv(resolve(SOURCES["split_v1"])).set_index("image_id")["split"]
    frame = frame.assign(ham_split=frame["image_id"].map(v1))

    group_ham = (frame.assign(rank=frame["ham_split"].map({"test": 3, "val": 2, "train": 1}))
                 .groupby("group_id")["rank"].max())
    locked = {"ham_test": set(group_ham[group_ham == 3].index),
              "ham_val": set(group_ham[group_ham == 2].index),
              "ham_train": set(group_ham[group_ham == 1].index)}

    free = [g for g in frame["group_id"].unique()
            if g not in locked["ham_test"] and g not in locked["ham_val"]
            and g not in locked["ham_train"]]

    # Stratify the free (non-HAM) groups on age band x escalation, which is what the reserved
    # cohort has to be balanced on for the primary endpoint to be estimable within each band.
    per_group = (frame[frame["group_id"].isin(free)]
                 .groupby("group_id")
                 .agg(age_band=("age_band", "first"),
                      escalating=("escalating_8", "max"))
                 .reset_index())
    per_group["stratum"] = per_group["age_band"] + "|" + per_group["escalating"].astype(str)

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    allocation: dict[str, dict] = {}
    for stratum, part in per_group.groupby("stratum"):
        ids = part["group_id"].to_numpy()
        rng.shuffle(ids)
        if stratum == SCARCE_STRATUM:
            # Fill to the S48 requirement by count, then leave the rest trainable. No validation
            # slice: at this scarcity a third split would leave neither half usable, and V4
            # selects methods on the pooled val split, never on this stratum alone.
            n_res, n_val = min(under40_target, len(ids)), 0
        else:
            n_res = int(round(len(ids) * RESERVED_FRACTION))
            n_val = int(round(len(ids) * V4_VAL_FRACTION))
        for gid in ids[:n_res]:
            assignment[gid] = "reserved"
        for gid in ids[n_res:n_res + n_val]:
            assignment[gid] = "val"
        for gid in ids[n_res + n_val:]:
            assignment[gid] = "train"
        allocation[stratum] = {"groups": int(len(ids)), "reserved": int(n_res),
                               "val": int(n_val), "train": int(len(ids) - n_res - n_val)}

    def resolve_split(gid):
        if gid in locked["ham_test"]:
            return "ham_test"
        if gid in locked["ham_val"]:
            return "ham_val"
        if gid in locked["ham_train"]:
            return "train"
        return assignment[gid]

    frame["split"] = frame["group_id"].map(resolve_split)

    counts = {k: int(v) for k, v in frame["split"].value_counts().items()}
    under40_esc = (frame[(frame["age_band"] == "<40") & frame["escalating_8"]]
                   .groupby("split")
                   .agg(images=("image_id", "size"), lesions=("group_id", "nunique")))
    stats = {
        "seed": seed,
        "reserved_fraction": RESERVED_FRACTION,
        "val_fraction": V4_VAL_FRACTION,
        "scarce_stratum": SCARCE_STRATUM,
        "under40_reserved_target": int(under40_target),
        "allocation_per_stratum": allocation,
        "images_per_split": counts,
        "groups_per_split": {k: int(v) for k, v in
                             frame.groupby("split")["group_id"].nunique().items()},
        "under40_escalating_by_split": {
            split: {"images": int(row["images"]), "lesions": int(row["lesions"])}
            for split, row in under40_esc.iterrows()
        },
        "ham_groups_locked": {k: len(v) for k, v in locked.items()},
    }
    return frame, stats


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

def verify(frame: pd.DataFrame, power: dict) -> dict:
    checks: dict[str, object] = {}

    graded = frame[frame["split"].isin(("train", "val", "reserved"))].copy()
    assert_no_leakage(graded.rename(columns={"group_id": "grp"}), "grp")
    checks["assert_no_leakage_v4_splits"] = "PASS"

    full = frame.rename(columns={"group_id": "grp"})
    assert_no_leakage(full, "grp")
    checks["assert_no_leakage_all_splits"] = "PASS"

    v1 = pd.read_csv(resolve(SOURCES["split_v1"])).set_index("image_id")["split"]
    held = frame[frame["image_id"].isin(v1[v1 != "train"].index)]
    checks["ham_val_test_images"] = int(len(held))
    checks["ham_val_test_in_v4_train"] = int((held["split"] == "train").sum())
    checks["ham_val_test_split_preserved"] = bool(
        (held["split"].str.replace("ham_", "", regex=False)
         == held["image_id"].map(v1)).all())

    groups = frame.groupby("group_id")["split"].nunique()
    checks["groups_spanning_splits"] = int((groups > 1).sum())
    dup = frame.dropna(subset=["dup_cluster"]).groupby("dup_cluster")["split"].nunique()
    checks["duplicate_clusters_spanning_splits"] = int((dup > 1).sum())

    image_dir = resolve(IMAGE_DIR)
    sample = frame["image_id"].sample(min(500, len(frame)), random_state=0)
    checks["sampled_paths_resolved"] = int(
        sum((image_dir / f"{i}.jpg").is_file() for i in sample))
    checks["sampled_paths_checked"] = int(len(sample))

    reserved = frame[(frame["split"] == "reserved") & (frame["age_band"] == "<40")
                     & frame["escalating_8"]]
    n_lesions = int(reserved["group_id"].nunique())
    target_lo = power["falsifier"]["required_for_single_arm_ci"]
    target_hi = power["falsifier"]["available_under40_escalating_lesions"] - 20
    checks["reserved_under40_escalating_lesions"] = n_lesions
    checks["reserved_under40_escalating_images"] = int(len(reserved))
    checks["s48_target_range"] = [target_lo, target_hi]
    checks["meets_s48_power_target"] = bool(n_lesions >= target_lo)
    checks["meets_paired_mcnemar_target"] = bool(
        n_lesions >= power["falsifier"]["required_for_paired_mcnemar_loss_ratio_05"])
    return checks


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def prune_prior_rows(path: str = "research/experiments.csv") -> int:
    import csv
    target = resolve(path)
    if not target.is_file():
        return 0
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames or [], list(reader)
    kept = [r for r in rows if not (r.get("session") == SESSION
                                    and str(r.get("method", "")).startswith(METHOD_PREFIX))]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="S49 -- build the V4 corpus.")
    parser.add_argument("--radius", type=int, default=CLUSTER_RADIUS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    power = json.loads(resolve(SOURCES["power_audit"]).read_text(encoding="utf-8"))

    print("S49 -- building the V4 corpus")
    frame, manifest_stats = build_manifest()
    print(f"  manifest    {manifest_stats['manifest_rows']} rows "
          f"({manifest_stats['in_ham']} in HAM, {manifest_stats['non_ham']} non-HAM), "
          f"{manifest_stats['dropped_unk']} UNK dropped")

    frame, group_stats = add_groups(frame, radius=args.radius)
    print(f"  grouping    {group_stats['n_effective_lesions']} effective lesions -> "
          f"{group_stats['n_groups']} groups "
          f"({group_stats['lesion_merges_caused_by_duplicates']} merges forced by duplicates)")

    target = power["falsifier"]["required_for_single_arm_ci"]
    frame, split_stats = assign_splits(frame, under40_target=target, seed=args.seed)
    print(f"  splits      {split_stats['images_per_split']}")

    checks = verify(frame, power)
    print(f"  under-40 escalating in reserved: "
          f"{checks['reserved_under40_escalating_lesions']} lesions / "
          f"{checks['reserved_under40_escalating_images']} images "
          f"(S48 target {checks['s48_target_range']}) -> "
          f"meets={checks['meets_s48_power_target']}")

    columns = ["image_id", "class_8", "class_7", "class_index_8", "class_index_7",
               "escalating_8", "escalating_7", "age_approx", "age_band", "sex",
               "anatom_site_general", "archive", "in_ham", "lesion_id",
               "effective_lesion_id", "lesion_id_was_null", "dup_cluster", "group_id",
               "ham_split", "split"]
    out = frame[columns].sort_values("image_id")
    resolve(MANIFEST_OUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(resolve(MANIFEST_OUT), index=False)

    split_dir = resolve(SPLIT_DIR)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "reserved"):
        out[out["split"] == name].to_csv(split_dir / f"split_v4_{name}.csv", index=False)

    report = {
        "session": "S49",
        "phase": "U_corpus",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": SOURCES,
        "test_read": False,
        "taxonomy": {"classes_8": list(CLASSES_8), "classes_7": list(CLASSES_7),
                     "collapse": COLLAPSE_7, "escalating_8": list(ESCALATING_8)},
        "manifest": manifest_stats,
        "grouping": group_stats,
        "splits": split_stats,
        "verification": checks,
    }
    resolve(REPORT_OUT).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    failed = [k for k in ("assert_no_leakage_v4_splits", "assert_no_leakage_all_splits")
              if checks[k] != "PASS"]
    hard = (checks["ham_val_test_in_v4_train"] == 0
            and checks["groups_spanning_splits"] == 0
            and checks["duplicate_clusters_spanning_splits"] == 0
            and checks["sampled_paths_resolved"] == checks["sampled_paths_checked"]
            and not failed)
    print(f"  verification: leakage={checks['groups_spanning_splits']} groups, "
          f"{checks['duplicate_clusters_spanning_splits']} dup clusters span splits; "
          f"{checks['ham_val_test_in_v4_train']} HAM val/test rows in V4 train; "
          f"paths {checks['sampled_paths_resolved']}/{checks['sampled_paths_checked']}")
    print(f"  ALL HARD CHECKS {'PASS' if hard else 'FAIL'}")

    if not args.no_ledger:
        removed = prune_prior_rows()
        if removed:
            print(f"  (replaced {removed} S49 ledger row(s))")
        log_experiment({
            "session": SESSION,
            "method": METHOD_PREFIX,
            "split": "v4_train+val+reserved",
            "notes": (
                f"Phase U; 8-class ISIC-2019 ({manifest_stats['manifest_rows']} images, "
                f"{manifest_stats['dropped_unk']} UNK dropped), 7-class collapse carried; "
                f"{group_stats['n_effective_lesions']} lesions -> {group_stats['n_groups']} "
                f"groups after perceptual dedupe (radius {args.radius}, "
                f"{group_stats['lesion_merges_caused_by_duplicates']} merges); splits "
                f"{split_stats['images_per_split']}; 0 leaks; reserved <40 escalating "
                f"{checks['reserved_under40_escalating_lesions']} lesions vs S48 target "
                f"{checks['s48_target_range']}; no test read"
            ),
        })
        print(f"  ledger      1 row under session={SESSION}")
    print(f"  written     {resolve(MANIFEST_OUT)}")
    return 0 if hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
