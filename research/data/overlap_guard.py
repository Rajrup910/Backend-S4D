"""The V7 overlap guard (V7 data plan, "Required check"). Every V7 entry point calls `require_receipt` before it reads a row.

It fails hard on: an unnormalised id; a missing or mostly empty field; fewer rows compared than declared; any evaluated row
with a confirmed or at-risk link to its training set (after the training-side removals below); an empty set being compared;
a planted copy not found; a component larger than the percolation limit.

Removals always take the TRAINING-side row, so evaluation rows and frozen splits never move:

* a training-only row (added pool, young extras) linked to any evaluation or spent row is dropped from all training;
* a development row linked to a spent row (HAM test, reserved, val, HAM val, HIBA) is dropped from all training;
* a development row linked to a row in fold f is dropped from training when fold f is held out (`exclude_by_fold`);
* for leave-one-archive-out, rows linked to the held-out archive and added rows that came from it are dropped
  (`exclude_by_holdout`). The added "ISIC 2019 Test" rows are its Barcelona part (`docs/V6_RUNSHEET.md`), so they leave
  the training set whenever BCN20000 is held out.

The receipt stores the SHA-256 of the master table, the link table and this package's code, plus every check's result.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from research.data import grouping
from research.data.ids import is_normalised, normalise_id

REQUIRED_COLUMNS = ("image_id", "archive_id", "archive", "source", "frozen_split", "fold", "master_group",
                    "group_method", "sha256", "phash", "dhash", "status")
FILLED_FIELDS = ("archive_id", "archive", "source", "frozen_split", "master_group", "sha256", "phash", "dhash", "status")
FILL_FLOOR = 0.99
DEV_SPLIT = "train"
#: Training-only from V7 (owner, 10 Nov 2026): the added pool, the young extras, HIBA, and the retired evaluation splits of V1-V4
#: (`reserved` read once at S63, `val` and `ham_val` selection sets). `frozen_split` values stay verbatim; only their role changes.
TRAINING_ONLY_SPLITS = ("train_added", "train_extra", "train_hiba", "reserved", "val", "ham_val")
SPENT_SPLITS = ("ham_test", "midas")  # never trained on: V1's published test set, and MIDAS, the sole sealed external read
DEV_FOLDS = (0, 1, 2, 3, 4)
ARCHIVES = ("ham", "bcn20000", "mskcc")
#: Added sources whose images come from a development archive's institution (declared, `docs/V6_RUNSHEET.md` line 523).
SOURCE_ARCHIVE_OF_ORIGIN = {"isic2019_test": "bcn20000"}
LINK_COLUMNS = ("a", "b", "tier", "evidence")
TIERS = ("confirmed", "at_risk")
CODE_DIR = Path(__file__).resolve().parent


class GuardFailure(SystemExit):
    """Raised when the data gate does not pass; the message names every failed check."""


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha() -> dict[str, str]:
    return {p.name: sha256_path(p) for p in sorted(CODE_DIR.glob("*.py"))}


def _adjacency(master: pd.DataFrame, links: pd.DataFrame) -> dict[str, set[str]]:
    """Each row's neighbours: every row it shares a master group with, every row it has a link to (either tier), and every
    member of that linked row's group. Group membership matters because confirmed links chain: a-b and b-c put a and c in
    one lesion group. A link reaches the whole group because a row that may be the same lesion as one image of a lesion may
    be the same lesion as all of them: before 12 Nov an at-risk link to image b of lesion {b, c} left a training row in
    place while c was evaluated (710 rows across all training on the gate of 12 Nov 14:42). At-risk links never chain to
    each other (that would percolate); only through a group of confirmed members."""
    adjacent: dict[str, set[str]] = defaultdict(set)
    group = dict(zip(master["image_id"].astype(str), master["master_group"].astype(str)))
    members: dict[str, set[str]] = {g: set(rows.astype(str)) for g, rows in master.groupby(master["master_group"].astype(str))["image_id"]}
    for a, b in zip(links["a"].astype(str), links["b"].astype(str)):
        for x, y in ((a, b), (b, a)):           # symmetric: x reaches y's group, and every member of y's group reaches x
            for z in members.get(group.get(y), {y}):
                if z != x:
                    adjacent[x].add(z)
                    adjacent[z].add(x)
    for rows in members.values():
        if len(rows) > 1:
            for r in rows:
                adjacent[r] |= rows - {r}
    return adjacent


def removals(master: pd.DataFrame, links: pd.DataFrame) -> dict:
    """Training-side rows to remove, globally, per held-out fold and per held-out archive."""
    split = dict(zip(master["image_id"].astype(str), master["frozen_split"].astype(str)))
    fold = dict(zip(master["image_id"].astype(str), master["fold"]))
    archive = dict(zip(master["image_id"].astype(str), master["archive"].astype(str)))
    source = dict(zip(master["image_id"].astype(str), master["source"].astype(str)))
    adjacent = _adjacency(master, links)
    training_side = {r for r, s in split.items() if s == DEV_SPLIT or s in TRAINING_ONLY_SPLITS}
    global_drop: set[str] = set()
    by_fold: dict[int, set[str]] = {f: set() for f in DEV_FOLDS}
    for row in training_side:
        for other in adjacent.get(row, ()):
            if split[other] in SPENT_SPLITS:
                global_drop.add(row)    # never-train rows are never in training: out of every run
            elif split[other] == DEV_SPLIT and (split[row] in TRAINING_ONLY_SPLITS or fold[other] != fold[row]):
                # out of the run whose held-out fold holds `other`. A training-only row linked to a development row was
                # removed from EVERY run until 12 Nov; it can only leak into the run that evaluates that row (every V7
                # fold model is read on its own fold only; holdouts have their own list below), so it stays in the other
                # four (about 1,300 rows per fold on the gate of 12 Nov 15:46).
                by_fold[int(fold[other])].add(row)
    by_holdout: dict[str, set[str]] = {a: set() for a in ARCHIVES}
    for row in training_side:
        origin = SOURCE_ARCHIVE_OF_ORIGIN.get(source[row])
        if origin:
            by_holdout[origin].add(row)
        if split[row] in TRAINING_ONLY_SPLITS and archive[row] in by_holdout:
            # A training-only row FROM the held-out archive (retired `reserved`/`val`/`ham_val` rows are BCN, MSKCC or HAM
            # images). Found 10 Nov by a BCN-holdout smoke: without this, 5,689 BCN images stayed in BCN-held-out training.
            by_holdout[archive[row]].add(row)
        for other in adjacent.get(row, ()):
            if split[other] == DEV_SPLIT and archive[other] != archive[row] and archive[other] in by_holdout:
                by_holdout[archive[other]].add(row)
    return {"global": sorted(global_drop), "by_fold": {f: sorted(v - global_drop) for f, v in by_fold.items()},
            "by_holdout": {a: sorted(v - global_drop) for a, v in by_holdout.items()}}


def training_rows(master: pd.DataFrame, drops: dict, fold: int | None = None, holdout: str | None = None) -> pd.DataFrame:
    """The rows a V7 run may train on, after the guard's removals. Exactly one of fold / holdout is given."""
    if (fold is None) == (holdout is None):
        raise ValueError("give exactly one of fold or holdout")
    split = master["frozen_split"].astype(str)
    keep = split.isin((DEV_SPLIT, *TRAINING_ONLY_SPLITS)) & ~master["image_id"].astype(str).isin(drops["global"])
    if fold is not None:
        keep &= ~((split == DEV_SPLIT) & (master["fold"] == fold))
        keep &= ~master["image_id"].astype(str).isin(drops["by_fold"][fold])
    else:
        keep &= master["archive"].astype(str) != holdout
        keep &= ~master["image_id"].astype(str).isin(drops["by_holdout"][holdout])
    return master[keep]


def check(master: pd.DataFrame, links: pd.DataFrame, declared_rows: int, plant_report: dict | None,
          limit: int = grouping.PERCOLATION_LIMIT) -> dict:
    """Run every guard check; returns the report. `GuardFailure` is raised by `enforce`, not here, so tests can read it."""
    failures: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        return {"passed": False, "failures": [f"master table lacks columns {missing}"]}
    if master.empty:
        return {"passed": False, "failures": ["master table is empty: an intersection on empty sets proves nothing"]}
    if len(master) < declared_rows:
        failures.append(f"{len(master):,} rows compared, fewer than the {declared_rows:,} declared")
    ids = master["image_id"].astype(str)
    if ids.duplicated().any():
        failures.append(f"{int(ids.duplicated().sum())} repeated image ids")
    bad = master.loc[~master["archive_id"].map(is_normalised), "archive_id"]
    if len(bad):
        failures.append(f"{len(bad)} unnormalised archive ids, e.g. {bad.iloc[0]!r}")
    expected = ids.map(lambda s: normalise_id(s.split(":", 1)[-1]))  # a source prefix such as `HIBA:` is not part of the id
    if not (expected == master["archive_id"].astype(str)).all():
        failures.append("archive_id is not normalise_id(image_id) on every row")
    collide = master.groupby("archive_id")["master_group"].nunique()
    if (collide > 1).any():
        failures.append(f"{int((collide > 1).sum())} normalised ids span more than one master group, e.g. {collide[collide > 1].index[0]}")
    fill = {c: float((master[c].notna() & (master[c].astype(str).str.strip().str.len() > 0)).mean()) for c in FILLED_FIELDS}
    for column, share in fill.items():
        if share < FILL_FLOOR:
            failures.append(f"field {column} is filled on {share:.1%} of rows (floor {FILL_FLOOR:.0%})")
    dev_fold = master.loc[master["frozen_split"] == DEV_SPLIT, "fold"]
    if dev_fold.isna().any() or not set(dev_fold.astype(int)) <= set(DEV_FOLDS):
        failures.append("a development row has no fold, or a fold outside 0-4")
    for name, rows in (("development", master["frozen_split"] == DEV_SPLIT),
                       ("training-only", master["frozen_split"].isin(TRAINING_ONLY_SPLITS))):
        if not rows.any():
            failures.append(f"the {name} set is empty: an intersection on empty sets proves nothing")

    link_missing = [c for c in LINK_COLUMNS if c not in links.columns]
    if link_missing:
        failures.append(f"link table lacks columns {link_missing}")
        return {"passed": False, "failures": failures, "fill": fill}
    unknown_tier = sorted(set(links["tier"]) - set(TIERS))
    if unknown_tier:
        failures.append(f"unknown link tiers {unknown_tier}")
    known = set(ids)
    stray = links.loc[~links["a"].astype(str).isin(known) | ~links["b"].astype(str).isin(known)]
    if len(stray):
        failures.append(f"{len(stray)} links name rows that are not in the master table")
        return {"passed": False, "failures": failures, "fill": fill}

    confirmed = links[links["tier"] == "confirmed"]
    groups = grouping.components(ids, zip(confirmed["a"].astype(str), confirmed["b"].astype(str)))
    split_of = dict(zip(ids, master["frozen_split"].astype(str)))
    attributes = {r: {"archive": a, "frozen_split": s} for r, a, s in zip(ids, master["archive"].astype(str), master["frozen_split"].astype(str))}
    sizes = grouping.size_distribution(groups)
    if sizes["largest"] > limit:
        failures.append(f"percolation: the largest confirmed component has {sizes['largest']} rows (limit {limit})")
    review = grouping.review_list(groups, attributes, limit)
    # master_group must be at least as coarse as the confirmed components: two confirmed rows never sit in different groups
    group_of = dict(zip(ids, master["master_group"].astype(str)))
    groups_per_component: dict[str, set[str]] = defaultdict(set)
    for row, component in groups.items():
        groups_per_component[component].add(group_of[row])
    split_groups = sum(len(v) > 1 for v in groups_per_component.values())
    if split_groups:
        failures.append(f"{split_groups} confirmed components are split across master groups")

    drops = removals(master, links)
    adjacent = _adjacency(master, links)
    residual: list[str] = []
    for f in DEV_FOLDS:
        evaluated = set(master.loc[(master["frozen_split"] == DEV_SPLIT) & (master["fold"] == f), "image_id"].astype(str))
        train = set(training_rows(master, drops, fold=f)["image_id"].astype(str))
        leaks = [(r, o) for r in evaluated for o in adjacent.get(r, ()) if o in train]
        if leaks:
            residual.append(f"fold {f}: {len(leaks)} links from evaluated rows to training rows remain, e.g. {leaks[0]}")
        shared = {group_of[r] for r in evaluated} & {group_of[r] for r in train}
        if shared:
            residual.append(f"fold {f}: {len(shared)} master groups sit on both sides, e.g. {sorted(shared)[0]}")
    for holdout in ARCHIVES:
        evaluated = set(master.loc[(master["frozen_split"] == DEV_SPLIT) & (master["archive"] == holdout), "image_id"].astype(str))
        if not evaluated:
            continue
        # The trainer applies the removal LISTS (not `training_rows`), so the lists alone must clear the held-out archive.
        same_archive = set(master.loc[master["frozen_split"].isin(TRAINING_ONLY_SPLITS) & (master["archive"] == holdout), "image_id"].astype(str))
        uncovered = same_archive - set(drops["global"]) - set(drops["by_holdout"][holdout])
        if uncovered:
            residual.append(f"holdout {holdout}: {len(uncovered)} training-only rows from the held-out archive are not in the removal lists")
        train = set(training_rows(master, drops, holdout=holdout)["image_id"].astype(str))
        leaks = [(r, o) for r in evaluated for o in adjacent.get(r, ()) if o in train]
        if leaks:
            residual.append(f"holdout {holdout}: {len(leaks)} links to training rows remain, e.g. {leaks[0]}")
    spent = {r for r, s in split_of.items() if s in SPENT_SPLITS}
    every_train = set(master.loc[master["frozen_split"].isin((DEV_SPLIT, *TRAINING_ONLY_SPLITS)), "image_id"].astype(str)) - set(drops["global"])
    spent_leaks = [(r, o) for r in spent for o in adjacent.get(r, ()) if o in every_train]
    if spent_leaks:
        residual.append(f"spent sets: {len(spent_leaks)} links to rows still in training, e.g. {spent_leaks[0]}")
    failures.extend(residual)

    if plant_report is None:
        failures.append("no planted-copy control was run")
    elif not plant_report.get("all_found"):
        missed = {k: v for k, v in plant_report.items() if isinstance(v, dict) and v["found"] != v["planted"]}
        failures.append(f"planted copies missed: {missed}")
    return {"passed": not failures, "failures": failures, "rows_compared": int(len(master)), "declared_rows": int(declared_rows),
            "links": {t: int((links["tier"] == t).sum()) for t in TIERS}, "fill": fill, "components": sizes,
            "review": {k: len(v) for k, v in review.items()}, "review_groups": review,
            "removals": {"global": len(drops["global"]), "by_fold": {str(f): len(v) for f, v in drops["by_fold"].items()},
                         "by_holdout": {a: len(v) for a, v in drops["by_holdout"].items()}},
            "removal_rows": {"global": drops["global"], "by_fold": {str(f): v for f, v in drops["by_fold"].items()},
                             "by_holdout": drops["by_holdout"]},
            "plants": plant_report}


def write_receipt(receipt_path: Path, master_path: Path, links_path: Path, report: dict) -> dict:
    receipt = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "passed": bool(report["passed"]),
               "master_sha256": sha256_path(master_path), "links_sha256": sha256_path(links_path), "code_sha256": code_sha(),
               "master": str(master_path), "links": str(links_path), "report": report}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=1, default=str), encoding="utf-8")
    return receipt


def require_receipt(receipt_path: Path, master_path: Path, links_path: Path) -> dict:
    """Refuse to continue unless a passing receipt matches the current master table, link table and guard code."""
    if not Path(receipt_path).is_file():
        raise GuardFailure(f"no data-gate receipt at {receipt_path}; run `python -m research.data.build_master_v7` first")
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    problems = []
    if not receipt.get("passed"):
        problems.append("the receipt records a failed gate: " + "; ".join(receipt["report"].get("failures", [])))
    if receipt.get("master_sha256") != sha256_path(master_path):
        problems.append("the master table changed after the receipt was written")
    if receipt.get("links_sha256") != sha256_path(links_path):
        problems.append("the link table changed after the receipt was written")
    if receipt.get("code_sha256") != code_sha():
        problems.append("the data-gate code changed after the receipt was written")
    if problems:
        raise GuardFailure("data gate: " + " | ".join(problems))
    return receipt
