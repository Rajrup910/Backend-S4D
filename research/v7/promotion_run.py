"""The V7 promotion read (plan 10.8 step 7; freeze blocker F7): the one entry point that decides "V7 replaces V6" or "V6 stays".

    python -m research.v7.promotion_run freeze-candidate --spec candidate_spec.json     # hashes files; opens no prediction
    python -m research.v7.promotion_run freeze-loao --candidate NAME --spec loao_spec.json
    python -m research.v7.promotion_run check --candidate NAME                          # dry run: guards and hashes only
    python -m research.v7.promotion_run read --candidate NAME --owner-approved          # the read, once

`research/v7/promotion.py` holds the rule; this module only loads the frozen files, builds the declared row sets and calls
those functions. It decides nothing itself.

**Order, each step refusing if the one before it is missing.**

1. `freeze-candidate` writes `results/v7/promotion/<name>/candidate.json`: every member's folds 1-4 prediction file and
   training summary with their SHA-256, and the Gate 0 record in force. It reads bytes only, never a label or a probability.
   After this the candidate cannot change: the read refuses any file whose hash differs.
2. `freeze-loao` does the same for the leave-one-archive-out predictions of the candidate and of the V6-recipe baseline
   (P3). The candidate's LOAO runs come after step 1 (plan 13.8, step 5b).
3. `read` runs once per candidate. A second read needs `--rerun-reason`, is recorded, and writes a new file beside the
   first; nothing is overwritten.

**The four row sets** (P1 and P2 are read on each; `promotion.ROW_SETS`). Each names the EVALUATION rows of folds 1-4 that
are left out of the read. The training side is identical in all four: the data gate's training-side removals apply to every
run of record, whatever the row set.

* `all_rows`: every development row of folds 1-4.
* `without_380_rows`: excluding the rows of the 380 lesion groups declared at V6 (`results/v6/leak_sensitivity_rows.csv`).
* `without_426_rows`: excluding the rows of the extended 426 declared groups (a superset of the 380).
* `without_v7_linked_rows`: excluding every development row the V7 data gate links to a row outside its own lesion group,
  or removed (`promotion.v7_linked_rows`).

P3 is read on each held-out archive's development rows, candidate vote against baseline vote, same rows, same counted
classes (`promotion.p3_holdout`).

**Refusals (fail closed).** Unsigned or changed Gate 0; a training run not made under a recorded signature; a smoke or
unfinished run; a run made under a different data-gate receipt; a missing, extra or duplicated row; an image id that is
not a development row (so no `ham_test`, HIBA or MIDAS row can enter); labels or lesion groups that differ from the
development table; a class column missing or out of order; a probability that is not finite, outside [0, 1] or not summing
to 1; a prediction or reference file whose hash changed; a banked V6 reference that no longer reproduces its recorded
Macro-F1; a member or archive missing; a read file that already exists.

Never read here: `ham_test` (test lock armed), HIBA outcomes, MIDAS labels (`research/v7/external_read.py` is the only
path to MIDAS, and it is separate). CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research import testguard
from research.v6 import head8
from research.v6 import system_reads as sr
from research.v7 import builds, plan_freeze, promotion, stats

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "v7" / "promotion"
RECEIPT = REPO / "results" / "v7" / "data_gate_receipt.json"
V6_MANIFEST = REPO / "results" / "v7" / "v6_reference_manifest.json"
V6_E_PRIMARY = REPO / "results" / "v6" / "e_primary.json"
V6_FINALISTS = ("anchor_h7", "panderm_b_h7", "dinov2_b_h7", "anchor384_h7")
V6_TAG, V6_CHECKPOINT = "v64", "swad"
#: P3's comparator (decision 16): the V6-recipe vote, member label -> (trainer member, image size). Option (b) sets
#: {"anchor_h7": ("anchor", 224)} before signing, and the candidate's cross-site arm is then its anchor alone.
P3_BASELINE_MEMBERS = {"anchor_h7": ("anchor", 224), "anchor384_h7": ("anchor", 384), "dinov2_b_h7": ("dinov2_b", 224),
                       "panderm_b_h7": ("panderm_b", 224)}
READ_FOLDS = (1, 2, 3, 4)
CLASSES = tuple(head8.CLASSES_7)
PROB_COLUMNS = [f"p_{c}" for c in CLASSES]
BANDS = ("40-59", "60+")
#: The declared secondary family (plan 10.4, 13.3). Holm is applied across exactly these; none of them promotes.
SECONDARY_FAMILY = ("gate_a2_pauc_u40", "e_histo_pauc_all", "e_histo_pauc_u40", "xauc_young_pos_older_neg",
                    "xauc_young_pos_older_confirmed_neg", "macro_f1_bcn20000", "macro_f1_ham", "macro_f1_mskcc",
                    "pauc_40-59", "pauc_60+")
ROW_SET_DEFINITION = ("each row set names evaluation rows of folds 1-4 left out of the read; the training side is the same in "
                      "all four (the data gate's training-side removals apply to every run of record)")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,40}$")


class Refusal(SystemExit):
    """A guard stopped the read. Nothing was written."""


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else REPO / path


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# ------------------------------------------------------------------ sources (replaced by synthetic ones in tests)
def load_rows() -> tuple[sr.Rows, np.ndarray]:
    """Development rows (all five folds) with labels, lesion groups and age band, from V5's common matrix as V6 read them."""
    rows, _ = sr.load_matrix((sr.V5_SYSTEM,))
    band = pd.read_csv(sr.MATRIX, usecols=["image_id", "age_band"], dtype={"image_id": str}, low_memory=False)
    if not (band["image_id"].to_numpy() == rows.image_id).all():
        raise Refusal("the development table's row order changed between two loads")
    return rows, band["age_band"].to_numpy()


def load_v6_reference(rows: sr.Rows) -> tuple[np.ndarray, dict]:
    """The banked V6 vote on folds 1-4, after checking every file against the reference manifest and the recorded Macro-F1."""
    manifest = json.loads(V6_MANIFEST.read_text(encoding="utf-8"))["predictions_tta"]
    hashes = {}
    for name in V6_FINALISTS:
        for fold in READ_FOLDS:
            path = sr.gate.pred_path(name, 42, V6_TAG, sr.V6_TTA_DIR, fold=fold, checkpoint=f"{V6_CHECKPOINT}_tta")
            key = _rel(path)
            if key not in manifest or manifest[key]["sha256"] != sha(path):
                raise Refusal(f"V6 reference file {key} is not the banked file (hash differs from v6_reference_manifest.json)")
            hashes[key] = manifest[key]["sha256"]
    members = [sr.load_member(rows, name, {f: V6_TAG for f in READ_FOLDS}, checkpoint=V6_CHECKPOINT, tta=True)["p7"] for name in V6_FINALISTS]
    vote = sr.vote(members)
    banked = json.loads(V6_E_PRIMARY.read_text(encoding="utf-8"))["descriptive"]["v6_endpoints"]["macro_f1"]
    got = sr.fast_endpoints(vote, rows, rows.where())["macro_f1"]
    if abs(got - banked) > 1e-9:
        raise Refusal(f"the banked V6 vote gives Macro-F1 {got:.6f}, not the recorded {banked:.6f}")
    return vote, hashes


def loao_dropped_groups() -> frozenset[str]:
    from research.v5 import train_v5   # imported late: it pulls in torch, which a dry run does not need
    return train_v5.LOAO_DROPPED_GROUPS


def dark_lesion_read(mass: pd.Series, name: str) -> dict:
    from research.v7 import dark_read
    return dark_read.read(mass, name)


@dataclass
class Sources:
    gate0: Callable[[], str] = plan_freeze.require
    signatures: Callable[[], set[str]] = plan_freeze.valid_signatures
    frozen_files: Callable[[], dict[str, str]] = plan_freeze.digest
    rows: Callable[[], tuple[sr.Rows, np.ndarray]] = load_rows
    reference: Callable[[sr.Rows], tuple[np.ndarray, dict]] = load_v6_reference
    declared_lists: Callable[[], dict[str, set[str]]] = promotion.declared_lists
    v7_linked: Callable[[], set[str]] = promotion.v7_linked_rows
    histo: Callable[[np.ndarray], np.ndarray] = promotion.histo_confirmed
    bar: Callable[[], float] = promotion.single_seed_bar
    receipt_sha: Callable[[], str] = lambda: sha(RECEIPT)
    loao_dropped: Callable[[], frozenset[str]] = loao_dropped_groups
    dark: Callable[[pd.Series, str], dict] = dark_lesion_read
    out_dir: Path = OUT_DIR
    n_boot: int = sr.N_BOOT


# ------------------------------------------------------------------ guards on training runs and files
def check_run(run_path: Path, src: Sources, fold: int | None, holdout: str | None, build: str | None = None) -> dict:
    """One training summary: a finished run of record, made under a recorded signature and the current data-gate receipt."""
    if not run_path.is_file():
        raise Refusal(f"training summary missing: {_rel(run_path)}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    problems = []
    if run.get("smoke") is not False:
        problems.append("a smoke, not a run of record")
    if run.get("epochs_run") != run.get("epochs") or run.get("optimizer_steps") != run.get("step_budget"):
        problems.append("not a complete run (epochs or optimiser steps short of the budget)")
    if run.get("plan_freeze_sha256") not in src.signatures():
        problems.append("not made under a recorded Gate 0 signature or amendment")
    if run.get("data_gate_receipt_sha256") != src.receipt_sha():
        problems.append("made under a different data-gate receipt (stale)")
    key = run.get("v7_build")
    if key not in builds.BUILDS or builds.BUILDS[key].status in ("NEEDS_CODE", "NEEDS_BENCHMARK", "NEEDS_ARTEFACT", "BLOCKED"):
        problems.append(f"build {key!r} is not a declared, runnable build")
    if build is not None and key != build:
        problems.append(f"build is {key!r}, expected {build!r}")
    if holdout is None and (run.get("fold") != fold or run.get("loao_holdout")):
        problems.append(f"not a fold-{fold} in-domain run")
    if holdout is not None and run.get("loao_holdout") != holdout:
        problems.append(f"not a leave-{holdout}-out run")
    if run.get("test_read") or run.get("reserved_read"):
        problems.append("the summary records a protected read")
    if not run.get("run_id"):
        problems.append("the summary names no run_id")
    if problems:
        raise Refusal(f"refusing {_rel(run_path)}: " + "; ".join(problems))
    return run


def _entry(item: dict, src: Sources, fold: int | None, holdout: str | None, build: str | None = None) -> dict:
    if set(item) != {"predictions", "run"}:
        raise Refusal(f"each entry names exactly 'predictions' and 'run', got {sorted(item)}")
    predictions, run_path = _path(item["predictions"]), _path(item["run"])
    if not predictions.is_file():
        raise Refusal(f"prediction file missing: {_rel(predictions)}")
    run = check_run(run_path, src, fold, holdout, build)
    # Bind the summary to the predictions (finding 2b/2d): every row must carry the summary's run_id. Reads one column only.
    ids = set(pd.read_csv(predictions, usecols=lambda c: c == "run_id", dtype=str).get("run_id", pd.Series(dtype=str)).dropna())
    if ids != {str(run["run_id"])}:
        raise Refusal(f"{_rel(predictions)} does not carry run_id {run['run_id']!r} on every row (found {sorted(ids)[:2]})")
    return {"predictions": _rel(predictions), "predictions_sha256": sha(predictions), "run": _rel(run_path), "run_sha256": sha(run_path),
            "run_id": run.get("run_id"), "v7_build": run.get("v7_build"), "plan_freeze_sha256": run.get("plan_freeze_sha256")}


def _directory(name: str, src: Sources) -> Path:
    if not NAME.match(name or ""):
        raise Refusal(f"candidate name {name!r} must be 2-41 letters, digits, '_', '.' or '-'")
    return src.out_dir / name


def freeze_candidate(spec: dict, src: Sources | None = None) -> dict:
    """Write the candidate manifest. Hashes bytes; opens no prediction and no label."""
    src = src or Sources()
    signature = src.gate0()
    directory = _directory(spec.get("name", ""), src)
    if (directory / "candidate.json").exists():
        raise Refusal(f"candidate {spec['name']!r} is already frozen; a frozen candidate is never rewritten")
    members = spec.get("members") or {}
    if not members:
        raise Refusal("the candidate has no members")
    frozen = {}
    for label, folds in members.items():
        if sorted(folds) != [str(f) for f in READ_FOLDS]:
            raise Refusal(f"member {label!r} must name folds {list(READ_FOLDS)} exactly, got {sorted(folds)}")
        frozen[label] = {f: _entry(folds[f], src, int(f), None) for f in sorted(folds)}
    hashes = [e["predictions_sha256"] for folds in frozen.values() for e in folds.values()]
    runs = [e["run_id"] for folds in frozen.values() for e in folds.values()]
    if len(set(hashes)) != len(hashes) or len(set(runs)) != len(runs):
        raise Refusal("two entries name the same prediction file or training run: a member counted twice (finding 2e)")
    manifest = {"name": spec["name"], "vote": "uniform soft vote of the members, per image", "members": frozen,
                "gate0_record_sha256": signature, "data_gate_receipt_sha256": src.receipt_sha(),
                "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "candidate.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def freeze_loao(name: str, spec: dict, src: Sources | None = None) -> dict:
    """Write the cross-site manifest: per archive, the candidate's and the V6-recipe baseline's LOAO prediction files."""
    src = src or Sources()
    signature = src.gate0()
    directory = _directory(name, src)
    if not (directory / "candidate.json").exists():
        raise Refusal("freeze the candidate first: its leave-one-archive-out runs come after it is frozen (plan 13.8)")
    if (directory / "loao.json").exists():
        raise Refusal(f"the cross-site files of {name!r} are already frozen")
    if sorted(spec) != ["baseline", "candidate"]:
        raise Refusal("the cross-site spec names exactly 'candidate' and 'baseline'")
    candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    builds_of = {label: {e["v7_build"] for e in folds.values()} for label, folds in candidate["members"].items()}
    out: dict = {"candidate": {}, "baseline": {}}
    for arm in ("candidate", "baseline"):
        if sorted(spec[arm]) != sorted(promotion.P3_HOLDOUTS):
            raise Refusal(f"{arm}: P3 needs every archive in {promotion.P3_HOLDOUTS}, got {sorted(spec[arm])}")
        for holdout, members in spec[arm].items():
            # Like for like (finding 2f): the candidate arm is the frozen candidate's own roster, the baseline the declared V6 vote.
            expected = sorted(builds_of) if arm == "candidate" else sorted(P3_BASELINE_MEMBERS)
            if sorted(members) != expected:
                raise Refusal(f"{arm}, {holdout}: members {sorted(members)} are not {expected} (P3 compares the frozen candidate's vote "
                              "with the declared V6-recipe vote, member for member)")
            out[arm][holdout] = {label: _entry(item, src, None, holdout, "L0" if arm == "baseline" else None)
                                 for label, item in members.items()}
            for label, entry in out[arm][holdout].items():
                run = json.loads(_path(entry["run"]).read_text(encoding="utf-8"))
                if arm == "candidate" and entry["v7_build"] not in builds_of[label]:
                    raise Refusal(f"candidate, {holdout}, {label}: build {entry['v7_build']} is not the frozen member's {sorted(builds_of[label])}")
                if arm == "baseline" and (run.get("member"), run.get("image_size")) != P3_BASELINE_MEMBERS[label]:
                    raise Refusal(f"baseline, {holdout}, {label}: run is {run.get('member')} at {run.get('image_size')} px, not {P3_BASELINE_MEMBERS[label]}")
    manifest = {"name": name, "comparison": "candidate vote against the V6-recipe baseline vote, same rows, per held-out archive",
                **out, "gate0_record_sha256": signature, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (directory / "loao.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _entries(candidate: dict, loao: dict | None) -> list[tuple[dict, int | None, str | None, str | None]]:
    """Every frozen entry with the fold or archive, and the build, its training run must have."""
    items = [(entry, int(fold), None, None) for folds in candidate["members"].values() for fold, entry in folds.items()]
    for arm in ("candidate", "baseline"):
        for holdout, members in (loao or {}).get(arm, {}).items():
            items += [(entry, None, holdout, "L0" if arm == "baseline" else None) for entry in members.values()]
    return items


def verify(name: str, src: Sources) -> tuple[dict, dict | None, str]:
    """Guards shared by the dry run and the read: Gate 0, manifests, every hash, every training summary. Opens no CSV."""
    signature = src.gate0()
    directory = _directory(name, src)
    if not (directory / "candidate.json").is_file():
        raise Refusal(f"candidate {name!r} is not frozen: run `freeze-candidate` first")
    candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    loao = json.loads((directory / "loao.json").read_text(encoding="utf-8")) if (directory / "loao.json").is_file() else None
    for manifest in filter(None, (candidate, loao)):
        if manifest["gate0_record_sha256"] not in src.signatures():
            raise Refusal("a manifest was frozen under a signature that is not in the Gate 0 record")
    if candidate["data_gate_receipt_sha256"] != src.receipt_sha():
        raise Refusal("the data-gate receipt changed after the candidate was frozen (stale)")
    for entry, fold, holdout, build in _entries(candidate, loao):
        for kind in ("predictions", "run"):
            path = _path(entry[kind])
            if not path.is_file() or sha(path) != entry[f"{kind}_sha256"]:
                raise Refusal(f"{entry[kind]} is missing or changed after the candidate was frozen")
        check_run(_path(entry["run"]), src, fold, holdout, build)
    return candidate, loao, signature


# ------------------------------------------------------------------ strict prediction loading
def load_predictions(path: Path, rows: sr.Rows, position: dict[str, int], expected: np.ndarray, run_id: str | None = None) -> np.ndarray:
    """Probabilities for exactly the `expected` row positions, in that order. Any mismatch refuses."""
    frame = pd.read_csv(path, dtype={"image_id": str}, low_memory=False)
    missing = [c for c in ("image_id", *PROB_COLUMNS) if c not in frame.columns]
    if missing:
        raise Refusal(f"{_rel(path)}: missing columns {missing}")
    ordered = [c for c in frame.columns if c in PROB_COLUMNS]
    if ordered != PROB_COLUMNS:
        raise Refusal(f"{_rel(path)}: class columns are {ordered}, not the frozen order {PROB_COLUMNS}")
    if frame["image_id"].duplicated().any():
        raise Refusal(f"{_rel(path)}: {int(frame['image_id'].duplicated().sum())} duplicated rows")
    unknown = [i for i in frame["image_id"] if i not in position]
    if unknown:
        raise Refusal(f"{_rel(path)}: {len(unknown)} image ids are not development rows (a protected or outside row is refused)")
    where = np.array([position[i] for i in frame["image_id"]], dtype=int)
    if len(where) != len(expected) or set(where.tolist()) != set(expected.tolist()):
        extra, short = len(set(where.tolist()) - set(expected.tolist())), len(set(expected.tolist()) - set(where.tolist()))
        raise Refusal(f"{_rel(path)}: rows do not match the declared set ({extra} extra, {short} missing)")
    probs = frame[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or np.abs(probs.sum(1) - 1).max() > 1e-3:
        raise Refusal(f"{_rel(path)}: probabilities are not finite, in [0, 1] and summing to 1")
    if "y_true" in frame and not (frame["y_true"].to_numpy() == rows.y[where]).all():
        raise Refusal(f"{_rel(path)}: labels differ from the development table")
    if "group_id" in frame and not (frame["group_id"].astype(str).to_numpy() == rows.group[where].astype(str)).all():
        raise Refusal(f"{_rel(path)}: lesion groups differ from the development table")
    if run_id and "run_id" in frame and set(frame["run_id"].astype(str)) != {run_id}:
        raise Refusal(f"{_rel(path)}: predictions come from run {sorted(set(frame['run_id'].astype(str)))[:2]}, not {run_id}")
    out = np.full((len(rows.y), len(CLASSES)), np.nan)
    out[where] = probs
    return out


def _vote(parts: list[np.ndarray]) -> np.ndarray:
    return np.mean(parts, axis=0)


def _paired(metric: Callable[[np.ndarray, np.ndarray], float], target: np.ndarray, a: np.ndarray, b: np.ndarray,
            groups: np.ndarray, n_boot: int) -> dict:
    return stats.paired_cluster_bootstrap(lambda i: metric(target[i], a[i]), lambda i: metric(target[i], b[i]), groups, n_boot=n_boot)


def read(name: str, owner_approved: bool, allow_missing_p3: bool = False, rerun_reason: str = "", src: Sources | None = None) -> dict:
    """The promotion read. Once per candidate; returns the report it wrote."""
    src = src or Sources()
    if not owner_approved:
        raise Refusal("the promotion read is the owner's step: pass --owner-approved")
    candidate, loao, signature = verify(name, src)
    directory = _directory(name, src)
    earlier = sorted(directory.glob("read*.json"))
    if earlier and not rerun_reason.strip():
        raise Refusal(f"{name!r} has been read ({earlier[0].name}): folds 1-4 are read once per frozen candidate. A repeat needs --rerun-reason, which is recorded.")
    # Multiplicity across candidates (finding 2h): every earlier candidate's read is listed in this report, and a second
    # candidate needs a stated reason. The first candidate read is the read of record; later ones are disclosed beside it.
    others = sorted(p.parent.name for p in src.out_dir.glob("*/read*.json") if p.parent.name != name)
    if others and not rerun_reason.strip():
        raise Refusal(f"other candidates have already been read on folds 1-4 ({sorted(set(others))}). Reading another needs "
                      "--rerun-reason; the report lists every earlier candidate, so the multiplicity is visible.")
    if loao is None and not allow_missing_p3:
        raise Refusal("no cross-site files are frozen, so P3 cannot be read and nothing can be promoted. Freeze them (`freeze-loao`), "
                      "or pass --allow-missing-p3 to record an in-domain read whose verdict is NOT_ESTIMABLE.")
    rows, band = src.rows()
    position = {image: i for i, image in enumerate(rows.image_id)}
    if len(position) != len(rows.image_id):
        raise Refusal("the development table has duplicated image ids")
    every = rows.where()
    reference, reference_hashes = src.reference(rows)

    # -- candidate vote on folds 1-4
    members = []
    for label, folds in candidate["members"].items():
        parts = np.full((len(rows.y), len(CLASSES)), np.nan)
        for fold, entry in folds.items():
            expected = np.where(rows.fold == int(fold))[0]
            loaded = load_predictions(_path(entry["predictions"]), rows, position, expected, entry.get("run_id"))
            parts[expected] = loaded[expected]
        members.append(parts)
    vote = _vote(members)
    if np.isnan(vote[every]).any() or np.isnan(reference[every]).any():
        raise Refusal("a fold 1-4 row has no prediction from the candidate or from the V6 reference")

    # -- the four row sets (evaluation rows only)
    lists = src.declared_lists()
    excluded = {"all_rows": set(), "without_380_rows": set(lists["without_380_rows"]), "without_426_rows": set(lists["without_426_rows"]),
                "without_v7_linked_rows": set(src.v7_linked())}
    if list(excluded) != list(promotion.ROW_SETS):
        raise Refusal("the runner's row sets are not promotion.ROW_SETS")
    row_sets = {label: every[~np.isin(rows.image_id[every], list(drop))] if drop else every for label, drop in excluded.items()}
    bar = src.bar()
    in_domain = {label: promotion.p1_p2(vote, reference, rows, where, bar, n_boot=src.n_boot) for label, where in row_sets.items()}

    # -- P3: candidate vote against baseline vote on each held-out archive
    cross_site = None
    histo = src.histo(rows.image_id)
    if loao is not None:
        dropped = src.loao_dropped()
        cross_site = {}
        for holdout in promotion.P3_HOLDOUTS:
            expected = np.where((rows.archive == holdout) & ~np.isin(rows.group.astype(str), list(dropped)))[0]
            arms = {arm: _vote([load_predictions(_path(e["predictions"]), rows, position, expected, e.get("run_id"))
                                for e in loao[arm][holdout].values()]) for arm in ("candidate", "baseline")}
            cross_site[holdout] = {"rows": int(len(expected)), "lesion_groups": int(len(np.unique(rows.group[expected]))),
                                   "members": {arm: sorted(loao[arm][holdout]) for arm in arms},
                                   **promotion.p3_holdout(rows.y[expected], rows.esc[expected].astype(int), rows.group[expected],
                                                          arms["candidate"][expected], arms["baseline"][expected], n_boot=src.n_boot,
                                                          histo=histo[expected])}

    # -- declared secondaries (never decisive), with Holm across the declared family
    primary = in_domain["all_rows"]
    e_histo = promotion.e_histo(vote, reference, rows, every, histo, n_boot=src.n_boot)
    xauc = promotion.xauc_u40(vote, reference, rows, every, n_boot=src.n_boot)
    confirmed = every[rows.esc[every] | histo[every]]
    xauc_confirmed = promotion.xauc_u40(vote, reference, rows, confirmed, n_boot=src.n_boot)
    by_archive = {}
    for a in promotion.P3_HOLDOUTS:
        sub = every[rows.archive[every] == a]
        by_archive[a] = {"rows": int(len(sub)), **_paired(sr._mf1, rows.y[sub], vote[sub], reference[sub], rows.group[sub], src.n_boot)}
    by_band = {}
    for b in BANDS:
        sub = every[band[every] == b]
        by_band[b] = {"rows": int(len(sub)), **_paired(sr._pauc, rows.esc[sub], vote[sub], reference[sub], rows.group[sub], src.n_boot)}
    p_values = {"gate_a2_pauc_u40": primary["secondary_gate_a2"]["p_boot"], "e_histo_pauc_all": e_histo["pauc_all"]["p_boot"],
                "e_histo_pauc_u40": e_histo["pauc_u40"]["p_boot"], "xauc_young_pos_older_neg": xauc["xauc_young_pos_older_neg"]["p_boot"],
                "xauc_young_pos_older_confirmed_neg": xauc_confirmed["xauc_young_pos_older_neg"]["p_boot"],
                **{f"macro_f1_{a}": by_archive[a]["p_boot"] for a in promotion.P3_HOLDOUTS}, **{f"pauc_{b}": by_band[b]["p_boot"] for b in BANDS}}
    if tuple(p_values) != SECONDARY_FAMILY:
        raise Refusal("the secondary reads are not the declared family")
    esc_columns = [CLASSES.index(c) for c in ("akiec", "bcc", "mel")]
    mass = pd.Series(vote[every][:, esc_columns].sum(1), index=rows.image_id[every])
    secondary = {"decisive": False, "gate_a2": primary["secondary_gate_a2"], "e_histo": e_histo, "xauc": xauc,
                 "xauc_confirmed_negatives": xauc_confirmed, "per_archive_macro_f1": by_archive, "per_band_escalation_pauc": by_band,
                 "holm_family": {"members": list(SECONDARY_FAMILY), "size": len(SECONDARY_FAMILY), "p_boot": p_values, "p_holm": stats.holm(p_values),
                                 "note": "bootstrap p-values, two-sided, add-one; the dark-lesion read is descriptive and outside the family"},
                 "dark_lesion_read": src.dark(mass, name)}

    verdict = promotion.verdict(in_domain, cross_site)
    report = {
        "what": "V7 promotion read: P1 and P2 on four row sets against the banked V6 vote; P3 per held-out archive; declared secondaries",
        "status": verdict["status"], "decision": verdict["decision"], "verdict": verdict,
        "executed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rerun_reason": rerun_reason.strip() or None,
        "earlier_reads": [p.name for p in earlier], "other_candidates_read_before": sorted(set(others)),
        "read_of_record": not earlier and not others, "command": sys.argv,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "candidate": name, "gate0_record_sha256": signature,
        "hashes": {"candidate_manifest": sha(directory / "candidate.json"), "loao_manifest": sha(directory / "loao.json") if loao else None,
                   "v6_reference_predictions": reference_hashes, "frozen_files": src.frozen_files(), "data_gate_receipt": src.receipt_sha()},
        "rule": {"bar_pauc_k1": bar, "p1": "pAUC delta > bar AND Macro-F1 delta > 0 with lower 95% bound > 0", "p2": primary["p2"]["form"],
                 "p2_noninferiority_margins": promotion.P2_NONINFERIORITY_MARGINS, "p3_margin": promotion.NON_INFERIORITY,
                 "p3_holdouts": list(promotion.P3_HOLDOUTS), "p3_min_class_groups": promotion.P3_MIN_CLASS_GROUPS,
                 "fpr_max": promotion.FPR_MAX, "n_boot": src.n_boot, "bootstrap_unit": "lesion group"},
        "row_set_definition": ROW_SET_DEFINITION,
        "rows": {label: int(len(where)) for label, where in row_sets.items()},
        "lesion_groups": {label: int(len(np.unique(rows.group[where]))) for label, where in row_sets.items()},
        "in_domain": in_domain, "cross_site": cross_site, "secondary": secondary,
        "protected_reads": {"fold_0_pooled": False, "test_read": False, "reserved_read": False, "hiba_read": False, "midas_read": False},
    }
    out = directory / ("read.json" if not earlier else f"read_rerun_{len(earlier)}.json")
    if out.exists():
        raise Refusal(f"{out.name} exists; nothing is overwritten")
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    table = [{"row_set": label, "endpoint": metric, "candidate": r["read"][metric]["a"], "v6": r["read"][metric]["b"],
              "delta": r["read"][metric]["delta"], "lo95": r["read"][metric]["ci95"][0], "hi95": r["read"][metric]["ci95"][1],
              "p1": r["p1"]["passed"], "p2": r["p2"]["passed"]} for label, r in in_domain.items() for metric in sr.BOOT_METRICS]
    pd.DataFrame(table).to_csv(out.with_suffix(".csv"), index=False)
    return report


def check(name: str, src: Sources | None = None) -> dict:
    """Dry run: every guard that needs no prediction content. Opens no CSV and no label."""
    src = src or Sources()
    candidate, loao, signature = verify(name, src)
    directory = _directory(name, src)
    return {"candidate": name, "gate0_record_sha256": signature, "members": sorted(candidate["members"]),
            "in_domain_files": sum(len(f) for f in candidate["members"].values()),
            "cross_site_frozen": loao is not None, "already_read": [p.name for p in sorted(directory.glob("read*.json"))],
            "would_read": "P1, P2 on four row sets; P3 on " + (", ".join(promotion.P3_HOLDOUTS) if loao else "nothing (NOT_ESTIMABLE)"),
            "predictions_opened": 0, "labels_opened": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a V7 candidate and run its one promotion read (plan 10.4, 10.8 step 7).")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("freeze-candidate", help="hash the candidate's folds 1-4 files into a manifest; opens no prediction")
    one.add_argument("--spec", required=True, help="JSON: {name, members: {label: {fold: {predictions, run}}}}")
    two = sub.add_parser("freeze-loao", help="hash the candidate's and the baseline's leave-one-archive-out files")
    two.add_argument("--candidate", required=True)
    two.add_argument("--spec", required=True, help="JSON: {candidate: {archive: {label: {predictions, run}}}, baseline: {...}}")
    three = sub.add_parser("check", help="dry run: guards and hashes only; opens no prediction and no label")
    three.add_argument("--candidate", required=True)
    four = sub.add_parser("read", help="the promotion read, once per frozen candidate")
    four.add_argument("--candidate", required=True)
    four.add_argument("--owner-approved", action="store_true")
    four.add_argument("--allow-missing-p3", action="store_true", help="record an in-domain read without P3; the verdict is NOT_ESTIMABLE")
    four.add_argument("--rerun-reason", default="")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V7 promotion read: development rows only")
    if args.command == "freeze-candidate":
        out = freeze_candidate(json.loads(Path(args.spec).read_text(encoding="utf-8")))
        print(f"frozen {out['name']}: {len(out['members'])} member(s) under Gate 0 record {out['gate0_record_sha256'][:16]}")
    elif args.command == "freeze-loao":
        out = freeze_loao(args.candidate, json.loads(Path(args.spec).read_text(encoding="utf-8")))
        print(f"frozen cross-site files of {out['name']}")
    elif args.command == "check":
        print(json.dumps(check(args.candidate), indent=1))
    else:
        report = read(args.candidate, args.owner_approved, args.allow_missing_p3, args.rerun_reason)
        print(f"{report['status']}: {report['decision']}")
        for reason in report["verdict"]["reasons"]:
            print("  " + reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
