"""V7 data gate: id normalisation, fingerprints and planted copies, grouping, the overlap guard and its receipt.

Every guard check has a test that makes it fire (V7 data plan, build step 4: "a test for each guard that could never fire")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from research.data import fingerprints as fp
from research.data import grouping, overlap_guard, plants
from research.data.build_master_v7 import identifier_links
from research.data.ids import is_normalised, normalise_id, normalise_lesion_id


# ------------------------------------------------------------------ ids
@pytest.mark.parametrize("raw, expected", [
    ("ISIC_0000026", "ISIC_0000026"), ("ISIC_0000026_downsampled", "ISIC_0000026"), ("isic_0000026.JPG", "ISIC_0000026"),
    ("data/x/ISIC_0000026_downsampled.jpg", "ISIC_0000026"), ("C:\\a\\ISIC_0000026.png", "ISIC_0000026"),
    (" ISIC 0000026 ", "ISIC_0000026"), ("ISIC_0000026_downsampled_downsampled", "ISIC_0000026"), ("HAM_0000118", "HAM_0000118")])
def test_normalise_id(raw: str, expected: str) -> None:
    assert normalise_id(raw) == expected


@pytest.mark.parametrize("raw", [None, float("nan"), "", "   "])
def test_an_empty_id_raises_so_two_missing_ids_never_compare_equal(raw) -> None:
    with pytest.raises(ValueError):
        normalise_id(raw)
    assert normalise_lesion_id(raw, "BCN") is None
    assert not is_normalised(raw)


def test_is_normalised_flags_suffixes_and_extensions() -> None:
    assert is_normalised("ISIC_0000026")
    assert not is_normalised("ISIC_0000026_downsampled")
    assert not is_normalised("ISIC_0000026.jpg")


# ------------------------------------------------------------------ fingerprints and plants
def _lesion_image(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    h, w = 450, 600
    yy, xx = np.mgrid[0:h, 0:w]
    skin = np.stack([200 + 20 * np.sin(xx / (40 + seed)), 160 + 15 * np.cos(yy / (55 + seed)), 140 + 0 * xx], -1)
    cy, cx = h / 2 + rng.integers(-40, 40), w / 2 + rng.integers(-60, 60)
    r = np.hypot((yy - cy) / (110 + rng.integers(0, 40)), (xx - cx) / (150 + rng.integers(0, 40)))
    blob = np.clip(1.3 - r, 0, 1)[..., None] * np.array([120, 110, 90]) * (0.6 + 0.4 * rng.random((h, w, 1)))
    image = np.clip(skin - blob + rng.normal(0, 6, (h, w, 3)), 0, 255)
    rim = np.hypot(yy - h / 2, xx - w / 2) > 0.55 * w
    image[rim] = 5
    Image.fromarray(image.astype(np.uint8)).save(path, quality=92)
    return path


@pytest.fixture()
def sources(tmp_path: Path) -> dict[str, Path]:
    return {f"ISIC_00000{i:02d}": _lesion_image(tmp_path / f"ISIC_00000{i:02d}.jpg", i) for i in range(1, 7)}


def test_every_pixel_plant_is_found_and_unrelated_images_are_not_linked(sources: dict[str, Path], tmp_path: Path) -> None:
    planted = []
    for image_id, path in list(sources.items())[:3]:
        planted += plants.make_plants(image_id, path, None, None, tmp_path / "plants")
    pixel = [p for p in planted if p.path]
    assert {p.family for p in pixel} == set(plants.PIXEL_FAMILIES)
    refs = [fp.fingerprint_file(i, p) for i, p in sources.items()]
    hits = fp.match([fp.fingerprint_file(p.image_id, p.path) for p in pixel], refs)
    found = {(h["query"], h["reference"]) for h in hits}
    for p in pixel:
        assert (p.image_id, p.source_id) in found, p.family
    wrong = [h for h in hits if not h["query"].startswith(f"PLANT_{h['reference']}") and h["query"] != f"{h['reference']}_downsampled"]
    assert not wrong, wrong
    real = fp.match(refs, refs)
    assert real == [], "six unrelated synthetic lesions must not link to one another"


def test_match_refuses_an_empty_side() -> None:
    with pytest.raises(ValueError, match="empty"):
        fp.match([], [fp.Fingerprint("x", "s", 0, 0, ((0, 0),) * 4, ((0, 0),) * 8)])


def test_identifier_plants_are_found_through_star_links(sources: dict[str, Path], tmp_path: Path) -> None:
    image_id, path = next(iter(sources.items()))
    planted = plants.make_plants(image_id, path, "IL_1", "IP_9", tmp_path / "plants")
    frame = pd.DataFrame({"image_id": [image_id, "OTHER_1", *[p.image_id for p in planted]],
                          "archive_id": [image_id, "OTHER_1", *[normalise_id(p.image_id) for p in planted]],
                          "src_lesion_id": ["IL_1", "IL_1", *[p.lesion_id for p in planted]],
                          "src_patient_id": ["IP_9", None, *[p.patient_id for p in planted]],
                          "group_id": [f"g{i}" for i in range(len(planted) + 2)]})
    found = {(a, b) for a, b, _, _ in identifier_links(frame)}
    report = plants.plant_recall(planted, found)
    for family in ("downsampled_variant", "lesion_link", "patient_link"):
        assert report[family]["found"] == 1, family
    assert report["renamed_copy"]["found"] == 0  # bytes, not ids: the pixel tier's job


def test_patient_links_are_at_risk_and_never_confirmed() -> None:
    frame = pd.DataFrame({"image_id": ["a", "b", "c"], "archive_id": ["a", "b", "c"], "src_lesion_id": [None, None, None],
                          "src_patient_id": ["P", "P", "P"], "group_id": ["g1", "g2", "g3"]})
    links = identifier_links(frame)
    assert {t for _, _, t, _ in links} == {"at_risk"}
    assert len(links) == 3  # a clique, so every pair is adjacent


# ------------------------------------------------------------------ grouping
def test_components_follow_confirmed_chains_and_percolation_is_reported() -> None:
    groups = grouping.components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert groups == {"a": "a", "b": "a", "c": "a", "d": "d"}
    assert grouping.size_distribution(groups)["largest"] == 3
    attrs = {"a": {"archive": "ham", "frozen_split": "train"}, "b": {"archive": "bcn20000", "frozen_split": "train"},
             "c": {"archive": "ham", "frozen_split": "ham_test"}, "d": {"archive": "ham", "frozen_split": "train"}}
    review = grouping.review_list(groups, attrs, limit=2)
    assert review == {"oversized": ["a"], "spans_archives": ["a"], "spans_frozen_splits": ["a"]}


def test_an_edge_to_an_unknown_row_raises() -> None:
    with pytest.raises(KeyError):
        grouping.components(["a"], [("a", "zz")])


# ------------------------------------------------------------------ guard
def _master() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append({"image_id": f"ISIC_10000{i:02d}", "archive": "ham" if i % 2 else "bcn20000", "source": "isic2019_train",
                     "frozen_split": "train", "fold": i // 2})
    rows += [{"image_id": "ISIC_2000001", "archive": "isic2020", "source": "isic2020", "frozen_split": "train_added", "fold": None},
             {"image_id": "ISIC_2000002", "archive": "isic2020", "source": "isic2020", "frozen_split": "train_added", "fold": None},
             {"image_id": "ISIC_2000003", "archive": "isic2019_test", "source": "isic2019_test", "frozen_split": "train_added", "fold": None},
             {"image_id": "ISIC_3000001", "archive": "ham", "source": "isic2019_train", "frozen_split": "ham_test", "fold": None}]
    frame = pd.DataFrame(rows)
    frame["fold"] = frame["fold"].astype("Int64")
    frame = frame.assign(archive_id=frame["image_id"], master_group=frame["image_id"], group_method="singleton_or_member",
                         sha256=[f"{i:064x}" for i in range(len(frame))], phash="0" * 16, dhash="0" * 16, status="kept")
    return frame


def _links(*pairs: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"a": a, "b": b, "tier": t, "evidence": "test"} for a, b, t in pairs], columns=list(overlap_guard.LINK_COLUMNS))


PLANTS_OK = {"all_found": True}
D = [f"ISIC_10000{i:02d}" for i in range(10)]


def test_a_clean_table_passes() -> None:
    report = overlap_guard.check(_master(), _links(), 14, PLANTS_OK)
    assert report["passed"], report["failures"]


def test_a_cross_fold_link_removes_the_training_side_row_for_each_held_out_fold() -> None:
    master, links = _master(), _links((D[0], D[2], "at_risk"))  # fold 0 and fold 1
    drops = overlap_guard.removals(master, links)
    assert drops["by_fold"][0] == [D[2]] and drops["by_fold"][1] == [D[0]] and drops["global"] == []
    assert D[2] not in set(overlap_guard.training_rows(master, drops, fold=0)["image_id"])
    assert overlap_guard.check(master, links, 14, PLANTS_OK)["passed"]


def test_rows_linked_to_spent_rows_leave_all_training_and_added_rows_leave_only_the_linked_fold() -> None:
    master = _master()
    drops = overlap_guard.removals(master, _links(("ISIC_2000001", D[3], "at_risk"), (D[4], "ISIC_3000001", "confirmed")))
    assert set(drops["global"]) == {D[4]}                                  # linked to ham_test: out of every run
    assert drops["by_fold"][1] == ["ISIC_2000001"]                         # D3 is fold 1: out of fold 1's run only
    assert all("ISIC_2000001" not in drops["by_fold"][f] for f in (0, 2, 3, 4))
    assert "ISIC_2000001" in set(overlap_guard.training_rows(master, drops, fold=0)["image_id"])
    assert "ISIC_2000001" not in set(overlap_guard.training_rows(master, drops, fold=1)["image_id"])


def test_confirmed_chains_remove_the_whole_group_not_only_direct_neighbours() -> None:
    master = _master()
    links = _links((D[0], D[2], "confirmed"), (D[2], D[4], "confirmed"))  # folds 0, 1, 2
    groups = grouping.components(master["image_id"], [(D[0], D[2]), (D[2], D[4])])
    master["master_group"] = master["image_id"].map(groups)
    drops = overlap_guard.removals(master, links)
    assert set(drops["by_fold"][0]) == {D[2], D[4]}
    assert overlap_guard.check(master, links, 14, PLANTS_OK)["passed"]


def test_an_at_risk_link_to_one_image_of_a_lesion_reaches_its_other_images() -> None:
    """Added row x looks like image D0 of lesion {D0, D2}; D2 is evaluated in fold 1. x must leave fold 1's training even
    though its only link is to D0 (in fold 0). Before 12 Nov the guard followed the link to D0 alone."""
    master = _master()
    groups = grouping.components(master["image_id"], [(D[0], D[2])])
    master["master_group"] = master["image_id"].map(groups)
    links = _links((D[0], D[2], "confirmed"), ("ISIC_2000001", D[0], "at_risk"))
    drops = overlap_guard.removals(master, links)
    assert "ISIC_2000001" in drops["by_fold"][0] and "ISIC_2000001" in drops["by_fold"][1]   # D0's and D2's folds
    assert not any("ISIC_2000001" in drops["by_fold"][f] for f in (2, 3, 4)) and "ISIC_2000001" not in drops["global"]
    links = _links((D[0], D[2], "confirmed"), (D[4], D[0], "at_risk"))   # a development row of fold 2
    drops = overlap_guard.removals(master, links)
    assert D[4] in drops["by_fold"][0] and D[4] in drops["by_fold"][1]   # out when D0's fold OR D2's fold is held out
    assert {D[0], D[2]} <= set(drops["by_fold"][2])                     # and both lesion images out when D4 is evaluated
    assert overlap_guard.check(master, links, 14, PLANTS_OK)["passed"]


def test_a_bcn_holdout_drops_the_barcelona_part_of_isic2019_test() -> None:
    master = _master()
    drops = overlap_guard.removals(master, _links())
    assert drops["by_holdout"]["bcn20000"] == ["ISIC_2000003"]
    train = set(overlap_guard.training_rows(master, drops, holdout="bcn20000")["image_id"])
    assert "ISIC_2000003" not in train and not train & set(master.loc[master["archive"] == "bcn20000", "image_id"])


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m.assign(archive_id=m["archive_id"].where(m.index != 0, "ISIC_1000000.jpg")), "unnormalised"),
    (lambda m: m.assign(sha256=m["sha256"].where(m.index % 2 == 0, None)), "field sha256 is filled"),
    (lambda m: m[m["frozen_split"] != "train_added"], "training-only set is empty"),
    (lambda m: m.assign(fold=m["fold"].where(m.index != 0, pd.NA)), "has no fold"),
])
def test_each_table_check_fires(mutate, message: str) -> None:
    master = mutate(_master())
    report = overlap_guard.check(master, _links(), len(master), PLANTS_OK)
    assert not report["passed"] and any(message in f for f in report["failures"]), report["failures"]


def test_fewer_rows_than_declared_fails() -> None:
    report = overlap_guard.check(_master(), _links(), 15, PLANTS_OK)
    assert any("fewer than the 15 declared" in f for f in report["failures"])


def test_an_empty_master_table_fails() -> None:
    report = overlap_guard.check(_master().iloc[0:0], _links(), 0, PLANTS_OK)
    assert not report["passed"] and "empty" in report["failures"][0]


def test_missed_or_absent_plants_fail() -> None:
    assert not overlap_guard.check(_master(), _links(), 14, None)["passed"]
    missed = {"all_found": False, "rotated_copy": {"planted": 3, "found": 2, "recall": 2 / 3}}
    report = overlap_guard.check(_master(), _links(), 14, missed)
    assert any("planted copies missed" in f for f in report["failures"])


def test_percolation_fails_above_the_limit() -> None:
    master, pairs = _master(), [(D[0], D[2]), (D[2], D[4])]
    master["master_group"] = master["image_id"].map(grouping.components(master["image_id"], pairs))
    report = overlap_guard.check(master, _links(*[(a, b, "confirmed") for a, b in pairs]), 14, PLANTS_OK, limit=2)
    assert any("percolation" in f for f in report["failures"])


def test_a_confirmed_component_split_across_master_groups_fails() -> None:
    report = overlap_guard.check(_master(), _links((D[0], D[2], "confirmed")), 14, PLANTS_OK)
    assert any("split across master groups" in f for f in report["failures"])


def test_a_link_to_an_unknown_row_fails() -> None:
    report = overlap_guard.check(_master(), _links((D[0], "ISIC_9999999", "at_risk")), 14, PLANTS_OK)
    assert any("not in the master table" in f for f in report["failures"])


def test_the_residual_leak_check_fires_when_removals_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = {"global": [], "by_fold": {f: [] for f in overlap_guard.DEV_FOLDS}, "by_holdout": {a: [] for a in overlap_guard.ARCHIVES}}
    monkeypatch.setattr(overlap_guard, "removals", lambda master, links: empty)
    report = overlap_guard.check(_master(), _links((D[0], D[2], "at_risk"), ("ISIC_3000001", D[5], "at_risk")), 14, PLANTS_OK)
    assert any(f.startswith("fold 0:") for f in report["failures"])
    assert any(f.startswith("spent sets:") for f in report["failures"])


# ------------------------------------------------------------------ receipt
def test_the_receipt_binds_the_tables_and_refuses_changes(tmp_path: Path) -> None:
    master_path, links_path, receipt = tmp_path / "m.csv", tmp_path / "l.csv", tmp_path / "r.json"
    _master().to_csv(master_path, index=False)
    _links().to_csv(links_path, index=False)
    with pytest.raises(overlap_guard.GuardFailure, match="no data-gate receipt"):
        overlap_guard.require_receipt(receipt, master_path, links_path)
    overlap_guard.write_receipt(receipt, master_path, links_path, overlap_guard.check(_master(), _links(), 14, PLANTS_OK))
    assert overlap_guard.require_receipt(receipt, master_path, links_path)["passed"]
    master_path.write_text(master_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(overlap_guard.GuardFailure, match="master table changed"):
        overlap_guard.require_receipt(receipt, master_path, links_path)


def test_a_failed_gate_receipt_is_refused(tmp_path: Path) -> None:
    master_path, links_path, receipt = tmp_path / "m.csv", tmp_path / "l.csv", tmp_path / "r.json"
    _master().to_csv(master_path, index=False)
    _links().to_csv(links_path, index=False)
    overlap_guard.write_receipt(receipt, master_path, links_path, overlap_guard.check(_master(), _links(), 14, None))
    with pytest.raises(overlap_guard.GuardFailure, match="failed gate"):
        overlap_guard.require_receipt(receipt, master_path, links_path)


def test_patient_keyed_existing_groups_never_merge_lesion_groups() -> None:
    """DERM12345 rows carry V6 groups keyed by patient (`__v6__IP_...`); the data plan forbids patient ids merging groups."""
    frame = pd.DataFrame({"image_id": ["d1", "d2", "d3", "h1", "h2"], "archive_id": ["d1", "d2", "d3", "h1", "h2"],
                          "src_lesion_id": [None, None, None, "HAM_1", "HAM_1"], "src_patient_id": [None] * 5,
                          "group_id": ["__v6__IP_7", "__v6__IP_7", "__v6__IP_7", "g_h", "g_h"]})
    links = identifier_links(frame)
    confirmed = {frozenset((a, b)) for a, b, t, _ in links if t == "confirmed"}
    at_risk = {frozenset((a, b)) for a, b, t, e in links if t == "at_risk" and e == "patient_id"}
    assert not any({"d1", "d2", "d3"} & set(pair) for pair in confirmed)
    assert at_risk == {frozenset(("d1", "d2")), frozenset(("d1", "d3")), frozenset(("d2", "d3"))}
    assert frozenset(("h1", "h2")) in confirmed


def test_a_holdout_drops_training_only_rows_from_the_held_out_archive_itself() -> None:
    """Retired evaluation splits are training-only rows of a development archive; a BCN holdout must drop the BCN ones."""
    master = _master()
    extra = pd.DataFrame([{"image_id": "ISIC_4000001", "archive": "bcn20000", "source": "isic2019_train", "frozen_split": "reserved", "fold": pd.NA},
                          {"image_id": "ISIC_4000002", "archive": "ham", "source": "isic2019_train", "frozen_split": "ham_val", "fold": pd.NA}])
    extra = extra.assign(archive_id=extra["image_id"], master_group=extra["image_id"], group_method="singleton_or_member",
                         sha256=["e1" * 32, "e2" * 32], phash="0" * 16, dhash="0" * 16, status="kept")
    master = pd.concat([master, extra], ignore_index=True)
    master["fold"] = master["fold"].astype("Int64")
    drops = overlap_guard.removals(master, _links())
    assert "ISIC_4000001" in drops["by_holdout"]["bcn20000"] and "ISIC_4000002" in drops["by_holdout"]["ham"]
    assert "ISIC_4000001" not in drops["by_holdout"]["ham"]
    assert overlap_guard.check(master, _links(), len(master), PLANTS_OK)["passed"]


def test_the_check_fails_if_removal_lists_leave_a_held_out_archive_row_in_training(monkeypatch: pytest.MonkeyPatch) -> None:
    master = _master()
    extra = pd.DataFrame([{"image_id": "ISIC_4000001", "archive": "bcn20000", "source": "isic2019_train", "frozen_split": "reserved",
                           "fold": pd.NA, "archive_id": "ISIC_4000001", "master_group": "ISIC_4000001", "group_method": "s",
                           "sha256": "e1" * 32, "phash": "0" * 16, "dhash": "0" * 16, "status": "kept"}])
    master = pd.concat([master, extra], ignore_index=True)
    master["fold"] = master["fold"].astype("Int64")
    real = overlap_guard.removals

    def without_archive_rule(m, links):
        drops = real(m, links)
        drops["by_holdout"]["bcn20000"] = [r for r in drops["by_holdout"]["bcn20000"] if r != "ISIC_4000001"]
        return drops

    monkeypatch.setattr(overlap_guard, "removals", without_archive_rule)
    report = overlap_guard.check(master, _links(), len(master), PLANTS_OK)
    assert any("held-out archive are not in the removal lists" in f for f in report["failures"])


# ------------------------------------------------------------------ declared sources (freeze audit, 11 Nov 2026)
def test_a_declared_source_that_loads_zero_rows_stops_the_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The 10 Nov bug: every HIBA row was dropped by a filter and the gate still passed. The builder now stops."""
    from research.data import build_master_v7 as bm

    hiba = tmp_path / "hiba_cohort.csv"
    pd.DataFrame({"isic_id": ["ISIC_1", "ISIC_2"], "lesion_key": ["L1", "L2"], "patient_id": ["P1", "P2"],
                  "excluded": ["duplicate", "no image"]}).to_csv(hiba, index=False)          # every row excluded: the source is empty
    monkeypatch.setattr(bm, "HIBA", hiba)
    with pytest.raises(AssertionError, match="loaded zero rows"):
        bm.load_rows()


def test_an_absent_empty_or_resized_source_is_a_binding_failure() -> None:
    """The builder adds HIBA and MIDAS only if their file exists; the data-role bindings catch a source that went missing."""
    from research.v7 import data_roles

    declared = dict(data_roles.DECLARED_ROWS)
    assert sum(declared.values()) == 57_488 and sum(v for k, v in declared.items() if k != "midas") == 54_079
    assert data_roles.check_sources(declared) == []
    without_hiba = {k: v for k, v in declared.items() if k != "train_hiba"}
    assert any("'train_hiba' has 0 rows" in p for p in data_roles.check_sources(without_hiba))
    assert any("'midas' has 0 rows" in p for p in data_roles.check_sources({**declared, "midas": 0}))
    assert any("not the declared 15,294" in p for p in data_roles.check_sources({**declared, "train": 15_000}))
    assert any("undeclared source 'new_archive'" in p for p in data_roles.check_sources({**declared, "new_archive": 10}))


def test_a_hash_only_match_stays_at_risk_and_never_merges_lesions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for the seven chained HAM lesions (10 Nov): only identical bytes confirm on pixels alone. A perceptual-hash
    match is a candidate (at risk), so a chain of such matches cannot join different lesions into one group."""
    from research.data import build_master_v7 as bm

    hits = [{"query": "a", "reference": "b", "evidence": "phash"}, {"query": "b", "reference": "c", "evidence": "dhash"},
            {"query": "c", "reference": "d", "evidence": "sha256"}, {"query": "b", "reference": "a", "evidence": "phash"}]
    monkeypatch.setattr(bm.fp, "match", lambda queries, references: hits)
    links = bm.pixel_links({k: object() for k in "abcd"})
    assert links == [("a", "b", "at_risk", "phash"), ("b", "c", "at_risk", "dhash"), ("c", "d", "confirmed", "sha256")]
    confirmed = [(a, b) for a, b, tier, _ in links if tier == "confirmed"]
    groups = grouping.components(list("abcd"), confirmed) if hasattr(grouping, "components") else None
    if groups is not None:
        assert len({groups[k] for k in "abcd"}) == 3          # a, b and {c, d}: the hash chain a-b-c did not merge


def test_the_second_opinion_drops_only_unsupported_single_detector_links(tmp_path, monkeypatch) -> None:
    """Hash / keypoint-hit links that neither M1 nor SIFT supports are dropped; supported, unscored and other links stay."""
    import research.data.build_master_v7 as bm
    monkeypatch.setattr(bm, "OUT_DIR", tmp_path)
    monkeypatch.setattr(bm, "WEAK_RAW", tmp_path / "weak_links_raw.csv")
    frame = pd.DataFrame([("x", "y", "at_risk", "hash_at_risk"), ("x", "z", "at_risk", "hash_at_risk"),
                          ("p", "q", "at_risk", "sift_eval_hits"), ("x", "y", "at_risk", "patient_id"),
                          ("u", "v", "at_risk", "hash_at_risk")], columns=["a", "b", "tier", "evidence"])
    monkeypatch.setattr(bm, "SECOND_OPINION", tmp_path / "absent.csv")
    same, info = bm.second_opinion_filter(frame)
    assert len(same) == 5 and info["dropped"] == 0                           # no opinion file: nothing dropped
    opinion = tmp_path / "second_opinion.csv"
    pd.DataFrame([("x", "y", "hash_at_risk", "unsupported"), ("x", "z", "hash_at_risk", "supported"),
                  ("p", "q", "sift_eval_hits", "unsupported")], columns=["a", "b", "evidence", "verdict"]).to_csv(opinion, index=False)
    monkeypatch.setattr(bm, "SECOND_OPINION", opinion)
    kept, info = bm.second_opinion_filter(frame)
    assert set(map(tuple, kept[["a", "b", "evidence"]].values)) == {("x", "z", "hash_at_risk"), ("x", "y", "patient_id"), ("u", "v", "hash_at_risk")}
    assert info["dropped"] == 2 and info["kept_unscored"] == 1
    assert len(pd.read_csv(tmp_path / "weak_links_raw.csv")) == 4             # every weak link written before filtering
