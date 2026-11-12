"""The V7 promotion runner (freeze blocker F7), on synthetic rows and synthetic prediction files only.

No real prediction, label or signature is read: every source the runner touches is replaced through `Sources`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.v6 import system_reads as sr
from research.v7 import promotion, promotion_run as pr

SIGNATURE = "sig-gate0"
RECEIPT = "receipt-1"
ARCHIVES = ("bcn20000", "ham", "mskcc")


def _rows(n_groups: int = 900, seed: int = 7) -> tuple[sr.Rows, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), 2)
    y = np.repeat(rng.choice(7, n_groups, p=[0.07, 0.1, 0.13, 0.05, 0.15, 0.45, 0.05]), 2)
    band = np.repeat(rng.choice(["<40", "40-59", "60+"], n_groups, p=[0.3, 0.3, 0.4]), 2)
    rows = sr.Rows(image_id=np.array([f"ISIC_{k:05d}" for k in range(len(y))]), fold=np.repeat(rng.integers(0, 5, n_groups), 2), y=y,
                   esc=np.isin(y, (0, 1, 4)), u40=band == "<40", group=np.array([f"g{g}" for g in groups]),
                   archive=np.repeat(rng.choice(ARCHIVES, n_groups), 2))
    return rows, band


def _probs(rows: sr.Rows, strength: float, seed: int) -> np.ndarray:
    p = np.random.default_rng(seed).random((len(rows.y), 7))
    p[np.arange(len(rows.y)), rows.y] += strength
    return p / p.sum(1, keepdims=True)


def _write_predictions(path: Path, rows: sr.Rows, probs: np.ndarray, where: np.ndarray, run_id: str) -> None:
    frame = pd.DataFrame({"image_id": rows.image_id[where], "group_id": rows.group[where], "y_true": rows.y[where]})
    for i, c in enumerate(pr.CLASSES):
        frame[f"p_{c}"] = probs[where, i]
    frame["run_id"] = run_id
    frame.to_csv(path, index=False)


def _write_run(path: Path, run_id: str, fold: int | None, holdout: str | None, build: str = "B3", member: str = "anchor",
               image_size: int = 384, **changes) -> None:
    run = {"run_id": run_id, "member": member, "image_size": image_size, "smoke": False, "epochs": 30, "epochs_run": 30, "optimizer_steps": 13410, "step_budget": 13410,
           "plan_freeze_sha256": SIGNATURE, "data_gate_receipt_sha256": RECEIPT, "v7_build": build, "fold": fold if fold is not None else 0,
           "loao_holdout": holdout, "test_read": False, "reserved_read": False, **changes}
    path.write_text(json.dumps(run), encoding="utf-8")


class World:
    """A synthetic repository: rows, a V6 reference, a candidate's files, and the Sources that point at them."""

    def __init__(self, tmp: Path, strength: float = 1.6) -> None:
        self.tmp, (self.rows, self.band) = tmp, _rows()
        self.reference = _probs(self.rows, 0.7, 1)
        self.candidate = _probs(self.rows, strength, 2)
        self.calls = {"rows": 0, "reference": 0, "dark": 0}
        ids = self.rows.image_id
        self.src = pr.Sources(
            gate0=lambda: SIGNATURE, signatures=lambda: {SIGNATURE}, frozen_files=lambda: {"rule.py": "abc"},
            rows=self._load_rows, reference=self._load_reference,
            declared_lists=lambda: {"without_380_rows": set(ids[:40]), "without_426_rows": set(ids[:60])},
            v7_linked=lambda: set(ids[100:180]), histo=lambda image_ids: np.arange(len(image_ids)) % 2 == 0, bar=lambda: 0.008,
            receipt_sha=lambda: RECEIPT, loao_dropped=lambda: frozenset({"g3"}), dark=self._dark, out_dir=tmp / "promotion", n_boot=60)

    def _load_rows(self):
        self.calls["rows"] += 1
        return self.rows, self.band

    def _load_reference(self, rows):
        self.calls["reference"] += 1
        return self.reference, {"v6.csv": "hash"}

    def _dark(self, mass, name):
        self.calls["dark"] += 1
        return {"candidate": name, "rows": int(len(mass))}

    def spec(self, name: str = "c1", members: tuple[str, ...] = ("m1", "m2")) -> dict:
        out: dict = {"name": name, "members": {}}
        for m in members:
            out["members"][m] = {}
            for fold in pr.READ_FOLDS:
                pred, run = self.tmp / f"{name}_{m}_f{fold}.csv", self.tmp / f"{name}_{m}_f{fold}.json"
                _write_predictions(pred, self.rows, self.candidate, np.where(self.rows.fold == fold)[0], f"{m}_f{fold}")
                _write_run(run, f"{m}_f{fold}", fold, None)
                out["members"][m][str(fold)] = {"predictions": str(pred), "run": str(run)}
        return out

    def loao_spec(self, name: str = "c1", candidate_members: tuple[str, ...] = ("m1", "m2")) -> dict:
        """The candidate arm is the frozen candidate's own members; the baseline is the declared V6-recipe vote."""
        out: dict = {"candidate": {}, "baseline": {}}
        arms = (("candidate", self.candidate, {m: ("B3", "anchor", 384) for m in candidate_members}),
                ("baseline", self.reference, {k: ("L0", *v) for k, v in pr.P3_BASELINE_MEMBERS.items()}))
        for arm, probs, roster in arms:
            for a in ARCHIVES:
                where = np.where((self.rows.archive == a) & (self.rows.group != "g3"))[0]
                out[arm][a] = {}
                for label, (build, member, size) in roster.items():
                    pred, run = self.tmp / f"{name}_{arm}_{a}_{label}.csv", self.tmp / f"{name}_{arm}_{a}_{label}.json"
                    _write_predictions(pred, self.rows, probs, where, f"{arm}_{a}_{label}")
                    _write_run(run, f"{arm}_{a}_{label}", None, a, build, member, size)
                    out[arm][a][label] = {"predictions": str(pred), "run": str(run)}
        return out


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_the_full_workflow_reads_once_writes_hashes_and_promotes_a_clearly_better_candidate(world: World) -> None:
    pr.freeze_candidate(world.spec(), world.src)
    pr.freeze_loao("c1", world.loao_spec(), world.src)
    report = pr.read("c1", owner_approved=True, src=world.src)
    assert report["status"] == "PROMOTED" and report["decision"] == "V7 candidate replaces V6"
    assert list(report["in_domain"]) == list(promotion.ROW_SETS) and sorted(report["cross_site"]) == sorted(ARCHIVES)
    n_all = int((world.rows.fold != 0).sum())
    assert report["rows"]["all_rows"] == n_all and report["rows"]["without_426_rows"] < report["rows"]["without_380_rows"] < n_all
    assert report["secondary"]["decisive"] is False and report["secondary"]["holm_family"]["size"] == len(pr.SECONDARY_FAMILY) == 10
    assert report["hashes"]["candidate_manifest"] and report["hashes"]["loao_manifest"] and report["gate0_record_sha256"] == SIGNATURE
    assert report["protected_reads"] == {"fold_0_pooled": False, "test_read": False, "reserved_read": False, "hiba_read": False, "midas_read": False}
    out = world.src.out_dir / "c1"
    assert json.loads((out / "read.json").read_text(encoding="utf-8"))["status"] == "PROMOTED" and (out / "read.csv").is_file()
    with pytest.raises(SystemExit, match="read once per frozen candidate"):
        pr.read("c1", owner_approved=True, src=world.src)
    again = pr.read("c1", owner_approved=True, rerun_reason="checking a seed", src=world.src)
    assert again["rerun_reason"] == "checking a seed" and (out / "read_rerun_1.json").is_file() and (out / "read.json").is_file()


def test_a_worse_candidate_keeps_v6(tmp_path: Path) -> None:
    world = World(tmp_path, strength=0.3)
    pr.freeze_candidate(world.spec(), world.src)
    pr.freeze_loao("c1", world.loao_spec(), world.src)
    report = pr.read("c1", owner_approved=True, src=world.src)
    assert report["status"] == "NOT_PROMOTED" and report["decision"] == "V6 stays" and report["verdict"]["reasons"]


def test_without_cross_site_files_nothing_is_promoted(world: World) -> None:
    pr.freeze_candidate(world.spec(), world.src)
    with pytest.raises(SystemExit, match="P3 cannot be read"):
        pr.read("c1", owner_approved=True, src=world.src)
    report = pr.read("c1", owner_approved=True, allow_missing_p3=True, src=world.src)
    assert report["status"] == "NOT_ESTIMABLE" and report["decision"] == "V6 stays" and report["cross_site"] is None


def test_the_read_needs_the_owner_and_a_frozen_candidate(world: World) -> None:
    with pytest.raises(SystemExit, match="owner-approved"):
        pr.read("c1", owner_approved=False, src=world.src)
    with pytest.raises(SystemExit, match="not frozen"):
        pr.read("c1", owner_approved=True, src=world.src)
    with pytest.raises(SystemExit, match="freeze the candidate first"):
        pr.freeze_loao("c1", world.loao_spec(), world.src)
    assert world.calls == {"rows": 0, "reference": 0, "dark": 0}


def test_an_unsigned_or_changed_gate_0_stops_everything(world: World) -> None:
    def unsigned() -> str:
        raise SystemExit("V7 Gate 0 is unsigned")
    world.src.gate0 = unsigned
    spec = world.spec()
    with pytest.raises(SystemExit, match="unsigned"):
        pr.freeze_candidate(spec, world.src)
    world.src.gate0 = lambda: SIGNATURE
    pr.freeze_candidate(spec, world.src)
    world.src.gate0 = unsigned
    for call in (lambda: pr.check("c1", world.src), lambda: pr.read("c1", owner_approved=True, allow_missing_p3=True, src=world.src)):
        with pytest.raises(SystemExit, match="unsigned"):
            call()
    assert world.calls["rows"] == 0


@pytest.mark.parametrize(("change", "message"), [
    ({"smoke": True}, "a smoke"), ({"epochs_run": 12}, "not a complete run"), ({"optimizer_steps": 100}, "not a complete run"),
    ({"plan_freeze_sha256": "made-before-signing"}, "not made under a recorded Gate 0"), ({"plan_freeze_sha256": None}, "not made under a recorded Gate 0"),
    ({"data_gate_receipt_sha256": "old"}, "different data-gate receipt"), ({"v7_build": "B99"}, "not a declared"),
    ({"v7_build": "R4"}, "not a declared"), ({"fold": 3}, "not a fold-1 in-domain run"), ({"loao_holdout": "ham"}, "not a fold-1 in-domain run"),
    ({"test_read": True}, "protected read")])
def test_a_training_run_that_is_not_a_signed_complete_run_of_record_is_refused(world: World, change: dict, message: str) -> None:
    spec = world.spec()
    run = Path(spec["members"]["m1"]["1"]["run"])
    run.write_text(json.dumps({**json.loads(run.read_text(encoding="utf-8")), **change}), encoding="utf-8")
    with pytest.raises(SystemExit, match=message):
        pr.freeze_candidate(spec, world.src)
    assert not (world.src.out_dir / "c1").exists()


def test_a_candidate_must_cover_folds_1_to_4_and_is_frozen_once(world: World) -> None:
    spec = world.spec()
    short = {"name": "c2", "members": {"m1": {k: v for k, v in spec["members"]["m1"].items() if k != "4"}}}
    with pytest.raises(SystemExit, match="must name folds"):
        pr.freeze_candidate(short, world.src)
    with_fold_0 = {"name": "c3", "members": {"m1": {**spec["members"]["m1"], "0": spec["members"]["m1"]["1"]}}}
    with pytest.raises(SystemExit, match="must name folds"):
        pr.freeze_candidate(with_fold_0, world.src)
    with pytest.raises(SystemExit, match="no members"):
        pr.freeze_candidate({"name": "c4", "members": {}}, world.src)
    with pytest.raises(SystemExit, match="candidate name"):
        pr.freeze_candidate({**spec, "name": "../escape"}, world.src)
    pr.freeze_candidate(spec, world.src)
    with pytest.raises(SystemExit, match="already frozen"):
        pr.freeze_candidate(spec, world.src)


def test_a_file_changed_after_freezing_or_a_stale_receipt_is_refused(world: World) -> None:
    spec = world.spec()
    pr.freeze_candidate(spec, world.src)
    pred = Path(spec["members"]["m2"]["3"]["predictions"])
    original = pred.read_bytes()
    pred.write_bytes(original + b"\n")
    with pytest.raises(SystemExit, match="changed after the candidate was frozen"):
        pr.check("c1", world.src)
    pred.write_bytes(original)
    assert pr.check("c1", world.src)["in_domain_files"] == 8
    world.src.receipt_sha = lambda: "a-new-gate-run"
    with pytest.raises(SystemExit, match="receipt changed"):
        pr.check("c1", world.src)


def _rewrite(spec: dict, member: str, fold: str, edit) -> None:
    path = Path(spec["members"][member][fold]["predictions"])
    frame = edit(pd.read_csv(path, dtype={"image_id": str}))
    frame.to_csv(path, index=False)


@pytest.mark.parametrize(("edit", "message"), [
    (lambda f: f.iloc[:-3], "missing"),
    (lambda f: pd.concat([f, f.iloc[:2]]), "duplicated"),
    (lambda f: f.assign(image_id=["HAM_TEST_0001", *f["image_id"][1:]]), "not development rows"),
    (lambda f: f.assign(image_id=["MIDAS_0001", *f["image_id"][1:]]), "not development rows"),
    (lambda f: f.assign(p_mel=np.nan), "not finite"),
    (lambda f: f.assign(p_mel=f["p_mel"] + 0.5), "summing to 1"),
    (lambda f: f.assign(p_nv=-0.1), "not finite, in"),
    (lambda f: f.drop(columns="p_vasc"), "missing columns"),
    (lambda f: f[["image_id", "group_id", "y_true", "p_bcc", "p_akiec", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc", "run_id"]], "frozen order"),
    (lambda f: f.assign(y_true=(f["y_true"] + 1) % 7), "labels differ"),
    (lambda f: f.assign(group_id="gX"), "lesion groups differ")])
def test_predictions_that_do_not_match_the_declared_rows_are_refused(world: World, edit, message: str) -> None:
    spec = world.spec()
    _rewrite(spec, "m1", "2", edit)
    pr.freeze_candidate(spec, world.src)
    with pytest.raises(SystemExit, match=message):
        pr.read("c1", owner_approved=True, allow_missing_p3=True, src=world.src)
    assert not list((world.src.out_dir / "c1").glob("read*"))


def test_predictions_are_bound_to_their_training_run(world: World) -> None:
    spec = world.spec()
    spec["members"]["m1"]["2"]["predictions"] = spec["members"]["m1"]["3"]["predictions"]       # another fold's file
    with pytest.raises(SystemExit, match="does not carry run_id 'm1_f2'"):
        pr.freeze_candidate(spec, world.src)
    spec = world.spec("c2")
    _rewrite(spec, "m1", "2", lambda f: f.assign(run_id="another_run"))
    with pytest.raises(SystemExit, match="does not carry run_id"):
        pr.freeze_candidate(spec, world.src)
    _rewrite(spec, "m1", "2", lambda f: f.drop(columns="run_id"))
    with pytest.raises(SystemExit, match="does not carry run_id"):
        pr.freeze_candidate(spec, world.src)


def test_a_member_listed_twice_is_refused(world: World) -> None:
    spec = world.spec()
    spec["members"]["m3"] = spec["members"]["m1"]
    with pytest.raises(SystemExit, match="counted twice"):
        pr.freeze_candidate(spec, world.src)


def test_a_second_candidate_is_read_only_with_a_reason_and_the_first_is_disclosed(world: World) -> None:
    pr.freeze_candidate(world.spec("c1"), world.src)
    pr.freeze_candidate(world.spec("c2"), world.src)
    first = pr.read("c1", owner_approved=True, allow_missing_p3=True, src=world.src)
    assert first["read_of_record"] and first["other_candidates_read_before"] == []
    with pytest.raises(SystemExit, match="other candidates have already been read"):
        pr.read("c2", owner_approved=True, allow_missing_p3=True, src=world.src)
    second = pr.read("c2", owner_approved=True, allow_missing_p3=True, rerun_reason="second candidate, disclosed", src=world.src)
    assert not second["read_of_record"] and second["other_candidates_read_before"] == ["c1"]


def test_the_cross_site_files_need_every_archive_both_arms_and_the_l0_baseline(world: World) -> None:
    pr.freeze_candidate(world.spec(), world.src)
    loao = world.loao_spec()
    with pytest.raises(SystemExit, match="every archive"):
        pr.freeze_loao("c1", {"candidate": loao["candidate"], "baseline": {k: v for k, v in loao["baseline"].items() if k != "ham"}}, world.src)
    with pytest.raises(SystemExit, match="exactly 'candidate' and 'baseline'"):
        pr.freeze_loao("c1", {"candidate": loao["candidate"]}, world.src)
    weak = {**loao, "baseline": {a: {"anchor_h7": m["anchor_h7"]} for a, m in loao["baseline"].items()}}
    with pytest.raises(SystemExit, match="member for member"):                         # one anchor against a two-member vote
        pr.freeze_loao("c1", weak, world.src)
    with pytest.raises(SystemExit, match="member for member"):                         # the candidate arm is its frozen roster
        pr.freeze_loao("c1", world.loao_spec("c1x", candidate_members=("m1",)), world.src)
    run = Path(loao["baseline"]["ham"]["dinov2_b_h7"]["run"])
    run.write_text(json.dumps({**json.loads(run.read_text(encoding="utf-8")), "image_size": 392}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not \\('dinov2_b', 224\\)"):
        pr.freeze_loao("c1", loao, world.src)
    run.write_text(json.dumps({**json.loads(run.read_text(encoding="utf-8")), "image_size": 224}), encoding="utf-8")
    run = Path(loao["baseline"]["ham"]["anchor_h7"]["run"])
    run.write_text(json.dumps({**json.loads(run.read_text(encoding="utf-8")), "v7_build": "B3"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="expected 'L0'"):
        pr.freeze_loao("c1", loao, world.src)
    run.write_text(json.dumps({**json.loads(run.read_text(encoding="utf-8")), "v7_build": "L0", "loao_holdout": "mskcc"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a leave-ham-out run"):
        pr.freeze_loao("c1", loao, world.src)
    assert not (world.src.out_dir / "c1" / "loao.json").exists()


def test_cross_site_arms_must_score_exactly_the_held_out_archives_rows(world: World) -> None:
    pr.freeze_candidate(world.spec(), world.src)
    loao = world.loao_spec()
    path = Path(loao["candidate"]["bcn20000"]["m1"]["predictions"])
    pd.read_csv(path, dtype={"image_id": str}).iloc[:-10].to_csv(path, index=False)
    pr.freeze_loao("c1", loao, world.src)
    with pytest.raises(SystemExit, match="do not match the declared set"):
        pr.read("c1", owner_approved=True, src=world.src)


def test_the_dry_run_opens_no_prediction_and_no_label(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    pr.freeze_candidate(world.spec(), world.src)
    pr.freeze_loao("c1", world.loao_spec(), world.src)

    def forbidden(*args, **kwargs):
        raise AssertionError("the dry run read a CSV")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    out = pr.check("c1", world.src)
    assert out["predictions_opened"] == 0 and out["cross_site_frozen"] and out["already_read"] == []
    assert world.calls == {"rows": 0, "reference": 0, "dark": 0}


def test_help_runs_without_touching_any_source(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as stop:
        pr.main(["--help"])
    assert stop.value.code == 0 and "freeze-candidate" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pr.main(["read", "--help"])


def test_the_row_sets_are_the_declared_four_and_only_remove_evaluation_rows(world: World) -> None:
    pr.freeze_candidate(world.spec(), world.src)
    report = pr.read("c1", owner_approved=True, allow_missing_p3=True, src=world.src)
    ids, folds = world.rows.image_id, world.rows.fold
    read_rows = folds != 0
    assert report["rows"]["without_380_rows"] == int((read_rows & ~np.isin(ids, ids[:40])).sum())
    assert report["rows"]["without_v7_linked_rows"] == int((read_rows & ~np.isin(ids, ids[100:180])).sum())
    assert "training side is the same" in report["row_set_definition"]
    assert tuple(report["in_domain"]) == promotion.ROW_SETS == ("all_rows", "without_380_rows", "without_426_rows", "without_v7_linked_rows")
