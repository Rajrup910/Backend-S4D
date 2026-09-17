"""Tests for the S60 decision service, with the model stubbed (probs supplied directly).

No torch, no ONNX, no GPU: `DecisionEngine`/the FastAPI app score pre-computed softmax
vectors against frozen artifacts already on disk (S56 abstention, S6/S9 conformal, S5/S12
age-lambda, S55's deployed Dirichlet). If any of those files move, these tests fail loudly
rather than silently drifting -- that is the point of anchoring to `--selftest`-verified
values in `research/v4/s60_contract.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from research.v4.s60_api import app, get_engine
from research.v4.s60_contract import DecisionEngine, raps_scores_single


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    return DecisionEngine()


@pytest.fixture(scope="module")
def client() -> TestClient:
    get_engine.cache_clear()
    return TestClient(app)


CONFIDENT_NV = [0.01, 0.01, 0.02, 0.01, 0.02, 0.90, 0.03]
UNCERTAIN_MIXED = [0.12, 0.10, 0.15, 0.05, 0.18, 0.30, 0.10]


# ------------------------------------------------------------------------- DecisionEngine
def test_class_order_matches_mapping(engine: DecisionEngine) -> None:
    assert engine.class_info.codes == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert engine.class_info.escalating_codes == ("akiec", "bcc", "mel")


def test_confident_call_not_referred(engine: DecisionEngine) -> None:
    d = engine.decide(CONFIDENT_NV, age=65, calibrated=True)
    assert d.predicted_class == "nv"
    assert d.age.band == "60+"
    assert not d.abstain


def test_uncertain_under40_is_referred(engine: DecisionEngine) -> None:
    d = engine.decide(UNCERTAIN_MIXED, age=25, calibrated=True)
    assert d.age.band == "<40"
    assert d.abstain


def test_unknown_age_falls_back_to_global_threshold(engine: DecisionEngine) -> None:
    d = engine.decide(CONFIDENT_NV, age=None, calibrated=True)
    assert d.age.band == "unknown"
    assert d.age.source == "unknown"
    assert d.abstention_threshold == engine.abstention.tau_global


def test_conformal_set_always_contains_argmax_class(engine: DecisionEngine) -> None:
    # score(argmax) = cumulative mass up to and including it, minus its own mass, plus its own
    # mass (u=1) = mass of everything ranked >= it, which is always <= the top rank's own
    # quantile as long as the quantile is finite; check membership directly for both fixtures.
    for probs, age in [(CONFIDENT_NV, 65), (UNCERTAIN_MIXED, 25)]:
        d = engine.decide(probs, age=age, calibrated=True)
        assert d.predicted_class in d.conformal_set, (probs, d.conformal_set)


def test_false_reassurance_never_fires_when_referred(engine: DecisionEngine) -> None:
    for probs, age in [(CONFIDENT_NV, 65), (UNCERTAIN_MIXED, 25), (CONFIDENT_NV, None)]:
        d = engine.decide(probs, age=age, calibrated=True)
        if d.abstain:
            assert not d.false_reassurance_risk


def test_lambda_rule_is_informational_only(engine: DecisionEngine) -> None:
    """The returned `predicted_class` must be the plain argmax, never the lambda-shifted one."""
    d = engine.decide(UNCERTAIN_MIXED, age=25, calibrated=True)
    raw = np.asarray(UNCERTAIN_MIXED)
    assert d.predicted_class == engine.class_info.codes[int(raw.argmax())]


def test_ood_and_gradcam_are_honestly_absent(engine: DecisionEngine) -> None:
    d = engine.decide(CONFIDENT_NV, age=65, calibrated=True)
    payload = d.to_dict()
    assert payload["ood"]["available"] is False
    assert payload["ood"]["score"] is None
    assert payload["gradcam"]["overlay"] is None


def test_uncalibrated_input_is_calibrated_before_scoring(engine: DecisionEngine) -> None:
    raw = [0.01, 0.01, 0.02, 0.01, 0.02, 0.90, 0.03]
    d_raw = engine.decide(raw, age=65, calibrated=False)
    d_pre = engine.decide(raw, age=65, calibrated=True)
    assert d_raw.probs_calibrated != d_pre.probs_calibrated


def test_rejects_wrong_length(engine: DecisionEngine) -> None:
    with pytest.raises(ValueError):
        engine.decide([0.5, 0.5], age=30, calibrated=True)


def test_rejects_probs_not_summing_to_one(engine: DecisionEngine) -> None:
    with pytest.raises(ValueError):
        engine.decide([0.5] * 7, age=30, calibrated=True)


def test_raps_scores_monotone_with_rank() -> None:
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
    scores = raps_scores_single(probs, escalating_idx=(0, 1, 4), k_reg=1, penalty=0.05)
    order = np.argsort(-probs)
    assert np.all(np.diff(scores[order]) >= -1e-9)


# --------------------------------------------------------------------------------- FastAPI
def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_contract_endpoint_carries_fairness_notice(client: TestClient) -> None:
    r = client.get("/contract")
    assert r.status_code == 200
    body = r.json()
    assert "Fitzpatrick" in body["fairness_notice"]
    assert body["floor_oof"] == 0.855
    assert len(body["plan_sha256"]) == 64


def test_contract_reports_s59_external_failure(client: TestClient) -> None:
    """The service must not hide that its contract failed off-distribution (S59)."""
    body = client.get("/contract").json()
    assert body["s59_stack"] == "S56"
    ext = body["external_contract"]
    assert ext["panel"] == "reserved"
    assert ext["joint"] == "CONTRACT_FAILS"
    assert all(t["status"] == "NOT_MET" for t in ext["terms"].values())
    pred = client.post("/predict", json={"probs": CONFIDENT_NV, "age": 65, "calibrated": True})
    assert pred.json()["contract"]["external_contract"]["joint"] == "CONTRACT_FAILS"


def test_predict_endpoint_happy_path(client: TestClient) -> None:
    r = client.post("/predict", json={"probs": CONFIDENT_NV, "age": 65, "calibrated": True})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "nv"
    assert body["age"]["band"] == "60+"
    assert "Fitzpatrick" in body["contract"]["fairness_notice"]
    assert body["nnb"] is not None


def test_predict_endpoint_rejects_bad_payload(client: TestClient) -> None:
    r = client.post("/predict", json={"probs": [0.1, 0.9], "age": 40})
    assert r.status_code == 422


def test_predict_endpoint_rejects_non_normalized_probs(client: TestClient) -> None:
    r = client.post("/predict", json={"probs": [0.5] * 7, "age": 40, "calibrated": True})
    assert r.status_code == 422


def test_predict_endpoint_age_optional(client: TestClient) -> None:
    r = client.post("/predict", json={"probs": CONFIDENT_NV, "calibrated": True})
    assert r.status_code == 200
    assert r.json()["age"]["source"] == "unknown"
