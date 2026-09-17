"""S60 -- FastAPI service exposing `s60_contract.DecisionEngine`.

Stubbed tonight, deliberately (V4 runbook Sec.5 S60; a GPU training run has the machine
tonight, and the ONNX export is gated on S53r/S63 deciding the winning system): `/predict`
takes the model's 7-class softmax output as input rather than an image, so no model is loaded
and no ONNX runtime is imported. (`torch` is imported transitively by
`research.calibration.methods`, which applies the Dirichlet map; it stays on CPU and allocates
nothing on the GPU.) Wiring a real model in later means adding one
function that turns an image into `probs` before calling `DecisionEngine.decide` --
everything downstream (calibration, age rule, abstention, conformal set, contract) is
already real, frozen-artifact-backed logic, not a mock.

    $py -m research.v4.s60_api                 # runs on 127.0.0.1:8060
    $py -m research.v4.s60_api --port 9000
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from research.v4.s60_contract import DecisionEngine

app = FastAPI(
    title="Skin Lesion Triage -- Decision Service (S60, stub)",
    description=(
        "Every response carries its own guarantee: class, conformal set, band abstention "
        "and threshold, the age used and the lambda it produced, and the policy contract "
        "the decision was made under. Model is stubbed -- callers supply the 7-class "
        "softmax vector directly; no image inference happens in this process."
    ),
    version="0.1.0-stub",
)


@lru_cache(maxsize=1)
def get_engine() -> DecisionEngine:
    return DecisionEngine()


class PredictRequest(BaseModel):
    probs: list[float] = Field(
        ..., min_length=7, max_length=7,
        description="7-class softmax vector in class_mapping.json index order "
                    "(akiec, bcc, bkl, df, mel, nv, vasc).",
    )
    age: Optional[float] = Field(
        None, description="Patient age in years, if recorded. Omit or null if unknown -- "
                           "this service does NOT estimate age from the image tonight "
                           "(S57a: the estimator has a documented +19.5y bias under 40 and "
                           "needs backbone features this stub does not compute).",
    )
    calibrated: bool = Field(
        False, description="True if `probs` is already Dirichlet-calibrated. False (default) "
                            "applies the deployed global Dirichlet map "
                            "(research/selective/results_oof/fit_state.json) first.",
    )


class HealthResponse(BaseModel):
    status: str
    engine_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_engine()
        return HealthResponse(status="ok", engine_loaded=True)
    except Exception as exc:  # noqa: BLE001 -- health check must report, not raise
        raise HTTPException(status_code=503, detail=f"engine failed to load: {exc}") from exc


@app.get("/contract")
def contract() -> dict:
    """Static policy metadata: what is deployed, its frozen plan hash, its known limits.

    Intended for S61's drift hooks and for any client that wants to display the system's
    guarantees without submitting a case.
    """
    c = get_engine().contract
    return {
        "plan_sha256": c.plan_sha256,
        "primary_budget": c.primary_budget,
        "floor_oof": c.floor_oof,
        "abstention_score": c.abstention_score,
        "decision_rule": c.decision_rule,
        "route": c.route,
        "fairness_notice": c.fairness_notice,
        "under40_notice": c.under40_notice,
        "known_limitations": c.known_limitations,
        "band_operating_point": c.band_operating_point,
        "s59_plan_sha256": c.s59_plan_sha256,
        "s59_stack": c.s59_stack,
        "external_contract": c.external_contract,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    engine = get_engine()
    try:
        decision = engine.decide(req.probs, age=req.age, calibrated=req.calibrated)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return decision.to_dict()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("research.v4.s60_api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
