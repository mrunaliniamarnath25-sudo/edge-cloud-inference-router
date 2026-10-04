import os
import time
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException

from app.model import score
from app.schemas import Prediction, Transaction

CLOUD_URL = os.getenv("CLOUD_URL", "http://127.0.0.1:8001/cloud/predict")
CLOUD_TIMEOUT_S = float(os.getenv("CLOUD_TIMEOUT_S", "1.0"))
LOW = float(os.getenv("ROUTE_LOW", "0.1"))
HIGH = float(os.getenv("ROUTE_HIGH", "0.9"))

app = FastAPI(title="Edge-Cloud Inference Router")
client = httpx.AsyncClient(timeout=CLOUD_TIMEOUT_S)
state = {"cloud_down": False}


def ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def call_cloud(tx: Transaction) -> tuple[Prediction, int]:
    if state["cloud_down"]:
        raise httpx.ConnectError("simulated outage")
    payload = tx.model_dump_json().encode()
    r = await client.post(
        CLOUD_URL, content=payload, headers={"Content-Type": "application/json"}
    )
    r.raise_for_status()
    return Prediction(**r.json()), len(payload)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/outage")
def set_outage(down: bool):
    state["cloud_down"] = down
    return state


@app.post("/edge/predict", response_model=Prediction)
def edge_predict(tx: Transaction):
    start = time.perf_counter()
    return Prediction(
        fraud_probability=score(tx.features), route="edge", latency_ms=ms(start)
    )


@app.post("/predict", response_model=Prediction)
async def predict(
    tx: Transaction,
    mode: Literal["hybrid", "cloud_only", "edge_only"] = "hybrid",
):
    start = time.perf_counter()

    if mode == "cloud_only":
        try:
            result, sent = await call_cloud(tx)
        except httpx.HTTPError:
            raise HTTPException(status_code=503, detail="cloud unavailable")
        return Prediction(
            fraud_probability=result.fraud_probability,
            route="cloud",
            latency_ms=ms(start),
            bytes_to_cloud=sent,
        )

    p = score(tx.features)
    if mode == "edge_only" or p <= LOW or p >= HIGH:
        return Prediction(fraud_probability=p, route="edge", latency_ms=ms(start))

    try:
        result, sent = await call_cloud(tx)
    except httpx.HTTPError:
        return Prediction(
            fraud_probability=p, route="edge", latency_ms=ms(start), fell_back=True
        )
    return Prediction(
        fraud_probability=result.fraud_probability,
        route="cloud",
        latency_ms=ms(start),
        bytes_to_cloud=sent,
    )
