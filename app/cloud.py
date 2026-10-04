import asyncio
import os
import time

from fastapi import FastAPI

from app.model import score
from app.schemas import Prediction, Transaction

DELAY_MS = float(os.getenv("CLOUD_DELAY_MS", "500"))

app = FastAPI(title="Simulated Cloud")


@app.post("/cloud/predict", response_model=Prediction)
async def cloud_predict(tx: Transaction):
    start = time.perf_counter()
    await asyncio.sleep(DELAY_MS / 1000)
    p = score(tx.features)
    return Prediction(
        fraud_probability=p,
        route="cloud",
        latency_ms=(time.perf_counter() - start) * 1000,
    )
