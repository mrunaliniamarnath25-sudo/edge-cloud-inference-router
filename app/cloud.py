import asyncio
import os
import random
import time

from fastapi import FastAPI, Query

from app.model import cloud_score
from app.schemas import Prediction, Transaction

DELAY_MS = float(os.getenv("CLOUD_DELAY_MS", "500"))
JITTER_MS = float(os.getenv("CLOUD_JITTER_MS", "0"))
DROP_PCT = float(os.getenv("CLOUD_DROP_PCT", "0"))
HANG_S = float(os.getenv("CLOUD_HANG_S", "5"))
_seed = os.getenv("CLOUD_SEED")
rng = random.Random(int(_seed) if _seed else None)

app = FastAPI(title="Simulated Cloud")
cloud_score([0.0] * 30)  # load and warm the cloud model at startup


@app.post("/cloud/predict", response_model=Prediction)
async def cloud_predict(tx: Transaction):
    start = time.perf_counter()
    if DROP_PCT > 0 and rng.random() * 100 < DROP_PCT:
        await asyncio.sleep(HANG_S)  # a request that never answers in time
    extra_ms = rng.expovariate(1.0 / JITTER_MS) if JITTER_MS > 0 else 0.0
    await asyncio.sleep((DELAY_MS + extra_ms) / 1000)
    p = cloud_score(tx.features)
    return Prediction(
        fraud_probability=p,
        route="cloud",
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@app.post("/admin/chaos")
def set_chaos(
    jitter_ms: float = Query(0.0, ge=0),
    drop_pct: float = Query(0.0, ge=0, le=100),
):
    global JITTER_MS, DROP_PCT
    JITTER_MS = jitter_ms
    DROP_PCT = drop_pct
    return {"jitter_ms": JITTER_MS, "drop_pct": DROP_PCT}
