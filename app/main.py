import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_model.onnx"
N_FEATURES = 30

app = FastAPI(title="Edge-Cloud Inference Router")
session = ort.InferenceSession(str(MODEL_PATH))
input_name = session.get_inputs()[0].name


class Transaction(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


class Prediction(BaseModel):
    fraud_probability: float
    route: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/edge/predict", response_model=Prediction)
def edge_predict(tx: Transaction):
    start = time.perf_counter()
    x = np.asarray([tx.features], dtype=np.float32)
    probs = session.run(None, {input_name: x})[1]
    p = float(probs[0][1])
    return Prediction(
        fraud_probability=p,
        route="edge",
        latency_ms=(time.perf_counter() - start) * 1000,
    )
