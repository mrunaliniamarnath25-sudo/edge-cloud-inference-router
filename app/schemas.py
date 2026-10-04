from pydantic import BaseModel, Field

from app.model import N_FEATURES


class Transaction(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


class Prediction(BaseModel):
    fraud_probability: float
    route: str
    latency_ms: float
    fell_back: bool = False
    bytes_to_cloud: int = 0
