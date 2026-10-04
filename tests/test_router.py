import httpx
from fastapi.testclient import TestClient

from app import main
from app.schemas import Prediction

client = TestClient(main.app)
BODY = {"features": [0.0] * 30}


def cloud_ok(prob):
    async def fake(tx):
        return Prediction(fraud_probability=prob, route="cloud", latency_ms=1.0), 123

    return fake


async def cloud_down(tx):
    raise httpx.ConnectError("down")


def test_confident_score_stays_on_edge(monkeypatch):
    monkeypatch.setattr(main, "score", lambda f: 0.01)
    monkeypatch.setattr(main, "call_cloud", cloud_down)  # would set fell_back if called
    body = client.post("/predict", json=BODY).json()
    assert body["route"] == "edge"
    assert body["fell_back"] is False


def test_uncertain_score_goes_to_cloud(monkeypatch):
    monkeypatch.setattr(main, "score", lambda f: 0.5)
    monkeypatch.setattr(main, "call_cloud", cloud_ok(0.7))
    body = client.post("/predict", json=BODY).json()
    assert body["route"] == "cloud"
    assert body["fraud_probability"] == 0.7
    assert body["bytes_to_cloud"] == 123


def test_cloud_failure_falls_back_to_edge(monkeypatch):
    monkeypatch.setattr(main, "score", lambda f: 0.5)
    monkeypatch.setattr(main, "call_cloud", cloud_down)
    body = client.post("/predict", json=BODY).json()
    assert body["route"] == "edge"
    assert body["fell_back"] is True


def test_cloud_only_fails_during_outage(monkeypatch):
    monkeypatch.setattr(main, "call_cloud", cloud_down)
    r = client.post("/predict?mode=cloud_only", json=BODY)
    assert r.status_code == 503
