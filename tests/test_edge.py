from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_edge_predict_returns_probability():
    r = client.post("/edge/predict", json={"features": [0.0] * 30})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["route"] == "edge"


def test_edge_rejects_wrong_length():
    r = client.post("/edge/predict", json={"features": [0.0] * 5})
    assert r.status_code == 422
