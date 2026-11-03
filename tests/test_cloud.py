import pytest
from fastapi.testclient import TestClient

from app import cloud, main

cloud_client = TestClient(cloud.app)
router_client = TestClient(main.app)
BODY = {"features": [0.0] * 30}


@pytest.fixture(autouse=True)
def reset_outage():
    main.state["cloud_down"] = False
    yield
    main.state["cloud_down"] = False


def test_cloud_returns_probability_and_route():
    r = cloud_client.post("/cloud/predict", json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["route"] == "cloud"


def test_cloud_rejects_wrong_feature_count():
    r = cloud_client.post("/cloud/predict", json={"features": [0.0] * 5})
    assert r.status_code == 422


def test_cloud_latency_reflects_the_configured_delay(monkeypatch):
    monkeypatch.setattr(cloud, "DELAY_MS", 50.0)
    r = cloud_client.post("/cloud/predict", json=BODY)
    # a small margin below 50 ms, because async timers can fire slightly early
    assert r.json()["latency_ms"] >= 45.0


def test_outage_switch_makes_cloud_only_return_503():
    r = router_client.post("/admin/outage", params={"down": "true"})
    assert r.json()["cloud_down"] is True
    r = router_client.post("/predict", params={"mode": "cloud_only"}, json=BODY)
    assert r.status_code == 503
