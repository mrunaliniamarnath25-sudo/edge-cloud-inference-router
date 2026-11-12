import itertools
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import gateway

NODES = ["http://n1", "http://n2", "http://n3"]
BODY = {"features": [0.0] * 30}


class FakeClient:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(url)
        return self.behavior(url)


def ok(url):
    node = url.split("//")[1].split("/")[0]
    return httpx.Response(200, json={"route": "edge"}, headers={"x-node-id": node})


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setattr(gateway, "NODES", list(NODES))
    monkeypatch.setattr(gateway, "down_until", {n: 0.0 for n in NODES})
    monkeypatch.setattr(gateway, "_counter", itertools.count())
    return TestClient(gateway.app)


def test_round_robin_cycles_through_nodes(gw, monkeypatch):
    monkeypatch.setattr(gateway, "client", FakeClient(ok))
    seen = [gw.post("/predict", json=BODY).headers["x-node-id"] for _ in range(6)]
    assert seen == ["n1", "n2", "n3", "n1", "n2", "n3"]


def test_failover_skips_a_dead_node(gw, monkeypatch):
    def behavior(url):
        if url.startswith("http://n1"):
            raise httpx.ConnectError("down")
        return ok(url)

    fake = FakeClient(behavior)
    monkeypatch.setattr(gateway, "client", fake)
    r = gw.post("/predict", json=BODY)
    assert r.status_code == 200
    assert r.headers["x-node-id"] == "n2"
    assert r.headers["x-gateway-attempts"] == "2"
    for _ in range(4):
        assert gw.post("/predict", json=BODY).status_code == 200
    assert sum(u.startswith("http://n1") for u in fake.calls) == 1


def test_503_from_a_node_is_not_treated_as_node_failure(gw, monkeypatch):
    fake = FakeClient(
        lambda url: httpx.Response(
            503, json={"detail": "cloud unavailable"}, headers={"x-node-id": "n1"}
        )
    )
    monkeypatch.setattr(gateway, "client", fake)
    r = gw.post("/predict?mode=cloud_only", json=BODY)
    assert r.status_code == 503
    assert len(fake.calls) == 1
    assert gateway.down_until["http://n1"] == 0.0


def test_all_nodes_down_returns_503(gw, monkeypatch):
    def behavior(url):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(gateway, "client", FakeClient(behavior))
    r = gw.post("/predict", json=BODY)
    assert r.status_code == 503
    assert r.headers["x-gateway-attempts"] == "3"


def test_load_stays_balanced_when_a_node_is_down(gw, monkeypatch):
    monkeypatch.setattr(gateway, "client", FakeClient(ok))
    gateway.down_until["http://n1"] = time.monotonic() + 60
    seen = [gw.post("/predict", json=BODY).headers["x-node-id"] for _ in range(10)]
    assert seen.count("n2") == 5
    assert seen.count("n3") == 5
