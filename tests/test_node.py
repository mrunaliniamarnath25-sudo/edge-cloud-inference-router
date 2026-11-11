from fastapi.testclient import TestClient

from app import main


def test_node_id_header_is_set(monkeypatch):
    monkeypatch.setattr(main, "NODE_ID", "edge-test")
    r = TestClient(main.app).get("/health")
    assert r.headers["x-node-id"] == "edge-test"
