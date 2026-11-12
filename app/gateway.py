import itertools
import os
import time

import httpx
from fastapi import FastAPI, Request, Response

NODES = [
    n.strip().rstrip("/")
    for n in os.getenv("NODES", "http://127.0.0.1:8000").split(",")
    if n.strip()
]
COOLDOWN_S = float(os.getenv("NODE_COOLDOWN_S", "5"))
TIMEOUT_S = float(os.getenv("NODE_TIMEOUT_S", "3"))

app = FastAPI(title="Edge Gateway")
client = httpx.AsyncClient(timeout=TIMEOUT_S)
down_until = {n: 0.0 for n in NODES}
_counter = itertools.count()


def candidates():
    """Live nodes in round-robin order; if every node is cooling down, try them all."""
    now = time.monotonic()
    live = [n for n in NODES if down_until[n] <= now]
    pool = live or list(NODES)
    start = next(_counter) % len(pool)
    return pool[start:] + pool[:start]


def is_node_failure(status_code: int) -> bool:
    # 503 is the router's own answer ("cloud unavailable" in cloud_only mode),
    # not a sign that the node is down.
    return status_code >= 500 and status_code != 503


@app.get("/health")
def health():
    return {"status": "ok", "nodes": len(NODES)}


@app.get("/nodes")
def nodes():
    now = time.monotonic()
    return {n: ("up" if down_until[n] <= now else "cooling down") for n in NODES}


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    params = dict(request.query_params)
    attempts = 0
    for node in candidates():
        attempts += 1
        try:
            r = await client.post(
                f"{node}/predict",
                params=params,
                content=body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError:
            down_until[node] = time.monotonic() + COOLDOWN_S
            continue
        if is_node_failure(r.status_code):
            down_until[node] = time.monotonic() + COOLDOWN_S
            continue
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type="application/json",
            headers={
                "X-Node-Id": r.headers.get("x-node-id", "unknown"),
                "X-Gateway-Attempts": str(attempts),
            },
        )
    return Response(
        content=b'{"detail":"no edge node available"}',
        status_code=503,
        media_type="application/json",
        headers={"X-Gateway-Attempts": str(attempts)},
    )
