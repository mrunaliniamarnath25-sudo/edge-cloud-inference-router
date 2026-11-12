import argparse
import csv
import subprocess
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

from bench.run_benchmark import load_rows, synthetic_rows

GATEWAY = "http://127.0.0.1:8000"
DIRECT = "http://127.0.0.1:8010"
COMPOSE = ["docker", "compose", "-f", "docker-compose.nodes.yml"]


def compose(*args, wait=True):
    p = subprocess.Popen(
        COMPOSE + list(args), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if wait:
        p.wait()
    return p


def run_scenario(client, base, name, rows, kill_at=None):
    out, stopper = [], None
    for i, (feats, label) in enumerate(rows):
        if kill_at is not None and i == kill_at:
            stopper = compose("stop", "edge2", wait=False)
        t = time.perf_counter()
        status, node, attempts = 0, "", ""
        try:
            r = client.post(
                f"{base}/predict", params={"mode": "hybrid"}, json={"features": feats}
            )
            status = r.status_code
            node = r.headers.get("x-node-id", "")
            attempts = r.headers.get("x-gateway-attempts", "")
        except httpx.HTTPError:
            pass
        out.append(
            {
                "scenario": name,
                "i": i,
                "label": label,
                "status": status,
                "client_ms": round((time.perf_counter() - t) * 1000, 3),
                "node": node,
                "attempts": attempts,
            }
        )
    if stopper:
        stopper.wait()
    return out


def summarize(rows):
    print(
        f"\n{'scenario':20} {'ok%':>6} {'retried%':>9} {'p50':>8} {'p95':>8} {'p99':>8}  nodes"
    )
    for name in dict.fromkeys(r["scenario"] for r in rows):
        r = [x for x in rows if x["scenario"] == name]
        ok = [x for x in r if x["status"] == 200]
        lat = [x["client_ms"] for x in ok] or [float("nan")]
        retried = 100 * sum(x["attempts"] not in ("", "1") for x in r) / len(r)
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        nodes = dict(sorted(Counter(x["node"] for x in ok).items()))
        print(
            f"{name:20} {100 * len(ok) / len(r):6.1f} {retried:9.1f} "
            f"{p50:8.2f} {p95:8.2f} {p99:8.2f}  {nodes}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark_sample.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    if Path(args.data).exists():
        rows = load_rows(args.data, args.n)
    else:
        print(f"NOTE: {args.data} not found, using synthetic rows (latency only).")
        rows = synthetic_rows(args.n)
    probe = rows[0][0]
    kill_at = len(rows) // 2

    with httpx.Client(timeout=15) as client:
        try:
            client.get(f"{GATEWAY}/health").raise_for_status()
            client.get(f"{DIRECT}/health").raise_for_status()
        except httpx.HTTPError:
            raise SystemExit(
                "start the stack first: docker compose -f docker-compose.nodes.yml up -d"
            )
        compose("start", "edge2")
        time.sleep(3)
        for _ in range(20):  # warm-up, not recorded
            client.post(f"{GATEWAY}/predict", params={"mode": "edge_only"}, json={"features": probe})

        results = []
        plan = [
            ("direct_edge1", DIRECT, None),
            ("gateway_3_nodes", GATEWAY, None),
            ("gateway_kill_node", GATEWAY, kill_at),
        ]
        for name, base, kill in plan:
            print(f"running {name} ({len(rows)} requests)...", flush=True)
            results += run_scenario(client, base, name, rows, kill)
        compose("start", "edge2")

    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark_nodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    summarize(results)

    after = [
        x for x in results
        if x["scenario"] == "gateway_kill_node" and x["i"] >= kill_at
        and x["status"] == 200 and x["node"] == "edge-2"
    ]
    print(f"\nedge-2 stopped at request {kill_at}; it still answered {len(after)} request(s) after that.")
    print("saved results/benchmark_nodes.csv (edge2 was started again)")


if __name__ == "__main__":
    main()
