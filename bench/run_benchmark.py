import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROUTER = "http://127.0.0.1:8000"
CLOUD = "http://127.0.0.1:8001"
STRESS = "http://127.0.0.1:8002"

# (name, mode, cloud_down, router, unstable_cloud)
BASE = [
    ("cloud_only", "cloud_only", False, ROUTER, False),
    ("hybrid", "hybrid", False, ROUTER, False),
    ("edge_only", "edge_only", False, ROUTER, False),
    ("cloud_only_outage", "cloud_only", True, ROUTER, False),
    ("hybrid_outage", "hybrid", True, ROUTER, False),
]
CHAOS = [
    ("cloud_only", "cloud_only", False, ROUTER, False),
    ("cloud_only_unstable", "cloud_only", False, ROUTER, True),
    ("hybrid", "hybrid", False, ROUTER, False),
    ("hybrid_unstable", "hybrid", False, ROUTER, True),
    ("hybrid_stress", "hybrid", False, STRESS, False),
    ("hybrid_unstable_stress", "hybrid", False, STRESS, True),
]


def load_rows(path, n):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c != "Class"]
        rows = [([float(r[c]) for c in cols], int(float(r["Class"]))) for r in reader]
    return rows[:n]


def synthetic_rows(n):
    rng = np.random.default_rng(0)
    return [(rng.normal(size=30).tolist(), 0) for _ in range(n)]


def start(app_name, port, **env):
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app_name, "--port", port],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_ready(client, base, probe):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if client.get(f"{base}/health").status_code == 200:
                r = client.post(
                    f"{base}/predict", params={"mode": "cloud_only"}, json={"features": probe}
                )
                if r.status_code == 200:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"{base} did not start in time")


def set_chaos(client, jitter_ms, drop_pct):
    client.post(f"{CLOUD}/admin/chaos", params={"jitter_ms": jitter_ms, "drop_pct": drop_pct})


def run_scenario(client, name, mode, outage, base, chaos, rows):
    set_chaos(client, *chaos)
    client.post(f"{base}/admin/outage", params={"down": str(outage).lower()})
    out = []
    for i, (feats, label) in enumerate(rows):
        t = time.perf_counter()
        r = client.post(f"{base}/predict", params={"mode": mode}, json={"features": feats})
        ms = (time.perf_counter() - t) * 1000
        b = r.json() if r.status_code == 200 else {}
        out.append(
            {
                "scenario": name,
                "i": i,
                "label": label,
                "status": r.status_code,
                "client_ms": round(ms, 3),
                "route": b.get("route", ""),
                "fell_back": b.get("fell_back", ""),
                "bytes_to_cloud": b.get("bytes_to_cloud", 0),
                "fraud_probability": b.get("fraud_probability", ""),
            }
        )
    client.post(f"{base}/admin/outage", params={"down": "false"})
    set_chaos(client, 0.0, 0.0)
    return out


def summarize(rows, names):
    print(
        f"\n{'scenario':24} {'ok%':>6} {'fb%':>6} {'p50':>8} {'p95':>8} {'p99':>8} "
        f"{'edge%':>7} {'KB->cloud':>10}"
    )
    for name in names:
        r = [x for x in rows if x["scenario"] == name]
        ok = [x for x in r if x["status"] == 200]
        lat = [x["client_ms"] for x in ok] or [float("nan")]
        edge = 100 * sum(x["route"] == "edge" for x in ok) / max(len(ok), 1)
        fb = 100 * sum(x["fell_back"] is True for x in r) / max(len(r), 1)
        kb = sum(x["bytes_to_cloud"] for x in r) / 1024
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        print(
            f"{name:24} {100 * len(ok) / len(r):6.1f} {fb:6.1f} {p50:8.2f} {p95:8.2f} "
            f"{p99:8.2f} {edge:7.1f} {kb:10.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark_sample.csv")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--synthetic", action="store_true", help="use random rows instead of the CSV")
    ap.add_argument("--chaos", action="store_true", help="run the unstable-cloud scenarios")
    ap.add_argument("--jitter-ms", type=float, default=150.0, help="mean extra delay")
    ap.add_argument("--drop-pct", type=float, default=5.0, help="share of requests that hang")
    ap.add_argument("--stress-pct", type=float, default=70.0, help="edge-score percentile for the stress band")
    args = ap.parse_args()

    synthetic = args.synthetic or not Path(args.data).exists()
    if synthetic:
        why = "--synthetic" if args.synthetic else f"{args.data} not found"
        print(
            f"NOTE: using {args.n} synthetic rows ({why}). Latency is valid; "
            "the routing mix and accuracy are not representative."
        )
        rows = synthetic_rows(args.n)
    else:
        rows = load_rows(args.data, args.n)

    scenarios = CHAOS if args.chaos else BASE
    parts = ["benchmark"] + (["unstable"] if args.chaos else []) + (["synthetic"] if synthetic else [])
    out_path = f"results/{'_'.join(parts)}.csv"
    probe = rows[0][0]

    procs = [
        start("app.cloud:app", "8001", CLOUD_SEED="1"),
        start("app.main:app", "8000"),
    ]
    try:
        with httpx.Client(timeout=15) as client:
            wait_ready(client, ROUTER, probe)
            if args.chaos:
                scores = [
                    client.post(f"{ROUTER}/edge/predict", json={"features": f}).json()["fraud_probability"]
                    for f, _ in rows
                ]
                low = min(float(np.percentile(scores, args.stress_pct)), 0.5)
                share = 100 * float(np.mean([(low < s < 0.9) for s in scores]))
                print(
                    f"stress band: {low:.6g} to 0.9, about {share:.0f}% of rows fall inside it; "
                    f"unstable cloud: jitter mean {args.jitter_ms:g} ms, {args.drop_pct:g}% of requests hang"
                )
                procs.append(start("app.main:app", "8002", ROUTE_LOW=str(low), ROUTE_HIGH="0.9"))
                wait_ready(client, STRESS, probe)
            for _ in range(20):  # warm-up, not recorded
                client.post(f"{ROUTER}/predict", params={"mode": "edge_only"}, json={"features": probe})
            results = []
            for name, mode, outage, base, unstable in scenarios:
                chaos = (args.jitter_ms, args.drop_pct) if unstable else (0.0, 0.0)
                print(f"running {name} ({len(rows)} requests)...", flush=True)
                results += run_scenario(client, name, mode, outage, base, chaos, rows)
    finally:
        for p in procs:
            p.terminate()

    Path("results").mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    summarize(results, [s[0] for s in scenarios])
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
