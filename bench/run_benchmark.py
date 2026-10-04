import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROUTER = "http://127.0.0.1:8000"
SCENARIOS = [
    ("cloud_only", "cloud_only", False),
    ("hybrid", "hybrid", False),
    ("edge_only", "edge_only", False),
    ("cloud_only_outage", "cloud_only", True),
    ("hybrid_outage", "hybrid", True),
]


def load_rows(path, n):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c != "Class"]
        rows = [([float(r[c]) for c in cols], int(float(r["Class"]))) for r in reader]
    return rows[:n]


def wait_ready(client, probe):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if client.get(f"{ROUTER}/health").status_code == 200:
                r = client.post(
                    f"{ROUTER}/predict", params={"mode": "cloud_only"}, json={"features": probe}
                )
                if r.status_code == 200:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit("servers did not start in time")


def run_scenario(client, name, mode, outage, rows):
    client.post(f"{ROUTER}/admin/outage", params={"down": str(outage).lower()})
    out = []
    for i, (feats, label) in enumerate(rows):
        t = time.perf_counter()
        r = client.post(f"{ROUTER}/predict", params={"mode": mode}, json={"features": feats})
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
    client.post(f"{ROUTER}/admin/outage", params={"down": "false"})
    return out


def summarize(rows):
    print(f"\n{'scenario':20} {'ok%':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'edge%':>7} {'KB->cloud':>10}")
    for name, _, _ in SCENARIOS:
        r = [x for x in rows if x["scenario"] == name]
        ok = [x for x in r if x["status"] == 200]
        lat = [x["client_ms"] for x in ok] or [float("nan")]
        edge = 100 * sum(x["route"] == "edge" for x in ok) / max(len(ok), 1)
        kb = sum(x["bytes_to_cloud"] for x in r) / 1024
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        print(f"{name:20} {100*len(ok)/len(r):6.1f} {p50:8.2f} {p95:8.2f} {p99:8.2f} {edge:7.1f} {kb:10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark_sample.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    rows = load_rows(args.data, args.n)
    probe = rows[0][0]
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app, "--port", port],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for app, port in [("app.cloud:app", "8001"), ("app.main:app", "8000")]
    ]
    try:
        with httpx.Client(timeout=10) as client:
            wait_ready(client, probe)
            for _ in range(20):  # warm-up, not recorded
                client.post(f"{ROUTER}/predict", params={"mode": "edge_only"}, json={"features": probe})
            results = []
            for name, mode, outage in SCENARIOS:
                print(f"running {name} ({len(rows)} requests)...", flush=True)
                results += run_scenario(client, name, mode, outage, rows)
    finally:
        for p in procs:
            p.terminate()

    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    summarize(results)
    print("\nsaved results/benchmark.csv")


if __name__ == "__main__":
    main()
