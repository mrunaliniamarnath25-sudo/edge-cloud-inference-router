import csv

import numpy as np

from app.model import cloud_score, score


def load(path="data/benchmark_sample.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c != "Class"]
        rows = list(reader)
    x = [[float(r[c]) for c in cols] for r in rows]
    y = np.array([int(float(r["Class"])) for r in rows])
    return x, y


def average_precision(y, s):
    order = np.argsort(-s)
    hits = y[order]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at_k * hits).sum() / hits.sum())


def report(name, y, p, cloud_pct):
    pred = p >= 0.5
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    ap = average_precision(y, p)
    print(f"{name:24} {cloud_pct:6.1f} {prec:7.3f} {rec:7.3f} {ap:6.3f} {fp:4d} {fn:4d}")


def main():
    x, y = load()
    edge = np.array([score(r) for r in x])
    cloud = np.array([cloud_score(r) for r in x])
    print(f"{len(y)} rows, {int(y.sum())} fraud ({100 * y.mean():.1f}%)\n")
    print(f"{'setup':24} {'cloud%':>6} {'prec':>7} {'recall':>7} {'AP':>6} {'FP':>4} {'FN':>4}")
    report("edge only", y, edge, 0.0)
    report("cloud only", y, cloud, 100.0)
    for lo, hi in [(0.02, 0.98), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]:
        unsure = (edge > lo) & (edge < hi)
        hybrid = np.where(unsure, cloud, edge)
        report(f"hybrid {lo}-{hi}", y, hybrid, 100 * unsure.mean())


if __name__ == "__main__":
    main()
