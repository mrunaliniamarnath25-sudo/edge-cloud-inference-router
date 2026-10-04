import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmark.csv")
ok = df[df.status == 200]

scen = ["cloud_only", "hybrid", "edge_only"]
names = ["Cloud only", "Hybrid router", "Edge only"]
pcts = [50, 95, 99]
stats = {s: np.percentile(ok[ok.scenario == s].client_ms, pcts) for s in scen}
n = int((df.scenario == "cloud_only").sum())

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8))

w = 0.25
x = np.arange(len(scen))
for j, p in enumerate(pcts):
    vals = [stats[s][j] for s in scen]
    bars = a1.bar(x + (j - 1) * w, vals, w, label=f"p{p}")
    a1.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
a1.set_yscale("log")
a1.set_xticks(x, names)
a1.set_ylabel("Latency (ms, log scale)")
a1.set_title("Request latency")
a1.legend()

kb = [df[df.scenario == s].bytes_to_cloud.sum() / 1024 for s in scen]
bars = a2.bar(names, kb)
a2.bar_label(bars, labels=[f"{v:.1f}" for v in kb], padding=2)
a2.set_title("Data sent to cloud (KB)")

out = ["cloud_only_outage", "hybrid_outage"]
succ = [(df[df.scenario == s].status == 200).mean() * 100 for s in out]
bars = a3.bar(["Cloud only", "Hybrid router"], succ)
a3.set_ylim(0, 115)
a3.bar_label(bars, labels=[f"{v:.0f}%" for v in succ], padding=2)
a3.set_title("Requests answered during cloud outage")

fig.suptitle(f"Edge-cloud inference router: {n} requests per scenario, simulated 500 ms cloud")
fig.tight_layout()
fig.savefig("results/latency_chart.png", dpi=200)

c, h = stats["cloud_only"], stats["hybrid"]
hy = ok[ok.scenario == "hybrid"]
print(f"p50 reduction: {100 * (1 - h[0] / c[0]):.1f}%  ({c[0]:.1f} -> {h[0]:.2f} ms)")
print(f"p95 reduction: {100 * (1 - h[1] / c[1]):.1f}%  ({c[1]:.1f} -> {h[1]:.2f} ms)")
print(f"cloud traffic reduction: {100 * (1 - kb[1] / kb[0]):.1f}%")
print(f"hybrid handled at edge: {100 * (hy.route == 'edge').mean():.1f}%")
pred, pos = hy.fraud_probability >= 0.5, hy.label == 1
tp = (pred & pos).sum()
print(f"hybrid recall {tp / max(pos.sum(), 1):.2f}, precision {tp / max(pred.sum(), 1):.2f} (fraud cases: {pos.sum()})")
print("saved results/latency_chart.png")
