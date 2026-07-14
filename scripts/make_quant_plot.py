"""Scatter plot of the quantization trade-off: VRAM vs accuracy, sized by speed."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)
QUANT_DIR = "results/quant"

bench, evals = {}, {}
for p in glob.glob(f"{QUANT_DIR}/bench_*.json"):
    d = json.load(open(p)); bench[d["run_name"]] = d
for p in glob.glob(f"{QUANT_DIR}/metrics_*.json"):
    d = json.load(open(p)); evals[d["run_name"]] = d

LABEL = {"bf16": "bf16", "int8": "INT8", "nf4": "NF4", "awq": "AWQ"}
COLOR = {"bf16": "#3b82f6", "int8": "#f59e0b", "nf4": "#10b981", "awq": "#a855f7"}

fig, ax = plt.subplots(figsize=(7.5, 5))
for run in bench:
    if run not in evals:
        continue
    vram = bench[run]["peak_vram_gb"]
    f1 = evals[run]["metrics"]["extract_eligibility"]["f1"]
    speed = bench[run]["decode_tok_per_s_mean"]
    ax.scatter(vram, f1, s=speed * 18, alpha=0.55, color=COLOR.get(run, "#888"),
               edgecolors="black", linewidths=1, zorder=3)
    ax.annotate(f"{LABEL.get(run, run)}\n{speed:.0f} tok/s",
                (vram, f1), textcoords="offset points", xytext=(10, 8), fontsize=9)

ax.set_xlabel("Peak VRAM (GB)  →  smaller is cheaper")
ax.set_ylabel("Eligibility F1  →  higher is better")
ax.set_title("Quantization trade-off (bubble size = decode tok/s)\nClinTrial-LM · Qwen2.5-7B · H100 · batch 1")
ax.grid(alpha=0.3, zorder=0)
ax.margins(0.15)
fig.tight_layout()
fig.savefig("assets/quant_tradeoff.png", dpi=150)
print("wrote assets/quant_tradeoff.png")
