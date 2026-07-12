"""Generate the SFT loss curve and GRPO reward curve PNGs for the README."""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

# ---------- SFT loss curve (from trainer_state.json) ----------
states = glob.glob("outputs/qlora-sft/checkpoint-*/trainer_state.json")
state = max(states, key=lambda p: len(json.load(open(p)).get("log_history", [])))
hist = json.load(open(state))["log_history"]
tr = [(e["step"], e["loss"]) for e in hist if "loss" in e]
ev = [(e["step"], e["eval_loss"]) for e in hist if "eval_loss" in e]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([s for s, _ in tr], [l for _, l in tr], lw=1.2, label="train loss", color="#3b82f6")
if ev:
    ax.plot([s for s, _ in ev], [l for _, l in ev], lw=2, marker="o", ms=4,
            label="eval loss", color="#ef4444")
ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.set_title("SFT (QLoRA) — Qwen2.5-7B on clinical-trial instructions")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("assets/sft_loss_curve.png", dpi=150)
print("wrote assets/sft_loss_curve.png  (train pts:", len(tr), "eval pts:", len(ev), ")")

# ---------- GRPO reward curve (parsed from grpo.log) ----------
rew, steps = [], []
step = 0
for line in open("grpo.log", errors="ignore"):
    m = re.search(r"'reward': '?([0-9.]+)", line)
    if m:
        step += 5  # logging_steps
        steps.append(step); rew.append(float(m.group(1)))

if rew:
    # simple moving average to show the trend through the noise
    k = 5
    ma = [sum(rew[max(0, i - k + 1):i + 1]) / len(rew[max(0, i - k + 1):i + 1]) for i in range(len(rew))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, rew, lw=1, alpha=0.35, color="#10b981", label="reward (per log step)")
    ax.plot(steps, ma, lw=2, color="#059669", label=f"moving avg ({k})")
    ax.set_xlabel("step"); ax.set_ylabel("mean verifiable reward")
    ax.set_title("GRPO / RLVR — mean reward over training")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("assets/grpo_reward_curve.png", dpi=150)
    print("wrote assets/grpo_reward_curve.png  (points:", len(rew), ")")
