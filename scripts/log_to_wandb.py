"""Replay a finished run's trainer_state.json into a Weights & Biases run.

Useful when a training run completed without live W&B logging — this reconstructs
the loss / eval-loss curves as a dashboard after the fact.

    python scripts/log_to_wandb.py --output-dir outputs/qlora-sft --run-name qlora-sft
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import wandb


def find_state(output_dir: str) -> str:
    cands = glob.glob(os.path.join(output_dir, "checkpoint-*/trainer_state.json"))
    cands += glob.glob(os.path.join(output_dir, "trainer_state.json"))
    if not cands:
        raise FileNotFoundError(f"no trainer_state.json found under {output_dir}")
    # The checkpoint with the longest log history holds the fullest curve.
    def hist_len(p):
        try:
            return len(json.load(open(p)).get("log_history", []))
        except Exception:
            return 0
    return max(cands, key=hist_len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--state", default=None, help="Explicit path to a trainer_state.json")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "clintrial-lm"))
    args = ap.parse_args()

    state_path = args.state or find_state(args.output_dir)
    state = json.load(open(state_path))
    history = state.get("log_history", [])

    run = wandb.init(project=args.project, name=args.run_name, job_type="train-replay")
    for entry in history:
        step = entry.get("step")
        metrics = {k: v for k, v in entry.items() if isinstance(v, (int, float)) and k != "step"}
        if metrics:
            wandb.log(metrics, step=step)
    run.summary["source_state"] = state_path
    run.summary["best_metric"] = state.get("best_metric")
    run.finish()
    print(f"[wandb] replayed {len(history)} log entries from {state_path}")


if __name__ == "__main__":
    main()
