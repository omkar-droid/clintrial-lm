"""Serve the merged model with vLLM's OpenAI-compatible API server.

    python src/serve_vllm.py --config configs/dpo.yaml --port 8000

Then query it like the OpenAI API:

    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "clintrial-lm",
      "messages": [
        {"role": "system", "content": "You are a clinical research assistant."},
        {"role": "user", "content": "Summarise NCT... in plain language."}
      ]
    }'

vLLM gives paged-attention + continuous batching, so this is the config you would
actually put behind an API. To A/B test adapters without merging, serve the base
model with `--enable-lora` and pass `--lora-modules name=path` instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-name", default="clintrial-lm")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = cfg.output.merged_dir

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", args.served_name,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.90",
    ]
    print("[serve] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
