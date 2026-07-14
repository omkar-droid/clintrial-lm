"""Benchmark inference cost for a model under a given quantization mode.

Measures the three numbers a deployment decision actually turns on:
  * **peak VRAM**       — can it fit on the GPU you can afford?
  * **TTFT**            — time to first token (prefill latency, what users feel)
  * **decode throughput** — tokens/sec once generation is flowing

Pair with `evaluate.py --quant ...` (same modes) to get the accuracy side, and the
two together answer the only question that matters: *how much quality do I lose for
how much speed/memory?*

    python src/benchmark.py --config configs/qlora_sft.yaml --quant bf16 --run-name bf16
    python src/benchmark.py --config configs/qlora_sft.yaml --quant nf4  --run-name nf4
    python src/benchmark.py --config configs/qlora_sft.yaml --model-path merged/qlora-sft-awq --quant awq --run-name awq
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import torch
from datasets import load_dataset

from utils import load_config, load_model_for_inference, load_tokenizer, set_seed


def dir_size_gb(path: str) -> float:
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True, check=True)
        return round(int(out.stdout.split()[0]) / 1e9, 2)
    except Exception:
        return -1.0


@torch.no_grad()
def _generate(model, enc, max_new_tokens, pad_id):
    return model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path", default=None, help="Defaults to output.merged_dir")
    ap.add_argument("--quant", choices=["bf16", "nf4", "int8", "awq", "gptq"], default="bf16")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    model_path = args.model_path or cfg.output.merged_dir
    os.makedirs(args.out_dir, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer(model_path if args.quant in ("awq", "gptq") else cfg.model.base_model_id)

    t_load = time.perf_counter()
    model = load_model_for_inference(
        model_path, args.quant, cfg.model.base_model_id, cfg.model.attn_implementation
    )
    model.eval()
    load_s = round(time.perf_counter() - t_load, 1)
    weights_vram = round(torch.cuda.memory_allocated() / 1e9, 2)

    # Real prompts from the held-out test set, so prefill lengths are realistic.
    test = load_dataset("json", data_files={"test": cfg.data.test_file})["test"]
    prompts = [row["messages"][:-1] for row in test.select(range(min(args.n_prompts, len(test))))]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def encode(msgs):
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return {k: v.to(model.device) for k, v in enc.items()}

    # Warm-up (kernel autotune / cache alloc) — never time the first call.
    _generate(model, encode(prompts[0]), 8, pad_id)
    torch.cuda.synchronize()

    ttfts, tok_per_s, gen_tokens_total, gen_time_total = [], [], 0, 0.0

    for msgs in prompts:
        enc = encode(msgs)
        n_in = enc["input_ids"].shape[1]

        # TTFT = prefill + 1 decoded token
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _generate(model, enc, 1, pad_id)
        torch.cuda.synchronize()
        ttfts.append((time.perf_counter() - t0) * 1000)

        # Full generation -> decode throughput
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = _generate(model, enc, args.max_new_tokens, pad_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        n_new = out.shape[1] - n_in
        if n_new > 0 and dt > 0:
            tok_per_s.append(n_new / dt)
            gen_tokens_total += n_new
            gen_time_total += dt

    peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    result = {
        "run_name": args.run_name,
        "quant": args.quant,
        "model_path": model_path,
        "disk_size_gb": dir_size_gb(model_path),
        "weights_vram_gb": weights_vram,
        "peak_vram_gb": peak_vram,
        "load_seconds": load_s,
        "ttft_ms_mean": round(sum(ttfts) / len(ttfts), 1),
        "decode_tok_per_s_mean": round(sum(tok_per_s) / len(tok_per_s), 1),
        "aggregate_tok_per_s": round(gen_tokens_total / gen_time_total, 1),
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
    }

    path = os.path.join(args.out_dir, f"bench_{args.run_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[bench] -> {path}")


if __name__ == "__main__":
    main()
