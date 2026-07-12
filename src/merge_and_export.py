"""Merge a trained LoRA/QLoRA adapter into the base weights to produce a single
standalone model for fast inference, and print export recipes for GGUF / AWQ.

    python src/merge_and_export.py --config configs/qlora_sft.yaml   # -> merged/qlora-sft
    python src/merge_and_export.py --config configs/dpo.yaml         # -> merged/dpo

Note: merging always happens in fp16/bf16 (not 4-bit) — you can't merge into a
quantized base. QLoRA adapters merge back into the full-precision base cleanly.
"""
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from utils import load_config, load_tokenizer


GGUF_HELP = """
# Export to GGUF (for llama.cpp / Ollama / local CPU inference):
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
python convert_hf_to_gguf.py {merged} --outfile clintrial.gguf --outtype f16
./llama-quantize clintrial.gguf clintrial-q4_k_m.gguf Q4_K_M

# Export to AWQ (4-bit weight-only quant for fast GPU serving with vLLM):
pip install autoawq
python -c "from awq import AutoAWQForCausalLM; from transformers import AutoTokenizer; \\
m=AutoAWQForCausalLM.from_pretrained('{merged}'); t=AutoTokenizer.from_pretrained('{merged}'); \\
m.quantize(t, quant_config={{'w_bit':4,'q_group_size':128,'zero_point':True,'version':'GEMM'}}); \\
m.save_quantized('{merged}-awq'); t.save_pretrained('{merged}-awq')"
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter-dir", default=None, help="Override adapter path (default: training.output_dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    adapter_dir = args.adapter_dir or cfg.training.output_dir
    merged_dir = cfg.output.merged_dir
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[merge] base    = {cfg.model.base_model_id}")
    print(f"[merge] adapter = {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = load_tokenizer(cfg.model.base_model_id)
    tokenizer.save_pretrained(merged_dir)
    print(f"[merge] merged model -> {merged_dir}")

    hub_id = getattr(cfg.output, "hub_model_id", None)
    if hub_id:
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        print(f"[merge] pushed to https://huggingface.co/{hub_id}")

    print(GGUF_HELP.format(merged=merged_dir))


if __name__ == "__main__":
    main()
