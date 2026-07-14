"""Produce AWQ / GPTQ 4-bit checkpoints from the merged fine-tuned model.

Unlike bitsandbytes (which quantizes at load time), AWQ and GPTQ are *calibrated*
post-training quantization: they run real data through the model to decide which
weights matter, so they typically retain more accuracy at 4-bit AND run faster,
because the kernels are built for inference rather than for training.

Calibration uses prompts from our own training split — quantizing on in-domain data
matters: calibrating a clinical model on generic web text loses more accuracy.

    python src/quantize.py --config configs/qlora_sft.yaml --method awq
    python src/quantize.py --config configs/qlora_sft.yaml --method gptq
"""
from __future__ import annotations

import argparse
import json

from utils import load_config, load_tokenizer


def calibration_texts(cfg, tokenizer, n: int = 128) -> list[str]:
    """In-domain calibration set: real prompts+answers from the training split."""
    texts = []
    with open(cfg.data.train_file) as f:
        for line in f:
            row = json.loads(line)
            texts.append(
                tokenizer.apply_chat_template(row["messages"], tokenize=False)
            )
            if len(texts) >= n:
                break
    return texts


def quantize_awq(cfg, out_dir: str, n_calib: int) -> None:
    from awq import AutoAWQForCausalLM

    tokenizer = load_tokenizer(cfg.model.base_model_id)
    model = AutoAWQForCausalLM.from_pretrained(cfg.output.merged_dir, device_map="cuda")
    quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}

    print(f"[awq] calibrating on {n_calib} in-domain examples ...")
    model.quantize(tokenizer, quant_config=quant_config,
                   calib_data=calibration_texts(cfg, tokenizer, n_calib))

    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[awq] saved -> {out_dir}")


def quantize_gptq(cfg, out_dir: str, n_calib: int) -> None:
    from gptqmodel import GPTQModel, QuantizeConfig

    tokenizer = load_tokenizer(cfg.model.base_model_id)
    qcfg = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(cfg.output.merged_dir, qcfg)

    print(f"[gptq] calibrating on {n_calib} in-domain examples ...")
    model.quantize(calibration_texts(cfg, tokenizer, n_calib), batch_size=1)

    model.save(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[gptq] saved -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["awq", "gptq"], required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-calib", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or f"{cfg.output.merged_dir}-{args.method}"

    if args.method == "awq":
        quantize_awq(cfg, out_dir, args.n_calib)
    else:
        quantize_gptq(cfg, out_dir, args.n_calib)

    print("QUANTIZE_DONE")


if __name__ == "__main__":
    main()
