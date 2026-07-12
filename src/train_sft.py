"""Supervised fine-tuning with QLoRA / LoRA via TRL's SFTTrainer.

    python src/train_sft.py --config configs/qlora_sft.yaml
    python src/train_sft.py --config configs/lora_sft.yaml --set training.learning_rate=1e-4

The same script drives both the 4-bit QLoRA run and the bf16 LoRA run — the only
difference lives in the YAML, which keeps the two directly comparable.
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from utils import (
    build_bnb_config,
    build_lora_config,
    count_trainable_parameters,
    dtype_from_str,
    load_config,
    load_tokenizer,
    resolve_attn_implementation,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="Dotted overrides, e.g. training.learning_rate=1e-4")
    return ap.parse_args()


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        key, _, val = p.partition("=")
        try:
            val = eval(val, {}, {})  # allow ints/floats/lists in overrides
        except Exception:
            pass
        out[key] = val
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg.training.seed)

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    # --- tokenizer + model ------------------------------------------------- #
    tokenizer = load_tokenizer(cfg.model.base_model_id)
    bnb_config = build_bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(cfg.model.attn_implementation),
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # --- data -------------------------------------------------------------- #
    dataset = load_dataset(
        "json",
        data_files={"train": cfg.data.train_file, "validation": cfg.data.val_file},
    )

    # --- trainer ----------------------------------------------------------- #
    t = cfg.training
    sft_config = SFTConfig(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        max_length=cfg.model.max_seq_length,
        packing=t.packing,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=t.bf16,
        seed=t.seed,
        report_to=report_to,
        run_name=cfg.wandb.run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )
    print(count_trainable_parameters(trainer.model))

    trainer.train()

    # Save the adapter + tokenizer.
    trainer.save_model(t.output_dir)
    tokenizer.save_pretrained(t.output_dir)
    print(f"[train_sft] adapter saved to {t.output_dir}")

    hub_id = getattr(cfg.output, "hub_model_id", None)
    if hub_id:
        trainer.push_to_hub(hub_id)
        print(f"[train_sft] pushed to https://huggingface.co/{hub_id}")


if __name__ == "__main__":
    main()
