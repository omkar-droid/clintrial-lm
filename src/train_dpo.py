"""DPO preference tuning with TRL's DPOTrainer.

Runs on top of the *merged* SFT model (`make merge-sft` first). A fresh QLoRA
adapter is trained; with PEFT, TRL derives the frozen reference policy from the
same base by disabling the adapter, so no separate reference model is needed and
everything fits on one H100.

    python src/train_dpo.py --config configs/dpo.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from utils import (
    build_bnb_config,
    build_lora_config,
    count_trainable_parameters,
    load_config,
    load_tokenizer,
    resolve_attn_implementation,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    tokenizer = load_tokenizer(cfg.model.base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        quantization_config=build_bnb_config(cfg),
        dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(cfg.model.attn_implementation),
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    dataset = load_dataset(
        "json",
        data_files={"train": cfg.data.train_file, "validation": cfg.data.val_file},
    )

    t = cfg.training
    dpo_config = DPOConfig(
        output_dir=t.output_dir,
        beta=cfg.dpo.beta,
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.model.max_seq_length,
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
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=t.bf16,
        seed=t.seed,
        report_to=report_to,
        run_name=cfg.wandb.run_name,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,               # PEFT: reference = base with adapter disabled
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )
    print(count_trainable_parameters(trainer.model))

    trainer.train()
    trainer.save_model(t.output_dir)
    tokenizer.save_pretrained(t.output_dir)
    print(f"[train_dpo] adapter saved to {t.output_dir}")

    hub_id = getattr(cfg.output, "hub_model_id", None)
    if hub_id:
        trainer.push_to_hub(hub_id)
        print(f"[train_dpo] pushed to https://huggingface.co/{hub_id}")


if __name__ == "__main__":
    main()
