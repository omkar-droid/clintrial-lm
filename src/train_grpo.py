"""GRPO (RL with Verifiable Rewards) fine-tuning on the clinical-trial tasks.

Starts from the merged SFT policy and improves it against a programmatic reward
(src/rewards.py) — no reward model, no human labels. For each prompt GRPO samples a
GROUP of completions, scores them, and pushes the policy toward the above-average
ones (group-relative advantage), which is why it needs no value network.

    python src/train_grpo.py --config configs/grpo.yaml
    python src/train_grpo.py --config configs/grpo.yaml --set training.max_steps=50
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from rewards import make_reward_func
from utils import (
    build_bnb_config,
    build_lora_config,
    count_trainable_parameters,
    load_config,
    load_tokenizer,
    resolve_attn_implementation,
    set_seed,
)


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        key, _, val = p.partition("=")
        try:
            val = eval(val, {}, {})
        except Exception:
            pass
        out[key] = val
    return out


def to_grpo_row(ex: dict) -> dict:
    """SFT chat example -> GRPO row: prompt (messages up to user) + gold + task."""
    msgs = ex["messages"]
    return {"prompt": msgs[:-1], "task": ex["task"], "gold": msgs[-1]["content"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
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

    ds = load_dataset("json", data_files={"train": cfg.data.train_file})["train"]
    ds = ds.map(to_grpo_row, remove_columns=ds.column_names)
    max_prompts = getattr(cfg.data, "max_prompts", None)
    if max_prompts:
        ds = ds.shuffle(seed=cfg.training.seed).select(range(min(max_prompts, len(ds))))

    t = cfg.training
    grpo_config = GRPOConfig(
        output_dir=t.output_dir,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        max_grad_norm=t.max_grad_norm,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_generations=cfg.grpo.num_generations,
        max_completion_length=cfg.grpo.max_completion_length,
        temperature=cfg.grpo.temperature,
        beta=cfg.grpo.beta,
        log_completions=True,
        num_completions_to_print=3,
        max_steps=t.max_steps,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=t.bf16,
        seed=t.seed,
        report_to=report_to,
        run_name=cfg.wandb.run_name,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_func(),
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )
    print(count_trainable_parameters(trainer.model))

    trainer.train()
    trainer.save_model(t.output_dir)
    tokenizer.save_pretrained(t.output_dir)
    print(f"[train_grpo] adapter saved to {t.output_dir}")

    hub_id = getattr(cfg.output, "hub_model_id", None)
    if hub_id:
        trainer.push_to_hub(hub_id)


if __name__ == "__main__":
    main()
