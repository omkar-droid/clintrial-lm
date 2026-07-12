"""Shared helpers: config loading, seeding, quantization and LoRA setup.

Keeping this logic in one place lets the train / eval / merge scripts read like a
recipe instead of a wall of boilerplate.
"""
from __future__ import annotations

import os
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _to_namespace(obj: Any) -> Any:
    """Recursively turn nested dicts into attribute-accessible namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str, overrides: dict | None = None) -> SimpleNamespace:
    """Load a YAML config into a namespace. `overrides` uses dotted keys, e.g.
    {"training.learning_rate": 1e-4} — handy for command-line sweeps."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    cfg = _to_namespace(raw)
    cfg._raw = raw  # keep the plain dict around for logging to W&B
    return cfg


def get(ns: SimpleNamespace, name: str, default: Any = None) -> Any:
    """Safe getattr for optional config fields."""
    return getattr(ns, name, default)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Model building blocks (imported lazily so the data pipeline stays torch-free)
# --------------------------------------------------------------------------- #
def dtype_from_str(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_bnb_config(cfg: SimpleNamespace):
    """BitsAndBytesConfig for QLoRA, or None when quantization is disabled."""
    q = getattr(cfg, "quantization", None)
    if q is None or not getattr(q, "load_in_4bit", False):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype_from_str(q.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
    )


def build_lora_config(cfg: SimpleNamespace):
    from peft import LoraConfig

    lora = cfg.lora
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=list(lora.target_modules),
    )


def resolve_attn_implementation(requested: str) -> str:
    """Use flash-attention-2 if it imports, otherwise fall back to SDPA so the
    code runs on machines without the flash-attn build."""
    if requested != "flash_attention_2":
        return requested
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        print("[utils] flash-attn not available -> falling back to attn_implementation='sdpa'")
        return "sdpa"


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        # Llama has no pad token; reuse EOS so batched training/generation works.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def count_trainable_parameters(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0
    return f"trainable params: {trainable:,} / {total:,} ({pct:.4f}%)"
