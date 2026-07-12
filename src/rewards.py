"""Verifiable reward functions for GRPO (RL with Verifiable Rewards / RLVR).

Every clinical-trial task has a programmatic "is this answer good?" signal, so the
model can be rewarded WITHOUT a learned reward model or human labels — the whole
appeal of RLVR (the technique behind DeepSeek-R1), and the reason this domain is a
great fit for it:

  * extract_eligibility    -> 0.5 * valid-JSON  +  0.5 * criterion-level F1 vs gold
  * phase_classification   -> exact match {0,1}
  * condition_qa           -> token F1 vs gold
  * plain_language_summary -> ROUGE-L vs reference

Reward is in [0, 1]. Consumed by src/train_grpo.py via `make_reward_func()`.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

_WORD = re.compile(r"[a-z0-9]+")
_rouge = None  # lazy — rouge_score import is slow


def _norm(s) -> str:
    return " ".join(_WORD.findall(str(s).lower()))


def token_f1(pred: str, ref: str) -> float:
    p, r = _norm(pred).split(), _norm(ref).split()
    if not p or not r:
        return float(p == r)
    ref_counts = defaultdict(int)
    for w in r:
        ref_counts[w] += 1
    overlap, seen = 0, defaultdict(int)
    for w in p:
        seen[w] += 1
        if seen[w] <= ref_counts[w]:
            overlap += 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def extract_json_block(text: str):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _flatten(d) -> set:
    items = set()
    for key in ("inclusion", "exclusion"):
        for it in d.get(key, []) or []:
            items.add((key, _norm(str(it))))
    return items


def eligibility_reward(pred: str, gold: str) -> float:
    """0.5 for producing valid JSON, +0.5 * criterion F1 for being correct."""
    pj = extract_json_block(pred)
    if pj is None:
        return 0.0                       # invalid JSON -> no reward (fixes the flat metric)
    reward = 0.5
    gj = extract_json_block(gold)
    if gj is None:
        return reward
    ps, gs = _flatten(pj), _flatten(gj)
    if not gs:
        return reward
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return reward + 0.5 * f1


def rouge_l(pred: str, ref: str) -> float:
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def score(task: str, pred: str, gold: str) -> float:
    if task == "extract_eligibility":
        return eligibility_reward(pred, gold)
    if task == "phase_classification":
        return float(_norm(pred) == _norm(gold))
    if task == "condition_qa":
        return token_f1(pred, gold)
    if task == "plain_language_summary":
        return rouge_l(pred, gold)
    return 0.0


def _completion_text(c) -> str:
    # GRPO passes conversational completions as [{"role": "assistant", "content": ...}]
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return c if isinstance(c, str) else str(c)


def make_reward_func():
    """Return a reward function with the TRL GRPO signature.

    TRL calls it as reward_func(prompts, completions, **columns) where each extra
    dataset column (here `task`, `gold`) arrives as a per-sample list.
    """
    def clinical_reward(prompts, completions, task=None, gold=None, **kwargs):
        out = []
        for i, c in enumerate(completions):
            t = task[i] if task is not None else "extract_eligibility"
            g = gold[i] if gold is not None else ""
            out.append(float(score(t, _completion_text(c), g)))
        return out

    return clinical_reward
