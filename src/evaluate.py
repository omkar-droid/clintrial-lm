"""Evaluate a model on the held-out clinical-trial test set.

Reports task-appropriate metrics so improvements are measurable, not vibes:
  * extract_eligibility  -> JSON validity + criterion-level precision/recall/F1
  * plain_language_summary / condition_qa -> ROUGE-1 / ROUGE-L
  * phase_classification  -> exact-match accuracy
Optionally adds an LLM-as-judge faithfulness score (needs ANTHROPIC_API_KEY).

    python src/evaluate.py --config configs/qlora_sft.yaml --which base    --run-name base
    python src/evaluate.py --config configs/qlora_sft.yaml --which adapter --run-name sft
    python src/evaluate.py --config configs/dpo.yaml       --which adapter --run-name dpo --judge
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM

from utils import load_config, load_tokenizer, resolve_attn_implementation, set_seed

_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def token_f1(pred: str, ref: str) -> float:
    p, r = _norm(pred).split(), _norm(ref).split()
    if not p or not r:
        return float(p == r)
    common = defaultdict(int)
    ref_counts = defaultdict(int)
    for w in r:
        ref_counts[w] += 1
    overlap = 0
    seen = defaultdict(int)
    for w in p:
        seen[w] += 1
        if seen[w] <= ref_counts[w]:
            overlap += 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_json_block(text: str) -> dict | None:
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


def criteria_prf(pred: str, ref: str) -> tuple[float, float, float, int]:
    """Criterion-level precision/recall/F1 for the extraction task, plus json_valid."""
    pj = extract_json_block(pred)
    rj = extract_json_block(ref)
    if pj is None:
        return 0.0, 0.0, 0.0, 0
    if rj is None:
        return 0.0, 0.0, 0.0, 1

    def flatten(d):
        items = []
        for key in ("inclusion", "exclusion"):
            for it in d.get(key, []) or []:
                items.append((key, _norm(str(it))))
        return set(items)

    pred_set, ref_set = flatten(pj), flatten(rj)
    if not ref_set:
        return 0.0, 0.0, 0.0, 1
    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, 1


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(cfg, which: str):
    kwargs = dict(
        dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(cfg.model.attn_implementation),
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    base = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_id, **kwargs)
    if which == "base":
        return base
    if which == "merged":
        return AutoModelForCausalLM.from_pretrained(cfg.output.merged_dir, **kwargs)
    # which == "adapter"
    from peft import PeftModel

    return PeftModel.from_pretrained(base, cfg.training.output_dir)


# --------------------------------------------------------------------------- #
# Generation + scoring loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, tokenizer, messages, max_new_tokens, temperature):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--which", choices=["base", "adapter", "merged"], default="adapter")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--num-samples", type=int, default=None, help="Override eval.num_samples (subset for a faster pass)")
    ap.add_argument("--judge", action="store_true", help="Add LLM-as-judge faithfulness (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg.model.base_model_id)
    model = load_model(cfg, args.which)
    model.eval()

    test = load_dataset("json", data_files={"test": cfg.data.test_file})["test"]
    n = args.num_samples if args.num_samples is not None else cfg.eval.num_samples
    if n:
        test = test.select(range(min(n, len(test))))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    per_task = defaultdict(lambda: defaultdict(list))
    examples_dump = []

    for i, row in enumerate(test):
        messages = row["messages"]
        prompt, reference = messages[:-1], messages[-1]["content"]
        pred = generate(model, tokenizer, prompt, cfg.eval.max_new_tokens, cfg.eval.temperature)
        task = row["task"]

        if task == "extract_eligibility":
            p, r, f1, valid = criteria_prf(pred, reference)
            per_task[task]["precision"].append(p)
            per_task[task]["recall"].append(r)
            per_task[task]["f1"].append(f1)
            per_task[task]["json_valid"].append(valid)
        elif task == "phase_classification":
            per_task[task]["exact_match"].append(float(_norm(pred) == _norm(reference)))
        else:  # summary / condition_qa
            s = scorer.score(reference, pred)
            per_task[task]["rouge1"].append(s["rouge1"].fmeasure)
            per_task[task]["rougeL"].append(s["rougeL"].fmeasure)
            per_task[task]["token_f1"].append(token_f1(pred, reference))

        if i < 12:
            examples_dump.append({"task": task, "prompt": prompt[-1]["content"],
                                  "reference": reference, "prediction": pred})
        if (i + 1) % 25 == 0:
            print(f"[eval] {i + 1}/{len(test)}", end="\r")
    print()

    # Aggregate.
    summary = {}
    for task, metrics in per_task.items():
        summary[task] = {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}
        summary[task]["n"] = len(next(iter(metrics.values())))

    if args.judge:
        summary["_llm_judge_faithfulness"] = llm_judge(examples_dump)

    result = {"run_name": args.run_name, "which": args.which, "model": cfg.model.base_model_id,
              "n_test": len(test), "metrics": summary}
    out_path = os.path.join(args.out_dir, f"metrics_{args.run_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[eval] metrics -> {out_path}")

    ex_path = os.path.join(args.out_dir, f"examples_{args.run_name}.md")
    with open(ex_path, "w") as f:
        f.write(f"# Qualitative examples — {args.run_name}\n\n")
        for e in examples_dump:
            f.write(f"### Task: {e['task']}\n\n**Prompt:** {e['prompt'][:500]}\n\n")
            f.write(f"**Reference:** {e['reference'][:500]}\n\n")
            f.write(f"**Prediction:** {e['prediction'][:500]}\n\n---\n\n")
    print(f"[eval] examples -> {ex_path}")


def llm_judge(examples: list[dict]) -> float:
    """Average faithfulness score (1-5) from Claude as a judge. Optional."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[eval] --judge set but ANTHROPIC_API_KEY missing; skipping")
        return -1.0
    try:
        import anthropic
    except ImportError:
        print("[eval] `anthropic` not installed; skipping judge")
        return -1.0

    client = anthropic.Anthropic(api_key=key)
    scores = []
    for e in examples:
        msg = (
            "Score 1-5 how faithful and complete the prediction is versus the reference for a "
            "clinical-trial task. Reply with only the integer.\n\n"
            f"PROMPT: {e['prompt'][:1500]}\n\nREFERENCE: {e['reference'][:1500]}\n\n"
            f"PREDICTION: {e['prediction'][:1500]}"
        )
        try:
            resp = client.messages.create(
                model="claude-sonnet-5", max_tokens=5,
                messages=[{"role": "user", "content": msg}],
            )
            scores.append(int(re.search(r"[1-5]", resp.content[0].text).group()))
        except Exception as exc:  # network / parse errors shouldn't kill eval
            print(f"[eval] judge error: {exc}")
    return round(sum(scores) / len(scores), 3) if scores else -1.0


if __name__ == "__main__":
    main()
