---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - clinical-trials
  - healthcare
  - biomedical
  - information-extraction
  - qlora
  - peft
  - trl
  - grpo
---

# ClinTrial-LM — Qwen2.5-7B fine-tuned for clinical-trial understanding

A **QLoRA**-fine-tuned [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
specialised for clinical-trial tasks, trained on ~26k instruction examples derived from the
[ClinicalTrials.gov](https://clinicaltrials.gov) registry.

📦 **Code, full pipeline & write-up:** https://github.com/omkar-droid/clintrial-lm

> ⚕️ **Research/education only. Not medical advice.** Do not use for clinical decision-making.

## What it does

| Task | Description |
|---|---|
| **Eligibility extraction** | Free-text inclusion/exclusion criteria → structured JSON |
| **Plain-language summary** | Trial description → patient-friendly summary |
| **Condition Q&A** | "What conditions does this trial study?" |
| **Phase classification** | Identify the trial phase |

## Results

Evaluated on a **held-out test set split by trial ID** (no trial appears in both train and test),
500 sampled examples.

| Task | Metric | Base Qwen2.5-7B | **This model (SFT)** |
|---|---|---|---|
| Eligibility extraction | criterion F1 | 0.840 | **0.968** |
| Eligibility output | **JSON validity** | 0.986 | **1.000** |
| Phase classification | exact match | 0.000 | **0.794** |
| Condition Q&A | token F1 | 0.298 | **0.742** |
| Plain-language summary | ROUGE-L | 0.195 | **0.290** |

Eligibility extraction reaches **precision 0.962 / recall 0.978** with **100% schema-valid JSON**.

The base model scores 0.000 on phase classification not because it lacks the knowledge, but because
it won't answer in the required format — fine-tuning buys **format discipline and faithfulness**.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "OmkarShewale/clintrial-qwen2.5-7b-sft"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

SYSTEM = (
    "You are a clinical research assistant. You help patients and clinicians understand "
    "clinical trials. Answer only from the information provided, be precise, and never "
    "invent eligibility criteria, conditions, or outcomes."
)

criteria = """Inclusion Criteria:
1. Adults aged 18 years or older with confirmed type 2 diabetes
2. HbA1c between 7.0% and 10.5% at screening

Exclusion Criteria:
1. History of severe hypoglycaemia within the last 6 months
2. Pregnancy or breastfeeding
"""

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content":
        'Extract the eligibility criteria from the trial text below into JSON with two lists, '
        '"inclusion" and "exclusion". Copy each criterion verbatim; do not add any.\n\n'
        f"Trial eligibility text:\n{criteria}"},
]

enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                    return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=1024, do_sample=False)
print(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Note:** eligibility answers can be long (up to ~1.7k tokens). Use `max_new_tokens >= 1024` or the
JSON will be truncated and fail to parse.

## Training

| | |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Method | QLoRA — 4-bit NF4 + LoRA (r=16, α=32, dropout 0.05) |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Trainable params | ~0.5% of total |
| Precision | bf16 compute, gradient checkpointing |
| Effective batch | 32 (8 × 4 grad-accum) |
| LR / schedule | 2e-4, cosine, 3% warmup |
| Hardware | 1× NVIDIA H100 NVL (95 GB), ~19 GB used |
| Framework | HuggingFace TRL + PEFT + Transformers |

**Best-checkpoint selection matters here:** eval loss bottomed around step 500 and then *rose* while
train loss kept falling (overfitting). `load_best_model_at_end` on `eval_loss` selected the step-500
checkpoint — this model. Training for 3 epochs was more than necessary.

A **GRPO / RLVR** variant (RL against a programmatic reward: JSON validity + criterion F1) was also
trained; it matched this SFT model but did not beat it, because SFT had already saturated the reward.
Details in the [repo](https://github.com/omkar-droid/clintrial-lm).

## Data

Built from 8,000 real ClinicalTrials.gov studies. Targets come from the registry's **own structured
fields** (not a teacher LLM), so labels are auditable. Splits are partitioned **by trial**, not by
example, to prevent leakage.

## Limitations

- **Not medical advice.** Outputs may be wrong; a human expert must review anything clinical.
- Trained on English registry text only; performance on other formats/languages is unknown.
- Summary quality (ROUGE-L 0.29) is the weakest task.
- Only 8k of ~500k available trials were used — more data would likely improve generalisation.

## License

Apache-2.0, inherited from Qwen2.5. Source registry data is public-domain U.S. government work.
