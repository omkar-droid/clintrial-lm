"""Build a clinical-trial instruction-tuning dataset from ClinicalTrials.gov.

Design choices worth calling out (these are the interesting engineering bits):

* **Programmatic ground truth.** Rather than distilling from a teacher LLM, targets
  come straight from structured registry fields (brief summary, parsed eligibility
  criteria, conditions, phase). That means the labels are auditable and free of
  another model's hallucinations.
* **Leakage-safe splits.** Train/val/test are split on *trial* (NCT id), never on
  individual examples, so no trial leaks across splits even though each trial spawns
  several instruction examples.
* **DPO pairs.** Preference data is synthesised by perturbing the gold answer
  (dropping criteria, injecting a hallucinated one, or swapping in an unrelated
  answer). The `chosen` response is faithful and complete; the `rejected` one is
  degraded — exactly the behaviour we want DPO to push away from.
* **Offline mode.** `--offline` generates a small synthetic sample so the whole
  pipeline can be smoke-tested with no network / no GPU.

Usage:
    python data/build_dataset.py --max-studies 8000 --out-dir data/processed
    python data/build_dataset.py --offline --out-dir data/processed
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from typing import Iterable

try:
    import requests
except ImportError:  # requests is only needed for online mode
    requests = None

API_URL = "https://clinicaltrials.gov/api/v2/studies"

SYSTEM_PROMPT = (
    "You are a clinical research assistant. You help patients and clinicians understand "
    "clinical trials. Answer only from the information provided, be precise, and never "
    "invent eligibility criteria, conditions, or outcomes."
)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def fetch_studies(max_studies: int, page_size: int = 100, query_term: str | None = None) -> list[dict]:
    """Page through the ClinicalTrials.gov v2 API and return raw study dicts."""
    if requests is None:
        raise RuntimeError("`requests` is required for online mode. pip install requests")
    studies: list[dict] = []
    page_token: str | None = None
    while len(studies) < max_studies:
        params = {
            "pageSize": min(page_size, max_studies - len(studies)),
            "format": "json",
        }
        if query_term:
            params["query.term"] = query_term
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("studies", [])
        if not batch:
            break
        studies.extend(batch)
        page_token = payload.get("nextPageToken")
        print(f"[fetch] {len(studies)} studies", end="\r")
        if not page_token:
            break
    print()
    return studies[:max_studies]


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|[a-z][.)])\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\r", " ")).strip()


def parse_eligibility(text: str) -> tuple[list[str], list[str]]:
    """Split free-text eligibility criteria into (inclusion, exclusion) item lists."""
    if not text:
        return [], []
    # Find the exclusion header and split there.
    m = re.search(r"exclusion criteria\s*:?", text, flags=re.IGNORECASE)
    if m:
        inclusion_block = re.sub(r"inclusion criteria\s*:?", "", text[: m.start()], flags=re.IGNORECASE)
        exclusion_block = text[m.end():]
    else:
        inclusion_block = re.sub(r"inclusion criteria\s*:?", "", text, flags=re.IGNORECASE)
        exclusion_block = ""

    def to_items(block: str) -> list[str]:
        items: list[str] = []
        for line in block.split("\n"):
            line = _clean(_BULLET_RE.sub("", line))
            if len(line) >= 8:  # drop stray fragments / headers
                items.append(line)
        return items

    return to_items(inclusion_block), to_items(exclusion_block)


def _get(study: dict, *path, default=None):
    node = study
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #
def _chat(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_examples(study: dict) -> list[dict]:
    """Turn one registry study into 0..N instruction examples."""
    ps = study.get("protocolSection", {})
    nct = _get(ps, "identificationModule", "nctId", default="")
    title = _clean(_get(ps, "identificationModule", "officialTitle")
                   or _get(ps, "identificationModule", "briefTitle", default=""))
    brief = _clean(_get(ps, "descriptionModule", "briefSummary", default=""))
    conditions = _get(ps, "conditionsModule", "conditions", default=[]) or []
    phases = _get(ps, "designModule", "phases", default=[]) or []
    interventions = [
        _clean(i.get("name", "")) for i in _get(ps, "armsInterventionsModule", "interventions", default=[]) or []
    ]
    elig_text = _get(ps, "eligibilityModule", "eligibilityCriteria", default="")
    inclusion, exclusion = parse_eligibility(elig_text)

    examples: list[dict] = []

    # Task 1: structured eligibility extraction (programmatic ground truth).
    if inclusion or exclusion:
        target = json.dumps({"inclusion": inclusion, "exclusion": exclusion}, indent=2)
        user = (
            "Extract the eligibility criteria from the trial text below into JSON with two "
            "lists, \"inclusion\" and \"exclusion\". Copy each criterion verbatim; do not add any.\n\n"
            f"Trial eligibility text:\n{_clean(elig_text)}"
        )
        examples.append({"task": "extract_eligibility", "nct_id": nct, "messages": _chat(user, target)})

    # Task 2: plain-language summary (target = registry brief summary).
    if brief and len(brief.split()) >= 25 and title:
        parts = [f"Title: {title}"]
        if conditions:
            parts.append("Conditions: " + ", ".join(conditions))
        if interventions:
            parts.append("Interventions: " + ", ".join(interventions))
        user = (
            "Write a short, plain-language summary (3-5 sentences) of this clinical trial for a "
            "patient audience.\n\n" + "\n".join(parts)
        )
        examples.append({"task": "plain_language_summary", "nct_id": nct, "messages": _chat(user, brief)})

    # Task 3: condition Q&A (programmatic ground truth).
    if conditions and brief:
        user = f"Based on this trial summary, what medical conditions is it studying? List them.\n\n{brief}"
        target = "This trial studies: " + "; ".join(conditions) + "."
        examples.append({"task": "condition_qa", "nct_id": nct, "messages": _chat(user, target)})

    # Task 4: phase classification (programmatic ground truth).
    if phases and brief:
        user = f"What clinical trial phase is described below? Answer with the phase(s) only.\n\n{brief}"
        target = ", ".join(p.replace("_", " ").title() for p in phases)
        examples.append({"task": "phase_classification", "nct_id": nct, "messages": _chat(user, target)})

    return examples


# --------------------------------------------------------------------------- #
# DPO preference pairs (perturb the gold answer to create a `rejected`)
# --------------------------------------------------------------------------- #
def make_dpo_pair(example: dict, other_examples: list[dict], rng: random.Random) -> dict | None:
    """Return a {prompt, chosen, rejected} triple, or None if not applicable."""
    task = example["task"]
    messages = example["messages"]
    prompt = messages[:-1]                       # system + user
    chosen = messages[-1]["content"]

    if task == "extract_eligibility":
        data = json.loads(chosen)
        bad = copy.deepcopy(data)
        # Drop half of the inclusion criteria (incompleteness) ...
        if len(bad["inclusion"]) > 1:
            keep = max(1, len(bad["inclusion"]) // 2)
            bad["inclusion"] = bad["inclusion"][:keep]
        # ... and inject a hallucinated exclusion criterion.
        bad["exclusion"] = bad["exclusion"] + ["Pregnant or breastfeeding women are excluded."]
        rejected = json.dumps(bad, indent=2)
    else:
        # For free-text tasks: reject an unrelated answer (faithfulness signal) or a
        # truncated one.
        if rng.random() < 0.5 and other_examples:
            other = rng.choice(other_examples)
            rejected = other["messages"][-1]["content"]
        else:
            words = chosen.split()
            rejected = " ".join(words[: max(4, len(words) // 3)]) + " ..."
    if rejected.strip() == chosen.strip():
        return None
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


# --------------------------------------------------------------------------- #
# Offline synthetic sample
# --------------------------------------------------------------------------- #
def synthetic_studies(n: int = 40) -> list[dict]:
    rng = random.Random(0)
    conditions_pool = ["Type 2 Diabetes", "Hypertension", "Breast Cancer", "Asthma", "Depression"]
    drugs = ["Metformin", "Lisinopril", "Pembrolizumab", "Albuterol", "Sertraline"]
    out = []
    for i in range(n):
        cond = rng.choice(conditions_pool)
        drug = rng.choice(drugs)
        elig = (
            "Inclusion Criteria:\n"
            f"- Adults aged 18-75 with confirmed {cond}\n"
            "- Able to provide informed consent\n"
            "- Stable medication regimen for 3 months\n\n"
            "Exclusion Criteria:\n"
            "- Participation in another trial within 30 days\n"
            "- Known allergy to study drug\n"
        )
        out.append({
            "protocolSection": {
                "identificationModule": {
                    "nctId": f"NCT9{i:07d}",
                    "briefTitle": f"A Study of {drug} in {cond}",
                    "officialTitle": f"A Randomized Phase 2 Study of {drug} for the Treatment of {cond}",
                },
                "descriptionModule": {
                    "briefSummary": (
                        f"This clinical trial evaluates whether {drug} improves outcomes in adults with "
                        f"{cond}. Participants are randomly assigned to receive either {drug} or a placebo "
                        "and are followed for 24 weeks. The goal is to measure changes in disease markers "
                        "and to assess the safety of the treatment."
                    ),
                },
                "conditionsModule": {"conditions": [cond]},
                "designModule": {"phases": ["PHASE2"], "studyType": "INTERVENTIONAL"},
                "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": drug}]},
                "eligibilityModule": {"eligibilityCriteria": elig, "sex": "ALL", "minimumAge": "18 Years"},
            }
        })
    return out


# --------------------------------------------------------------------------- #
# Split + write
# --------------------------------------------------------------------------- #
def split_by_trial(examples: list[dict], ratios=(0.9, 0.05, 0.05), seed: int = 42):
    rng = random.Random(seed)
    nct_ids = sorted({e["nct_id"] for e in examples})
    rng.shuffle(nct_ids)
    n = len(nct_ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_ids = set(nct_ids[:n_train])
    val_ids = set(nct_ids[n_train:n_train + n_val])
    buckets = {"train": [], "val": [], "test": []}
    for e in examples:
        split = "train" if e["nct_id"] in train_ids else "val" if e["nct_id"] in val_ids else "test"
        buckets[split].append(e)
    return buckets


def write_jsonl(rows: Iterable[dict], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-studies", type=int, default=8000)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--query-term", type=str, default=None, help="Optional ClinicalTrials.gov search term")
    ap.add_argument("--out-dir", type=str, default="data/processed")
    ap.add_argument("--offline", action="store_true", help="Use a small synthetic sample (no network)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print("[data] fetching studies ...")
    studies = synthetic_studies() if args.offline else fetch_studies(
        args.max_studies, args.page_size, args.query_term
    )
    print(f"[data] {len(studies)} studies")

    examples: list[dict] = []
    for study in studies:
        examples.extend(build_examples(study))
    rng.shuffle(examples)
    print(f"[data] {len(examples)} SFT examples across tasks")

    buckets = split_by_trial(examples, seed=args.seed)
    for split, rows in buckets.items():
        n = write_jsonl(rows, os.path.join(args.out_dir, f"{split}.jsonl"))
        print(f"[data]   {split}: {n} examples")

    # DPO pairs built only from the training trials to avoid leakage into eval.
    # Restricted to tasks with a long, meaningful answer so the chosen/rejected gap
    # is a real faithfulness/completeness signal (short labels give a weak signal).
    dpo_tasks = {"extract_eligibility", "plain_language_summary"}
    summaries_pool = [e for e in buckets["train"] if e["task"] == "plain_language_summary"]
    dpo_rows = []
    for e in buckets["train"]:
        if e["task"] not in dpo_tasks:
            continue
        pair = make_dpo_pair(e, summaries_pool, rng)
        if pair:
            dpo_rows.append(pair)
    rng.shuffle(dpo_rows)
    n_val = max(1, len(dpo_rows) // 20)
    write_jsonl(dpo_rows[n_val:], os.path.join(args.out_dir, "dpo_train.jsonl"))
    write_jsonl(dpo_rows[:n_val], os.path.join(args.out_dir, "dpo_val.jsonl"))
    print(f"[data]   dpo: {len(dpo_rows) - n_val} train / {n_val} val pairs")

    # Task distribution report (nice to show in the README / dataset card).
    dist: dict[str, int] = {}
    for e in examples:
        dist[e["task"]] = dist.get(e["task"], 0) + 1
    print("[data] task distribution:", json.dumps(dist, indent=2))


if __name__ == "__main__":
    main()
