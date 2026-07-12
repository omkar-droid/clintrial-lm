"""Collate results/metrics_*.json into a single markdown comparison table.

    python scripts/make_report.py            # prints markdown to stdout
    python scripts/make_report.py > results/REPORT.md

Paste the output into the README's Results section, or commit results/REPORT.md.
Keeps the reported numbers honest: they come straight from the eval JSON, never
hand-typed.
"""
from __future__ import annotations

import glob
import json
import os

# (task, metric) columns to surface in the headline table.
COLUMNS = [
    ("extract_eligibility", "f1", "Eligibility F1"),
    ("extract_eligibility", "json_valid", "JSON valid"),
    ("plain_language_summary", "rougeL", "Summary ROUGE-L"),
    ("condition_qa", "token_f1", "Condition F1"),
    ("phase_classification", "exact_match", "Phase acc."),
]

# Preferred row order if these runs exist.
ORDER = ["base", "sft", "lora", "dpo"]


def main() -> None:
    files = glob.glob("results/metrics_*.json")
    if not files:
        print("No results/metrics_*.json found. Run `make eval-base eval-sft eval-dpo` first.")
        return
    runs = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        runs[data["run_name"]] = data["metrics"]

    ordered = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]

    header = "| Run | " + " | ".join(label for *_, label in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    print(header)
    print(sep)
    for run in ordered:
        cells = []
        for task, metric, _ in COLUMNS:
            val = runs[run].get(task, {}).get(metric)
            cells.append(f"{val:.3f}" if isinstance(val, (int, float)) else "—")
        print(f"| {run} | " + " | ".join(cells) + " |")

    for run in ordered:
        judge = runs[run].get("_llm_judge_faithfulness")
        if isinstance(judge, (int, float)) and judge >= 0:
            print(f"\n_LLM-judge faithfulness ({run}): {judge}/5_")


if __name__ == "__main__":
    main()
