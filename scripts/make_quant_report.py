"""Join the quantization benchmarks with the accuracy evals into one table.

Answers the only question a deployment actually asks:
**how much quality do I give up, for how much memory and speed?**

    python scripts/make_quant_report.py > results/QUANT_REPORT.md
"""
from __future__ import annotations

import glob
import json
import os

QUANT_DIR = "results/quant"
ORDER = ["bf16", "int8", "nf4", "awq", "gptq"]
LABEL = {
    "bf16": "bf16 (baseline)",
    "int8": "INT8 (bitsandbytes)",
    "nf4": "NF4 4-bit (bitsandbytes)",
    "awq": "AWQ 4-bit",
    "gptq": "GPTQ 4-bit",
}


def load(pattern: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(QUANT_DIR, pattern)):
        with open(path) as f:
            d = json.load(f)
        out[d["run_name"]] = d
    return out


def main() -> None:
    bench = load("bench_*.json")
    evals = load("metrics_*.json")
    runs = [r for r in ORDER if r in bench or r in evals]
    runs += [r for r in set(bench) | set(evals) if r not in ORDER]
    if not runs:
        print("No results in results/quant/. Run the quantization sweep first.")
        return

    def elig(run, key):
        return evals.get(run, {}).get("metrics", {}).get("extract_eligibility", {}).get(key)

    base_f1 = elig("bf16", "f1")

    print("| Method | Disk | Peak VRAM | TTFT | Decode | Eligibility F1 | JSON valid | Quality kept |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        b, f1, jv = bench.get(r, {}), elig(r, "f1"), elig(r, "json_valid")
        kept = f"{100 * f1 / base_f1:.1f}%" if (f1 and base_f1) else "—"
        print(
            f"| {LABEL.get(r, r)} "
            f"| {b.get('disk_size_gb', '—')} GB "
            f"| {b.get('peak_vram_gb', '—')} GB "
            f"| {b.get('ttft_ms_mean', '—')} ms "
            f"| {b.get('decode_tok_per_s_mean', '—')} tok/s "
            f"| {f1 if f1 is not None else '—'} "
            f"| {jv if jv is not None else '—'} "
            f"| {kept} |"
        )

    if base_f1:
        print(f"\n_Baseline = bf16 (F1 {base_f1}). 'Quality kept' = F1 relative to baseline._")


if __name__ == "__main__":
    main()
