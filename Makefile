# ClinTrial-LM pipeline. Run `make help` for the list of stages.
.PHONY: help setup data data-offline train-sft train-lora merge-sft train-grpo merge-grpo \
        train-dpo merge-dpo eval-base eval-sft eval-grpo eval-dpo report plots serve clean

PYTHON ?= python
CONFIG_SFT   ?= configs/qlora_sft.yaml
CONFIG_LORA  ?= configs/lora_sft.yaml
CONFIG_GRPO  ?= configs/grpo.yaml
CONFIG_DPO   ?= configs/dpo.yaml

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:            ## Install training dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:             ## Build the clinical-trial instruction dataset from ClinicalTrials.gov
	$(PYTHON) data/build_dataset.py --max-studies 8000 --out-dir data/processed

data-offline:     ## Build a tiny synthetic sample (no network) to smoke-test the pipeline
	$(PYTHON) data/build_dataset.py --offline --out-dir data/processed

train-sft:        ## QLoRA supervised fine-tuning (4-bit)
	$(PYTHON) src/train_sft.py --config $(CONFIG_SFT)

train-lora:       ## bf16 LoRA supervised fine-tuning (comparison run)
	$(PYTHON) src/train_sft.py --config $(CONFIG_LORA)

merge-sft:        ## Merge the SFT adapter into a standalone fp16 model
	$(PYTHON) src/merge_and_export.py --config $(CONFIG_SFT)

train-grpo:       ## GRPO / RLVR on top of the merged SFT policy (verifiable rewards)
	$(PYTHON) src/train_grpo.py --config $(CONFIG_GRPO)

merge-grpo:       ## Merge the GRPO adapter for serving
	$(PYTHON) src/merge_and_export.py --config $(CONFIG_GRPO)

train-dpo:        ## DPO preference tuning on top of the merged SFT model
	$(PYTHON) src/train_dpo.py --config $(CONFIG_DPO)

merge-dpo:        ## Merge the DPO adapter for serving
	$(PYTHON) src/merge_and_export.py --config $(CONFIG_DPO)

eval-base:        ## Evaluate the untuned base model (baseline)
	$(PYTHON) src/evaluate.py --config $(CONFIG_SFT) --which base --run-name base

eval-sft:         ## Evaluate the SFT model
	$(PYTHON) src/evaluate.py --config $(CONFIG_SFT) --which adapter --run-name sft

eval-grpo:        ## Evaluate the GRPO model
	$(PYTHON) src/evaluate.py --config $(CONFIG_GRPO) --which adapter --run-name grpo

eval-dpo:         ## Evaluate the DPO model
	$(PYTHON) src/evaluate.py --config $(CONFIG_DPO) --which adapter --run-name dpo

report:           ## Rebuild the results table from results/metrics_*.json
	$(PYTHON) scripts/make_report.py | tee results/REPORT.md

quantize-awq:     ## Produce a calibrated AWQ 4-bit checkpoint from the merged model
	$(PYTHON) src/quantize.py --config $(CONFIG_SFT) --method awq

benchmark:        ## Benchmark inference cost for a quant mode: make benchmark Q=nf4
	$(PYTHON) src/benchmark.py --config $(CONFIG_SFT) --quant $(or $(Q),bf16) --run-name $(or $(Q),bf16) --out-dir results/quant

quant-report:     ## Join benchmark + accuracy into the quantization table
	$(PYTHON) scripts/make_quant_report.py | tee results/QUANT_REPORT.md

plots:            ## Regenerate all charts in assets/
	$(PYTHON) scripts/make_plots.py
	$(PYTHON) scripts/make_quant_plot.py

serve:            ## Serve the merged model with vLLM (OpenAI-compatible API on :8000)
	$(PYTHON) src/serve_vllm.py --config $(CONFIG_GRPO)

clean:            ## Remove local outputs (does NOT touch pushed Hub models)
	rm -rf outputs checkpoints wandb merged
