#!/usr/bin/env bash
# One-shot environment setup for an on-prem H100 box (Ubuntu + CUDA 12.x driver).
# Creates a Python 3.11 venv and installs the training stack.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

echo ">> Creating venv at ${VENV_DIR} using ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel

echo ">> Installing PyTorch (CUDA 12.1 build)"
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo ">> Installing training requirements"
pip install -r requirements.txt

echo ">> Attempting flash-attention 2 (optional, big speedup on H100)"
pip install flash-attn --no-build-isolation || echo "flash-attn skipped; will use SDPA attention"

echo ">> Logging in to Hugging Face + Weights & Biases (reads .env if present)"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
[ -n "${HF_TOKEN:-}" ]       && huggingface-cli login --token "${HF_TOKEN}" || echo "HF_TOKEN not set; skipping"
[ -n "${WANDB_API_KEY:-}" ]  && wandb login "${WANDB_API_KEY}"              || echo "WANDB_API_KEY not set; skipping"

echo ">> GPU visible to torch:"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo ">> Done. Activate with: source ${VENV_DIR}/bin/activate"
