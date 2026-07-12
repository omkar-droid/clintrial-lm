# Training image for ClinTrial-LM. Built on the official CUDA runtime so it runs
# on any H100 box with the NVIDIA Container Toolkit installed.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Install torch matching the CUDA base first, then the rest of the stack.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# flash-attention needs torch present at build time.
RUN python -m pip install flash-attn --no-build-isolation || \
    echo "flash-attn build skipped; training falls back to SDPA attention"

COPY . .

CMD ["bash"]
