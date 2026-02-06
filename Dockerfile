# === Build stage: nvcc needed for qtorch CUDA extensions ===
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
RUN uv venv --python python3.11 .venv
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

COPY STEP/cls/requirements.txt requirements-step.txt
COPY requirements.txt requirements.txt

RUN uv pip install --no-cache-dir -r requirements-step.txt && \
    uv pip install --no-cache-dir -r requirements.txt && \
    uv pip install --no-cache-dir git+https://github.com/braincog-X/Brain-Cog.git

# === Runtime stage: smaller image for deployment ===
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /workspace/.venv /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

COPY . .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/STEP/cls
