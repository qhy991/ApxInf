#!/usr/bin/env bash
set -euo pipefail

# Reference bring-up backend. Replace this launcher with the real apxinf
# service while keeping spec.json, generated manifests, and run_benchmark.py.
VENV_DIR="${VENV_DIR:-/root/venvs/qwen38-vllm-0.27.1}"
MODEL_DIR="${MODEL_DIR:-/mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-awq-int4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# 32K prompt + 128 measured output tokens + room for chat-template variance.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-33792}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

exec "${VENV_DIR}/bin/vllm" serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-chunked-prefill \
  --enforce-eager \
  --trust-remote-code
