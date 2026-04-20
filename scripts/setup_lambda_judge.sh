#!/bin/bash
# Bootstrap script to run on the Lambda A10 instance.
# Installs vLLM and serves Llama-3-8B-Instruct on port 8000 (localhost only).
# Run as: ssh -i ~/.ssh/lambda_claude ubuntu@<IP> 'bash -s' < setup_lambda_judge.sh
set -euo pipefail

echo "=== Checking GPU ==="
nvidia-smi -L

echo "=== Installing uv (fast Python installer) ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Setting up venv and installing vllm ==="
cd $HOME
if [ ! -d llm-venv ]; then
  uv venv llm-venv --python 3.11
fi
source llm-venv/bin/activate
uv pip install --quiet vllm==0.6.6 || uv pip install vllm

echo "=== Starting vLLM OpenAI server on :8000 (Qwen2.5-7B-Instruct) ==="
# Use Qwen 2.5-7B-Instruct: fits in A10 24GB at fp16; open weights; strong
# instruction-following for short 0/1 judge outputs.
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --host 0.0.0.0 --port 8000 \
  > ~/vllm.log 2>&1 &

echo "Started. PID: $!"
sleep 5
tail -20 ~/vllm.log || true
