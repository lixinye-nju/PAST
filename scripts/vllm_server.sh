#!/bin/bash

MODEL=$1
echo "Using model ${MODEL}"

if [ -z "$2" ]; then
    PORT=8000
else
    PORT=$2
fi

echo "Using port ${PORT}"


NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))


VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --dtype bfloat16 \
    --port ${PORT} \
    --tensor-parallel-size ${NUM_GPUS} \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384