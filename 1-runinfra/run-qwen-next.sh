#!/bin/bash
# run-qwen-next.sh - one instance. Override any of these via env:
#   GGML_VK_VISIBLE_DEVICES=2,3 PORT=9932 NCMOE=44 CTX=32768 ./run-qwen-next.sh
set -u

MODEL=${MODEL:-../0-build/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
PORT=${PORT:-9931}
NCMOE=${NCMOE:-44}
CTX=${CTX:-32768}
THREADS=${THREADS:-24}
export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-0}

/llama-vulkan/build/bin/llama-server \
  -m "$MODEL" \
  -ngl 99 -ot "per_layer_token_embd=CPU" -ncmoe "$NCMOE" \
  -lm mmap -lzm off \
  -c "$CTX" -np 1 -fa on -ctk q8_0 -ctv q8_0 \
  --no-context-shift -b 2048 -ub 512 \
  -t "$THREADS" -tb "$THREADS" --jinja --tools all \
  --host 0.0.0.0 --port "$PORT"
