#!/bin/bash
# this script is used when we need an in-container LLM to assist vscode to execute ThereminQuantumOPS commands

cd thereminq-autoresearch/0-build/
wget https://huggingface.co/mradermacher/Qwen3.8-9B-heretic-uncensored-GGUF/resolve/main/Qwen3.8-9B-heretic-uncensored.IQ4_XS.gguf
/llama-vulkan/build/bin/llama-server   -m ../0-build/Qwen3.8-9B-heretic-uncensored.IQ4_XS.gguf   -c 196608  --no-cache-idle-slots  -np 1  -ngl 999   --device Vulkan0   --kv-unified   -fa on   --split-mode none   --cache-type-k q8_0   --cache-type-v q4_0    --no-mmap   --spec-type draft-mtp   --spec-draft-n-max 3   --host 0.0.0.0   --port 8033   --tools all   --fit off --jinja
