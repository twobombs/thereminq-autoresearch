#!/bin/bash
../0-build/llama.cpp/build/bin/llama-server   -m ../0-build/Qwen3.8-27B-UD-IQ3_XXS.gguf   -c 131072  --no-cache-idle-slots -np 1   -ngl 999   --device Vulkan2   --kv-unified   -fa on   --split-mode none   --cache-type-k q8_0   --cache-type-v q4_0   -t 6   -tb 6   --no-mmap   --spec-type draft-mtp   --spec-draft-n-max 3    --host 0.0.0.0   --port 8033   --fit off --tools all --jinja
