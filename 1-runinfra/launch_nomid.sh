#!/bin/bash
/media/aryan/nvme/llama.cpp/llama.cpp-embedded/build/bin/llama-server -m /media/aryan/nvme/models/nomic-embed-text-v1.5.Q6_K.gguf --device VULKAN0 --embedding --host 0.0.0.0 --port 8034 --ctx-size 8192   -b 8192   -ub 8192  --batch-size 1024  --n-gpu-layers 99 -np 1
