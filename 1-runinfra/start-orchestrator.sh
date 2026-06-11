#!/bin/bash
/media/aryan/nvme/llama.orch/build/bin/llama-server   -m /media/aryan/nvme/models/nvidia_Orchestrator-8B-Q6_K.gguf   -ngl 99  --no-cache-idle-slots --cache-ram 0 -c 40960   -b 512   -ub 512   --parallel 2   --no-mmap   --tools all   --jinja   --kv-unified   -fa on   -ctk q8_0   -ctv q4_0   -fit off   --host 0.0.0.0   --port 8080 --device VULKAN1
