#!/bin/bash
GGML_VK_VISIBLE_DEVICES=0 ../0-build/llama.cpp/build/bin/llama-server   -m /media/aryan/nvme/models/Qwen3.8-27B-UD-IQ3_XXS.gguf  --tools all   -c 65536   -np 1   -ngl 999   --load-mode mmap   -fa on   -t 48   -tb 48   --reasoning on  --host 0.0.0.0   --port 8033   --cors-origins "*"   -ctk q8_0   -ctv q4_0   --spec-type draft-mtp   --spec-draft-n-max 3
