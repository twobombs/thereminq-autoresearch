# this script starts all requirements for autoresearch to start on a six gpu machine

# 27B 'Apex' substitute - replace with GLM or fronteer Qwen variant
# GGML_VK_VISIBLE_DEVICES=0,4 /root/llama-vulkan/build/bin/llama-server   -m /media/aryan/nvme/models/Qwen3.6-27B-UD-IQ3_XXS.gguf   --tools all   -c 65536   -np 1   -ngl 999   --load-mode mmap   -fa on   -t 48   -tb 48   --reasoning on   --reasoning-preserve   --host 0.0.0.0   --port 8081   --cors-origins "*"   -ctk q8_0   -ctv q4_0   --spec-type draft-mtp   --spec-draft-n-max 3 &

# we leverage the CPU for the large model to give room for the workers and orchestration 
# the 'apex' model does not need to do a whole lotm so this helps a lot of GPU idle time
/root/llama-vulkan/build/bin/llama-server   -m /media/aryan/nvme/models/Qwen3.5-122B-A10B-UD/Qwen3.5-122B-A10B-UD-IQ1_M.gguf   --tools all   -c 131072   -np 1   -ngl 0   --load-mode mmap   -fa on   -t 48   -tb 48   --reasoning on   --reasoning-preserve   --host 0.0.0.0   --port 8081   --cors-origins "*"   -ctk q8_0   -ctv q4_0   --spec-type draft-mtp   --spec-draft-n-max 5


# stitcher workers 
GGML_VK_VISIBLE_DEVICES=0 /root/llama-vulkan-worker1/build/bin/llama-server   -m /media/aryan/nvme/models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf   --model-draft /media/aryan/nvme/models/mtp-gemma-4-E4B-it-Q4_0.gguf   -c 196608   --no-cache-idle-slots   -np 2   -ngl 999   --kv-unified   -fa on   --split-mode none   --cache-type-k q8_0   --cache-type-v q4_0   --load-mode none   --spec-type draft-mtp   --spec-draft-n-max 3   --host 0.0.0.0   --port 8070   --tools all   --fit off   --jinja
GGML_VK_VISIBLE_DEVICES=4 /root/llama-vulkan-worker1/build/bin/llama-server   -m /media/aryan/nvme/models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf   --model-draft /media/aryan/nvme/models/mtp-gemma-4-E4B-it-Q4_0.gguf   -c 196608   --no-cache-idle-slots   -np 2   -ngl 999   --kv-unified   -fa on   --split-mode none   --cache-type-k q8_0   --cache-type-v q4_0   --load-mode none   --spec-type draft-mtp   --spec-draft-n-max 3   --host 0.0.0.0   --port 8071   --tools all   --fit off   --jinja

# orchestrator
GGML_VK_VISIBLE_DEVICES=2 /root/llama-vulkan-orchestrator/build/bin/llama-server   -m /media/aryan/nvme/models/nvidia_Orchestrator-8B-Q5_K_S.gguf   -ngl 99  --no-cache-idle-slots --cache-ram 0 -c 40960   -b 512   -ub 512   --parallel 1   --no-mmap   --tools all   --jinja   --kv-unified   -fa on   -ctk q8_0   -ctv q4_0   -fit off   --host 0.0.0.0   --port 8080 &
GGML_VK_VISIBLE_DEVICES=5 /root/llama-vulkan-orchestrator2/build/bin/llama-server   -m /media/aryan/nvme/models/nvidia_Orchestrator-8B-Q5_K_S.gguf   -ngl 99  --no-cache-idle-slots --cache-ram 0 -c 40960   -b 512   -ub 512   --parallel 1   --no-mmap   --tools all   --jinja   --kv-unified   -fa on   -ctk q8_0   -ctv q4_0   -fit off   --host 0.0.0.0   --port 8079 &

# workers
GGML_VK_VISIBLE_DEVICES=3 /root/llama-vulkan-worker1/build/bin/llama-server   -m /media/aryan/nvme/models/Qwen3.5-9B-IQ4_XS.gguf   -c 196608  --no-cache-idle-slots  -np 2   -ngl 999   --device Vulkan0   --kv-unified   -fa on   --split-mode none   --cache-type-k q8_0   --cache-type-v q4_0    --no-mmap   --spec-type draft-mtp   --spec-draft-n-max 3   --host 0.0.0.0   --port 8033   --tools all   --fit off --jinja &
GGML_VK_VISIBLE_DEVICES=1 /root/llama-vulkan-worker2/build/bin/llama-server   -m /media/aryan/nvme/models/Qwen3.5-9B-IQ4_XS.gguf   -c 196608  --no-cache-idle-slots  -np 2   -ngl 999   --device Vulkan0   --kv-unified   -fa on   --split-mode none   --cache-type-k q8_0   --cache-type-v q4_0    --no-mmap   --spec-type draft-mtp   --spec-draft-n-max 3   --host 0.0.0.0   --port 8034   --tools all   --fit off --jinja &
