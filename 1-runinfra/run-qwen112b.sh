GGML_VK_VISIBLE_DEVICES=1,2,3,4,5 ./build/bin/llama-server   -m /media/aryan/nvme/models/Qwen3.5-122B-A10B-UD/Qwen3.5-122B-A10B-UD-IQ1_M.gguf   --tools all   -c 131072   -np 2   -ngl 9
99   --load-mode mmap   -fa on   -t 48   -tb 48   --reasoning on   --reasoning-preserve   --host 0.0.0.0   --port 8033   --cors-origins "*"   -ctk q8_0   -ctv q4_0   --tensor-split 1,1,1,1,1   --spec-type draf
t-mtp   --spec-draft-n-max 5
