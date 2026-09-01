#!/bin/bash

# semi-apex model
wget https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ3_XXS.gguf &

# advisary/alt angle stitching model 
wget https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf &
wget https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/MTP/mtp-gemma-4-E4B-it-Q4_0.gguf

# worker node image plus 3.8 alt. distills
wget https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/Qwen3.5-9B-IQ4_XS.gguf &
wget https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF/resolve/main/Qwen3.8-9B-Q4_K_M.gguf &
wget https://huggingface.co/mradermacher/Qwen3.8-9B-heretic-uncensored-GGUF/resolve/main/Qwen3.8-9B-heretic-uncensored.IQ4_XS.gguf &

# distilled Qwen 3.8 2B/4B for alt stitching
wget https://huggingface.co/empero-ai/Qwen3.8-2B-Distill-GGUF/resolve/main/Qwen3.8-2B-Q4_K_M.gguf
wget https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF/resolve/main/Qwen3.8-4B-Q4_K_M.gguf

# orchestrator phased out for now in favour for MTP and nextgen model consolidation
wget https://huggingface.co/bartowski/nvidia_Orchestrator-8B-GGUF/resolve/main/nvidia_Orchestrator-8B-Q6_K.gguf &
wget https://huggingface.co/bartowski/nvidia_Orchestrator-8B-GGUF/resolve/main/nvidia_Orchestrator-8B-Q5_K_S.gguf

# Qwen 3.8 nextgen Apex midrange model
wget https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf &
wget https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf &
wget https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf
# MTP
wget https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF/resolve/main/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf

# midApex CPUonly model - not leveraged in the current setup - placeholder for distilled Apex model 
wget https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF/resolve/main/Qwen3.5-122B-A10B-UD-IQ1_M.gguf &
wget https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF/resolve/main/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf
