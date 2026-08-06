#!/bin/bash

apt install -y build-essential cmake glslang-tools spirv-headers 
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

cd ..

cp -r ./llama.cpp ./llama.orch
cp -r ./llama.cpp ./llama-vulkan
cp -r ./llama.cpp ./llama-vulkan-orchestrator
cp -r ./llama.cpp ./llama-vulkan-worker1
cp -r ./llama.cpp ./llama-vulkan-worker2
cp -r ./llama.cpp ./llama-vulkan-worker3
