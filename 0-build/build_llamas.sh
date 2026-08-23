#!/bin/bash
apt update
apt install -y build-essential cmake glslang-tools spirv-headers 
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j $(grep -c ^processor /proc/cpuinfo)

cd ..

cp -r ./llama.cpp ./llama.orch
cp -r ./llama.cpp ./llama-vulkan
cp -r ./llama.cpp ./llama-vulkan-cpu
cp -r ./llama.cpp ./llama-vulkan-orchestrator
cp -r ./llama.cpp ./llama-vulkan-orchestrator2
cp -r ./llama.cpp ./llama-vulkan-worker1
cp -r ./llama.cpp ./llama-vulkan-worker2
cp -r ./llama.cpp ./llama-vulkan-worker3
