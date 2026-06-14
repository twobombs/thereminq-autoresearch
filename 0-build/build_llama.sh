#!/bin/bash

apt install -y build-essential cmake glslang-tools spirv-headers 
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

cd ..

cp -r ./llama.cpp ./llama.orch
