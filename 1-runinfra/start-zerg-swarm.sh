#!/bin/bash

# ==============================================================================
# ThereminQ-HPC Agentic Swarm Orchestrator
# 6x Qwen 3.6 9B MTP | 100% VRAM-Resident Pipeline
# Strict NUMA-to-PCIe Affinity | Auto-Restart | Shader Cache Isolation
# ==============================================================================

# Configuration
MODEL="/media/aryan/nvme/models/Qwen3.5-9B-IQ4_XS.gguf"
SERVER_BIN="../0-build/llama.cpp/build/bin/llama-server"
LOG_DIR="./agent_logs"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Define the Swarm Topology: "Vulkan_ID  NUMA_Node  API_Port"
# Mapped directly from the physical sysfs PCIe tree
SWARM=(
  "0 0 8030"
  "1 0 8031"
  "2 1 8032"
  "3 1 8033"
  "4 6 8034"
  "5 6 8035"
)

# Graceful Shutdown Sequence
shutdown_swarm() {
    # Unbind traps to prevent recursive triggering during teardown
    trap - SIGINT SIGTERM EXIT 
    echo -e "\n[ThereminQ] Shutting down all agentic nodes..."
    
    # Kill the background restart loops
    kill $(jobs -p) 2>/dev/null
    
    # Explicitly kill surviving llama-server processes to guarantee VRAM release
    pkill -f "$SERVER_BIN" 2>/dev/null
    
    wait 2>/dev/null
    echo "[ThereminQ] Swarm offline."
    exit 0
}

# Catch Ctrl+C/Termination and trigger the shutdown sequence
trap shutdown_swarm SIGINT SIGTERM EXIT

# Prerequisite Checks
if [ ! -f "$MODEL" ]; then
    echo "[!] Error: Model file not found at $MODEL"
    exit 1
fi

if [ ! -x "$SERVER_BIN" ]; then
    echo "[!] Error: llama-server executable not found or not executable at $SERVER_BIN"
    exit 1
fi

if ! command -v numactl &> /dev/null; then
    echo "[!] Error: numactl is not installed. Please install it to continue."
    exit 1
fi

echo "[ThereminQ] Initiating 6-Node Agentic Swarm with Auto-Restart..."

# Auto-Restart Wrapper Function
launch_node() {
    local VULKAN_ID=$1
    local NUMA_NODE=$2
    local PORT=$3
    local LOG_FILE=$4
    local CACHE_DIR=$5

    # Isolate the RADV Shader Cache for this specific Vulkan device thread
    export MESA_SHADER_CACHE_DIR="$CACHE_DIR"

    while true; do
        echo "[+] Booting Instance -> Physical Vulkan${VULKAN_ID} | NUMA Node ${NUMA_NODE} | Port ${PORT}"

        # Using >> to append to the log file so crash data isn't overwritten on restart
        numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" "$SERVER_BIN" \
            -m "$MODEL" \
            -c 196608 \
            -np 2 \
            -ngl 999 \
            -mg "${VULKAN_ID}" \
            --kv-unified \
            -fa on \
            --no-cache-idle-slots \
            --split-mode none \
            --cache-type-k q8_0 \
            --cache-type-v q4_0 \
            --no-mmap \
            --spec-type draft-mtp \
            --spec-draft-n-max 3 \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --tools all \
            --fit off >> "$LOG_FILE" 2>&1
        
        # If execution reaches this line, the server process has stopped
        echo "[!] Warning: Vulkan${VULKAN_ID} on Port ${PORT} stopped unexpectedly. Restarting in 5 seconds..."
        echo -e "\n[$(date)] -> Process stopped unexpectedly. Restarting in 5 seconds...\n" >> "$LOG_FILE"
        sleep 5
    done
}

# Loop through the topology array and ignite each instance using the wrapper
for node_config in "${SWARM[@]}"; do
    read -r VULKAN_ID NUMA_NODE PORT <<< "$node_config"
    LOG_TARGET="${LOG_DIR}/vulkan${VULKAN_ID}_port${PORT}.log"
    CACHE_TARGET="${LOG_DIR}/shader_cache_vk${VULKAN_ID}"
    
    # Create the isolated cache directory
    mkdir -p "$CACHE_TARGET"
    
    # Launch the auto-restart wrapper function in the background
    launch_node "$VULKAN_ID" "$NUMA_NODE" "$PORT" "$LOG_TARGET" "$CACHE_TARGET" &
    
    # Stagger the boot sequence by 8 seconds to prevent PCIe DMA saturation
    echo "[ThereminQ] Pausing 8 seconds to allow Vulkan graph initialization for node ${VULKAN_ID}..."
    sleep 8
done

echo "=============================================================================="
echo "[ThereminQ] Swarm boot sequence active."
echo "[ThereminQ] View individual initialization and crash logs in: $LOG_DIR"
echo "[ThereminQ] Press [Ctrl+C] to gracefully terminate all instances."
echo "=============================================================================="

# Keep the script alive to hold the background loops
wait
