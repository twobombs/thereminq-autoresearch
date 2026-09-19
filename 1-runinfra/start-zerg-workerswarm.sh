#!/bin/bash

# ==============================================================================
# ThereminQ-HPC Agentic Swarm Orchestrator
# 6x Qwen 9B MTP | 100% VRAM-Resident Pipeline
# Container-first | Optional NUMA-to-PCIe Affinity | Auto-Restart
# Shader Cache Isolation
# ==============================================================================
#
# NUMA pinning is OFF by default (container-safe).
#   NUMA_PIN=off  -> run llama-server directly (default)
#   NUMA_PIN=on   -> wrap each node in numactl, if available
#
# Inside containers, prefer doing affinity at the runtime level instead:
#   docker run --cpuset-cpus=... --cpuset-mems=... --device /dev/dri ...
# ==============================================================================

# Configuration
MODEL="${MODEL:-../0-build/Qwen3.8-9B-Q4_K_M.gguf}"
SERVER_BIN="${SERVER_BIN:-../0-build/llama.cpp/build/bin/llama-server}"
LOG_DIR="${LOG_DIR:-./agent_logs}"
NUMA_PIN="${NUMA_PIN:-off}"
BOOT_STAGGER="${BOOT_STAGGER:-8}"

# Define the Swarm Topology: "Vulkan_ID  NUMA_Node  API_Port"
# NUMA_Node is only used when NUMA_PIN=on (kept here to document topology)
SWARM=(
  "0 0 8030"
  "1 0 8031"
  "2 1 8032"
  "3 1 8033"
  "4 6 8034"
  "5 6 8035"
)

# Prerequisite Checks (before traps, so failures keep a non-zero exit code)
if [ ! -f "$MODEL" ]; then
    echo "[!] Error: Model file not found at $MODEL"
    exit 1
fi

if [ ! -x "$SERVER_BIN" ]; then
    echo "[!] Error: llama-server executable not found or not executable at $SERVER_BIN"
    exit 1
fi

case "$NUMA_PIN" in
    on)
        if command -v numactl &> /dev/null; then
            echo "[ThereminQ] NUMA pinning: ON (numactl)"
        else
            echo "[!] Warning: NUMA_PIN=on but numactl not found -> running unpinned."
            NUMA_PIN=off
        fi
        ;;
    off)
        echo "[ThereminQ] NUMA pinning: OFF (container default)"
        ;;
    *)
        echo "[!] Error: NUMA_PIN must be 'on' or 'off' (got '$NUMA_PIN')"
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"

# Graceful Shutdown Sequence
shutdown_swarm() {
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

# Catch Ctrl+C / docker stop (SIGTERM) / exit
trap shutdown_swarm SIGINT SIGTERM EXIT

echo "[ThereminQ] Initiating ${#SWARM[@]}-Node Agentic Swarm with Auto-Restart..."

# Auto-Restart Wrapper Function
launch_node() {
    local VULKAN_ID=$1
    local NUMA_NODE=$2
    local PORT=$3
    local LOG_FILE=$4
    local CACHE_DIR=$5

    # Isolate the RADV Shader Cache for this specific Vulkan device
    export MESA_SHADER_CACHE_DIR="$CACHE_DIR"

    # Optional affinity prefix; empty array = direct launch
    local PREFIX=()
    local AFFINITY="unpinned"
    if [ "$NUMA_PIN" = "on" ]; then
        PREFIX=(numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}")
        AFFINITY="NUMA Node ${NUMA_NODE}"
    fi

    while true; do
        echo "[+] Booting Instance -> Physical Vulkan${VULKAN_ID} | ${AFFINITY} | Port ${PORT}"

        "${PREFIX[@]}" "$SERVER_BIN" \
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
            --spec-type draft-mtp \
            --spec-draft-n-max 3 \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --tools all \
            --fit off >> "$LOG_FILE" 2>&1

        echo "[!] Warning: Vulkan${VULKAN_ID} on Port ${PORT} stopped unexpectedly. Restarting in 5 seconds..."
        echo -e "\n[$(date)] -> Process stopped unexpectedly. Restarting in 5 seconds...\n" >> "$LOG_FILE"
        sleep 5
    done
}

for node_config in "${SWARM[@]}"; do
    read -r VULKAN_ID NUMA_NODE PORT <<< "$node_config"
    LOG_TARGET="${LOG_DIR}/vulkan${VULKAN_ID}_port${PORT}.log"
    CACHE_TARGET="${LOG_DIR}/shader_cache_vk${VULKAN_ID}"

    mkdir -p "$CACHE_TARGET"

    launch_node "$VULKAN_ID" "$NUMA_NODE" "$PORT" "$LOG_TARGET" "$CACHE_TARGET" &

    echo "[ThereminQ] Pausing ${BOOT_STAGGER}s to allow Vulkan graph initialization for node ${VULKAN_ID}..."
    sleep "$BOOT_STAGGER"
done

echo "=============================================================================="
echo "[ThereminQ] Swarm boot sequence active."
echo "[ThereminQ] View individual initialization and crash logs in: $LOG_DIR"
echo "[ThereminQ] Press [Ctrl+C] or 'docker stop' to gracefully terminate all instances."
echo "=============================================================================="

wait
