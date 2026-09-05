#!/bin/bash
# this script is used when we need an in-container LLM to assist vscode to execute ThereminQuantumOPS commands
set -o pipefail

MODEL_DIR="/thereminq-autoresearch/0-build"

# Qwen 3.8 nextgen Apex midrange model (split GGUF, 3 shards)
MODEL_BASE="Qwen3.8-Flash-Next-UD-IQ4_XS"
MODEL_PARTS=3
MODEL_URL_BASE="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS"

# llama.cpp only needs shard 1 on -m; it resolves the rest from the same directory
MODEL_FILE="$(printf '%s-%05d-of-%05d.gguf' "$MODEL_BASE" 1 "$MODEL_PARTS")"

cd "$MODEL_DIR" || { echo "cannot cd to $MODEL_DIR"; exit 1; }

# --- download each shard only if missing, in parallel ---
fetch_part() {
    local file="$1"
    if [ -s "$file" ]; then
        echo "model shard already present: $file ($(du -h "$file" | cut -f1))"
        return 0
    fi
    echo "downloading $file ..."
    # -c resumes a partial .part; only promote to the real name on success
    if wget -c -O "${file}.part" "${MODEL_URL_BASE}/${file}"; then
        mv "${file}.part" "$file"
    else
        echo "download failed, partial kept at ${file}.part"
        return 1
    fi
}

PIDS=()
FILES=()
for i in $(seq 1 "$MODEL_PARTS"); do
    f="$(printf '%s-%05d-of-%05d.gguf' "$MODEL_BASE" "$i" "$MODEL_PARTS")"
    FILES+=("$f")
    fetch_part "$f" &
    PIDS+=($!)
done

DL_FAIL=0
for idx in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$idx]}"; then
        echo "shard failed: ${FILES[$idx]}"
        DL_FAIL=1
    fi
done
[ "$DL_FAIL" -ne 0 ] && { echo "one or more shards missing, aborting"; exit 1; }

# --- thread sizing: fit -t / -tb to the CPUs this container can actually use ---
# ceilings preserve the hand-tuned host values; raise them on a bigger box
T_MAX=24
TB_MAX=48

cpu_quota() {
    local q p
    if [ -r /sys/fs/cgroup/cpu.max ]; then                    # cgroup v2
        read -r q p < /sys/fs/cgroup/cpu.max
        if [ "$q" != "max" ]; then echo $(( (q + p - 1) / p )); return; fi
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then     # cgroup v1
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "$q" -gt 0 ]; then echo $(( (q + p - 1) / p )); return; fi
    fi
    echo 0
}

CPUS=$(nproc)                       # already honours cpuset / taskset affinity
QUOTA=$(cpu_quota)                  # but not a CFS bandwidth cap
[ "$QUOTA" -gt 0 ] && [ "$QUOTA" -lt "$CPUS" ] && CPUS=$QUOTA

# -tb gets every logical CPU (batch/prompt is throughput-bound),
# -t gets physical cores only (generation gains nothing from SMT siblings)
if [ "$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)" = "1" ]; then
    GEN_THREADS=$(( CPUS / 2 ))
else
    GEN_THREADS=$CPUS
fi
BATCH_THREADS=$CPUS

[ "$GEN_THREADS" -lt 1 ] && GEN_THREADS=1
[ "$BATCH_THREADS" -lt 1 ] && BATCH_THREADS=1
[ "$GEN_THREADS" -gt "$T_MAX" ] && GEN_THREADS=$T_MAX
[ "$BATCH_THREADS" -gt "$TB_MAX" ] && BATCH_THREADS=$TB_MAX

echo "cpus available: ${CPUS} -> -t ${GEN_THREADS} -tb ${BATCH_THREADS}"

# --- supervisor loop ---
trap 'echo "shutting down"; exit 0' INT TERM

# main model on Vulkan1, MTP/draft on Vulkan0
export GGML_VK_VISIBLE_DEVICES=0,2

BACKOFF=2
while true; do
    START=$(date +%s)

    /llama-vulkan/build/bin/llama-server \
        -m "./${MODEL_FILE}" \
        --device Vulkan0 --device-draft Vulkan0 \
        -ngl 99 -ngld 99 \
        -ot "\.ffn_(gate|up|down)_exps\.=CPU" \
        -c 131072 -np 1 -fa on -ctk f16 -ctv f16 \
        --no-context-shift -b 2048 -ub 512 \
        -t "$GEN_THREADS" -tb "$BATCH_THREADS" --jinja --tools all \
        --host 0.0.0.0 --port 9931

    RC=$?
    RUNTIME=$(( $(date +%s) - START ))
    echo "llama-server exited rc=$RC after ${RUNTIME}s"

    # clean shutdown -> stop supervising
    [ "$RC" -eq 0 ] && break

    # reset backoff if it ran fine for a while, otherwise grow it
    if [ "$RUNTIME" -ge 60 ]; then
        BACKOFF=2
    else
        BACKOFF=$(( BACKOFF * 2 ))
        [ "$BACKOFF" -gt 60 ] && BACKOFF=60
    fi

    echo "restarting in ${BACKOFF}s ..."
    sleep "$BACKOFF"
done
