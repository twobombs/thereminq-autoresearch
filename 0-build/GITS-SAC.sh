#!/bin/bash
# this script is used when we need an in-container LLM to assist vscode to execute ThereminQuantumOPS commands
set -o pipefail

MODEL_DIR="/thereminq-autoresearch/0-build"
MODEL_FILE="Qwen3.8-9B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF/resolve/main/${MODEL_FILE}"

cd "$MODEL_DIR" || { echo "cannot cd to $MODEL_DIR"; exit 1; }

# --- download only if missing ---
if [ -s "$MODEL_FILE" ]; then
    echo "model already present: $(du -h "$MODEL_FILE" | cut -f1)"
else
    echo "downloading $MODEL_FILE ..."
    # -c resumes a partial .part; only promote to the real name on success
    if wget -c -O "${MODEL_FILE}.part" "$MODEL_URL"; then
        mv "${MODEL_FILE}.part" "$MODEL_FILE"
    else
        echo "download failed, partial kept at ${MODEL_FILE}.part"
        exit 1
    fi
fi

# --- supervisor loop ---
trap 'echo "shutting down"; exit 0' INT TERM

BACKOFF=2
while true; do
    START=$(date +%s)

    /llama-vulkan/build/bin/llama-server \
        -m "./${MODEL_FILE}" \
        -c 163840 --no-cache-idle-slots -np 1 -ngl 999 \
        --device Vulkan0 --kv-unified -fa on --split-mode none \
        --cache-type-k q8_0 --cache-type-v q4_0 --no-mmap \
        --spec-type draft-mtp --spec-draft-n-max 3 \
        --host 0.0.0.0 --port 8033 --tools all --fit off --jinja

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
