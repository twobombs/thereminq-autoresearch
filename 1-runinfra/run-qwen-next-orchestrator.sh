#!/bin/bash
# run-qwen-next-orchestrate.sh - one instance per V340 card (two dies each),
# then one generator against each simultaneously.
# Overrides: DEVICES="0,1 2,3" NCMOE=44 CTX=32768 ./run-qwen-next-orchestrate.sh
set -u

DEVICES=${DEVICES:-"0,1 2,3 4,5"}
BASE_PORT=${BASE_PORT:-9931}
STAGGER=${STAGGER:-20}
GEN_DIR=${GEN_DIR:-../2-startcore}
PROMPT=${PROMPT:-../8-workdir/testprompt.txt}
RESULTS=${RESULTS:-$PWD/testrun-$(date +%Y%m%d_%H%M%S)}

export NCMOE=${NCMOE:-32}
export CTX=${CTX:-262144}
export THREADS=${THREADS:-28}
export LLM_MODEL=${LLM_MODEL:-Qwen3.8-Flash-Next-UD-IQ4_XS}
export OPENAI_API_KEY=sk-local

mkdir -p "$RESULTS"

PROMPT_ABS=$(cd "$(dirname "$PROMPT")" && pwd)/$(basename "$PROMPT")
GEN_ABS=$(cd "$GEN_DIR" && pwd)

vmtouch -t ../0-build/Qwen3.8-Flash-Next-UD-IQ4_XS-*.gguf

trap 'echo "stopping"; kill $(jobs -p) 2>/dev/null' EXIT INT TERM

PORTS=""
i=0
for dev in $DEVICES; do
  port=$((BASE_PORT + i))
  PORTS="$PORTS $port"
  echo "launch: dies $dev -> port $port"
  GGML_VK_VISIBLE_DEVICES=$dev PORT=$port ./run-qwen-next.sh \
    > "$RESULTS/server-${port}.log" 2>&1 &
  i=$((i + 1))
  sleep "$STAGGER"
done

echo "waiting for health"
for port in $PORTS; do
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 && break
    sleep 5
  done
  echo "  port $port ready"
done

echo "starting generators at $(date +%T)"
cd "$GEN_ABS" || exit 1

GENPIDS=""
for port in $PORTS; do
  (
    start=$(date +%s)
    OPENAI_API_BASE="http://localhost:${port}/v1" \
      python3 0-generate-macrotask.py -f "$PROMPT_ABS" \
        -d "$RESULTS/raw-${port}" -c articles \
        > "$RESULTS/gen-${port}.log" 2>&1
    echo "port $port done in $(( $(date +%s) - start ))s" \
      | tee -a "$RESULTS/summary.txt"
  ) &
  GENPIDS="$GENPIDS $!"
done

wait $GENPIDS
echo "--- all generators finished at $(date +%T) ---"
cat "$RESULTS/summary.txt"
echo "servers still running; ctrl-c to stop"
wait
