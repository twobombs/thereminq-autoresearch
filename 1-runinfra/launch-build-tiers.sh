#!/bin/bash
# ==============================================================================
# 1-runinfra/launch-tiers.sh
#
# ThereminQ build-machine inference tiers - Vulkan, role-resolved, VRAM-resident.
#
# Replaces vk-pick.sh + kvfit.py + launch2B-*.sh + launch9B-*.sh.
#
# Design constraints this encodes:
#   - Vulkan only. Cards get swapped, so nothing here may name a card, an
#     enumeration index, or a vendor.
#   - The box is specced 50/50 (RAM == total VRAM). Nothing may spill to host
#     memory, and the guards below fail loudly rather than run slowly.
#   - Context is derived from the VRAM the device actually reports at boot,
#     never baked in from whatever card used to be in the slot.
#
# Usage:
#   ./launch-tiers.sh list              enumerate Vulkan devices
#   ./launch-tiers.sh plan              resolve every tier, print the budget
#   ./launch-tiers.sh run small         foreground, one tier (systemd ExecStart)
#   ./launch-tiers.sh all               staggered background start of all tiers
#   ./launch-tiers.sh stop              tear down
#   ./launch-tiers.sh health            probe every tier endpoint + check RSS
#   ./launch-tiers.sh purge             drop all scratch state (do this after
#                                       any card swap - shader caches are
#                                       architecture-specific)
#
# All runtime state - shader caches, logs, SPIR-V temp - lives under
# 8-workdir/.runtime, alongside testprompt.txt. Nothing is written to /var.
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BUILD=${BUILD:-../0-build}
BIN=${BIN:-$BUILD/llama.cpp/build/bin/llama-server}
GGUF_PY=${GGUF_PY:-$BUILD/llama.cpp/gguf-py}
# All churning, disposable state lives under the workdir that already holds
# testprompt.txt - not /var. That directory is where the NVMe RAID is, it is
# removable as a unit, and nothing here is worth surviving a card swap. Shader
# caches in particular are architecture-specific and must not outlive the card
# that compiled them.
WORKDIR=${WORKDIR:-../8-workdir}
SCRATCH=${SCRATCH:-$WORKDIR/.runtime}
LOG_DIR=${LOG_DIR:-$SCRATCH/logs}
CACHE_ROOT=${CACHE_ROOT:-$SCRATCH/vkcache}
CACHE_MAX=${CACHE_MAX:-2147483648}      # 2 GiB ceiling per driver stack

# ------------------------------------------------------------------------------
# Tier table. Add a row, get a tier. Fields:
#   name  model  port  role  role_arg  batch  ubatch  mtp_mib  extra
#
# role is how the tier finds a card:
#   largest             - take the biggest device (reasoning tier)
#   smallest-fitting N  - smallest device with >= N MiB free, so the big card
#                         stays free for the tier that needs it
#   match STR           - substring match on device name, for pinned oddities
#
# mtp_mib is the VRAM reserved for the MTP head and its own draft KV. Set it to
# 0 on any tier where --spec-type is absent. The 400 below is a placeholder:
# measure it (see NOTES at the bottom) before you trust the derived ctx.
# ------------------------------------------------------------------------------
TIERS=(
  "small|Qwen3.8-2B-Q4_K_M.gguf|8036|smallest-fitting|2400|256|128|0|"
  "big|Qwen3.8-9B-Q4_K_M.gguf|8033|largest||512|512|400|--spec-type draft-mtp --spec-draft-n-max 3"
)

CTK=${CTK:-q8_0}
CTV=${CTV:-q4_0}
RESERVE=${RESERVE:-192}       # MiB held back for driver + fragmentation

die() { echo "[!] $*" >&2; exit 1; }
note() { echo "[*] $*" >&2; }

# The workdir is removable by design, so treat its absence as a hard error
# rather than silently recreating it somewhere slow. If the RAID is not
# mounted, a shader cache landing on the root filesystem is worse than a
# refusal - it is slow, invisible, and it survives the next card swap.
scratch_check() {
  [ -d "$WORKDIR" ] || die "workdir $WORKDIR not present - is the NVMe array mounted?"
  mkdir -p "$SCRATCH" 2>/dev/null || die "cannot create scratch under $WORKDIR"
  [ -w "$SCRATCH" ] || die "$SCRATCH is not writable"
  printf 'scratch state for launch-tiers.sh - safe to delete at any time\n' \
    > "$SCRATCH/.disposable"
}

# ------------------------------------------------------------------------------
# Device enumeration
#
# --list-devices lines look like:
#   Vulkan0: NVIDIA CMP 50HX (10240 MiB, 10102 MiB free)
# Emits: VulkanN|Name|FreeMiB
# ------------------------------------------------------------------------------
vk_devices() {
  "$BIN" --list-devices 2>/dev/null | sed -n \
    's/^[[:space:]]*\(Vulkan[0-9]\+\): \(.*\) (\([0-9]\+\) MiB, \([0-9]\+\) MiB free)$/\1|\2|\4/p'
}

# vk_pick <mode> [arg] -> sets VK_DEV VK_FREE VK_NAME VK_SLUG
vk_pick() {
  local mode=$1 arg=${2:-} devs pick
  devs=$(vk_devices) || true
  [ -n "$devs" ] || die "no Vulkan devices enumerated by $BIN"

  case "$mode" in
    largest)
      pick=$(echo "$devs" | sort -t'|' -k3 -nr | head -1) ;;
    smallest-fitting)
      pick=$(echo "$devs" | awk -F'|' -v n="$arg" '$3+0 >= n+0' \
             | sort -t'|' -k3 -n | head -1) ;;
    match)
      pick=$(echo "$devs" | grep -i -- "$arg" | head -1) ;;
    *) die "unknown role: $mode" ;;
  esac
  [ -n "${pick:-}" ] || die "no device satisfies role: $mode $arg"

  VK_DEV=${pick%%|*}
  VK_FREE=${pick##*|}
  VK_NAME=$(echo "$pick" | cut -d'|' -f2)
  VK_SLUG=$(echo "$VK_NAME" | tr -cs 'A-Za-z0-9' '-' | tr 'A-Z' 'a-z' | sed 's/-*$//')
}

# ------------------------------------------------------------------------------
# KV budget solver. Reads block_count / head_count_kv / key_length straight out
# of the GGUF so a model swap needs no edit here either.
#
# bytes-per-value by cache type, per llama.cpp block layout:
#   f16 2.0 | q8_0 34/32 | q5_1 24/32 | q5_0 22/32 | q4_1 20/32 | q4_0 18/32
# ------------------------------------------------------------------------------
kv_fit() {
  local model=$1 vram=$2 ub=$3 mtp=$4
  PYTHONPATH="$GGUF_PY:${PYTHONPATH:-}" python3 - "$model" "$vram" "$ub" "$mtp" \
      "$CTK" "$CTV" "$RESERVE" <<'PY'
import os, sys
model, vram, ub, mtp, ctk, ctv, reserve = sys.argv[1:8]
vram, ub, mtp, reserve = float(vram), int(ub), float(mtp), float(reserve)
BPV = {"f32":4.0,"f16":2.0,"bf16":2.0,"q8_0":34/32,"q5_1":24/32,
       "q5_0":22/32,"q4_1":20/32,"q4_0":18/32}
MIB = 1024*1024
try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("gguf-py not importable; set GGUF_PY to <llama.cpp>/gguf-py")
r = GGUFReader(model)
def get(*sfx):
    for name, f in r.fields.items():
        if any(name.endswith(s) for s in sfx):
            try: return int(f.parts[f.data[0]][0])
            except Exception: return None
    return None
layers  = get(".block_count")
kvh     = get(".attention.head_count_kv")
heads   = get(".attention.head_count")
n_embd  = get(".embedding_length")
klen    = get(".attention.key_length")
vlen    = get(".attention.value_length")
if klen is None and heads and n_embd:
    klen = vlen = n_embd // heads
vlen = vlen or klen
vocab = len(r.fields["tokenizer.ggml.tokens"].data) \
    if "tokenizer.ggml.tokens" in r.fields else 151936
if not all([layers, kvh, klen]):
    sys.exit("could not read KV geometry from %s" % model)
weights = os.path.getsize(model)/MIB
per_tok = layers * (kvh*klen*BPV[ctk] + kvh*vlen*BPV[ctv])
logits  = vocab*ub*4/MIB
graph   = max(64.0, ub*layers*kvh*klen*4/MIB*0.25)
budget  = vram - weights - logits - graph - reserve - mtp
ctx     = int(budget*MIB/per_tok) if budget > 0 else 0
ctx    -= ctx % 256
safe    = int(ctx*0.92); safe -= safe % 256
print("%d %d %.2f %.0f %.0f %.0f" %
      (safe, ctx, per_tok/1024, weights, logits, budget))
PY
}

# ------------------------------------------------------------------------------
# Per-tier resolution: role -> device -> ctx. Sets TIER_* for the caller.
# ------------------------------------------------------------------------------
resolve_tier() {
  local row name model port role rarg batch ub mtp extra
  row=$(printf '%s\n' "${TIERS[@]}" | grep "^$1|") \
    || die "no such tier: $1 (have: $(printf '%s\n' "${TIERS[@]}" | cut -d'|' -f1 | paste -sd,))"
  IFS='|' read -r name model port role rarg batch ub mtp extra <<<"$row"

  TIER_NAME=$name
  TIER_MODEL="$BUILD/$model"
  TIER_PORT=$port
  TIER_BATCH=$batch
  TIER_UB=$ub
  TIER_EXTRA=$extra
  [ -f "$TIER_MODEL" ] || die "$name: model not found at $TIER_MODEL"

  vk_pick "$role" "$rarg"
  TIER_DEV=$VK_DEV; TIER_VRAM=$VK_FREE; TIER_CARD=$VK_NAME; TIER_SLUG=$VK_SLUG

  read -r TIER_CTX TIER_CTX_MAX TIER_KVTOK TIER_W TIER_LOGITS TIER_BUDGET \
    < <(kv_fit "$TIER_MODEL" "$VK_FREE" "$ub" "$mtp")
  [ "${TIER_CTX:-0}" -gt 0 ] \
    || die "$name: $TIER_CARD has ${VK_FREE} MiB free, model needs more than that"
  TIER_CTX=${CTX_OVERRIDE:-$TIER_CTX}
}

# ------------------------------------------------------------------------------
# Guards. These exist because Vulkan will not fail for you.
#
# The Vulkan backend picks memory types by preference and can fall back from
# DEVICE_LOCAL to a host-visible heap. --fit off stops layer-shedding, not that
# fallback, so an over-budget allocation runs at PCIe speed instead of dying.
# On a 50/50 box that is the failure mode you least want to discover in Jenkins.
# ------------------------------------------------------------------------------
guard_env() {
  # If the checkout exposes a sysmem-fallback toggle, use it. The name has moved
  # between releases, so discover it rather than hardcoding one that may be gone.
  local var
  var=$(grep -rhos 'GGML_VK_[A-Z_]*SYSMEM[A-Z_]*' \
        "$BUILD/llama.cpp/ggml/src/ggml-vulkan/" 2>/dev/null | sort -u | head -1)
  if [ -n "$var" ]; then
    export "$var=0"
    note "sysmem fallback disabled via $var"
  else
    note "no sysmem-fallback toggle in this checkout - rely on MemoryMax= in the unit"
  fi

  # Pascal-class cards advertise shaderFloat16 while running it at 1/64 rate.
  # Opt out per device rather than globally.
  if [ "${VK_DISABLE_F16:-}" = "1" ]; then
    export GGML_VK_DISABLE_F16=1
    note "GGML_VK_DISABLE_F16=1 for $TIER_CARD"
  fi

  # Shader cache keyed on the card, so a swap invalidates instead of loading
  # SPIR-V compiled for the previous architecture. Both stacks, either vendor.
  export MESA_SHADER_CACHE_DIR="$CACHE_ROOT/vk-$TIER_SLUG"
  export MESA_SHADER_CACHE_MAX_SIZE="$((CACHE_MAX / 1073741824))G"
  export __GL_SHADER_DISK_CACHE_PATH="$CACHE_ROOT/vk-$TIER_SLUG"
  export __GL_SHADER_DISK_CACHE_SIZE="$CACHE_MAX"

  # SPIR-V compilation scratch goes to the same NVMe, not to /tmp. On a 50/50
  # box /tmp is usually tmpfs, and tmpfs is RAM - the one budget that must not
  # move. This is the same reason --no-mmap is staggered in cmd_all.
  export TMPDIR="$SCRATCH/tmp"
  mkdir -p "$MESA_SHADER_CACHE_DIR" "$LOG_DIR" "$TMPDIR"

  export GGML_VK_VISIBLE_DEVICES=${TIER_DEV#Vulkan}
}

guard_coopmat() {
  # Speculation and prompt processing both lean on cooperative matrix. If the
  # card in this slot lacks it, say so at boot, not via a Jenkins timeout.
  case "$TIER_EXTRA" in
    *draft-mtp*)
      vulkaninfo 2>/dev/null | grep -qi 'cooperative_matrix' \
        || note "warning: no VK_KHR_cooperative_matrix - MTP draft will run scalar shaders" ;;
  esac
}

# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------
cmd_list() {
  printf '%-9s %-34s %10s\n' DEVICE CARD "FREE MiB"
  vk_devices | while IFS='|' read -r d n f; do printf '%-9s %-34s %10s\n' "$d" "$n" "$f"; done
}

cmd_plan() {
  local t
  scratch_check
  for t in $(printf '%s\n' "${TIERS[@]}" | cut -d'|' -f1); do
    resolve_tier "$t"
    printf '%s\n' "--- tier: $TIER_NAME (port $TIER_PORT)"
    printf '  card      %s (%s, %s MiB free)\n' "$TIER_CARD" "$TIER_DEV" "$TIER_VRAM"
    printf '  model     %s (%s MiB)\n' "$(basename "$TIER_MODEL")" "$TIER_W"
    printf '  logits    %s MiB at -ub %s\n' "$TIER_LOGITS" "$TIER_UB"
    printf '  KV/token  %s KiB (%s K / %s V)\n' "$TIER_KVTOK" "$CTK" "$CTV"
    printf '  KV budget %s MiB\n' "$TIER_BUDGET"
    printf '  ctx       %s  (ceiling %s)\n' "$TIER_CTX" "$TIER_CTX_MAX"
  done
}

cmd_run() {
  scratch_check
  resolve_tier "$1"
  guard_env
  guard_coopmat
  note "$TIER_NAME -> $TIER_CARD ($TIER_DEV), ctx $TIER_CTX, port $TIER_PORT"
  # shellcheck disable=SC2086
  exec "$BIN" \
    -m "$TIER_MODEL" \
    -c "$TIER_CTX" \
    -np 1 --kv-unified --no-cache-idle-slots \
    -ngl 999 --device "$TIER_DEV" --split-mode none \
    -fa on --cache-type-k "$CTK" --cache-type-v "$CTV" \
    --no-mmap --fit off \
    -b "$TIER_BATCH" -ub "$TIER_UB" \
    --no-context-shift \
    $TIER_EXTRA \
    --host 0.0.0.0 --port "$TIER_PORT" \
    --tools all --jinja
}

cmd_all() {
  scratch_check
  # Staggered on purpose. --no-mmap allocates a full host-side copy during load;
  # two tiers loading at once on a 50/50 box with Jenkins resident is how you
  # meet the OOM killer. Each tier must be serving before the next one loads.
  mkdir -p "$LOG_DIR"
  local t port
  for t in $(printf '%s\n' "${TIERS[@]}" | cut -d'|' -f1); do
    port=$(printf '%s\n' "${TIERS[@]}" | grep "^$t|" | cut -d'|' -f3)
    note "starting tier $t"
    ( while true; do
        "$0" run "$t" >>"$LOG_DIR/$t.log" 2>&1
        note "tier $t exited, restarting in 10s"; sleep 10
      done ) &
    for _ in $(seq 1 120); do
      curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
      sleep 2
    done
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
      || die "tier $t did not become healthy - see $LOG_DIR/$t.log"
    note "tier $t healthy on :$port"
  done
  trap 'note "shutting down"; kill $(jobs -p) 2>/dev/null; pkill -f "$BIN"; exit 0' \
    SIGINT SIGTERM
  wait
}

cmd_health() {
  local t port pid rss
  for t in $(printf '%s\n' "${TIERS[@]}" | cut -d'|' -f1); do
    port=$(printf '%s\n' "${TIERS[@]}" | grep "^$t|" | cut -d'|' -f3)
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      # Backend-agnostic spill check: with full residency, RSS after warmup is
      # the loader's working set, not the weights. A multi-GB RSS means the
      # allocator took a host-visible heap and you are running over PCIe.
      pid=$(pgrep -f "port $port" | head -1 || true)
      rss=$( [ -n "$pid" ] && awk '/VmRSS/{print $2/1024" MiB"}' "/proc/$pid/status" || echo "?")
      printf '%-7s :%s  up    RSS %s\n' "$t" "$port" "$rss"
    else
      printf '%-7s :%s  DOWN\n' "$t" "$port"
    fi
  done
}

cmd_stop() { pkill -f "$BIN" 2>/dev/null && note "stopped" || note "nothing running"; }

# Everything under $SCRATCH is regenerable. Purge after a card swap so no tier
# loads SPIR-V compiled for an architecture that is no longer in the box.
cmd_purge() {
  [ -f "$SCRATCH/.disposable" ] || die "$SCRATCH is not a scratch dir I created"
  rm -rf "${SCRATCH:?}"
  note "purged $SCRATCH"
}

case "${1:-plan}" in
  list)   cmd_list ;;
  plan)   cmd_plan ;;
  run)    shift; cmd_run "${1:?tier name required}" ;;
  all)    cmd_all ;;
  health) cmd_health ;;
  stop)   cmd_stop ;;
  purge)  cmd_purge ;;
  *)      sed -n '2,30p' "$0"; exit 1 ;;
esac

# ==============================================================================
# NOTES
#
# Measuring mtp_mib: start the tier once with mtp set to 0 and CTX_OVERRIDE set
# low (say 8192), read the KV and model buffer sizes llama-server prints at
# load, subtract from the device total, and put the remainder in the table.
# Guessing here costs you either context or a failed start.
#
# Jenkins: run tiers as systemd units, not as Jenkins-managed processes - VRAM
# state should survive job churn. ExecStart=.../launch-tiers.sh run big, plus
# MemoryMax= sized to the loader working set so a host-visible spill cannot
# allocate. Label agents per tier and gate with lockable-resources, one lock per
# port: both tiers run -np 1, so two concurrent jobs queue inside llama-server
# where Jenkins cannot see the wait, and --no-cache-idle-slots means the second
# one also pays a full prompt reprocess.
# ==============================================================================
