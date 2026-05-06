#!/usr/bin/env bash
# encode_batch_parallel.sh — shard a permutation JSONL and run N parallel Cave.exe video encodes.
#
# Usage:
#   tools/encode_batch_parallel.sh \
#       --json data/search/.../random500_for_video.jsonl \
#       --procs 4 \
#       --gpus 0,1 \
#       --output data/videos/random500_par4 \
#       -- --grid 49 --spawn 25 --spawn-mode filled --width 640 --height 360 --fps 30 --duration 150
#
# Everything after `--` is forwarded to each Cave.exe invocation.

set -euo pipefail

INPUT_JSONL=""
# N=2 is the empirical sweet spot at 49^3/360p with the full optimization stack
# (shader cache + --keep-h264-only + --headless): 39s for 500 videos. With per-video
# time slashed, per-proc startup overhead and encoder-queue contention dominate at N>=4.
NUM_PROCS=2
GPUS="0,1"
OUTPUT_BASE=""
EXE="${EXE:-./build/Release/Cave.exe}"
KEEP_SHARDS=0

# --- parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json) INPUT_JSONL="$2"; shift 2 ;;
        --procs) NUM_PROCS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --output) OUTPUT_BASE="$2"; shift 2 ;;
        --exe) EXE="$2"; shift 2 ;;
        --keep-shards) KEEP_SHARDS=1; shift ;;
        --) shift; PASSTHROUGH=("$@"); break ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
PASSTHROUGH=("${PASSTHROUGH[@]:-}")

if [[ -z "$INPUT_JSONL" || -z "$OUTPUT_BASE" ]]; then
    echo "Required: --json <input.jsonl> --output <base_dir>" >&2
    exit 1
fi
if [[ ! -f "$INPUT_JSONL" ]]; then
    echo "Input not found: $INPUT_JSONL" >&2
    exit 1
fi
if [[ ! -x "$EXE" ]]; then
    echo "Cave.exe not found or not executable: $EXE" >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"

mkdir -p "$OUTPUT_BASE"
SHARD_DIR="$OUTPUT_BASE/_shards"
LOG_DIR="$OUTPUT_BASE/_logs"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

# --- shard the input round-robin -----------------------------------------
# awk splits each non-empty line into shard_(NR % NUM_PROCS).jsonl
total=$(grep -c . "$INPUT_JSONL" || true)
echo "[driver] Input: $INPUT_JSONL ($total permutations)"
echo "[driver] Sharding into $NUM_PROCS shards (round-robin)..."
for (( k=0; k<NUM_PROCS; k++ )); do : > "$SHARD_DIR/shard_${k}.jsonl"; done
awk -v N="$NUM_PROCS" -v dir="$SHARD_DIR" '
    NF { idx = (NR - 1) % N; print >> (dir "/shard_" idx ".jsonl") }
' "$INPUT_JSONL"

for (( k=0; k<NUM_PROCS; k++ )); do
    cnt=$(grep -c . "$SHARD_DIR/shard_${k}.jsonl" || true)
    printf '[driver]   shard_%d: %s perms\n' "$k" "$cnt"
done

# --- launch N procs ------------------------------------------------------
START_TIME=$(date +%s)
PIDS=()
for (( k=0; k<NUM_PROCS; k++ )); do
    GPU_INDEX="${GPU_ARR[$(( k % ${#GPU_ARR[@]} ))]}"
    SHARD="$SHARD_DIR/shard_${k}.jsonl"
    OUT="$OUTPUT_BASE/shard_${k}"
    LOG="$LOG_DIR/shard_${k}.log"
    mkdir -p "$OUT"
    echo "[driver] Launching shard $k (GPU $GPU_INDEX) → $OUT"
    # Trailing slash forces Cave.exe to treat --output as a folder rather than a prefix.
    "$EXE" \
        --mode video \
        --json "$SHARD" --encode-all \
        --gpu "$GPU_INDEX" \
        --output "$OUT/" \
        "${PASSTHROUGH[@]}" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# --- wait + report -------------------------------------------------------
echo "[driver] Waiting for $NUM_PROCS shards (PIDs: ${PIDS[*]})..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[driver] PID $pid exited non-zero" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

# Count outputs — Cave.exe may write to subdirs (with trailing-slash --output) or
# as path-prefixed siblings (no slash). Find all .mp4s under OUTPUT_BASE recursively
# and exclude any older files by mtime > $START_TIME.
TOTAL_OUT=$(find "$OUTPUT_BASE" -type f -name '*.mp4' -newermt "@$START_TIME" 2>/dev/null | wc -l || echo 0)

echo "[driver] === Done ==="
echo "[driver] Wall time: ${ELAPSED}s"
echo "[driver] Videos produced: $TOTAL_OUT (input was $total)"
echo "[driver] Failed shards: $FAILED"
[[ -n "${total:-}" && "$ELAPSED" -gt 0 && "$total" -gt 0 ]] && \
    echo "[driver] Throughput: $(awk -v t="$total" -v s="$ELAPSED" 'BEGIN{printf "%.1f videos/sec, %.0f ms/video", t/s, s*1000/t}')"

if [[ "$KEEP_SHARDS" -eq 0 ]]; then
    rm -rf "$SHARD_DIR"
fi

exit "$FAILED"
