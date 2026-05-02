#!/usr/bin/env bash
# Encode a short low-res preview MP4 for each entry in a Cave search-mode JSONL,
# so a human can scroll through the results and pick out visually interesting rules
# for full-quality re-encoding.
#
# The Cave video encoder already supports JSONL ingestion + --encode-all per-entry
# batching. This script is just a thin wrapper that:
#   - finds the JSONL (defaulting to most recent _viable.jsonl in data/search/)
#   - optionally sorts/limits entries before encoding (so big searches stay tractable)
#   - invokes Cave.exe with sensible preview defaults (640x360 @ 20fps, 5 sec each)
#
# Usage:
#   bash scripts/preview_search_results.sh \
#     [--json <path>] [--shape cube|ed] [--grid <N>] [--out <folder>] \
#     [--limit <N>] [--sort-by ticks|births|final-alive|peak-alive|random]
#
# Defaults:
#   --shape cube  --grid 64  --out videos/previews/<timestamp>/
#   --sort-by ticks  (no --limit means encode every entry)

set -euo pipefail

JSON_PATH=""
SHAPE="cube"
GRID=64
OUT=""
LIMIT=""
SORT_BY="ticks"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--json)     JSON_PATH="$2"; shift 2 ;;
		--shape)    SHAPE="$2"; shift 2 ;;
		--grid)     GRID="$2"; shift 2 ;;
		--out)      OUT="$2"; shift 2 ;;
		--limit)    LIMIT="$2"; shift 2 ;;
		--sort-by)  SORT_BY="$2"; shift 2 ;;
		-h|--help)
			sed -n '2,21p' "$0"
			exit 0
			;;
		*)
			echo "unknown arg: $1" >&2
			exit 1
			;;
	esac
done

# Resolve script dir so the script works from any cwd; resolve all paths against repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default JSON: most recently modified *_viable.jsonl under data/search/
if [[ -z "$JSON_PATH" ]]; then
	JSON_PATH="$(find data/search -name '*_viable.jsonl' -type f -printf '%T@ %p\n' 2>/dev/null \
		| sort -nr | head -1 | cut -d' ' -f2- || true)"
	if [[ -z "$JSON_PATH" ]]; then
		echo "error: no _viable.jsonl found under data/search/. Pass --json <path>." >&2
		exit 1
	fi
	echo "auto-selected JSON: $JSON_PATH"
fi

if [[ ! -f "$JSON_PATH" ]]; then
	echo "error: JSON file not found: $JSON_PATH" >&2
	exit 1
fi

# Default output folder: videos/previews/<timestamp>/
if [[ -z "$OUT" ]]; then
	OUT="videos/previews/$(date +%Y%m%d_%H%M%S)/"
fi
mkdir -p "$OUT"

# Build the input JSONL we'll actually feed to Cave.exe. If sort-by/limit are
# specified, write a filtered+sorted copy to a temp file inside the output folder.
# Uses Python (not jq) for the sort step since jq isn't always available on Windows.
INPUT_JSONL="$JSON_PATH"
if [[ "$SORT_BY" != "ticks" || -n "$LIMIT" ]]; then
	INPUT_JSONL="$OUT/_input.jsonl"

	case "$SORT_BY" in
		ticks)       SORT_KEY="ticksSurvived" ;;
		births)      SORT_KEY="totalBirths" ;;
		final-alive) SORT_KEY="finalAliveCount" ;;
		peak-alive)  SORT_KEY="maxAliveCount" ;;
		random)      SORT_KEY="" ;;  # python random.shuffle handles this
		*)
			echo "error: unknown --sort-by: $SORT_BY (use ticks|births|final-alive|peak-alive|random)" >&2
			exit 1
			;;
	esac

	# Resolve to an absolute Windows path so python sees it correctly when running
	# via Git-bash (/c/Users/... vs C:/Users/...).
	if command -v cygpath >/dev/null 2>&1; then
		PY_INPUT="$(cygpath -w "$JSON_PATH")"
		PY_OUTPUT="$(cygpath -w "$INPUT_JSONL")"
	else
		PY_INPUT="$JSON_PATH"
		PY_OUTPUT="$INPUT_JSONL"
	fi

	python -c "
import json, sys, random
key = '$SORT_KEY'
limit = ${LIMIT:-0}
with open(r'$PY_INPUT', 'r') as f:
    entries = [json.loads(line) for line in f if line.strip()]
if key:
    entries.sort(key=lambda e: e.get(key, 0), reverse=True)
else:
    random.shuffle(entries)
if limit > 0:
    entries = entries[:limit]
with open(r'$PY_OUTPUT', 'w') as f:
    for e in entries:
        f.write(json.dumps(e) + '\n')
"
	echo "filtered/sorted input: $INPUT_JSONL ($(wc -l < "$INPUT_JSONL") entries)"
fi

ENTRY_COUNT=$(wc -l < "$INPUT_JSONL")
echo "encoding $ENTRY_COUNT preview(s) → $OUT"
echo "  shape=$SHAPE grid=${GRID}^3 res=640x360 @ 20fps × 100 ticks (5 sec each)"

# Find the Cave.exe — prefer Debug, fall back to Release.
if [[ -x "build/Debug/Cave.exe" ]]; then
	CAVE_EXE="build/Debug/Cave.exe"
elif [[ -x "build/Release/Cave.exe" ]]; then
	CAVE_EXE="build/Release/Cave.exe"
else
	echo "error: no Cave.exe found in build/Debug/ or build/Release/" >&2
	exit 1
fi

"$CAVE_EXE" \
	--mode video \
	--json "$INPUT_JSONL" \
	--encode-all \
	--shape "$SHAPE" \
	--grid "$GRID" \
	--width 640 --height 360 \
	--fps 20 --duration 100 \
	--dual-gpu --headless \
	--orbit-speed 0.005 \
	--output "$OUT"

echo "done. previews in $OUT"
