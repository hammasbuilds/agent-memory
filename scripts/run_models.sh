#!/usr/bin/env bash
# Run the whole model arm end to end: dense/hybrid retrieval, LLM fact extraction,
# answer accuracy. Checks free RAM, free GPU memory and that Ollama has both models
# before starting. Every call is cached (data/model_cache.sqlite), so rerunning resumes.
#
#   scripts/run_models.sh --dry-run        # job list and call counts; touches nothing
#   scripts/run_models.sh                  # the full run
#   scripts/run_models.sh --stage answer --per-type 20
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-8}"
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-11000}"   # qwen2.5:14b q4 needs ~9-10 GB

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    exec uv run python scripts/run_model_arm.py "$@"
  fi
done

free_ram_gb() {
  if command -v powershell >/dev/null 2>&1; then
    powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r'
  else
    awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo
  fi
}

ram=$(free_ram_gb)
if (( ram < MIN_FREE_RAM_GB )); then
  echo "only ${ram} GB RAM free (< ${MIN_FREE_RAM_GB}); not starting" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d '\r ')
  if (( vram < MIN_FREE_VRAM_MB )); then
    echo "only ${vram} MB GPU memory free (< ${MIN_FREE_VRAM_MB}); is another job training?" >&2
    exit 1
  fi
else
  echo "warning: nvidia-smi not found; running without a GPU check" >&2
fi

tags=$(curl -sf "$OLLAMA_URL/api/tags") || { echo "Ollama not reachable at $OLLAMA_URL" >&2; exit 1; }
for model in qwen2.5:14b-instruct nomic-embed-text; do
  if ! grep -q "\"$model" <<<"$tags"; then
    echo "model $model is not pulled: run 'ollama pull $model'" >&2
    exit 1
  fi
done

echo "RAM ${ram} GB free, Ollama at $OLLAMA_URL has both models; starting"
uv run python scripts/run_model_arm.py --url "$OLLAMA_URL" "$@"
