#!/usr/bin/env bash
# Read-only 8-GPU launcher for the GREEN v2.1 postmortem AD shards.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_ROOT="${1:-/mnt/sdb/ccj/iclr_1_runs/idle_1_green_bridge_v200_f99626f/outputs/green_bridge_v200}"
SCRATCH_ROOT="${2:-/mnt/sdb/ccj/iclr_1_postmortem/green_v21}"
PYTHON="/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python"
CACHE_ROOT="/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime"
RUNTIME_ROOT="/mnt/sdb/ccj/iclr_1_runs/green_bridge_v200_runtime"
SHARD_ROOT="$SCRATCH_ROOT/gpu_shards"
LOG_ROOT="$SCRATCH_ROOT/logs"

[[ -d "$FORMAL_ROOT" ]] || { echo "missing formal v2 root: $FORMAL_ROOT" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "missing frozen Python: $PYTHON" >&2; exit 2; }
[[ -d "$CACHE_ROOT/huggingface/hub" ]] || { echo "missing frozen Hugging Face cache" >&2; exit 2; }
for gpu in {0..7}; do
  [[ ! -e "$SHARD_ROOT/worker_$gpu" ]] || {
    echo "refusing to overwrite shard: $SHARD_ROOT/worker_$gpu" >&2
    exit 2
  }
done
mkdir -p "$SHARD_ROOT" "$LOG_ROOT" "$RUNTIME_ROOT/tmp"

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR="$RUNTIME_ROOT/pip" TORCH_HOME="$RUNTIME_ROOT/torch"
export TMPDIR="$RUNTIME_ROOT/tmp"
export GREEN_V136_PREDECESSOR_ROOT="/mnt/sdb/ccj/iclr_1_runs/idle_1_green_bridge_v136/outputs/green_bridge_v136"
export PYTHONHASHSEED=20260805 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR/src"

pids=()
for gpu in {0..7}; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON" "$PROJECT_DIR/src/green_bridge_v300_postmortem_worker.py" \
      --formal-v200-root "$FORMAL_ROOT" \
      --output "$SHARD_ROOT/worker_$gpu" \
      --worker-index "$gpu" --worker-count 8 --physical-gpu "$gpu"
  ) >"$LOG_ROOT/worker_$gpu.log" 2>&1 &
  pids+=("$!")
done

failed=0
for gpu in {0..7}; do
  if ! wait "${pids[$gpu]}"; then
    echo "worker $gpu failed; see $LOG_ROOT/worker_$gpu.log" >&2
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit 1
echo "all eight GREEN v2.1 postmortem workers completed"
