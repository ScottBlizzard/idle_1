#!/usr/bin/env bash
# Frozen GREEN v3.0.0 one-shot launcher: prepare only, physical GPU 4 only.
set -euo pipefail

GPU_ID="${1:-4}"
PHASE="${2:-prepare}"
[[ "$GPU_ID" == "4" ]] || { echo "prepare coordinator must use physical GPU 4" >&2; exit 2; }
[[ "$PHASE" == "prepare" ]] || {
  echo "UNAUTHORIZED_PHASE_REQUIRES_NEW_GPTPRO_DECISION" >&2
  exit 2
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTION_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
[[ -z "$(git -C "$PROJECT_DIR" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "dirty worktree at v3 launch" >&2
  exit 2
}

export GREEN_BASE=/mnt/sdb/ccj
export GREEN_RUNTIME_ROOT="/mnt/sdb/ccj/iclr_1_runs/green_bridge_v300_${EXECUTION_COMMIT}"
export HF_HOME=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface
export HF_HUB_CACHE=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface/transformers
export TORCH_HOME=/mnt/sdb/ccj/cache/torch
export XDG_CACHE_HOME=/mnt/sdb/ccj/cache
export PIP_CACHE_DIR=/mnt/sdb/ccj/cache/pip
export TMPDIR="/mnt/sdb/ccj/tmp/green_bridge_v300_${EXECUTION_COMMIT}"
export TEMP="$TMPDIR" TMP="$TMPDIR"
OUTPUT_ROOT="$GREEN_RUNTIME_ROOT/outputs/green_bridge_v300"
WORKER_LOG_ROOT="$GREEN_RUNTIME_ROOT/logs/workers"
COORDINATOR_LOG_ROOT="$GREEN_RUNTIME_ROOT/logs/coordinator"
ENDPOINT_ROOT="$GREEN_RUNTIME_ROOT/endpoint_ledgers"
PARQUET_ROOT="$GREEN_RUNTIME_ROOT/parquet_staging"

for path in "$GREEN_RUNTIME_ROOT" "$OUTPUT_ROOT" "$HF_HOME" "$TORCH_HOME" \
            "$TMPDIR" "$WORKER_LOG_ROOT" "$COORDINATOR_LOG_ROOT" \
            "$ENDPOINT_ROOT" "$PARQUET_ROOT"; do
  resolved="$(readlink -m "$path")"
  [[ "$resolved" == /mnt/sdb/* ]] || { echo "path escaped /mnt/sdb: $resolved" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "formal v3 root already exists" >&2; exit 2; }
[[ -d "$HF_HUB_CACHE" ]] || { echo "frozen Hugging Face cache missing" >&2; exit 2; }
mkdir -p "$GREEN_RUNTIME_ROOT" "$TMPDIR" "$WORKER_LOG_ROOT" \
         "$COORDINATOR_LOG_ROOT" "$ENDPOINT_ROOT" "$PARQUET_ROOT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONHASHSEED=20260805 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
PYTHON=/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
[[ -x "$PYTHON" ]] || { echo "frozen Python missing" >&2; exit 2; }

cd "$PROJECT_DIR"
"$PYTHON" -m pip check
TEST_LOG="$COORDINATOR_LOG_ROOT/green_v300_combined_272_tests.log"
"$PYTHON" src/test_green_bridge_v300_combined.py 2>&1 | tee "$TEST_LOG"
grep -q "Ran 272 tests" "$TEST_LOG"
grep -q '^OK$' "$TEST_LOG"
! grep -qi 'skipped=' "$TEST_LOG"

"$PYTHON" src/exp_green_bridge_v300.py \
  --phase prepare --device cuda:0 --output-root "$OUTPUT_ROOT"
