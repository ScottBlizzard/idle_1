#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
STORAGE_ROOT="/mnt/sdb/ccj/green_v400_formal_prepare_runtime"
OUTPUT_ROOT="/mnt/sdb/ccj/outputs/green_bridge_v400_formal_prepare"
PACKAGE_ROOT="${STORAGE_ROOT}/site-packages"
PYTHON_BIN="${GREEN_V400_PYTHON:-/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python}"
LOG_ROOT="${STORAGE_ROOT}/logs"
TEMP_ROOT="${STORAGE_ROOT}/tmp"
CACHE_ROOT="${STORAGE_ROOT}/cache"
MODEL_CACHE_ROOT="${GREEN_V400_MODEL_CACHE:-/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface}"
BINDING_PARENT="48182844a43d391439704f27aa26d513d33adaa0"

cd "${PROJECT_ROOT}"
test "$(git branch --show-current)" = "codex/green-v400-joint-witness-formal-prepare"
git merge-base --is-ancestor "${BINDING_PARENT}" HEAD
test -z "$(git status --porcelain=v1 --untracked-files=all)"
for path in "${STORAGE_ROOT}" "${OUTPUT_ROOT}" "${PACKAGE_ROOT}" "${LOG_ROOT}" "${TEMP_ROOT}" "${CACHE_ROOT}" "${MODEL_CACHE_ROOT}"; do
  resolved="$(realpath -m "${path}")"
  case "${resolved}" in /mnt/sdb|/mnt/sdb/*) ;; *) exit 72 ;; esac
done
test ! -e "${OUTPUT_ROOT}" || test -z "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)"
mkdir -p "${LOG_ROOT}" "${TEMP_ROOT}" "${CACHE_ROOT}"

export TMPDIR="${TEMP_ROOT}"
export TEMP="${TEMP_ROOT}"
export TMP="${TEMP_ROOT}"
export HF_HOME="${MODEL_CACHE_ROOT}"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"
export PYTHONPATH="${PACKAGE_ROOT}:${PROJECT_ROOT}/src"
export GREEN_V400_DEVICE="${GREEN_V400_DEVICE:-cuda:0}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

test -x "${PYTHON_BIN}"
test -d "${MODEL_CACHE_ROOT}/hub/models--openai-community--gpt2"
mkdir -p "${PACKAGE_ROOT}"
"${PYTHON_BIN}" -m pip install --disable-pip-version-check --target "${PACKAGE_ROOT}" -r requirements/green_v400_validated_numerics.lock
"${PYTHON_BIN}" src/test_green_bridge_v300_combined.py 2>&1 | tee "${LOG_ROOT}/historical_regression.log"
"${PYTHON_BIN}" -m pytest -q -W error::DeprecationWarning \
  tests/test_green_bridge_v400_interval_core.py \
  tests/test_green_bridge_v400_interval_jet.py \
  tests/test_green_bridge_v400_transformer_ops.py \
  tests/test_green_bridge_v400_relational_graph.py \
  tests/test_green_bridge_v400_endpoint_certificate.py \
  tests/test_green_bridge_v400_repository_contract.py 2>&1 | tee "${LOG_ROOT}/theorem_barrier.log"
"${PYTHON_BIN}" src/green_bridge_v400_prepare.py \
  --config configs/green_bridge_v400_formal_prepare.json 2>&1 | tee "${LOG_ROOT}/formal_prepare.log"
"${PYTHON_BIN}" analysis/green_v400_formal_prepare_audit.py \
  --output-root "${OUTPUT_ROOT}" 2>&1 | tee "${LOG_ROOT}/independent_audit.log"
printf '%s\n' "STOP_AFTER_FORMAL_PREPARE_RETURN_TO_GPTPRO" | tee "${LOG_ROOT}/terminal_report.txt"
