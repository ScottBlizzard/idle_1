#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
STORAGE_ROOT="/mnt/sdb/green_v400_formal_prepare_runtime"
OUTPUT_ROOT="/mnt/sdb/outputs/green_bridge_v400_formal_prepare"
VENV_ROOT="${STORAGE_ROOT}/venv"
LOG_ROOT="${STORAGE_ROOT}/logs"
TEMP_ROOT="${STORAGE_ROOT}/tmp"
CACHE_ROOT="${STORAGE_ROOT}/cache"
BINDING_PARENT="48182844a43d391439704f27aa26d513d33adaa0"

cd "${PROJECT_ROOT}"
test "$(git branch --show-current)" = "codex/green-v400-joint-witness-formal-prepare"
git merge-base --is-ancestor "${BINDING_PARENT}" HEAD
test -z "$(git status --porcelain=v1 --untracked-files=all)"
for path in "${STORAGE_ROOT}" "${OUTPUT_ROOT}" "${VENV_ROOT}" "${LOG_ROOT}" "${TEMP_ROOT}" "${CACHE_ROOT}"; do
  resolved="$(realpath -m "${path}")"
  case "${resolved}" in /mnt/sdb|/mnt/sdb/*) ;; *) exit 72 ;; esac
done
test ! -e "${OUTPUT_ROOT}" || test -z "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)"
mkdir -p "${LOG_ROOT}" "${TEMP_ROOT}" "${CACHE_ROOT}"

export TMPDIR="${TEMP_ROOT}"
export TEMP="${TEMP_ROOT}"
export TMP="${TEMP_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"
export PYTHONPATH="${PROJECT_ROOT}/src"
export GREEN_V400_DEVICE="${GREEN_V400_DEVICE:-cuda:0}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

if test ! -x "${VENV_ROOT}/bin/python"; then
  python3 -m venv --system-site-packages "${VENV_ROOT}"
fi
"${VENV_ROOT}/bin/python" -m pip install --disable-pip-version-check -r requirements/green_v400_validated_numerics.lock
"${VENV_ROOT}/bin/python" src/test_green_bridge_v300_combined.py 2>&1 | tee "${LOG_ROOT}/historical_regression.log"
"${VENV_ROOT}/bin/python" -m pytest -q -W error::DeprecationWarning \
  tests/test_green_bridge_v400_interval_core.py \
  tests/test_green_bridge_v400_interval_jet.py \
  tests/test_green_bridge_v400_transformer_ops.py \
  tests/test_green_bridge_v400_relational_graph.py \
  tests/test_green_bridge_v400_endpoint_certificate.py \
  tests/test_green_bridge_v400_repository_contract.py 2>&1 | tee "${LOG_ROOT}/theorem_barrier.log"
"${VENV_ROOT}/bin/python" src/green_bridge_v400_prepare.py \
  --config configs/green_bridge_v400_formal_prepare.json 2>&1 | tee "${LOG_ROOT}/formal_prepare.log"
"${VENV_ROOT}/bin/python" analysis/green_v400_formal_prepare_audit.py \
  --output-root "${OUTPUT_ROOT}" 2>&1 | tee "${LOG_ROOT}/independent_audit.log"
printf '%s\n' "STOP_AFTER_FORMAL_PREPARE_RETURN_TO_GPTPRO" | tee "${LOG_ROOT}/terminal_report.txt"
