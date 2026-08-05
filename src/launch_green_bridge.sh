#!/usr/bin/env bash
# Frozen GREEN bridge launcher. Usage: bash src/launch_green_bridge.sh [GPU_ID]
set -euo pipefail

GPU_ID="${1:-0}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="green_bridge_20260805"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "STOP 01_ENVIRONMENT: conda initialization not found" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11.13 pip
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade "pip==25.1.1"
python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$PROJECT_DIR/requirements-green-bridge.lock"
python -m pip install --no-deps --force-reinstall \
  "git+https://github.com/TransformerLensOrg/TransformerLens.git@4a4dc26c750475b29e6f54b362c2aab988702c9c"

cd "$PROJECT_DIR"
mkdir -p outputs/green_bridge
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONHASHSEED=20260805
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python src/test_green_bridge_contract.py \
  2>&1 | tee outputs/green_bridge/contract_test.log
python src/exp_green_bridge_gpt2.py --phase all --device cuda:0 \
  2>&1 | tee outputs/green_bridge/run.log
