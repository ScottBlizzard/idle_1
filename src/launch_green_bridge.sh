#!/usr/bin/env bash
# Frozen v1.3 launcher. Usage: bash src/launch_green_bridge.sh GPU_ID PHASE
set -euo pipefail

GPU_ID="${1:-4}"
PHASE="${2:-prepare}"
if [[ "$PHASE" != "prepare" && "$PHASE" != "development" && "$PHASE" != "confirmation" ]]; then
  echo "phase must be prepare, development, or confirmation" >&2
  exit 2
fi
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
if python - <<'PY'
from pathlib import Path
import hashlib
import transformer_lens

expected = {
    "HookedTransformer.py": "f80ee1ec42039a287a2b9366c75f98eec23ff33c6e941ffeee03f0374eb20af3",
    "HookedRootModule.py": "e7144971a973ec2d63bf7400db6443caba5d03f22f310f6789d52fa4a56ad245",
    "components/mlps/mlp.py": "615cb178d3ce65d8784af18dec86fbfe2b3957ddc02d3b99bdd2d45aa6759b32",
    "utilities/addmm.py": "f9e72f6a3d6c508814fa8e69918c20e1cb72cbc9ae7bcb1a1abb2476e246bc38",
}
root = Path(transformer_lens.__file__).resolve().parent
actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
raise SystemExit(0 if actual == expected else 1)
PY
then
  echo "TransformerLens source matches frozen commit 4a4dc26"
else
  python -m pip install --no-deps --force-reinstall \
    "git+https://github.com/TransformerLensOrg/TransformerLens.git@4a4dc26c750475b29e6f54b362c2aab988702c9c"
fi

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONHASHSEED=20260805
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TEST_LOG="/tmp/green_bridge_v13_contract_${PHASE}.log"
RUN_LOG="/tmp/green_bridge_v13_${PHASE}.log"
python src/test_green_bridge_contract.py \
  2>&1 | tee "$TEST_LOG"
python src/exp_green_bridge_gpt2.py --phase "$PHASE" --device cuda:0 \
  --output-root outputs/green_bridge 2>&1 | tee "$RUN_LOG"
