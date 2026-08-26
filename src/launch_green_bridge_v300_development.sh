#!/usr/bin/env bash
# Formal GREEN v3.0.0 development launcher. Confirmation remains sealed.
set -euo pipefail

PREPARE_ROOT="${1:?usage: launch_green_bridge_v300_development.sh PREPARE_ROOT}"
MODE="${2:-initial}"
[[ "$MODE" == "initial" || "$MODE" == "integrity-finalization-recovery" ]] || {
  echo "invalid development launch mode" >&2
  exit 2
}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTION_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
[[ -z "$(git -C "$PROJECT_DIR" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "dirty worktree at v3 development launch" >&2
  exit 2
}

PREPARE_ROOT="$(readlink -m "$PREPARE_ROOT")"
[[ "$PREPARE_ROOT" == /mnt/sdb/* ]] || { echo "prepare root escaped /mnt/sdb" >&2; exit 2; }
RUN_ROOT="$(dirname "$(dirname "$PREPARE_ROOT")")"
WORKER_ROOT="$RUN_ROOT/development_workers_${EXECUTION_COMMIT}"
LOG_ROOT="$RUN_ROOT/logs/development_${EXECUTION_COMMIT}"
TMP_ROOT="/mnt/sdb/ccj/tmp/green_bridge_v300_development_${EXECUTION_COMMIT}"
for path in "$RUN_ROOT" "$WORKER_ROOT" "$LOG_ROOT" "$TMP_ROOT"; do
  resolved="$(readlink -m "$path")"
  [[ "$resolved" == /mnt/sdb/* ]] || { echo "development path escaped /mnt/sdb: $resolved" >&2; exit 2; }
done
[[ -f "$PREPARE_ROOT/prepare_result.json" ]] || { echo "formal prepare result missing" >&2; exit 2; }
[[ ! -e "$WORKER_ROOT" ]] || { echo "formal development worker root exists" >&2; exit 2; }
[[ ! -e "$PREPARE_ROOT/dev_result.json" ]] || { echo "formal development already exists" >&2; exit 2; }

export GREEN_BASE=/mnt/sdb/ccj
export GREEN_RUNTIME_ROOT="$RUN_ROOT"
export HF_HOME=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface
export HF_HUB_CACHE=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface/transformers
export TORCH_HOME=/mnt/sdb/ccj/cache/torch
export XDG_CACHE_HOME=/mnt/sdb/ccj/cache
export PIP_CACHE_DIR=/mnt/sdb/ccj/cache/pip
export TMPDIR="$TMP_ROOT"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=20260805 TOKENIZERS_PARALLELISM=false
export GREEN_V300_FINITE_MODE=float64_response_only
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
PYTHON=/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
[[ -x "$PYTHON" ]] || { echo "frozen Python missing" >&2; exit 2; }
[[ -d "$HF_HUB_CACHE" ]] || { echo "frozen Hugging Face cache missing" >&2; exit 2; }
mkdir -p "$WORKER_ROOT" "$LOG_ROOT" "$TMP_ROOT"

cd "$PROJECT_DIR"
"$PYTHON" -m pip check
"$PYTHON" src/test_green_bridge_v300_combined.py 2>&1 | tee "$LOG_ROOT/combined_contract.log"
grep -q "Ran 272 tests" "$LOG_ROOT/combined_contract.log"
grep -q '^OK$' "$LOG_ROOT/combined_contract.log"
! grep -qi 'skipped=' "$LOG_ROOT/combined_contract.log"

"$PYTHON" - "$PREPARE_ROOT" "$EXECUTION_COMMIT" "$MODE" "$RUN_ROOT" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
mode = sys.argv[3]
run_root = pathlib.Path(sys.argv[4])
result = json.load(open(root / "prepare_result.json", encoding="utf-8"))
ledger = json.load(open(root / "run_ledger.json", encoding="utf-8"))
if result.get("verdict") != "PREPARE_PASS":
    raise SystemExit("PREPARE_PASS_REQUIRED")
if ledger.get("confirmation_started") or (root / "dev_result.json").exists():
    raise SystemExit("DEVELOPMENT_OR_CONFIRMATION_ALREADY_TERMINAL")
if mode == "initial" and ledger.get("development_started"):
    raise SystemExit("DEVELOPMENT_PHASE_STATE_INVALID")
if mode == "integrity-finalization-recovery":
    if not ledger.get("development_started") or ledger.get("development_completed"):
        raise SystemExit("DEVELOPMENT_RECOVERY_STATE_INVALID")
    if ledger.get("development_execution_commit") != "2ef281794d08933ec686e624f2b01252b3b8ace1":
        raise SystemExit("DEVELOPMENT_INCIDENT_COMMIT_MISMATCH")
    old_logs = run_root / "logs/development_2ef281794d08933ec686e624f2b01252b3b8ace1"
    logs = sorted(old_logs.glob("worker_*.log"))
    if len(logs) != 8 or any(
        "DEVELOPMENT_ACTIVE_MODEL_INTEGRITY_FAILURE" not in path.read_text(encoding="utf-8")
        for path in logs
    ):
        raise SystemExit("DEVELOPMENT_INCIDENT_EVIDENCE_MISMATCH")
hash_file = root / ("sha256sums.txt" if mode == "initial" else "prepare_sha256sums.txt")
listed = {}
for line in hash_file.read_text(encoding="utf-8").splitlines():
    digest, name = line.split("  ", 1)
    listed[name] = digest
for name, digest in listed.items():
    if mode != "initial" and name == "run_ledger.json":
        continue
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"PREPARE_HASH_FAILURE:{name}")
if mode == "initial":
    (root / "prepare_sha256sums.txt").write_bytes((root / "sha256sums.txt").read_bytes())
ledger.update({
    "development_started": True,
    "development_completed": False,
    "development_execution_commit": commit,
    "development_authorization_id": "CODEX-GREEN-V300-DEVELOPMENT-v1-20260826",
    "confirmation_started": False,
})
if mode == "integrity-finalization-recovery":
    ledger["development_execution_incident"] = {
        "incident_commit": "2ef281794d08933ec686e624f2b01252b3b8ace1",
        "failure": "integrity verdict read before context-manager exit",
        "scientific_result_artifacts_written": False,
        "threshold_or_metric_output_observed": False,
        "recovery_commit": commit,
        "recovery_changes_scientific_protocol": False,
    }
tmp = root / ".run_ledger.development.tmp"
tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, root / "run_ledger.json")
PY

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  worker_dir="$WORKER_ROOT/worker_$(printf '%02d' "$gpu")"
  mkdir -p "$worker_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" src/green_bridge_v300_multigpu_worker.py \
      --phase development --role transport \
      --worker-index "$gpu" --worker-count 8 --physical-gpu "$gpu" \
      --prepare-root "$PREPARE_ROOT" --output "$worker_dir/transport"
    "$PYTHON" src/green_bridge_v300_multigpu_worker.py \
      --phase development --role joint \
      --worker-index "$gpu" --worker-count 8 --physical-gpu "$gpu" \
      --prepare-root "$PREPARE_ROOT" --output "$worker_dir/joint"
  ) >"$LOG_ROOT/worker_$(printf '%02d' "$gpu").log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || {
  echo "DEVELOPMENT_WORKER_FAILURE" >&2
  for log in "$LOG_ROOT"/worker_*.log; do tail -n 80 "$log" >&2; done
  exit 3
}

"$PYTHON" src/green_bridge_v300_development_merge.py \
  --worker-root "$WORKER_ROOT" --output-root "$PREPARE_ROOT" \
  2>&1 | tee "$LOG_ROOT/development_merge.log"

"$PYTHON" - "$PREPARE_ROOT" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
ledger = json.load(open(root / "run_ledger.json", encoding="utf-8"))
result = json.load(open(root / "dev_result.json", encoding="utf-8"))
ledger.update({
    "development_completed": True,
    "development_verdict": result["verdict"],
    "confirmation_started": False,
    "confirmation_authorized": False,
})
tmp = root / ".run_ledger.development.tmp"
tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, root / "run_ledger.json")
paths = sorted(p for p in root.iterdir() if p.is_file() and p.name != "sha256sums.txt")
(root / "sha256sums.txt").write_text(
    "\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in paths) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, sort_keys=True))
PY
