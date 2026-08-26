#!/usr/bin/env bash
# Recompute only the v3 development joint role after correcting the frozen
# recoverable-set contraction.  The original terminal merge is archived and
# confirmation remains sealed.
set -euo pipefail

PREPARE_ROOT="${1:?usage: launch_green_bridge_v300_joint_bound_corrigendum.sh PREPARE_ROOT}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTION_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
ORIGINAL_EXECUTION_COMMIT="0ad916b26f4bb225c474795460fac7a775a2c7e4"
ORIGINAL_MERGE_COMMIT="2f8e1cf6b061a554e7afc584589ccf5529cff08d"
[[ -z "$(git -C "$PROJECT_DIR" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "dirty worktree at v3 joint-bound corrigendum launch" >&2
  exit 2
}

PREPARE_ROOT="$(readlink -m "$PREPARE_ROOT")"
[[ "$PREPARE_ROOT" == /mnt/sdb/* ]] || { echo "prepare root escaped /mnt/sdb" >&2; exit 2; }
RUN_ROOT="$(dirname "$(dirname "$PREPARE_ROOT")")"
OLD_WORKER_ROOT="$RUN_ROOT/development_workers_${ORIGINAL_EXECUTION_COMMIT}"
WORKER_ROOT="$RUN_ROOT/development_workers_${EXECUTION_COMMIT}_joint_bound_corrigendum"
LOG_ROOT="$RUN_ROOT/logs/development_${EXECUTION_COMMIT}_joint_bound_corrigendum"
ARCHIVE_ROOT="$RUN_ROOT/development_initial_merge_${ORIGINAL_MERGE_COMMIT}_POSTER_ONLY"
TMP_ROOT="/mnt/sdb/ccj/tmp/green_bridge_v300_joint_bound_corrigendum_${EXECUTION_COMMIT}"
for path in "$RUN_ROOT" "$OLD_WORKER_ROOT" "$WORKER_ROOT" "$LOG_ROOT" "$ARCHIVE_ROOT" "$TMP_ROOT"; do
  resolved="$(readlink -m "$path")"
  [[ "$resolved" == /mnt/sdb/* ]] || { echo "corrigendum path escaped /mnt/sdb: $resolved" >&2; exit 2; }
done
[[ -d "$OLD_WORKER_ROOT" ]] || { echo "original worker root missing" >&2; exit 2; }
[[ ! -e "$WORKER_ROOT" ]] || { echo "corrigendum worker root exists" >&2; exit 2; }
[[ ! -e "$ARCHIVE_ROOT" ]] || { echo "initial terminal archive exists" >&2; exit 2; }

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
mkdir -p "$WORKER_ROOT" "$LOG_ROOT" "$TMP_ROOT"

cd "$PROJECT_DIR"
"$PYTHON" -m pip check
"$PYTHON" src/test_green_bridge_v300_combined.py 2>&1 | tee "$LOG_ROOT/combined_contract.log"
grep -q "Ran 272 tests" "$LOG_ROOT/combined_contract.log"
grep -q '^OK$' "$LOG_ROOT/combined_contract.log"
! grep -qi 'skipped=' "$LOG_ROOT/combined_contract.log"

# Validate and independently archive the original terminal evidence before any
# official result path is replaced.
"$PYTHON" - "$PREPARE_ROOT" "$ARCHIVE_ROOT" "$ORIGINAL_EXECUTION_COMMIT" "$ORIGINAL_MERGE_COMMIT" <<'PY'
import hashlib, json, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
execution_commit = sys.argv[3]
merge_commit = sys.argv[4]
result = json.load(open(root / "dev_result.json", encoding="utf-8"))
ledger = json.load(open(root / "run_ledger.json", encoding="utf-8"))
if result.get("verdict") != "POSTER_ONLY":
    raise SystemExit("CORRIGENDUM_REQUIRES_ORIGINAL_POSTER_ONLY")
if result.get("confirmation_started") or result.get("confirmation_authorized"):
    raise SystemExit("CONFIRMATION_MUST_REMAIN_SEALED")
if not ledger.get("development_completed") or ledger.get("confirmation_started"):
    raise SystemExit("DEVELOPMENT_LEDGER_STATE_MISMATCH")
if ledger.get("development_execution_commit") != execution_commit:
    raise SystemExit("ORIGINAL_EXECUTION_COMMIT_MISMATCH")
if ledger.get("development_merge_commit") != merge_commit:
    raise SystemExit("ORIGINAL_MERGE_COMMIT_MISMATCH")
listed = {}
for line in (root / "sha256sums.txt").read_text(encoding="utf-8").splitlines():
    digest, name = line.split("  ", 1)
    listed[name] = digest
for name, digest in listed.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"ORIGINAL_TERMINAL_HASH_FAILURE:{name}")
names = (
    "dev_transport_scores.parquet", "dev_joint_targets.parquet",
    "dev_cells.json", "dev_result.json", "frozen_analysis.json",
    "run_ledger.json", "sha256sums.txt",
)
archive.mkdir(parents=False, exist_ok=False)
for name in names:
    shutil.copy2(root / name, archive / name)
(archive / "archive_manifest.json").write_text(json.dumps({
    "schema_version": "green-bridge-v3.0.0-initial-development-archive-v1",
    "verdict": "POSTER_ONLY",
    "development_execution_commit": execution_commit,
    "development_merge_commit": merge_commit,
    "confirmation_started": False,
    "reason_archived": "pre-corrigendum recoverable joint-bound contraction and radius-stability denominator",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(archive / "archive_sha256sums.txt").write_text(
    "\n".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(archive.iterdir()) if path.name != "archive_sha256sums.txt"
    ) + "\n", encoding="utf-8",
)
PY

# Reuse only immutable, hash-checked transport result files.  Joint outputs are
# recomputed from scratch under the corrigendum commit.
for gpu in 0 1 2 3 4 5 6 7; do
  old_transport="$OLD_WORKER_ROOT/worker_$(printf '%02d' "$gpu")/transport"
  new_transport="$WORKER_ROOT/worker_$(printf '%02d' "$gpu")/transport"
  mkdir -p "$new_transport"
  for name in worker_result.json transport_rows.parquet joint_rows.parquet record_rows.parquet; do
    [[ -f "$old_transport/$name" ]] || { echo "old transport file missing: $gpu/$name" >&2; exit 2; }
    cp -al "$old_transport/$name" "$new_transport/$name"
  done
done

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  worker_dir="$WORKER_ROOT/worker_$(printf '%02d' "$gpu")"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
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
  echo "JOINT_BOUND_CORRIGENDUM_WORKER_FAILURE" >&2
  for log in "$LOG_ROOT"/worker_*.log; do tail -n 80 "$log" >&2; done
  exit 3
}

"$PYTHON" src/green_bridge_v300_development_merge.py \
  --worker-root "$WORKER_ROOT" --output-root "$PREPARE_ROOT" \
  2>&1 | tee "$LOG_ROOT/development_merge.log"

"$PYTHON" - "$PREPARE_ROOT" "$ARCHIVE_ROOT" "$EXECUTION_COMMIT" "$ORIGINAL_MERGE_COMMIT" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
commit = sys.argv[3]
original_merge = sys.argv[4]
ledger = json.load(open(root / "run_ledger.json", encoding="utf-8"))
result = json.load(open(root / "dev_result.json", encoding="utf-8"))
ledger.update({
    "development_completed": True,
    "development_verdict": result["verdict"],
    "development_joint_bound_corrigendum_commit": commit,
    "development_postcorrigendum_merge_commit": commit,
    "development_initial_terminal_archive": str(archive),
    "development_initial_merge_commit": original_merge,
    "development_corrigendum_scope": {
        "joint_role_recomputed": True,
        "transport_role_reused_unchanged": True,
        "recoverable_joint_bound_formula_corrected": True,
        "coarse_fine_stability_definition_restored_from_v2": True,
        "detectability_definition_or_values_changed": False,
    },
    "confirmation_started": False,
    "confirmation_authorized": False,
})
tmp = root / ".run_ledger.corrigendum.tmp"
tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, root / "run_ledger.json")
paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "sha256sums.txt")
(root / "sha256sums.txt").write_text(
    "\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, sort_keys=True))
PY

