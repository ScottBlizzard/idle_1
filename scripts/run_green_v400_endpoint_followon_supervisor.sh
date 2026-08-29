#!/usr/bin/env bash
set -euo pipefail

# Development endpoints only. This waits for both numerical-replay assemblies,
# prepares typed authorizations, and evaluates the frozen endpoint queue. It
# cannot launch confirmation because the activated plans keep it locked.

readonly REPO=/mnt/sdb/ccj/iclr_1_runs/idle_1_green_v400_postfreeze_validation_20260828
readonly PYTHON=/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
readonly MODEL_MANIFEST="$REPO/analysis/GREEN_V400_FORMAL_PREPARE_ARTIFACTS_20260826/model_manifest.json"
readonly AUTHORIZATION="$REPO/configs/green_v400_development_authorization_20260829.json"
readonly PYTHON_PATH="$REPO/src:$REPO:/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages:/mnt/sdb/ccj/green_v400_baseline_runtime/site-packages"
readonly REPLAY_SUPERVISOR=/mnt/sdb/ccj/iclr_1_runs/green_v400_replay_followon_supervisor_20260829_v1
readonly ENDPOINT_SUPERVISOR=/mnt/sdb/ccj/iclr_1_runs/green_v400_endpoint_followon_supervisor_20260829_v1

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

acquire_supervisor_lock() {
  mkdir -p "$ENDPOINT_SUPERVISOR"
  exec 9> "$ENDPOINT_SUPERVISOR/supervisor.lock"
  if ! flock -n 9; then
    log "ERROR another endpoint follow-on supervisor or inherited shard is active"
    return 1
  fi
}

wait_for_gpu_memory() {
  local gpu=$1
  local required_free_mib=8192
  while true; do
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits --id="$gpu" | tr -d '[:space:]')
    if [[ $free_mib =~ ^[0-9]+$ ]] && (( free_mib >= required_free_mib )); then
      log "GPU $gpu has ${free_mib} MiB free; endpoint may start"
      return 0
    fi
    log "GPU $gpu has ${free_mib:-unknown} MiB free; waiting for ${required_free_mib} MiB"
    sleep 60
  done
}

wait_for_replay_supervisor() {
  while [[ ! -f "$REPLAY_SUPERVISOR/status.txt" ]]; do
    if [[ ! -f "$REPLAY_SUPERVISOR/supervisor.pid" ]]; then
      log "ERROR replay supervisor pid file is missing"
      return 1
    fi
    local pid
    pid=$(<"$REPLAY_SUPERVISOR/supervisor.pid")
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR replay supervisor exited before COMPLETE"
      return 1
    fi
    log "waiting for replay supervisor"
    sleep 60
  done
  if [[ $(<"$REPLAY_SUPERVISOR/status.txt") != COMPLETE ]]; then
    log "ERROR replay supervisor status is not COMPLETE"
    return 1
  fi
}

prepare_authorizations() {
  local plan=$1
  local ledger=$2
  local universe=$3
  local prediction_root=$4
  local grant_receipts=$5
  local replay_receipts=$6
  local output_directory=$7
  if [[ -f "$output_directory/prepare_summary.json" ]]; then
    log "using existing endpoint authorizations: $output_directory"
    return 0
  fi
  env PYTHONPATH="$PYTHON_PATH" \
    "$PYTHON" "$REPO/analysis/green_v400_endpoint_authorization_prepare.py" \
    --plan "$plan" \
    --phase development \
    --phase-ledger "$ledger" \
    --universe "$universe" \
    --prediction-root "$prediction_root" \
    --grant-receipt-directory "$grant_receipts" \
    --replay-receipt-directory "$replay_receipts" \
    --output-directory "$output_directory"
}

run_endpoint_shard() {
  local plan=$1
  local parent=$2
  local universe=$3
  local registry=$4
  local endpoint_directions=$5
  local authorization_root=$6
  local endpoint_root=$7
  local shard=$8
  local gpu=$((shard + 4))
  local output_directory="$endpoint_root/shard_$shard"
  local temporary_directory="$endpoint_root/tmp_gpu_$gpu"
  local retry_directory="$endpoint_root/retry_logs"
  mkdir -p "$output_directory" "$temporary_directory" "$endpoint_root/pycache" \
    "$retry_directory"
  mapfile -t jobs < <(
    "$PYTHON" - "$plan" "$shard" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
shard = int(sys.argv[2])
for ordinal, job in enumerate(plan["queues"]["development_endpoint"]):
    if ordinal % 4 == shard:
        print(job["job_id"])
PY
  )
  local job_id output_path
  for job_id in "${jobs[@]}"; do
    output_path="$output_directory/${job_id}.json"
    if [[ -f "$output_path" ]]; then
      log "resume skip endpoint shard=$shard job=$job_id"
      continue
    fi
    while [[ ! -f "$output_path" ]]; do
      wait_for_gpu_memory "$gpu"
      local prior_attempts attempt attempt_log exit_code
      prior_attempts=$(find "$retry_directory" -maxdepth 1 -type f \
        -name "${job_id}_attempt_*.log" | wc -l)
      attempt=$((prior_attempts + 1))
      attempt_log="$retry_directory/${job_id}_attempt_${attempt}.log"
      if env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        BLIS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        PYTHONHASHSEED=0 \
        TOKENIZERS_PARALLELISM=false \
        HF_HOME=/mnt/sdb/ccj/iclr_1_runs/green_bridge_v136_runtime/huggingface \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TMPDIR="$temporary_directory" \
        TEMP="$temporary_directory" \
        TMP="$temporary_directory" \
        PYTHONPYCACHEPREFIX="$endpoint_root/pycache" \
        PYTHONPATH="$PYTHON_PATH" \
        "$PYTHON" -u "$REPO/analysis/green_v400_formal_worker.py" \
        --mode endpoint \
        --plan "$plan" \
        --parent-plan "$parent" \
        --development-authorization "$AUTHORIZATION" \
        --universe "$universe" \
        --model-manifest "$MODEL_MANIFEST" \
        --direction-registry "$registry" \
        --direction-payload "$endpoint_directions" \
        --job-id "$job_id" \
        --output "$output_path" \
        --device cuda:0 \
        --prediction-commitment \
          "$authorization_root/prediction_commitments/${job_id}.json" \
        --endpoint-authorization-receipt \
          "$authorization_root/endpoint_authorizations/${job_id}.json" \
          > "$attempt_log" 2>&1; then
        cat "$attempt_log"
        break
      else
        exit_code=$?
        cat "$attempt_log"
        if grep -q -E "OutOfMemoryError|CUDA out of memory" "$attempt_log"; then
          log "OOM retry shard=$shard job=$job_id attempt=$attempt"
          sleep 60
          continue
        fi
        log "ERROR non-OOM endpoint failure shard=$shard job=$job_id"
        return "$exit_code"
      fi
    done
  done
  printf '%s\n' COMPLETE > "$endpoint_root/shard_${shard}.status"
}

launch_endpoint_fleet() {
  local plan=$1
  local parent=$2
  local universe=$3
  local registry=$4
  local endpoint_directions=$5
  local authorization_root=$6
  local endpoint_root=$7
  mkdir -p "$endpoint_root"
  local shard gpu pid
  for shard in 0 1 2 3; do
    gpu=$((shard + 4))
    run_endpoint_shard \
      "$plan" "$parent" "$universe" "$registry" "$endpoint_directions" \
      "$authorization_root" "$endpoint_root" "$shard" \
      >> "$endpoint_root/endpoint_shard_${shard}_gpu_${gpu}.log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$endpoint_root/endpoint_shard_${shard}_gpu_${gpu}.pid"
    log "launched endpoint shard=$shard gpu=$gpu pid=$pid"
  done
}

wait_for_endpoint_fleet() {
  local endpoint_root=$1
  while true; do
    local complete=0
    local shard gpu pid_file pid
    for shard in 0 1 2 3; do
      if [[ -f "$endpoint_root/shard_${shard}.status" ]] \
        && [[ $(<"$endpoint_root/shard_${shard}.status") == COMPLETE ]]; then
        complete=$((complete + 1))
        continue
      fi
      gpu=$((shard + 4))
      pid_file="$endpoint_root/endpoint_shard_${shard}_gpu_${gpu}.pid"
      if [[ ! -f "$pid_file" ]]; then
        log "ERROR missing endpoint pid file: $pid_file"
        return 1
      fi
      pid=$(<"$pid_file")
      if ! kill -0 "$pid" 2>/dev/null; then
        log "ERROR endpoint shard $shard exited before COMPLETE"
        return 1
      fi
    done
    log "endpoint fleet progress: $complete/4 completed shards"
    if [[ $complete -eq 4 ]]; then
      return 0
    fi
    sleep 30
  done
}

run_protocol_endpoint() {
  local label=$1
  local plan=$2
  local parent=$3
  local prepare=$4
  local universe=$5
  local prediction_root=$6
  local batch_ledger_root=$7
  local replay_receipt_root=$8
  local authorization_root=$9
  local endpoint_root=${10}
  log "preparing $label endpoint authorizations"
  prepare_authorizations \
    "$plan" "$replay_receipt_root/phase_ledger.json" "$universe" \
    "$prediction_root" "$batch_ledger_root/grant_receipts" \
    "$replay_receipt_root/layer_receipts" "$authorization_root"
  log "starting $label development endpoint fleet"
  launch_endpoint_fleet \
    "$plan" "$parent" "$universe" "$prepare/direction_registry.json" \
    "$prepare/directions/endpoint_directions.npy" "$authorization_root" \
    "$endpoint_root"
  wait_for_endpoint_fleet "$endpoint_root"
  log "$label development endpoint fleet complete"
}

acquire_supervisor_lock
wait_for_replay_supervisor

run_protocol_endpoint \
  IOI \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_development_plan_20260829_v1/development_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_execution_plan_20260829_v8/sealed_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_prepare_20260829_v4 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_prepare_20260829_v4/ioi_untouched_universe.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_prediction_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_development_batch_ledger_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_replay_receipts_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_endpoint_authorizations_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_endpoint_development_20260829_v1

run_protocol_endpoint \
  GT \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_development_plan_20260829_v1/development_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_execution_plan_20260829_v8/sealed_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_prepare_20260829_v3 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_prepare_20260829_v3/untouched_universe.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_prediction_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_development_batch_ledger_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_replay_receipts_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_endpoint_authorizations_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_endpoint_development_20260829_v1

printf '%s\n' COMPLETE > "$ENDPOINT_SUPERVISOR/status.txt"
log "all development endpoint fleets complete"
