#!/usr/bin/env bash
set -euo pipefail

# Development endpoints only. This waits for both numerical-replay assemblies,
# prepares typed authorizations, and evaluates the frozen endpoint queue. Four
# GPU workers claim jobs dynamically while preserving the plan-defined shard
# output topology. It cannot launch confirmation because the activated plans
# keep it locked.

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
  local plan=$2
  local endpoint_root=$3
  local required_free_mib=8192
  while true; do
    if [[ $(remaining_endpoint_jobs "$plan" "$endpoint_root") -eq 0 ]]; then
      log "endpoint queue completed while GPU $gpu was waiting"
      return 2
    fi
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

remaining_endpoint_jobs() {
  local plan=$1
  local endpoint_root=$2
  "$PYTHON" - "$plan" "$endpoint_root" <<'PY'
import json
import pathlib
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
remaining = 0
for ordinal, job in enumerate(plan["queues"]["development_endpoint"]):
    if not (root / f"shard_{ordinal % 4}" / f"{job['job_id']}.json").is_file():
        remaining += 1
print(remaining)
PY
}

run_endpoint_worker() {
  local plan=$1
  local parent=$2
  local universe=$3
  local registry=$4
  local endpoint_directions=$5
  local authorization_root=$6
  local endpoint_root=$7
  local worker=$8
  local gpu=$((worker + 4))
  local temporary_directory="$endpoint_root/tmp_gpu_$gpu"
  local retry_directory="$endpoint_root/retry_logs"
  local lock_directory="$endpoint_root/job_locks"
  mkdir -p "$endpoint_root/shard_0" "$endpoint_root/shard_1" \
    "$endpoint_root/shard_2" "$endpoint_root/shard_3" \
    "$temporary_directory" "$endpoint_root/pycache" "$retry_directory" \
    "$lock_directory"
  mapfile -t job_entries < <(
    "$PYTHON" - "$plan" "$worker" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
worker = int(sys.argv[2])
jobs = plan["queues"]["development_endpoint"]
for base in range(0, len(jobs), 4):
    for delta in range(4):
        ordinal = base + ((worker + delta) % 4)
        if ordinal < len(jobs):
            print(f"{ordinal}\t{jobs[ordinal]['job_id']}")
PY
  )
  local remaining entry ordinal nominal_shard job_id output_path lock_path
  local job_lock_fd prior_attempts attempt attempt_log exit_code wait_status
  while true; do
    remaining=$(remaining_endpoint_jobs "$plan" "$endpoint_root")
    if [[ $remaining -eq 0 ]]; then
      printf '%s\n' COMPLETE > "$endpoint_root/shard_${worker}.status"
      log "endpoint worker=$worker gpu=$gpu observed the complete frozen queue"
      return 0
    fi
    log "endpoint worker=$worker gpu=$gpu queue remaining=$remaining"
    for entry in "${job_entries[@]}"; do
      ordinal=${entry%%$'\t'*}
      job_id=${entry#*$'\t'}
      nominal_shard=$((ordinal % 4))
      output_path="$endpoint_root/shard_${nominal_shard}/${job_id}.json"
      if [[ -f "$output_path" ]]; then
        continue
      fi
      wait_status=0
      wait_for_gpu_memory "$gpu" "$plan" "$endpoint_root" || wait_status=$?
      if [[ $wait_status -eq 2 ]]; then
        printf '%s\n' COMPLETE > "$endpoint_root/shard_${worker}.status"
        return 0
      elif [[ $wait_status -ne 0 ]]; then
        return "$wait_status"
      fi
      lock_path="$lock_directory/${job_id}.lock"
      exec {job_lock_fd}> "$lock_path"
      if ! flock -n "$job_lock_fd"; then
        exec {job_lock_fd}>&-
        continue
      fi
      if [[ -f "$output_path" ]]; then
        exec {job_lock_fd}>&-
        continue
      fi
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
        exec {job_lock_fd}>&-
      else
        exit_code=$?
        cat "$attempt_log"
        if grep -q -E "OutOfMemoryError|CUDA out of memory" "$attempt_log"; then
          log "OOM retry worker=$worker gpu=$gpu job=$job_id attempt=$attempt"
          exec {job_lock_fd}>&-
          sleep 60
          continue
        fi
        log "ERROR non-OOM endpoint failure worker=$worker gpu=$gpu job=$job_id"
        exec {job_lock_fd}>&-
        return "$exit_code"
      fi
    done
    sleep 5
  done
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
  local worker gpu pid
  for worker in 0 1 2 3; do
    gpu=$((worker + 4))
    run_endpoint_worker \
      "$plan" "$parent" "$universe" "$registry" "$endpoint_directions" \
      "$authorization_root" "$endpoint_root" "$worker" \
      >> "$endpoint_root/endpoint_shard_${worker}_gpu_${gpu}.log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$endpoint_root/endpoint_shard_${worker}_gpu_${gpu}.pid"
    log "launched dynamic endpoint worker=$worker gpu=$gpu pid=$pid"
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
