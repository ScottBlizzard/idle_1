#!/usr/bin/env bash
set -euo pipefail

# Follow-on development orchestration. This waits for the prediction/Grant
# supervisor, ingests its typed batch receipts, and then runs independent A/B
# replay processes. It never launches an endpoint or confirmation job.

readonly REPO=/mnt/sdb/ccj/iclr_1_runs/idle_1_green_v400_postfreeze_validation_20260828
readonly PYTHON=/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
readonly MODEL_MANIFEST="$REPO/analysis/GREEN_V400_FORMAL_PREPARE_ARTIFACTS_20260826/model_manifest.json"
readonly AUTHORIZATION="$REPO/configs/green_v400_development_authorization_20260829.json"
readonly PYTHON_PATH="$REPO/src:$REPO:/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages:/mnt/sdb/ccj/green_v400_baseline_runtime/site-packages"
readonly FIRST_SUPERVISOR=/mnt/sdb/ccj/iclr_1_runs/green_v400_development_supervisor_20260829_v1
readonly FOLLOWON_RUN=/mnt/sdb/ccj/iclr_1_runs/green_v400_replay_followon_supervisor_20260829_v1

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

wait_for_first_supervisor() {
  while [[ ! -f "$FIRST_SUPERVISOR/status.txt" ]]; do
    if [[ ! -f "$FIRST_SUPERVISOR/supervisor.pid" ]]; then
      log "ERROR first supervisor pid file is missing"
      return 1
    fi
    local pid
    pid=$(<"$FIRST_SUPERVISOR/supervisor.pid")
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR first supervisor exited before COMPLETE"
      return 1
    fi
    log "waiting for prediction/Grant supervisor"
    sleep 60
  done
  if [[ $(<"$FIRST_SUPERVISOR/status.txt") != COMPLETE ]]; then
    log "ERROR first supervisor status is not COMPLETE"
    return 1
  fi
}

ingest_batches() {
  local plan=$1
  local prediction_root=$2
  local grant_root=$3
  local output_directory=$4
  if [[ -f "$output_directory/phase_ledger.json" ]]; then
    log "using existing batch ledger: $output_directory/phase_ledger.json"
    return 0
  fi
  env PYTHONPATH="$PYTHON_PATH" \
    "$PYTHON" "$REPO/analysis/green_v400_batch_ledger_ingest.py" \
    --plan "$plan" \
    --phase development \
    --prediction-root "$prediction_root" \
    --grant-root "$grant_root" \
    --output-directory "$output_directory"
}

run_replay_shard() {
  local plan=$1
  local parent=$2
  local universe=$3
  local registry=$4
  local endpoint_directions=$5
  local replay_root=$6
  local shard=$7
  local gpu=$((shard + 4))
  local output_directory="$replay_root/shard_$shard"
  local temporary_directory="$replay_root/tmp_gpu_$gpu"
  mkdir -p "$output_directory" "$temporary_directory" "$replay_root/pycache"
  mapfile -t jobs < <(
    "$PYTHON" - "$plan" "$shard" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
shard = int(sys.argv[2])
for ordinal, job in enumerate(plan["queues"]["endpoint_numerical_replay"]):
    if ordinal % 4 == shard:
        print(job["job_id"])
PY
  )
  local job_id replay_id output_path
  for job_id in "${jobs[@]}"; do
    for replay_id in A B; do
      output_path="$output_directory/${job_id}_${replay_id}.json"
      if [[ -f "$output_path" ]]; then
        log "resume skip replay shard=$shard job=$job_id replay=$replay_id"
        continue
      fi
      env \
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
        PYTHONPYCACHEPREFIX="$replay_root/pycache" \
        PYTHONPATH="$PYTHON_PATH" \
        "$PYTHON" -u "$REPO/analysis/green_v400_formal_worker.py" \
        --mode replay \
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
        --replay-id "$replay_id"
    done
  done
  printf '%s\n' COMPLETE > "$replay_root/shard_${shard}.status"
}

launch_replay_fleet() {
  local plan=$1
  local parent=$2
  local universe=$3
  local registry=$4
  local endpoint_directions=$5
  local replay_root=$6
  mkdir -p "$replay_root"
  local shard gpu pid
  for shard in 0 1 2 3; do
    gpu=$((shard + 4))
    run_replay_shard \
      "$plan" "$parent" "$universe" "$registry" \
      "$endpoint_directions" "$replay_root" "$shard" \
      > "$replay_root/replay_shard_${shard}_gpu_${gpu}.log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$replay_root/replay_shard_${shard}_gpu_${gpu}.pid"
    log "launched replay shard=$shard gpu=$gpu pid=$pid"
  done
}

wait_for_replay_fleet() {
  local replay_root=$1
  while true; do
    local complete=0
    local shard gpu pid_file pid
    for shard in 0 1 2 3; do
      if [[ -f "$replay_root/shard_${shard}.status" ]] \
        && [[ $(<"$replay_root/shard_${shard}.status") == COMPLETE ]]; then
        complete=$((complete + 1))
        continue
      fi
      gpu=$((shard + 4))
      pid_file="$replay_root/replay_shard_${shard}_gpu_${gpu}.pid"
      if [[ ! -f "$pid_file" ]]; then
        log "ERROR missing replay pid file: $pid_file"
        return 1
      fi
      pid=$(<"$pid_file")
      if ! kill -0 "$pid" 2>/dev/null; then
        log "ERROR replay shard $shard exited before COMPLETE"
        return 1
      fi
    done
    log "replay fleet progress: $complete/4 completed shards"
    if [[ $complete -eq 4 ]]; then
      return 0
    fi
    sleep 30
  done
}

assemble_replays() {
  local plan=$1
  local ledger=$2
  local replay_root=$3
  local output_directory=$4
  env PYTHONPATH="$PYTHON_PATH" \
    "$PYTHON" "$REPO/analysis/green_v400_replay_receipt_assembler.py" \
    --plan "$plan" \
    --phase-ledger "$ledger" \
    --replay-root "$replay_root" \
    --output-directory "$output_directory"
}

run_protocol_replay() {
  local label=$1
  local plan=$2
  local parent=$3
  local prepare=$4
  local universe=$5
  local prediction_root=$6
  local grant_root=$7
  local ledger_root=$8
  local replay_root=$9
  local receipt_root=${10}
  log "starting $label batch ledger ingestion"
  ingest_batches "$plan" "$prediction_root" "$grant_root" "$ledger_root"
  log "starting $label numerical replay fleet"
  launch_replay_fleet \
    "$plan" "$parent" "$universe" "$prepare/direction_registry.json" \
    "$prepare/directions/endpoint_directions.npy" "$replay_root"
  wait_for_replay_fleet "$replay_root"
  log "assembling $label numerical replay receipts"
  assemble_replays \
    "$plan" "$ledger_root/phase_ledger.json" "$replay_root" "$receipt_root"
  log "$label numerical replay complete and validated"
}

mkdir -p "$FOLLOWON_RUN"
wait_for_first_supervisor

run_protocol_replay \
  IOI \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_development_plan_20260829_v1/development_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_execution_plan_20260829_v8/sealed_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_prepare_20260829_v4 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_silent_failure_prepare_20260829_v4/ioi_untouched_universe.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_prediction_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_grant_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_development_batch_ledger_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_replay_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_replay_receipts_20260829_v1

run_protocol_replay \
  GT \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_development_plan_20260829_v1/development_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_execution_plan_20260829_v8/sealed_execution_plan.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_prepare_20260829_v3 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_prepare_20260829_v3/untouched_universe.json \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_prediction_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_grant_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_development_batch_ledger_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_replay_development_20260829_v1 \
  /mnt/sdb/ccj/iclr_1_runs/green_v400_gt_replay_receipts_20260829_v1

printf '%s\n' COMPLETE > "$FOLLOWON_RUN/status.txt"
log "all development numerical replay fleets and receipt assemblies complete"
