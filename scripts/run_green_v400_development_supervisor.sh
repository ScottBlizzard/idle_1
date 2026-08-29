#!/usr/bin/env bash
set -euo pipefail

# Development-only orchestration. Scientific workers revalidate the activated
# plan, parent plan, authorization sidecar, source hashes, and frozen numerics.

readonly REPO=/mnt/sdb/ccj/iclr_1_runs/idle_1_green_v400_postfreeze_validation_20260828
readonly PYTHON=/home/ccj/miniconda3/envs/green_bridge_20260805/bin/python
readonly MODEL_MANIFEST="$REPO/analysis/GREEN_V400_FORMAL_PREPARE_ARTIFACTS_20260826/model_manifest.json"
readonly CAPTURE_SPEC="$REPO/configs/green_v400_grant_capture_spec.json"
readonly AUTHORIZATION="$REPO/configs/green_v400_development_authorization_20260829.json"
readonly PYTHON_PATH="$REPO/src:$REPO:/mnt/sdb/ccj/green_v400_formal_prepare_runtime/site-packages:/mnt/sdb/ccj/green_v400_baseline_runtime/site-packages"
readonly SUPERVISOR_RUN=/mnt/sdb/ccj/iclr_1_runs/green_v400_development_supervisor_20260829_v1

readonly IOI_PREDICTION_RUN=/mnt/sdb/ccj/iclr_1_runs/green_v400_ioi_prediction_development_20260829_v1
readonly GT_PLAN=/mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_development_plan_20260829_v1/development_execution_plan.json
readonly GT_PARENT=/mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_execution_plan_20260829_v8/sealed_execution_plan.json
readonly GT_PREPARE=/mnt/sdb/ccj/iclr_1_runs/green_v400_greater_than_prepare_20260829_v3
readonly GT_UNIVERSE="$GT_PREPARE/untouched_universe.json"
readonly GT_REGISTRY="$GT_PREPARE/direction_registry.json"
readonly GT_GREEN="$GT_PREPARE/directions/green_directions.npy"
readonly GT_GRANT_RUN=/mnt/sdb/ccj/iclr_1_runs/green_v400_gt_grant_development_20260829_v1
readonly GT_PREDICTION_RUN=/mnt/sdb/ccj/iclr_1_runs/green_v400_gt_prediction_development_20260829_v1

mkdir -p "$SUPERVISOR_RUN"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

completion_path() {
  local run_directory=$1
  local mode=$2
  local shard=$3
  printf '%s/shard_%s/_completion_%s_development_%02d_of_04.json' \
    "$run_directory" "$shard" "$mode" "$shard"
}

wait_for_fleet() {
  local run_directory=$1
  local mode=$2
  while true; do
    local complete=0
    local shard
    for shard in 0 1 2 3; do
      if [[ -f "$(completion_path "$run_directory" "$mode" "$shard")" ]]; then
        complete=$((complete + 1))
        continue
      fi
      local pid_file="$run_directory/${mode}_shard_${shard}_gpu_$((shard + 4)).pid"
      if [[ ! -f "$pid_file" ]]; then
        log "ERROR missing pid file: $pid_file"
        return 1
      fi
      local pid
      pid=$(<"$pid_file")
      if ! kill -0 "$pid" 2>/dev/null; then
        log "ERROR shard $shard exited without a completion receipt"
        return 1
      fi
    done
    log "$mode fleet progress: $complete/4 completion receipts"
    if [[ $complete -eq 4 ]]; then
      return 0
    fi
    sleep 30
  done
}

launch_gt_fleet() {
  local mode=$1
  local run_directory=$2
  mkdir -p "$run_directory" "$run_directory/pycache"
  local shard
  for shard in 0 1 2 3; do
    local gpu=$((shard + 4))
    local output_directory="$run_directory/shard_$shard"
    local temporary_directory="$run_directory/tmp_gpu_$gpu"
    local log_path="$run_directory/${mode}_shard_${shard}_gpu_${gpu}.log"
    local pid_path="$run_directory/${mode}_shard_${shard}_gpu_${gpu}.pid"
    mkdir -p "$output_directory" "$temporary_directory"
    local mode_arguments=()
    if [[ $mode == grant ]]; then
      mode_arguments=(--grant-capture-spec "$CAPTURE_SPEC")
    elif [[ $mode == prediction ]]; then
      mode_arguments=(
        --direction-registry "$GT_REGISTRY"
        --direction-payload "$GT_GREEN"
        --integrated-gradients-steps 65
        --ms-hvp-segments 8
        --response-batch-chunk-size 16
      )
    else
      log "ERROR invalid fleet mode: $mode"
      return 1
    fi
    nohup env \
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
      PYTHONPYCACHEPREFIX="$run_directory/pycache" \
      PYTHONPATH="$PYTHON_PATH" \
      "$PYTHON" -u "$REPO/analysis/green_v400_formal_batch_worker.py" \
      --mode "$mode" \
      --phase development \
      --plan "$GT_PLAN" \
      --parent-plan "$GT_PARENT" \
      --development-authorization "$AUTHORIZATION" \
      --universe "$GT_UNIVERSE" \
      --model-manifest "$MODEL_MANIFEST" \
      "${mode_arguments[@]}" \
      --shard-index "$shard" \
      --shard-count 4 \
      --output-directory "$output_directory" \
      --device cuda:0 \
      --resume >"$log_path" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$pid_path"
    log "launched GT $mode shard=$shard gpu=$gpu pid=$pid"
  done
}

log "waiting for the already-running IOI development prediction fleet"
wait_for_fleet "$IOI_PREDICTION_RUN" prediction
log "IOI development prediction fleet complete"

launch_gt_fleet grant "$GT_GRANT_RUN"
wait_for_fleet "$GT_GRANT_RUN" grant
log "GT development Grant fleet complete"

launch_gt_fleet prediction "$GT_PREDICTION_RUN"
wait_for_fleet "$GT_PREDICTION_RUN" prediction
log "GT development prediction fleet complete"
printf '%s\n' "COMPLETE" > "$SUPERVISOR_RUN/status.txt"
