#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 HARNESS DESCRIPTOR BLOB RAW_AUDIT_JSON SEALED_AUDIT_JSON SANITIZER_LOG EXIT_JSON" >&2
  exit 2
fi

harness=$(realpath "$1")
descriptor=$(realpath "$2")
blob=$(realpath "$3")
raw_audit_json=$4
sealed_audit_json=$5
sanitizer_log=$6
exit_json=$7
project_root=$(cd "$(dirname "$0")/.." && pwd)

if [[ ! -x "$harness" ]]; then
  echo "native harness is not executable: $harness" >&2
  exit 2
fi
for input_path in "$descriptor" "$blob"; do
  if [[ ! -f "$input_path" ]]; then
    echo "required immutable input does not exist: $input_path" >&2
    exit 2
  fi
done
for output_path in "$raw_audit_json" "$sealed_audit_json" "$sanitizer_log" "$exit_json"; do
  if [[ -e "$output_path" ]]; then
    echo "refusing to overwrite existing evidence file: $output_path" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$output_path")"
done

set +e
TSAN_OPTIONS='halt_on_error=1:exitcode=66:second_deadlock_stack=1:history_size=7' \
  timeout --signal=TERM --kill-after=30s 30m \
  "$harness" "$descriptor" "$blob" >"$raw_audit_json" 2>"$sanitizer_log"
process_status=$?
set -e

termination_signal=null
if (( process_status >= 129 && process_status <= 192 )); then
  termination_signal=$((process_status - 128))
fi
printf '{"exit_code":%d,"schema_version":"green-v400-sanitizer-process-exit-v1","termination_signal":%s}\n' \
  "$process_status" "$termination_signal" >"$exit_json"

if (( process_status != 0 )); then
  exit "$process_status"
fi

# Python is intentionally outside the TSan process boundary: this sealer runs
# only after wait/timeout has returned and never loads the native DSO.
python3 "$project_root/analysis/green_v400_native_close_first_tsan_seal.py" \
  --raw-audit "$raw_audit_json" \
  --sanitizer-log "$sanitizer_log" \
  --process-exit-json "$exit_json" \
  --output "$sealed_audit_json"

exit "$process_status"
