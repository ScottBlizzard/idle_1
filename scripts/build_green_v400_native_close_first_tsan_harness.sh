#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TSAN_AUDIT_DSO OUTPUT_EXECUTABLE" >&2
  exit 2
fi

tsan_dso=$(realpath "$1")
output_executable=$2
project_root=$(cd "$(dirname "$0")/.." && pwd)
source_file="$project_root/native/green_v400_native_close_first_tsan_harness.cpp"

if [[ ! -f "$tsan_dso" ]]; then
  echo "TSan audit DSO does not exist: $tsan_dso" >&2
  exit 2
fi
if [[ -e "$output_executable" ]]; then
  echo "refusing to overwrite existing executable: $output_executable" >&2
  exit 2
fi
if ! readelf -d "$tsan_dso" | grep -q 'libtsan'; then
  echo "DSO has no dynamic dependency on libtsan: $tsan_dso" >&2
  exit 2
fi

required_symbols=(
  green_v400_native_plan_envelope_open_v1
  green_v400_native_precision_context_open_v1
  green_v400_native_precision_context_close_v1
  green_v400_native_precision_context_dispatch_cell_v1
  green_v400_native_dispatch_concurrency_reset_v1
  green_v400_native_dispatch_concurrency_info_v1
  green_v400_native_audit_after_find_hook_enable_v1
  green_v400_native_audit_after_find_hook_reached_v1
  green_v400_native_audit_after_find_hook_release_v1
)
exported_symbols=$(nm -D --defined-only "$tsan_dso" | awk '{print $3}')
for symbol in "${required_symbols[@]}"; do
  if ! grep -Fxq "$symbol" <<<"$exported_symbols"; then
    echo "TSan audit DSO is missing required symbol: $symbol" >&2
    exit 2
  fi
done

output_directory=$(dirname "$output_executable")
mkdir -p "$output_directory"
output_executable=$(realpath -m "$output_executable")
dso_directory=$(dirname "$tsan_dso")
dso_basename=$(basename "$tsan_dso")

g++ \
  -std=c++17 \
  -O1 \
  -g3 \
  -fno-omit-frame-pointer \
  -fno-optimize-sibling-calls \
  -D_GLIBCXX_ASSERTIONS \
  -fsanitize=thread \
  -pthread \
  "$source_file" \
  -L"$dso_directory" \
  -Wl,-rpath,"$dso_directory" \
  -Wl,-z,defs \
  "-l:$dso_basename" \
  -o "$output_executable"

printf 'TSAN_NATIVE_CLOSE_FIRST_HARNESS=%s\n' "$output_executable"
