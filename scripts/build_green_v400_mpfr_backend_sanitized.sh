#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MPFR_PREFIX OUTPUT_DIRECTORY" >&2
  exit 2
fi

prefix=$1
output_directory=$2
project_root=$(cd "$(dirname "$0")/.." && pwd)

if [[ ! -f "$prefix/include/mpfr.h" ]]; then
  echo "MPFR header is unavailable below prefix: $prefix" >&2
  exit 2
fi

mkdir -p "$output_directory"

common_flags=(
  -std=c++17
  -O1
  -g3
  -fno-omit-frame-pointer
  -fno-optimize-sibling-calls
  -fPIC
  -shared
  -pthread
  -D_GLIBCXX_ASSERTIONS
  -DGREEN_V400_NATIVE_AUDIT_TEST_HOOKS=1
  -I"$prefix/include"
)
sources=(
  "$project_root/native/green_v400_mpfr_backend.cpp"
  "$project_root/native/green_v400_native_plan_loader.cpp"
)
link_flags=(
  -L"$prefix/lib"
  -Wl,-rpath,"$prefix/lib"
  -lmpfr
  -lgmp
  -lm
)

asan_ubsan_output="$output_directory/libgreen_v400_mpfr_backend_asan_ubsan_audit.so"
tsan_output="$output_directory/libgreen_v400_mpfr_backend_tsan_audit.so"

g++ "${common_flags[@]}" \
  -fsanitize=address,undefined \
  -fsanitize-address-use-after-scope \
  -fno-sanitize-recover=all \
  "${sources[@]}" "${link_flags[@]}" \
  -o "$asan_ubsan_output"

g++ "${common_flags[@]}" \
  -fsanitize=thread \
  "${sources[@]}" "${link_flags[@]}" \
  -o "$tsan_output"

printf 'ASAN_UBSAN=%s\n' "$asan_ubsan_output"
printf 'TSAN=%s\n' "$tsan_output"
