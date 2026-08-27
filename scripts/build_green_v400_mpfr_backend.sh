#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MPFR_PREFIX OUTPUT_SO" >&2
  exit 2
fi

prefix=$1
output=$2
project_root=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$(dirname "$output")"
g++ -std=c++17 -O3 -fPIC -shared \
  -I"$prefix/include" \
  "$project_root/native/green_v400_mpfr_backend.cpp" \
  "$project_root/native/green_v400_native_plan_loader.cpp" \
  -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -lmpfr -lgmp -lm \
  -o "$output"
