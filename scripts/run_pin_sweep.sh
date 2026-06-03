#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /opt/rh/gcc-toolset-13/enable

mkdir -p results logs

out="results/pin-sweep.csv"
echo "label,kernel,m,n,k,repeats,seconds,gflops,max_abs_err,max_rel_err" > "$out"

cflags="-std=gnu11 -O3 -Wall -Wextra -Wpedantic -march=sapphirerapids -mtune=sapphirerapids"
make clean all CFLAGS="$cflags" >/dev/null

for label in pin0 pin1 pin60 pin61 nopin; do
    case "$label" in
        pin0) env AMX_PIN_CORE=0 ./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv ;;
        pin1) env AMX_PIN_CORE=1 ./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv ;;
        pin60) env AMX_PIN_CORE=60 ./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv ;;
        pin61) env AMX_PIN_CORE=61 ./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv ;;
        nopin) env AMX_DISABLE_PIN=1 ./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv ;;
    esac | sed "s/^/${label},/"
done >> "$out"

cat "$out"
