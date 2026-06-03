#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /opt/rh/gcc-toolset-13/enable
make clean all

mkdir -p results logs

./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv \
    | tee results/bench-10000-batch.csv
