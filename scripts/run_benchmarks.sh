#!/usr/bin/env bash
set -euo pipefail

module load gcc-toolset/13
make clean all

mkdir -p results
echo "kernel,m,n,k,repeats,seconds,gflops,max_abs_err,max_rel_err" > results/bench.csv

echo "=== CPU INFO ==="
lscpu
echo "=== CPU FLAGS (first match) ==="
grep -m1 '^flags' /proc/cpuinfo || true
echo "=== BASELINE SMOKE TEST ==="
./build/gemm_bf16 --kernel baseline --m 64 --n 64 --k 64 --repeats 1
echo "=== AMX BENCHMARKS ==="

AMX_TRACE=1 ./build/gemm_bf16 --kernel amx --m 256 --n 256 --k 256 --repeats 5 --csv >> results/bench.csv
./build/gemm_bf16 --kernel amx --m 512 --n 512 --k 512 --repeats 5 --csv >> results/bench.csv
./build/gemm_bf16 --kernel amx --m 1024 --n 1024 --k 1024 --repeats 3 --csv >> results/bench.csv
./build/gemm_bf16 --kernel amx --m 2048 --n 2048 --k 2048 --repeats 2 --csv >> results/bench.csv
./build/gemm_bf16 --kernel amx --m 4096 --n 4096 --k 4096 --repeats 1 --csv >> results/bench.csv
./build/gemm_bf16 --kernel amx --m 10000 --n 10000 --k 10000 --repeats 1 --no-verify --csv >> results/bench.csv
./build/gemm_bf16 --kernel baseline --m 256 --n 256 --k 256 --repeats 1 --csv >> results/bench.csv
./build/gemm_bf16 --kernel baseline --m 512 --n 512 --k 512 --repeats 1 --csv >> results/bench.csv
