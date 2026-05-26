# AMX BF16 GEMM Report

## Objective

Optimize single-core `bf16 * bf16 -> fp32` GEMM with Intel AMX on Sapphire Rapids.

## Current Implementation

- `src/gemm_bf16.c`
  - Baseline triple-loop GEMM
  - AMX BF16 GEMM with `16 x 16 x 32` blocking
  - B-matrix prepacking by `(K-block, N-block)` tile
  - On-the-fly A-panel packing
  - Correctness check against baseline
- `scripts/run_benchmarks.sh`
  - Compute-node benchmark driver
- `scripts/submit_pjm_bench.sh`
  - PJM submission script for `a-batch-low`, `node=1`, `elapse=10:00`

## Bugs Found and Fixed

1. Compute-node execution initially failed because the job script relied on `dirname "$0"`; fixed by preferring `PJM_O_WORKDIR`.
2. Early AMX runs failed with `Illegal instruction` because the whole binary was compiled for Sapphire Rapids. Fixed by using generic compilation and applying AMX target attributes only to AMX functions.
3. The AMX BF16 B-tile configuration was incorrect. `TDPBF16PS` requires each B tile row to hold `2 * N` bf16 values, i.e. `4 * N` bytes. Fixing `colsb[2]` resolved the AMX execution failure.

## Partial Results

Measured on compute node:

| Kernel | M | N | K | Repeats | Time [s] | GFLOPS | Max Abs Err | Max Rel Err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMX | 256 | 256 | 256 | 5 | 0.000304035 | 110.364 | 5.72e-06 | 2.11e-03 |
| AMX | 512 | 512 | 512 | 5 | 0.002418624 | 110.987 | 1.53e-05 | 6.87e-02 |
| AMX | 1024 | 1024 | 1024 | 3 | 0.024887946 | 86.286 | 4.20e-05 | 2.50e-01 |
| AMX | 2048 | 2048 | 2048 | 2 | 0.191187412 | 89.859 | 1.11e-04 | 1.80e+00 |

## Observations

- The AMX kernel is functioning on Sapphire Rapids compute nodes.
- Performance is roughly `86-111 GFLOPS` in the currently tested range.
- Relative error grows with problem size. This is expected to some extent because the reference and the AMX kernel accumulate in different orders, but it should still be monitored.
- The benchmark script was edited while jobs were running, so large-size results beyond `2048^3` should be re-run cleanly before treating them as final.

## Next Steps

1. Re-run a stable benchmark pass for `4096^3`, `10000^3`, and the baseline cases.
2. Try reducing AMX configuration overhead by keeping tile configuration loaded across inner loops.
3. Compare against MKL if available in the environment.
4. Generate plots from the final CSV and fold them into this report.
