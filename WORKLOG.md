# Worklog

## Scope

- Target: single-core `bf16 * bf16 -> fp32` GEMM on Sapphire Rapids with AMX
- Constraints: no OpenMP, no MPI, no references outside this repository

## Files Added

- `src/gemm_bf16.c`: baseline GEMM and AMX BF16 implementation
- `Makefile`: local build entry point
- `scripts/run_benchmarks.sh`: compute-node benchmark driver
- `scripts/submit_pjm_bench.sh`: PJM batch script
- `results/bench.csv`: partial benchmark results gathered on compute node
- `logs/pjm-bench.out`: latest PJM output log
- `report.md`: interim report

## Progress Record

1. Created a baseline triple-loop GEMM for correctness reference.
2. Implemented an AMX BF16 kernel using `TDPBF16PS`.
3. Added AMX state enablement through `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`.
4. Confirmed AMX-related CPU flags on compute node:
   - `amx_tile`
   - `amx_bf16`
   - `amx_int8`
5. Fixed a compute-node path issue by using `PJM_O_WORKDIR` in the submit script.
6. Fixed an `Illegal instruction` issue caused by compiling the whole binary for Sapphire Rapids; the AMX target is now limited to AMX functions.
7. Fixed the BF16 B-tile configuration. The B tile needed `4 * N` bytes per row, not `2 * N`.
8. Verified that the AMX kernel runs on compute node and collected partial performance results through `2048^3`.

## Partial Results

| Kernel | M | N | K | Repeats | Time [s] | GFLOPS | Max Abs Err | Max Rel Err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMX | 256 | 256 | 256 | 5 | 0.000304035 | 110.364 | 5.72e-06 | 2.11e-03 |
| AMX | 512 | 512 | 512 | 5 | 0.002418624 | 110.987 | 1.53e-05 | 6.87e-02 |
| AMX | 1024 | 1024 | 1024 | 3 | 0.024887946 | 86.286 | 4.20e-05 | 2.50e-01 |
| AMX | 2048 | 2048 | 2048 | 2 | 0.191187412 | 89.859 | 1.11e-04 | 1.80e+00 |

## Current Issues

- Large-size runs beyond `2048^3` were not finalized in a clean pass.
- Baseline comparisons for larger sizes are not yet collected on compute node.
- The benchmark script should be re-run in a stable pass before treating the report as final.

## Re-run Commands

Build:

```bash
module load gcc-toolset/13
make clean all
```

Batch submission:

```bash
pjsub scripts/submit_pjm_bench.sh
```
