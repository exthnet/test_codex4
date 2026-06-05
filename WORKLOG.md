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
- `results/flag-sweep.csv`: compiler-flag comparison on compute node
- `results/pin-sweep.csv`: pinning comparison for `10000^3`

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
9. Measured `10000^3` via PJM batch and initially recorded `65.150163505 s`, `30.698 GFLOPS` with `--no-verify`.
10. Compared several compute-node build configurations for large sizes:
   - generic `-O3`
   - `-O3 -march=sapphirerapids -mtune=sapphirerapids`
   - the above plus `-funroll-loops`
   - `-Ofast` with Sapphire Rapids tuning
11. Found that `-march=sapphirerapids -mtune=sapphirerapids` consistently improved throughput on the measured node:
   - `2048^3`: `37.168 -> 44.713 GFLOPS`
   - `4096^3`: `37.823 -> 42.451 GFLOPS`
   - `10000^3`: `30.579 -> 35.522 GFLOPS`
12. Added `AMX_PIN_CORE` / `AMX_DISABLE_PIN` support to explore pinning effects and confirmed that pinning differences are minor for `10000^3`.
13. Tried a small AMX-kernel restructuring aimed at keeping `C` tiles resident across K blocks; the result regressed and was intentionally discarded.
14. Added a wider AMX macro-kernel that updates two `16 x 16` output tiles across N per A-tile load (`16 x 32 x 32` effective blocking at the outer-kernel level).
15. Re-ran `10000^3` via PJM batch after the wider macro-kernel change and recorded `23.374218975 s`, `85.564 GFLOPS` with `--no-verify`.

## Partial Results

| Kernel | M | N | K | Repeats | Time [s] | GFLOPS | Max Abs Err | Max Rel Err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMX | 256 | 256 | 256 | 5 | 0.000304035 | 110.364 | 5.72e-06 | 2.11e-03 |
| AMX | 512 | 512 | 512 | 5 | 0.002418624 | 110.987 | 1.53e-05 | 6.87e-02 |
| AMX | 1024 | 1024 | 1024 | 3 | 0.024887946 | 86.286 | 4.20e-05 | 2.50e-01 |
| AMX | 2048 | 2048 | 2048 | 2 | 0.191187412 | 89.859 | 1.11e-04 | 1.80e+00 |
| AMX | 10000 | 10000 | 10000 | 1 | 23.374218975 | 85.564 | 0.00e+00 | 0.00e+00 |

## Current Issues

- The committed `results/bench.csv` now reflects the latest two-output-tile AMX kernel result for `10000^3` (`85.564 GFLOPS`).
- Baseline comparisons for larger sizes are not yet collected on compute node.
- The current AMX kernel still uses a simple packing strategy and now computes two neighboring output tiles per A-tile load; broader redesigns beyond this are still unexplored.

## Re-run Commands

Build:

```bash
module load gcc-toolset/13
make clean all CFLAGS="-std=gnu11 -O3 -Wall -Wextra -Wpedantic -march=sapphirerapids -mtune=sapphirerapids"
```

Batch submission:

```bash
pjsub scripts/submit_pjm_bench.sh
```
