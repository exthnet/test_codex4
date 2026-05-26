#!/usr/bin/env bash
#PJM -L rscgrp=a-batch-low,node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -o logs/pjm-bench.out

set -euo pipefail

if [[ -n "${PJM_O_WORKDIR:-}" ]]; then
    cd "${PJM_O_WORKDIR}"
else
    cd "$(dirname "$0")/.."
fi

mkdir -p logs results
bash scripts/run_benchmarks.sh
