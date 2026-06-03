#!/usr/bin/env bash
#PJM -L rscgrp=a-batch-low,node=1
#PJM -L elapse=20:00
#PJM -j
#PJM -o logs/pjm-pin-sweep.out

set -euo pipefail

if [[ -n "${PJM_O_WORKDIR:-}" ]]; then
    cd "${PJM_O_WORKDIR}"
else
    cd "$(dirname "$0")/.."
fi

bash scripts/run_pin_sweep.sh
