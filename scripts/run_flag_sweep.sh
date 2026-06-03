#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /opt/rh/gcc-toolset-13/enable

mkdir -p results logs

out="results/flag-sweep.csv"
echo "label,m,n,k,repeats,seconds,gflops,max_abs_err,max_rel_err" > "$out"

run_case() {
    local label="$1"
    local cflags="$2"

    make clean all CFLAGS="$cflags" >/dev/null

    while read -r m n k repeats extra; do
        args=(--kernel amx --m "$m" --n "$n" --k "$k" --repeats "$repeats" --csv)
        if [[ -n "${extra:-}" ]]; then
            args+=("$extra")
        fi
        ./build/gemm_bf16 "${args[@]}" | sed "s/^/${label},/"
    done <<'EOF'
2048 2048 2048 1
4096 4096 4096 1
10000 10000 10000 1 --no-verify
EOF
}

run_case baseline "-std=gnu11 -O3 -Wall -Wextra -Wpedantic" >> "$out"
run_case spr "-std=gnu11 -O3 -Wall -Wextra -Wpedantic -march=sapphirerapids -mtune=sapphirerapids" >> "$out"
run_case spr_unroll "-std=gnu11 -O3 -Wall -Wextra -Wpedantic -march=sapphirerapids -mtune=sapphirerapids -funroll-loops" >> "$out"
run_case spr_fast "-std=gnu11 -Ofast -Wall -Wextra -Wpedantic -march=sapphirerapids -mtune=sapphirerapids -fno-math-errno -fno-trapping-math -funroll-loops" >> "$out"

cat "$out"
