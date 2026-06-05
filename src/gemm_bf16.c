#define _GNU_SOURCE

#include <errno.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef SYS_arch_prctl
#define SYS_arch_prctl 158
#endif

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

#define BM 16
#define BN 16
#define BK 32
#define TILE_STRIDE_BYTES 64

typedef uint16_t bf16_t;

typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg_t;

typedef enum {
    KERNEL_BASELINE = 0,
    KERNEL_AMX = 1,
} kernel_t;

typedef struct {
    int m;
    int n;
    int k;
    int repeats;
    uint32_t seed;
    kernel_t kernel;
    bool verify;
    bool csv;
} options_t;

typedef struct {
    double seconds;
    double gflops;
    float max_abs_err;
    float max_rel_err;
} result_t;

static bool trace_enabled(void) {
    const char *env = getenv("AMX_TRACE");
    return env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
}

static inline uint64_t min_u64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void *aligned_calloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    memset(ptr, 0, size);
    return ptr;
}

static inline bf16_t fp32_to_bf16(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    const uint32_t lsb = (v.u >> 16U) & 1U;
    const uint32_t rounding_bias = 0x7FFFU + lsb;
    return (bf16_t)((v.u + rounding_bias) >> 16U);
}

static inline float bf16_to_fp32(bf16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)x << 16U;
    return v.f;
}

static void fill_bf16_random(bf16_t *dst, size_t count, uint32_t *state) {
    for (size_t i = 0; i < count; ++i) {
        *state = (*state * 1664525U) + 1013904223U;
        const uint32_t mant = (*state >> 9U) | 0x3F000000U;
        union {
            uint32_t u;
            float f;
        } v = {.u = mant};
        const float scaled = (v.f - 0.75f) * 4.0f;
        dst[i] = fp32_to_bf16(scaled);
    }
}

static void zero_fp32(float *dst, size_t count) {
    memset(dst, 0, count * sizeof(*dst));
}

static void gemm_baseline(
    int m,
    int n,
    int k,
    const bf16_t *a,
    const bf16_t *b,
    float *c
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) {
                acc += bf16_to_fp32(a[(size_t)i * (size_t)k + (size_t)p]) *
                       bf16_to_fp32(b[(size_t)p * (size_t)n + (size_t)j]);
            }
            c[(size_t)i * (size_t)n + (size_t)j] = acc;
        }
    }
}

static int request_amx_permission(void) {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        return -1;
    }
    return 0;
}

static bool cpu_supports_amx(void) {
#if defined(__x86_64__)
    return __builtin_cpu_supports("amx-tile") && __builtin_cpu_supports("amx-bf16");
#else
    return false;
#endif
}

__attribute__((target("amx-tile,amx-bf16")))
static void load_tilecfg_1col(int m_rows, int n_cols, int k_bytes) {
    tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    cfg.start_row = 0;

    cfg.colsb[0] = (uint16_t)(n_cols * (int)sizeof(float));
    cfg.rows[0] = (uint8_t)m_rows;
    cfg.colsb[1] = (uint16_t)k_bytes;
    cfg.rows[1] = (uint8_t)m_rows;
    cfg.colsb[2] = (uint16_t)(n_cols * (int)sizeof(float));
    cfg.rows[2] = (uint8_t)(k_bytes / 4);

    _tile_loadconfig(&cfg);
}

__attribute__((target("amx-tile,amx-bf16")))
static void load_tilecfg_2col(int m_rows, int n_cols0, int n_cols1, int k_bytes) {
    tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    cfg.start_row = 0;

    cfg.colsb[0] = (uint16_t)(n_cols0 * (int)sizeof(float));
    cfg.rows[0] = (uint8_t)m_rows;
    cfg.colsb[1] = (uint16_t)(n_cols1 * (int)sizeof(float));
    cfg.rows[1] = (uint8_t)m_rows;
    cfg.colsb[2] = (uint16_t)k_bytes;
    cfg.rows[2] = (uint8_t)m_rows;
    cfg.colsb[3] = (uint16_t)(n_cols0 * (int)sizeof(float));
    cfg.rows[3] = (uint8_t)(k_bytes / 4);
    cfg.colsb[4] = (uint16_t)(n_cols1 * (int)sizeof(float));
    cfg.rows[4] = (uint8_t)(k_bytes / 4);

    _tile_loadconfig(&cfg);
}

static void pack_a_panel(
    int rows,
    int k_chunk,
    const bf16_t *src,
    int lda,
    bf16_t *dst
) {
    memset(dst, 0, BM * TILE_STRIDE_BYTES);
    for (int i = 0; i < rows; ++i) {
        memcpy(
            (char *)dst + (size_t)i * TILE_STRIDE_BYTES,
            src + (size_t)i * (size_t)lda,
            (size_t)k_chunk * sizeof(*src)
        );
    }
}

static void pack_b_panel(
    int k_chunk,
    int cols,
    const bf16_t *src,
    int ldb,
    bf16_t *dst
) {
    memset(dst, 0, BM * TILE_STRIDE_BYTES);
    const int pair_rows = (k_chunk + 1) / 2;
    for (int kp = 0; kp < pair_rows; ++kp) {
        bf16_t *row = (bf16_t *)((char *)dst + (size_t)kp * TILE_STRIDE_BYTES);
        const int k0 = 2 * kp;
        const int k1 = k0 + 1;
        for (int j = 0; j < cols; ++j) {
            row[2 * j] = src[(size_t)k0 * (size_t)ldb + (size_t)j];
            row[2 * j + 1] = (k1 < k_chunk) ? src[(size_t)k1 * (size_t)ldb + (size_t)j] : 0;
        }
    }
}

static bf16_t *prepack_b_matrix(int k, int n, const bf16_t *b, size_t *tile_count_out) {
    const int kk_blocks = (k + BK - 1) / BK;
    const int jj_blocks = (n + BN - 1) / BN;
    const size_t tile_count = (size_t)kk_blocks * (size_t)jj_blocks;
    bf16_t *packed = aligned_calloc(64, tile_count * BM * TILE_STRIDE_BYTES);
    if (packed == NULL) {
        return NULL;
    }

    for (int kk = 0; kk < kk_blocks; ++kk) {
        const int k0 = kk * BK;
        const int k_chunk = (int)min_u64((uint64_t)BK, (uint64_t)(k - k0));
        for (int jj = 0; jj < jj_blocks; ++jj) {
            const int j0 = jj * BN;
            const int cols = (int)min_u64((uint64_t)BN, (uint64_t)(n - j0));
            bf16_t *tile = packed + ((size_t)kk * (size_t)jj_blocks + (size_t)jj) * (BM * TILE_STRIDE_BYTES / (int)sizeof(bf16_t));
            pack_b_panel(k_chunk, cols, b + (size_t)k0 * (size_t)n + (size_t)j0, n, tile);
        }
    }

    *tile_count_out = tile_count;
    return packed;
}

__attribute__((target("amx-tile,amx-bf16")))
static void amx_kernel_block(
    int rows,
    int cols,
    int k_chunk,
    const bf16_t *a_pack,
    const bf16_t *b_pack,
    float *c,
    int ldc
) {
    const int k_pairs = (k_chunk + 1) / 2;
    const int k_bytes = k_pairs * 4;

    if (trace_enabled()) {
        fprintf(stderr, "amx_kernel_block rows=%d cols=%d k_chunk=%d k_bytes=%d\n", rows, cols, k_chunk, k_bytes);
    }
    load_tilecfg_1col(rows, cols, k_bytes);
    if (trace_enabled()) {
        fprintf(stderr, "tilecfg loaded\n");
    }
    _tile_loadd(0, c, (size_t)ldc * sizeof(*c));
    _tile_loadd(1, a_pack, TILE_STRIDE_BYTES);
    _tile_loadd(2, b_pack, TILE_STRIDE_BYTES);
    if (trace_enabled()) {
        fprintf(stderr, "tiles loaded\n");
    }
    _tile_dpbf16ps(0, 1, 2);
    if (trace_enabled()) {
        fprintf(stderr, "dpbf16ps done\n");
    }
    _tile_stored(0, c, (size_t)ldc * sizeof(*c));
    _tile_release();
}

__attribute__((target("amx-tile,amx-bf16")))
static void amx_kernel_block_2col(
    int rows,
    int cols0,
    int cols1,
    int k_chunk,
    const bf16_t *a_pack,
    const bf16_t *b0_pack,
    const bf16_t *b1_pack,
    float *c0,
    float *c1,
    int ldc
) {
    const int k_pairs = (k_chunk + 1) / 2;
    const int k_bytes = k_pairs * 4;

    load_tilecfg_2col(rows, cols0, cols1, k_bytes);
    _tile_loadd(0, c0, (size_t)ldc * sizeof(*c0));
    _tile_loadd(1, c1, (size_t)ldc * sizeof(*c1));
    _tile_loadd(2, a_pack, TILE_STRIDE_BYTES);
    _tile_loadd(3, b0_pack, TILE_STRIDE_BYTES);
    _tile_loadd(4, b1_pack, TILE_STRIDE_BYTES);
    _tile_dpbf16ps(0, 2, 3);
    _tile_dpbf16ps(1, 2, 4);
    _tile_stored(0, c0, (size_t)ldc * sizeof(*c0));
    _tile_stored(1, c1, (size_t)ldc * sizeof(*c1));
    _tile_release();
}

__attribute__((target("amx-tile,amx-bf16")))
static void gemm_amx(
    int m,
    int n,
    int k,
    const bf16_t *a,
    const bf16_t *b,
    float *c
) {
    if (request_amx_permission() != 0) {
        fprintf(stderr, "failed to enable AMX state: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (trace_enabled()) {
        fprintf(stderr, "amx permission enabled\n");
    }
    if (!cpu_supports_amx()) {
        fprintf(stderr, "AMX BF16 is not reported by this CPU\n");
        exit(EXIT_FAILURE);
    }
    if (trace_enabled()) {
        fprintf(stderr, "cpu_supports_amx passed\n");
    }

    bf16_t *a_pack = aligned_calloc(64, BM * TILE_STRIDE_BYTES);
    size_t b_tile_count = 0;
    bf16_t *b_pack = prepack_b_matrix(k, n, b, &b_tile_count);
    if (a_pack == NULL || b_pack == NULL) {
        fprintf(stderr, "failed to allocate packed panels\n");
        free(a_pack);
        free(b_pack);
        exit(EXIT_FAILURE);
    }

    const int jj_blocks = (n + BN - 1) / BN;
    for (int ii = 0; ii < m; ii += BM) {
        const int rows = (int)min_u64((uint64_t)BM, (uint64_t)(m - ii));
        for (int kk = 0; kk < k; kk += BK) {
            const int kk_block = kk / BK;
            const int k_chunk = (int)min_u64((uint64_t)BK, (uint64_t)(k - kk));
            pack_a_panel(rows, k_chunk, a + (size_t)ii * (size_t)k + (size_t)kk, k, a_pack);
            for (int jj = 0; jj < n; jj += 2 * BN) {
                const int jj_block = jj / BN;
                const int cols = (int)min_u64((uint64_t)BN, (uint64_t)(n - jj));
                const bf16_t *b_tile0 =
                    b_pack + ((size_t)kk_block * (size_t)jj_blocks + (size_t)jj_block) *
                                 (BM * TILE_STRIDE_BYTES / (int)sizeof(bf16_t));
                if (trace_enabled() && ii == 0 && kk == 0 && jj == 0) {
                    fprintf(stderr, "entering first amx block\n");
                }
                const int jj_next = jj + BN;
                if (jj_next < n) {
                    const int jj_block1 = jj_next / BN;
                    const int cols1 = (int)min_u64((uint64_t)BN, (uint64_t)(n - jj_next));
                    const bf16_t *b_tile1 =
                        b_pack + ((size_t)kk_block * (size_t)jj_blocks + (size_t)jj_block1) *
                                     (BM * TILE_STRIDE_BYTES / (int)sizeof(bf16_t));
                    amx_kernel_block_2col(
                        rows,
                        cols,
                        cols1,
                        k_chunk,
                        a_pack,
                        b_tile0,
                        b_tile1,
                        c + (size_t)ii * (size_t)n + (size_t)jj,
                        c + (size_t)ii * (size_t)n + (size_t)jj_next,
                        n
                    );
                } else {
                    amx_kernel_block(
                        rows,
                        cols,
                        k_chunk,
                        a_pack,
                        b_tile0,
                        c + (size_t)ii * (size_t)n + (size_t)jj,
                        n
                    );
                }
                if (trace_enabled() && ii == 0 && kk == 0 && jj == 0) {
                    fprintf(stderr, "completed first amx block\n");
                }
            }
        }
    }

    (void)b_tile_count;
    free(a_pack);
    free(b_pack);
}

static void pin_to_core_zero(void) {
    const char *disable_pin = getenv("AMX_DISABLE_PIN");
    if (disable_pin != NULL && disable_pin[0] != '\0' && strcmp(disable_pin, "0") != 0) {
        return;
    }

    int core = 0;
    const char *core_env = getenv("AMX_PIN_CORE");
    if (core_env != NULL && core_env[0] != '\0') {
        core = atoi(core_env);
        if (core < 0) {
            core = 0;
        }
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);
}

static void parse_args(int argc, char **argv, options_t *opt) {
    *opt = (options_t){
        .m = 512,
        .n = 512,
        .k = 512,
        .repeats = 3,
        .seed = 1U,
        .kernel = KERNEL_AMX,
        .verify = true,
        .csv = false,
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opt->m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opt->n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            opt->k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            opt->repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            opt->seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            const char *name = argv[++i];
            if (strcmp(name, "baseline") == 0) {
                opt->kernel = KERNEL_BASELINE;
            } else if (strcmp(name, "amx") == 0) {
                opt->kernel = KERNEL_AMX;
            } else {
                fprintf(stderr, "unknown kernel: %s\n", name);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt->verify = false;
        } else if (strcmp(argv[i], "--csv") == 0) {
            opt->csv = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    if (opt->m <= 0 || opt->n <= 0 || opt->k <= 0 || opt->repeats <= 0) {
        fprintf(stderr, "all sizes and repeats must be positive\n");
        exit(EXIT_FAILURE);
    }
}

static void compute_error(
    const float *ref,
    const float *test,
    size_t count,
    float *max_abs_err,
    float *max_rel_err
) {
    float abs_err = 0.0f;
    float rel_err = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float diff = fabsf(ref[i] - test[i]);
        const float denom = fmaxf(fabsf(ref[i]), 1.0e-6f);
        if (diff > abs_err) {
            abs_err = diff;
        }
        const float rel = diff / denom;
        if (rel > rel_err) {
            rel_err = rel;
        }
    }
    *max_abs_err = abs_err;
    *max_rel_err = rel_err;
}

static result_t run_kernel(
    const options_t *opt,
    const bf16_t *a,
    const bf16_t *b,
    float *c,
    float *ref
) {
    result_t res = {0};
    double best = 1.0e300;
    for (int r = 0; r < opt->repeats; ++r) {
        zero_fp32(c, (size_t)opt->m * (size_t)opt->n);
        const double t0 = now_seconds();
        if (opt->kernel == KERNEL_BASELINE) {
            gemm_baseline(opt->m, opt->n, opt->k, a, b, c);
        } else {
            gemm_amx(opt->m, opt->n, opt->k, a, b, c);
        }
        const double t1 = now_seconds();
        const double elapsed = t1 - t0;
        if (elapsed < best) {
            best = elapsed;
        }
    }
    res.seconds = best;
    res.gflops = (2.0 * (double)opt->m * (double)opt->n * (double)opt->k) / best / 1.0e9;

    if (opt->verify) {
        zero_fp32(ref, (size_t)opt->m * (size_t)opt->n);
        gemm_baseline(opt->m, opt->n, opt->k, a, b, ref);
        compute_error(ref, c, (size_t)opt->m * (size_t)opt->n, &res.max_abs_err, &res.max_rel_err);
    }

    return res;
}

int main(int argc, char **argv) {
    options_t opt;
    parse_args(argc, argv, &opt);
    pin_to_core_zero();

    bf16_t *a = aligned_calloc(64, (size_t)opt.m * (size_t)opt.k * sizeof(*a));
    bf16_t *b = aligned_calloc(64, (size_t)opt.k * (size_t)opt.n * sizeof(*b));
    float *c = aligned_calloc(64, (size_t)opt.m * (size_t)opt.n * sizeof(*c));
    float *ref = aligned_calloc(64, (size_t)opt.m * (size_t)opt.n * sizeof(*ref));
    if (a == NULL || b == NULL || c == NULL || ref == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(c);
        free(ref);
        return EXIT_FAILURE;
    }

    uint32_t rng = opt.seed;
    fill_bf16_random(a, (size_t)opt.m * (size_t)opt.k, &rng);
    fill_bf16_random(b, (size_t)opt.k * (size_t)opt.n, &rng);

    result_t res = run_kernel(&opt, a, b, c, ref);

    if (opt.csv) {
        printf(
            "%s,%d,%d,%d,%d,%.9f,%.3f,%.8e,%.8e\n",
            opt.kernel == KERNEL_BASELINE ? "baseline" : "amx",
            opt.m,
            opt.n,
            opt.k,
            opt.repeats,
            res.seconds,
            res.gflops,
            res.max_abs_err,
            res.max_rel_err
        );
    } else {
        printf("kernel: %s\n", opt.kernel == KERNEL_BASELINE ? "baseline" : "amx");
        printf("shape: %d x %d x %d\n", opt.m, opt.n, opt.k);
        printf("repeats: %d\n", opt.repeats);
        printf("best_seconds: %.9f\n", res.seconds);
        printf("gflops: %.3f\n", res.gflops);
        if (opt.verify) {
            printf("max_abs_err: %.8e\n", res.max_abs_err);
            printf("max_rel_err: %.8e\n", res.max_rel_err);
        }
    }

    free(a);
    free(b);
    free(c);
    free(ref);
    return 0;
}
