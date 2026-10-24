/* threads_gpu.cu -- the CUDA backend's block size (all transfer modes) and host
 * staging threads (ring only, the one mode that copies on the host), timed on a corpus
 * law and checked bit for bit against the default options. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matern.h"
#include "corpus_dataset.h"
#include "common/wm_metrics.h"
#include "common/wm_timing.h"

static const struct { int mode; const char *name; } STRATEGIES[] = {
    { MATERN_CUDA_SYNC,         "sync"   },
    { MATERN_CUDA_ASYNC_DIRECT, "direct" },
    { MATERN_CUDA_ASYNC_RING,   "ring"   },
    { MATERN_CUDA_ASYNC_MAPPED, "mapped" },
};
enum { STRATEGY_COUNT = (int)(sizeof STRATEGIES / sizeof STRATEGIES[0]) };

static int call_once(const corpus_dataset *cs, double *out, int z_family, int mode,
                     const matern_cuda_options *o)
{
    const wm_dataset *s = &cs->set;
    if (z_family)
        return matern_covariance_cuda_z_tuned(out, s->argument, s->order, s->count,
                                              mode, o);
    return matern_covariance_cuda_tuned(out, s->h, s->order, s->length_scale,
                                        s->count, mode, o);
}

/* One configuration, timed under common/wm_timing.h.  The warm-up passes absorb
 * context setup and the ring's pinned allocation; only completed repetitions count. */
static wm_time time_config(const corpus_dataset *cs, double *out, int z_family,
                           int mode, const matern_cuda_options *o, int reps,
                           int *status, const char *family, const char *law,
                           const char *label)
{
    double *t = (double *)malloc((size_t)(reps > 0 ? reps : 1) * sizeof *t);
    wm_time z; memset(&z, 0, sizeof z);
    if (!t) { *status = MATERN_OUT_OF_MEMORY; return z; }

    *status = MATERN_OK;
    for (int w = 0; w < WM_WARMUP_REPS && *status == MATERN_OK; w++)
        *status = call_once(cs, out, z_family, mode, o);
    if (*status != MATERN_OK) { free(t); return z; }

    const double per = 1e9 / (double)cs->set.count;
    int done = 0;
    for (int r = 0; r < reps; r++) {
        const double a = wm_now_s();
        const int st = call_once(cs, out, z_family, mode, o);
        const double el = wm_now_s() - a;
        if (st != MATERN_OK) { *status = st; break; }
        t[done++] = el * per;
    }
    wm_raw_emit("cuda", "-", family, law, label, cs->set.count, t, done);
    const wm_time out_t = wm_time_reduce(t, done);
    free(t);
    return out_t;
}

/* One untimed call into a NaN-filled buffer, scored and compared with the
 * default-options output (`identical`); a failed configuration gives n/a. */
static int score_config(const corpus_dataset *cs, double *out, const double *ref,
                        int z_family, int mode, const matern_cuda_options *o,
                        int status, wm_metrics *M, const char **same)
{
    const size_t N = cs->set.count;
    memset(M, 0, sizeof *M); M->n = N; M->u_max = NAN;
    *same = "n/a";
    if (status != MATERN_OK) return status;
    for (size_t k = 0; k < N; k++) out[k] = NAN;
    const int st = call_once(cs, out, z_family, mode, o);
    if (st != MATERN_OK) return st;
    wm_metrics_compute(M, out, cs->set.reference_high, cs->set.reference_low,
                       cs->accuracy_bits, cs->set.order,
                       cs->set.has_length_scale ? cs->set.h : cs->set.argument,
                       cs->set.has_length_scale ? cs->set.length_scale : NULL,
                       N, WM_ULP_MODE_DEFAULT);
    *same = (memcmp(out, ref, N * sizeof *out) == 0) ? "yes" : "no";
    return MATERN_OK;
}

#define ACC_HDR "ulp_mode,n_scored,n_excluded,nonfinite,ref_zero,ref_subnormal," \
                "exact,min_bits,u_min,u_q1,u_med,u_q3,u_p99,u_p999,u_max,u_mean," \
                "u_iqr,u_fence,beyond_fence,u_med_nontrivial,rel_max,abs_max," \
                "worst_nu,worst_arg"
static void print_acc(const wm_metrics *M)
{
    printf(",%s,%zu,%zu,%zu,%zu,%zu,%zu,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
           "%.17g,%.17g,%.17g,%.17g,%zu,%.17g,%.17g,%.17g,%.17g,%.17g",
           wm_ulp_mode_name(WM_ULP_MODE_DEFAULT), M->n_scored, M->n_excluded, M->n_nonfinite,
           M->n_ref_zero, M->n_ref_subnormal, M->n_exact, M->min_bits,
           M->u_min, M->u_q1, M->u_med, M->u_q3, M->u_p99, M->u_p999, M->u_max,
           M->u_mean, M->u_iqr, M->u_fence_hi, M->n_beyond_fence,
           M->u_med_nontrivial, M->rel_max, M->abs_max,
           M->n_worst ? M->worst[0].nu : NAN, M->n_worst ? M->worst[0].arg : NAN);
}

static void usage(FILE *o, const char *p)
{
    fprintf(o, "usage: %s [--family z_nu|h_nu_s] [--law NAME] [--reps N] [--n POINTS]\n"
           "          [--data-dir data] [--format csv|text]\n"
           "  Sweeps `block` across all transfer strategies and `host_threads` under\n"
           "  ASYNC_RING only -- the ring is the only strategy that copies on the host,\n"
           "  so the knob is inert under the other three.\n", p);
}

int main(int argc, char **argv)
{
    const char *family = "h_nu_s", *law = "size_random";
    const char *root = "data", *format = "csv";
    int reps = WM_REPS_MAX_DEFAULT; size_t n = 0;   /* the pipeline's default count */

    for (int a = 1; a < argc; a++) {
        const int nx = (a + 1 < argc);
        if      (!strcmp(argv[a],"--family")   && nx) family = argv[++a];
        else if (!strcmp(argv[a],"--law")      && nx) law    = argv[++a];
        else if (!strcmp(argv[a],"--reps")     && nx) reps   = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--n")        && nx) n      = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--data-dir") && nx) root   = argv[++a];
        else if (!strcmp(argv[a],"--format")   && nx) format = argv[++a];
        else if (!strcmp(argv[a],"--help")) { usage(stdout, argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); usage(stderr, argv[0]); return 2; }
    }
    if (strchr(family, ',')) { fprintf(stderr, "--family takes one family\n"); return 2; }
    if (corpus_family_check(family)) return 2;

    if (!matern_cuda_available()) {
        fprintf(stderr, "threads_gpu: no usable CUDA device\n");
        return 3;
    }

    const int z_family = !strcmp(family, "z_nu");
    /* per-repetition samples, one file per invocation (run_gpu.sh names it) */
    wm_raw_open(getenv("WM_RAW_REPS"));

    corpus_dataset cs;
    /* only the first n points are evaluated, so only they are read */
    if (corpus_dataset_load_prefix(&cs, root, family, law, n) != 0) {
        fprintf(stderr, "threads_gpu: could not load %s/%s from %s\n", family, law, root);
        return 1;
    }
    if (n > 0 && n < cs.set.count) cs.set.count = n;
    const size_t N = cs.set.count;

    double *out = (double *)malloc(N * sizeof *out);
    double *ref = (double *)malloc(N * sizeof *ref);
    if (!out || !ref) { fprintf(stderr, "threads_gpu: out of memory\n"); return 1; }

    /* The comparison output: default options, sync strategy. */
    int rstatus = call_once(&cs, ref, z_family, MATERN_CUDA_SYNC, NULL);
    if (rstatus != MATERN_OK) {
        fprintf(stderr, "threads_gpu: reference call failed (%d)\n", rstatus);
        return 1;
    }

    static const int BLOCKS[] = { 64, 128, 256, 512, 1024 };
    static const int MT[]     = { 1, 2, 4, 8, 16 };

    const int csv = !strcmp(format, "csv");
    if (csv)
        /* Columns: docs/benchmark.md, "Output". */
        printf("schema,device,supplier,family,law,backend,knob,knob_value,transfer_mode,"
               "n,reps,ns_per_elem,ns_med,ns_spread,identical,status,"
               "ns_min,ns_q1,ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_points,"
               "timing_scope," ACC_HDR "\n");
    else
        printf("threads_gpu: %s %s, n = %zu, %d reps, median over reps\n", family, law, N, reps);

    int rc = 0;

    /* block size, across all strategies */
    if (!csv) printf("\n  block size, ns/elem\n  %8s", "block");
    if (!csv) { for (int s = 0; s < STRATEGY_COUNT; s++) printf(" %9s", STRATEGIES[s].name); printf("\n"); }
    for (size_t i = 0; i < sizeof BLOCKS / sizeof BLOCKS[0]; i++) {
        if (!csv) printf("  %8d", BLOCKS[i]);
        for (int s = 0; s < STRATEGY_COUNT; s++) {
            matern_cuda_options o; memset(&o, 0, sizeof o); o.block = BLOCKS[i];
            int status = 0;
            char label[64];
            snprintf(label, sizeof label, "block%d/%s", BLOCKS[i], STRATEGIES[s].name);
            const wm_time t = time_config(&cs, out, z_family, STRATEGIES[s].mode,
                                          &o, reps, &status, family, law, label);
            wm_metrics M; const char *same;
            status = score_config(&cs, out, ref, z_family, STRATEGIES[s].mode,
                                  &o, status, &M, &same);
            if (strcmp(same, "yes")) rc = 4;
            if (csv)
            {
                printf("%d,cuda,-,%s,%s,cuda,block,%d,%s,%zu,%d,%.6f,%.6f,%.6f,%s,%d,"
                       "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,call",
                       WM_SCHEMA, family, law, BLOCKS[i], STRATEGIES[s].name, N, t.reps,
                       t.min, t.med, t.spread, same, status,
                       t.min, t.q1, t.q3, t.p95, t.max, t.mean, t.iqr, N);
                print_acc(&M); printf("\n");
            }
            else
                printf(" %9.3f", t.med);
        }
        if (!csv) printf("\n");
    }

    /* host staging threads, ring only */
    if (!csv) printf("\n  host staging threads (ASYNC_RING only -- inert elsewhere)\n  %8s %9s\n",
                     "threads", "ring");
    for (size_t i = 0; i < sizeof MT / sizeof MT[0]; i++) {
        matern_cuda_options o; memset(&o, 0, sizeof o); o.host_threads = MT[i];
        int status = 0;
        char label[64];
        snprintf(label, sizeof label, "host_threads%d/ring", MT[i]);
        const wm_time t = time_config(&cs, out, z_family, MATERN_CUDA_ASYNC_RING,
                                      &o, reps, &status, family, law, label);
        wm_metrics M; const char *same;
        status = score_config(&cs, out, ref, z_family, MATERN_CUDA_ASYNC_RING,
                              &o, status, &M, &same);
        if (strcmp(same, "yes")) rc = 4;
        if (csv)
        {
            printf("%d,cuda,-,%s,%s,cuda,host_threads,%d,ring,%zu,%d,%.6f,%.6f,%.6f,%s,%d,"
                   "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,call",
                   WM_SCHEMA, family, law, MT[i], N, t.reps,
                   t.min, t.med, t.spread, same, status,
                   t.min, t.q1, t.q3, t.p95, t.max, t.mean, t.iqr, N);
            print_acc(&M); printf("\n");
        }
        else
            printf("  %8d %9.3f\n", MT[i], t.med);
    }

    free(out); free(ref);
    wm_raw_close();
    corpus_dataset_free(&cs);
    matern_cuda_release();
    return rc;
}
