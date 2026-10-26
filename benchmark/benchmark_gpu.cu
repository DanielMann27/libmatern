/* benchmark_gpu.cu -- the CUDA backend on the corpus under the four transfer modes,
 * scored and written like the CPU rows (device=cuda).  The modes must agree bit for
 * bit; a mismatch sets the exit status.  Without a CUDA device it exits 0, no rows. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "corpus_dataset.h"
}
#include "common/bench_metrics.h"
#include "common/bench_timing.h"
#include "common/bench_cuda_call.h"
#include "matern.h"

/* In build/cuda_bench.o, not in matern.h: the device routing compiled in. */
extern "C" int matern_cuda_route_flags(void);

/* The ULP mode of the scoring call and of the `ulp_mode` column. */
#define BENCH_ACCURACY_ULP_MODE BENCH_ULP_MODE_DEFAULT
#include "common/build_info_cuda.h"

struct mode_entry { int mode; const char *name; };
static const mode_entry MODES[] = {
    { MATERN_CUDA_SYNC,         "sync"         },
    { MATERN_CUDA_ASYNC_DIRECT, "async-direct" },
    { MATERN_CUDA_ASYNC_RING,   "async-ring"   },
    { MATERN_CUDA_ASYNC_MAPPED, "async-mapped" },
};
static const int N_MODES = (int)(sizeof MODES / sizeof *MODES);

/* Every corpus law, the size (prefix-ladder) laws included so that they can be
 * named; `--law all` excludes them (see main), as suppliers.c does. */
static const char *ALL_LAWS[] = { "grid","random","small_z","near_int","half_int",
                                  "large_z","large_nu","seam",
                                  "size_random","size_half",
                                  "size_random_large","size_half_large" };
static const int N_LAWS = (int)(sizeof ALL_LAWS / sizeof *ALL_LAWS);

/* One public call over the first n points, for bench_time_cuda_call. */
typedef struct { const corpus_dataset *cs; double *out; size_t n; int mode; } bg_call;
static int bg_call_once(void *p)
{
    const bg_call *c = (const bg_call *)p;
    if (c->cs->set.has_length_scale)
        matern_covariance_cuda(c->out, c->cs->set.h, c->cs->set.order,
                               c->cs->set.length_scale, c->n, c->mode);
    else
        matern_covariance_cuda_z(c->out, c->cs->set.argument, c->cs->set.order, c->n, c->mode);
    return 0;   /* as before: a failing mode is caught by its scored call */
}

/* Times the scored call with common/bench_cuda_call.h, transfers included, since the
 * transfer modes differ in them. */
static bench_time time_mode(const corpus_dataset *cs, double *out, size_t n,
                         int mode, int has_length_scale, int reps, size_t cap,
                         size_t *timed_points, const char *fam_label,
                         const char *law_label, const char *mode_label)
{
    (void)has_length_scale;
    /* The whole law unless --time-points caps it; the count goes to ns_points. */
    if (cap && n > cap) n = cap;
    *timed_points = n;
    double *t = (double *)malloc((size_t)reps * sizeof *t);
    if (!t) { bench_time z; memset(&z, 0, sizeof z); return z; }
    bg_call c = { cs, out, n, mode };
    bench_time_cuda_call(bg_call_once, &c, n, reps, t);
    bench_raw_emit("cuda", "-", fam_label, law_label, mode_label, n, t, reps);
    bench_time R = bench_time_reduce(t, reps);
    free(t);
    return R;
}

/* The ladder: rungs are prefixes of the `size_*` laws, as on the CPU ladder, and
 * ns_points equals n.  No comment may stand between the string literals below:
 * benchmark/tests/run_tests.sh counts header fields from them. */
static void ladder_header(void)
{
    printf("schema,device,supplier,family,law,backend,n,ns_min,ns_q1,ns_med,"
           "ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_spread,reps,nu_med,arg_med,"
           "timing_scope,ns_points\n");
}

static void ladder_row(const char *fam, const char *law, const char *mode,
                       size_t n, const bench_time *T, double nu_med, double arg_med)
{
    printf("%d,cuda,-,%s,%s,%s,%zu,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,"
           "call,%zu\n",
           BENCH_SCHEMA, fam, law, mode, n,
           T->min, T->q1, T->med, T->q3, T->p95, T->max, T->mean, T->iqr,
           T->spread, T->reps, nu_med, arg_med, n);
}

static int cmp_dbl_(const void *a, const void *b)
{ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

/* Per-rung input median, which shows the input mixture along the ladder. */
static double med_of(const double *v, size_t n)
{
    if (!n) return 0.0;
    double *c = (double *)malloc(n * sizeof *c);
    if (!c) return 0.0;
    memcpy(c, v, n * sizeof *c);
    qsort(c, n, sizeof *c, cmp_dbl_);
    double m = c[n/2];
    free(c);
    return m;
}

static void csv_header(void)
{
    printf("schema,device,supplier,family,law,backend,ulp_mode,n,n_scored,n_excluded,"
           "nonfinite,ref_zero,ref_subnormal,exact,min_bits,u_min,u_q1,u_med,u_q3,"
           "u_p99,u_p999,u_max,u_mean,u_iqr,u_fence,beyond_fence,u_med_nontrivial,"
           "rel_max,abs_max,worst_nu,worst_arg,ns_min,ns_q1,ns_med,ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_spread,reps,ns_points,timing_scope,n_file\n");
}

static void csv_row(const char *fam, const char *law, const char *mode,
                    const bench_metrics *M, const bench_time *T, size_t ns_points)
{
    printf("%d,cuda,-,%s,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%zu,call,%zu\n",
           BENCH_SCHEMA, fam, law, mode, bench_ulp_mode_name(BENCH_ACCURACY_ULP_MODE),
           M->n, M->n_scored, M->n_excluded, M->n_nonfinite,
           M->n_ref_zero, M->n_ref_subnormal, M->n_exact, M->min_bits,
           M->u_min, M->u_q1, M->u_med, M->u_q3, M->u_p99, M->u_p999, M->u_max,
           M->u_mean, M->u_iqr, M->u_fence_hi, M->n_beyond_fence,
           M->u_med_nontrivial, M->rel_max, M->abs_max,
           M->n_worst ? M->worst[0].nu : 0.0, M->n_worst ? M->worst[0].arg : 0.0,
           T->min, T->q1, T->med, T->q3, T->p95, T->max, T->mean, T->iqr,
           T->spread, T->reps, ns_points, M->n_file);
}

static void usage(FILE *o, const char *p)
{
    fprintf(o, "usage: %s [--family ...] [--law all|...] [--format text|csv]\n"
           "          [--data-dir data] [--max-points N] [--time-points N] [--reps N]\n"
           "          [--ladder [--ladder-max N]] [--no-mode-check] [--print-build]\n"
           "          [--build-info]\n", p);
}

int main(int argc, char **argv)
{
    const char *families = "z_nu,h_nu_s", *laws = "all", *root = "data";
    const char *format = "text";
    size_t maxpts = 0;
    size_t time_points = 0;          /* 0 == time the whole law */
    int ladder = 0;
    size_t ladder_max = (size_t)-1;  /* --ladder-max: the top rung (default: corpus size) */
    int check_modes = 1;
    /* the pipeline's fixed repetition count, settable with --reps */
    int reps = BENCH_REPS_MAX_DEFAULT;

    /* --build-info, before any device call, so that it runs without a GPU;
     * build/detect relays it into PROVENANCE.txt. */
    for (int a = 1; a < argc; a++)
        if (!strcmp(argv[a], "--build-info")) {
            bench_build_info_cuda("gpu_benchmark");
            bench_build_info_kv(stdout, "gpu_benchmark", "ulp_mode", bench_ulp_mode_name(BENCH_ACCURACY_ULP_MODE));
            return 0;
        }

    for (int a = 1; a < argc; a++) {
        const char *nx = (a + 1 < argc) ? argv[a+1] : NULL;
        if      (!strcmp(argv[a],"--family")   && nx) families = argv[++a];
        else if (!strcmp(argv[a],"--law")      && nx) laws     = argv[++a];
        else if (!strcmp(argv[a],"--format")   && nx) format   = argv[++a];
        else if (!strcmp(argv[a],"--data-dir") && nx) root     = argv[++a];
        else if (!strcmp(argv[a],"--max-points") && nx) maxpts = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--time-points") && nx) time_points = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--ladder")) ladder = 1;
        else if (!strcmp(argv[a],"--ladder-max") && nx) ladder_max = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--reps") && nx) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--no-mode-check")) check_modes = 0;
        /* The routing flags come from the library, where the device routing is fixed. */
        else if (!strcmp(argv[a],"--print-build")) {
            int f = matern_cuda_route_flags();
            printf("levin=%s\n", (f & 32) ? "on" : "off");   /* MATERN_ROUTE_LEVIN_BIT */
            printf("route_flags=%d\n", f);
            return 0;
        }
        else if (!strcmp(argv[a],"--help")) { usage(stdout, argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); usage(stderr, argv[0]); return 2; }
    }
    if (corpus_family_check(families)) return 2;

    /* --max-points is a stride over the file, and the ladder needs prefixes. */
    if (ladder && maxpts) {
        fprintf(stderr, "--max-points decimates the corpus (it is a STRIDE), which breaks the\n"
                        "ladder's prefix property and changes the input mixture.  Refusing.\n"
                        "  drop --max-points, or drop --ladder.\n");
        return 2;
    }

    if (!matern_cuda_available()) {
        fprintf(stderr,
            "benchmark_gpu: no CUDA device -- GPU rows are ABSENT from this run.\n"
            "               This is not a failure: every GPU figure is simply not\n"
            "               produced here, and the run record says so.\n");
        return 0;                       /* no device: no rows, not an error */
    }
    fprintf(stderr, "benchmark_gpu: device %s\n", matern_cuda_device_name());

    bench_raw_open(getenv("BENCH_TIMING_RAW"));
    bench_detail_open(getenv("BENCH_ACCURACY_DETAIL"));
    if (!strcmp(format,"csv")) { if (ladder) ladder_header(); else csv_header(); }

    char famb[64]; snprintf(famb, sizeof famb, "%s", families);
    char *fsave = NULL;
    int mismatches = 0;
    for (char *fs = strtok_r(famb, ",", &fsave); fs; fs = strtok_r(NULL, ",", &fsave)) {
        for (int li = 0; li < N_LAWS; li++) {
            const char *law = ALL_LAWS[li];
            /* `all` excludes the size laws, which repeat the distributions of
             * `random` and `half_int`; they stay reachable by name for --ladder. */
            {   const size_t _ln = strlen(law);
                const int _large = _ln > 6 && !strcmp(law + _ln - 6, "_large");
                if (!strcmp(laws,"all") && (_large || !strcmp(law,"size_random") ||
                                            !strcmp(law,"size_half")))
                    continue;
            }
            if (strcmp(laws,"all") != 0) {
                char tmp[512]; snprintf(tmp,sizeof tmp,"%s",laws);
                int hit = 0; char *ls = NULL;
                for (char *t = strtok_r(tmp,",",&ls); t; t = strtok_r(NULL,",",&ls))
                    if (!strcmp(t,law)) { hit = 1; break; }
                if (!hit) continue;
            }
            corpus_dataset cs;
            if ((ladder && ladder_max != (size_t)-1)
                    ? corpus_dataset_load_prefix(&cs, root, fs, law, ladder_max)   /* the rungs need no more */
                    : corpus_dataset_load(&cs, root, fs, law, maxpts))
                { fprintf(stderr, "corpus: no data for %s/%s under %s -- dataset skipped\n", fs, law, root); continue; }
            const size_t n = cs.set.count;
            const int has_length_scale = cs.set.has_length_scale;
            double *out  = (double *)malloc(n * sizeof *out);
            double *ref0 = (double *)malloc(n * sizeof *ref0);   /* the reference mode's answer */

            if (ladder) {
                /* Prefixes from 2^8 to the corpus size, one curve per transfer mode. */
                for (int m = 0; m < N_MODES; m++) {
                    for (size_t nn = 256; nn <= n && nn <= ladder_max; nn <<= 1) {
                        double *t = (double *)malloc((size_t)reps * sizeof *t);
                        if (!t) break;
                        bg_call c = { &cs, out, nn, MODES[m].mode };
                        bench_time_cuda_call(bg_call_once, &c, nn, reps, t);
                        bench_raw_emit("cuda", "-", fs, law, MODES[m].name, nn,
                                    t, reps);
                        bench_time T = bench_time_reduce(t, reps);
                        free(t);
                        double numed  = med_of(cs.set.order, nn);
                        double argmed = med_of(has_length_scale ? cs.set.h : cs.set.argument, nn);
                        if (!strcmp(format,"csv"))
                            ladder_row(fs, law, MODES[m].name, nn, &T, numed, argmed);
                        else
                            printf("  ladder %-9s %-12s %-13s n=%-9zu %.4f ns/elem\n",
                                   fs, law, MODES[m].name, nn, T.min);
                    }
                }
                corpus_dataset_free(&cs); free(out); free(ref0);
                continue;
            }

            /* The first mode that succeeds is the comparison reference. */
            int ref_mode = -1;
            for (int m = 0; m < N_MODES; m++) {
                /* NaN first: a point left unwritten counts as non-finite. */
                for (size_t k = 0; k < n; k++) out[k] = NAN;
                int rc = has_length_scale
                    ? matern_covariance_cuda(out, cs.set.h, cs.set.order,
                                             cs.set.length_scale, n, MODES[m].mode)
                    : matern_covariance_cuda_z(out, cs.set.argument, cs.set.order,
                                               n, MODES[m].mode);
                if (rc != 0) {
                    fprintf(stderr, "  %s/%s %s: returned %d, row omitted\n",
                            fs, law, MODES[m].name, rc);
                    continue;
                }
                /* The modes may differ only in how the data moves. */
                if (ref_mode < 0) { memcpy(ref0, out, n * sizeof *out); ref_mode = m; }
                else if (check_modes) {
                    size_t bad = 0;
                    /* bitwise, so two NaNs at the same point agree */
                    for (size_t i = 0; i < n; i++)
                        if (memcmp(&out[i], &ref0[i], sizeof out[i]) != 0) bad++;
                    if (bad) {
                        fprintf(stderr, "  *** %s/%s %s differs from %s on %zu of %zu "
                                "points -- the modes are NOT bit-identical\n",
                                fs, law, MODES[m].name, MODES[ref_mode].name, bad, n);
                        mismatches++;
                    }
                }
                /* Scored against hi + lo: a reference rounded to one double is 0 where
                 * M is below the double range and would score a zero result as exact. */
                bench_metrics M;
                bench_metrics_compute(&M, out, cs.set.reference_high, cs.set.reference_low,
                                   cs.accuracy_bits, cs.set.order,
                                   has_length_scale ? cs.set.h : cs.set.argument,
                                   has_length_scale ? cs.set.length_scale : NULL,
                                   n, BENCH_ACCURACY_ULP_MODE);
                M.n_file = cs.file_points;
                /* the detail sidecar (bench_metrics.h), before the timed loop reuses `out` */
                bench_detail_emit("cuda", "-", fs, law, MODES[m].name, &M, out,
                               cs.set.order, has_length_scale ? cs.set.h : cs.set.argument,
                               has_length_scale ? cs.set.length_scale : NULL, n);
                size_t tpts = 0;
                bench_time T = time_mode(&cs, out, n, MODES[m].mode,
                                      cs.set.has_length_scale,
                                      reps, time_points, &tpts,
                                      fs, law, MODES[m].name);
                if (!strcmp(format,"csv"))
                    csv_row(fs, law, MODES[m].name, &M, &T, tpts);
                else printf("%-9s %-10s %-14s n=%-8zu scored=%-8zu ULPmax=%-11.4g "
                            "med=%-8.3f nonfin=%zu\n",
                            fs, law, MODES[m].name, M.n, M.n_scored, M.u_max,
                            M.u_med, M.n_nonfinite);
            }
            free(out); free(ref0);
            corpus_dataset_free(&cs);
        }
    }
    bench_raw_close();
    bench_detail_close();
    matern_cuda_release();
    if (mismatches)
        fprintf(stderr, "benchmark_gpu: %d transfer-mode mismatch(es) -- see above\n",
                mismatches);
    return mismatches ? 1 : 0;
}
