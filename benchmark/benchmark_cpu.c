/* benchmark_cpu.c -- stage 2: accuracy and timing of the library over the corpus (see the
 * accompanying paper, accuracy and stability metrics; timing methodology).  Built with
 * -DBENCH_FOLD it is build/fold_cpu, the driver of the prefactor-fold stage.  The exit status
 * is structural only (2 arguments, 3 backend, 4 no match); no error figure affects it. */
#include "common/numa_bind.h"        /* first: sets _GNU_SOURCE */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include "corpus_dataset.h"
#include "common/bench_metrics.h"
#include "common/bench_timing.h"
#include "common/bench_progress.h"
#include "matern.h"
#include "../references/common/build_info.h"

static const char *ALL_LAWS[] = { "grid","random","small_z","near_int","half_int",
                                  "large_z","large_nu","seam","size_random","size_half",
                                  "size_random_large","size_half_large" };
/* The `_large` ladder laws (2^25 points) are not part of `--law all`; they are selected
 * by name. */
static int law_is_large(const char *law)
{ const size_t n = strlen(law); return n > 6 && !strcmp(law + n - 6, "_large"); }
#define N_LAWS ((int)(sizeof ALL_LAWS / sizeof *ALL_LAWS))

/* Timing primitives: common/bench_timing.h, shared with the supplier stage. */
#define now_s bench_now_s

#ifdef BENCH_FOLD
/* Fold-stage backends: scalar (through dispatch); scalar-control, the shipped source built
 * like the twin (Makefile, FOLD_VARIANTS); scalar-unfolded-direct/-explog, the unfolded twin
 * (src/backends/scalar/scalar_internal.h).  Control and twin run the instruction-set build
 * that dispatch selects for `scalar` (matern_scalar_isa()) and need no workspace. */
typedef int (*fold_entry)(double *, const double *, const double *, const double *,
                          size_t, matern_workspace *);
const char *matern_scalar_isa(void);
#define FOLD_DECLARE(v, isa) \
    int matern_evaluate_scalar_##v##_##isa(double *, const double *, const double *, \
                                           const double *, size_t, matern_workspace *); \
    int matern_evaluate_scalar_##v##_##isa##_z(double *, const double *, const double *, \
                                               const double *, size_t, matern_workspace *);
#define FOLD_DECLARE_ALL(v) FOLD_DECLARE(v, avx512) FOLD_DECLARE(v, avx2) \
                            FOLD_DECLARE(v, fma) FOLD_DECLARE(v, portable)
FOLD_DECLARE_ALL(control)
FOLD_DECLARE_ALL(unfolded_direct)
FOLD_DECLARE_ALL(unfolded_explog)

typedef struct { const char *isa; fold_entry h, z; } fold_build;
#define FOLD_BUILD(v, isa) { #isa, matern_evaluate_scalar_##v##_##isa, \
                             matern_evaluate_scalar_##v##_##isa##_z }
#define FOLD_BUILDS(v) { FOLD_BUILD(v, avx512), FOLD_BUILD(v, avx2), \
                         FOLD_BUILD(v, fma), FOLD_BUILD(v, portable) }
static const struct { const char *backend; fold_build build[4]; } FOLD_BACKENDS[] = {
    { "scalar-control",         FOLD_BUILDS(control)         },
    { "scalar-unfolded-direct", FOLD_BUILDS(unfolded_direct) },
    { "scalar-unfolded-explog", FOLD_BUILDS(unfolded_explog) },
};
#define FOLD_N_BACKENDS ((int)(sizeof FOLD_BACKENDS / sizeof *FOLD_BACKENDS))
#define FOLD_DEFAULT_BACKENDS "scalar,scalar-control,scalar-unfolded-direct,scalar-unfolded-explog"

/* The live entry of a control or twin backend; NULL for `scalar` (dispatch). */
static fold_entry fold_h = NULL, fold_z = NULL;
#endif

/* Returns the library status; only the scored call checks it. */
static int evaluate(const corpus_dataset *cs, double *out, size_t n,
                    matern_workspace *ws)
{
#ifdef BENCH_FOLD
    if (fold_h) {
        if (cs->set.has_length_scale)
            return fold_h(out, cs->set.h, cs->set.order, cs->set.length_scale, n, ws);
        return fold_z(out, cs->set.argument, cs->set.order, cs->set.order, n, ws);
    }
#endif
    if (cs->set.has_length_scale)
        return matern_covariance(out, cs->set.h, cs->set.order, cs->set.length_scale, n, ws);
    return matern_covariance_z(out, cs->set.argument, cs->set.order, n, ws);
}

/* Points [lo, lo + n) in one call, for the cumulative ladder pass. */
static int evaluate_range(const corpus_dataset *cs, double *out, size_t lo, size_t n,
                          matern_workspace *ws)
{
#ifdef BENCH_FOLD
    if (fold_h) {
        if (cs->set.has_length_scale)
            return fold_h(out + lo, cs->set.h + lo, cs->set.order + lo,
                          cs->set.length_scale + lo, n, ws);
        return fold_z(out + lo, cs->set.argument + lo, cs->set.order + lo,
                      cs->set.order + lo, n, ws);
    }
#endif
    if (cs->set.has_length_scale)
        return matern_covariance(out + lo, cs->set.h + lo, cs->set.order + lo,
                                 cs->set.length_scale + lo, n, ws);
    return matern_covariance_z(out + lo, cs->set.argument + lo, cs->set.order + lo, n, ws);
}

/* Whether the library uses a workspace on this backend: avx512 and avx2 do; the scalar
 * builds and the width-1 entries (matched, matched_fma) use none (dispatch.c, vector.c).
 * tests/test_workspace_need.c checks matern_evaluate_scalar and matern_evaluate_matched. */
static int backend_builds_workspace(const char *backend)
{
    return strncmp(backend, "avx", 3) == 0;      /* avx512 and avx2 */
}

/* Timing: on avx512 and avx2 every repetition creates and destroys its own workspace
 * inside the timer.
 * bench_rep_policy repeats until a budget is spent, between a floor and a ceiling; the
 * pipeline sets floor == ceiling, so every row is a minimum over the same count. */
#define cmp_d bench_cmp_d

/* Last-level cache size and sharing CPUs, set once in main, printed on every ladder row. */
static long bench_llc_bytes = 0;
static int  bench_llc_share = 1;
/* Backend label of the ladder's raw-repetition dump, set by run_ladder. */
static const char *bench_raw_backend = "-";

/* --ladder-max: the top rung, below the corpus size (default: no cap). */
static size_t ladder_max = (size_t)-1;

/* --ladder-pass-from S: rungs N >= S come from one cumulative pass (time_pass); 0 times
 * every rung on its own. */
static size_t ladder_pass_from = 0;

static bench_time time_call(const corpus_dataset *cs, double *out, size_t n,
                         bench_rep_policy pol, int ws_needed)
{
    double *t = (double *)malloc((size_t)pol.max_reps * sizeof *t);
    bench_time r; memset(&r, 0, sizeof r);
    bench_sched_begin();
    for (int w = 0; w < BENCH_WARMUP_REPS; w++) {                       /* untimed warm-up */
        matern_workspace *ws = NULL;
        if (ws_needed) matern_workspace_create(&ws, 0);
        evaluate(cs, out, n, ws);
        if (ws_needed) matern_workspace_destroy(ws);
    }
    int i = 0; double spent = 0.0;
    while (i < pol.max_reps && (i < pol.min_reps || spent < pol.budget)) {
        matern_workspace *ws = NULL;
        double t0 = now_s();
        if (ws_needed) matern_workspace_create(&ws, 0);
        evaluate(cs, out, n, ws);
        if (ws_needed) matern_workspace_destroy(ws);
        double dt = now_s() - t0;
        spent += dt;
        t[i++] = dt * 1e9 / (double)n;                  /* ns per element */
    }
    /* Raw samples are written before the shared reducer, which sorts in place. */
    bench_raw_emit("cpu", "-", cs->family, cs->law, bench_raw_backend, n, t, i);
    r = bench_time_reduce(t, i);
    free(t);
    return r;
}

/* time_call() with one workspace created outside the timer; the difference is the
 * per-call setup cost. */
static bench_time time_call_ws_reused(const corpus_dataset *cs, double *out, size_t n,
                              bench_rep_policy pol, int ws_needed)
{
    double *t = (double *)malloc((size_t)pol.max_reps * sizeof *t);
    bench_time r; memset(&r, 0, sizeof r);
    matern_workspace *ws = NULL;
    if (ws_needed) matern_workspace_create(&ws, 0);
    for (int w = 0; w < BENCH_WARMUP_REPS; w++) evaluate(cs, out, n, ws);
    int i = 0; double spent = 0.0;
    while (i < pol.max_reps && (i < pol.min_reps || spent < pol.budget)) {
        double t0 = now_s();
        evaluate(cs, out, n, ws);
        double dt = now_s() - t0;
        spent += dt;
        t[i++] = dt * 1e9 / (double)n;
    }
    if (ws_needed) matern_workspace_destroy(ws);
    /* Not written to timing_raw: it would share time_call's (family, law, backend, n) key. */
    r = bench_time_reduce(t, i);
    free(t);
    return r;
}

/* Workspace create+destroy alone, in ns per pair. */
static bench_time time_workspace(int reps)
{
    double *t = (double *)malloc((size_t)reps * sizeof *t);
    bench_time r; memset(&r, 0, sizeof r); r.reps = reps;
    /* Same untimed warm-up as time_call and time_call_ws_reused. */
    for (int w = 0; w < BENCH_WARMUP_REPS; w++) {
        matern_workspace *ws;
        matern_workspace_create(&ws, 0);
        matern_workspace_destroy(ws);
    }
    for (int i = 0; i < reps; i++) {
        matern_workspace *ws;
        double t0 = now_s();
        matern_workspace_create(&ws, 0);
        matern_workspace_destroy(ws);
        t[i] = (now_s() - t0) * 1e9;                    /* ns for the pair */
    }
    r = bench_time_reduce(t, reps);
    free(t);
    return r;
}

/* Cumulative pass: one pass over [0, top) per repetition, one library call per chunk
 * (chunk bounds doubling from S) with the clock read after each, so rung N is the time
 * for the first N points.  reused = 0: a workspace per chunk call inside the timer, as
 * time_call; reused = 1: one workspace built outside, as time_call_ws_reused. */
static void time_pass(const corpus_dataset *cs, double *out, size_t S, size_t top,
                      bench_rep_policy pol, int ws_needed, int reused, bench_time *T)
{
    int k = 0;
    size_t bnd[40];
    for (size_t b = S; b <= top && k < 40; b <<= 1) bnd[k++] = b;
    const int reps = pol.max_reps;
    double *t = (double *)malloc((size_t)k * (size_t)reps * sizeof *t);
    matern_workspace *wsr = NULL;
    if (reused && ws_needed) matern_workspace_create(&wsr, 0);
    if (!reused) bench_sched_begin();
    for (int r = -BENCH_WARMUP_REPS; r < reps; r++) {         /* r < 0: untimed warm-up */
        size_t lo = 0;
        const double t0 = now_s();
        for (int j = 0; j < k; j++) {
            matern_workspace *ws = wsr;
            if (!reused && ws_needed) matern_workspace_create(&ws, 0);
            evaluate_range(cs, out, lo, bnd[j] - lo, ws);
            if (!reused && ws_needed) matern_workspace_destroy(ws);
            if (r >= 0) t[(size_t)j * (size_t)reps + (size_t)r] = (now_s() - t0) * 1e9 / (double)bnd[j];
            lo = bnd[j];
        }
    }
    if (wsr) matern_workspace_destroy(wsr);
    /* every rung of the pass carries the pass's scheduler record */
    const bench_sched mark = bench_sched_mark;
    for (int j = 0; j < k; j++) {
        double *tj = t + (size_t)j * (size_t)reps;
        if (!reused) { bench_sched_mark = mark; bench_sched_marked = 1; }
        if (!reused) bench_raw_emit("cpu", "-", cs->family, cs->law, bench_raw_backend, bnd[j], tj, reps);
        T[j] = bench_time_reduce(tj, reps);
    }
    free(t);
}

/* Ladder: rungs 2^8 .. corpus size, rung N being the first N points of a `size_*` law.
 * The medians of nu and of the argument are printed per rung, so that a change of the
 * input mixture along the ladder is visible. */
static double median_of(const double *v, size_t n)
{
    if (!n) return 0.0;
    double *t = (double *)malloc(n * sizeof *t);
    memcpy(t, v, n * sizeof *t);
    qsort(t, n, sizeof *t, cmp_d);
    double m = t[n / 2];
    free(t);
    return m;
}

/* One ladder row: build series t, reused series t_reused, workspace alone w, and the
 * timing mode (`rung` or `pass`). */
static void ladder_row(const corpus_dataset *cs, const char *backend, const char *format,
                       size_t n, const bench_time *tp, const bench_time *wp, const bench_time *wsp,
                       const char *mode)
{
    const bench_time t = *tp, t_reused = *wp, w = *wsp;
    const double *nu  = cs->set.order;
    const double *arg = cs->set.has_length_scale ? cs->set.h : cs->set.argument;
    /* Without a workspace both loops are the same: the difference is noise. */
    double setup_per = t.min - t_reused.min;
    double qn = median_of(nu, n), qa = median_of(arg, n);
    if (strcmp(format, "csv") == 0)
        printf("%d,cpu,-,%s,%s,%s,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%ld,%d,%.0f,"
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%zu\n",
               BENCH_SCHEMA, cs->family, cs->law, backend, n,
               t.min, t_reused.min, setup_per, (double)t.reps, t_reused.med, qn, qa,
               w.min,
               t_reused.q1, t_reused.q3, t_reused.p95, t_reused.max, t_reused.mean, t_reused.iqr,
               t_reused.spread, bench_llc_bytes, bench_llc_share,
               (double)n * (double)sizeof(double)
                   * (cs->set.has_length_scale ? 4.0 : 3.0),
               t.med, t.q1, t.q3, t.p95, t.max, t.mean, t.iqr, t.spread,
               w.med, mode, ladder_pass_from);
    else
        printf("  %-9zu %-10.3f %-10.3f %-10.3f %-9d %-11.0f %-11.5g %-11.5g %s\n",
               n, t.min, t_reused.min, setup_per, t.reps, w.min, qn, qa, mode);
}

static void run_ladder(const corpus_dataset *cs, const char *backend,
                       bench_rep_policy pol, const char *format)
{
    bench_raw_backend = backend;
    const size_t full = cs->set.count;

    /* The CSV header is printed once, by the parent (print_ladder_header). */
    if (strcmp(format, "csv") != 0) {
        printf("\n# ladder %s/%s  %s   (prefix rungs; min over a %.2f s budget,"
               " %d..%d reps)\n",
               cs->family, cs->law, backend, pol.budget, pol.min_reps, pol.max_reps);
        printf("  %-9s %-10s %-10s %-10s %-9s %-11s %-11s %-11s %s\n",
               "N", "build", "reused", "setup", "reps", "ws alone", "nu median", "arg median",
               "timed as");
    }

    /* Warm the allocator before the first rung. */
    const int ws_needed = backend_builds_workspace(backend);
    if (ws_needed)
        for (int w = 0; w < 8; w++) {
            matern_workspace *ws; matern_workspace_create(&ws, 0);
            matern_workspace_destroy(ws);
        }

    double *out = (double *)bench_array_alloc(full * sizeof *out);
    /* The top rung is the corpus size or --ladder-max; `e <= 30` only guards the shift.
     * Rungs from --ladder-pass-from up are timed in one cumulative pass afterwards. */
    size_t top = 0;
    for (int e = 8; e <= 30; e++) {
        size_t n = (size_t)1 << e;
        if (n > full || n > ladder_max) break;
        top = n;
        if (ladder_pass_from && n >= ladder_pass_from) continue;
        bench_time t = time_call(cs, out, n, pol, ws_needed);   /* workspace per call */
        bench_time t_reused = time_call_ws_reused(cs, out, n, pol, ws_needed); /* warm ws     */
        /* Workspace create+destroy alone, same repetition count, at every rung; zero
         * without a workspace.  ns_setup_per_elem matches it only while N fits one tile. */
        bench_time w = ws_needed ? time_workspace(pol.max_reps)
                              : (bench_time){0};
        ladder_row(cs, backend, format, n, &t, &t_reused, &w, "rung");
    }
    if (ladder_pass_from && top >= ladder_pass_from) {
        bench_time T[40], TR[40];
        time_pass(cs, out, ladder_pass_from, top, pol, ws_needed, 0, T);
        time_pass(cs, out, ladder_pass_from, top, pol, ws_needed, 1, TR);
        int j = 0;
        for (size_t n = ladder_pass_from; n <= top; n <<= 1, j++) {
            bench_time w = ws_needed ? time_workspace(pol.max_reps) : (bench_time){0};
            ladder_row(cs, backend, format, n, &T[j], &TR[j], &w, "pass");
        }
    }
    free(out);
    if (strcmp(format, "csv") != 0)
        puts("  (build = a new workspace per call on avx512 and avx2, the library's timing.\n"
             "   reused = the same loop with the workspace created once.  On the other\n"
             "   backends both time the same call.  setup = build - reused, the difference\n"
             "   of the two minima.  nu/arg medians that change down the column show that\n"
             "   the input mixture, not only N, changes along the ladder.)");
}

static void print_accuracy_text(const char *fam, const char *law, const char *backend,
                                const bench_metrics *M)
{
    printf("%-9s %-12s %-8s n=%-8zu scored=%-8zu\n", fam, law, backend, M->n, M->n_scored);
    printf("   ULP  max=%-11.4f p999=%-9.4f p99=%-9.4f q3=%-8.4f med=%-8.4f q1=%-8.4f min=%.4f\n",
           M->u_max, M->u_p999, M->u_p99, M->u_q3, M->u_med, M->u_q1, M->u_min);
    printf("   BOX  iqr=%-9.4f fence=%-9.4f beyond=%-8zu mean=%-9.4f med_nontrivial=%.4f\n",
           M->u_iqr, M->u_fence_hi, M->n_beyond_fence, M->u_mean, M->u_med_nontrivial);
    printf("   POP  nonfinite=%-6zu excluded=%-6zu ref_zero=%-7zu ref_subnormal=%-7zu exact=%-7zu min_bits=%d\n",
           M->n_nonfinite, M->n_excluded, M->n_ref_zero, M->n_ref_subnormal,
           M->n_exact, M->min_bits);
    printf("   2ND  rel_max=%-12.4e abs_max=%-12.4e at nu=%.6g arg=%.6g\n",
           M->rel_max, M->abs_max, M->rel_max_nu, M->rel_max_arg);
    if (M->n_worst)
        printf("   WORST %+.4f ulp at nu=%.8g arg=%.8g  M=%.6e bits=%d\n",
               M->worst[0].ulp, M->worst[0].nu, M->worst[0].arg,
               M->worst[0].ref, M->worst[0].bits);
}

/* Columns: docs/benchmark.md, "Output".  Must match ladder_row(). */
static void print_ladder_header(void)
{
    printf("schema,device,supplier,family,law,backend,n,ns_ws_per_call_min,ns_ws_reused_min,"
           "ns_setup_per_elem,reps,ns_ws_reused_med,nu_med,arg_med,ns_workspace,"
           "ns_ws_reused_q1,ns_ws_reused_q3,ns_ws_reused_p95,ns_ws_reused_max,"
           "ns_ws_reused_mean,ns_ws_reused_iqr,ns_ws_reused_spread,"
           "llc_bytes,llc_shared_cpus,ws_bytes,"
           "ns_ws_per_call_med,ns_ws_per_call_q1,ns_ws_per_call_q3,"
           "ns_ws_per_call_p95,ns_ws_per_call_max,ns_ws_per_call_mean,"
           "ns_ws_per_call_iqr,ns_ws_per_call_spread,"
           "ns_workspace_med,"
           "ladder_mode,ladder_pass_from\n");
}

/* --time-points: 0 times the whole law. */
static size_t time_points_cap = 0;

/* Times the library on the law being scored, with the repetitions and point cap of the
 * supplier stage; on avx512 and avx2 every repetition builds its own workspace, as a
 * caller pays for it. */
static bench_time time_accuracy_law(const corpus_dataset *cs, double *out, size_t n,
                                 int reps, size_t *timed_points,
                                 const char *fam, const char *law,
                                 const char *backend)
{
    if (time_points_cap && n > time_points_cap) n = time_points_cap;
    *timed_points = n;
    const int ws_needed = backend_builds_workspace(backend);
    double *t = (double *)malloc((size_t)reps * sizeof *t);
    if (!t) { bench_time z; memset(&z, 0, sizeof z); return z; }
    bench_sched_begin();
    for (int w = 0; w < BENCH_WARMUP_REPS; w++) {          /* warmed like the reps */
        matern_workspace *ws = NULL;
        if (ws_needed) matern_workspace_create(&ws, 0);
        evaluate(cs, out, n, ws);
        if (ws_needed) matern_workspace_destroy(ws);
    }
    for (int r = 0; r < reps; r++) {
        matern_workspace *ws = NULL;
        double t0 = now_s();
        if (ws_needed) matern_workspace_create(&ws, 0);
        evaluate(cs, out, n, ws);
        if (ws_needed) matern_workspace_destroy(ws);
        t[r] = (now_s() - t0) * 1e9 / (double)n;
    }
    /* Before the reduce, which sorts in place: the raw samples keep their order. */
    bench_raw_emit("cpu", "-", fam, law, backend, n, t, reps);
    bench_time R = bench_time_reduce(t, reps);
    free(t);
    return R;
}

static void print_csv_header(void)
{
    printf("schema,device,supplier,family,law,backend,ulp_mode,n,n_scored,n_excluded,nonfinite,"
           "ref_zero,ref_subnormal,exact,min_bits,"
           "u_min,u_q1,u_med,u_q3,u_p99,u_p999,u_max,u_mean,u_iqr,u_fence,beyond_fence,"
           "u_med_nontrivial,rel_max,abs_max,worst_nu,worst_arg,"
           "ns_min,ns_q1,ns_med,ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_spread,reps,ns_points,timing_scope,n_file\n");
}

static void print_csv_row(const char *fam, const char *law, const char *backend,
                          bench_ulp_mode mode, const bench_metrics *M,
                          const bench_time *T, size_t ns_points)
{
    /* device and supplier are fixed (cpu, "-") for the library's CPU rows. */
    printf("%d,cpu,-,%s,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%zu,call,%zu\n",
           BENCH_SCHEMA, fam, law, backend, bench_ulp_mode_name(mode),
           M->n, M->n_scored, M->n_excluded, M->n_nonfinite,
           M->n_ref_zero, M->n_ref_subnormal, M->n_exact, M->min_bits,
           M->u_min, M->u_q1, M->u_med, M->u_q3, M->u_p99, M->u_p999, M->u_max,
           M->u_mean, M->u_iqr, M->u_fence_hi, M->n_beyond_fence,
           M->u_med_nontrivial, M->rel_max, M->abs_max,
           M->n_worst ? M->worst[0].nu : 0.0, M->n_worst ? M->worst[0].arg : 0.0,
           T->min, T->q1, T->med, T->q3, T->p95, T->max, T->mean, T->iqr,
           T->spread, T->reps, ns_points, M->n_file);
}

#ifdef BENCH_FOLD
/* Selects fold-stage backend `b` for this process.  Nonzero (exit 3) for an unknown name,
 * when dispatch does not resolve `scalar`, or when a control or twin does not behave as built. */
static int fold_select(const char *b)
{
    setenv("MATERN_BACKEND", "scalar", 1);
    { double probe_o, probe_z = 1.0, probe_nu = 1.5;
      matern_covariance_z(&probe_o, &probe_z, &probe_nu, 1, NULL); }
    const char *live = matern_backend_name();
    if (strcmp(live, "scalar") != 0) {
        fprintf(stderr, "fold_cpu: asked for backend 'scalar' but '%s' is live.\n"
                        "  Refusing to label %s's numbers as the scalar backend's.\n",
                live, live);
        return 1;
    }
    fold_h = fold_z = NULL;
    if (!strcmp(b, "scalar")) return 0;
    const char *isa = matern_scalar_isa();
    for (int i = 0; i < FOLD_N_BACKENDS; i++) {
        if (strcmp(b, FOLD_BACKENDS[i].backend) != 0) continue;
        for (int j = 0; j < 4; j++)
            if (!strcmp(isa, FOLD_BACKENDS[i].build[j].isa)) {
                fold_h = FOLD_BACKENDS[i].build[j].h;
                fold_z = FOLD_BACKENDS[i].build[j].z;
            }
        if (!fold_h) {
            fprintf(stderr, "fold_cpu: no %s build for the scalar instruction set '%s'\n", b, isa);
            return 1;
        }
        /* At (z, nu) = (1, 200.25) Gamma(nu) overflows: the direct twin is non-finite,
         * the exp-log twin finite, and the control equals the scalar backend bit for bit. */
        double ref = NAN, got = NAN, z1 = 1.0, nu1 = 200.25;
        matern_covariance_z(&ref, &z1, &nu1, 1, NULL);
        fold_z(&got, &z1, &nu1, &nu1, 1, NULL);
        int ok = isfinite(ref);
        if (!strcmp(b, "scalar-control"))         ok = ok && memcmp(&ref, &got, sizeof ref) == 0;
        if (!strcmp(b, "scalar-unfolded-direct")) ok = ok && !isfinite(got);
        if (!strcmp(b, "scalar-unfolded-explog")) ok = ok && isfinite(got);
        if (!ok) {
            fprintf(stderr, "fold_cpu: %s does not behave as built at (z, nu) = (1, 200.25):"
                            " %.17g against the scalar backend's %.17g.  Refusing to measure.\n",
                    b, got, ref);
            fold_h = fold_z = NULL;
            return 1;
        }
        return 0;
    }
    fprintf(stderr, "fold_cpu: unknown backend '%s'; the stage's backends are %s\n",
            b, FOLD_DEFAULT_BACKENDS);
    return 1;
}
#endif

static void usage(FILE *o, const char *p)
{
#ifdef BENCH_FOLD
    fprintf(o, "fold_cpu: the prefactor-fold stage.  benchmark_cpu built with -DBENCH_FOLD:\n"
               "  accuracy rows only (--metric accuracy, no --ladder), for the backends\n"
               "  %s (the default).\n\n", FOLD_DEFAULT_BACKENDS);
#endif
    fprintf(o, "usage: %s [--family z_nu,h_nu_s] [--law all|grid,...] \n"
           "          [--backend avx512,avx2,matched_fma,matched,scalar] [--metric accuracy|speed|all]\n"
           "          [--ulp-mode up|down|muller|straddle] [--max-points N] [--time-points N]\n"
           "          [--reps N] [--max-reps MAX] [--budget SECONDS]\n"
           "            --reps fixes the repetition count (default 25); with --max-reps and\n"
           "            --budget, repetitions continue until the budget is spent, between N\n"
           "            and MAX.  The count used is recorded per row.\n"
           "            --time-points times only the first N points of each dataset (default:\n"
           "            every point); the count is recorded in ns_points.\n"
           "          [--format text|csv] [--data-dir data] [--no-bind] [--ladder]\n"
           "  --ladder times 2^8..the corpus size as PREFIX rungs (2^20 for size_random |\n"
           "          size_half, 2^25 for size_random_large | size_half_large)\n"
           "          [--ladder-max N]  caps the top rung at N\n"
           "          [--ladder-pass-from S]  times the rungs N >= S (a power of two) in one\n"
           "            cumulative pass instead of one by one\n"
           "          [--print-build]  print `levin_vector=` and `levin_scalar=` and exit\n"
           "          [--build-info]  print this binary's build record and exit\n"
           "          [--help]  print this text and exit\n", p);
#ifdef BENCH_FOLD
    const char *backend = "as above";
#else
    const char *backend = "avx512";
#endif
    fprintf(o, "  defaults: --family z_nu,h_nu_s --law all --backend %s --metric accuracy\n"
               "            --ulp-mode up --reps 25 --format text --data-dir data; all points\n"
               "            (no --max-points, no --time-points); no --ladder-max; every rung\n"
               "            timed on its own (no --ladder-pass-from).\n", backend);
#ifndef BENCH_FOLD
    fprintf(o, "            benchmark/run_benchmark.sh passes --backend, by default with every\n"
               "            backend of avx512,avx2,matched_fma,matched,scalar this CPU runs.\n");
#endif
}

int main(int argc, char **argv)
{
    /* the last-level cache of CPU 0, read once */
    bench_llc_bytes = bench_llc_bytes_shared(0, &bench_llc_share);
#ifdef BENCH_FOLD
    const char *families = "z_nu,h_nu_s", *laws = "all", *backends = FOLD_DEFAULT_BACKENDS;
#else
    const char *families = "z_nu,h_nu_s", *laws = "all", *backends = "avx512";
#endif
    const char *metric = "accuracy", *format = "text", *root = "data";
    int bind = 1, ladder = 0; size_t maxpts = 0;
    bench_rep_policy pol = { BENCH_REPS_MIN_DEFAULT, BENCH_REPS_MAX_DEFAULT, BENCH_BUDGET_DEFAULT };
    bench_ulp_mode mode = BENCH_ULP_MODE_DEFAULT;

    for (int a = 1; a < argc; a++) {
        const char *nx = (a + 1 < argc) ? argv[a+1] : NULL;
        if      (!strcmp(argv[a],"--family")  && nx) families = argv[++a];
        else if (!strcmp(argv[a],"--law")     && nx) laws     = argv[++a];
        else if (!strcmp(argv[a],"--backend") && nx) backends = argv[++a];
        else if (!strcmp(argv[a],"--metric")  && nx) metric   = argv[++a];
        else if (!strcmp(argv[a],"--format")  && nx) format   = argv[++a];
        else if (!strcmp(argv[a],"--data-dir")&& nx) root     = argv[++a];
        /* --reps sets floor and ceiling; a later --max-reps raises the ceiling only */
        else if (!strcmp(argv[a],"--reps")    && nx) pol.min_reps = pol.max_reps = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--max-reps")&& nx) pol.max_reps = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--budget")  && nx) pol.budget   = atof(argv[++a]);
        else if (!strcmp(argv[a],"--max-points") && nx) maxpts = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--time-points") && nx) time_points_cap = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--no-bind")) bind = 0;
        else if (!strcmp(argv[a],"--ladder")) ladder = 1;
        else if (!strcmp(argv[a],"--ladder-max") && nx) ladder_max = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--ladder-pass-from") && nx) ladder_pass_from = (size_t)atol(argv[++a]);
        /* --print-build reports the compiled routing of both Levin routers; the Levin A/B
         * checks that its arms differ. */
        else if (!strcmp(argv[a],"--print-build")) {
            /* Independent macros: MATERN_WITHOUT_LEVIN (vector), MATERN_SCALAR_WITH_LEVIN (scalar). */
#ifdef MATERN_WITHOUT_LEVIN
            printf("levin_vector=off\n");
#else
            printf("levin_vector=on\n");
#endif
#ifdef MATERN_SCALAR_WITH_LEVIN
            printf("levin_scalar=on\n");
#else
            printf("levin_scalar=off\n");
#endif
#ifdef BENCH_FOLD
            printf("fold=on\n");
#endif
            return 0;
        }
        /* --build-info: this binary's build record, relayed by `detect --build-info`. */
        else if (!strcmp(argv[a],"--build-info")) {
#ifdef BENCH_FOLD
            BENCH_BUILD_INFO_COMMON(stdout, "cpu_fold");
            bench_build_info_kv(stdout, "cpu_fold", "twin_direct",
                             "M = 2 (z/2)^nu / Gamma(nu) * K_nu(z)");
            bench_build_info_kv(stdout, "cpu_fold", "twin_explog",
                             "M = exp((1-nu) ln 2 - lnGamma(nu) + nu ln z + ln K_nu(z))");
            bench_build_info_kv(stdout, "cpu_fold", "backends", FOLD_DEFAULT_BACKENDS);
#else
            BENCH_BUILD_INFO_COMMON(stdout, "cpu_benchmark");
#endif
            char um[160];
            snprintf(um, sizeof um, "%s (the default; --ulp-mode overrides it, and the "
                     "ulp_mode column records the mode of each row)",
                     bench_ulp_mode_name(BENCH_ULP_MODE_DEFAULT));
#ifdef BENCH_FOLD
            bench_build_info_kv(stdout, "cpu_fold", "ulp_mode", um);
#else
            bench_build_info_kv(stdout, "cpu_benchmark", "ulp_mode", um);
#endif
            return 0;
        }
        else if (!strcmp(argv[a],"--ulp-mode") && nx) {
            const char *m = argv[++a];
            if      (!strcmp(m,"up"))       mode = BENCH_ULP_UP;
            else if (!strcmp(m,"down"))     mode = BENCH_ULP_DOWN;
            else if (!strcmp(m,"muller"))   mode = BENCH_ULP_MULLER;
            else if (!strcmp(m,"straddle")) mode = BENCH_ULP_STRADDLE;
            else { fprintf(stderr, "unknown --ulp-mode %s\n", m); return 2; }
        }
        else if (!strcmp(argv[a],"--help")) { usage(stdout, argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); usage(stderr, argv[0]); return 2; }
    }
    if (corpus_family_check(families)) return 2;

#ifdef BENCH_FOLD
    /* The fold stage writes accuracy rows only. */
    if (ladder || strcmp(metric, "accuracy") != 0) {
        fprintf(stderr, "fold_cpu: the prefactor-fold stage writes accuracy rows only"
                        " (--metric accuracy, no --ladder)\n");
        return 2;
    }
#endif
    if (ladder_pass_from && (ladder_pass_from < 256 || (ladder_pass_from & (ladder_pass_from - 1)))) {
        fprintf(stderr, "--ladder-pass-from must be a power of two >= 256 (got %zu)\n", ladder_pass_from);
        return 2;
    }

    /* --max-points strides over the file, which breaks the ladder's prefix rungs. */
    if (ladder && maxpts) {
        fprintf(stderr, "--max-points decimates the corpus (it is a STRIDE), which breaks the\n"
                        "ladder's prefix property and changes the input mixture.  Refusing.\n"
                        "  drop --max-points, or drop --ladder.\n");
        return 2;
    }

    /* One backend per process: dispatch reads MATERN_BACKEND on the first call only and
     * caches it, so several backends run as re-executed children. */
    const int want_acc_hdr = (strcmp(metric,"speed") != 0) && !getenv("BENCH_CHILD");
    if (strchr(backends, ',')) {
        char list[128]; snprintf(list, sizeof list, "%s", backends);
        int rc = 0; char *save = NULL;
        /* Flush the header before forking: a block-buffered header would follow the
         * children's output. */
        if (!strcmp(format,"csv")) {
            if (ladder)             print_ladder_header();
            else if (want_acc_hdr)  print_csv_header();
            fflush(stdout);
        }
        for (char *b = strtok_r(list, ",", &save); b; b = strtok_r(NULL, ",", &save)) {
            char *child[32]; int c = 0;
            child[c++] = argv[0];
            for (int a = 1; a < argc && c < 28; a++) {
                if (!strcmp(argv[a], "--backend")) { a++; continue; }
                child[c++] = argv[a];
            }
            child[c++] = (char *)"--backend";
            child[c++] = b;
            child[c]   = NULL;
            pid_t pid = fork();
            if (pid == 0) { setenv("BENCH_CHILD","1",1); execv(argv[0], child); _exit(127); }
            int st = 0; waitpid(pid, &st, 0);
            if (st) rc = 1;
        }
        return rc;
    }

    bench_expect_single_thread = 1;
    if (bind) {
        char why[256]; int nodes = 0, best = bench_best_node(&nodes);
        const int st = bench_bind_node(best < 0 ? 0 : best, why, sizeof why);
        if (strcmp(format, "csv") != 0) printf("# placement: %s\n", why);
        bench_bind_warn(st, why);
    }

    const int want_acc   = strcmp(metric,"speed") != 0;
    /* `--ladder` requires `--metric speed`: accuracy and ladder rows have different columns. */
    if (ladder && want_acc) {
        fprintf(stderr, "benchmark_cpu: --ladder needs --metric speed, or the "
                        "accuracy rows and the ladder rows go into one file "
                        "under one header (two schemas, one file)\n");
        return 2;
    }
    const int want_speed = strcmp(metric,"accuracy") != 0;
    /* Only a process that is not a re-exec'd child prints the header. */
    if (!strcmp(format,"csv") && !getenv("BENCH_CHILD")) {
        if (ladder)        print_ladder_header();
        else if (want_acc) print_csv_header();
    }
    /* One raw file per backend process, <prefix>.<backend>, so children do not truncate
     * each other's files. */
    {
        const char *rp = getenv("BENCH_TIMING_RAW");
        if (rp && *rp) {
            char pth[512];
            snprintf(pth, sizeof pth, "%s.%s", rp, backends);
            bench_raw_open(pth);
        }
        /* the accuracy detail, one file per backend process, likewise */
        const char *dp = getenv("BENCH_ACCURACY_DETAIL");
        if (dp && *dp) {
            char pth[512];
            snprintf(pth, sizeof pth, "%s.%s", dp, backends);
            bench_detail_open(pth);
        }
    }

    /* strtok_r: the family, law and backend loops are nested.  No match is exit 4. */
    size_t rows_emitted = 0;
    char famb[64]; snprintf(famb, sizeof famb, "%s", families);
    char *fsave = NULL;
    for (char *fs = strtok_r(famb, ",", &fsave); fs; fs = strtok_r(NULL, ",", &fsave)) {
        char lawb[512]; snprintf(lawb, sizeof lawb, "%s", laws);
        for (int li = 0; li < N_LAWS; li++) {
            const char *law = ALL_LAWS[li];
            if (strcmp(laws,"all") == 0 && law_is_large(law)) continue;
            if (strcmp(laws,"all") != 0) {
                char tmp[512]; snprintf(tmp,sizeof tmp,"%s",laws);
                int hit = 0; char *lsave = NULL;
                for (char *t = strtok_r(tmp,",",&lsave); t; t = strtok_r(NULL,",",&lsave))
                    if (!strcmp(t,law)) { hit = 1; break; }
                if (!hit) continue;
            }
            corpus_dataset cs;
            if ((ladder && ladder_max != (size_t)-1)
                    ? corpus_dataset_load_prefix(&cs, root, fs, law, ladder_max)   /* the rungs need no more */
                    : corpus_dataset_load(&cs, root, fs, law, maxpts))
                { fprintf(stderr, "corpus: no data for %s/%s under %s -- dataset skipped\n", fs, law, root); continue; }
            size_t n = cs.set.count;
            double *out = (double *)bench_array_alloc(n * sizeof *out);

            char bkb[128]; snprintf(bkb, sizeof bkb, "%s", backends);
            char *bsave = NULL;
            for (char *b = strtok_r(bkb, ",", &bsave); b; b = strtok_r(NULL, ",", &bsave)) {
                /* before the block's warm-up, outside every timed interval */
                bench_progress("  %-8s %s/%s  %s", ladder ? "ladder" : want_acc ? "accuracy" : "speed",
                            fs, law, b);
#ifdef BENCH_FOLD
                if (fold_select(b) != 0) { free(out); corpus_dataset_free(&cs); return 3; }
#else
                setenv("MATERN_BACKEND", b, 1);   /* single backend per process */
                /* Dispatch falls back to its automatic choice for an unknown
                 * MATERN_BACKEND, so the live backend is checked after one probe call. */
                { double probe_o, probe_z = 1.0, probe_nu = 1.5;
                  matern_covariance_z(&probe_o, &probe_z, &probe_nu, 1, NULL);
                  const char *live = matern_backend_name();
                  if (strcmp(live, b) != 0) {
                      fprintf(stderr,
                          "benchmark_cpu: asked for backend '%s' but '%s' is live.\n"
                          "  Either the name is wrong, or this CPU cannot run it.\n"
                          "  Refusing to label %s's numbers as '%s'.\n", b, live, live, b);
                      free(out); corpus_dataset_free(&cs);
                      return 3;
                  } }
#endif
                if (want_acc) {
                    /* NaN prefill: a point the library does not write is non-finite and
                     * excluded, not scored on uninitialised memory. */
                    for (size_t k = 0; k < n; k++) out[k] = NAN;
                    const int ast = evaluate(&cs, out, n, NULL);
                    if (ast != MATERN_OK)
                        fprintf(stderr, "benchmark_cpu: %s/%s on %s returned status %d"
                                " -- row scored on what it wrote (unwritten = non-finite)\n",
                                fs, law, b, ast);
                    bench_metrics M;
                    bench_metrics_compute(&M, out, cs.set.reference_high, cs.set.reference_low,
                                       cs.accuracy_bits, cs.set.order,
                                       cs.set.has_length_scale ? cs.set.h : cs.set.argument,
                                       cs.set.length_scale, n, mode);
                    M.n_file = cs.file_points;
                    /* the detail sidecar (bench_metrics.h), before the timed loop reuses `out` */
                    bench_detail_emit("cpu", "-", fs, law, b, &M, out, cs.set.order,
                                   cs.set.has_length_scale ? cs.set.h : cs.set.argument,
                                   cs.set.has_length_scale ? cs.set.length_scale : NULL, n);
                    rows_emitted++;
                    size_t tpts = 0;
                    bench_time TA = time_accuracy_law(&cs, out, n, pol.max_reps, &tpts,
                                                   fs, law, b);
                    if (!strcmp(format,"csv"))
                        print_csv_row(fs, law, b, mode, &M, &TA, tpts);
                    else                        print_accuracy_text(fs, law, b, &M);
                }
                if (ladder) { run_ladder(&cs, b, pol, format); rows_emitted++; }
                if (want_speed && !ladder) {
                    rows_emitted++;
                    bench_time t = time_call(&cs, out, n, pol,
                                          backend_builds_workspace(b));
                    if (strcmp(format,"csv") != 0)
                        printf("   TIME %-8s min=%-9.3f med=%-9.3f max=%-9.3f spread=%.1f%% (ns/elem, %d reps)\n",
                               b, t.min, t.med, t.max, 100.0*(t.max-t.min)/t.min, t.reps);
                }
            }
            free(out);
            corpus_dataset_free(&cs);
        }
    }

    if (rows_emitted == 0) {
        fprintf(stderr, "benchmark_cpu: nothing matched --family '%s' --law '%s'."
                        "  Check the names; an empty table is not a result.\n",
                families, laws);
        return 4;
    }
    if (want_speed && strcmp(format,"csv") != 0) {
        bench_time w = time_workspace(30);
        printf("\n# workspace create+destroy alone: min=%.0f ns med=%.0f ns max=%.0f ns\n"
               "#   the workspace is tile-sized (~552 KiB), not N-sized: negligible at large N,\n"
               "#   dominant at small N, where the ladder starts.\n",
               w.min, w.med, w.max);
    }
    return 0;
}
