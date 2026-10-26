/* suppliers.c -- external K_nu suppliers on the corpus, assembled into M (direct, scaled,
 * logtrick; references/common/cpu_supplier.h) and scored like the library rows, with the assembly
 * in the `backend` column (see the accompanying paper, reference implementations). */
#include "common/numa_bind.h"        /* first: sets _GNU_SOURCE */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "corpus_dataset.h"
#include "common/bench_metrics.h"
#include "common/bench_timing.h"
#include "common/bench_progress.h"
#include "cpu_supplier.h"
#include "cpu_supplier_registry.h"
#include "build_info.h"

/* The ULP mode of the scoring call and of the `ulp_mode` column. */
#define BENCH_ACCURACY_ULP_MODE BENCH_ULP_MODE_DEFAULT

/* `--build-info`: the build record of this binary and of every linked supplier object,
 * relayed by `detect --build-info`. */
static void print_build_info(void)
{
    BENCH_BUILD_INFO_COMMON(stdout, "cpu_suppliers");
    bench_build_info_kv(stdout, "cpu_suppliers", "ulp_mode", bench_ulp_mode_name(BENCH_ACCURACY_ULP_MODE));
    bench_build_info_cpu_supplier_c(stdout);
#ifdef CPU_SUPPLIER_WITH_CXX
    bench_build_info_cpu_supplier_cxx(stdout);
#else
    bench_build_info_kv(stdout, "cpu_supplier_cxx", "compiler", "not built (no C++ compiler at build time)");
#endif
#ifdef CPU_SUPPLIER_WITH_GSL
    bench_build_info_cpu_supplier_gsl(stdout);
#else
    bench_build_info_kv(stdout, "cpu_supplier_gsl", "headers", "none (not built: no GSL at build time)");
#endif
}

/* Supplier list: references/common/cpu_supplier_registry.h, shared with threads_cpu.c. */
#define entry cpu_supplier_entry
static int build_list(entry *e, int cap) { return cpu_supplier_build_list(e, cap); }

enum { ASM_DIRECT, ASM_SCALED, ASM_LOGTRICK, ASM_COUNT };
static const char *asm_name(int a)
{ return a == ASM_DIRECT ? "direct" : a == ASM_SCALED ? "scaled" : "logtrick"; }

static double assemble(const cpu_supplier *s, int a, int has_length_scale,
                       double h, double nu, double scale, double z)
{
    if (has_length_scale) {
        switch (a) {
            case ASM_DIRECT:   return cpu_supplier_direct(s, h, nu, scale);
            case ASM_SCALED:   return cpu_supplier_scaled(s, h, nu, scale);
            default:           return cpu_supplier_logtrick(s, h, nu, scale);
        }
    }
    switch (a) {
        case ASM_DIRECT:   return cpu_supplier_direct_z(s, z, nu);
        case ASM_SCALED:   return cpu_supplier_scaled_z(s, z, nu);
        default:           return cpu_supplier_logtrick_z(s, z, nu);
    }
}

/* --time-points: 0 times the whole law. */
static size_t time_points_cap = 0;

/* Times the work that is scored (same assembly, points and order) as whole calls, as the
 * library rows are timed. */
static bench_time time_assembly(const cpu_supplier *sp, int a, int has_length_scale,
                             const corpus_dataset *cs, double *out, size_t n,
                             int reps, size_t *timed_points,
                             int ladder_mode,
                             const char *sp_label, const char *fam_label,
                             const char *law_label, const char *asm_label)
{
    /* The point cap applies outside the ladder only. */
    if (!ladder_mode && time_points_cap && n > time_points_cap) n = time_points_cap;
    *timed_points = n;
    double *t = (double *)malloc((size_t)reps * sizeof *t);
    if (!t) { bench_time z; memset(&z, 0, sizeof z); return z; }
    bench_sched_begin();
    for (int w = 0; w < BENCH_WARMUP_REPS; w++)
        for (size_t i = 0; i < n; i++)
            out[i] = assemble(sp, a, has_length_scale, has_length_scale ? cs->set.h[i] : 0.0,
                              cs->set.order[i],
                              has_length_scale ? cs->set.length_scale[i] : 0.0,
                              has_length_scale ? 0.0 : cs->set.argument[i]);
    for (int r = 0; r < reps; r++) {
        double t0 = bench_now_s();
        for (size_t i = 0; i < n; i++)
            out[i] = assemble(sp, a, has_length_scale, has_length_scale ? cs->set.h[i] : 0.0,
                              cs->set.order[i],
                              has_length_scale ? cs->set.length_scale[i] : 0.0,
                              has_length_scale ? 0.0 : cs->set.argument[i]);
        t[r] = (bench_now_s() - t0) * 1e9 / (double)n;
    }
    /* Before the reduce, which sorts in place: the raw samples keep their order. */
    bench_raw_emit("cpu", sp_label, fam_label, law_label, asm_label, n, t, reps);
    bench_time r = bench_time_reduce(t, reps);
    free(t);
    return r;
}

/* --ladder-pass-from S: one pass over [0, top) per repetition, chunk bounds doubling from
 * S, with the clock read after each chunk, so rung N is the time for the first N points.
 * T receives one bench_time per rung. */
static size_t ladder_pass_from = 0;

static void time_assembly_pass(const cpu_supplier *sp, int a, int has_length_scale,
                               const corpus_dataset *cs, double *out, size_t S,
                               size_t top, int reps, const char *sp_label,
                               const char *fam_label, const char *law_label,
                               const char *asm_label, bench_time *T)
{
    int k = 0;
    size_t bnd[40];
    for (size_t b = S; b <= top && k < 40; b <<= 1) bnd[k++] = b;
    double *t = (double *)malloc((size_t)k * (size_t)reps * sizeof *t);
    if (!t) { memset(T, 0, (size_t)k * sizeof *T); return; }
    bench_sched_begin();
    for (int r = -BENCH_WARMUP_REPS; r < reps; r++) {         /* r < 0: untimed warm-up */
        size_t lo = 0;
        const double t0 = bench_now_s();
        for (int j = 0; j < k; j++) {
            for (size_t i = lo; i < bnd[j]; i++)
                out[i] = assemble(sp, a, has_length_scale, has_length_scale ? cs->set.h[i] : 0.0,
                                  cs->set.order[i],
                                  has_length_scale ? cs->set.length_scale[i] : 0.0,
                                  has_length_scale ? 0.0 : cs->set.argument[i]);
            if (r >= 0) t[(size_t)j * (size_t)reps + (size_t)r] = (bench_now_s() - t0) * 1e9 / (double)bnd[j];
            lo = bnd[j];
        }
    }
    /* every rung of the pass carries the pass's scheduler record */
    const bench_sched mark = bench_sched_mark;
    for (int j = 0; j < k; j++) {
        double *tj = t + (size_t)j * (size_t)reps;
        bench_sched_mark = mark; bench_sched_marked = 1;
        bench_raw_emit("cpu", sp_label, fam_label, law_label, asm_label, bnd[j], tj, reps);
        T[j] = bench_time_reduce(tj, reps);
    }
    free(t);
}

/* Supplier ladder: the rungs of the library ladder; a row stops when a rung exceeds the
 * budget (see main), so the top rung reached is the `n` of its last row. */
static void ladder_header(void)
{
    printf("schema,device,supplier,family,law,backend,n,ns_min,ns_q1,ns_med,"
           "ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_spread,reps,ns_points,timing_scope,"
           "ladder_mode,ladder_pass_from\n");
}

/* mode: `rung` (timed on its own) or `pass` (read from the cumulative pass) */
static void ladder_row(const char *sup, const char *fam, const char *law,
                       const char *assembly, size_t n, const bench_time *T,
                       const char *mode)
{
    printf("%d,cpu,%s,%s,%s,%s,%zu,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%zu,call,%s,%zu\n",
           BENCH_SCHEMA, sup, fam, law, assembly, n,
           T->min, T->q1, T->med, T->q3, T->p95, T->max, T->mean, T->iqr,
           T->spread, T->reps, n, mode, ladder_pass_from);
}

static void csv_header(void)
{
    printf("schema,device,supplier,family,law,backend,ulp_mode,n,n_scored,n_excluded,"
           "nonfinite,ref_zero,ref_subnormal,exact,min_bits,u_min,u_q1,u_med,u_q3,"
           "u_p99,u_p999,u_max,u_mean,u_iqr,u_fence,beyond_fence,u_med_nontrivial,"
           "rel_max,abs_max,worst_nu,worst_arg,ns_min,ns_q1,ns_med,ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_spread,reps,ns_points,timing_scope,n_file\n");
}

static void csv_row(const char *supplier, const char *fam, const char *law,
                    const char *assembly, const bench_metrics *M, const bench_time *T,
                    size_t ns_points)
{
    printf("%d,cpu,%s,%s,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%zu,call,%zu\n",
           BENCH_SCHEMA, supplier, fam, law, assembly, bench_ulp_mode_name(BENCH_ACCURACY_ULP_MODE),
           M->n, M->n_scored, M->n_excluded, M->n_nonfinite,
           M->n_ref_zero, M->n_ref_subnormal, M->n_exact, M->min_bits,
           M->u_min, M->u_q1, M->u_med, M->u_q3, M->u_p99, M->u_p999, M->u_max,
           M->u_mean, M->u_iqr, M->u_fence_hi, M->n_beyond_fence,
           M->u_med_nontrivial, M->rel_max, M->abs_max,
           M->n_worst ? M->worst[0].nu : 0.0, M->n_worst ? M->worst[0].arg : 0.0,
           T->min, T->q1, T->med, T->q3, T->p95, T->max, T->mean, T->iqr,
           T->spread, T->reps, ns_points, M->n_file);
}

/* Every law of the corpus: the eight accuracy laws and the `size_*` prefix ladders. */
static const char *ALL_LAWS[] = { "grid","random","small_z","near_int","half_int",
                                  "large_z","large_nu","seam",
                                  "size_random","size_half",
                                  "size_random_large","size_half_large" };
#define N_LAWS ((int)(sizeof ALL_LAWS / sizeof *ALL_LAWS))

static void usage(FILE *o, const char *p)
{
    fprintf(o, "usage: %s [--family ...] [--law all|...]\n"
           "          [--supplier NAME (--list prints the names)]\n"
           "          [--format text|csv] [--data-dir data] [--max-points N]\n"
           "          [--time-points N] [--reps N] [--list] [--build-info]\n"
           "          [--ladder [--ladder-max N] [--ladder-budget SECONDS]\n"
           "                    [--ladder-pass-from S]]\n", p);
}

int main(int argc, char **argv)
{
    const char *families = "z_nu,h_nu_s", *laws = "all", *root = "data";
    const char *format = "text", *only = NULL;
    size_t maxpts = 0;
    /* The library stage's fixed repetition count: a minimum over a different number of
     * samples is a different estimator. */
    int reps = BENCH_REPS_MIN_DEFAULT;
    int ladder = 0;
    /* The ladder is bounded by the dataset and the budget; --ladder-max caps it. */
    size_t ladder_max = (size_t)-1;
    /* A row climbs while one rung fits the budget; the rungs double, so a whole ladder
     * costs about twice the budget. */
    double ladder_budget = 20.0;   /* 0 disables it -- see the break below */

    for (int a = 1; a < argc; a++)
        if (!strcmp(argv[a], "--build-info")) { print_build_info(); return 0; }

    for (int a = 1; a < argc; a++) {
        const char *nx = (a + 1 < argc) ? argv[a+1] : NULL;
        if      (!strcmp(argv[a],"--family")   && nx) families = argv[++a];
        else if (!strcmp(argv[a],"--law")      && nx) laws     = argv[++a];
        else if (!strcmp(argv[a],"--supplier") && nx) only     = argv[++a];
        else if (!strcmp(argv[a],"--format")   && nx) format   = argv[++a];
        else if (!strcmp(argv[a],"--data-dir") && nx) root     = argv[++a];
        else if (!strcmp(argv[a],"--max-points") && nx) maxpts = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--time-points") && nx) time_points_cap = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--reps") && nx) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--ladder")) ladder = 1;
        else if (!strcmp(argv[a],"--ladder-max") && nx) ladder_max = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--ladder-budget") && nx) ladder_budget = atof(argv[++a]);
        else if (!strcmp(argv[a],"--ladder-pass-from") && nx) ladder_pass_from = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--list")) {
            entry e[16]; int n = build_list(e, 16);
            printf("%d supplier(s) in this build:\n", n);
            for (int i = 0; i < n; i++)
                printf("  %-20s %-28s scaled=%s\n", e[i].label, e[i].s->origin,
                       cpu_supplier_has_scaled(e[i].s) ? "yes" : "no");
            cpu_supplier_print_absent();
            return 0;
        }
        else if (!strcmp(argv[a],"--help")) { usage(stdout, argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); usage(stderr, argv[0]); return 2; }
    }
    if (corpus_family_check(families)) return 2;

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

    char why[256]; int nodes = 0, best = bench_best_node(&nodes);
    bench_expect_single_thread = 1;
    const int st = bench_bind_node(best < 0 ? 0 : best, why, sizeof why);
    if (strcmp(format,"csv") != 0) printf("# placement: %s\n", why);
    bench_bind_warn(st, why);

    entry sup[16]; const int nsup = build_list(sup, 16);
    if (only) {                       /* an unknown supplier name is an error */
        int found = 0;
        for (int i = 0; i < nsup; i++) if (!strcmp(only, sup[i].label)) { found = 1; break; }
        if (!found) {
            fprintf(stderr, "suppliers: no supplier named '%s' in this build.\n"
                            "  Available:", only);
            for (int i = 0; i < nsup; i++) fprintf(stderr, " %s", sup[i].label);
            fprintf(stderr, "\n  (--list shows them with their origins)\n");
            return 3;
        }
    }
    size_t rows_emitted = 0;
    bench_raw_open(getenv("BENCH_TIMING_RAW"));
    bench_detail_open(getenv("BENCH_ACCURACY_DETAIL"));
    if (!strcmp(format,"csv")) { if (ladder) ladder_header(); else csv_header(); }

    char famb[64]; snprintf(famb, sizeof famb, "%s", families);
    char *fsave = NULL;
    for (char *fs = strtok_r(famb, ",", &fsave); fs; fs = strtok_r(NULL, ",", &fsave)) {
        for (int li = 0; li < N_LAWS; li++) {
            const char *law = ALL_LAWS[li];
            /* `all` means the eight accuracy laws; the `size_*` ladders (distributions of
             * `random` and `half_int`) are selected by name. */
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
            /* between measurements: before the law is loaded and warmed up */
            if (!ladder) bench_progress("    %s/%s", fs, law);
            corpus_dataset cs;
            if ((ladder && ladder_max != (size_t)-1)
                    ? corpus_dataset_load_prefix(&cs, root, fs, law, ladder_max)   /* the rungs need no more */
                    : corpus_dataset_load(&cs, root, fs, law, maxpts))
                { fprintf(stderr, "corpus: no data for %s/%s under %s -- dataset skipped\n", fs, law, root); continue; }
            const size_t n = cs.set.count;
            const int has_length_scale = cs.set.has_length_scale;
            double *out = (double *)malloc(n * sizeof *out);

            for (int si = 0; si < nsup; si++) {
                if (only && strcmp(only, sup[si].label)) continue;
                for (int a = 0; a < ASM_COUNT; a++) {
                    /* a supplier without a scaled K gets no scaled row */
                    if (a == ASM_SCALED && !cpu_supplier_has_scaled(sup[si].s)) continue;
                    if (ladder) bench_progress("    %s/%s  %s", fs, law, asm_name(a));
                    for (size_t i = 0; i < n; i++)
                        out[i] = assemble(sup[si].s, a, has_length_scale,
                                          has_length_scale ? cs.set.h[i] : 0.0,
                                          cs.set.order[i],
                                          has_length_scale ? cs.set.length_scale[i] : 0.0,
                                          has_length_scale ? 0.0 : cs.set.argument[i]);
                    bench_metrics M;
                    bench_metrics_compute(&M, out, cs.set.reference_high,
                                       cs.set.reference_low, cs.accuracy_bits,
                                       cs.set.order,
                                       has_length_scale ? cs.set.h : cs.set.argument,
                                       has_length_scale ? cs.set.length_scale : NULL,
                                       n, BENCH_ACCURACY_ULP_MODE);
                    M.n_file = cs.file_points;
                    /* the detail sidecar (bench_metrics.h), before the timed loop reuses `out` */
                    bench_detail_emit("cpu", sup[si].label, fs, law, asm_name(a), &M, out,
                                   cs.set.order, has_length_scale ? cs.set.h : cs.set.argument,
                                   has_length_scale ? cs.set.length_scale : NULL, n);
                    if (ladder) {
                        /* Rungs below ladder_pass_from are timed one by one; the cost
                         * of the last one projects the pass. */
                        const size_t lim = n < ladder_max ? n : ladder_max;
                        double last_cost = 0.0; size_t last_nn = 0; int stopped = 0;
                        for (size_t nn = 256; nn <= lim; nn <<= 1) {
                            if (ladder_pass_from && nn >= ladder_pass_from) break;
                            const double rung_t0 = bench_now_s();
                            size_t lp = 0;
                            bench_time LT = time_assembly(sup[si].s, a, has_length_scale, &cs,
                                                       out, nn, reps, &lp, 1,
                                                       sup[si].label, fs, law,
                                                       asm_name(a));
                            rows_emitted++;
                            if (!strcmp(format,"csv"))
                                ladder_row(sup[si].label, fs, law, asm_name(a),
                                           nn, &LT, "rung");
                            else
                                printf("  ladder %-18s %-9s %-10s %-9s n=%-9zu "
                                       "%.3f ns/elem\n", sup[si].label, fs, law,
                                       asm_name(a), nn, LT.min);
                            last_cost = bench_now_s() - rung_t0; last_nn = nn;
                            /* Stop once a rung costs more than the budget (the next
                             * is twice the work); a budget of 0 climbs to the top. */
                            if (ladder_budget > 0.0 && last_cost > ladder_budget) {
                                if (strcmp(format,"csv") != 0)
                                    printf("  ladder %-18s %-9s %-10s %-9s "
                                           "stopped after n=%zu (rung cost "
                                           "%.1f s > budget %.1f s)\n",
                                           sup[si].label, fs, law, asm_name(a),
                                           nn, last_cost, ladder_budget);
                                stopped = 1;
                                break;
                            }
                        }
                        /* The pass climbs while the cost projected linearly in N from
                         * the last timed rung fits the budget.  With no rung below S,
                         * rung S is evaluated as often as a timed rung, unrecorded, to
                         * give the projection its cost. */
                        if (ladder_pass_from && !stopped && ladder_pass_from <= lim) {
                            if (!last_nn && ladder_budget > 0.0) {
                                const double c0 = bench_now_s();
                                for (int r = 0; r < BENCH_WARMUP_REPS + reps; r++)
                                    for (size_t i = 0; i < ladder_pass_from; i++)
                                        out[i] = assemble(sup[si].s, a, has_length_scale,
                                                          has_length_scale ? cs.set.h[i] : 0.0,
                                                          cs.set.order[i],
                                                          has_length_scale ? cs.set.length_scale[i] : 0.0,
                                                          has_length_scale ? 0.0 : cs.set.argument[i]);
                                last_cost = bench_now_s() - c0; last_nn = ladder_pass_from;
                            }
                            size_t top = ladder_pass_from;
                            while (2 * top <= lim && (ladder_budget <= 0.0 ||
                                   last_cost * (double)top / (double)last_nn <= ladder_budget))
                                top *= 2;
                            bench_time PT[40];
                            time_assembly_pass(sup[si].s, a, has_length_scale, &cs, out,
                                               ladder_pass_from, top, reps,
                                               sup[si].label, fs, law, asm_name(a), PT);
                            int j = 0;
                            for (size_t nn = ladder_pass_from; nn <= top; nn <<= 1, j++) {
                                rows_emitted++;
                                if (!strcmp(format,"csv"))
                                    ladder_row(sup[si].label, fs, law, asm_name(a),
                                               nn, &PT[j], "pass");
                                else
                                    printf("  ladder %-18s %-9s %-10s %-9s n=%-9zu "
                                           "%.3f ns/elem (pass)\n", sup[si].label, fs,
                                           law, asm_name(a), nn, PT[j].min);
                            }
                        }
                        continue;
                    }
                    size_t tpts = 0;
                    bench_time T = time_assembly(sup[si].s, a, has_length_scale, &cs, out,
                                              n, reps, &tpts, 0, sup[si].label,
                                              fs, law, asm_name(a));
                    rows_emitted++;
                    if (!strcmp(format,"csv"))
                        csv_row(sup[si].label, fs, law, asm_name(a), &M, &T, tpts);
                    else
                        printf("%-18s %-9s %-10s %-9s n=%-8zu scored=%-8zu "
                               "ULPmax=%-12.4g relmax=%-11.3e nonfin=%-8zu "
                               "ns/elem=%.3f (min of %d)\n",
                               sup[si].label, fs, law, asm_name(a), M.n, M.n_scored,
                               M.u_max, M.rel_max, M.n_nonfinite, T.min, T.reps);
                }
            }
            free(out);
            corpus_dataset_free(&cs);
        }
    }
    bench_raw_close();
    bench_detail_close();
    if (rows_emitted == 0) {
        fprintf(stderr, "suppliers: nothing matched --family '%s' --law '%s'.\n", families, laws);
        return 4;
    }
    return 0;
}
