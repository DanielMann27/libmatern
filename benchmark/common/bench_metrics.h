/* bench_metrics.h -- accuracy statistics of one row (ULP primary; relative and absolute
 * error secondary, since relative error is undefined where the reference is 0; signed
 * histogram; worst points), the accuracy detail sidecar and the CSV schema number. */
#ifndef BENCH_METRICS_H
#define BENCH_METRICS_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "ulpmodes.h"

/* The histogram layout is part of the schema; benchmark/plots/runs.py duplicates it
 * (HIST_*): change both. */
#define BENCH_HIST_BINS   48                 /* log-spaced over [2^-10, 2^10) ulp */
#define BENCH_HIST_LO_LOG (-10.0)
#define BENCH_HIST_HI_LOG ( 10.0)
#define BENCH_WORST_KEEP  64                 /* located worst points kept */
/* The record layout version, the first column of every CSV row; readers refuse a
 * schema they do not accept, so benchmark/plots/runs.py SCHEMA_ACCEPTED must contain it.
 * Columns: docs/benchmark.md.  benchmark/tests/run_tests.sh reads the define line below. */
#define BENCH_SCHEMA      11

typedef struct {
    double ulp;                  /* signed */
    double nu, arg, aux;         /* nu, then z or h, s */
    double value, ref;
    int    bits;
    size_t index;
} bench_worst;

typedef struct {
    /* population */
    size_t n, n_scored, n_excluded, n_nonfinite;
    size_t n_ref_zero, n_ref_subnormal, n_exact;
    size_t n_beyond_fence;
    int    min_bits;
    /* Points of the law on disk (`n_file`); set by the caller after
     * bench_metrics_compute, which zeroes it. */
    size_t n_file;

    /* ULP, primary */
    double u_min, u_q1, u_med, u_q3, u_p99, u_p999, u_max, u_mean;
    double u_iqr, u_fence_hi;
    double u_med_nontrivial;     /* median over points whose reference is not 0 */

    /* secondary */
    double rel_max, abs_max;
    double rel_max_nu, rel_max_arg;

    /* signed histogram: [0] = exactly zero, [1..BINS] = log bins, [BINS+1] = over */
    long   hist_neg[BENCH_HIST_BINS + 2];
    long   hist_pos[BENCH_HIST_BINS + 2];

    bench_worst worst[BENCH_WORST_KEEP];
    size_t   n_worst;
} bench_metrics;

static inline int bench_cmp_double(const void *a, const void *b)
{ double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* Nearest rank on the exact sorted sample, index round(f*(m-1)): every reported
 * value occurred. */
static inline double bench_quantile(const double *sorted, size_t m, double f)
{ return m ? sorted[(size_t)(f * (double)(m - 1) + 0.5)] : 0.0; }

static inline int bench_hist_bin(double au)
{
    if (au == 0.0) return 0;
    double L = log2(au);
    if (L < BENCH_HIST_LO_LOG) return 1;                          /* underflow bin */
    if (L >= BENCH_HIST_HI_LOG) return BENCH_HIST_BINS + 1;       /* overflow bin */
    int b = 1 + (int)((L - BENCH_HIST_LO_LOG) *
                      (BENCH_HIST_BINS / (BENCH_HIST_HI_LOG - BENCH_HIST_LO_LOG)));
    if (b < 1) b = 1;
    if (b > BENCH_HIST_BINS) b = BENCH_HIST_BINS;
    return b;
}

/* `aux` may be NULL (a (z, nu) set has no scale parameter s). */
static inline void bench_metrics_compute(bench_metrics *M, const double *value,
                               const double *hi, const double *lo, const int *bits,
                               const double *nu, const double *arg, const double *aux,
                               size_t n, bench_ulp_mode mode)
{
    memset(M, 0, sizeof *M);
    M->n = n;
    M->min_bits = 1 << 30;

    double *U  = (double *)malloc(n ? n * sizeof *U : 1);   /* |ulp|, for quantiles */
    double *UN = (double *)malloc(n ? n * sizeof *UN : 1);  /* |ulp|, non-trivial only */
    size_t m = 0, mn = 0;
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (bits && bits[i] < M->min_bits) M->min_bits = bits[i];
        if (!isfinite(value[i])) { M->n_nonfinite++; M->n_excluded++; continue; }

        const double r = hi[i] + lo[i];
        if (hi[i] == 0.0)                                   M->n_ref_zero++;
        else if (fabs(hi[i]) < DBL_MIN)                     M->n_ref_subnormal++;

        const double su = bench_ulp_signed(value[i], hi[i], lo[i], mode);
        const double au = fabs(su);
        if (au == 0.0) M->n_exact++;

        long *H = (su < 0.0) ? M->hist_neg : M->hist_pos;
        H[bench_hist_bin(au)]++;

        U[m++] = au; sum += au;
        if (hi[i] != 0.0) UN[mn++] = au;

        const double ae = fabs(value[i] - hi[i] - lo[i]);
        if (ae > M->abs_max) M->abs_max = ae;
        if (r != 0.0) {
            double re = ae / fabs(r);
            if (re > M->rel_max) { M->rel_max = re;
                                   M->rel_max_nu = nu[i]; M->rel_max_arg = arg[i]; }
        }
        /* The k worst by |ulp|, by insertion; the stored ulp is signed, so both
         * sides compare magnitudes (benchmark/tests/test_metrics.c). */
        if (M->n_worst < BENCH_WORST_KEEP || au > fabs(M->worst[M->n_worst - 1].ulp)) {
            size_t pos = (M->n_worst < BENCH_WORST_KEEP) ? M->n_worst : BENCH_WORST_KEEP - 1;
            while (pos > 0 && fabs(M->worst[pos - 1].ulp) < au) {
                M->worst[pos] = M->worst[pos - 1]; pos--;
            }
            M->worst[pos].ulp = su; M->worst[pos].nu = nu[i];
            M->worst[pos].arg = arg[i]; M->worst[pos].aux = aux ? aux[i] : 0.0;
            M->worst[pos].value = value[i]; M->worst[pos].ref = r;
            M->worst[pos].bits = bits ? bits[i] : 0; M->worst[pos].index = i;
            if (M->n_worst < BENCH_WORST_KEEP) M->n_worst++;
        }
    }

    M->n_scored = m;
    if (!m) { free(U); free(UN); M->min_bits = 0; return; }

    qsort(U, m, sizeof *U, bench_cmp_double);
    M->u_min  = U[0];                      M->u_q1   = bench_quantile(U, m, 0.25);
    M->u_med  = bench_quantile(U, m, 0.50); M->u_q3  = bench_quantile(U, m, 0.75);
    M->u_p99  = bench_quantile(U, m, 0.99); M->u_p999 = bench_quantile(U, m, 0.999);
    M->u_max  = U[m - 1];                  M->u_mean = sum / (double)m;
    M->u_iqr  = M->u_q3 - M->u_q1;
    M->u_fence_hi = M->u_q3 + 1.5 * M->u_iqr;
    for (size_t i = 0; i < m; i++) if (U[i] > M->u_fence_hi) M->n_beyond_fence++;

    if (mn) { qsort(UN, mn, sizeof *UN, bench_cmp_double);
              M->u_med_nontrivial = bench_quantile(UN, mn, 0.50); }

    free(U); free(UN);
}

/* The accuracy detail sidecar: one CSV per process (path BENCH_ACCURACY_DETAIL), read by
 * benchmark/plots/runs.py (DETAIL_HEADER: change both).  Rows are keyed like their
 * accuracy row and are of three kinds:
 *   hist     key = bin, sign (+/-), [lo_log2, hi_log2), count; bin 0 = exactly zero,
 *            bin 1 also holds everything below 2^-10, bin 49 = [2^10, inf); every
 *            bin is written, empty or not
 *   worst    key = rank (0 = the u_max point), ulp (signed), nu, arg, aux (s on
 *            h_nu_s, empty on z_nu), value, ref, bits, index
 *   excluded key = index, nu, arg, aux, reason (nan | +inf | -inf); value empty
 * `index` is into the scored arrays (with a --max-points stride, file row = index *
 * stride). */
#include <stdio.h>
static FILE *bench_detail_fp = NULL;

static inline void bench_detail_open(const char *path)
{
    if (!path || !*path) return;
    bench_detail_fp = fopen(path, "w");
    if (bench_detail_fp)
        fprintf(bench_detail_fp, "device,supplier,family,law,backend,kind,key,sign,"
                "lo_log2,hi_log2,count,ulp,nu,arg,aux,value,ref,bits,index,reason\n");
}

static inline void bench_detail_close(void)
{ if (bench_detail_fp) { fclose(bench_detail_fp); bench_detail_fp = NULL; } }

static inline void bench_detail_emit(const char *device, const char *supplier,
                                  const char *fam, const char *law,
                                  const char *backend, const bench_metrics *M,
                                  const double *value, const double *nu,
                                  const double *arg, const double *aux, size_t n)
{
    FILE *f = bench_detail_fp;
    if (!f) return;
    const double step = (BENCH_HIST_HI_LOG - BENCH_HIST_LO_LOG) / BENCH_HIST_BINS;
    for (int sgn = 0; sgn < 2; sgn++) {
        const long *H = sgn ? M->hist_neg : M->hist_pos;
        for (int b = 0; b < BENCH_HIST_BINS + 2; b++) {
            double lo_l, hi_l;
            if (b == 0)                 { lo_l = -INFINITY; hi_l = -INFINITY; }
            else if (b == 1)            { lo_l = -INFINITY; hi_l = BENCH_HIST_LO_LOG + step; }
            else if (b == BENCH_HIST_BINS + 1) { lo_l = BENCH_HIST_HI_LOG; hi_l = INFINITY; }
            else { lo_l = BENCH_HIST_LO_LOG + (b - 1) * step; hi_l = lo_l + step; }
            fprintf(f, "%s,%s,%s,%s,%s,hist,%d,%s,%.17g,%.17g,%ld,,,,,,,,,\n",
                    device, supplier, fam, law, backend, b, sgn ? "-" : "+",
                    lo_l, hi_l, H[b]);
        }
    }
    for (size_t k = 0; k < M->n_worst; k++) {
        const bench_worst *w = &M->worst[k];
        if (aux)
            fprintf(f, "%s,%s,%s,%s,%s,worst,%zu,,,,,%.17g,%.17g,%.17g,%.17g,%.17g,"
                    "%.17g,%d,%zu,\n", device, supplier, fam, law, backend, k,
                    w->ulp, w->nu, w->arg, w->aux, w->value, w->ref, w->bits, w->index);
        else
            fprintf(f, "%s,%s,%s,%s,%s,worst,%zu,,,,,%.17g,%.17g,%.17g,,%.17g,"
                    "%.17g,%d,%zu,\n", device, supplier, fam, law, backend, k,
                    w->ulp, w->nu, w->arg, w->value, w->ref, w->bits, w->index);
    }
    for (size_t i = 0; i < n; i++) {
        if (isfinite(value[i])) continue;
        const char *why = isnan(value[i]) ? "nan" : (value[i] > 0 ? "+inf" : "-inf");
        if (aux)
            fprintf(f, "%s,%s,%s,%s,%s,excluded,%zu,,,,,,%.17g,%.17g,%.17g,,,,,%s\n",
                    device, supplier, fam, law, backend, i, nu[i], arg[i], aux[i],
                    why);
        else
            fprintf(f, "%s,%s,%s,%s,%s,excluded,%zu,,,,,,%.17g,%.17g,,,,,,%s\n",
                    device, supplier, fam, law, backend, i, nu[i], arg[i], why);
    }
    fflush(f);
}

#endif /* BENCH_METRICS_H */
