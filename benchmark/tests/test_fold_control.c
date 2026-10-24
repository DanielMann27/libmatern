/* test_fold_control.c -- per scalar instruction set: the direct twin equals the backend on
 * the unfolded large-argument region and differs where the backend folds; the control
 * equals the backend everywhere.  Run by benchmark/tests/run_tests.sh. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "matern.h"

typedef int (*entry)(double *, const double *, const double *, const double *,
                     size_t, matern_workspace *);
#define DECL(name) int name(double *, const double *, const double *, const double *, \
                            size_t, matern_workspace *);
DECL(matern_evaluate_scalar_z)
DECL(matern_evaluate_scalar_avx512_z)
DECL(matern_evaluate_scalar_avx2_z)
DECL(matern_evaluate_scalar_fma_z)
#define DECL_ISA(v) DECL(matern_evaluate_scalar_##v##_avx512_z) DECL(matern_evaluate_scalar_##v##_avx2_z) \
                    DECL(matern_evaluate_scalar_##v##_fma_z) DECL(matern_evaluate_scalar_##v##_portable_z)
DECL_ISA(control)
DECL_ISA(unfolded_direct)

static int always(void) { return 1; }
static int has_avx512(void)
{ return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("fma"); }
static int has_avx2(void) { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
static int has_fma(void)  { return __builtin_cpu_supports("fma") && __builtin_cpu_supports("avx"); }
static const struct { const char *isa; entry folded, control, direct; int (*runs)(void); } BUILDS[] = {
    { "portable", matern_evaluate_scalar_z,        matern_evaluate_scalar_control_portable_z,
      matern_evaluate_scalar_unfolded_direct_portable_z, always },
    { "fma",      matern_evaluate_scalar_fma_z,    matern_evaluate_scalar_control_fma_z,
      matern_evaluate_scalar_unfolded_direct_fma_z,      has_fma },
    { "avx2",     matern_evaluate_scalar_avx2_z,   matern_evaluate_scalar_control_avx2_z,
      matern_evaluate_scalar_unfolded_direct_avx2_z,     has_avx2 },
    { "avx512",   matern_evaluate_scalar_avx512_z, matern_evaluate_scalar_control_avx512_z,
      matern_evaluate_scalar_unfolded_direct_avx512_z,   has_avx512 },
};

#define N 4096

/* Fills (z, nu) with n points: nu on a non-half-integer ladder in [lo_nu, hi_nu],
 * z log-spaced in [lo_z, hi_z]. */
static void fill(double *z, double *nu, int n, double lo_nu, double hi_nu,
                 double lo_z, double hi_z)
{
    for (int i = 0; i < n; i++) {
        const double u = (double)(i % 64) / 63.0, v = (double)(i / 64) / (double)(n / 64 - 1);
        nu[i] = lo_nu + u * (hi_nu - lo_nu) + 0.0123;     /* off the half-integers */
        z[i]  = exp(log(lo_z) + v * (log(hi_z) - log(lo_z)));
    }
}

static int count_differing(const double *a, const double *b, int n)
{
    int d = 0;
    for (int i = 0; i < n; i++) d += memcmp(&a[i], &b[i], sizeof a[i]) != 0;
    return d;
}

int main(void)
{
    __builtin_cpu_init();
    static double z[N], nu[N], f[N], c[N], d[N];
    int bad = 0;
    for (size_t b = 0; b < sizeof BUILDS / sizeof *BUILDS; b++) {
        if (!BUILDS[b].runs()) { printf("%-8s not run on this CPU\n", BUILDS[b].isa); continue; }
        /* large argument: nu <= 30, z above the flattened border 22, below 708.396 */
        fill(z, nu, N, 0.02, 29.9, 22.5, 708.0);
        BUILDS[b].folded (f, z, nu, nu, N, NULL);
        BUILDS[b].control(c, z, nu, nu, N, NULL);
        BUILDS[b].direct (d, z, nu, nu, N, NULL);
        const int dl = count_differing(f, d, N), cl = count_differing(f, c, N);
        /* Debye: nu >= 30; continued fraction: 1 <= z <= 18, nu < 30 */
        fill(z, nu, N, 30.5, 200.0, 0.5, 400.0);
        BUILDS[b].folded(f, z, nu, nu, N, NULL);
        BUILDS[b].control(c, z, nu, nu, N, NULL);
        BUILDS[b].direct(d, z, nu, nu, N, NULL);
        const int dd = count_differing(f, d, N), cd = count_differing(f, c, N);
        fill(z, nu, N, 0.6, 29.0, 1.0, 18.0);
        BUILDS[b].folded(f, z, nu, nu, N, NULL);
        BUILDS[b].control(c, z, nu, nu, N, NULL);
        BUILDS[b].direct(d, z, nu, nu, N, NULL);
        const int dc = count_differing(f, d, N), cc = count_differing(f, c, N);
        printf("%-8s large argument: direct differs at %d of %d; Debye: %d; continued fraction: %d;"
               " control differs at %d\n", BUILDS[b].isa, dl, N, dd, dc, cl + cd + cc);
        if (dl != 0)            { printf("  FAIL: the direct twin must equal the backend on the large-argument region\n"); bad = 1; }
        if (dd == 0 || dc == 0) { printf("  FAIL: the direct twin equals the backend where it folds: not built as a twin\n"); bad = 1; }
        if (cl + cd + cc != 0)  { printf("  FAIL: the control differs from the backend\n"); bad = 1; }
    }
    if (!bad) printf("fold controls hold\n");
    return bad;
}
