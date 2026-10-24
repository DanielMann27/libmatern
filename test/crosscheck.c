/* crosscheck.c -- the scalar backend (src/backends/scalar/, MATERN_BACKEND=scalar or
 * scalar_portable) against the vector kernels (src/bessel/, the matched backend called
 * directly), which share no code; the gap is reported, a one-sided zero or non-finite fails. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>
#include "matern.h"

/* Library-internal entries (src/backends/internal.h), local in build/libmatern.a, so
 * `make crosscheck` links build/libmatern_internal.a. */
int matern_evaluate_matched(double*, const double*, const double*, const double*,
                            size_t, matern_workspace*);
const char *matern_scalar_isa(void);

/* Called directly, the entries return a status and write nothing on an invalid argument,
 * so return codes are checked. */

/* splitmix64: deterministic inputs without a data file. */
static uint64_t S = 0x9E3779B97F4A7C15ULL;
static double u01(void){
    uint64_t z = (S += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
}
static double loguniform(double a, double b){ return a * pow(b/a, u01()); }

#define N 200000

int main(void)
{
    /* The dispatcher reads MATERN_BACKEND here, on the first call. */
    const char *backend = matern_backend_name();
    if (strcmp(backend, "scalar") != 0 && strcmp(backend, "scalar_portable") != 0) {
        printf("crosscheck: the backend in use is %s; run the test with "
               "MATERN_BACKEND=scalar or\n  MATERN_BACKEND=scalar_portable "
               "(make crosscheck runs both)\n", backend);
        return 1;
    }

    double *h  = malloc(N*sizeof *h),  *nu = malloc(N*sizeof *nu);
    double *r  = malloc(N*sizeof *r);
    double *a  = malloc(N*sizeof *a),  *b  = malloc(N*sizeof *b);
    if (!h || !nu || !r || !a || !b) { puts("out of memory"); return 1; }

    /* The nu range of the contract domain (matern.h, DOMAIN AND VALIDITY), with z
     * capped at 760: M ~ e^{-z} underflows to zero past z ~ 745, and 760 still
     * exercises the extended range past the exponential limit.  The skip count
     * is printed. */
    for (size_t i = 0; i < N; i++) {
        nu[i] = loguniform(1e-3, 1e3);
        r [i] = loguniform(0.1, 10.0);
        h [i] = loguniform(1e-3, 760.0) * r[i] / sqrt(2.0*nu[i]);
    }

    matern_workspace *ws = NULL;
    if (matern_workspace_create(&ws, 0) != 0) { puts("workspace_create failed"); return 1; }
    const int rc_s = matern_covariance      (a, h, nu, r, N, ws);
    const int rc_m = matern_evaluate_matched(b, h, nu, r, N, ws);
    if (rc_s != 0 || rc_m != 0) {
        printf("evaluate returned scalar=%d matched=%d -- not 0, so the outputs are "
               "not results\n", rc_s, rc_m);
        return 1;
    }

    double worst = 0.0; size_t wi = 0; size_t scored = 0, skipped = 0, nonfinite = 0, onesided = 0;
    for (size_t i = 0; i < N; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            /* both non-finite for the same reason is agreement, not a difference */
            if (isnan(a[i]) && isnan(b[i])) { skipped++; continue; }
            nonfinite++; continue;
        }
        /* Below DBL_MIN a relative comparison is meaningless, so points where
         * both results are below DBL_MIN are skipped and counted.  A point where
         * only one side is below DBL_MIN (one path delivers a value, the other
         * truncates) is counted as a one-sided zero and fails. */
        if (fabs(a[i]) < DBL_MIN && fabs(b[i]) < DBL_MIN) { skipped++; continue; }
        if (fabs(a[i]) < DBL_MIN || fabs(b[i]) < DBL_MIN) { onesided++; continue; }
        const double d = fabs(a[i] - b[i]) / fabs(b[i]);
        scored++;
        if (d > worst) { worst = d; wi = i; }
    }

    printf("crosscheck: %s (the %s build) vs matched, %d points over "
           "z in [1e-3,760], nu in [1e-3,1e3]\n\n", backend, matern_scalar_isa(), N);
    printf("  scored            %zu\n", scored);
    printf("  skipped           %zu  (M < DBL_MIN, or both NaN)\n", skipped);
    printf("  one-sided non-finite %zu  (must be 0)\n", nonfinite);
    printf("  one-sided zero    %zu  (must be 0)\n", onesided);
    printf("  worst gap         %.3e   at z = %.6g, nu = %.6g\n",
           worst, sqrt(2.0*nu[wi])*h[wi]/r[wi], nu[wi]);
    printf("  scalar %.17g\n  matched %.17g\n\n", a[wi], b[wi]);
    printf("  `scalar` evaluates the approximations in plain doubles; `matched` carries\n"
           "  the double-double argument and exponent.  The gap is expected to be of\n"
           "  the order of max(z,nu)*eps (z*eps from the argument, nu*eps from the\n"
           "  Debye exponent); this is a scale, not a limit.\n");
    /* An order of magnitude, not a bound: the gap at a point can exceed it, and
     * it also depends on the exact (h, s) that derive a given z. */
    printf("  max(z,nu)*eps     %.3e   at that point\n\n",
           (sqrt(2.0*nu[wi])*h[wi]/r[wi] > nu[wi] ? sqrt(2.0*nu[wi])*h[wi]/r[wi] : nu[wi])
           * 2.220446049250313e-16);
    printf("  The gap is reported, not asserted: the test fails only when one\n"
           "  implementation returns no value (zero or non-finite) where the other\n"
           "  does, or when no point is scored.\n\n");

    int bad = (nonfinite != 0) || (onesided != 0) || (scored == 0);
    printf("  %s\n", scored == 0 ? "FAILED: no point was scored"
                 : bad ? "FAILED: one implementation returned no value where the other did"
                       : "both implementations return a value at every scored point");
    matern_workspace_destroy(ws);
    free(h); free(nu); free(r); free(a); free(b);
    return bad ? 1 : 0;
}
