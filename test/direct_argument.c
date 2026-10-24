/* direct_argument.c -- the (z, nu) entry matern_covariance_z(): z = 0 returns exactly 1.0,
 * z = +inf returns 0, invalid input returns NaN; agreement with matern_covariance() is
 * reported, not asserted, since that entry derives z again from the rounded h = z/sqrt(2 nu). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matern.h"

static int failures = 0;

static void fail(const char *what, double got, double want)
{
    printf("    FAIL %-46s got %.17g, want %.17g\n", what, got, want);
    failures++;
}

int main(void)
{
    const char *backend = matern_backend_name();
    printf("direct argument: backend %s\n", backend);

    /* z = 0 is exactly 1, at every order.
     * Exact equality with 1.0, as on the (h, nu, s) entry. */
    {
        double z[19], nu[19], out[19];
        for (int i = 0; i < 19; i++) { z[i] = 0.0; nu[i] = 0.25 * (double)(i + 1); }
        if (matern_covariance_z(out, z, nu, 19, NULL) != MATERN_OK)
            fail("z = 0 sweep returned non-OK", 0, 0);
        for (int i = 0; i < 19; i++)
            if (out[i] != 1.0) fail("M_nu(0) is exactly 1", out[i], 1.0);
    }

    /* invalid input returns NaN, including in combination */
    {
        const double bad_nu[] = { 0.0, -1.0, (double)NAN, (double)INFINITY };
        const double bad_z [] = { -1.0, -0x1p-1074, -(double)INFINITY, (double)NAN };
        for (size_t a = 0; a < sizeof bad_nu / sizeof *bad_nu; a++) {
            double z = 2.0, nu = bad_nu[a], out = 0.0;
            matern_covariance_z(&out, &z, &nu, 1, NULL);
            if (!isnan(out)) fail("invalid nu returns NaN", out, (double)NAN);
        }
        for (size_t b = 0; b < sizeof bad_z / sizeof *bad_z; b++) {
            double z = bad_z[b], nu = 1.5, out = 0.0;
            matern_covariance_z(&out, &z, &nu, 1, NULL);
            if (!isnan(out)) fail("invalid z returns NaN", out, (double)NAN);
        }
        /* Two invalid arguments together: (nu < 0, z < 0), the analogue of
         * (h < 0, s < 0) on the (h, nu, s) entry; z is an input here, so the
         * signs cannot cancel, and the pair is checked nonetheless. */
        {
            double z = -2.0, nu = -1.5, out = 0.0;
            matern_covariance_z(&out, &z, &nu, 1, NULL);
            if (!isnan(out)) fail("nu < 0 AND z < 0 returns NaN", out, (double)NAN);
        }
    }

    /* nu = +inf is invalid also as a lane of the classifier's vector body and tile
     * remainder (67 = 64 + 3), and at z = 0 */
    {
        enum { NB = 67 };
        const int at[2] = { 10, 65 };
        const double zs[2] = { 2.0, 0.0 };
        for (int k = 0; k < 2; k++) for (int q = 0; q < 2; q++) {
            double z[NB], nu[NB], out[NB];
            for (int j = 0; j < NB; j++) { z[j] = 0.1 + 0.3*j; nu[j] = 2.3 + 0.0123*j; }
            z[at[k]] = zs[q]; nu[at[k]] = (double)INFINITY;
            matern_covariance_z(out, z, nu, NB, NULL);
            if (!isnan(out[at[k]])) fail("nu = +inf in a batch returns NaN", out[at[k]], (double)NAN);
            for (int j = 0; j < NB; j++)
                if (j != at[k] && !(out[j] > 0.0 && out[j] <= 1.0))
                    fail("a neighbour of nu = +inf stays in (0, 1]", out[j], 0.5);
        }
    }

    /* z = +inf returns 0, alone, in the same two lanes and in a constant half-integer run */
    {
        enum { NB = 67, NR = 512, LANE = 300 };
        const double inf = (double)INFINITY, orders[] = { 0.5, 1.5, 2.3, 35.0 };
        const int at[2] = { 10, 65 };
        double z[NR], nu[NR], out[NR];
        for (size_t k = 0; k < sizeof orders / sizeof *orders; k++) {
            double got = -1.0;
            matern_covariance_z(&got, &inf, &orders[k], 1, NULL);
            if (got != 0.0) fail("z = +inf returns 0", got, 0.0);
            for (int q = 0; q < 2; q++) {
                for (int j = 0; j < NB; j++) { z[j] = 0.1 + 0.3*j; nu[j] = 2.3 + 0.0123*j; }
                z[at[q]] = inf; nu[at[q]] = orders[k];
                matern_covariance_z(out, z, nu, NB, NULL);
                if (out[at[q]] != 0.0) fail("z = +inf in a batch returns 0", out[at[q]], 0.0);
            }
        }
        for (int j = 0; j < NR; j++) { z[j] = 0.001 + 0.01*j; nu[j] = 1.5; }
        z[LANE] = inf;
        matern_covariance_z(out, z, nu, NR, NULL);
        if (out[LANE] != 0.0) fail("z = +inf in a half-integer run returns 0", out[LANE], 0.0);
    }

    /* the two entries agree to the derivation's own error */
    {
        enum { N = 4096 };
        double *z     = malloc(N * sizeof *z);
        double *nu    = malloc(N * sizeof *nu);
        double *h     = malloc(N * sizeof *h);
        double *scale = malloc(N * sizeof *scale);
        double *a     = malloc(N * sizeof *a);
        double *b     = malloc(N * sizeof *b);
        if (!z || !nu || !h || !scale || !a || !b) return 2;

        /* log-uniform over the validated box, deterministic */
        unsigned long long s = 0x243F6A8885A308D3ull;
        for (int i = 0; i < N; i++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const double u1 = (double)((s >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const double u2 = (double)((s >> 11) & ((1ull << 53) - 1)) * 0x1.0p-53;
            nu[i]    = exp(log(1e-2) + (log(60.0)  - log(1e-2)) * u1);
            z[i]     = exp(log(1e-2) + (log(700.0) - log(1e-2)) * u2);
            scale[i] = 1.0;
            h[i]     = z[i] / sqrt(2.0 * nu[i]);   /* the only h that means this z */
        }

        if (matern_covariance_z(a, z, nu, N, NULL) != MATERN_OK)
            fail("direct entry returned non-OK", 0, 0);
        if (matern_covariance(b, h, nu, scale, N, NULL) != MATERN_OK)
            fail("derived entry returned non-OK", 0, 0);

        /* The difference is a few ulp of displacement in z; M behaves like
         * e^-z, so that is a relative difference of a few z*eps. */
        size_t compared = 0;
        double worst = 0.0, worst_z = 0.0;
        for (int i = 0; i < N; i++) {
            if (!isfinite(a[i]) || !isfinite(b[i]) || a[i] == 0.0) continue;
            const double rel = fabs(a[i] - b[i]) / fabs(a[i]);
            compared++;
            if (rel > worst) { worst = rel; worst_z = z[i]; }
        }
        printf("    %zu points compared, largest relative difference %.3e at z = %.4g"
               " (z*eps there: %.3e)\n",
               compared, worst, worst_z, worst_z * 2.220446049250313e-16);
        if (!compared) fail("no comparable points -- the test proved nothing", 0, 1);

        free(z); free(nu); free(h); free(scale); free(a); free(b);
    }

    if (failures) {
        printf("direct argument: %d FAILURE%s on backend %s\n",
               failures, failures == 1 ? "" : "S", backend);
        return 1;
    }
    printf("direct argument: all checks pass on backend %s\n", backend);
    return 0;
}
