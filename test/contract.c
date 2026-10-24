/* contract.c -- properties promised by matern.h: 0 < M <= 1 where M is representable,
 * monotone in h, NaN for invalid input (including h < 0 with s < 0 and infinite nu or s),
 * 0 at h = +infinity, a positive value past the exponential limit, the library version. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matern.h"

static int failures = 0;

static void report(const char *name, int bad, const char *unit)
{
    printf("  %-58s %s\n", name, bad ? "FAIL" : "ok");
    if (bad) { printf("      %d %s\n", bad, unit); failures += bad; }
}

/* 0. the library linked is the release of this header */
static void check_version(void)
{
    report("matern_version() is MATERN_VERSION_STRING",
           strcmp(matern_version(), MATERN_VERSION_STRING) != 0,
           "release differs between the library and the header");
}

/* 1. 0 < M <= 1, and finite, on a sample where the true M is positive.
 * At z <= 700 the true M is a positive double at every point (subnormal at
 * z = 700, nu = 1e-3), so a 0 is a defect; past that, M can be below the double
 * range and the computed value 0 (matern.h, ACCURACY). */
static void check_range(void)
{
    const size_t n = 200000;
    double *h = malloc(n*8), *nu = malloc(n*8), *scale = malloc(n*8), *out = malloc(n*8);
    unsigned long long s = 0x243F6A8885A308D3ull;
    for (size_t i = 0; i < n; i++) {
        unsigned long long x = (s += 0x9E3779B97F4A7C15ull);
        x = (x^(x>>30))*0xBF58476D1CE4E5B9ull; x = (x^(x>>27))*0x94D049BB133111EBull; x ^= x>>31;
        const double u = (double)(x>>11)*0x1.0p-53;
        unsigned long long y = (s += 0x9E3779B97F4A7C15ull);
        y = (y^(y>>30))*0xBF58476D1CE4E5B9ull; y = (y^(y>>27))*0x94D049BB133111EBull; y ^= y>>31;
        const double v = (double)(y>>11)*0x1.0p-53;
        nu[i]    = 1e-3 * pow(1e3/1e-3, u);
        scale[i] = 1.0;
        h[i]     = (1e-3 * pow(700.0/1e-3, v)) / sqrt(2.0*nu[i]);
    }
    matern_covariance(out, h, nu, scale, n, NULL);
    int bad = 0;
    for (size_t i = 0; i < n; i++)
        if (!(out[i] > 0.0) || !(out[i] <= 1.0) || !isfinite(out[i])) bad++;
    report("0 < M <= 1 and finite, 200000 points, z <= 700", bad, "points outside the range");
    free(h); free(nu); free(scale); free(out);
}

/* 2. non-increasing in h on a log grid */
static void check_monotone(void)
{
    const int steps = 4000;
    int bad = 0;
    const double orders[] = {0.5, 1.5, 2.5, 7.0, 24.7, 35.0, 100.0};
    for (size_t k = 0; k < sizeof orders/sizeof orders[0]; k++) {
        double *h = malloc(steps*8), *nu = malloc(steps*8), *scale = malloc(steps*8),
               *out = malloc(steps*8);
        for (int i = 0; i < steps; i++) {
            h[i] = 1e-4 * pow(1e4/1e-4, (double)i/(double)(steps-1));
            nu[i] = orders[k]; scale[i] = 1.0;
        }
        matern_covariance(out, h, nu, scale, (size_t)steps, NULL);
        /* Where M is representable the true M falls by far more per step than its
         * rounding, so a rise is a defect.  Non-strict, since M can round to 0 at
         * both ends of a step.  Between arguments a few ulp apart the computed M
         * can rise in its last bits, which this spacing does not sample. */
        for (int i = 1; i < steps; i++)
            if (out[i] > out[i-1]) bad++;
        free(h); free(nu); free(scale); free(out);
    }
    report("M non-increasing in h at 0.46 % steps, 7 orders", bad, "increases");
}

/* 3. invalid input returns NaN, including combinations */
static void check_validity(void)
{
    struct { double h, nu, scale; const char *what; } c[] = {
        { -1.0,  2.0,  1.0, "h < 0" },
        {  1.0, -2.0,  1.0, "nu < 0" },
        {  1.0,  2.0, -1.0, "scale < 0" },
        {  1.0,  0.0,  1.0, "nu = 0" },
        {  1.0,  2.0,  0.0, "scale = 0" },
        { -1.0,  2.0, -1.0, "h < 0 AND scale < 0 -- the signs CANCEL in z" },
        { -3.0,  2.0, -7.0, "h < 0 AND scale < 0, different magnitudes" },
        { -1.0, -2.0, -1.0, "all three negative" },
        {  NAN,  2.0,  1.0, "h = NaN" },
        {  1.0,  NAN,  1.0, "nu = NaN" },
        {  1.0,  2.0,  NAN, "scale = NaN" },
        { -1.0,  2.0,  0.0, "h < 0 AND scale = 0" },
    };
    const int n = (int)(sizeof c / sizeof c[0]);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double got = 0.0;
        matern_covariance(&got, &c[i].h, &c[i].nu, &c[i].scale, 1, NULL);
        if (!isnan(got)) {
            printf("      %-46s -> %.17g, want NaN\n", c[i].what, got);
            bad++;
        }
    }
    report("invalid input returns NaN, 12 cases", bad, "cases returned a number");
}

/* 3b. +infinity in nu or s returns NaN in a single call, as a lane of the vector body and
 * of the tile remainder of a valid batch, and in a constant half-integer run; the rest of
 * each batch stays in (0, 1]. */
static void check_infinite_parameters(void)
{
    const double inf = (double)INFINITY;
    static const struct { double h, nu, scale; const char *what; } c[] = {
        {  1.0,  inf,  1.0, "nu = +inf" },
        {  0.0,  inf,  1.0, "nu = +inf AND h = 0" },
        {  inf,  inf,  1.0, "nu = +inf AND h = +inf" },
        {  1.0,  2.0,  inf, "scale = +inf" },
        {  0.0,  2.0,  inf, "scale = +inf AND h = 0" },
        {  inf,  2.0,  inf, "scale = +inf AND h = +inf" },
        {  1.0,  1.5,  inf, "scale = +inf at a half-integer order" },
        {  1.0,  inf,  inf, "nu = +inf AND scale = +inf" },
    };
    const int nc = (int)(sizeof c / sizeof c[0]);
    enum { NB = 67, BODY = 10, TAIL = 65, NR = 512, LANE = 300 };   /* 67 = 64 + 3 */
    double h[NR], nu[NR], scale[NR], out[NR];
    int bad = 0, neighbours = 0;
    for (int i = 0; i < nc; i++) {
        double got = 0.0;
        matern_covariance(&got, &c[i].h, &c[i].nu, &c[i].scale, 1, NULL);
        if (!isnan(got)) { printf("      single %-38s -> %.17g, want NaN\n", c[i].what, got); bad++; }
        const int at[2] = { BODY, TAIL };
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < NB; j++) { h[j] = 0.05 + 0.01*j; nu[j] = 2.3 + 0.0123*j; scale[j] = 1.0; }
            h[at[k]] = c[i].h; nu[at[k]] = c[i].nu; scale[at[k]] = c[i].scale;
            matern_covariance(out, h, nu, scale, NB, NULL);
            if (!isnan(out[at[k]])) {
                printf("      batch %s %-34s -> %.17g, want NaN\n", k ? "tail" : "body", c[i].what, out[at[k]]);
                bad++;
            }
            for (int j = 0; j < NB; j++)
                if (j != at[k] && !(out[j] > 0.0 && out[j] <= 1.0)) neighbours++;
        }
    }
    for (int j = 0; j < NR; j++) { h[j] = 0.001 + 0.01*j; nu[j] = 1.5; scale[j] = 1.0; }
    scale[LANE] = inf;
    matern_covariance(out, h, nu, scale, NR, NULL);
    if (!isnan(out[LANE])) { printf("      half-integer run, scale = +inf -> %.17g, want NaN\n", out[LANE]); bad++; }
    for (int j = 0; j < NR; j++) if (j != LANE && !(out[j] > 0.0 && out[j] <= 1.0)) neighbours++;
    report("infinite nu or scale returns NaN, 8 cases x 3 layouts + a run", bad,
           "cases returned a number");
    report("the rest of those batches stays in (0, 1]", neighbours, "neighbouring points");
}

/* 3c. h = +infinity returns 0, in the layouts of 3b. */
static void check_infinite_separation(void)
{
    const double inf = (double)INFINITY, orders[] = {0.5, 1.5, 2.3, 35.0};
    const int no = (int)(sizeof orders / sizeof orders[0]);
    enum { NB = 67, BODY = 10, TAIL = 65, NR = 512, LANE = 300 };
    double h[NR], nu[NR], scale[NR], out[NR];
    int bad = 0;
    for (int i = 0; i < no; i++) {
        double got = -1.0, s = 0.3;
        matern_covariance(&got, &inf, &orders[i], &s, 1, NULL);
        if (got != 0.0) { printf("      single nu = %-8g -> %.17g, want 0\n", orders[i], got); bad++; }
        const int at[2] = { BODY, TAIL };
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < NB; j++) { h[j] = 0.05 + 0.01*j; nu[j] = 2.3 + 0.0123*j; scale[j] = 0.3; }
            h[at[k]] = inf; nu[at[k]] = orders[i];
            matern_covariance(out, h, nu, scale, NB, NULL);
            if (out[at[k]] != 0.0) {
                printf("      batch %s nu = %-8g -> %.17g, want 0\n", k ? "tail" : "body", orders[i], out[at[k]]);
                bad++;
            }
        }
    }
    for (int j = 0; j < NR; j++) { h[j] = 0.001 + 0.01*j; nu[j] = 1.5; scale[j] = 0.3; }
    h[LANE] = inf;
    matern_covariance(out, h, nu, scale, NR, NULL);
    if (out[LANE] != 0.0) { printf("      half-integer run, h = +inf -> %.17g, want 0\n", out[LANE]); bad++; }
    report("h = +infinity returns 0, 4 orders x 3 layouts + a run", bad, "cases returned a nonzero value");
}

/* 4. the extended range delivers a value, not zero */
static void check_extended_range(void)
{
    /* Past z = 708.396 the ordinary assembly underflows; the header promises a
     * value rather than a truncation, positive until M leaves the double range. */
    int bad = 0;
    const double orders[] = {0.5, 2.5, 20.0, 40.0};
    for (size_t k = 0; k < sizeof orders/sizeof orders[0]; k++)
        for (double z = 709.0; z <= 740.0; z += 1.0) {
            const double nu = orders[k], scale = 1.0, h = z / sqrt(2.0*nu);
            double got = 0.0;
            matern_covariance(&got, &h, &nu, &scale, 1, NULL);
            if (!(got > 0.0) || !isfinite(got)) bad++;
        }
    report("M > 0 past the exponential limit, z = 709..740, 4 orders", bad,
           "points returned 0 or non-finite");
}

/* 4b. the extended range at orders nu >= 30 (uniform-order expansion, one folded
 * exponent).  Every point has a true M >= DBL_MIN, so a 0 is a defect; the scalar
 * backend maps a NaN to 0 (`matern_scalar_clamp_to_unit`). */
static void check_extended_range_deep(void)
{
    static const struct { double nu, z; const char *why; } p[] = {
        { 30.0,   710.0, "arb: representable" }, { 30.0,   750.0, "arb: representable" },
        { 30.0,   795.2, "arb: representable" },
        { 32.5,   710.0, "arb: representable" }, { 32.5,   795.2, "arb: representable" },
        { 40.0,   710.0, "arb: representable" }, { 40.0,   750.0, "arb: representable" },
        { 40.0,   795.2, "arb: representable" },
        { 96.64,  710.0, "arb: representable" }, { 96.64,  750.0, "arb: representable" },
        { 96.64,  795.2, "arb: representable" }, { 96.64,  850.0, "arb: representable" },
        { 200.0,  710.0, "arb: representable" }, { 200.0,  795.2, "arb: representable" },
        { 200.0,  850.0, "arb: representable" },
        { 400.0,  710.0, "arb: representable" }, { 400.0,  795.2, "arb: representable" },
        { 400.0,  850.0, "arb: representable" },
    };
    const int n = (int)(sizeof p / sizeof p[0]);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        const double scale = 1.0/3.0;             /* not 1, so the derived z is inexact */
        const double h     = p[i].z * scale / sqrt(2.0 * p[i].nu);
        double got = 0.0;
        matern_covariance(&got, &h, &p[i].nu, &scale, 1, NULL);
        if (!(got > 0.0) || !isfinite(got)) {
            printf("      z = %-8g nu = %-8g -> %.17g, %s\n",
                   p[i].z, p[i].nu, got, p[i].why);
            bad++;
        }
    }
    report("M > 0 past the exponential limit, 18 points with nu >= 30", bad,
           "points returned 0 or non-finite where the true M >= DBL_MIN");
}

/* The dispatcher resolves the backend once per process, so `make check` runs this
 * file once per backend (MATERN_BACKEND); agreement between backends is not checked. */

int main(void)
{
    printf("contract: backend %s\n\n", matern_backend_name());
    check_version();
    check_range();
    check_monotone();
    check_validity();
    check_infinite_parameters();
    check_infinite_separation();
    check_extended_range();
    check_extended_range_deep();
    printf("\n%s\n", failures ? "CONTRACT FAILED" : "contract holds");
    return failures != 0;
}
