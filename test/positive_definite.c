/* positive_definite.c -- the largest leading block of 200 x 200 library covariance matrices
 * (one-dimensional grid, s = 1) that admits a Cholesky factorisation, the unit diagonal, and
 * exact symmetry and batch invariance on every backend: a point gets the same bits alone, in
 * one call, and in calls of other lengths. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matern.h"

static int failures = 0;

static void report(const char *what, int bad, const char *note)
{
    printf("  %-58s %s\n", what, bad ? "FAILED" : "ok");
    if (bad) { printf("      %d %s\n", bad, note); failures++; }
}

/* Cholesky of the leading m x m block of an n x n matrix W into A.  Returns the
 * smallest pivot on success, or -1.0 if a pivot is not positive. */
static double cholesky_leading(const double *W, int n, int m, double *A)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) A[(size_t)i*m + j] = W[(size_t)i*n + j];

    double smallest = 1e300;
    for (int j = 0; j < m; j++) {
        double d = A[(size_t)j*m + j];
        for (int k = 0; k < j; k++) d -= A[(size_t)j*m + k] * A[(size_t)j*m + k];
        if (!(d > 0.0)) return -1.0;
        d = sqrt(d);
        A[(size_t)j*m + j] = d;
        if (d < smallest) smallest = d;
        for (int i = j+1; i < m; i++) {
            double t = A[(size_t)i*m + j];
            for (int k = 0; k < j; k++) t -= A[(size_t)i*m + k] * A[(size_t)j*m + k];
            A[(size_t)i*m + j] = t/d;
        }
    }
    return smallest;
}

/* The largest leading block that factors.  If a leading block is positive
 * definite, so is every smaller one, so bisection over m is sound. */
static int largest_factorable(const double *W, int n, double *A, double *pivot)
{
    double p = cholesky_leading(W, n, n, A);
    if (p > 0.0) { *pivot = p; return n; }
    int lo = 1, hi = n;
    while (hi - lo > 1) {
        const int mid = (lo + hi)/2;
        if (cholesky_leading(W, n, mid, A) > 0.0) lo = mid; else hi = mid;
    }
    *pivot = cholesky_leading(W, n, lo, A);
    return lo;
}

int main(void)
{
    printf("positive definiteness: backend %s\n\n", matern_backend_name());

    const int n = 200;
    const size_t m2 = (size_t)n * (size_t)n;
    double *W = malloc(m2 * sizeof *W), *A = malloc(m2 * sizeof *A);
    double *h = malloc(m2 * sizeof *h), *order = malloc(m2 * sizeof *order);
    double *scale = malloc(m2 * sizeof *scale), *x = malloc((size_t)n * sizeof *x);
    if (!W || !A || !h || !order || !scale || !x) { puts("out of memory"); return 1; }

    matern_workspace *work = NULL;
    if (matern_workspace_create(&work, 0) != 0) { puts("workspace failed"); return 1; }

    static const double ORDERS[] = { 0.5, 1.5, 2.5, 5.0, 20.0, 100.0 };
    const int norders = (int)(sizeof ORDERS / sizeof ORDERS[0]);
    static const double SPACINGS[] = { 0.05, 0.2, 0.8 };

    int asserted_bad = 0, asymmetric = 0, diagonal_bad = 0;
    double asym_worst = 0.0;

    printf("  %8s %8s %12s %12s   %s\n",
           "spacing", "nu", "factors", "min pivot", "");
    for (int s = 0; s < 3; s++) {
        const double spacing = SPACINGS[s];
        for (int k = 0; k < norders; k++) {
            const double nu = ORDERS[k], length_scale = 1.0;
            for (int i = 0; i < n; i++) x[i] = spacing * (double)i;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    h    [(size_t)i*n + j] = fabs(x[i] - x[j]);
                    order[(size_t)i*n + j] = nu;
                    scale[(size_t)i*n + j] = length_scale;
                }
            if (matern_covariance(W, h, order, scale, m2, work) != 0) {
                printf("  matern_covariance failed at nu = %g\n", nu);
                failures++; continue;
            }

            /* Symmetry and the unit diagonal.  A result does not depend on its position
             * in the batch (matern.h, THREAD SAFETY), so W[i][j] and W[j][i] are equal. */
            for (int i = 0; i < n; i++) {
                if (W[(size_t)i*n + i] != 1.0) diagonal_bad++;
                for (int j = 0; j < i; j++) {
                    const double a = W[(size_t)i*n + j], b = W[(size_t)j*n + i];
                    if (a == b) continue;
                    asymmetric++;
                    const double d = fabs(a - b) / fabs(a);
                    if (d > asym_worst) asym_worst = d;
                }
            }

            double pivot = 0.0;
            const int got = largest_factorable(W, n, A, &pivot);

            /* Asserted at spacing 0.8 and for nu <= 5, where the pivot margin is large.
             * For large nu on a dense grid the true eigenvalues fall below double
             * precision and the matrix is numerically singular. */
            const int must_be_full = (spacing >= 0.8) || (nu <= 5.0);
            const char *mark = (got == n) ? ""
                             : must_be_full ? "   <-- REGRESSION: this must factor fully"
                                            : "   partial: conditioning, not asserted";
            if (must_be_full && got != n) asserted_bad++;
            printf("  %8.2f %8g %6d of %3d %12.2e%s\n",
                   spacing, nu, got, n, pivot, mark);
        }
    }
    putchar('\n');
    report("every asserted configuration factors fully", asserted_bad,
           "configurations did not, and their margin was three orders of magnitude");
    report("every matrix is exactly symmetric", asymmetric,
           "pairs W[i][j] != W[j][i] -- a result depends on its place in the batch");
    if (asymmetric)
        printf("      largest relative difference %.3e\n", asym_worst);

    /* Batch invariance: every point evaluated alone against the same point inside one
     * call, and inside calls of 37 points.  The first set is the nu = 20 grid above; the
     * second mixes orders in short runs (half-integers among them, so that the closed form
     * and the router share tiles), separations across the regions and a length that is not
     * a multiple of any vector width. */
    {
        const double spacing = 0.05, nu = 20.0, length_scale = 1.0;
        for (int i = 0; i < n; i++) x[i] = spacing * (double)i;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                h    [(size_t)i*n + j] = fabs(x[i] - x[j]);
                order[(size_t)i*n + j] = nu;
                scale[(size_t)i*n + j] = length_scale;
            }
        static const double MIXED[] = { 0.5, 2.3, 1.5, 7.0, 12.5, 29.5, 20.0, 2.5, 31.0, 0.75 };
        const size_t nmixed = sizeof MIXED / sizeof MIXED[0];
        const size_t mixed_points = m2 - 13;
        double *mh = malloc(mixed_points * sizeof *mh), *mnu = malloc(mixed_points * sizeof *mnu);
        double *ms = malloc(mixed_points * sizeof *ms), *one = malloc(mixed_points * sizeof *one);
        double *chunked = malloc(mixed_points * sizeof *chunked);
        if (!mh || !mnu || !ms || !one || !chunked) { puts("out of memory"); return 1; }
        for (size_t i = 0, k = 0; i < mixed_points; k++)           /* runs of 1 .. 5 points */
            for (size_t r = 0; r <= k % 5 && i < mixed_points; r++, i++) {
                mnu[i] = MIXED[k % nmixed];
                mh[i]  = 0.013 * (double)(i % 3001);                   /* z from 0 to ~250 */
                ms[i]  = (i % 2) ? 1.0 : 0.7;
            }
        size_t moved_grid = 0, moved_mixed = 0, moved_chunked = 0;
        matern_covariance(W, h, order, scale, m2, work);
        for (size_t i = 0; i < m2; i++) {
            double alone = 0.0;
            matern_covariance(&alone, &h[i], &order[i], &scale[i], 1, work);
            if (alone != W[i]) moved_grid++;
        }
        matern_covariance(W, mh, mnu, ms, mixed_points, work);
        for (size_t i = 0; i < mixed_points; i += 37) {
            const size_t len = (mixed_points - i < 37) ? mixed_points - i : 37;
            matern_covariance(chunked + i, mh + i, mnu + i, ms + i, len, work);
        }
        for (size_t i = 0; i < mixed_points; i++) {
            matern_covariance(one + i, mh + i, mnu + i, ms + i, 1, work);
            if (one[i] != W[i]) moved_mixed++;
            if (chunked[i] != W[i]) moved_chunked++;
        }
        report("nu = 20 grid: one call equals single-point calls", (int)moved_grid,
               "points differ between the batch and the point alone");
        report("mixed orders: one call equals single-point calls", (int)moved_mixed,
               "points differ between the batch and the point alone");
        report("mixed orders: one call equals calls of 37 points", (int)moved_chunked,
               "points differ between one call and shorter calls");
        free(mh); free(mnu); free(ms); free(one); free(chunked);
    }
    report("the diagonal is exactly 1.0", diagonal_bad,
           "diagonal entries were not 1.0 -- M(0) = 1 at every order");

    matern_workspace_destroy(work);
    free(W); free(A); free(h); free(order); free(scale); free(x);
    printf("\n%s\n", failures ? "POSITIVE DEFINITENESS FAILED" : "positive definiteness holds");
    return failures != 0;
}
