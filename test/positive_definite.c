/* positive_definite.c -- the largest leading block of 200 x 200 library covariance matrices
 * (one-dimensional grid, s = 1) that admits a Cholesky factorisation, the unit diagonal, and
 * symmetry and batch invariance (asserted exact on scalar, reported on the other backends). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
    const int is_scalar = (strncmp(matern_backend_name(), "scalar", 6) == 0);

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

            /* Symmetry and the unit diagonal.  On the CPU backends other than scalar
             * a result can depend on its position in the batch, so W[i][j] and
             * W[j][i] can differ in the last bit (reported); scalar routes per point
             * and is checked for exact symmetry. */
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
    if (is_scalar) {
        printf("  %-58s %s\n", "the matrix is exactly symmetric (scalar)",
               asymmetric == 0 ? "ok" : "FAILED");
        if (asymmetric) {
            printf("      %d pairs differ, largest relative %.3e\n", asymmetric, asym_worst);
            failures++;
        }
    } else {
        printf("  %-58s %d pairs differ, largest relative %.3e\n",
               "symmetry (reported)", asymmetric, asym_worst);
    }

    /* Batch invariance: a point evaluated alone against the same point inside a
     * batch.  Reported on the vector backends (matern.h, THREAD SAFETY), exact on
     * scalar. */
    {
        const double spacing = 0.05, nu = 20.0, length_scale = 1.0;
        for (int i = 0; i < n; i++) x[i] = spacing * (double)i;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                h    [(size_t)i*n + j] = fabs(x[i] - x[j]);
                order[(size_t)i*n + j] = nu;
                scale[(size_t)i*n + j] = length_scale;
            }
        matern_covariance(W, h, order, scale, m2, work);
        size_t moved = 0; double worst = 0.0;
        for (size_t i = 0; i < m2; i++) {
            double alone = 0.0;
            matern_covariance(&alone, &h[i], &order[i], &scale[i], 1, work);
            if (alone == W[i]) continue;
            moved++;
            const double d = fabs(alone - W[i]) / fabs(alone);
            if (d > worst) worst = d;
        }
        if (is_scalar) {
            printf("  %-58s %s\n", "batch and single-point calls agree exactly (scalar)",
                   moved == 0 ? "ok" : "FAILED");
            if (moved) {
                printf("      %zu of %zu points moved, largest relative %.3e\n", moved, m2, worst);
                failures++;
            }
        } else {
            printf("  %-58s %zu of %zu points differ, largest relative %.3e\n",
                   "batch vs single-point calls (reported)", moved, m2, worst);
        }
    }
    report("the diagonal is exactly 1.0", diagonal_bad,
           "diagonal entries were not 1.0 -- M(0) = 1 at every order");

    matern_workspace_destroy(work);
    free(W); free(A); free(h); free(order); free(scale); free(x);
    printf("\n%s\n", failures ? "POSITIVE DEFINITENESS FAILED" : "positive definiteness holds");
    return failures != 0;
}
