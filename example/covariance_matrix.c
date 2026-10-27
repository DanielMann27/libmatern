/* covariance_matrix.c -- example: builds the Matern covariance matrix of 512 equally
 * spaced points on [0, 1] and Cholesky-factors it; exits non-zero if it does not factor.
 * Build and run:   make example && ./build/example */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matern.h"

/* In-place Cholesky, lower triangle.  Returns the index of the first
 * non-positive pivot, or -1 if the factorisation succeeded. */
static int cholesky_lower(double *a, int n)
{
    for (int j = 0; j < n; j++) {

        double pivot = a[j * n + j];
        for (int k = 0; k < j; k++)
            pivot -= a[j * n + k] * a[j * n + k];

        if (!(pivot > 0.0))
            return j;

        pivot = sqrt(pivot);
        a[j * n + j] = pivot;

        for (int i = j + 1; i < n; i++) {
            double sum = a[i * n + j];
            for (int k = 0; k < j; k++)
                sum -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = sum / pivot;
        }
    }
    return -1;
}

int main(void)
{
    const int    n     = 512;         /* points, hence a 512 x 512 matrix   */
    const double order = 1.5;         /* Matern 3/2                          */
    const double scale = 0.30;        /* the scale parameter s               */

    printf("libmatern, backend %s\n\n", matern_backend_name());

    /* The points, equally spaced on [0, 1]. */
    double *position = malloc((size_t)n * sizeof *position);
    for (int i = 0; i < n; i++)
        position[i] = (double)i / (double)(n - 1);

    /* One workspace, reused for every row. */
    matern_workspace *work = NULL;
    if (matern_workspace_create(&work, 0) != MATERN_OK) {
        fprintf(stderr, "could not create a workspace\n");
        return 1;
    }

    /* The matrix, one call per row.  The constant order and scale parameter
     * are passed as arrays of one repeated value. */
    double *matrix   = malloc((size_t)n * (size_t)n * sizeof *matrix);
    double *distance = malloc((size_t)n * sizeof *distance);
    double *orders   = malloc((size_t)n * sizeof *orders);
    double *scales   = malloc((size_t)n * sizeof *scales);

    for (int i = 0; i < n; i++) {
        orders[i] = order;
        scales[i] = scale;
    }

    int status = MATERN_OK;
    for (int row = 0; row < n && status == MATERN_OK; row++) {
        for (int col = 0; col < n; col++)
            distance[col] = fabs(position[row] - position[col]);

        status = matern_covariance(matrix + (size_t)row * (size_t)n,
                                   distance, orders, scales,
                                   (size_t)n, work);
    }

    if (status != MATERN_OK) {
        printf("matern_covariance returned %d.\n", status);
        return 1;
    }

    printf("matrix built: %d x %d, order %.1f, scale parameter %.2f\n",
           n, n, order, scale);
    printf("  M(0)          = %.17g   (must be exactly 1)\n", matrix[0]);
    printf("  M(h_max)      = %.17g\n", matrix[n - 1]);
    /* A point's result does not depend on the other points of the call (matern.h,
     * THREAD SAFETY), so M(i,j) and M(j,i) are equal and the check prints 0. */
    printf("  symmetry check: max |M(i,j) - M(j,i)| = ");

    double asymmetry = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            const double d = fabs(matrix[(size_t)i * n + j] - matrix[(size_t)j * n + i]);
            if (d > asymmetry)
                asymmetry = d;
        }
    printf("%.3g\n\n", asymmetry);

    const int failed_at = cholesky_lower(matrix, n);

    if (failed_at < 0)
        printf("Cholesky factorisation succeeded: the matrix is positive definite.\n");
    else
        printf("Cholesky failed at pivot %d -- the matrix is not positive definite.\n",
               failed_at);

    matern_workspace_destroy(work);
    free(matrix); free(distance); free(orders); free(scales); free(position);
    return failed_at < 0 ? 0 : 1;
}
