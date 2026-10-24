/* test_workspace_need.c -- scalar succeeds with a NULL workspace and matched (the vector
 * pipeline at width 1) refuses it, as the harness exemption in benchmark_cpu.c assumes.
 * Run by benchmark/tests/run_tests.sh; exit status 1 on any failure. */
#include <stdio.h>
#include "matern.h"

int matern_evaluate_scalar (double *, const double *, const double *,
                            const double *, size_t, matern_workspace *);
int matern_evaluate_matched(double *, const double *, const double *,
                            const double *, size_t, matern_workspace *);

int main(void)
{
    double h[4] = {0.5, 1.0, 1.5, 2.0}, nu[4] = {0.75, 1.25, 2.5, 3.5};
    double scale[4] = {1, 1, 1, 1}, out[4];
    int bad = 0;

    if (matern_evaluate_scalar(out, h, nu, scale, 4, NULL) != MATERN_OK) {
        printf("scalar REQUIRES a workspace -- the harness exemption is now wrong\n");
        bad = 1;
    }
    if (matern_evaluate_matched(out, h, nu, scale, 4, NULL) == MATERN_OK) {
        printf("matched runs WITHOUT a workspace -- it must not be exempted, but\n"
               "the reason it is not has gone away; re-check pipeline_vector.h\n");
        bad = 1;
    }
    if (!bad) printf("scalar needs no workspace, matched does\n");
    return bad;
}
