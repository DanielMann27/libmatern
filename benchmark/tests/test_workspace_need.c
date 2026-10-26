/* test_workspace_need.c -- the backends without a workspace in dispatch.c (each scalar build
 * this CPU runs, matched, matched_fma) succeed with a NULL one, as benchmark_cpu.c and
 * threads_cpu.c assume.  Run by benchmark/tests/run_tests.sh; exit status 1 on any failure. */
#include <stdio.h>
#include "matern.h"

typedef int (*entry)(double *, const double *, const double *,
                     const double *, size_t, matern_workspace *);
int matern_evaluate_scalar       (double *, const double *, const double *,
                                  const double *, size_t, matern_workspace *);
int matern_evaluate_scalar_fma   (double *, const double *, const double *,
                                  const double *, size_t, matern_workspace *);
int matern_evaluate_scalar_avx2  (double *, const double *, const double *,
                                  const double *, size_t, matern_workspace *);
int matern_evaluate_scalar_avx512(double *, const double *, const double *,
                                  const double *, size_t, matern_workspace *);
int matern_evaluate_matched      (double *, const double *, const double *,
                                  const double *, size_t, matern_workspace *);
int matern_evaluate_matched_fma  (double *, const double *, const double *,
                                  const double *, size_t, matern_workspace *);

int main(void)
{
    double h[4] = {0.5, 1.0, 1.5, 2.0}, nu[4] = {0.75, 1.25, 2.5, 3.5};
    double scale[4] = {1, 1, 1, 1}, out[4];
    int bad = 0;

    /* The feature tests of dispatch.c: an entry runs only where its objects cannot fault. */
    __builtin_cpu_init();
    const int has_fma    = __builtin_cpu_supports("fma") && __builtin_cpu_supports("avx");
    const int has_avx2   = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const int has_avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
                           && __builtin_cpu_supports("fma");
    const struct { const char *name; entry f; int runs; } e[] = {
        {"scalar (portable)", matern_evaluate_scalar,        1},
        {"scalar (fma)",      matern_evaluate_scalar_fma,    has_fma},
        {"scalar (avx2)",     matern_evaluate_scalar_avx2,   has_avx2},
        {"scalar (avx512)",   matern_evaluate_scalar_avx512, has_avx512},
        {"matched",           matern_evaluate_matched,       1},
        {"matched_fma",       matern_evaluate_matched_fma,   has_fma},
    };

    for (size_t i = 0; i < sizeof e / sizeof e[0]; i++) {
        if (!e[i].runs) {
            printf("%s: not run, this CPU lacks its instruction set\n", e[i].name);
            continue;
        }
        if (e[i].f(out, h, nu, scale, 4, NULL) != MATERN_OK) {
            printf("%s requires a workspace, which the harnesses do not build for it\n",
                   e[i].name);
            bad = 1;
        }
    }
    if (!bad) printf("scalar and matched* need no workspace\n");
    return bad;
}
