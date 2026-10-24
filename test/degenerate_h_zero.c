/* degenerate_h_zero.c -- M(0) = 1 exactly (the bit pattern of 1.0), the diagonal of every
 * covariance matrix, at every order and in three call shapes (make h-zero). */
#include <stdio.h>
#include "matern.h"

int main(void)
{
    printf("backend %s\n", matern_backend_name());
    /* the last two: 2 nu overflows (nu > DBL_MAX/2), so the derived z is NaN */
    const double orders[] = {0.001,0.25,0.5,1.0,1.5,2.0,2.5,2.9999,3.0,5.5,
                             12.0,25.5,26.0,29.5,30.0,40.5,100.0,1000.0,
                             0x1p+1023,0x1.fffffffffffffp+1023};
    const size_t k = sizeof orders/sizeof orders[0];

    /* one call per order, then one batched call: the run-splitter and the
     * router take different paths through h = 0 */
    int bad = 0;
    for (size_t i = 0; i < k; i++) {
        const double h = 0.0, scale = 1.0;
        double out = -1.0;
        const int st = matern_covariance(&out, &h, &orders[i], &scale, 1, NULL);
        if (st != 0 || out != 1.0) {
            printf("  SINGLE nu=%-10g status=%d out=%.17g\n", orders[i], st, out);
            bad++;
        }
    }
    printf("  single calls: %d of %zu not exactly 1.0\n", bad, k);

    double h[64], nu[64], scale[64], out[64];
    for (size_t i = 0; i < k; i++) { h[i]=0.0; nu[i]=orders[i]; scale[i]=1.0; out[i]=-1.0; }
    const int st = matern_covariance(out, h, nu, scale, k, NULL);
    int badb = 0;
    for (size_t i = 0; i < k; i++)
        if (out[i] != 1.0) { printf("  BATCH  nu=%-10g out=%.17g\n", orders[i], out[i]); badb++; }
    printf("  batched call: status=%d, %d of %zu not exactly 1.0\n", st, badb, k);

    /* mixed batch: h = 0 interleaved with ordinary points in one vector */
    for (size_t i = 0; i < 64; i++) { h[i]=(i%2)?0.0:0.3; nu[i]=1.5; scale[i]=1.0; out[i]=-1.0; }
    matern_covariance(out, h, nu, scale, 64, NULL);
    int badm = 0;
    for (size_t i = 1; i < 64; i += 2) if (out[i] != 1.0) badm++;
    printf("  mixed vector (nu=1.5): %d of 32 h=0 lanes not exactly 1.0\n", badm);
    return (bad || badb || badm) ? 1 : 0;
}
