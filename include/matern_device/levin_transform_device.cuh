/* levin_transform_device.cuh -- device Levin transform of the large-argument series, with the
 * term ratios, weights and phases of the host AVX2/width-1 form and the exponent folded in
 * through matern_large_argument_folded_prefactor_device.  Routed only under
 * MATERN_ROUTE_LEVIN_BIT. */
#ifndef MATERN_LEVIN_TRANSFORM_DEVICE_CUH
#define MATERN_LEVIN_TRANSFORM_DEVICE_CUH

#include "asymptotic_large_argument_device.cuh"   /* matern_large_argument_folded_prefactor_device */

#define MATERN_LEVIN_TERMS_DEVICE 15

/* Must equal LEVIN_WEIGHTS in levin_transform_{avx2,avx512,scalar}.h. */
static __device__ __constant__ double MATERN_LEVIN_WEIGHTS_DEVICE[15] = {
    3.7391005219784331e-07, -3.9260555480773548e-05, 0.0013610325900001495,
    -0.023137554030002543,   0.22906178489702517,    -1.4507246376811593,
    6.2173913043478262,     -18.652173913043477,     39.894927536231883,
    -61.172222222222224,     66.733333333333334,    -50.555555555555557,
    25.277777777777779,     -7.5,                    1.0 };

/* M(h) directly: the Levin ratio times the folded exponent. */
static __device__ __forceinline__ double matern_levin_fold_dd_argument_device(double Zh,
                                                              double Zl, double nu)
{
    if (!(Zh > 0.0) || !(Zh < CUDART_INF)) return 0.0;

    const double mu = 4.0 * nu * nu;

    /* phase 1a: the independent term ratios. */
    double rho[MATERN_LEVIN_TERMS_DEVICE];
#pragma unroll
    for (int i = 1; i <= MATERN_LEVIN_TERMS_DEVICE; i++) {
        const double t = (double)(2 * i - 1);
        rho[i - 1] = (mu - t * t) / (Zh * (double)(8 * i));
    }
    /* phase 1b: the partial sums A_i and the remainder estimates omega_i = t_{i+1}. */
    double seq[MATERN_LEVIN_TERMS_DEVICE], wt[MATERN_LEVIN_TERMS_DEVICE];
    double S = 0.0, T = 1.0;
#pragma unroll
    for (int i = 0; i < MATERN_LEVIN_TERMS_DEVICE; i++) {
        S += T;
        T *= rho[i];
        seq[i] = S;
        wt[i]  = T;
    }
    /* phases 2 and 3: num = sum c_k A_k/omega_k, den = sum c_k/omega_k. */
    double num = 0.0, den = 0.0;
#pragma unroll
    for (int k = 0; k < MATERN_LEVIN_TERMS_DEVICE; k++) {
        const double inv = 1.0 / wt[k];
        const double c   = MATERN_LEVIN_WEIGHTS_DEVICE[k];
        num = fma(c, seq[k] * inv, num);
        den = fma(c, inv, den);
    }
    /* Half-integer nu terminates the series (factor 0 at 2nu = 2i-1) and num/den would be NaN.
     * T then stays 0, so seq[14] is the exact sum and testing the last weight suffices. */
    const double ratio = (wt[MATERN_LEVIN_TERMS_DEVICE - 1] == 0.0)
                       ? seq[MATERN_LEVIN_TERMS_DEVICE - 1] : (num / den);

    const double eE = matern_large_argument_folded_prefactor_device(Zh, Zl, nu);
    const double v  = eE * ratio;
    return (v > 0.0 && v <= 1.0) ? v : 0.0;    /* M lies in (0, 1] */
}

#endif /* MATERN_LEVIN_TRANSFORM_DEVICE_CUH */
