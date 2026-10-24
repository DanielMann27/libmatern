/* temme_small_argument_device.cuh -- device small-argument (Temme) kernel: Temme's series
 * at order mu = xnu - nl, then the order recurrence with the Matern prefactor
 * folded into it (see the accompanying paper, Temme formulation).  Host counterpart:
 * temme_small_argument.h. */
#ifndef MATERN_TEMME_SMALL_ARGUMENT_DEVICE_CUH
#define MATERN_TEMME_SMALL_ARGUMENT_DEVICE_CUH

#include <float.h>            /* DBL_EPSILON, DBL_MIN */

/* Series term ceiling, relative convergence tolerance, and pi. */
#define MATERN_TEMME_MAX_TERMS_DEVICE 20
#define MATERN_TEMME_EPS_DEVICE       DBL_EPSILON
#define MATERN_TEMME_PI_DEVICE        3.141592653589793238462643383279502884

/* gam1, gam2 as cancellation-free series in mu^2: same values as RECIPROCAL_GAMMA_ODD_B (G1S)
 * and RECIPROCAL_GAMMA_EVEN_C (G2S) in reciprocal_gamma_coefficients.h, 11 terms each. */
static __device__ __constant__ double MATERN_TMWS_G1S[11] = {
    5.77215664901532866e-01, -4.20026350340952370e-02, -4.21977345555443334e-02,
    7.21894324666309990e-03, -2.15241674114950975e-04, -2.01348547807882387e-05,
    1.13302723198169593e-06, 6.11609510448141609e-09, -1.18127457048702004e-09,
    7.78226343990507081e-12, 5.10037028745447575e-13 };
static __device__ __constant__ double MATERN_TMWS_G2S[11] = {
    1.00000000000000000e+00, -6.55878071520253902e-01, 1.66538611382291479e-01,
    -9.62197152787697303e-03, -1.16516759185906517e-03, 1.28050282388116196e-04,
    -1.25049348214267063e-06, -2.05633841697760707e-07, 5.00200764446922295e-09,
    1.04342671169110054e-10, -3.69680561864220598e-12 };

static __device__ __forceinline__ double matern_temme_gam1_series_device(double m2){
    double r = MATERN_TMWS_G1S[10];
    for (int j = 9; j >= 0; j--) r = fma(r, m2, MATERN_TMWS_G1S[j]);
    return -r;
}

static __device__ __forceinline__ double matern_temme_gam2_series_device(double m2){
    double r = MATERN_TMWS_G2S[10];
    for (int j = 9; j >= 0; j--) r = fma(r, m2, MATERN_TMWS_G2S[j]);
    return r;
}

/* Classical Temme names (sum = K_xmu, sum1 = (x/2) K_(xmu+1)); the folded pair is
 * g_a = 2 (z/2)^a K_a / Gamma(a) and v_a = g_(a-1) (z/2)^2 / (a-1),
 * with g_(a+1) = g_a + v_a / a. */
/* lx2 = ln(x/2) formed by the caller from the inputs; read only where x is subnormal or 0
 * (the derived z lost bits or underflowed, or x/2 rounds to 0 at x = 2^-1074), where
 * -log(x/2) has lost bits or is infinite.  x/2 is then exp(lx2), which is negligible
 * unless the derivation underflowed in an intermediate product. */
static __device__ double matern_temme_small_argument_fold_device(double x, double xnu, double lx2)
{
    int nl = (int)(xnu + 0.5);
    double xmu = xnu - nl, xmu2 = xmu * xmu, x2 = 0.5 * x;
    double pimu = MATERN_TEMME_PI_DEVICE * xmu;
    /* nl = (int)(xnu + 0.5) makes xmu exactly 0 at integer orders, where both quotients
     * are 0/0; the singularities are removable with limit 1, hence the exact-zero tests */
    double fact = (pimu == 0.0) ? 1.0 : pimu / sin(pimu);
    double d = -log(x2);
    if (!(x >= DBL_MIN)) { x2 = exp(lx2); d = -lx2; }
    double e = xmu * d;
    double fact2 = (e == 0.0) ? 1.0 : sinh(e) / e;
    double gam1 = matern_temme_gam1_series_device(xmu2);
    double gam2 = matern_temme_gam2_series_device(xmu2);
    double gampl = gam2 - xmu * gam1, gammi = gam2 + xmu * gam1;
    double ff = fact * (gam1 * cosh(e) + gam2 * fact2 * d), sum = ff;
    e = exp(e);
    double p = 0.5 * e / gampl, q = 0.5 / (e * gammi), c = 1.0;
    const double ZG = 1.0 / p;              /* = 2 (z/2)^xmu / Gamma(1+xmu) */
    d = x2 * x2;                            /* (z/2)^2 */
    double sum1 = p;
    for (int i = 1; i < MATERN_TEMME_MAX_TERMS_DEVICE; i++)
    {
        ff = (i * ff + p + q) / (i * i - xmu2);
        c *= d / i; p /= (i - xmu); q /= (i + xmu);
        double del = c * ff; sum += del;
        double del1 = c * (p - i * ff); sum1 += del1;
        if (fabs(del) < fabs(sum) * MATERN_TEMME_EPS_DEVICE) break;
    }
    /* nl == 0: g_0 = ZG*xmu*sum, since 1/Gamma(xmu) = xmu/Gamma(1+xmu).
     * nl >= 1: start at (g_1, v_1) and advance nl-1 times; xmu cancels in v_1. */
    if (nl == 0) return ZG * xmu * sum;
    double g = ZG * sum1;                   /* g_1 */
    double v = ZG * sum * d;                /* v_1 = g_0 (z/2)^2 / xmu */
    for (int i = 1; i <= nl - 1; i++){
        double r  = 1.0 / (xmu + i);
        double gn = fma(v, r, g);
        v = g * d * r;
        g = gn;
    }
    return g;                               /* M(h) at order xnu */
}

#endif /* MATERN_TEMME_SMALL_ARGUMENT_DEVICE_CUH */
