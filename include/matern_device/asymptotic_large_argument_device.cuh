/* asymptotic_large_argument_device.cuh -- device large-argument series with the first-order
 * z_lo correction, and its single-exponent extended-range form (see the accompanying paper,
 * large-argument formulation).  Host counterpart: asymptotic_large_argument.h. */
#ifndef MATERN_ASYMPTOTIC_LARGE_ARGUMENT_DEVICE_CUH
#define MATERN_ASYMPTOTIC_LARGE_ARGUMENT_DEVICE_CUH

#include <float.h>            /* DBL_EPSILON */
#include <math_constants.h>   /* CUDART_INF  */
#include "dd_arithmetic_device.cuh"
#include "debye_uniform_order_device.cuh"   /* the tail falls back to Debye */

/* -log(DBL_MIN), and ln 2 and 0.5 ln(pi/2) as double-doubles: same values as LN2_HI/LO and
 * HALF_LN_PI_OVER_2_HI/LO in large_argument_coefficients.h. */
#define MATERN_EXP_RANGE_MAX_DEVICE      708.396418532264
#define MATERN_LN2_HI_DEVICE             0.6931471805599453
#define MATERN_LN2_LO_DEVICE             2.3190468138462996e-17
#define MATERN_HALF_LN_PI_2_HI_DEVICE    0.22579135264472738
#define MATERN_HALF_LN_PI_2_LO_DEVICE  (-1.3082408289203276e-17)

/* pref times the series, corrected to v (1 + g Zl) with g = d lnM/dz including Sigma'/Sigma.
 * length_sum is a parameter because the cap is two-sided: past the series minimum the
 * terms diverge, so more terms can be less accurate. */
static __device__ __forceinline__ double matern_large_argument_dd_argument_device(double Zh, double Zl, double nu,
                                                         double pref, int length_sum, double tol)
{
    double nu42 = 4.0 * nu * nu;
    double w    = 1.0 / (8.0 * Zh);
    double sum = 1.0, start = 1.0, D = 0.0;
    for (int j = 1; j < length_sum; j++){
        double t = (double)(2*j - 1);
        start = start * ((nu42 - t*t) / (double)j) * w;
        sum   = start + sum;
        D     = fma((double)j, start, D);            /* SUM k a_k, for Sigma'/Sigma */
        if (fabs(start) < tol * fabs(sum)) break;      /* relative to the sum */
    }
    double v  = pref * (rsqrt(Zh) * exp(-Zh) * sum * sqrt(M_PI / 2));
    double rz = 8.0 * w;                             /* = fl(1/Zh) exactly (8 is a power of two) */
    double g  = fma(nu - 0.5, rz, -1.0) - (D * rz) / sum;
    return fma(v * Zl, g, v);
}

/* exp(E), E = (nu-1/2) ln z - z - lnGamma(nu) + (1-nu) ln 2 + 1/2 ln(pi/2) in double-double
 * with Zl absorbed exactly; shared by the asymptotic sum and its Levin transform. */
static __device__ __forceinline__ double matern_wm_exponent_device(double Zh, double Zl,
                                                                   double nu)
{
    const double rz = 1.0 / Zh;
    /* ln z in dd, Zl absorbed: ln(Zh+Zl) = ln Zh + Zl/Zh */
    double lzh, lzl; matern_dd_log_device(Zh, &lzh, &lzl);
    lzl = fma(Zl, rz, lzl);
    const double a  = nu - 0.5;
    double ph = a * lzh;
    double pl = fma(a, lzl, fma(a, lzh, -ph));
    double Eh, e1; matern_two_sum_device(ph, -Zh, &Eh, &e1);
    double El = (e1 + pl) - Zl;
    { double gh, gl; matern_dd_log_device(tgamma(nu), &gh, &gl);
      double h2, e2; matern_two_sum_device(Eh, -gh, &h2, &e2);
      El = (e2 + El) - gl; Eh = h2; }
    { const double b = 1.0 - nu;
      const double qh = b * MATERN_LN2_HI_DEVICE;
      const double ql = fma(b, MATERN_LN2_LO_DEVICE, fma(b, MATERN_LN2_HI_DEVICE, -qh));
      double h2, e2; matern_two_sum_device(Eh, qh, &h2, &e2);
      El = (e2 + El) + ql; Eh = h2; }
    { double h2, e2; matern_two_sum_device(Eh, MATERN_HALF_LN_PI_2_HI_DEVICE, &h2, &e2);
      El = (e2 + El) + MATERN_HALF_LN_PI_2_LO_DEVICE; Eh = h2; }
    /* Second order in El, since the dd_log low words are not normalised (as on the CPU). */
    double eE = exp(Eh);
    return fma(eE, fma(0.5 * El, El, El), eE);     /* exp(Eh+El) = exp(Eh)(1 + El + El^2/2) */
}

static __device__ __forceinline__ double matern_large_argument_fold_device(double Zh, double Zl,
                                                             double nu, int length_sum)
{
    if (!(Zh < CUDART_INF)) return 0.0;                       /* z = inf: M is 0 (a NaN z also returns 0) */
    /* nu >= 30 goes to the Debye expansion, the more accurate there; this also keeps the
     * series clear of its domain limit z >= nu^2/18 for nu > 50.  Same clamp as below:
     * where the square of x = z/nu overflows, Debye returns NaN and M is 0. */
    if (nu >= 30.0){
        const double v = matern_debye_fold_dd_argument_device(Zh, Zl, nu);
        return (v > 0.0 && v <= 1.0) ? v : 0.0;    /* M lies in (0, 1] */
    }

    /* Same recurrence as matern_large_argument_dd_argument_device; no derivative is needed,
     * the double-double exponent absorbs Zl exactly. */
    const double nu42 = 4.0 * nu * nu;
    const double w    = 1.0 / (8.0 * Zh);
    double sum = 1.0, start = 1.0;
    for (int j = 1; j < length_sum; j++){
        const double t = (double)(2*j - 1);
        start = start * ((nu42 - t*t) / (double)j) * w;
        sum   = start + sum;
        if (fabs(start) < DBL_EPSILON * fabs(sum)) break;
    }

    const double eE = matern_wm_exponent_device(Zh, Zl, nu);
    const double v = eE * sum;
    return (v > 0.0 && v <= 1.0) ? v : 0.0;        /* M lies in (0, 1] */
}

#endif /* MATERN_ASYMPTOTIC_LARGE_ARGUMENT_DEVICE_CUH */
