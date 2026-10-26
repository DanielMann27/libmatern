/* continued_fraction_wronskian.c -- scalar continued-fraction kernel for 1 <= z <= C(nu):
 * ascending I-series pair, CF2 by modified Lentz and the Wronskian, with the Matern
 * prefactor cancelled (see the accompanying paper, continued-fraction formulation). */
#include <float.h>
#include <math.h>

#include "scalar_internal.h"

/* gnu = (z/2)^nu / Gamma(nu) is a common factor of both series and cancels identically,
 * so the seeds are S = 1/nu, T = 2/z; with D = z K_nu/K_(nu-1) and the Wronskian,
 * 2 / (z (O2 + O1 z/D)) = 2^(1-nu) z^nu K_nu(z) / Gamma(nu) = M. */
double matern_scalar_continued_fraction(double z, double nu){
    const double zh = z * 0.5, z4 = zh * zh;          /* z/2 and (z/2)^2 */
    /* S, T: current terms; O1, O2: partial sums.  The term ratios (z/2)^2/(k(nu+k)) and
     * (z/2)^2/(k(nu+k-1)) differ by one index, so both series share one loop. */
#ifdef MATERN_SCALAR_UNFOLDED
    /* The twin keeps gnu in the seeds, so O1 = I_nu(z) and O2 = I_(nu-1)(z). */
    const double gnu = pow(zh, nu) / tgamma(nu);
    double S = gnu/nu, T = gnu/zh, O1 = 0.0, O2 = 0.0;
#else
    double S = 1.0/nu, T = 1.0/zh, O1 = 0.0, O2 = 0.0;
#endif
    for (int i = 0; i < SCALAR_FRACTION_SERIES_TERMS; i++){
        O1 += S; O2 += T;
        if (fabs(S) <= DBL_EPSILON * fabs(O1)) break;    /* per-element exit */
        const double f1 = (double)(i + 1);
        S = (S / ((nu + f1) * f1)) * z4;
        T = (T / ((nu + (double)i) * f1)) * z4;
    }
    /* CF2 by modified Lentz; step k has A = (nu-1)^2 - (2k-1)^2/4 and B = 2(z+k).  1e-50 is
     * the nonzero start for a fraction with no leading term and cancels in the first step;
     * at half-integer nu A = 0 gives PROD = 1 to within rounding, which ends the loop. */
    const double nun = nu - 1.0, nunu = nun * nun;
    double C = 1.0, low1 = 1e-50, low2 = 1e-50, H = 0.0;
    for (int i = 0; i < SCALAR_FRACTION_LENTZ_STEPS; i++){
        const double t2 = 2.0*C - 1.0;
        const double A  = nunu - 0.25*(t2*t2);
        const double B  = 2.0*(z + C);
        low2 = B + A/low2;
        H    = 1.0/(A*H + B);
        const double PROD = H * low2;
        low1 *= PROD;
        if (fabs(PROD - 1.0) <= DBL_EPSILON) break;
        C += 1.0;
    }
    const double D = nun + z + 0.5 + low1;            /* z K_nu / K_(nu-1)        */
#ifdef MATERN_SCALAR_UNFOLDED
    const double K = 1.0 / (z * (O2 + O1 * z / D));   /* K_nu, by the Wronskian   */
    return MATERN_SCALAR_K_RESULT(K);
#else
    return 2.0 / (z * (O2 + O1 * z / D));             /* the covariance, as above */
#endif
}
