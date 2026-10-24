/* debye_uniform_order.c -- scalar uniform-order (Debye) kernel for nu >= 30 at every z,
 * M = exp(nu G + cc) POLY / sqrt(t) with cc the Stirling tail of -lnGamma(nu) (see the
 * accompanying paper, large-order formulation). */
#include <math.h>

#include "scalar_internal.h"

/* t = sqrt(1 + (z/nu)^2), G = ln((1+t)/2) - t + 1, POLY the u_k Debye correction series. */
double matern_scalar_debye(double z, double nu){
    const double x  = z / nu;
    const double t  = sqrt(1.0 + x*x);
    const double p  = 1.0 / t;
    const double q  = p * p;

    /* POLY_k(q), the coefficient polynomials of the u_k series */
    double poly[DEBYE_SCALAR_TERM_COUNT];
    for (int k = 0; k <= SCALAR_DEBYE_TOP; k++){
        double acc = (k & 1) ? -MATERN_DEBYE_SCALAR_C[k][k] : MATERN_DEBYE_SCALAR_C[k][k];
        for (int m = k - 1; m >= 0; m--)
            acc = acc * q + ((m & 1) ? -MATERN_DEBYE_SCALAR_C[k][m] : MATERN_DEBYE_SCALAR_C[k][m]);
        poly[k] = acc;
    }
    /* POLY = SUM_k (-p/nu)^k POLY_k(q) in one Horner chain (the same polynomial as a
     * split by the parity of k). */
    const double in = -p / nu;
    double POLY = poly[SCALAR_DEBYE_TOP];
    for (int k = SCALAR_DEBYE_TOP - 1; k >= 0; k--) POLY = POLY * in + poly[k];

#ifdef WMS_UNFOLDED
    /* The twin: K_nu(nu x) = sqrt(pi/2nu) e^{-nu eta} / (1 + x^2)^{1/4} POLY with
     * eta = t + ln(x / (1 + t)) as written; the router applies the prefactor. */
    const double eta = t + log(x / (1.0 + t));
#  if WMS_UNFOLDED == 2
    return 0.5 * log(WMS_PI_2 / nu) - nu * eta - 0.5 * log(t) + log(POLY);
#  else
    return sqrt(WMS_PI_2 / nu) * exp(-nu * eta) / sqrt(t) * POLY;
#  endif
#else

    /* Cancels to O(x^2); evaluated as written, not as log1p(d/2) - d, d = t - 1. */
    const double G = log(0.5 * (1.0 + t)) - t + 1.0;

    /* Stirling tail of -lnGamma(nu) */
    const double rn = 1.0/nu, rn2 = rn*rn;
    static const double WMS_STIRL[6] = { -1.0/12.0, 1.0/360.0, -1.0/1260.0,
                                         1.0/1680.0, -1.0/1188.0, 691.0/360360.0 };
    double cc = WMS_STIRL[SCALAR_STIRLING_TERMS-1];
    for (int j = SCALAR_STIRLING_TERMS-2; j >= 0; j--) cc = cc*rn2 + WMS_STIRL[j];
    cc = rn * cc;

    const double E = nu * G + cc;
    return exp(E) * POLY / sqrt(t);
#endif
}
