/* temme_small_argument.c -- scalar small-argument kernel for z < 1, nu <= 30: Temme's
 * series at |mu| <= 1/2, then the order recurrence with the Matern prefactor folded into
 * it, as the vector and CUDA kernels (see the accompanying paper, Temme formulation). */
#include <float.h>
#include <math.h>

#include "scalar_internal.h"

/* Hidden under -std=c11; the same tokens as glibc's M_PI, so a redefinition is benign. */
#define M_PI    3.14159265358979323846

/* T_i = 2 (z/2)^(mu0+i) K_(mu0+i) / Gamma(mu0+i) with T_(i+1) = T_i + l_i r_i,
 * l_(i+1) = T_i (z/2)^2 r_i, r_i = 1/(mu0+i): a contraction on O(1) values, so rounding
 * errors are not amplified.  F_T = 1/p_0 = 2 (z/2)^mu0 / Gamma(1+mu0) comes from the
 * series' own seed, so no pow or tgamma follows the series. */
double matern_scalar_small_argument(double z, double nu){
    /* base: mu in [0.5, 1.5), nl upward steps to reach nu */
    int nl = (int)floor(nu - 0.5);
    if (nl < 0) nl = 0;
    double mu = nu - (double)nl;                     /* in [0.5, 1.5) when nu >= 0.5 */
    if (nu < 0.5){ nl = 0; mu = nu; }                /* nu < 1/2: no recurrence at all */
    const double mu0 = (nu < 0.5) ? nu : mu - 1.0;   /* Temme's series wants |mu0|<=1/2 */

    /* Temme series for K_mu0 and K_(mu0+1), |mu0| <= 1/2 */
    const double x2 = 0.5 * z;
    const double pimu = M_PI * mu0;
    /* At integer nu (mu0 = 0) pimu/sin(pimu) and sinh(e)/e are removable singularities;
     * the tests select the limit 1. */
    const double fact = (pimu == 0.0) ? 1.0 : pimu / sin(pimu);
    double d = -log(x2);
    double e = mu0 * d;
    const double fact2 = (e == 0.0) ? 1.0 : sinh(e) / e;
    /* 1/Gamma(1 -+ mu0) directly: the argument is in [1/2, 3/2], no reflection needed */
    const double gampl = 1.0 / tgamma(1.0 + mu0);
    const double gammi = 1.0 / tgamma(1.0 - mu0);
    const double mu0sq = mu0 * mu0;
    /* gam1, gam2 from the even/odd series: (gammi - gampl)/(2 mu0) cancels near nu = 0 and
     * integer nu; the series gives gam1(0) = -gamma, gam2(0) = 1 exactly, so mu0 = 0 needs
     * no special case. */
    double gam1 = RECIPROCAL_GAMMA_ODD_B[SCALAR_GAMMA_TERMS-1];
    double gam2 = RECIPROCAL_GAMMA_EVEN_C[SCALAR_GAMMA_TERMS-1];
    for (int j = SCALAR_GAMMA_TERMS-2; j >= 0; j--) gam1 = gam1*mu0sq + RECIPROCAL_GAMMA_ODD_B[j];
    for (int j = SCALAR_GAMMA_TERMS-2; j >= 0; j--) gam2 = gam2*mu0sq + RECIPROCAL_GAMMA_EVEN_C[j];
    gam1 = -gam1;
    double ff = fact * (gam1 * cosh(e) + gam2 * fact2 * d);
    double sum = ff;
    const double ee = exp(e);
    double pp = 0.5 * ee / gampl;
    double qq = 0.5 / (ee * gammi);
#ifndef WMS_UNFOLDED
    const double zg = 1.0 / pp;          /* F_T */
#endif
    double c = 1.0;
    const double dd2 = x2 * x2;
    double sum1 = pp;
    /* terms 0 .. SCALAR_SMALL_ARGUMENT_MAX_TERMS-1, the count of the vector and CUDA kernels */
    for (int i = 1; i < SCALAR_SMALL_ARGUMENT_MAX_TERMS; i++){
        const double di = (double)i;
        ff = (di * ff + pp + qq) / (di * di - mu0sq);
        c *= dd2 / di;
        pp /= (di - mu0);
        qq /= (di + mu0);
        const double del = c * ff;
        sum += del;
        sum1 += c * (pp - di * ff);
        if (fabs(del) < fabs(sum) * DBL_EPSILON) break;
    }
    const double K_mu0 = sum;            /* K_mu0(z)     */

#ifdef WMS_UNFOLDED
    const double K_mu1 = sum1 * 2.0 / z; /* K_(mu0+1)(z) */
    /* The twin: upward recurrence on K itself in nl steps (base order nu for nu < 1/2). */
    double K;
    if (nu < 0.5) K = K_mu0;
    else {
        double k_prev = K_mu0, k_cur = K_mu1, a = mu;            /* K_(mu-1), K_mu */
        for (int i = 0; i < nl; i++){
            const double k_next = (2.0 * a / z) * k_cur + k_prev;
            k_prev = k_cur; k_cur = k_next; a += 1.0;
        }
        K = k_cur;
    }
    return WMS_K_RESULT(K);
#else
    /* nu < 1/2: T_0 = F_T mu0 K_mu0, as 1/Gamma(mu0) = mu0/Gamma(1+mu0).  Otherwise
     * T_1 = F_T sum1 at order mu0 + 1 and l_1 = F_T K_mu0 (z/2)^2, in which mu0 cancels, then
     * nl steps to order nu (the operation order of temme_small_argument_device.cuh). */
    if (nu < 0.5) return zg * mu0 * K_mu0;
    double g = zg * sum1;
    double v = zg * K_mu0 * dd2;
    for (int i = 1; i <= nl; i++){
        const double r  = 1.0 / (mu0 + (double)i);
        const double gn = v * r + g;
        v = g * dd2 * r;
        g = gn;
    }
    return g;
#endif
}
