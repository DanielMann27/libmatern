/* asymptotic_large_argument.c -- scalar large-argument kernel for z > C(nu), nu <= 30, its
 * single-exponent form past the exponential limit, and the assembly of the unfolded twin
 * (see the accompanying paper, large-argument formulation). */
/* Before any include, or it has no effect: exposes `lgamma_r`, the reentrant
 * lnGamma (plain `lgamma` writes the sign to a global, a data race). */
#define _DEFAULT_SOURCE 1

#include <float.h>
#include <math.h>

#include "scalar_internal.h"

/* ln|Gamma|: lgamma_r on the listed platforms; the fallback lgamma writes the global
 * signgam, a data race between threads. */
#if defined(__linux__) || defined(__GLIBC__) || defined(__BIONIC__) \
 || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) \
 || defined(__APPLE__)
static inline double scalar_log_gamma(double x)
{
    int sign;                      /* discarded: the order is positive here */
    return lgamma_r(x, &sign);
}
#else
#  define scalar_log_gamma(x)  lgamma(x)
#endif

/* Not ISO C; the same tokens as glibc's M_PI_2, so a redefinition is benign. */
#define M_PI_2  1.57079632679489661923

/* K_nu(z) ~ sqrt(pi/2z) e^-z SUM t_k, t_k/t_(k-1) = (4nu^2-(2k-1)^2)/(8kz).  The unfused
 * assembly 2 (z/2)^nu / Gamma(nu) * K_nu is safe because the router enters only for
 * nu <= 30 and z <= 708.396, where Gamma(nu) and (z/2)^nu are finite. */
double matern_scalar_large_argument_at(double zh, double nu){
    const double w  = 1.0 / zh;
    const double mu = 4.0 * nu * nu;
    double t = 1.0, s = 1.0;
    for (int k = 1; k < SCALAR_LARGE_ARGUMENT_MAX_TERMS; k++){
        const double kk = (double)k;
        const double tk = (double)(2*k - 1);
        t *= (mu - tk*tk) / (8.0*kk) * w;
        s += t;
        if (fabs(t) <= DBL_EPSILON * fabs(s)) break;     /* per-element exit, relative */
    }
#ifdef WMS_UNFOLDED
    /* matern_scalar_compose_unfolded writes the prefactor as the branch below does, so the
     * direct twin equals this backend bit for bit in this region. */
#  if WMS_UNFOLDED == 2
    return 0.5 * log(M_PI_2 / zh) - zh + log(s);
#  else
    const double kn   = sqrt(M_PI_2 / zh) * s * exp(-zh);
    return kn;
#  endif
#else
    const double pref = 2.0 * pow(zh*0.5, nu) / tgamma(nu);
    const double kn   = sqrt(M_PI_2 / zh) * s * exp(-zh);
    return pref * kn;
#endif
}


/* z > 708.396: the series with every factor folded into one exponent; the Debye kernel at
 * nu >= EXTENDED_RANGE_ORDER_SPLIT. */
double matern_scalar_large_argument_folded_at(double zh, double nu)
{
    if (nu >= EXTENDED_RANGE_ORDER_SPLIT) return matern_scalar_debye(zh, nu);

    const double w = 1.0/zh, mu = 4.0*nu*nu;
    double t = 1.0, S = 1.0;
    /* terms 0 .. SCALAR_LARGE_ARGUMENT_MAX_TERMS-1, as matern_scalar_large_argument_at */
    for (int k = 1; k < SCALAR_LARGE_ARGUMENT_MAX_TERMS; k++){
        const double tk = (double)(2*k - 1);
        t *= (mu - tk*tk) / (8.0*(double)k) * w;
        /* add, then |t_k| <= eps |S_k| on the new sum, as matern_scalar_large_argument_at */
        S += t;
        if (!(fabs(t) > DBL_EPSILON*fabs(S))) break;
        if (fabs(t) > 1e290) break;                 /* divergent tail guard */
    }
    /* Every factor goes into one exponent, so only the answer must be representable:
     *   E = (nu - 1/2) ln z - z - lnGamma(nu) + (1-nu) ln 2 + 1/2 ln(pi/2),  M = e^E * S.
     * lgamma, not log(tgamma): tgamma overflows past nu ~ 171.6. */
#ifdef WMS_UNFOLDED
#  if WMS_UNFOLDED == 2
    return 0.5 * log(M_PI_2 / zh) - zh + log(S);
#  else
    return sqrt(M_PI_2 / zh) * S * exp(-zh);
#  endif
#else
    const double E = (nu - 0.5) * log(zh)
                   - zh
                   - scalar_log_gamma(nu)
                   + (1.0 - nu) * 0.6931471805599453   /* ln 2        */
                   + 0.22579135264472738;              /* 1/2 ln(pi/2) */
    return exp(E) * S;
#endif
}

#ifdef WMS_UNFOLDED
/* The twin's assembly.  The direct prefactor is written as in
 * matern_scalar_large_argument_at; the exp-log sum keeps the evaluation order of the
 * log assembly in references/common/cpu_wm.h. */
double matern_scalar_log_gamma_unfolded(double nu)
{
    return scalar_log_gamma(nu);
}

double matern_scalar_compose_unfolded(double k, double z, double nu)
{
#  if WMS_UNFOLDED == 2
    return exp((1.0 - nu) * WMS_LN2 - scalar_log_gamma(nu) + nu * log(z) + k);
#  else
    const double pref = 2.0 * pow(z*0.5, nu) / tgamma(nu);
    return pref * k;
#  endif
}

/* The clamp applies only to a finite value, so an overflow stays non-finite. */
double matern_scalar_finish_unfolded(double k, double z, double nu)
{
    const double c = matern_scalar_compose_unfolded(k, z, nu);
    return isfinite(c) ? matern_scalar_clamp_to_unit(c) : c;
}
#endif
