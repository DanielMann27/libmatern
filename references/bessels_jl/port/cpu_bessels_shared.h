/* cpu_bessels_shared.h -- the parts of Bessels.jl's besselk.jl common to the
 * pinned revision and release 0.2.8 (besselk0/k0x/k1/k1x, besselk_large_orders
 * and its scaled form, the forward recurrence), included by both ports.
 * Port of MIT-licensed code, see ../LICENSE; the K0/K1 coefficients (cpu_bessels_data.h)
 * come from Boost.Math, see ../LICENSE_BOOST_1_0.txt.  Julia `muladd` -> C `fma` throughout. */
#ifndef CPU_BESSELS_SHARED_H
#define CPU_BESSELS_SHARED_H

#include <math.h>
#include "cpu_bessels_data.h"

#define BJL_EPS    2.220446049250313e-16   /* eps(Float64)               */
#define BJL_SQPIO2 1.25331413731550025     /* math_constants.jl, Float64 */

/* evalpoly(x, (c0, c1, ...)) -- Julia's ascending Horner over a tuple. */
static inline double bjl_evalpoly(double x, const double *c, int n)
{
    double s = c[n - 1];
    for (int i = n - 2; i >= 0; i--) s = fma(x, s, c[i]);
    return s;
}


/* In place of Julia's `sinpi` (../PROVENANCE.md, Substitutions in the ports):
 * the same kind of exact reduction (x - nearbyint(x) is exact in IEEE double),
 * then libm sin(M_PI * f) on |f| <= 1/2, so the result can differ from Julia's
 * in the last bits.  The reduction keeps sin(M_PI * x) accurate near a large
 * integer. */
static inline double bjl_sinpi(double x)
{
    const double q = nearbyint(x);
    const double f = x - q;                       /* exact, |f| <= 0.5        */
    const double s = sin(M_PI * f);
    const double qh = q * 0.5;
    return (qh == nearbyint(qh)) ? s : -s;        /* q even -> +, q odd -> -  */
}

/* Math.check_convergence(term) = abs(term) <= eps(T) */
static inline int bjl_converged(double t) { return fabs(t) <= BJL_EPS; }

/* Math.isnearint(x) = abs(x - round(x)) < 1e-5 */
static inline int bjl_isnearint(double x) { return fabs(x - nearbyint(x)) < 1e-5; }

/* besselk0 / besselk1 and their scaled forms. */

static inline double bjl_besselk0(double x)
{
    if (x <= 1.0) {
        double a = x * x / 4.0;
        double s = fma(bjl_evalpoly(a, P1_k0, P1_k0_N),
                       1.0 / bjl_evalpoly(a, Q1_k0, Q1_k0_N), Y_k0);
        a = fma(s, a, 1.0);
        return fma(-a, log(x), bjl_evalpoly(x * x, P2_k0, P2_k0_N));
    } else {
        const double s = exp(-x / 2.0);
        const double a = fma(bjl_evalpoly(1.0 / x, P3_k0, P3_k0_N),
                             1.0 / bjl_evalpoly(1.0 / x, Q3_k0, Q3_k0_N), 1.0)
                         * s / sqrt(x);
        return a * s;
    }
}

static inline double bjl_besselk0x(double x)
{
    if (x <= 1.0) {
        double a = x * x / 4.0;
        double s = fma(bjl_evalpoly(a, P1_k0, P1_k0_N),
                       1.0 / bjl_evalpoly(a, Q1_k0, Q1_k0_N), Y_k0);
        a = fma(s, a, 1.0);
        return fma(-a, log(x), bjl_evalpoly(x * x, P2_k0, P2_k0_N)) * exp(x);
    } else {
        return fma(bjl_evalpoly(1.0 / x, P3_k0, P3_k0_N),
                   1.0 / bjl_evalpoly(1.0 / x, Q3_k0, Q3_k0_N), 1.0) / sqrt(x);
    }
}

static inline double bjl_besselk1(double x)
{
    if (x <= 1.0) {
        const double z = x * x;
        double a = z / 4.0;
        double pq = fma(bjl_evalpoly(a, P1_k1, P1_k1_N),
                        1.0 / bjl_evalpoly(a, Q1_k1, Q1_k1_N), Y_k1);
        pq = fma(pq * a, a, (a / 2.0 + 1.0));
        a = pq * x / 2.0;
        pq = fma(bjl_evalpoly(z, P2_k1, P2_k1_N) / bjl_evalpoly(z, Q2_k1, Q2_k1_N),
                 x, 1.0 / x);
        return fma(a, log(x), pq);
    } else {
        const double s = exp(-x / 2.0);
        const double a = fma(bjl_evalpoly(1.0 / x, P3_k1, P3_k1_N),
                             1.0 / bjl_evalpoly(1.0 / x, Q3_k1, Q3_k1_N), Y2_k1)
                         * s / sqrt(x);
        return a * s;
    }
}

static inline double bjl_besselk1x(double x)
{
    if (x <= 1.0) {
        const double z = x * x;
        double a = z / 4.0;
        double pq = fma(bjl_evalpoly(a, P1_k1, P1_k1_N),
                        1.0 / bjl_evalpoly(a, Q1_k1, Q1_k1_N), Y_k1);
        pq = fma(pq * a, a, (a / 2.0 + 1.0));
        a = pq * x / 2.0;
        pq = fma(bjl_evalpoly(z, P2_k1, P2_k1_N) / bjl_evalpoly(z, Q2_k1, Q2_k1_N),
                 x, 1.0 / x);
        return fma(a, log(x), pq) * exp(x);
    } else {
        return fma(bjl_evalpoly(1.0 / x, P3_k1, P3_k1_N),
                   1.0 / bjl_evalpoly(1.0 / x, Q3_k1, Q3_k1_N), Y2_k1) / sqrt(x);
    }
}

/* Debye uniform asymptotic (DLMF 10.41).
 * Uk_poly_Kn(p, v, p2, Float64) = Uk_poly10(p, v, p2)[2]
 *                               = split_evalpoly(-p/v, (1, u1, ..., u10))[2]
 * split_evalpoly walks the tuple in strides of two and returns
 * (even - odd, even + odd); [2] is the second, i.e. sum_k (-p/v)^k u_k, which is
 * sum_k (-1)^k u_k(p) / v^k as DLMF 10.41.4 requires.  The u_k are polynomials
 * in p2 = p^2, taken from U_polynomials.jl by extract_upstream.sh. */
static double bjl_Uk_poly_Kn(double p, double v, double p2)
{
    const double u[11] = {
        1.0,
        bjl_evalpoly(p2, BJL_U1,  BJL_U1_N),  bjl_evalpoly(p2, BJL_U2,  BJL_U2_N),
        bjl_evalpoly(p2, BJL_U3,  BJL_U3_N),  bjl_evalpoly(p2, BJL_U4,  BJL_U4_N),
        bjl_evalpoly(p2, BJL_U5,  BJL_U5_N),  bjl_evalpoly(p2, BJL_U6,  BJL_U6_N),
        bjl_evalpoly(p2, BJL_U7,  BJL_U7_N),  bjl_evalpoly(p2, BJL_U8,  BJL_U8_N),
        bjl_evalpoly(p2, BJL_U9,  BJL_U9_N),  bjl_evalpoly(p2, BJL_U10, BJL_U10_N)
    };
    /* split_evalpoly(x, P), N = 11 (odd) */
    const double x  = -p / v;
    const double xx = x * x;
    double out  = u[10];   /* P[end]   */
    double out2 = u[9];    /* P[end-1] */
    for (int i = 9; i >= 2; i -= 2) {       /* Julia: for i in N-2:-2:2, 1-based */
        out  = fma(xx, out,  u[i - 1]);     /* P[i]   */
        out2 = fma(xx, out2, u[i - 2]);     /* P[i-1] */
    }
    out  = fma(xx, out, u[0]);              /* N odd branch */
    out2 *= x;
    return out2 + out;                      /* the [2] component */
}

static double bjl_besselk_large_orders(double v, double x)
{
    const double z  = x / v;
    const double zs = hypot(1.0, z);
    const double n  = zs + log(z) - log1p(zs);
    const double coef = BJL_SQPIO2 * sqrt(1.0 / v) * exp(-v * n) / sqrt(zs);
    const double p  = 1.0 / zs;
    const double mx = v > x ? v : x, mn = v > x ? x : v;
    const double p2 = v * v / fma(mx, mx, mn * mn);
    return coef * bjl_Uk_poly_Kn(p, v, p2);
}

static double bjl_besselk_large_orders_scaled(double v, double x)
{
    const double z  = x / v;
    const double zs = hypot(1.0, z);
    const double n  = zs + log(z) - log1p(zs);
    const double coef = BJL_SQPIO2 * sqrt(1.0 / v) * exp(-v * n + x) / sqrt(zs);
    const double p  = 1.0 / zs;
    const double mx = v > x ? v : x, mn = v > x ? x : v;
    const double p2 = v * v / fma(mx, mx, mn * mn);
    return coef * bjl_Uk_poly_Kn(p, v, p2);
}

static inline int bjl_debye_cutoff(double nu, double x) { return nu > 25.0 || x > 35.0; }

/* Forward recurrence (recurrence.jl). */

static void bjl_up_recurrence(double x, double jnu, double jnum1,
                              double nu_start, double nu_end,
                              double *o1, double *o2)
{
    const double x2 = 2.0 / x;
    while (nu_start < nu_end + 0.5) {
        const double prev = jnu;
        jnu   = fma(nu_start * x2, jnu, jnum1);
        jnum1 = prev;
        nu_start += 1.0;
    }
    *o1 = jnum1;
    *o2 = jnu;
}


#endif /* CPU_BESSELS_SHARED_H */
