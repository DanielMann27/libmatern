/* cpu_bessels_jl.c -- C port of Bessels.jl besselk/besselkx (MIT, see
 * ../LICENSE) at commit 009b615347d91d1dd69962641647521901362af3,
 * src/BesselFunctions/besselk.jl.
 *
 * The double-precision path is reproduced branch for branch and expression
 * for expression.  Also from upstream: constants.jl (K0/K1 coefficients,
 * cpu_bessels_data.h), U_polynomials.jl (Debye u_1..u_10, split_evalpoly,
 * Uk_poly10), Math/Math.jl (levin_transform, isnearint, check_convergence),
 * recurrence.jl (besselk_up_recurrence), math_constants.jl (SQPIO2).
 *
 * Substitutions (../PROVENANCE.md, Substitutions in the ports): `muladd` ->
 * `fma`; `gamma` (Gamma.jl) -> libm `tgamma`, in besselk_power_series only;
 * `sinpi` -> bjl_sinpi (exact reduction, then libm sin); exp, log, ^, cosh,
 * hypot, log1p -> the C library's.  The results therefore depend on the libm. */

#include <math.h>
#include <float.h>
#include "cpu_wm.h"
#include "cpu_bessels_shared.h"


/* Large-argument expansion (DLMF 10.40.2). */

static double bjl_besselkx_large_args(double v, double x)
{
    const int MaxIter = 75;
    double t = 1.0, s = 1.0;
    const double invx = 1.0 / (8.0 * x);
    for (int i = 1; i <= MaxIter; i++) {
        t *= invx * ((4.0 * v * v - (2.0 * i - 1.0) * (2.0 * i - 1.0)) / i);
        s += t;
        if (bjl_converged(t)) break;
    }
    return s * sqrt(M_PI / (2.0 * x));
}

static double bjl_besselk_large_args(double v, double x)
{
    const double e = exp(-x / 2.0);
    return (e * bjl_besselkx_large_args(v, x)) * e;
}

static inline int bjl_large_args_cutoff(double v, double x) { return x > v * v / 36.0 + 18.0; }

/* Levin sequence transform, Val(16). */

#define BJL_LEVIN_N 16

/* levin_scale(B, n, k) = -(B+n+k)(B+n+k-1) / ((B+n+2k)(B+n+2k-1)), B = 1 */
static inline double bjl_levin_scale(double n, double k)
{
    return -(1.0 + n + k) * (1.0 + n + k - 1.0) /
            ((1.0 + n + 2.0 * k) * (1.0 + n + 2.0 * k - 1.0));
}

/* Math.levin_transform: a_i = (s_i/w_i, 1/w_i) then a triangular sweep of FMAs.
 * Julia holds the pair in a Vec{2} and fmadds both lanes at once; the two scalar
 * fmas below are the same two operations, in the same order. */
static double bjl_levin_transform(const double *s, const double *w, int N)
{
    double an[BJL_LEVIN_N], ad[BJL_LEVIN_N];
    for (int i = 0; i < N; i++) {
        if (w[i] == 0.0) return s[i];        /* Julia: iszero(w[i]) && return s[i] */
        an[i] = s[i] / w[i];
        ad[i] = 1.0 / w[i];
    }
    const int len = N - 1;
    for (int k = 1; k <= len; k++)                    /* @nexprs len k        */
        for (int i = 1; i <= len - k; i++) {          /* @nexprs (len-k) i    */
            const double sc = bjl_levin_scale((double)i, (double)(k - 1));
            an[i - 1] = fma(an[i - 1], sc, an[i]);
            ad[i - 1] = fma(ad[i - 1], sc, ad[i]);
        }
    return an[0] / ad[0];
}

static double bjl_besselkx_levin(double v, double x)
{
    double seq[BJL_LEVIN_N], wt[BJL_LEVIN_N];
    double s = 0.0, t = 1.0;
    for (int i = 1; i <= BJL_LEVIN_N; i++) {
        s += t;
        t *= (4.0 * v * v - (2.0 * i - 1.0) * (2.0 * i - 1.0)) / (8.0 * x * i);
        seq[i - 1] = s;
        wt[i - 1]  = t;
    }
    return bjl_levin_transform(seq, wt, BJL_LEVIN_N) * sqrt(M_PI / (2.0 * x));
}

static inline int    bjl_levin_cutoff(double v, double x) { return x > v * v * v * v / 2401.0 + 1.5; }
static inline double bjl_levin_min(void) { return 1.5; }

/* Power series (non-integer nu). */

static double bjl_besselk_power_series(double v, double x)
{
    const int MaxIter = 50;
    const double gam  = tgamma(v);                        /* Gamma.jl -> libm  */
    const double ngam = M_PI / (bjl_sinpi(-fabs(v)) * gam * v);       /* sinpi(-|v|) */
    double s1 = 0.0, s2 = 0.0, t1 = 1.0, t2 = 1.0;
    for (int k = 1; k <= MaxIter; k++) {
        s1 += t1;
        s2 += t2;
        t1 *= x * x / (4.0 * k * (k - v));
        t2 *= x * x / (4.0 * k * (k + v));
        if (bjl_converged(t1)) break;
    }
    const double xpv = pow(x / 2.0, v);
    const double s   = gam * s1 + xpv * xpv * ngam * s2;
    return s / (2.0 * xpv);
}

/* Temme series (near-integer nu), |v| < 1e-5. */

static const double bjl_pk_poly[4] = { 1.0, -0.5772156649015329,
                                       0.9890559953279725, -0.23263776388631713 };
static const double bjl_sp_poly[4] = { 1.0, 1.6449340668482264,
                                       1.8940656589944918, 1.9711021825948702 };
static const double bjl_g1_poly[3] = { -0.5772156649015329, 0.04200263503409518,
                                       0.042197734555544306 };
static const double bjl_g2_poly[3] = { 1.0, -0.6558780715202539, 0.16653861138229145 };
static const double bjl_sh_poly[5] = { 1.0, 0.16666666666666666, 0.008333333333333333,
                                       0.0001984126984126984, 2.7557319223985893e-6 };

static double bjl_f0_local_expansion_v0(double v, double x)
{
    const double l2dx = log(2.0) - log(x);
    const double mu = v * l2dx;
    const double vv = v * v;
    const double sp = bjl_evalpoly(vv, bjl_sp_poly, 4);
    const double g1 = bjl_evalpoly(vv, bjl_g1_poly, 3);
    const double g2 = bjl_evalpoly(vv, bjl_g2_poly, 3);
    const double sh = bjl_evalpoly(mu * mu, bjl_sh_poly, 5);
    return sp * (g1 * cosh(mu) + g2 * sh * l2dx);
}

static void bjl_besselk_temme_series(double v, double x, double *kv, double *kvp1)
{
    const int Max_Iter = 500;
    const double z = x / 2.0, zz = z * z;
    double fk = bjl_f0_local_expansion_v0(v, x);
    const double zv = pow(z, v);
    double pk = bjl_evalpoly( v, bjl_pk_poly, 4) / (2.0 * zv);
    double qk = zv * bjl_evalpoly(-v, bjl_pk_poly, 4) / 2.0;
    double ck = 1.0, out_v = 0.0, out_vp1 = 0.0;
    for (int k = 1; k <= Max_Iter; k++) {
        const double term_v   = ck * fk;
        const double term_vp1 = ck * (pk - (k - 1) * fk);
        out_v   += term_v;
        out_vp1 += term_vp1;
        if (bjl_converged(term_v) && bjl_converged(term_vp1)) break;
        fk = (k * fk + pk + qk) / ((double)k * k - v * v);
        pk /= k - v;
        qk /= k + v;
        ck *= zz / k;
    }
    *kv   = out_v;
    *kvp1 = out_vp1 / z;
}

/* Dispatch: _besselk(v::Float64, x::Float64). */

static double bjl_besselk(double v, double x)
{
    v = fabs(v);
    if (x < 0.0) return NAN;                 /* Julia throws DomainError      */

    if (bjl_large_args_cutoff(v, x))  return bjl_besselk_large_args(v, x);
    if (bjl_levin_cutoff(v, x))       return bjl_besselkx_levin(v, x) * exp(-x);
    if (bjl_debye_cutoff(v, x))       return bjl_besselk_large_orders(v, x);

    double ip;
    double v_floor = modf(v, &ip);           /* Julia: v_floor, _ = modf(v)   */
    if (x > bjl_levin_min()) {
        const double kv   = bjl_besselkx_levin(v_floor, x);
        const double kvp1 = bjl_besselkx_levin(v_floor + 1.0, x);
        double o1, o2;
        bjl_up_recurrence(x, kvp1, kv, v_floor + 1.0, v, &o1, &o2);
        return o1 * exp(-x);
    }
    if (bjl_isnearint(v)) {
        if (v_floor > 0.5) v_floor -= 1.0;
        double kv, kvp1, o1, o2;
        bjl_besselk_temme_series(v_floor, x, &kv, &kvp1);
        bjl_up_recurrence(x, kvp1, kv, v_floor + 1.0, v, &o1, &o2);
        return o1;
    }
    return bjl_besselk_power_series(v, x);
}

/* Dispatch: _besselkx(v::Float64, x::Float64). */

static double bjl_besselkx(double v, double x)
{
    v = fabs(v);
    if (x < 0.0) return NAN;

    if (bjl_large_args_cutoff(v, x))  return bjl_besselkx_large_args(v, x);
    if (bjl_levin_cutoff(v, x))       return bjl_besselkx_levin(v, x);
    if (bjl_debye_cutoff(v, x))       return bjl_besselk_large_orders_scaled(v, x);

    double ip;
    double v_floor = modf(v, &ip);
    if (x > bjl_levin_min()) {
        const double kv   = bjl_besselkx_levin(v_floor, x);
        const double kvp1 = bjl_besselkx_levin(v_floor + 1.0, x);
        double o1, o2;
        bjl_up_recurrence(x, kvp1, kv, v_floor + 1.0, v, &o1, &o2);
        return o1;
    }
    if (bjl_isnearint(v)) {
        if (v_floor > 0.5) v_floor -= 1.0;
        double kv, kvp1, o1, o2;
        bjl_besselk_temme_series(v_floor, x, &kv, &kvp1);
        bjl_up_recurrence(x, kvp1, kv, v_floor + 1.0, v, &o1, &o2);
        return o1 * exp(x);
    }
    return bjl_besselk_power_series(v, x) * exp(x);
}

static double sup_kv (double nu, double z) { return bjl_besselk (nu, z); }
static double sup_kxv(double nu, double z) { return bjl_besselkx(nu, z); }

const wmc_supplier wmc_bessels_jl = {
    "Bessels.jl besselk, commit 009b615 (MIT)",
    WMC_LOG_SCALED,        /* besselkx exists, so log K = log(Kx) - z         */
    sup_kv, sup_kxv, NULL
};
