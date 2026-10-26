/* cpu_bessels_old.c -- C port of Bessels.jl besselk.jl, release 0.2.8 (tag
 * v0.2.8, commit 1475e9e, src/besselk.jl; fetched as upstream/besselk_old.jl)
 * (MIT, see ../LICENSE).
 *
 * Three definitions that besselk.jl uses and release 0.2.8 defines in other
 * files are taken from the pinned commit and marked below: the one-line
 * predicates `besseli_large_argument_cutoff` (besseli.jl) and
 * `sphericalbesselk_cutoff`, and the function `sphericalbesselk_int`; the
 * last two are in modifiedsphericalbessel.jl, which is the same file at both
 * commits.  Only the first differs (release 0.2.8: x > 23 and
 * x > nu^2/1.8 + 23), so the large-argument border (x > nu^2/2 + 19) is the
 * pinned commit's (../PROVENANCE.md).
 *
 * Substitutions as in cpu_bessels_jl.c (../PROVENANCE.md): fma for muladd,
 * libm tgamma for gamma (here release 0.2.8's own, src/gamma.jl), bjl_sinpi
 * for sinpi, and the C library's elementary functions for Julia's. */

#include <math.h>
#include "cpu_supplier.h"
#include "cpu_bessels_shared.h"   /* besselk0/1(x), Debye, up_recurrence, sinpi */

/* Modified spherical Bessel k_n, the half-integer closed form:
 * modifiedsphericalbessel.jl:68 (pinned commit; the same file in release
 * 0.2.8).  K_{n+1/2}(x) = k_n(x) sqrt(pi/2) sqrt(x) with this normalisation;
 * the caller applies the factor. */
static double bol_sphericalbesselk_int(int v, double x)
{
    const double xinv = 1.0 / x;
    double b0 = exp(-x) * xinv;
    double b1 = b0 * (x + 1.0) * xinv;
    if (v == 0) return b0;
    double _v = 1.0;
    const double invx = 1.0 / x;
    while (_v < (double)v) {
        _v += 1.0;
        const double t = b1;
        b1 = b0 + (2.0 * _v - 1.0) * b1 * invx;
        b0 = t;
    }
    return b1;
}

/* modifiedsphericalbessel.jl:66, pinned commit */
static inline int bol_spherical_cutoff(double nu) { return nu < 41.5; }

/* besseli.jl:609, pinned commit */
static inline int bol_large_argument_cutoff(double nu, double x)
{
    return x > nu * nu / 2.0 + 19.0;
}

/* Large-argument asymptotic (besselk_old.jl:546). */

static double bol_besselk_large_argument_core(double v, double x)
{
    const int MaxIter = 5000;
    const double fv2 = 4.0 * v * v;
    double term = 1.0, res = 1.0;
    const double s = 1.0;
    for (int i = 1; i <= MaxIter; i++) {
        const double offset = fma(2.0, (double)i, -1.0);
        term *= fma(offset, -offset, fv2) / (8.0 * x * i);
        res = fma(term, s, res);
        if (fabs(term) <= BJL_EPS) break;
    }
    return res;
}

static double bol_besselk_large_argument(double v, double x)
{
    const double a = exp(-x / 2.0);
    const double coef = a * sqrt(M_PI / (2.0 * x));
    return bol_besselk_large_argument_core(v, x) * coef * a;
}

static double bol_besselk_large_argument_scaled(double v, double x)
{
    return bol_besselk_large_argument_core(v, x) * sqrt(M_PI / (2.0 * x));
}

/* Power series of release 0.2.8 (besselk_old.jl:501): gamma(1-v) and
 * gamma(1+v) are carried through the loop. */
static double bol_besselk_power_series(double v, double x)
{
    const int MaxIter = 1000;
    const double z = x / 2.0, zz = z * z;
    const double logz = log(z);
    const double xd2_v  = exp(v * logz);
    const double xd2_nv = 1.0 / xd2_v;

    const double gam_v  = tgamma(v);                      /* src/gamma.jl -> libm */
    const double gam_nv = M_PI / (bjl_sinpi(-fabs(v)) * gam_v * v);
    double gam_1mv  = -gam_nv * v;
    double gam_1mnv =  gam_v * v;

    const double _t1 = gam_v  * xd2_nv * gam_1mv;
    const double _t2 = gam_nv * xd2_v  * gam_1mnv;
    double xd2_pow = 1.0, fact_k = 1.0, out = 0.0;
    for (int k = 0; k <= MaxIter; k++) {
        const double t1 = xd2_pow * 0.5;
        double tmp = fma(_t1, gam_1mnv, _t2 * gam_1mv);
        tmp *= 1.0 / (gam_1mv * gam_1mnv * fact_k);
        const double term = t1 * tmp;
        out += term;
        if (fabs(term / out) < BJL_EPS) break;
        const double g1 = gam_1mnv * (1.0 + v + k);
        const double g2 = gam_1mv  * (1.0 - v + k);
        gam_1mnv = g1;
        gam_1mv  = g2;
        xd2_pow *= zz;
        fact_k  *= k + 1.0;
    }
    return out;
}

static inline int bol_power_series_cutoff(double nu, double x)
{
    return x < 2.0 || nu > 1.6 * x - 1.0;
}

/* Continued fraction (besselk_old.jl:427). */

static void bol_besseli_power_series_inu_inum1(double v, double x,
                                               double *inu, double *inum1)
{
    const int MaxIter = 3000;
    double out = 0.0, out2 = 0.0;
    const double x2 = x / 2.0;
    const double xs = pow(x2, v);
    const double gmx = xs / tgamma(v);
    double a = gmx / v;
    double b = gmx / x2;
    const double t2 = x2 * x2;
    for (int i = 0; i <= MaxIter; i++) {
        out  += a;
        out2 += b;
        if (fabs(a) < BJL_EPS * fabs(out)) break;
        a *= 1.0 / ((v + i + 1.0) * (i + 1.0)) * t2;
        b *= 1.0 / ((v + i) * (i + 1.0)) * t2;
    }
    *inu = out;
    *inum1 = out2;
}

/* K_{v+1}/K_v by modified Lentz */
static double bol_besselk_ratio_knu_knup1(double v, double x)
{
    const int MaxIter = 1000;
    double hn = 1e-50, Dn = 0.0, Cn = 1e-50;
    double jf = 1.0;
    const double vv = v * v;
    for (int i = 0; i < MaxIter; i++) {
        const double an = vv - ((2.0 * jf - 1.0) * (2.0 * jf - 1.0)) * 0.25;
        const double bn = 2.0 * (x + jf);
        Cn = an / Cn + bn;
        Dn = 1.0 / fma(an, Dn, bn);
        const double del = Dn * Cn;
        hn *= del;
        if (fabs(del - 1.0) < BJL_EPS) break;
        jf += 1.0;
    }
    const double xinv = 1.0 / x;
    return xinv * (v + x + 0.5) + xinv * hn;
}

static double bol_besselk_continued_fraction(double nu, double x)
{
    double inu, inum1;
    bol_besseli_power_series_inu_inum1(nu, x, &inu, &inum1);
    const double H_knu = bol_besselk_ratio_knu_knup1(nu - 1.0, x);
    return 1.0 / (x * (inum1 + inu / H_knu));
}

/* Dispatch: besselk_positive_args (besselk_old.jl:317). */

static double bol_besselk(double nu, double x)
{
    nu = fabs(nu);
    x  = fabs(x);
    if (x == 0.0)     return INFINITY;
    if (isinf(x))     return 0.0;
    if (nu == 0.0)    return bjl_besselk0(x);

    if (nu - 0.5 == nearbyint(nu - 0.5) && bol_spherical_cutoff(nu))
        return bol_sphericalbesselk_int((int)(nu - 0.5), x) * BJL_SQPIO2 * sqrt(x);

    if (bol_large_argument_cutoff(nu, x)) return bol_besselk_large_argument(nu, x);
    if (bjl_debye_cutoff(nu, x))          return bjl_besselk_large_orders(nu, x);

    if (nu == nearbyint(nu)) {
        double o1, o2;
        bjl_up_recurrence(x, bjl_besselk1(x), bjl_besselk0(x), 1.0, nu, &o1, &o2);
        return o1;
    }
    if (bol_power_series_cutoff(nu, x))   return bol_besselk_power_series(nu, x);
    return bol_besselk_continued_fraction(nu, x);
}

/* Dispatch: _besselkx (besselk_old.jl:351). */

static double bol_besselkx(double nu, double x)
{
    if (nu == 0.0) return bjl_besselk0x(x);
    if (bol_large_argument_cutoff(nu, x)) return bol_besselk_large_argument_scaled(nu, x);
    if (bjl_debye_cutoff(nu, x))          return bjl_besselk_large_orders_scaled(nu, x);
    if (nu == nearbyint(nu)) {
        double o1, o2;
        bjl_up_recurrence(x, bjl_besselk1x(x), bjl_besselk0x(x), 1.0, nu, &o1, &o2);
        return o1;
    }
    if (bol_power_series_cutoff(nu, x))   return bol_besselk_power_series(nu, x) * exp(x);
    return bol_besselk_continued_fraction(nu, x) * exp(x);
}

/* Upstream _besselkx has no half-integer branch, so besselk and besselkx take
 * different routes at nu = 1/2, 3/2, ...; reproduced as in upstream. */

static double sup_kv (double nu, double z) { return bol_besselk (nu, z); }
static double sup_kxv(double nu, double z) { return bol_besselkx(nu, z); }

const cpu_supplier cpu_supplier_bessels_old = {
    "Bessels.jl besselk, earlier algorithm (MIT)",
    CPU_SUPPLIER_LOG_SCALED,
    sup_kv, sup_kxv, NULL
};
