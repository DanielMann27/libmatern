/* cpu_wm.h -- interface of the external K_nu suppliers and the covariance assemblies shared
 * by all of them (direct, scaled, exp-log), M_nu(z) = 2^(1-nu)/Gamma(nu) z^nu K_nu(z) with
 * z = sqrt(2 nu) h / s (see the accompanying paper, reference implementations). */
#ifndef CPU_WM_H
#define CPU_WM_H

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* how a supplier can deliver ln K_nu */
#define WMC_LOG_NONE   0   /* only K_nu -> ln K = log(K)                   */
#define WMC_LOG_SCALED 1   /* also K_nu e^{z} -> ln K = log(Kx) - z        */
#define WMC_LOG_NATIVE 2   /* ln K_nu directly, no intermediate to overflow */

typedef struct {
    const char *origin;    /* upstream file / library the port came from    */
    int   logmode;         /* WMC_LOG_*                                     */
    double (*kv)   (double nu, double z);   /* K_nu(z)                      */
    double (*kxv)  (double nu, double z);   /* K_nu(z) e^{z}, or NULL       */
    double (*logkv)(double nu, double z);   /* ln K_nu(z), or NULL          */
} wmc_supplier;

#define WMC_LN2 0.6931471805599453094172321214581766

/* z = sqrt(2 nu) h / s, formed identically for every supplier. */
static inline double wmc_z(double h, double nu, double scale)
{
    return sqrt(2.0 * nu) * h / scale;
}

/* ln K_nu(z) by the best route this supplier offers. */
static inline double wmc_logk(const wmc_supplier *s, double nu, double z)
{
    switch (s->logmode) {
        case WMC_LOG_NATIVE: return s->logkv(nu, z);
        case WMC_LOG_SCALED: return log(s->kxv(nu, z)) - z;
        default:             return log(s->kv(nu, z));
    }
}

/* The assemblies return exactly 1.0 at h == 0 (z = 0 is outside the domain of every
 * K_nu routine); no other input is special-cased.  Direct: the formula as written. */
static inline double wmc_direct(const wmc_supplier *s, double h, double nu, double scale)
{
    if (h == 0.0) return 1.0;
    const double z = wmc_z(h, nu, scale);
    return pow(2.0, 1.0 - nu) / tgamma(nu) * pow(z, nu) * s->kv(nu, z);
}

/* Nonzero when the supplier publishes K_nu(z) e^{z} (Bessels.jl, GSL); only those
 * get a scaled row, and wmc_scaled returns NaN for the others. */
static inline int wmc_has_scaled(const wmc_supplier *s) { return s->kxv != NULL; }

static inline double wmc_scaled(const wmc_supplier *s, double h, double nu, double scale)
{
    if (h == 0.0) return 1.0;
    const double z = wmc_z(h, nu, scale);
    /* Prefactor and e^{-z} in one exponent A, applied as two halves,
     * (exp(A/2) * Kx) * exp(A/2), in the multiplication order of Bessels.jl's
     * besselk_large_args.  For nu >> z, Kx is large while exp(A) underflows, so
     * exp(A) * Kx would give 0 for a representable M.  The halves cannot
     * overflow: M <= 1 bounds A + ln(Kx) <= 0, and ln(Kx) is O(1) wherever A is
     * positive. */
    if (!wmc_has_scaled(s)) return NAN;
    const double e = (1.0 - nu) * WMC_LN2 - lgamma(nu) + nu * log(z) - z;
    const double half = exp(0.5 * e);
    return (half * s->kxv(nu, z)) * half;
}

/* ln M(h), the exponent of the exp-log assembly below.  For a supplier without a
 * native log it inherits the range of that supplier's K. */
static inline double wmc_logc(const wmc_supplier *s, double h, double nu, double scale)
{
    if (h == 0.0) return 0.0;
    const double z = wmc_z(h, nu, scale);
    return (1.0 - nu) * WMC_LN2 - lgamma(nu) + nu * log(z) + wmc_logk(s, nu, z);
}

/* The exp-log assembly: exp(wmc_logc).  At h == 0 it returns exp(0) = 1. */
static inline double wmc_logtrick(const wmc_supplier *s, double h, double nu, double scale)
{
    return exp(wmc_logc(s, h, nu, scale));
}

/* The same assemblies taking z directly, for the (z, nu) datasets: re-deriving z
 * from h = z/sqrt(2 nu) does not in general return the stored z, and since M
 * behaves like e^-z an absolute error in z is a relative error in M. */

static inline double wmc_direct_z(const wmc_supplier *s, double z, double nu)
{
    if (z == 0.0) return 1.0;
    return pow(2.0, 1.0 - nu) / tgamma(nu) * pow(z, nu) * s->kv(nu, z);
}

static inline double wmc_scaled_z(const wmc_supplier *s, double z, double nu)
{
    if (z == 0.0) return 1.0;
    if (!wmc_has_scaled(s)) return NAN;
    const double e = (1.0 - nu) * WMC_LN2 - lgamma(nu) + nu * log(z) - z;
    const double half = exp(0.5 * e);
    return (half * s->kxv(nu, z)) * half;     /* halved exponent -- see wmc_scaled */
}

static inline double wmc_logc_z(const wmc_supplier *s, double z, double nu)
{
    if (z == 0.0) return 0.0;
    return (1.0 - nu) * WMC_LN2 - lgamma(nu) + nu * log(z) + wmc_logk(s, nu, z);
}

static inline double wmc_logtrick_z(const wmc_supplier *s, double z, double nu)
{
    return exp(wmc_logc_z(s, z, nu));
}

static inline void wmc_direct_array_z(const wmc_supplier *s, const double *z,
                                      const double *nu, double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = wmc_direct_z(s, z[i], nu[i]);
}

static inline void wmc_logtrick_array_z(const wmc_supplier *s, const double *z,
                                        const double *nu, double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = wmc_logtrick_z(s, z[i], nu[i]);
}

static inline void wmc_scaled_array_z(const wmc_supplier *s, const double *z,
                                      const double *nu, double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = wmc_scaled_z(s, z[i], nu[i]);
}

/* Array drivers: one call per array, as for the library's own entry points. */
static inline void wmc_direct_array(const wmc_supplier *s, const double *h,
                                    const double *nu, const double *scale,
                                    double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = wmc_direct(s, h[i], nu[i], scale[i]);
}

static inline void wmc_logtrick_array(const wmc_supplier *s, const double *h,
                                      const double *nu, const double *scale,
                                      double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = wmc_logtrick(s, h[i], nu[i], scale[i]);
}

static inline void wmc_scaled_array(const wmc_supplier *s, const double *h,
                                    const double *nu, const double *scale,
                                    double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = wmc_scaled(s, h[i], nu[i], scale[i]);
}

extern const wmc_supplier wmc_bessels_jl;      /* Bessels.jl besselk.jl, pinned commit */
extern const wmc_supplier wmc_bessels_old;     /* Bessels.jl, release 0.2.8   */
/* The KAUST symbols are defined by kaust_logbesselk/kaust_port.c, built only when
 * `make kaust DOWNLOAD=yes` has generated its port (docs/benchmark.md, External
 * suppliers).  The declarations are unconditional; consumers guard their uses with
 * WMC_WITH_KAUST. */
extern const wmc_supplier wmc_kaust;           /* logbesselk.cu, 128 panels    */
extern const wmc_supplier wmc_kaust40;         /*   ... setIntervals(40)       */
extern const wmc_supplier wmc_kaust4;          /* kaust-adaptive, described in
                                               * kaust_logbesselk/PROVENANCE.md */
/* Defined in cpp_stdlib/cpu_cxx.cpp when a C++ compiler is present
 * (WMC_WITH_CXX), the Boost one only when a Boost is selected (WMC_WITH_BOOST). */
extern const wmc_supplier wmc_cxx_std;         /* libstdc++ std::cyl_bessel_k */
extern const wmc_supplier wmc_cxx_boost;       /* boost::math::cyl_bessel_k   */
#ifdef WMC_WITH_GSL
extern const wmc_supplier wmc_gsl;             /* GSL -- all three forms native */
#endif

/* Assembly numbering shared by the harnesses, and a dispatcher over the array
 * drivers above. */
enum { WMC_ASM_DIRECT, WMC_ASM_SCALED, WMC_ASM_LOGTRICK };

static inline const char *wmc_asm_name(int a)
{
    return a == WMC_ASM_DIRECT ? "direct"
         : a == WMC_ASM_SCALED ? "scaled" : "logtrick";
}

/* One call per array.  `has_length_scale` selects the argument form: nonzero
 * reads (h, nu, scale), zero reads (z, nu).  Suppliers take no workspace. */
static inline void wmc_eval_array(const wmc_supplier *s, int assembly,
                                  int has_length_scale,
                                  const double *h, const double *nu,
                                  const double *scale, const double *z,
                                  double *out, size_t n)
{
    if (has_length_scale) {
        switch (assembly) {
            case WMC_ASM_DIRECT: wmc_direct_array  (s, h, nu, scale, out, n); return;
            case WMC_ASM_SCALED: wmc_scaled_array  (s, h, nu, scale, out, n); return;
            default:             wmc_logtrick_array(s, h, nu, scale, out, n); return;
        }
    }
    switch (assembly) {
        case WMC_ASM_DIRECT: wmc_direct_array_z  (s, z, nu, out, n); return;
        case WMC_ASM_SCALED: wmc_scaled_array_z  (s, z, nu, out, n); return;
        default:             wmc_logtrick_array_z(s, z, nu, out, n); return;
    }
}

#ifdef __cplusplus
}
#endif
#endif /* CPU_WM_H */
