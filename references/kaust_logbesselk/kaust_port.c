/* kaust_port.c -- KAUST CuBesselK rows kaust-128, kaust-40 and kaust-adaptive (Geng et al.,
 * arXiv:2502.00356v2), on the port kaust_derived.h that `make kaust DOWNLOAD=yes` derives from
 * the user's copy; no upstream code is in this file.  One source for host C and CUDA rows. */

#include <math.h>
#include <float.h>
#include "cpu_supplier.h"

/* Under nvcc every internal function and table is a device object; the tables
 * need it because ka_cheb holds a pointer to one. */
#if defined(__CUDACC__)
  #define KA_FN   static __device__
  #define KA_DATA static __device__ const
#else
  #define KA_FN   static
  #define KA_DATA static const
#endif

/* A Chebyshev series: coefficients, interval [a, b], order. */
typedef struct { const double *c; double a, b; int order; } ka_cheb;

/* log(cosh(y)) without forming cosh(y), which overflows for |y| > 710; the
 * quadrature reaches nu*t = 9 nu. */
KA_FN double ka_logcosh(double y)
{
    y = fabs(y);
    if (y > 20.0) return y - M_LN2 + log1p(exp(-2.0 * y));
    return log(cosh(y));
}

/* The integrand log cosh(v t) - x cosh t, through ka_logcosh. */
KA_FN double ka_f_stable(double t, double v, double x)
{
    return ka_logcosh(v * t) - x * cosh(t);
}

#include "kaust_derived.h"

/* The integration range of Geng et al.'s Eq. (7), in place of the released fixed [0, 9]:
 * [t0, t1] = { t : g(t) >= log(eps) + g(t_max) },  g(t) = log cosh(nu t) - x cosh t,
 * with t_max found by doubling until g' < 0 and then bisection. */
KA_FN double ka_gp(double t, double v, double x)
{ return v*tanh(v*t) - x*sinh(t); }

KA_FN double ka_find_tmax(double v, double x)
{
    if (v*v <= x) return 0.0;                  /* g decreasing from the origin */
    double hi = 1.0;
    for (int m = 0; m < 60 && ka_gp(hi, v, x) > 0.0; m++) hi *= 2.0;
    double lo = 0.5*hi;
    for (int i = 0; i < 100; i++){
        const double md = 0.5*(lo + hi);
        if (ka_gp(md, v, x) > 0.0) lo = md; else hi = md;
    }
    return 0.5*(lo + hi);
}

/* bisect g(t) = target on [a,b] where g is monotone and the level is bracketed */
KA_FN double ka_find_level(double a, double b, double v, double x, double target)
{
    double ga = ka_f_stable(a, v, x) - target;
    for (int i = 0; i < 100; i++){
        const double md = 0.5*(a + b);
        const double gm = ka_f_stable(md, v, x) - target;
        if ((ga < 0.0) == (gm < 0.0)) { a = md; ga = gm; } else b = md;
    }
    return 0.5*(a + b);
}

KA_FN double ka_trapezoid_log_adaptive(double nu, double x, int iv)
{
    const double tmax = ka_find_tmax(nu, x);
    const double targ = log(KA_DBL_EPSILON) + ka_f_stable(tmax, nu, x);
    double t0 = 0.0;
    if (nu*nu > x && ka_f_stable(0.0, nu, x) < targ)
        t0 = ka_find_level(0.0, tmax, nu, x, targ);
    double hi = (tmax > 0.0 ? 2.0*tmax : 1.0) + 1.0;
    for (int i = 0; i < 60 && ka_f_stable(hi, nu, x) > targ; i++) hi *= 2.0;
    const double t1 = ka_find_level(tmax, hi, nu, x, targ);
    return ka_trapezoid_log_range(nu, x, iv, t0, t1);
}

/* Configurations:
 *   fix    0: as released (e10 dropped, cosh(nu t) formed before the log)
 *          1: e10 applied, ka_f_stable
 *   iv     the trapezoid panel count (upstream default 128, setIntervals)
 *   adapt  1: the Eq. (7) range instead of [0, 9] */
#define KA_BINS_DEFAULT 128
#define KA_BINS_COARSE   40

/* K_nu(x); with fix = 0 the return expressions of the released code.  ka_small_core
 * returns K_nu e^{x} and a decimal exponent e10; e10 is applied only when fix = 1. */
KA_FN double ka_kv(double nu, double x, int fix, int iv, int adapt)
{
    if (x <= 0) return INFINITY;
    if (nu < 0) nu = -nu;
    if (x <= 0.1) {
        double K; int e10;
        ka_small_core(nu, x, &K, &e10);
        const double v = K * exp(-x);
        return fix ? v * pow(10.0, (double)e10) : v;
    }
    return exp(adapt ? ka_trapezoid_log_adaptive(nu, x, iv)
                     : fix ? ka_trapezoid_log_stable(nu, x, iv)
                           : ka_trapezoid_log(nu, x, iv));
}

/* ln K_nu(x): for x > 0.1 the quadrature's logarithm before its final exp. */
KA_FN double ka_logkv(double nu, double x, int fix, int iv, int adapt)
{
    if (x <= 0) return INFINITY;
    if (nu < 0) nu = -nu;
    if (x <= 0.1) {
        double K; int e10;
        ka_small_core(nu, x, &K, &e10);
        return log(K) - x + (fix ? e10 * M_LN10 : 0.0);
    }
    return adapt ? ka_trapezoid_log_adaptive(nu, x, iv)
                 : fix ? ka_trapezoid_log_stable(nu, x, iv)
                       : ka_trapezoid_log(nu, x, iv);
}

#if defined(__CUDACC__)
/* The device entry points, called across translation units with -rdc=true by
 * benchmark/suppliers_gpu.cu. */
__device__ double gpu_supplier_kaust_kv   (double nu, double x, int fix, int iv, int adapt)
{ return ka_kv(nu, x, fix, iv, adapt); }
__device__ double gpu_supplier_kaust_logkv(double nu, double x, int fix, int iv, int adapt)
{ return ka_logkv(nu, x, fix, iv, adapt); }
#else
#define KA_SUPPLIER_FNS(SUF, FIX, IV, AD)                                             \
    static double sup_kv##SUF   (double nu,double z){ return ka_kv   (nu,z,FIX,IV,AD); } \
    static double sup_logkv##SUF(double nu,double z){ return ka_logkv(nu,z,FIX,IV,AD); }

KA_SUPPLIER_FNS(_128,   0, KA_BINS_DEFAULT, 0)
KA_SUPPLIER_FNS(_40,    0, KA_BINS_COARSE,  0)
KA_SUPPLIER_FNS(_128k4, 1, KA_BINS_DEFAULT, 1)

const cpu_supplier cpu_supplier_kaust_128 = {
    "CuBesselK logbesselk.cu @94ec9ca, 128 panels, as released",
    CPU_SUPPLIER_LOG_NATIVE,
    sup_kv_128, NULL, sup_logkv_128
};

const cpu_supplier cpu_supplier_kaust_40 = {
    "CuBesselK logbesselk.cu @94ec9ca, setIntervals(40), as released",
    CPU_SUPPLIER_LOG_NATIVE,
    sup_kv_40, NULL, sup_logkv_40
};

const cpu_supplier cpu_supplier_kaust_adaptive = {
    "CuBesselK logbesselk.cu @94ec9ca, 128 panels + e10 + stable log cosh + Eq.(7) range",
    CPU_SUPPLIER_LOG_NATIVE,
    sup_kv_128k4, NULL, sup_logkv_128k4
};
#endif
