/* region_router.c -- scalar router and the entries matern_evaluate_scalar() and _z():
 * constant half-integer runs go to the closed form, every other point is routed on its
 * own (see the accompanying paper, domain decomposition). */
#include <float.h>
#include <math.h>
#include <stddef.h>

#include "scalar_internal.h"

/* The Makefile builds this backend once more per instruction set (scalar_avx512,
 * scalar_avx2, scalar_fma) with these two entry names set, and makes every other symbol
 * of such a build local. */
#ifndef MATERN_SCALAR_ENTRY
#define MATERN_SCALAR_ENTRY   matern_evaluate_scalar
#define MATERN_SCALAR_ENTRY_Z matern_evaluate_scalar_z
#endif

double matern_scalar_clamp_to_unit(double v){
    if(!(v > 0.0)) return 0.0;      /* also catches a NaN out of a kernel */
    if(v > 1.0)    return 1.0;
    return v;
}

/* The (z, nu) router; validity is tested on (z, nu), and an infinite nu is invalid.
 * WMS_FINISH is the clamp, or in the unfolded twin the prefactor followed by the clamp
 * of a finite value. */
double matern_scalar_one_point_z(double z, double nu)
{
    if (!(nu > 0.0 && nu <= DBL_MAX) || !(z >= 0.0)) return (double)NAN;
    if (!(z > 0.0))        return 1.0;                   /* z = 0: M(0) = 1 */
    /* Past the exponential limit: the folded single-exponent tail. */
    if (!(z <= MATERN_EXP_RANGE_LIMIT))  return WMS_FINISH(matern_scalar_large_argument_folded_at(z, nu), z, nu);

    double Cb = nu*nu/36.0 + 18.0;                       /* C(nu), as in region_partition.h */
    if (nu <= LARGE_ARGUMENT_ORDER_CAP) Cb = fmin(Cb, LARGE_ARGUMENT_Z_FLOOR);   /* fmin == the compare form (region_partition.h) */
    /* Large argument: z > C(nu) and nu <= 30, inclusive; above nu = 30 Debye at every z. */
    if (nu <= LARGE_ARGUMENT_ORDER_CAP && z > Cb)
        return WMS_FINISH(matern_scalar_large_argument_at(z, nu), z, nu);
    /* Below z = 1 the small-argument kernel up to order 30 (26 in the vector and CUDA
     * routers, whose Debye is the more accurate kernel there; the scalar one is not), Debye
     * above; then Debye for nu >= 30.  The order of the tests is theirs. */
    if (z < SMALL_ARGUMENT_Z_CEILING) {
        if (nu <= LARGE_ARGUMENT_ORDER_CAP) return WMS_FINISH(matern_scalar_small_argument(z, nu), z, nu);
        return WMS_FINISH(matern_scalar_debye(z, nu), z, nu);
    }
    if (nu >= LARGE_ARGUMENT_ORDER_CAP) return WMS_FINISH(matern_scalar_debye(z, nu), z, nu);
    /* No upper bound on z is needed (region_partition.h). */
#ifdef WMS_LEVIN
    /* Levin band: off by default, unlike the vector router; tested at that router's position. */
    if (nu <= 15.0){ double L = nu*nu*nu*nu/2401.0 + 1.5;
                     if (z > L) return WMS_FINISH(matern_scalar_levin(z, nu), z, nu); }
#endif
    return WMS_FINISH(matern_scalar_continued_fraction(z, nu), z, nu);
}

/* The (h, nu, s) entry; guarded here because two sign errors in h and s cancel in z. */
static double matern_scalar_one_point(double h, double nu, double scale)
{
    if (!(nu > 0.0 && nu <= DBL_MAX) || !(scale > 0.0 && scale <= DBL_MAX) || !(h >= 0.0))
        return (double)NAN;
    /* M(0) = 1, tested on h: past nu = DBL_MAX/2, 2 nu overflows and sqrt(2 nu) * 0 is NaN. */
    if (h == 0.0) return 1.0;
    return matern_scalar_one_point_z(sqrt(2.0*nu) * h / scale, nu);
}

/* The (z, nu) entry; `scale` is unused, kept for the signature shared by all backends. */
int MATERN_SCALAR_ENTRY_Z(double *out, const double *z, const double *nu,
                   const double *scale, size_t n, struct matern_workspace *ws)
{
    (void)ws; (void)scale;
    if (!n) return MATERN_OK;
    if (!out || !z || !nu) return MATERN_INVALID_ARGUMENT;
    /* A half-integer run of any length goes to the closed form, so a result does not
     * depend on its neighbours. */
    size_t off = 0;
    while (off < n){
        const size_t run = constant_run_length(nu + off, n - off);
        const int    p   = half_integer_degree(nu[off], HALF_INTEGER_SCALAR_CAP);
        if (p >= 0)
            matern_scalar_half_integer_run_z(out+off, z+off, nu[off], run, p);
        else
            for (size_t i = off; i < off + run; i++) out[i] = matern_scalar_one_point_z(z[i], nu[i]);
        off += run;
    }
    return MATERN_OK;
}

int MATERN_SCALAR_ENTRY(double *out, const double *h, const double *nu,
                   const double *scale, size_t n, struct matern_workspace *ws)
{
    (void)ws;
    if (!n) return MATERN_OK;
    if (!out || !h || !nu || !scale) return MATERN_INVALID_ARGUMENT;
    /* A half-integer run of any length goes to the closed form, so a result does not
     * depend on its neighbours. */
    size_t off = 0;
    while (off < n){
        const size_t run = constant_run_length(nu + off, n - off);
        /* Cap nu <= 40.5 here, 29.5 on the vector backends (half_integer_common.h). */
        const int    p   = half_integer_degree(nu[off], HALF_INTEGER_SCALAR_CAP);
        if (p >= 0)
            matern_scalar_half_integer_run(out+off, h+off, nu[off], scale+off, run, p);
        else
            for (size_t i = off; i < off + run; i++) out[i] = matern_scalar_one_point(h[i], nu[i], scale[i]);
        off += run;
    }
    return MATERN_OK;
}
