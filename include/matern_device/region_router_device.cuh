/* region_router_device.cuh -- per-point device router (see the accompanying paper, domain
 * decomposition): validity, half-integer closed form, extended range, then the region cascade.
 * The order of the branch tests defines the partition at every border and must not change. */
#ifndef MATERN_REGION_ROUTER_DEVICE_CUH
#define MATERN_REGION_ROUTER_DEVICE_CUH

#include <float.h>            /* DBL_EPSILON, DBL_MAX */
#include "derive_argument_device.cuh"
#include "asymptotic_large_argument_device.cuh"
#include "levin_transform_device.cuh"
#include "debye_uniform_order_device.cuh"
#include "temme_small_argument_device.cuh"
#include "continued_fraction_wronskian_device.cuh"
#include "half_integer_closed_form_device.cuh"

/* Routing flags, the compile-time template argument F.  The values are reported by
 * matern_cuda_route_flags (cuda.cu, under MATERN_CUDA_BENCHMARK_HOOKS) and must not be
 * renumbered; 4 and 16 are unused.  Every set contains the five bits of
 * MATERN_ROUTE_FLAGS_BASE; the LEVIN, HALF_INTEGER and DIRECT_ARGUMENT bits are switches;
 * any other bit does not compile. */
#define MATERN_ROUTE_LARGE_ARGUMENT_Z_FLOOR_BIT      1  /* flat large-argument cap min(C,22) for nu <= 30   */
#define MATERN_ROUTE_LARGE_ARGUMENT_ORDER_CAP_BIT    2  /* the large-argument order cap, nu <= 30           */
#define MATERN_ROUTE_TEMME_FOLD_BIT                  8  /* Temme with the Matern prefactor folded in        */
#define MATERN_ROUTE_LEVIN_BIT                      32  /* route the Levin band to the Levin kernel         */
#define MATERN_ROUTE_SMALL_ARGUMENT_ORDER_MAX_BIT   64  /* small-z <-> Debye border at nu <= 26             */
#define MATERN_ROUTE_FULL_DOMAIN_BIT               128  /* invalid input NaN, z = 0 gives 1, extended range */
#define MATERN_ROUTE_HALF_INTEGER_BIT              256  /* the elementary half-integer closed form          */
#define MATERN_ROUTE_DIRECT_ARGUMENT_BIT           512  /* direct argument: `zc` is z, no derivation        */

/* The half-integer closed form, nu = p + 1/2 for p <= this cap. */
#define MATERN_ROUTE_HALF_INTEGER_CAP         29
/* The large-argument term cap is two-sided: more terms can pass the series' minimum term. */
#define MATERN_ROUTE_LARGE_ARGUMENT_MAX_TERMS 50
#define MATERN_ROUTE_CF_SERIES_TERMS          50
#define MATERN_ROUTE_CF_LENTZ_TERMS           50

/* The shipped flag sets. */
#define MATERN_ROUTE_FLAGS_BASE      (MATERN_ROUTE_LARGE_ARGUMENT_Z_FLOOR_BIT \
                                      | MATERN_ROUTE_LARGE_ARGUMENT_ORDER_CAP_BIT \
                                      | MATERN_ROUTE_SMALL_ARGUMENT_ORDER_MAX_BIT \
                                      | MATERN_ROUTE_TEMME_FOLD_BIT | MATERN_ROUTE_FULL_DOMAIN_BIT)
#define MATERN_ROUTE_FLAGS_DEFAULT   (MATERN_ROUTE_FLAGS_BASE | MATERN_ROUTE_HALF_INTEGER_BIT)
#define MATERN_ROUTE_FLAGS_DEFAULT_Z (MATERN_ROUTE_FLAGS_DEFAULT | MATERN_ROUTE_DIRECT_ARGUMENT_BIT)

/* Levin variants: without MATERN_ROUTE_LEVIN_BIT the Levin band goes to the continued fraction. */
#define MATERN_ROUTE_FLAGS_DEFAULT_LEVIN   (MATERN_ROUTE_FLAGS_DEFAULT   | MATERN_ROUTE_LEVIN_BIT)
#define MATERN_ROUTE_FLAGS_DEFAULT_LEVIN_Z (MATERN_ROUTE_FLAGS_DEFAULT_Z | MATERN_ROUTE_LEVIN_BIT)

#define MATERN_ROUTE_FLAGS_VALID(F) \
    (((F) & MATERN_ROUTE_FLAGS_BASE) == MATERN_ROUTE_FLAGS_BASE && \
     ((F) & ~(MATERN_ROUTE_FLAGS_BASE | MATERN_ROUTE_LEVIN_BIT | MATERN_ROUTE_HALF_INTEGER_BIT \
              | MATERN_ROUTE_DIRECT_ARGUMENT_BIT)) == 0)

/* lx2 = ln(z/2) from the inputs, read by the Temme kernel only where Zh is subnormal or 0. */
template<int F>
static __device__ __forceinline__ double matern_route_at_argument_device(double Zh, double Zl, double nuc,
                                                                         double lx2)
{
    static_assert(MATERN_ROUTE_FLAGS_VALID(F),
                  "a routing flag set is MATERN_ROUTE_FLAGS_BASE plus any of the LEVIN, HALF_INTEGER and DIRECT_ARGUMENT bits");

    /* the large-argument border C(nu), flattened to min(C, 22) for nu <= 30 */
    double C = nuc * nuc / 36.0 + 18.0;
    if (nuc <= 30.0 && C > 22.0) C = 22.0;

    /* the Matern prefactor 2^(1-nu) / Gamma(nu) * z^nu, which the
     * large-argument kernel applies */
    double pref = pow(2.0, 1.0 - nuc) / tgamma(nuc) * pow(Zh, nuc);

    /* the large-argument region; above nu = 30 these points go to Debye */
    int lz = (nuc <= 30.0) && (Zh > C);

    if (lz)
        return matern_large_argument_dd_argument_device(Zh, Zl, nuc, pref,
                                MATERN_ROUTE_LARGE_ARGUMENT_MAX_TERMS, DBL_EPSILON);

    if (Zh < 1.0){
        if (nuc <= 26.0)
            return matern_temme_small_argument_fold_device(Zh, nuc, lx2);   /* M(h) directly */
        return matern_debye_fold_dd_argument_device(Zh, Zl, nuc);
    }
    /* For nu <= 30 the flattened border is at most 22, so the large-argument
     * region has already taken every point with z >= 22 there. */
    if (nuc >= 30.0)
        return matern_debye_fold_dd_argument_device(Zh, Zl, nuc);
    /* The Levin band sits where region_partition.h places it, immediately before the
     * continued fraction, so MATERN_ROUTE_LEVIN_BIT changes nothing else. */
    if (F & MATERN_ROUTE_LEVIN_BIT) {
        const double Lfloor = fma(nuc * nuc, (nuc * nuc) / 2401.0, 1.5);
        if (nuc <= 15.0 && Zh > Lfloor)
            return matern_levin_fold_dd_argument_device(Zh, Zl, nuc);
    }
    return matern_continued_fraction_dd_argument_device(Zh, Zl, nuc, MATERN_ROUTE_CF_SERIES_TERMS, MATERN_ROUTE_CF_LENTZ_TERMS, DBL_EPSILON);
}

template<int F>
static __device__ __forceinline__ double matern_route_device(double zc, double nuc, double scalec)
{
    static_assert(MATERN_ROUTE_FLAGS_VALID(F),
                  "a routing flag set is MATERN_ROUTE_FLAGS_BASE plus any of the LEVIN, HALF_INTEGER and DIRECT_ARGUMENT bits");
    /* Order: validity, z = 0, then the exponential range.  Under MATERN_ROUTE_DIRECT_ARGUMENT_BIT
     * z is an input, so there is no scale to test and no sign cancellation. */
    /* An infinite nu or s is invalid, as on the CPU backends. */
    const bool bad = (F & MATERN_ROUTE_DIRECT_ARGUMENT_BIT) ? (!(nuc > 0.0 && nuc <= DBL_MAX) || !(zc >= 0.0))
                                 : (!(nuc > 0.0 && nuc <= DBL_MAX) || !(scalec > 0.0 && scalec <= DBL_MAX)
                                    || !(zc >= 0.0));
    if (bad)
        return __longlong_as_double(0x7ff8000000000000LL);   /* NaN */
    if (!(zc > 0.0)) return 1.0;
    /* the single derivation site; under MATERN_ROUTE_DIRECT_ARGUMENT_BIT z is taken with a
     * zero low word */
    double Zh, Zl;
    if (F & MATERN_ROUTE_DIRECT_ARGUMENT_BIT){ Zh = zc; Zl = 0.0; }
    else          { matern_derive_argument_device(nuc, zc, scalec, &Zh, &Zl); }
    /* The half-integer closed form precedes the exponential-range test and carries its
     * own: e^-z P(z) below the limit, the exp-log form above.  No clamp is needed: c_j > 0
     * and Zh >= 0 give s >= 1, and e^-Zh >= DBL_MIN below the limit, so the result is
     * finite, positive and mathematically at most 1. */
    if (F & MATERN_ROUTE_HALF_INTEGER_BIT){
        const int p = matern_half_integer_degree_device(nuc);
        if (p >= 0 && p <= MATERN_ROUTE_HALF_INTEGER_CAP)          /* precision border, not the table bound */
            return (Zh <= MATERN_EXP_RANGE_MAX_DEVICE) ? matern_half_integer_direct_form_device(Zh, Zl, p)
                                    : matern_half_integer_exp_log_device(Zh, Zl, p);
    }
    if (!(Zh <= MATERN_EXP_RANGE_MAX_DEVICE))                      /* folded tail */
        return matern_large_argument_fold_device(Zh, Zl, nuc, MATERN_ROUTE_LARGE_ARGUMENT_MAX_TERMS);
    /* Below DBL_MIN the derived z has lost bits or underflowed to 0, and at z = 2^-1074
     * the Temme kernel's z/2 rounds to 0; it then takes ln(z/2) formed here from the
     * inputs.  At small nu, M = 1 - O((z/2)^(2nu)) can differ from 1 even at such z. */
    double lx2 = 0.0;
    if (!(Zh >= DBL_MIN))
        lx2 = ((F & MATERN_ROUTE_DIRECT_ARGUMENT_BIT) ? log(zc)
                                    : log(zc) - log(scalec) + 0.5 * log(2.0 * nuc)) - M_LN2;
    double v = matern_route_at_argument_device<F>(Zh, Zl, nuc, lx2);
    if (!(v > 0.0)) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

#endif /* MATERN_REGION_ROUTER_DEVICE_CUH */
