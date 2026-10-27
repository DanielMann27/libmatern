/* region_partition.h -- region codes, border constants and the scalar statement of the
 * partition, matern_region_of() (see the accompanying paper, domain decomposition).  The
 * vector backends classify every point with matern_classify (pipeline_vector.h), which
 * makes the same decisions except within an ulp of the C(nu) and L(nu) borders, where it
 * rounds those curves differently; matern_region_of has no caller in the library and
 * documents the partition that backends/scalar/region_router.c and the CUDA router
 * write out. */
#ifndef MATERN_REGION_PARTITION_H
#define MATERN_REGION_PARTITION_H

#include <math.h>

/* The low-order large-argument code runs the same kernel as its sibling; the split
 * exists only for the input sort and changes no result. */
enum {
    REGION_LARGE_ARGUMENT_LOW_ORDER = 0,  /* z > C(nu), nu < 8   -- sort class    */
    REGION_LARGE_ARGUMENT,                /* z > C(nu), nu >= 8                   */
    REGION_LEVIN,                         /* nu <= 15, z above the band floor     */
    REGION_CONTINUED_FRACTION,            /* the middle                           */
    REGION_DEBYE,                         /* nu >= 30, and small z above order 26 */
    REGION_SMALL_ARGUMENT,                /* z < 1, nu <= 26                      */
#ifdef MATERN_ROUTE_HALF_INTEGER
    /* avx512 and avx2 only (vector.c): nu = p + 1/2 within the vector cap, served by the
     * closed form per point.  A routing class, not a region of the partition below; last,
     * so that the output clamp exempts it together with the trivial bucket. */
    REGION_HALF_INTEGER,
#endif
    REGION_COUNT
};

/* The trivial bucket (invalid input, z = 0, z past the exponential limit) is last, so
 * the output clamp exempts it with a bound; its NaN for invalid input must not become 0. */
#define REGION_TRIVIAL  (REGION_COUNT)
#define REGION_TOTAL    (REGION_COUNT + 1)

/* Borders.  The scalar backend reads them too, except SMALL_ARGUMENT_ORDER_MAX and
 * GAMMA_RECURRENCE_ORDER; the CUDA router (region_router_device.cuh) defines its own.
 * MATERN_EXP_RANGE_LIMIT is also defined in bessel/half_integer_closed_form.h and must
 * stay token-identical. */
#define SMALL_ARGUMENT_Z_CEILING   1.0    /* below this z, Temme's series        */
#define SMALL_ARGUMENT_ORDER_MAX   26.0   /* above this order, Debye instead     */
#define LARGE_ARGUMENT_Z_FLOOR     22.0   /* the flattened C(nu), for nu <= 30   */
#define LARGE_ARGUMENT_ORDER_CAP   30.0   /* above this order, Debye instead     */
#define GAMMA_RECURRENCE_ORDER     8.0    /* the sort class boundary             */
#define EXTENDED_RANGE_ORDER_SPLIT 30.0   /* which tail assembly serves a point  */
#define MATERN_EXP_RANGE_LIMIT    708.396418532264  /* = -log(DBL_MIN)          */

/* Scalar classifier; the vector path makes the same decisions.  The tests form a cascade,
 * each assuming the earlier ones failed.  No test needs an upper bound on z: for
 * nu <= 30 the large-argument test takes every z > 22, for nu >= 30 Debye takes the rest. */
static inline int matern_region_of(double z,double nu){
    if(!(z>0.0))    return REGION_TRIVIAL;                       /* z <= 0, invalid, NaN */
    if(!(z<=MATERN_EXP_RANGE_LIMIT)) return REGION_TRIVIAL;                     /* past the exp limit */
    /* C(nu) is also written out in pipeline_vector.h, backends/scalar/region_router.c
     * and region_router_device.cuh.  18: the series' smallest term, about e^{-2z},
     * falls below 2^-52 at z >= ln(2^52)/2 = 18.02. */
    double C = nu*nu/36.0 + 18.0;
    /* fmin equals the compare form: a NaN nu gives a NaN z, rejected above. */
    if (nu <= LARGE_ARGUMENT_ORDER_CAP) C = fmin(C, LARGE_ARGUMENT_Z_FLOOR);
    if (nu <= LARGE_ARGUMENT_ORDER_CAP && z > C)
        return (nu < GAMMA_RECURRENCE_ORDER) ? REGION_LARGE_ARGUMENT_LOW_ORDER : REGION_LARGE_ARGUMENT;
    if (z < SMALL_ARGUMENT_Z_CEILING)  return (nu <= SMALL_ARGUMENT_ORDER_MAX) ? REGION_SMALL_ARGUMENT : REGION_DEBYE;
    if (nu >= LARGE_ARGUMENT_ORDER_CAP) return REGION_DEBYE;
#ifndef MATERN_WITHOUT_LEVIN
    /* The Levin band; -DMATERN_WITHOUT_LEVIN sends it to the continued fraction.
     * Must stay in step with matern_classify in pipeline_vector.h. */
    if (nu <= 15.0){ double L=nu*nu*nu*nu/2401.0+1.5; if (z > L) return REGION_LEVIN; }
#endif
    return REGION_CONTINUED_FRACTION;
}

#endif /* MATERN_REGION_PARTITION_H */
