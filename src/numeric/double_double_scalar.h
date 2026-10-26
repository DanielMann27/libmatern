/* double_double_scalar.h -- width-1 dd_two_sum and dd_log, a line-by-line transcription of
 * double_double_avx512.h (same reduction, table, coefficients and operation order); dd_log
 * requires a positive normal argument, as the vector getmant/getexp pair does, and is
 * accurate to about 2^-58 absolute, not to double-double precision. */
#ifndef MATERN_DOUBLE_DOUBLE_SCALAR_H
#define MATERN_DOUBLE_DOUBLE_SCALAR_H

#include "simd_scalar.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* (s,e) = TwoSum(a,b): a + b = s + e exactly, for either ordering of |a| and |b|.
 * The operations must not be reassociated. */
static inline void dd_two_sum(double a, double b, double *s, double *e)
{
    const double sum = a + b;
    const double bb  = sum - a;
    *s = sum;
    *e = (a - (sum - bb)) + (b - bb);
}

/* The AVX-512 C_lo/C_hi and LC_lo/LC_hi pairs as permutex2var indexes them: 16 entries,
 * _lo first, same values. */
static const double MATERN_DDLOG_C[16] = {
    0x1.66dc1a064aef0p+0, 0x1.57cbc2f6cb044p+0, 0x1.49f23241c41ecp+0, 0x1.3d2b498096841p+0,
    0x1.31584df11803ep+0, 0x1.265ef05ad41adp+0, 0x1.1c2888a4bd60ep+0, 0x1.12a178f6bb705p+0,
    0x1.09b8af69ba7e6p+0, 0x1.0000000000000p+0, 0x1.e6de48066a58bp-1, 0x1.cb8b8e97829fap-1,
    0x1.b3207789e8ce7p-1, 0x1.9d2c5f955aceap-1, 0x1.89544cafebaa1p-1, 0x1.774df5e0f8dcdp-1
};
static const double MATERN_DDLOG_LC[16] = {
   -0x1.59dc40bffdf67p-2,-0x1.2df29fc2eb653p-2,-0x1.03d78316163c6p-2,-0x1.b6ccaf9eb5e35p-3,
   -0x1.68fd6fb8fc365p-3,-0x1.1e079a70103c0p-3,-0x1.ab6f437266f32p-4,-0x1.1fbe7f190542ap-4,
   -0x1.31530d08c0289p-5, 0x0.0p+0,            0x1.9c4fc07f27e26p-5, 0x1.bab9e58337c0bp-4,
    0x1.4d2ee3308680bp-3, 0x1.b7357487f3e5ap-3, 0x1.0e0176feb050ap-2, 0x1.3e0a7a097f046p-2
};

/* log(rh) as lh + ll; operation order as in double_double_avx512.h.  Not a full
 * double-double logarithm: log(1+r) is a double polynomial and r is rounded, so lh + ll is
 * accurate to about 2^-58 absolute; near rh = 1, where log rh is small, that is about
 * 2^-52.6 relative, no better than one double. */
static inline void dd_log(double rh, double *lh, double *ll)
{
    const double   thresh = 0x1.6555555555556p+0;
    const double   ln2_hi = 0x1.62e42fee00000p-1;
    const double   ln2_lo = 0x1.a39ef35793c76p-33;
    const uint64_t base   = 0x3FE6555555555556ULL;

    /* getmant(NORM_1_2) + getexp by bit extraction; exact for a positive normal. */
    uint64_t bits;
    memcpy(&bits, &rh, sizeof bits);
    double e = (double)(int64_t)(((bits >> 52) & 0x7FFULL) - 1023ULL);
    uint64_t mbits = (bits & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m;
    memcpy(&m, &mbits, sizeof m);

    if (m >= thresh) { m = m * 0.5; e = e + 1.0; }      /* the masked mul/add */

    memcpy(&mbits, &m, sizeof mbits);
    const unsigned idx = (unsigned)((mbits - base) >> 48) & 15u;   /* permutex2var */
    const double c    = MATERN_DDLOG_C[idx];
    const double logc = MATERN_DDLOG_LC[idx];

    const double r  = fma_cancel(m, c, -1.0);
    const double r2 = r * r;
    double B = 0x1.c7ca050ba3963p-4;
    B = fma(B, r, -0x1.005f77015466cp-3);
    B = fma(B, r,  0x1.24923c827ddc1p-3);
    B = fma(B, r, -0x1.5555476feb961p-3);
    B = fma(B, r,  0x1.9999999a30370p-3);
    B = fma(B, r, -0x1.0000000052d61p-2);
    B = fma(B, r,  0x1.555555555554dp-2);
    B = fma(B, r, -0x1.ffffffffffff6p-2);
    const double P = fma(r2, B, r);        /* ~log(1+r) */

    /* recomposition: log = e*ln2_hi (exact) + e*ln2_lo + logc + P */
    double s, es;   dd_two_sum(logc, P, &s, &es);
    const double lo  = fma(e, ln2_lo, es);
    const double eh2 = e * ln2_hi;         /* exact: ln2_hi has 32 significant bits */
    double hh, ee2; dd_two_sum(eh2, s, &hh, &ee2);
    *lh = hh;
    *ll = ee2 + lo;
}

#endif /* MATERN_DOUBLE_DOUBLE_SCALAR_H */
