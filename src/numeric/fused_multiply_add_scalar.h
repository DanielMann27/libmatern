/* fused_multiply_add_scalar.h -- the width-1 fma: under -mfma (matched_fma) fma() and
 * fma_cancel() are the instruction; without it (matched) fma() is a*b + c and fma_cancel()
 * uses two_product and two_sum, or libm fma() outside their safe range. */
#ifndef MATERN_FUSED_MULTIPLY_ADD_SCALAR_H
#define MATERN_FUSED_MULTIPLY_ADD_SCALAR_H

/* Included from simd_scalar.h only, under -DMATERN_TARGET_SCALAR and before any kernel
 * text, so the fma macro reaches every call site.  A new fma site whose operands can
 * cancel must be written VEC_*_CANCEL. */
#include <math.h>

#if defined(__FMA__)

/* matched_fma: the instruction, at every site. */
#define fma_cancel(A, B, C) fma((A), (B), (C))

#else

/* two_sum and Dekker's two_product, in plain C. */
#include "double_double_portable.h"

/* Calls libm fma(), so it must precede the fma macro at the end of this file. */
static __attribute__((noinline, cold, unused))
double matern_fma_libm(double a, double b, double c)
{
    return fma(a, b, c);
}

/* Returns s = a*b + c when |s| >= 2|a*b| (the product's rounding is then at most
 * ulp(s)/4); otherwise the result of the exact two_product and two_sum, which reproduces a
 * residual fma(a, b, -fl(a*b)) exactly and elsewhere equals fma() or an adjacent double.
 * Outside Dekker's safe range the call goes to libm fma(). */
static inline __attribute__((always_inline, unused))
double matern_fma_guarded(double a, double b, double c)
{
    if (a == 0.0 || b == 0.0)
        return a * b + c;                 /* exact: the product is a signed zero */

    const double p = a * b;

    /* No significant cancellation.  The negated compare returns a NaN s here; an
     * out-of-range product falls through to the libm call below. */
    const double s = p + c;
    if (!(fabs(s) < 2.0 * fabs(p)) && fabs(p) < 0x1p1000)
        return s;

    const double ap = fabs(a), bp = fabs(b), pp = fabs(p);
    if (!(ap < 0x1p995 && bp < 0x1p995 && pp < 0x1p1000 && pp > 0x1p-900))
        return matern_fma_libm(a, b, c);

    const matern_double_double prod = matern_two_product(a, b);   /* hi == p */
    const matern_double_double sum  = matern_two_sum(prod.hi, c);
    return sum.hi + (sum.lo + prod.lo);
}

#define fma_cancel(A, B, C) matern_fma_guarded((A), (B), (C))

/* Every later fma( in the translation unit: two roundings. */
#define fma(A, B, C) ((A) * (B) + (C))

#endif /* __FMA__ */

#endif /* MATERN_FUSED_MULTIPLY_ADD_SCALAR_H */
