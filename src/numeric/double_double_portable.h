/* double_double_portable.h -- error-free TwoSum (Knuth) and TwoProduct (Dekker splitting)
 * in plain C, used by the cancellation-guarded fma of the width-1 build without hardware
 * FMA (fused_multiply_add_scalar.h). */
#ifndef MATERN_DOUBLE_DOUBLE_PORTABLE_H
#define MATERN_DOUBLE_DOUBLE_PORTABLE_H

typedef struct { double hi, lo; } matern_double_double;


/* hi + lo == a + b exactly for finite operands whose sum does not overflow, for either
 * ordering of |a| and |b|.  The operations must not be reassociated. */
static inline matern_double_double matern_two_sum(double a, double b)
{
    matern_double_double result;

    result.hi = a + b;

    const double b_part = result.hi - a;
    result.lo = (a - (result.hi - b_part)) + (b - b_part);

    return result;
}


/* hi + lo == a * b exactly when no intermediate overflows or underflows: the splitting
 * overflows for |a| above about DBL_MAX / 2^27 and the partial products underflow near
 * DBL_MIN, so the caller keeps operands in a safe range (fused_multiply_add_scalar.h).
 * Exact also under -ffp-contract=fast, since every partial product is representable. */
static inline matern_double_double matern_two_product(double a, double b)
{
    matern_double_double result;

    result.hi = a * b;

    /* Dekker's splitting into halves of 26 significant bits. */
    const double splitter = 134217729.0;               /* 2^27 + 1 */

    const double a_scaled = a * splitter;
    const double a_high   = a_scaled - (a_scaled - a);
    const double a_low    = a - a_high;

    const double b_scaled = b * splitter;
    const double b_high   = b_scaled - (b_scaled - b);
    const double b_low    = b - b_high;

    result.lo = ((a_high * b_high - result.hi) + a_high * b_low
                 + a_low * b_high) + a_low * b_low;

    return result;
}

#endif /* MATERN_DOUBLE_DOUBLE_PORTABLE_H */
