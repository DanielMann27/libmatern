/* debye_polynomial.h -- the Debye polynomials u_k (Horner in q = p^2) and their sum
 * SUM_k (-1)^k u_k(p)/nu^k (even/odd Horner in (p/nu)^2) in the vector vocabulary, over
 * tables padded to one vector per coefficient, indexed C + m * INDEX_FACTOR. */
#ifndef MATERN_DEBYE_POLYNOMIAL_H
#define MATERN_DEBYE_POLYNOMIAL_H

#include "vector_current.h"

#if defined(MATERN_TARGET_SCALAR)
  #include "debye_coefficients_scalar_lanes.h"
#elif defined(MATERN_TARGET_AVX2)
  #include "debye_coefficients_avx2.h"
#else
  #include "debye_coefficients_avx512.h"
#endif

/* One u_k without its p^k factor, by Horner in q.  The coefficients are magnitudes:
 * each step adds or subtracts by the parity of m and `deg & 1` sets the leading sign;
 * the caller passes deg = k. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT debye_polynomial(const DATA_TYPE_FLOAT *C, int deg, SIMD_TYPE_FLOAT q)
{
    SIMD_TYPE_FLOAT acc = VEC_VLD(C + deg * INDEX_FACTOR);
    if (deg & 1) acc = VEC_SUB(VEC_VSET(0.0), acc);

    /* gcc leaves this file's loops rolled at -O2; unrolling does not reassociate. */
#pragma GCC unroll 16
    for (int m = deg - 1; m >= 0; m--) {
        const SIMD_TYPE_FLOAT c = VEC_VLD(C + m * INDEX_FACTOR);
        acc = (m & 1) ? VEC_FMS_CANCEL(acc, q, c) : VEC_FMA_CANCEL(acc, q, c);
    }
    return acc;
}

/* sum_{k=0..top} (-1)^k u_k(p)/nu^k, even/odd Horner in in2 = (p/nu)^2 */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT debye_polynomial_sum(SIMD_TYPE_FLOAT P1, SIMD_TYPE_FLOAT p,
                                     SIMD_TYPE_FLOAT NU, int top)
{
    SIMD_TYPE_FLOAT poly[11];
    poly[0] = VEC_VSET(1.0);

#pragma GCC unroll 16
    for (int k = 1; k <= top; k++)
        poly[k] = debye_polynomial(DEBYE_COEFFICIENT_ROWS[k], k, P1);

    /* in = -p/nu carries the alternating sign of the sum: (-1)^k (p/nu)^k = in^k. */
    const SIMD_TYPE_FLOAT in  = VEC_DIV(VEC_SUB(VEC_VSET(0.0), p), NU);
    const SIMD_TYPE_FLOAT in2 = VEC_MUL(in, in);

    /* Two Horner chains in in2, even and odd k, combined once at the end; te and to
     * are the highest even and odd indices not above top. */
    const int te = (top & 1) ? top - 1 : top;
    const int to = (top & 1) ? top     : top - 1;

    SIMD_TYPE_FLOAT e = poly[te];
#pragma GCC unroll 16
    for (int k = te - 2; k >= 0; k -= 2) e = VEC_FMA_CANCEL(e, in2, poly[k]);

    SIMD_TYPE_FLOAT o = poly[to];
#pragma GCC unroll 16
    for (int k = to - 2; k >= 1; k -= 2) o = VEC_FMA_CANCEL(o, in2, poly[k]);

    return VEC_FMA_CANCEL(o, in, e);       /* even + in * odd */
}

#endif /* MATERN_DEBYE_POLYNOMIAL_H */
