/* exponential_avx512.h -- e^x for eight doubles: Cody-Waite reduction by ln2/8, a 2^(j/8)
 * double-double table read by an in-register permute, degree-7 minimax; finite x in
 * [ln DBL_MIN, 708] (callers bound the argument). */
#ifndef MATERN_EXPONENTIAL_AVX512_H
#define MATERN_EXPONENTIAL_AVX512_H

#include <immintrin.h>

#if !defined(__AVX512F__) || !defined(__AVX512DQ__) || !defined(__FMA__)
#  error "exponential_avx512.h needs AVX-512F, AVX-512DQ and FMA (-mavx512f -mavx512dq -mfma)."
#endif

/* p * 2^k by an integer add of k << 52 to the bits of p; exact while the biased
 * exponent of p plus k stays in [1, 2046], which holds for x in [ln DBL_MIN, 708]. */
static inline __m512d matern_exp_scale_by_power_of_two(__m512d p, __m512i k)
{
    __m512i bits = _mm512_add_epi64(_mm512_castpd_si512(p), _mm512_slli_epi64(k, 52));
    return _mm512_castsi512_pd(bits);
}


/* 2^(j/8), j = 0..7: TH[j] = RN(2^(j/8)), TL[j] = RN(2^(j/8) - TH[j]).  The same
 * values are in exponential_avx2.h, gamma_avx512.h, gamma_coefficients_avx2.h,
 * power_avx512.h and power_coefficients_avx2.h and must stay equal. */
static const double MATERN_EXP_TH[8] __attribute__((aligned(64))) = {
    0x1.0000000000000p+0, 0x1.172b83c7d517bp+0, 0x1.306fe0a31b715p+0, 0x1.4bfdad5362a27p+0,
    0x1.6a09e667f3bcdp+0, 0x1.8ace5422aa0dbp+0, 0x1.ae89f995ad3adp+0, 0x1.d5818dcfba487p+0
};
static const double MATERN_EXP_TL[8] __attribute__((aligned(64))) = {
    0x0.0p+0, -0x1.19041b9d78a76p-55, 0x1.6f46ad23182e4p-55, 0x1.d4397afec42e2p-56,
    -0x1.bdd3413b26456p-54, 0x1.6e9f156864b27p-54, 0x1.7a1cd345dcc81p-54, 0x1.2ed02d75b3707p-55
};

/* ln2/8 split for Cody-Waite; the high part ends in 27 zero bits, so q*HI is exact. */
#define MATERN_EXP_8_BY_LN2  0x1.71547652b82fep+3   /* 8 / ln2 */
#define MATERN_EXP_LN2_8_HI  0x1.62e42f8000000p-4   /* ln2/8, high */
#define MATERN_EXP_LN2_8_LO  0x1.be8e7bcd5e4f2p-30  /* ln2/8, low  */

/* Degree-7 minimax for e^r on [-ln2/16, ln2/16], C0 = C1 = 1. */
#define MATERN_EXP_C1  0x1.0000000000000p+0
#define MATERN_EXP_C2  0x1.0000000000138p-1
#define MATERN_EXP_C3  0x1.55555555552ccp-3
#define MATERN_EXP_C4  0x1.55555547c6b2ap-5
#define MATERN_EXP_C5  0x1.11111125558f7p-7
#define MATERN_EXP_C6  0x1.6c1cbae98c662p-10
#define MATERN_EXP_C7  0x1.a00d960e1cab3p-13

static inline __m512d matern_exponential_pd(__m512d x)
{
    /* q = round(x*8/ln2): adding 1.5*2^52 leaves q in the low mantissa bits, so q is
     * available as integer bits and, after subtracting SHIFT, as a double without a
     * conversion.  Valid since |x*8/ln2| < 2^51. */
    const __m512d  SHIFT = _mm512_set1_pd(0x1.8p52);          /* 1.5 * 2^52 */
    const __m512i  BIAS  = _mm512_set1_epi64(0x4338000000000000LL); /* bits of SHIFT */
    __m512d s  = _mm512_fmadd_pd(x, _mm512_set1_pd(MATERN_EXP_8_BY_LN2), SHIFT);
    __m512i qi = _mm512_castpd_si512(s);
    __m512d qd = _mm512_sub_pd(s, SHIFT);
    __m512i j  = _mm512_and_si512(qi, _mm512_set1_epi64(7));
    __m512i k  = _mm512_srai_epi64(_mm512_sub_epi64(qi, BIAS), 3); /* floor(q/8) */

    /* r = x - q*(ln2/8) in [-ln2/16, ln2/16]. */
    __m512d r = _mm512_fnmadd_pd(qd, _mm512_set1_pd(MATERN_EXP_LN2_8_HI), x);
    r         = _mm512_fnmadd_pd(qd, _mm512_set1_pd(MATERN_EXP_LN2_8_LO), r);

    __m512d th = _mm512_permutexvar_pd(j, _mm512_loadu_pd(MATERN_EXP_TH));
    __m512d tl = _mm512_permutexvar_pd(j, _mm512_loadu_pd(MATERN_EXP_TL));

    /* e^r = 1 + r*w(r), w = (e^r - 1)/r by Horner. */
    __m512d w = _mm512_set1_pd(MATERN_EXP_C7);
    w = _mm512_fmadd_pd(w, r, _mm512_set1_pd(MATERN_EXP_C6));
    w = _mm512_fmadd_pd(w, r, _mm512_set1_pd(MATERN_EXP_C5));
    w = _mm512_fmadd_pd(w, r, _mm512_set1_pd(MATERN_EXP_C4));
    w = _mm512_fmadd_pd(w, r, _mm512_set1_pd(MATERN_EXP_C3));
    w = _mm512_fmadd_pd(w, r, _mm512_set1_pd(MATERN_EXP_C2));
    w = _mm512_fmadd_pd(w, r, _mm512_set1_pd(MATERN_EXP_C1));

    /* y = th + (th*(r*w) + tl); the order is fixed.  Folding tl in through the fma leaves
     * one rounding (the final add); forming th + tl first exposes the table rounding.
     * th*u is exact inside the fma, the dropped tl*(e^r - 1) is <= 2^-58, and
     * y in [0.9576, 1.916] stays normal, so the exponent-add scaling is exact. */
    __m512d u   = _mm512_mul_pd(r, w);
    __m512d tmp = _mm512_fmadd_pd(th, u, tl);
    __m512d y   = _mm512_add_pd(th, tmp);

    return matern_exp_scale_by_power_of_two(y, k);
}

#endif /* MATERN_EXPONENTIAL_AVX512_H */
