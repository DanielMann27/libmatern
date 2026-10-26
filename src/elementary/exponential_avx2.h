/* exponential_avx2.h -- e^x for four doubles with the reduction, table and polynomial of
 * exponential_avx512.h; the 2^(j/8) table is assembled from 32-bit halves by permutes.
 * Finite x in [ln DBL_MIN, 708]. */
#ifndef MATERN_EXPONENTIAL_AVX2_H
#define MATERN_EXPONENTIAL_AVX2_H

#include <immintrin.h>
#include <stdint.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "exponential_avx2.h needs AVX2 and FMA (-mavx2 -mfma)."
#endif

/* MATERN_EXP_TH/TL of exponential_avx512.h as 32-bit halves; the values must stay equal. */
static const uint32_t MATERN_EXP_TH_LO32[8] = {
    0x00000000u, 0x3c7d517bu, 0x0a31b715u, 0xd5362a27u,
    0x667f3bcdu, 0x422aa0dbu, 0x995ad3adu, 0xdcfba487u
};
static const uint32_t MATERN_EXP_TH_HI32[8] = {
    0x3ff00000u, 0x3ff172b8u, 0x3ff306feu, 0x3ff4bfdau,
    0x3ff6a09eu, 0x3ff8ace5u, 0x3ffae89fu, 0x3ffd5818u
};
static const uint32_t MATERN_EXP_TL_LO32[8] = {
    0x00000000u, 0xb9d78a76u, 0xd23182e4u, 0xafec42e2u,
    0x13b26456u, 0x56864b27u, 0x345dcc81u, 0xd75b3707u
};
static const uint32_t MATERN_EXP_TL_HI32[8] = {
    0x00000000u, 0xbc819041u, 0x3c86f46au, 0x3c7d4397u,
    0xbc9bdd34u, 0x3c96e9f1u, 0x3c97a1cdu, 0x3c82ed02u
};

/* Same values as the MATERN_EXP_* constants of exponential_avx512.h. */
#define MATERN_EXP_8_BY_LN2  0x1.71547652b82fep+3
#define MATERN_EXP_LN2_8_HI  0x1.62e42f8000000p-4
#define MATERN_EXP_LN2_8_LO  0x1.be8e7bcd5e4f2p-30

#define MATERN_EXP_C2  0x1.0000000000138p-1
#define MATERN_EXP_C3  0x1.55555555552ccp-3
#define MATERN_EXP_C4  0x1.55555547c6b2ap-5
#define MATERN_EXP_C5  0x1.11111125558f7p-7
#define MATERN_EXP_C6  0x1.6c1cbae98c662p-10
#define MATERN_EXP_C7  0x1.a00d960e1cab3p-13

/* 2^(j/8) lookup: idxHi = j << 32 puts j in the odd 32-bit slot, so vpermd places low
 * halves in even slots and high halves in odd ones; blend 0xAA joins them. */
static inline void matern_exp_table_lookup(__m256i j, __m256d *th, __m256d *tl)
{
    const __m256i idxLo = j;
    const __m256i idxHi = _mm256_slli_epi64(j, 32);

    __m256i th_lo = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256((const __m256i *)MATERN_EXP_TH_LO32), idxLo);
    __m256i th_hi = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256((const __m256i *)MATERN_EXP_TH_HI32), idxHi);
    *th = _mm256_castsi256_pd(_mm256_blend_epi32(th_lo, th_hi, 0xAA));

    __m256i tl_lo = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256((const __m256i *)MATERN_EXP_TL_LO32), idxLo);
    __m256i tl_hi = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256((const __m256i *)MATERN_EXP_TL_HI32), idxHi);
    *tl = _mm256_castsi256_pd(_mm256_blend_epi32(tl_lo, tl_hi, 0xAA));
}

/* SHIFT = 1.5*2^52: the fma rounds x*8/ln2 to the integer q and leaves it in the low
 * mantissa bits, so s - SHIFT is q as a double and bits(s) - BIAS (BIAS = bits of SHIFT)
 * is q as a two's-complement integer, without a conversion instruction. */
static inline __m256d matern_exponential_pd(__m256d x)
{
    const __m256d SHIFT = _mm256_set1_pd(0x1.8p52);
    const __m256i BIAS  = _mm256_set1_epi64x(0x4338000000000000LL);
    __m256d s   = _mm256_fmadd_pd(x, _mm256_set1_pd(MATERN_EXP_8_BY_LN2), SHIFT);
    __m256d qd  = _mm256_sub_pd(s, SHIFT);
    __m256i q   = _mm256_sub_epi64(_mm256_castpd_si256(s), BIAS);
    __m256i j   = _mm256_and_si256(q, _mm256_set1_epi64x(7));
    __m256i k8  = _mm256_and_si256(q, _mm256_set1_epi64x(~7LL));  /* 8k */

    /* r = x - q*(ln2/8) in [-ln2/16, ln2/16]. */
    __m256d r = _mm256_fnmadd_pd(qd, _mm256_set1_pd(MATERN_EXP_LN2_8_HI), x);
    r         = _mm256_fnmadd_pd(qd, _mm256_set1_pd(MATERN_EXP_LN2_8_LO), r);

    __m256d th, tl;
    matern_exp_table_lookup(j, &th, &tl);

    /* e^r = 1 + r*w(r), w = (e^r - 1)/r by Horner. */
    __m256d w = _mm256_set1_pd(MATERN_EXP_C7);
    w = _mm256_fmadd_pd(w, r, _mm256_set1_pd(MATERN_EXP_C6));
    w = _mm256_fmadd_pd(w, r, _mm256_set1_pd(MATERN_EXP_C5));
    w = _mm256_fmadd_pd(w, r, _mm256_set1_pd(MATERN_EXP_C4));
    w = _mm256_fmadd_pd(w, r, _mm256_set1_pd(MATERN_EXP_C3));
    w = _mm256_fmadd_pd(w, r, _mm256_set1_pd(MATERN_EXP_C2));
    w = _mm256_fmadd_pd(w, r, _mm256_set1_pd(1.0));         /* C1 = 1 */

    /* y = th + (th*(r*w) + tl), in the order of exponential_avx512.h. */
    __m256d u   = _mm256_mul_pd(r, w);
    __m256d tmp = _mm256_fmadd_pd(th, u, tl);
    __m256d y   = _mm256_add_pd(th, tmp);

    /* 2^k * y with (8k) << 49 == k << 52 (AVX2 has no srai_epi64). */
    __m256i bits = _mm256_add_epi64(_mm256_castpd_si256(y), _mm256_slli_epi64(k8, 49));
    return _mm256_castsi256_pd(bits);
}

#endif /* MATERN_EXPONENTIAL_AVX2_H */
