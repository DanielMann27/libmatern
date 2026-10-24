/* logarithm_reduction_avx2.h -- AVX2 logarithm reduction helpers: a 16-entry table read
 * from 32-bit halves, the borrow reduction (recentring in the exponent field) and the
 * plain mantissa/exponent split. */
#ifndef MATERN_LOGARITHM_REDUCTION_AVX2_H
#define MATERN_LOGARITHM_REDUCTION_AVX2_H

#include <stdint.h>
#include "simd_avx2.h"

/* table[j] from four uint32 arrays: LOA/HIA hold the low/high halves of entries 0..7,
 * LOB/HIB those of entries 8..15.  The caller passes idxLo = j, idxHi = j << 32 and
 * m8 = (j > 7); blend 0xAA joins the low (even) and high (odd) 32-bit slots. */
static inline __m256d matern_log_table_16(const uint32_t *LOA, const uint32_t *LOB,
                                   const uint32_t *HIA, const uint32_t *HIB,
                                   __m256i idxLo, __m256i idxHi, __m256i m8)
{
    __m256i lo_a = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)LOA), idxLo);
    __m256i lo_b = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)LOB), idxLo);
    __m256i hi_a = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)HIA), idxHi);
    __m256i hi_b = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)HIB), idxHi);
    __m256i lo = _mm256_blendv_epi8(lo_a, lo_b, m8);
    __m256i hi = _mm256_blendv_epi8(hi_a, hi_b, m8);
    return _mm256_castsi256_pd(_mm256_blend_epi32(lo, hi, 0xAA));
}

/* x = 2^e m, m in [T/2, T) for the table threshold T, SUBC = frac(T).  Subtracting
 * SUBC borrows out of the exponent field when the mantissa is below T; with the
 * exponent rebiased by 1022, mantissas at or above T come out halved and e raised by
 * one.  Exact, with no compare or blend. */
static inline void matern_log_reduce_borrow(__m256d x, long long SUBC,
                                        __m256d *m_out, __m256d *e_out)
{
    __m256i bits = _mm256_castpd_si256(x);
    __m256i t2   = _mm256_sub_epi64(bits, _mm256_set1_epi64x(SUBC));
    __m256i be   = _mm256_srli_epi64(t2, 52);
    *e_out = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(be,
                 _mm256_set1_epi64x(0x4330000000000000LL))),
                 _mm256_set1_pd(0x1.0p52 + 1022.0));
    *m_out = _mm256_castsi256_pd(_mm256_add_epi64(
                 _mm256_sub_epi64(bits, _mm256_and_si256(t2,
                     _mm256_set1_epi64x(0xFFF0000000000000LL))),
                 _mm256_set1_epi64x(0x3FE0000000000000LL)));
}

/* m in [1, 2), e the unbiased exponent as a double; bit-identical to
 * VGETMANT(_MM_MANT_NORM_1_2)/VGETEXP for finite positive normals. */
static inline void matern_log_reduce_normal(__m256d x, __m256d *m_out, __m256d *e_out)
{
    const __m256i frac    = _mm256_set1_epi64x(0x000fffffffffffffLL);
    const __m256i onebits = _mm256_set1_epi64x(0x3ff0000000000000LL);
    const __m256i magic   = _mm256_set1_epi64x(0x4330000000000000LL); /* 2^52 */
    __m256i ix = _mm256_castpd_si256(x);
    *m_out = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(ix, frac), onebits));
    __m256i ei = _mm256_srli_epi64(ix, 52);                 /* biased exp (x>0) */
    __m256d ed = _mm256_castsi256_pd(_mm256_or_si256(ei, magic)); /* 2^52 + biased */
    *e_out = _mm256_sub_pd(ed, _mm256_set1_pd(0x1.0p52 + 1023.0));
}

#endif /* MATERN_LOGARITHM_REDUCTION_AVX2_H */
