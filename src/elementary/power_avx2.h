/* power_avx2.h -- x^y for four doubles by the method of power_avx512.h, with tables as
 * 32-bit halves and e + lc2 formed with an explicit error term (wlo); finite normal
 * x > 0 with a finite normal result. */
#ifndef MATERN_POWER_AVX2_H
#define MATERN_POWER_AVX2_H

#include <stdint.h>
#include "logarithm_reduction_avx2.h"   /* matern_log_table_16 */
#include "power_coefficients_avx2.h"

/* 8-entry table read (j in [0,7]): one VPERMD per 32-bit half, joined by blend 0xAA. */
static inline __m256d matern_power_table_8(const uint32_t *LO32, const uint32_t *HI32,
                                  __m256i idxLo, __m256i idxHi)
{
    __m256i lo = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)LO32), idxLo);
    __m256i hi = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)HI32), idxHi);
    return _mm256_castsi256_pd(_mm256_blend_epi32(lo, hi, 0xAA));
}

#define MATERN_POWER_THRESH_SUBC 0x0006555555555556LL   /* bits(0x1.6555555555556p+0) - bits(1.0) */

/* log2 x = e + lc2 + log2(1 + r) with m c = 1 + r, lc2 = -log2 c from the table and the
 * series r log2(e) + r^2 B(r), as the unnormalised pair (HI, LO); z = y log2 x as a
 * pair; 2^z with z = q/8 + g, q = n8 + j: 2^(j/8) from the table, 2^g from a polynomial,
 * 2^(n8/8) added to the exponent field as n8 << 49 == (n8/8) << 52. */
static inline __attribute__((always_inline)) __m256d matern_power_pd(__m256d x, __m256d y)
{
    /* log2 x as a double-double. */
    const __m256d one    = _mm256_set1_pd(1.0);
    const __m256d LE_hi  = _mm256_set1_pd(0x1.71547652b82fep+0);   /* log2(e) hi */
    const __m256d LE_lo  = _mm256_set1_pd(0x1.777d0ffda0d24p-56);  /* log2(e) lo */
    const __m256i base   = _mm256_set1_epi64x(0x3FE6555555555556LL);

    __m256d m, e;
    /* Borrow reduction with the THRESH centring folded in: m in [THRESH/2, THRESH). */
    matern_log_reduce_borrow(x, MATERN_POWER_THRESH_SUBC, &m, &e);

    __m256i j     = _mm256_srli_epi64(_mm256_sub_epi64(_mm256_castpd_si256(m), base), 48);
    __m256i idxLo = j;
    __m256i idxHi = _mm256_slli_epi64(j, 32);
    __m256i m8    = _mm256_cmpgt_epi64(j, _mm256_set1_epi64x(7));
    __m256d c   = matern_log_table_16(MATERN_POWER_C_LO32A,   MATERN_POWER_C_LO32B,   MATERN_POWER_C_HI32A,   MATERN_POWER_C_HI32B,   idxLo, idxHi, m8);
    __m256d lc2 = matern_log_table_16(MATERN_POWER_LC2_LO32A, MATERN_POWER_LC2_LO32B, MATERN_POWER_LC2_HI32A, MATERN_POWER_LC2_HI32B, idxLo, idxHi, m8);

    __m256d p   = _mm256_mul_pd(m, c);
    __m256d rhi = _mm256_sub_pd(p, one);
    __m256d rlo = _mm256_fmsub_pd(m, c, p);

    __m256d r2 = _mm256_mul_pd(rhi, rhi);
    __m256d B = _mm256_set1_pd( 0x1.48c8491ceb6cdp-3);
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd(-0x1.71de305ad0c77p-3));
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd( 0x1.a617506de3767p-3));
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd(-0x1.ec7089b75d1fap-3));
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd( 0x1.2776c50f66653p-2));
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd(-0x1.715476532fb1dp-2));
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd( 0x1.ec709dc3a03f1p-2));
    B = _mm256_fmadd_pd(B, rhi, _mm256_set1_pd(-0x1.71547652b82f7p-1));
    __m256d Pcorr = _mm256_mul_pd(r2, B);
    __m256d cross = _mm256_mul_pd(rhi, rlo);

    __m256d g2hi = _mm256_mul_pd(rhi, LE_hi);
    __m256d g2lo = _mm256_fmadd_pd(rhi, LE_lo, _mm256_fmsub_pd(rhi, LE_hi, g2hi));

    __m256d whi  = _mm256_add_pd(e, lc2);
    __m256d wlo  = _mm256_add_pd(_mm256_sub_pd(e, whi), lc2);

    __m256d T2   = _mm256_add_pd(whi, g2hi);
    __m256d lo_a = _mm256_add_pd(_mm256_sub_pd(whi, T2), g2hi);
    __m256d HI   = _mm256_add_pd(T2, Pcorr);
    __m256d lo_b = _mm256_add_pd(_mm256_sub_pd(T2, HI), Pcorr);

    __m256d LO = _mm256_add_pd(wlo, lo_a);
    LO = _mm256_add_pd(LO, lo_b);
    LO = _mm256_add_pd(LO, g2lo);
    LO = _mm256_fmadd_pd(LE_hi, rlo, LO);
    LO = _mm256_fnmadd_pd(LE_hi, cross, LO);

    /* y-multiply into the unnormalised (HI, LO). */
    __m256d zhi = _mm256_mul_pd(y, HI);
    __m256d zlo = _mm256_fmadd_pd(y, LO, _mm256_fmsub_pd(y, HI, zhi));

    /* 2^z by a reduction to q/8 + g. */
    const __m256d SHIFT = _mm256_set1_pd(0x1.8p52);
    const __m256i BIAS  = _mm256_set1_epi64x(0x4338000000000000LL);
    __m256d s  = _mm256_fmadd_pd(zhi, _mm256_set1_pd(8.0), SHIFT);
    __m256d qd = _mm256_sub_pd(s, SHIFT);
    __m256i q  = _mm256_sub_epi64(_mm256_castpd_si256(s), BIAS);
    __m256i j8 = _mm256_and_si256(q, _mm256_set1_epi64x(7));
    __m256i n8 = _mm256_and_si256(q, _mm256_set1_epi64x(~7LL));

    __m256d g = _mm256_fnmadd_pd(qd, _mm256_set1_pd(0.125), zhi);
    g         = _mm256_add_pd(g, zlo);

    __m256i jHi = _mm256_slli_epi64(j8, 32);
    __m256d th = matern_power_table_8(MATERN_POWER_TH8_LO32, MATERN_POWER_TH8_HI32, j8, jHi);
    __m256d tl = matern_power_table_8(MATERN_POWER_TL8_LO32, MATERN_POWER_TL8_HI32, j8, jHi);

    __m256d w = _mm256_set1_pd(0x1.ffd1f6633bd3ap-17);
    w = _mm256_fmadd_pd(w, g, _mm256_set1_pd(0x1.430deca12d584p-13));
    w = _mm256_fmadd_pd(w, g, _mm256_set1_pd(0x1.5d87fe72ac65ep-10));
    w = _mm256_fmadd_pd(w, g, _mm256_set1_pd(0x1.3b2ab6f1f1910p-7));
    w = _mm256_fmadd_pd(w, g, _mm256_set1_pd(0x1.c6b08d704a11fp-5));
    w = _mm256_fmadd_pd(w, g, _mm256_set1_pd(0x1.ebfbdff82c6c5p-3));
    w = _mm256_fmadd_pd(w, g, _mm256_set1_pd(0x1.62e42fefa39efp-1));

    __m256d u   = _mm256_mul_pd(g, w);
    __m256d tmp = _mm256_fmadd_pd(th, u, tl);
    __m256d yv  = _mm256_add_pd(th, tmp);

    __m256i bits = _mm256_add_epi64(_mm256_castpd_si256(yv), _mm256_slli_epi64(n8, 49));
    return _mm256_castsi256_pd(bits);
}

#endif /* MATERN_POWER_AVX2_H */
