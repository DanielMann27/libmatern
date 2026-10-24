/* power_avx512.h -- x^y for eight doubles as 2^(y log2 x): double-double log2 (a relative
 * error eps in log2 x becomes |y log2 x| ln2 eps in x^y), double-double product with y,
 * 2^(j/8) table and polynomial; finite normal x > 0 with a finite normal result. */
#ifndef MATERN_POWER_AVX512_H
#define MATERN_POWER_AVX512_H

#include "exponential_avx512.h"   /* matern_exp_scale_by_power_of_two */
#include "power_coefficients_avx512.h"

/* THRESH = 0x1.6555555555556p+0 is the recentring point of the reduction. */
#define MATERN_POWER_BASE    0x3FE6555555555556LL   /* bits(THRESH/2), the index origin */
/* frac(THRESH) = bits(THRESH) - bits(1.0), the borrow constant: the subtraction borrows
 * out of the exponent field when the mantissa is below THRESH; with the exponent
 * rebiased by 1022, mantissas at or above THRESH come out halved and n raised by one. */
#define MATERN_POWER_SUBC    0x0006555555555556LL
#define MATERN_POWER_LE_HI   0x1.71547652b82fep+0   /* log2(e) = 1/ln2, hi */
#define MATERN_POWER_LE_LO   0x1.777d0ffda0d24p-56  /* log2(e), lo */

/* exp2: w(g) = (2^g - 1)/g, degree 6, |g| <= 1/16. */
#define MATERN_POWER_E1      0x1.62e42fefa39efp-1   /* ln2 */
#define MATERN_POWER_E2      0x1.ebfbdff82c6c5p-3
#define MATERN_POWER_E3      0x1.c6b08d704a11fp-5
#define MATERN_POWER_E4      0x1.3b2ab6f1f1910p-7
#define MATERN_POWER_E5      0x1.5d87fe72ac65ep-10
#define MATERN_POWER_E6      0x1.430deca12d584p-13
#define MATERN_POWER_E7      0x1.ffd1f6633bd3ap-17

/* 2^(j/8) double-double; same values as MATERN_EXP_TH/TL in exponential_avx512.h. */
static const double MATERN_POWER_TH[8] __attribute__((aligned(64))) = {
    0x1.0000000000000p+0, 0x1.172b83c7d517bp+0, 0x1.306fe0a31b715p+0, 0x1.4bfdad5362a27p+0,
    0x1.6a09e667f3bcdp+0, 0x1.8ace5422aa0dbp+0, 0x1.ae89f995ad3adp+0, 0x1.d5818dcfba487p+0
};
static const double MATERN_POWER_TL[8] __attribute__((aligned(64))) = {
    0x0.0p+0, -0x1.19041b9d78a76p-55, 0x1.6f46ad23182e4p-55, 0x1.d4397afec42e2p-56,
    -0x1.bdd3413b26456p-54, 0x1.6e9f156864b27p-54, 0x1.7a1cd345dcc81p-54, 0x1.2ed02d75b3707p-55
};

static inline __m512d matern_power_pd(__m512d x, __m512d y)
{
    /* log2 x as the unnormalised double-double (HI, LO). */
    const __m512d one    = _mm512_set1_pd(1.0);
    const __m512d LE_hi  = _mm512_set1_pd(MATERN_POWER_LE_HI);
    const __m512d LE_lo  = _mm512_set1_pd(MATERN_POWER_LE_LO);
    const __m512i lbase  = _mm512_set1_epi64(MATERN_POWER_BASE);

    /* lo = buckets 0..7, hi = 8..15; bucket 9 (m = 1) is c = 1, lc2 = 0, and
     * lc2 = -log2(c) lies on a 2^-42 grid. */
    const __m512d C_lo  = _mm512_load_pd(MATERN_POWER_C);
    const __m512d C_hi  = _mm512_load_pd(MATERN_POWER_C + 8);
    const __m512d LC2_lo = _mm512_load_pd(MATERN_POWER_LC2);
    const __m512d LC2_hi = _mm512_load_pd(MATERN_POWER_LC2 + 8);

    /* Borrow reduction: m in [THRESH/2, THRESH), n = be - 1022, no recentring step. */
    __m512i bits = _mm512_castpd_si512(x);
    __m512i t2   = _mm512_sub_epi64(bits, _mm512_set1_epi64(MATERN_POWER_SUBC));
    __m512i be   = _mm512_srli_epi64(t2, 52);          /* biased exp, THRESH-centred */
    __m512d e    = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(be,
                       _mm512_set1_epi64(0x4330000000000000LL))),
                       _mm512_set1_pd(0x1.0p52 + 1022.0));   /* = n, exact integer */
    __m512d m    = _mm512_castsi512_pd(_mm512_add_epi64(
                       _mm512_sub_epi64(bits, _mm512_and_si512(t2,
                           _mm512_set1_epi64(0xFFF0000000000000LL))),
                       _mm512_set1_epi64(0x3FE0000000000000LL)));

    /* m spans 2^52 in bit space over [THRESH/2, THRESH), so the top four fraction
     * bits above the base give the bucket. */
    __m512i mb   = _mm512_castpd_si512(m);
    __m512i idx  = _mm512_srli_epi64(_mm512_sub_epi64(mb, lbase), 48);
    __m512d c    = _mm512_permutex2var_pd(C_lo,   idx, C_hi);
    __m512d lc2  = _mm512_permutex2var_pd(LC2_lo, idx, LC2_hi);

    /* m*c - 1 as the exact pair (rhi, rlo). */
    __m512d p   = _mm512_mul_pd(m, c);
    __m512d rhi = _mm512_sub_pd(p, one);              /* exact (Sterbenz) */
    __m512d rlo = _mm512_fmsub_pd(m, c, p);           /* exact product tail: m*c = p + rlo */

    /* log2(1+r) = LOG2E*rhi (dd) + rhi^2*B2(rhi) + LOG2E*rlo*(1-rhi).
     * B2 is a degree-7 near-minimax of (log2(1+r) - LOG2E*r)/r^2 on |r| <= 0.0302. */
    __m512d r2 = _mm512_mul_pd(rhi, rhi);
    __m512d B = _mm512_set1_pd( 0x1.48c8491ceb6cdp-3);
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd(-0x1.71de305ad0c77p-3));
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd( 0x1.a617506de3767p-3));
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd(-0x1.ec7089b75d1fap-3));
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd( 0x1.2776c50f66653p-2));
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd(-0x1.715476532fb1dp-2));
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd( 0x1.ec709dc3a03f1p-2));
    B = _mm512_fmadd_pd(B, rhi, _mm512_set1_pd(-0x1.71547652b82f7p-1));
    __m512d Pcorr = _mm512_mul_pd(r2, B);             /* rhi^2 * B2 */

    /* rlo*(1 - rhi) in one rounding of about 2^-106 absolute, since |rlo| <= 2^-53. */
    __m512d rt = _mm512_fnmadd_pd(rhi, rlo, rlo);     /* rlo - rhi*rlo */

    /* LOG2E*rhi as a double-double (two-product), for precision near x = 1. */
    __m512d g2hi = _mm512_mul_pd(rhi, LE_hi);
    __m512d g2lo = _mm512_fmadd_pd(rhi, LE_lo, _mm512_fmsub_pd(rhi, LE_hi, g2hi));

    /* e + lc2 is exact by table construction: lc2 is a multiple of 2^-42
     * (power_coefficients_avx512.h) and e an integer with |e| <= 1074, so the sum needs
     * bits 2^10..2^-42, 53 in all, and no error term. */
    __m512d whi  = _mm512_add_pd(e, lc2);

    /* HI = whi + g2hi + Pcorr (two fast2sum steps); LO gathers the rest. */
    __m512d T2   = _mm512_add_pd(whi, g2hi);
    __m512d lo_a = _mm512_add_pd(_mm512_sub_pd(whi, T2), g2hi);    /* fast2sum tail (exact) */
    __m512d HI   = _mm512_add_pd(T2, Pcorr);
    __m512d lo_b = _mm512_add_pd(_mm512_sub_pd(T2, HI), Pcorr);    /* fast2sum tail */

    __m512d LO = lo_a;                                /* e + lc2 exact: no tail */
    LO = _mm512_add_pd(LO, lo_b);
    LO = _mm512_add_pd(LO, g2lo);
    LO = _mm512_fmadd_pd(LE_hi, rt, LO);              /* + LOG2E*rlo*(1-rhi) */

    /* zhi + zlo = y*(HI + LO) to about 2^-92 relative: the inner fmsub is the exact
     * tail of y*HI, y*LO the correction. */
    __m512d zhi = _mm512_mul_pd(y, HI);
    __m512d zlo = _mm512_fmadd_pd(y, LO, _mm512_fmsub_pd(y, HI, zhi));

    /* 2^z by a reduction to q/8 + g, without Cody-Waite (zhi - q/8 is exact). */
    const __m512d SHIFT = _mm512_set1_pd(0x1.8p52);
    const __m512i BIAS  = _mm512_set1_epi64(0x4338000000000000LL);
    __m512d s  = _mm512_fmadd_pd(zhi, _mm512_set1_pd(8.0), SHIFT);
    __m512i qi = _mm512_castpd_si512(s);
    __m512d qd = _mm512_sub_pd(s, SHIFT);             /* round(zhi*8) */
    __m512i j  = _mm512_and_si512(qi, _mm512_set1_epi64(7));
    __m512i n  = _mm512_srai_epi64(_mm512_sub_epi64(qi, BIAS), 3);

    __m512d g = _mm512_fnmadd_pd(qd, _mm512_set1_pd(0.125), zhi);  /* zhi - q/8, exact */
    g         = _mm512_add_pd(g, zlo);                /* fold the double-double tail */

    __m512d th = _mm512_permutexvar_pd(j, _mm512_load_pd(MATERN_POWER_TH));
    __m512d tl = _mm512_permutexvar_pd(j, _mm512_load_pd(MATERN_POWER_TL));

    __m512d w = _mm512_set1_pd(MATERN_POWER_E7);
    w = _mm512_fmadd_pd(w, g, _mm512_set1_pd(MATERN_POWER_E6));
    w = _mm512_fmadd_pd(w, g, _mm512_set1_pd(MATERN_POWER_E5));
    w = _mm512_fmadd_pd(w, g, _mm512_set1_pd(MATERN_POWER_E4));
    w = _mm512_fmadd_pd(w, g, _mm512_set1_pd(MATERN_POWER_E3));
    w = _mm512_fmadd_pd(w, g, _mm512_set1_pd(MATERN_POWER_E2));
    w = _mm512_fmadd_pd(w, g, _mm512_set1_pd(MATERN_POWER_E1));

    __m512d u   = _mm512_mul_pd(g, w);
    __m512d tmp = _mm512_fmadd_pd(th, u, tl);
    __m512d yv  = _mm512_add_pd(th, tmp);

    return matern_exp_scale_by_power_of_two(yv, n);
}

#endif /* MATERN_POWER_AVX512_H */
