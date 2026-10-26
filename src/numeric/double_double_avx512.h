/* double_double_avx512.h -- double-double primitives (hi + lo) for the AVX-512 kernels:
 * dd_two_sum (Knuth TwoSum) and dd_log for positive normal x (16-entry table reduction,
 * degree-9 polynomial for log(1+r), recomposition whose low word is not normalised).
 * dd_log is accurate to about 2^-58 absolute, not to double-double precision (see dd_log). */
#ifndef MATERN_DOUBLE_DOUBLE_AVX512_H
#define MATERN_DOUBLE_DOUBLE_AVX512_H

#include <immintrin.h>

/* (s,e) = TwoSum(a,b): s = fl(a+b), e = (a - (s - bb)) + (b - bb) with bb = s - a, so
 * a + b = s + e exactly.  The six-operation form is valid for either ordering of |a| and
 * |b| (callers do not guarantee |a| >= |b|); the operations must not be reassociated. */
static inline void dd_two_sum(__m512d a, __m512d b, __m512d *s, __m512d *e){
    __m512d ss = _mm512_add_pd(a, b);
    __m512d bb = _mm512_sub_pd(ss, a);
    *e = _mm512_add_pd(_mm512_sub_pd(a, _mm512_sub_pd(ss, bb)), _mm512_sub_pd(b, bb));
    *s = ss;
}

/* log(rh) as lh + ll, for positive normal rh.  Not a full double-double logarithm:
 * log(1+r) is a double polynomial and r is rounded, so lh + ll is accurate to about 2^-58
 * absolute; near rh = 1, where log rh is small, that is about 2^-52.6 relative, no better
 * than one double. */
static inline void dd_log(__m512d rh, __m512d *lh, __m512d *ll){
    const __m512d thresh = _mm512_set1_pd(0x1.6555555555556p+0);
    const __m512d half   = _mm512_set1_pd(0.5);
    const __m512d one    = _mm512_set1_pd(1.0);
    const __m512d neg1   = _mm512_set1_pd(-1.0);
    const __m512d ln2_hi = _mm512_set1_pd(0x1.62e42fee00000p-1);
    const __m512d ln2_lo = _mm512_set1_pd(0x1.a39ef35793c76p-33);
    const __m512i base   = _mm512_set1_epi64(0x3FE6555555555556LL);
    const __m512d C_lo   = _mm512_setr_pd(0x1.66dc1a064aef0p+0,0x1.57cbc2f6cb044p+0,0x1.49f23241c41ecp+0,0x1.3d2b498096841p+0,0x1.31584df11803ep+0,0x1.265ef05ad41adp+0,0x1.1c2888a4bd60ep+0,0x1.12a178f6bb705p+0);
    const __m512d C_hi   = _mm512_setr_pd(0x1.09b8af69ba7e6p+0,0x1.0000000000000p+0,0x1.e6de48066a58bp-1,0x1.cb8b8e97829fap-1,0x1.b3207789e8ce7p-1,0x1.9d2c5f955aceap-1,0x1.89544cafebaa1p-1,0x1.774df5e0f8dcdp-1);
    const __m512d LC_lo  = _mm512_setr_pd(-0x1.59dc40bffdf67p-2,-0x1.2df29fc2eb653p-2,-0x1.03d78316163c6p-2,-0x1.b6ccaf9eb5e35p-3,-0x1.68fd6fb8fc365p-3,-0x1.1e079a70103c0p-3,-0x1.ab6f437266f32p-4,-0x1.1fbe7f190542ap-4);
    const __m512d LC_hi  = _mm512_setr_pd(-0x1.31530d08c0289p-5,0x0.0p+0,0x1.9c4fc07f27e26p-5,0x1.bab9e58337c0bp-4,0x1.4d2ee3308680bp-3,0x1.b7357487f3e5ap-3,0x1.0e0176feb050ap-2,0x1.3e0a7a097f046p-2);

    /* x = 2^e m, m in [1,2) recentred to [thresh/2, thresh) so that r is small on both
     * sides of 1; halving m is exact. */
    __m512d m = _mm512_getmant_pd(rh, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    __m512d e = _mm512_getexp_pd(rh);
    const __mmask8 hm = _mm512_cmp_pd_mask(m, thresh, _CMP_GE_OQ);
    m = _mm512_mask_mul_pd(m, hm, m, half);
    e = _mm512_mask_add_pd(e, hm, e, one);
    /* Index from the leading mantissa bits into sixteen entries (two registers, two-source
     * permute).  Each c ~ 1/m is chosen so that log c is close to a double.  Entry 9 is
     * c = 1, log c = 0, so the reduction is exact for x near 1. */
    __m512i idx  = _mm512_srli_epi64(_mm512_sub_epi64(_mm512_castpd_si512(m), base), 48);
    __m512d c    = _mm512_permutex2var_pd(C_lo,  idx, C_hi);
    __m512d logc = _mm512_permutex2var_pd(LC_lo, idx, LC_hi);
    __m512d r  = _mm512_fmadd_pd(m, c, neg1);
    __m512d r2 = _mm512_mul_pd(r, r);
    __m512d B = _mm512_set1_pd( 0x1.c7ca050ba3963p-4);
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd(-0x1.005f77015466cp-3));
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd( 0x1.24923c827ddc1p-3));
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd(-0x1.5555476feb961p-3));
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd( 0x1.9999999a30370p-3));
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd(-0x1.0000000052d61p-2));
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd( 0x1.555555555554dp-2));
    B = _mm512_fmadd_pd(B, r, _mm512_set1_pd(-0x1.ffffffffffff6p-2));
    __m512d P = _mm512_fmadd_pd(r2, B, r);            /* ~log(1+r) */
    /* log = e*ln2_hi + e*ln2_lo + logc + P, smallest terms first so that each TwoSum
     * captures the rounding of the next addition; ln2 is split so that e*ln2_hi is exact
     * for every reachable e. */
    __m512d s, es; dd_two_sum(logc, P, &s, &es);
    __m512d lo = _mm512_fmadd_pd(e, ln2_lo, es);
    __m512d eh2 = _mm512_mul_pd(e, ln2_hi);           /* exact: ln2_hi has 32 significant bits, e small */
    __m512d hh, ee2; dd_two_sum(eh2, s, &hh, &ee2);
    *lh = hh;
    *ll = _mm512_add_pd(ee2, lo);
}

#endif /* MATERN_DOUBLE_DOUBLE_AVX512_H */
