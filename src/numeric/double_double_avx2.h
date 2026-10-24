/* double_double_avx2.h -- dd_two_sum and dd_log for four doubles, with the coefficients
 * and operation order of double_double_avx512.h; the dd_log tables are stored as 32-bit
 * halves. */
#ifndef MATERN_DOUBLE_DOUBLE_AVX2_H
#define MATERN_DOUBLE_DOUBLE_AVX2_H

#include <immintrin.h>
#include "logarithm_reduction_avx2.h"   /* reduction and table read */

/* dd_log's 16-entry c and log c tables as 32-bit halves, reassembled by two 32-bit
 * permutes (matern_log_table_16).  Same values as the AVX-512 tables. */
static const uint32_t DD_LOG_C_LO32A[8] = { 0xa064aef0u, 0x2f6cb044u, 0x241c41ecu, 0x98096841u, 0xdf11803eu, 0x05ad41adu, 0x8a4bd60eu, 0x8f6bb705u };
static const uint32_t DD_LOG_C_LO32B[8] = { 0xf69ba7e6u, 0x00000000u, 0x8066a58bu, 0xe97829fau, 0x789e8ce7u, 0xf955aceau, 0xcafebaa1u, 0x5e0f8dcdu };
static const uint32_t DD_LOG_C_HI32A[8] = { 0x3ff66dc1u, 0x3ff57cbcu, 0x3ff49f23u, 0x3ff3d2b4u, 0x3ff31584u, 0x3ff265efu, 0x3ff1c288u, 0x3ff12a17u };
static const uint32_t DD_LOG_C_HI32B[8] = { 0x3ff09b8au, 0x3ff00000u, 0x3fee6de4u, 0x3fecb8b8u, 0x3feb3207u, 0x3fe9d2c5u, 0x3fe89544u, 0x3fe774dfu };
static const uint32_t DD_LOG_LOGC_LO32A[8] = { 0x0bffdf67u, 0xfc2eb653u, 0x316163c6u, 0xf9eb5e35u, 0xfb8fc365u, 0xa70103c0u, 0x37266f32u, 0xf190542au };
static const uint32_t DD_LOG_LOGC_LO32B[8] = { 0xd08c0289u, 0x00000000u, 0x07f27e26u, 0x58337c0bu, 0x3308680bu, 0x487f3e5au, 0x6feb050au, 0xa097f046u };
static const uint32_t DD_LOG_LOGC_HI32A[8] = { 0xbfd59dc4u, 0xbfd2df29u, 0xbfd03d78u, 0xbfcb6ccau, 0xbfc68fd6u, 0xbfc1e079u, 0xbfbab6f4u, 0xbfb1fbe7u };
static const uint32_t DD_LOG_LOGC_HI32B[8] = { 0xbfa31530u, 0x00000000u, 0x3fa9c4fcu, 0x3fbbab9eu, 0x3fc4d2eeu, 0x3fcb7357u, 0x3fd0e017u, 0x3fd3e0a7u };


/* (s,e) = TwoSum(a,b): a + b = s + e exactly, for either ordering of |a| and |b|.
 * The operations must not be reassociated. */
static inline void dd_two_sum(__m256d a, __m256d b, __m256d *s, __m256d *e){
    __m256d ss = _mm256_add_pd(a, b);
    __m256d bb = _mm256_sub_pd(ss, a);
    *e = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(ss, bb)), _mm256_sub_pd(b, bb));
    *s = ss;
}

/* log(rh) as lh + ll, for positive normal rh; same coefficients and operation order as
 * the AVX-512 version. */
static inline void dd_log(__m256d rh, __m256d *lh, __m256d *ll){
    const __m256d thresh = _mm256_set1_pd(0x1.6555555555556p+0);
    const __m256d one  = _mm256_set1_pd(1.0), half = _mm256_set1_pd(0.5), neg1 = _mm256_set1_pd(-1.0);
    const __m256d ln2_hi = _mm256_set1_pd(0x1.62e42fee00000p-1);
    const __m256d ln2_lo = _mm256_set1_pd(0x1.a39ef35793c76p-33);
    const __m256i base   = _mm256_set1_epi64x(0x3FE6555555555556LL);
    __m256d m, e; matern_log_reduce_normal(rh, &m, &e);
    __m256d ge = _mm256_cmp_pd(m, thresh, _CMP_GE_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), ge);
    e = _mm256_blendv_pd(e, _mm256_add_pd(e, one),  ge);
    __m256i j     = _mm256_srli_epi64(_mm256_sub_epi64(_mm256_castpd_si256(m), base), 48);
    __m256i idxLo = j;
    __m256i idxHi = _mm256_slli_epi64(j, 32);
    __m256i m8    = _mm256_cmpgt_epi64(j, _mm256_set1_epi64x(7));
    __m256d c    = matern_log_table_16(DD_LOG_C_LO32A,    DD_LOG_C_LO32B,    DD_LOG_C_HI32A,    DD_LOG_C_HI32B,    idxLo, idxHi, m8);
    __m256d logc = matern_log_table_16(DD_LOG_LOGC_LO32A, DD_LOG_LOGC_LO32B, DD_LOG_LOGC_HI32A, DD_LOG_LOGC_HI32B, idxLo, idxHi, m8);
    __m256d r  = _mm256_fmadd_pd(m, c, neg1);
    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d B = _mm256_set1_pd( 0x1.c7ca050ba3963p-4);
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd(-0x1.005f77015466cp-3));
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd( 0x1.24923c827ddc1p-3));
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd(-0x1.5555476feb961p-3));
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd( 0x1.9999999a30370p-3));
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd(-0x1.0000000052d61p-2));
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd( 0x1.555555555554dp-2));
    B = _mm256_fmadd_pd(B, r, _mm256_set1_pd(-0x1.ffffffffffff6p-2));
    __m256d P = _mm256_fmadd_pd(r2, B, r);            /* ~log(1+r) */
    /* recomposition, smallest terms first: log = e*ln2_hi + e*ln2_lo + logc + P,
     * with e*ln2_hi exact */
    __m256d s, es; dd_two_sum(logc, P, &s, &es);
    __m256d lo  = _mm256_fmadd_pd(e, ln2_lo, es);
    __m256d eh2 = _mm256_mul_pd(e, ln2_hi);
    __m256d hh, ee2; dd_two_sum(eh2, s, &hh, &ee2);
    *lh = hh;
    *ll = _mm256_add_pd(ee2, lo);
}

#endif /* MATERN_DOUBLE_DOUBLE_AVX2_H */
