/* logarithm_avx512.h -- ln x for eight doubles: borrow reduction, 16-entry accurate table
 * (Gal), sigma = r/(2+r) with a degree-2 odd series, single-form recomposition; finite
 * normal x > 0, round-to-nearest. */
#ifndef MATERN_LOGARITHM_AVX512_H
#define MATERN_LOGARITHM_AVX512_H

#include <immintrin.h>

#if !defined(__AVX512F__) || !defined(__AVX512DQ__) || !defined(__FMA__)
#  error "logarithm_avx512.h needs AVX-512F, AVX-512DQ and FMA (-mavx512f -mavx512dq -mfma)."
#endif

/* x = 2^n m, m in [THRESH/2, THRESH) (exact borrow reduction), r = fma(m, c_j, -1),
 * sigma = r/(2+r), v = sigma^2, qq(v) ~ (2 atanh(sqrt v) - 2 sqrt v)/v^1.5,
 * log(x) = fma(-n, -LN2HI, A + B), A = r + sigma*(v*qq - r), B = fma(-n, -LN2LO, logc_j).
 * r leads and sigma only scales a correction, so one plain divide suffices; the largest
 * term is added last, with one rounding.  |r| <= 0.03016 permits degree 2.
 * The order is fixed: folding logc into the inner fma (fma(sigma, w, fma(-n, -LN2LO, logc)),
 * then + r) rounds the small sigma*w correction to logc's ulp; A combines the small
 * terms first and meets logc only in the final add.
 */

/* Degree-2 weighted-minimax qq(v), v in [0, 2.207e-4]. */
#define LOG_G5_QQ0 0x1.55555555557f6p-1
#define LOG_G5_QQ1 0x1.99999930f8dccp-2
#define LOG_G5_QQ2 0x1.24a5931e848b7p-2
/* LN2HI has 32 significant bits, so n*LN2HI is exact for |n| <= 1075. */
#define LOG_G5_LN2HI 0x1.62e42fee00000p-1
#define LOG_G5_LN2LO 0x1.a39ef35793c76p-33
/* bits(THRESH) - 1, THRESH = 0x1.6555555555556p+0  =>  m in [THRESH/2, THRESH) */
#define LOG_G5_OFFN  0x3FF6555555555555LL
/* bits(THRESH/2), the table index base */
#define LOG_G5_BASE  0x3FE6555555555556LL

static inline __m512d matern_logarithm_pd(__m512d x)
{
    const __m512d neg1 = _mm512_set1_pd(-1.0);
    const __m512d two  = _mm512_set1_pd(2.0);
    const __m512i base = _mm512_set1_epi64(LOG_G5_BASE);

    /* lo = buckets 0..7, hi = 8..15.  Bucket 9 is c = 1, logc = 0, so r is exact near
     * x = 1; c_j is chosen so that -log(c_j) is within about 2^-67 of a double. */
    const __m512d C_lo  = _mm512_setr_pd(0x1.66dc1a064aef0p+0, 0x1.57cbc2f6cb044p+0, 0x1.49f23241c41ecp+0, 0x1.3d2b498096841p+0, 0x1.31584df11803ep+0, 0x1.265ef05ad41adp+0, 0x1.1c2888a4bd60ep+0, 0x1.12a178f6bb705p+0);
    const __m512d C_hi  = _mm512_setr_pd(0x1.09b8af69ba7e6p+0, 0x1.0000000000000p+0, 0x1.e6de48066a58bp-1, 0x1.cb8b8e97829fap-1, 0x1.b3207789e8ce7p-1, 0x1.9d2c5f955aceap-1, 0x1.89544cafebaa1p-1, 0x1.774df5e0f8dcdp-1);
    const __m512d LC_lo = _mm512_setr_pd(-0x1.59dc40bffdf67p-2, -0x1.2df29fc2eb653p-2, -0x1.03d78316163c6p-2, -0x1.b6ccaf9eb5e35p-3, -0x1.68fd6fb8fc365p-3, -0x1.1e079a70103c0p-3, -0x1.ab6f437266f32p-4, -0x1.1fbe7f190542ap-4);
    const __m512d LC_hi = _mm512_setr_pd(-0x1.31530d08c0289p-5, 0x0.0p+0, 0x1.9c4fc07f27e26p-5, 0x1.bab9e58337c0bp-4, 0x1.4d2ee3308680bp-3, 0x1.b7357487f3e5ap-3, 0x1.0e0176feb050ap-2, 0x1.3e0a7a097f046p-2);

    /* Borrow reduction, centring folded into OFFN; nn = -n, so the ln2 constants are
     * negated below. */
    __m512i bits = _mm512_castpd_si512(x);
    __m512i t    = _mm512_sub_epi64(_mm512_set1_epi64(LOG_G5_OFFN), bits);
    __m512d m    = _mm512_castsi512_pd(_mm512_add_epi64(bits,
                       _mm512_and_si512(t, _mm512_set1_epi64(0xFFF0000000000000LL))));
    __m512d nn   = _mm512_cvtepi64_pd(_mm512_srai_epi64(t, 52));

    __m512i idx  = _mm512_srli_epi64(_mm512_sub_epi64(_mm512_castpd_si512(m), base), 48);
    __m512d c    = _mm512_permutex2var_pd(C_lo,  idx, C_hi);
    __m512d logc = _mm512_permutex2var_pd(LC_lo, idx, LC_hi);

    __m512d r = _mm512_fmadd_pd(m, c, neg1);              /* |r| <= 0.03016; exact in bucket 9 */
    __m512d s = _mm512_div_pd(r, _mm512_add_pd(two, r));   /* |sigma| <= 0.014856 */
    __m512d v = _mm512_mul_pd(s, s);                      /* v <= 2.21e-4 */

    __m512d q = _mm512_set1_pd(LOG_G5_QQ2);
    q = _mm512_fmadd_pd(q, v, _mm512_set1_pd(LOG_G5_QQ1));
    q = _mm512_fmadd_pd(q, v, _mm512_set1_pd(LOG_G5_QQ0));

    /* A: r leads, sigma scales the correction.  B: logc and the ln2 low word, both
     * <= 0.35 in magnitude.  Order as in the note above. */
    __m512d w = _mm512_fmsub_pd(v, q, r);                             /* v*qq - r */
    __m512d A = _mm512_fmadd_pd(s, w, r);                             /* log(1+r)  */
    __m512d B = _mm512_fmadd_pd(nn, _mm512_set1_pd(-LOG_G5_LN2LO), logc);
    return _mm512_fmadd_pd(nn, _mm512_set1_pd(-LOG_G5_LN2HI), _mm512_add_pd(A, B));
}


#endif /* MATERN_LOGARITHM_AVX512_H */
