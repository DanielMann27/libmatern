/* logarithm_avx2.h -- ln x for four doubles: borrow reduction to m in [1/sqrt2, sqrt2),
 * s = (m-1)/(m+1), degree-6 series in s^2 and a four-operation tail; finite normal
 * x > 0, round-to-nearest. */
#ifndef MATERN_LOGARITHM_AVX2_H
#define MATERN_LOGARITHM_AVX2_H

#include <immintrin.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "logarithm_avx2.h needs AVX2 and FMA (-mavx2 -mfma)."
#endif


/* x = 2^n m, m in [1/sqrt2, sqrt2); l = m - 1 (exact, Sterbenz); s = l/(m+1), u = s^2;
 * log(m) = 2 atanh(s) = l + s*(u*q - l).  Since l is exact and leads, s only scales a
 * correction and one plain divide suffices.  Tail, in this order:
 *     w = u*q - l;  A = fma(s, w, l) (l leads);  B = fma(n, LN2LO, A) (ln2 low word
 *     through the chain);  out = fma(n, LN2HI, B) (largest term last, one rounding).
 * LN2HI has 42 significant bits, so n*LN2HI is exact for |n| <= 1075.
 */

/* Degree-6 minimax q(u) = (2 atanh(sqrt u) - 2 sqrt u)/u^1.5 on u in [0, 0.0294373]. */
#define MATERN_LOG_G6_Q0 0x1.5555555555592p-1
#define MATERN_LOG_G6_Q1 0x1.999999997fee9p-2
#define MATERN_LOG_G6_Q2 0x1.24924941e0c27p-2
#define MATERN_LOG_G6_Q3 0x1.c71c52164ca89p-3
#define MATERN_LOG_G6_Q4 0x1.74663c5376b90p-3
#define MATERN_LOG_G6_Q5 0x1.39a1fb9d86af1p-3
#define MATERN_LOG_G6_Q6 0x1.2f02e5a54db10p-3
#define MATERN_LOG_G6_LN2HI 0x1.62e42fefa3800p-1
#define MATERN_LOG_G6_LN2LO 0x1.ef35793c76730p-45
/* bits(fl(sqrt2)) - bits(1.0): the borrow constant with the sqrt2 centring folded in.
 * x > 0, so the logical srli is safe (AVX2 has no srai_epi64). */
#define MATERN_LOG_G6_SUBC 0x0006A09E667F3BCDLL

static inline __m256d matern_logarithm_pd(__m256d x)
{
    const __m256d one = _mm256_set1_pd(1.0);

    /* m in [1/sqrt2, sqrt2); n, the unbiased exponent as a double, keeps its own sign,
     * so the ln2 constants are not negated (unlike logarithm_avx512.h). */
    __m256i bits = _mm256_castpd_si256(x);
    __m256i t2   = _mm256_sub_epi64(bits, _mm256_set1_epi64x(MATERN_LOG_G6_SUBC));
    __m256i be   = _mm256_srli_epi64(t2, 52);
    __m256d n    = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(be,
                       _mm256_set1_epi64x(0x4330000000000000LL))),
                       _mm256_set1_pd(0x1.0p52 + 1022.0));
    __m256d m = _mm256_castsi256_pd(_mm256_add_epi64(
                    _mm256_sub_epi64(bits, _mm256_and_si256(t2,
                        _mm256_set1_epi64x(0xFFF0000000000000LL))),
                    _mm256_set1_epi64x(0x3FE0000000000000LL)));

    __m256d l = _mm256_sub_pd(m, one);                            /* exact (Sterbenz) */
    __m256d s = _mm256_div_pd(l, _mm256_add_pd(m, one));           /* |s| <= 0.17157  */
    __m256d u = _mm256_mul_pd(s, s);                               /* u  <= 0.029437  */

    __m256d q = _mm256_set1_pd(MATERN_LOG_G6_Q6);
    q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(MATERN_LOG_G6_Q5));
    q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(MATERN_LOG_G6_Q4));
    q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(MATERN_LOG_G6_Q3));
    q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(MATERN_LOG_G6_Q2));
    q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(MATERN_LOG_G6_Q1));
    q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(MATERN_LOG_G6_Q0));

    /* Four-operation tail, in the order of the note above. */
    __m256d w = _mm256_fmsub_pd(u, q, l);                          /* u*q - l */
    __m256d A = _mm256_fmadd_pd(s, w, l);                          /* log(m) core */
    __m256d B = _mm256_fmadd_pd(n, _mm256_set1_pd(MATERN_LOG_G6_LN2LO), A);
    return _mm256_fmadd_pd(n, _mm256_set1_pd(MATERN_LOG_G6_LN2HI), B);
}

#endif /* MATERN_LOGARITHM_AVX2_H */
