/* vector_ops_avx2.h -- arithmetic vocabulary of the region kernels, 4 doubles per vector.
 * VEC_FMA a*b + c, VEC_FMS a*b - c and VEC_FNMA c - a*b round once, which the
 * double-double residuals rely on. */
#ifndef MATERN_VECTOR_OPS_AVX2_H
#define MATERN_VECTOR_OPS_AVX2_H

#include <immintrin.h>

#define VEC_VLD   _mm256_loadu_pd
#define VEC_VST   _mm256_storeu_pd
#define VEC_VSET  _mm256_set1_pd
#define VEC_MUL   _mm256_mul_pd
#define VEC_ADD   _mm256_add_pd
#define VEC_SUB   _mm256_sub_pd
#define VEC_DIV   _mm256_div_pd
#define VEC_FMA   _mm256_fmadd_pd
#define VEC_FNMA  _mm256_fnmadd_pd
#define VEC_FMS   _mm256_fmsub_pd
/* Sites whose operands can cancel: the plain names here; they differ only at width 1
 * without -mfma (fused_multiply_add_scalar.h). */
#define VEC_FMA_CANCEL   VEC_FMA
#define VEC_FNMA_CANCEL  VEC_FNMA
#define VEC_FMS_CANCEL   VEC_FMS
#define fma_cancel(A, B, C) fma((A), (B), (C))
#define VEC_SQRT  _mm256_sqrt_pd
#define VEC_MIN   _mm256_min_pd
#define VEC_MAX   _mm256_max_pd
/* |x|, by clearing the sign bit (exact). */
#define VEC_ABS(X) _mm256_andnot_pd(_mm256_set1_pd(-0.0), (X))
#define VEC_LANES 4

/* The lane-0 value, used where one point is evaluated in a whole vector. */
static inline double vec_first_lane(__m256d v) { return _mm256_cvtsd_f64(v); }

/* A where CMPA < CMPB, B elsewhere, lane by lane. */
#define VEC_SELECT_LT(A, B, CMPA, CMPB) \
    _mm256_blendv_pd((B),(A),_mm256_cmp_pd((CMPA),(CMPB),_CMP_LT_OQ))

/* x/d by the plain, correctly rounded divide (AVX-512 uses a refined reciprocal). */
static inline __attribute__((always_inline)) __m256d vec_divide_refined(__m256d x, __m256d d){
    return _mm256_div_pd(x, d);
}

/* The largest value across the lanes, for trip counts set by the worst lane. */
static inline double vec_reduce_max(__m256d d)
{
    __m128d lo = _mm256_castpd256_pd128(d), hi = _mm256_extractf128_pd(d, 1);
    __m128d m  = _mm_max_pd(lo, hi);
    m = _mm_max_sd(m, _mm_unpackhi_pd(m, m));
    return _mm_cvtsd_f64(m);
}

#endif /* MATERN_VECTOR_OPS_AVX2_H */
