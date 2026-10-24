/* vector_ops_avx512.h -- arithmetic vocabulary of the region kernels, 8 doubles per
 * vector.  VEC_FMA a*b + c, VEC_FMS a*b - c and VEC_FNMA c - a*b round once, which the
 * double-double residuals rely on. */
#ifndef MATERN_VECTOR_OPS_AVX512_H
#define MATERN_VECTOR_OPS_AVX512_H

#include <immintrin.h>

#define VEC_VLD   _mm512_loadu_pd
#define VEC_VST   _mm512_storeu_pd
#define VEC_VSET  _mm512_set1_pd
#define VEC_MUL   _mm512_mul_pd
#define VEC_ADD   _mm512_add_pd
#define VEC_SUB   _mm512_sub_pd
#define VEC_DIV   _mm512_div_pd
#define VEC_FMA   _mm512_fmadd_pd
#define VEC_FNMA  _mm512_fnmadd_pd
#define VEC_FMS   _mm512_fmsub_pd
/* Sites whose operands can cancel: the plain names here; they differ only at width 1
 * without -mfma (fused_multiply_add_scalar.h). */
#define VEC_FMA_CANCEL   VEC_FMA
#define VEC_FNMA_CANCEL  VEC_FNMA
#define VEC_FMS_CANCEL   VEC_FMS
#define fma_cancel(A, B, C) fma((A), (B), (C))
#define VEC_SQRT  _mm512_sqrt_pd
#define VEC_MIN   _mm512_min_pd
#define VEC_MAX   _mm512_max_pd
#define VEC_ABS(X) _mm512_abs_pd(X)
#define VEC_LANES 8

/* The lane-0 value, used where one point is evaluated in a whole vector. */
static inline double vec_first_lane(__m512d v) { return _mm512_cvtsd_f64(v); }

/* A where CMPA < CMPB, B elsewhere, lane by lane. */
#define VEC_SELECT_LT(A, B, CMPA, CMPB) \
    _mm512_mask_blend_pd(_mm512_cmp_pd_mask((CMPA),(CMPB),_CMP_LT_OQ),(B),(A))

/* x/d by a 14-bit reciprocal estimate, one Newton step and the residual correction
 * q + (x - d q) y; not always correctly rounded, unlike the AVX2 and width-1 divide. */
static inline __attribute__((always_inline)) __m512d vec_divide_refined(__m512d x, __m512d d){
    __m512d y = _mm512_rcp14_pd(d);
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, _mm512_set1_pd(2.0)));
    __m512d q = _mm512_mul_pd(x, y);
    __m512d r = _mm512_fnmadd_pd(d, q, x);
    return _mm512_fmadd_pd(r, y, q);
}

/* The largest value across the lanes, for trip counts set by the worst lane. */
static inline double vec_reduce_max(__m512d d)
{
    return _mm512_reduce_max_pd(d);
}

#endif /* MATERN_VECTOR_OPS_AVX512_H */
