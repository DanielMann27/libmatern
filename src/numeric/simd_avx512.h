/* simd_avx512.h -- AVX-512 (F + DQ + FMA) intrinsic layer, one macro per vector
 * operation, eight doubles per vector; exactly one simd_*.h per translation unit
 * (vector_current.h). */
#ifndef MATERN_SIMD_AVX512_H
#define MATERN_SIMD_AVX512_H

/* `restrict` is C only; the headers are also compiled as C++, so it is defined away. */
#if defined(__cplusplus) && !defined(restrict)
#define restrict
#endif

#include <float.h>
#include <immintrin.h>


#define DATA_TYPE_FLOAT double
#define SIMD_TYPE_FLOAT __m512d
#define MASK_FLOAT __mmask8
/* INDEX_FACTOR is the lane count and the stride of every broadcast table (constant k at
 * offset k * INDEX_FACTOR).  WHILE_FACTOR is MASK_REDUCE of a mask with every lane set, so
 * `MASK_REDUCE(cond) == WHILE_FACTOR` is a test over all lanes, not an arithmetic test. */
#define INDEX_FACTOR 8
#define WHILE_FACTOR 255
#define LOAD_FLOAT(X) _mm512_loadu_pd(X)
#define SET_FLOAT(X) _mm512_set1_pd(X)
#define EQUAL_FLOAT(X, Y) _mm512_cmpeq_pd_mask(X, Y)
#define LESS_THAN_FLOAT(X, Y) _mm512_cmplt_pd_mask(X, Y)
#define LESS_THAN_EQUAL_FLOAT(X, Y) _mm512_cmple_pd_mask(X, Y)
/* BLEND_FLOAT(mask, A, B) gives B where the mask bit is set and A elsewhere, the reverse
 * of a C conditional (MASK_BLEND has the same order). */
#define BLEND_FLOAT(X, Y, Z) _mm512_mask_blend_pd(X, Y, Z)

#define ADD_FLOAT(X, Y) _mm512_add_pd(X, Y)

#define SUB_FLOAT(X, Y) _mm512_sub_pd(X, Y)

/* Fused forms, rounded once: FMADD a*b + c, FMSUB a*b - c, FNMADD c - a*b.  The single
 * rounding is required: FMSUB(a, b, a*b) recovers a product's rounding error exactly. */
#define FMSUB_FLOAT(X, Y, Z) _mm512_fmsub_pd(X, Y, Z)
#define FMADD_FLOAT(X, Y, Z) _mm512_fmadd_pd(X, Y, Z)
#define FNMADD_FLOAT(X, Y, Z) _mm512_fnmadd_pd(X, Y, Z)
#define MUL_FLOAT(X, Y) _mm512_mul_pd(X, Y)
#define DIV_FLOAT(X, Y) _mm512_div_pd(X, Y)


#define FLOOR(X) _mm512_roundscale_pd(X, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)

#define ABS(X) _mm512_abs_pd(X)


#define SQRT_ROUND(X, Y) _mm512_sqrt_round_pd(X, Y)

/* The mask as an integer whose bit k is lane k; compared against WHILE_FACTOR. */
#define MASK_REDUCE(X) _cvtmask8_u32(X)

#endif /* MATERN_SIMD_AVX512_H */
