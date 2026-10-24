/* simd_avx2.h -- AVX2 + FMA intrinsic layer, one macro per vector operation, four doubles
 * per vector; masks are full-width vectors. */
#ifndef MATERN_SIMD_AVX2_H
#define MATERN_SIMD_AVX2_H

/* `restrict` is C only; the headers are also compiled as C++, so it is defined away. */
#if defined(__cplusplus) && !defined(restrict)
#define restrict
#endif

#include <float.h>
#include <immintrin.h>

#define DATA_TYPE_FLOAT double
#define SIMD_TYPE_FLOAT __m256d
#define MASK_FLOAT __m256d
/* INDEX_FACTOR and WHILE_FACTOR as in simd_avx512.h: the lane count and broadcast stride,
 * and MASK_REDUCE of a mask with all four lanes set. */
#define INDEX_FACTOR 4
#define WHILE_FACTOR 15
#define LOAD_FLOAT(X) _mm256_loadu_pd((X))
#define SET_FLOAT(X) _mm256_set1_pd(X)
#define EQUAL_FLOAT(X, Y) _mm256_cmp_pd(X, Y, _CMP_EQ_OQ)
#define LESS_THAN_EQUAL_FLOAT(X, Y) _mm256_cmp_pd(X, Y, _CMP_LE_OQ)
#define LESS_THAN_FLOAT(X, Y) _mm256_cmp_pd(X, Y, _CMP_LT_OQ)
/* BLEND_FLOAT(mask, A, B) gives B where the mask lane is set and A elsewhere, the
 * argument order of the AVX-512 file; blendv's own order is (A, B, mask). */
#define BLEND_FLOAT(X, Y, Z) _mm256_blendv_pd(Y, Z, X)

#define ADD_FLOAT(X, Y) _mm256_add_pd(X, Y)

#define SUB_FLOAT(X, Y) _mm256_sub_pd(X, Y)

#define FMSUB_FLOAT(X, Y, Z) _mm256_fmsub_pd(X, Y, Z)
#define FMADD_FLOAT(X, Y, Z) _mm256_fmadd_pd(X, Y, Z)
#define FNMADD_FLOAT(X, Y, Z) _mm256_fnmadd_pd(X, Y, Z)
#define MUL_FLOAT(X, Y) _mm256_mul_pd(X, Y)
#define DIV_FLOAT(X, Y) _mm256_div_pd(X, Y)

#define FLOOR(X) _mm256_floor_pd(X)


#define SQRT_ROUND(X, Y) _mm256_sqrt_pd(X) 

/* Gathers the four lane sign bits into an integer, hence WHILE_FACTOR 15. */
#define MASK_REDUCE(X) _mm256_movemask_pd (X)
#define ABS(X) _mm256_max_pd((X), _mm256_mul_pd(_mm256_set1_pd(-1.0), (X)))

#endif /* MATERN_SIMD_AVX2_H */
