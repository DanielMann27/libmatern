/* vector_masks_avx2.h -- per-lane predicates as full-width vectors of all-ones or
 * all-zeros lanes; MASK_BLEND(m, e, t) gives t where m is set, e elsewhere.  Comparisons
 * are ordered and quiet, so a NaN lane compares false. */
#ifndef MATERN_VECTOR_MASKS_AVX2_H
#define MATERN_VECTOR_MASKS_AVX2_H

#include <immintrin.h>

typedef __m256d vec_mask;
#define MASK_AND(A,B) _mm256_and_pd(A,B)
#define MASK_OR(A,B)  _mm256_or_pd(A,B)
#define MASK_NOT(A)   _mm256_xor_pd(A,_mm256_castsi256_pd(_mm256_set1_epi64x(-1)))
#define MASK_GREATER(A,B)  _mm256_cmp_pd(A,B,_CMP_GT_OQ)
#define MASK_LESS(A,B)  _mm256_cmp_pd(A,B,_CMP_LT_OQ)
#define MASK_LESS_EQUAL(A,B)  _mm256_cmp_pd(A,B,_CMP_LE_OQ)
#define MASK_EQUAL(A,B)  _mm256_cmp_pd(A,B,_CMP_EQ_OQ)
#define MASK_ANY(M)   (_mm256_movemask_pd(M)!=0)
#define MASK_COUNT(M) ((size_t)__builtin_popcount((unsigned)_mm256_movemask_pd(M)))   /* lanes set */
#define MASK_BLEND(M,ELSE,THEN) _mm256_blendv_pd(ELSE,THEN,M)

#endif /* MATERN_VECTOR_MASKS_AVX2_H */
