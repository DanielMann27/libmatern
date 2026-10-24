/* vector_masks_avx512.h -- per-lane predicates as 8-bit mask registers;
 * MASK_BLEND(m, e, t) gives t where m is set, e elsewhere.  Comparisons are ordered and
 * quiet, so a NaN lane compares false. */
#ifndef MATERN_VECTOR_MASKS_AVX512_H
#define MATERN_VECTOR_MASKS_AVX512_H

#include <immintrin.h>

typedef __mmask8 vec_mask;
#define MASK_AND(A,B) _kand_mask8(A,B)
#define MASK_OR(A,B)  _kor_mask8(A,B)
#define MASK_NOT(A)   _knot_mask8(A)
#define MASK_GREATER(A,B)  _mm512_cmp_pd_mask(A,B,_CMP_GT_OQ)
#define MASK_LESS(A,B)  _mm512_cmp_pd_mask(A,B,_CMP_LT_OQ)
#define MASK_LESS_EQUAL(A,B)  _mm512_cmp_pd_mask(A,B,_CMP_LE_OQ)
#define MASK_EQUAL(A,B)  _mm512_cmp_pd_mask(A,B,_CMP_EQ_OQ)
#define MASK_ANY(M)   ((M)!=0)
#define MASK_COUNT(M) ((size_t)__builtin_popcount((unsigned)(M)))   /* lanes set */
#define MASK_BLEND(M,ELSE,THEN) _mm512_mask_blend_pd(M,ELSE,THEN)

#endif /* MATERN_VECTOR_MASKS_AVX512_H */
