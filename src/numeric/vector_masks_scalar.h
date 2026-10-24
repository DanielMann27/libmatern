/* vector_masks_scalar.h -- the mask vocabulary at width 1: a mask is one truth value and a
 * horizontal reduction is that value. */
#ifndef MATERN_VECTOR_MASKS_SCALAR_H
#define MATERN_VECTOR_MASKS_SCALAR_H

#include "simd_scalar.h"

typedef int vec_mask;

#define MASK_AND(A, B)          ((A) && (B))
#define MASK_OR(A, B)           ((A) || (B))
#define MASK_NOT(A)             (!(A))
#define MASK_GREATER(A, B)      ((A) >  (B))
#define MASK_LESS(A, B)         ((A) <  (B))
#define MASK_LESS_EQUAL(A, B)   ((A) <= (B))
#define MASK_EQUAL(A, B)        ((A) == (B))
#define MASK_ANY(M)             ((M) != 0)
#define MASK_COUNT(M)           ((size_t)((M) != 0))
#define MASK_REDUCE(M)          ((M) != 0)
#define MASK_BLEND(M, ELSE, THEN)  ((M) ? (THEN) : (ELSE))

#endif /* MATERN_VECTOR_MASKS_SCALAR_H */
