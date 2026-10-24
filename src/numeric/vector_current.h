/* vector_current.h -- binds the vector vocabulary (VEC_*, MASK_*, dd_*, broadcast pools)
 * of the ISA-neutral region kernels to the AVX-512, AVX2 or width-1 implementation. */
#ifndef MATERN_VECTOR_CURRENT_H
#define MATERN_VECTOR_CURRENT_H

#if defined(MATERN_TARGET_SCALAR)
  #include "simd_scalar.h"
  #include "vector_ops_scalar.h"
  #include "vector_masks_scalar.h"
  #include "double_double_scalar.h"
  #include "broadcast_pools_scalar.h"
#elif defined(MATERN_TARGET_AVX2)
  #include "simd_avx2.h"
  #include "vector_ops_avx2.h"
  #include "vector_masks_avx2.h"
  #include "double_double_avx2.h"
  #include "broadcast_pools_avx2.h"
#elif defined(__AVX512F__)
  #include "simd_avx512.h"
  #include "vector_ops_avx512.h"
  #include "vector_masks_avx512.h"
  #include "double_double_avx512.h"
  #include "broadcast_pools_avx512.h"
#else
  #error "vector_current.h: build with -mavx512f -mavx512dq -mfma, or with -mavx2 -mfma -DMATERN_TARGET_AVX2"
#endif

#endif /* MATERN_VECTOR_CURRENT_H */
