/* engines_current.h -- selects the elementary-function engines (exp, log, pow, gamma)
 * of the target instruction set; every path defines matern_exponential_pd,
 * matern_logarithm_pd, matern_gamma_pd and matern_power_pd. */
#ifndef MATERN_ENGINES_CURRENT_H
#define MATERN_ENGINES_CURRENT_H

#if defined(MATERN_TARGET_SCALAR)
  #include "engines_scalar.h"
#elif defined(MATERN_TARGET_AVX2)
  #include "exponential_avx2.h"
  #include "gamma_avx2.h"
  #include "power_avx2.h"
  #include "logarithm_avx2.h"
#elif defined(__AVX512F__)
  #include "exponential_avx512.h"
  #include "gamma_avx512.h"
  #include "power_avx512.h"
  #include "logarithm_avx512.h"
#else
  #error "engines_current.h: no supported instruction set selected"
#endif

#endif /* MATERN_ENGINES_CURRENT_H */
