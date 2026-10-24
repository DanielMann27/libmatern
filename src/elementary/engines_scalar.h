/* engines_scalar.h -- width-1 elementary functions from libm (exp, log, pow, tgamma)
 * under -DMATERN_TARGET_SCALAR, for the matched builds of the AVX2 region code (fma
 * per fused_multiply_add_scalar.h).  matern_gamma_pd returns Gamma(x), not ln Gamma(x). */
#ifndef MATERN_ENGINES_SCALAR_H
#define MATERN_ENGINES_SCALAR_H

#include <math.h>

static inline double matern_exponential_pd(double x) { return exp(x); }
static inline double matern_logarithm_pd(double x)   { return log(x); }
static inline double matern_power_pd(double x, double y) { return pow(x, y); }
static inline double matern_gamma_pd(double x)       { return tgamma(x); }

#endif /* MATERN_ENGINES_SCALAR_H */
