/* scalar_internal.h -- constants and cross-file declarations of the scalar backend, which
 * uses plain doubles and libm (see the accompanying paper, CPU implementation and SIMD
 * strategy). */
#ifndef MATERN_SCALAR_INTERNAL_H
#define MATERN_SCALAR_INTERNAL_H

#include <stddef.h>

#include "matern.h"
/* No double-double header: the scalar kernels use plain doubles. */
#include "region_partition.h"
#include "half_integer_common.h"
#include "debye_coefficients_scalar.h"
#include "reciprocal_gamma_coefficients.h"

/* Iteration ceilings: hard stops behind convergence tests, not expected trip counts. */
#define SCALAR_LARGE_ARGUMENT_MAX_TERMS    50
#define SCALAR_DEBYE_TOP                   10
#define SCALAR_STIRLING_TERMS              4
#define SCALAR_GAMMA_TERMS                 11
#define SCALAR_FRACTION_SERIES_TERMS       50
#define SCALAR_FRACTION_LENTZ_STEPS        50
#define SCALAR_LEVIN_TERMS                 16
#define SCALAR_SMALL_ARGUMENT_MAX_TERMS    20

double matern_scalar_large_argument_at(double z, double order);
double matern_scalar_large_argument_folded_at(double z, double order);
double matern_scalar_debye(double z, double order);
double matern_scalar_continued_fraction(double z, double order);
double matern_scalar_small_argument(double z, double order);
double matern_scalar_levin(double z, double order);
void   matern_scalar_half_integer_run_z(double *out, const double *z, double order,
                                        size_t n, int degree);
void   matern_scalar_half_integer_run(double *out, const double *h, double order,
                                      const double *length_scale, size_t n, int degree);

/* Clamp to [0, 1], NaN -> 0, for every kernel output.  The half-integer extended arm maps
 * a value outside (0, 1] to 0 instead (half_integer_closed_form.c). */
double matern_scalar_clamp_to_unit(double value);

/* Unfolded twin, built only with MATERN_SCALAR_ENTRY/_Z set, for build/fold_cpu: with
 * -DMATERN_SCALAR_UNFOLDED=1 (direct) or =2 (exp-log) every region kernel returns K_nu(z) or
 * ln K_nu(z), and MATERN_SCALAR_FINISH applies the prefactor and clamps only a finite value. */
#ifdef MATERN_SCALAR_UNFOLDED
#  if !defined(MATERN_SCALAR_ENTRY) || !defined(MATERN_SCALAR_ENTRY_Z)
#    error "MATERN_SCALAR_UNFOLDED builds only as a renamed variant: define MATERN_SCALAR_ENTRY and MATERN_SCALAR_ENTRY_Z"
#  endif
#  ifdef MATERN_SCALAR_WITH_LEVIN
#    error "MATERN_SCALAR_UNFOLDED and MATERN_SCALAR_WITH_LEVIN exclude each other"
#  endif
#  if MATERN_SCALAR_UNFOLDED != 1 && MATERN_SCALAR_UNFOLDED != 2
#    error "MATERN_SCALAR_UNFOLDED must be 1 (direct) or 2 (exp-log)"
#  endif
#  define MATERN_SCALAR_PI_2  1.57079632679489661923       /* pi/2, as glibc's M_PI_2 */
#  define MATERN_SCALAR_LN2   0.6931471805599453           /* ln 2                    */
#  if MATERN_SCALAR_UNFOLDED == 2
#    define MATERN_SCALAR_K_RESULT(k)  log(k)              /* the kernels return ln K */
#  else
#    define MATERN_SCALAR_K_RESULT(k)  (k)                 /* the kernels return K    */
#  endif
/* Defined in asymptotic_large_argument.c, which has lnGamma. */
double matern_scalar_compose_unfolded(double k, double z, double order);
double matern_scalar_finish_unfolded(double k, double z, double order);
double matern_scalar_log_gamma_unfolded(double order);
#  define MATERN_SCALAR_FINISH(k, z, nu)  matern_scalar_finish_unfolded((k), (z), (nu))
#else
#  define MATERN_SCALAR_FINISH(k, z, nu)  matern_scalar_clamp_to_unit(k)
#endif

#endif /* MATERN_SCALAR_INTERNAL_H */
