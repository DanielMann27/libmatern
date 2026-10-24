/* half_integer_closed_form.c -- scalar closed form for a run of constant half-integer order
 * nu = p + 1/2, p <= 40: M = e^{-z} P_p(z), as exp(log P - z) past the exponential limit
 * (see the accompanying paper, half-integer orders). */
#include <float.h>
#include <math.h>
#include <stddef.h>

#include "scalar_internal.h"

/* Above the exponential limit exp(log(acc) - z): e^{-z} leaves the normal range there
 * while the product may not.  Both arms need the Horner sum, so it precedes the branch.
 * There M is far below 1, so a value above 1 or not finite is not rounding: P(z) has
 * overflowed, or z = +inf.  M is then far below the double range, so the value is 0. */
static inline double matern_scalar_half_integer_point(double z, const double *c, int p)
{
    if (z == 0.0) return 1.0;
    double acc = c[p];                               /* Horner, P(z) */
    for (int k = p-1; k >= 0; k--) acc = acc*z + c[k];
    if (!(z <= MATERN_EXP_RANGE_LIMIT)){
        const double v = exp(log(acc) - z);
        return (v > 0.0 && v <= 1.0) ? v : 0.0;
    }
    return acc * exp(-z);
}

#ifdef WMS_UNFOLDED
/* The twin: K_nu(z) = sqrt(pi/2z) e^{-z} SUM_{k=0..p} a_k z^{-k} with
 * a_k = (p+k)! / (k! (p-k)! 2^k); the order part of the prefactor is computed once per run,
 * and the clamp is applied where this backend applies it. */
typedef struct {
    double a[HALF_INTEGER_MAX_DEGREE + 1];
    double order_part;
    double nu;
    int    p;
} wms_half_integer_run;

static void wms_half_integer_setup(wms_half_integer_run *r, double nu, int p)
{
    r->a[0] = 1.0;
    for (int k = 1; k <= p; k++)
        r->a[k] = r->a[k-1] * (double)(p + k) * (double)(p - k + 1) / (2.0 * (double)k);
#  if WMS_UNFOLDED == 2
    r->order_part = (1.0 - nu) * WMS_LN2 - matern_scalar_log_gamma_unfolded(nu);
#  else
    r->order_part = 2.0 / tgamma(nu);
#  endif
    r->nu = nu;
    r->p  = p;
}

static inline double wms_half_integer_point(double z, const wms_half_integer_run *r)
{
    if (z == 0.0) return 1.0;
    const double w = 1.0 / z;
    double S = r->a[r->p];                           /* Horner, SUM a_k z^-k */
    for (int k = r->p - 1; k >= 0; k--) S = S*w + r->a[k];
#  if WMS_UNFOLDED == 2
    const double lnK = 0.5 * log(WMS_PI_2 / z) - z + log(S);
    const double c = exp(r->order_part + r->nu * log(z) + lnK);
#  else
    const double K = sqrt(WMS_PI_2 / z) * S * exp(-z);
    const double c = r->order_part * pow(z*0.5, r->nu) * K;
#  endif
    if (!(z <= MATERN_EXP_RANGE_LIMIT))
        return isfinite(c) ? matern_scalar_clamp_to_unit(c) : c;
    return c;
}
#endif

/* The (z, nu) run: the same closed form as the (h, nu, s) entry, so the two entries agree. */
void matern_scalar_half_integer_run_z(double *out, const double *z, double nu,
                          size_t n, int p)
{
#ifdef WMS_UNFOLDED
    wms_half_integer_run r;
    wms_half_integer_setup(&r, nu, p);
    for (size_t i = 0; i < n; i++){
        if (!(z[i] >= 0.0)){ out[i] = (double)NAN; continue; }
        out[i] = wms_half_integer_point(z[i], &r);
    }
#else
    double c[HALF_INTEGER_MAX_DEGREE + 1];
    half_integer_coefficients(p, c);
    (void)nu;                       /* only the degree p is needed once z is given */
    for (size_t i = 0; i < n; i++){
        if (!(z[i] >= 0.0)){ out[i] = (double)NAN; continue; }
        out[i] = matern_scalar_half_integer_point(z[i], c, p);
    }
#endif
}

void matern_scalar_half_integer_run(double *out, const double *h, double nu,
                          const double *length_scale, size_t n, int p)
{
#ifdef WMS_UNFOLDED
    wms_half_integer_run r;
    wms_half_integer_setup(&r, nu, p);
    const double sh = sqrt(2.0 * nu);
    for (size_t i = 0; i < n; i++){
        if (!(h[i] >= 0.0) || !(length_scale[i] > 0.0 && length_scale[i] <= DBL_MAX)){
            out[i] = (double)NAN; continue; }
        out[i] = wms_half_integer_point(sh * h[i] / length_scale[i], &r);
    }
#else
    double c[HALF_INTEGER_MAX_DEGREE + 1];
    half_integer_coefficients(p, c);
    const double sh = sqrt(2.0 * nu);
    for (size_t i = 0; i < n; i++){
        if (!(h[i] >= 0.0) || !(length_scale[i] > 0.0 && length_scale[i] <= DBL_MAX)){
            out[i] = (double)NAN; continue; }
        out[i] = matern_scalar_half_integer_point(sh * h[i] / length_scale[i], c, p);
    }
#endif
}
