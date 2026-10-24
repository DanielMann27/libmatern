/* half_integer_closed_form.h -- covariance at half-integer order nu = p + 1/2 in the vector
 * vocabulary: M = e^-z P(z), P(z) = SUM_k c_k z^k, as exp(-z + ln P(z)) with a double-double
 * exponent past z = 708.396 (see the accompanying paper, half-integer orders). */
#ifndef MATERN_HALF_INTEGER_CLOSED_FORM_H
#define MATERN_HALF_INTEGER_CLOSED_FORM_H

#include <float.h>
#include <math.h>
#include <stddef.h>

#include "vector_current.h"
#include "engines_current.h"

/* -log(DBL_MIN): exp(-x) leaves the normal range beyond this x.  Also defined in
 * router/region_partition.h; both definitions must stay token-identical. */
#define MATERN_EXP_RANGE_LIMIT 708.396418532264

#include "half_integer_common.h"

/* One vector of the closed form.  A run's remainder goes through this function on a
 * padded vector, so a result does not depend on how an array is split. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT half_integer_cell(SIMD_TYPE_FLOAT H, SIMD_TYPE_FLOAT R,
                                SIMD_TYPE_FLOAT S, SIMD_TYPE_FLOAT SL,
                                const double *c, int p)
{
    const SIMD_TYPE_FLOAT one = VEC_VSET(1.0), zero = VEC_VSET(0.0);
    const SIMD_TYPE_FLOAT zmax = VEC_VSET(MATERN_EXP_RANGE_LIMIT), nan_ = VEC_VSET((double)NAN);
#ifdef MATERN_ENTRY_Z
    /* The caller supplies z exactly in H; S, SL and R are unused and the residual
     * is zero. */
    (void)R; (void)S; (void)SL;
    SIMD_TYPE_FLOAT Z  = H;
    SIMD_TYPE_FLOAT ZL = zero;
#else
    SIMD_TYPE_FLOAT ph = VEC_MUL(S, H);
    SIMD_TYPE_FLOAT pl = VEC_FMA_CANCEL(SL, H, VEC_FMS_CANCEL(S, H, ph));
    SIMD_TYPE_FLOAT Z  = VEC_DIV(ph, R);
    SIMD_TYPE_FLOAT ZL = VEC_DIV(VEC_ADD(VEC_FNMA_CANCEL(Z, R, ph), pl), R);
#endif
    /* Simultaneous Horner: acc ends as P(Z) and dac as P'(Z).  Since
     * d/dz [P(z) e^-z] = (P'(z) - P(z)) e^-z, the argument correction is
     * M(z+zl) ~ (P + (P' - P) zl) e^-z. */
    SIMD_TYPE_FLOAT acc = VEC_VSET(c[p]), dac = zero;
    for(int k = p-1; k >= 0; k--){
        dac = VEC_FMA_CANCEL(dac, Z, acc);               /* -> P'(Z) */
        acc = VEC_FMA_CANCEL(acc, Z, VEC_VSET(c[k]));    /* -> P(Z)  */
    }
    SIMD_TYPE_FLOAT v = VEC_MUL(VEC_FMA(VEC_SUB(dac, acc), ZL, acc),
                                matern_exponential_pd(VEC_SUB(zero, Z)));
#ifdef MATERN_ENTRY_Z
    const vec_mask valid = MASK_AND(MASK_LESS_EQUAL(zero, H), MASK_LESS(zero, R));
#else
    /* s = +inf is invalid; it would give z = 0 and M = 1 */
    const vec_mask valid = MASK_AND(MASK_LESS_EQUAL(zero, H),
                               MASK_AND(MASK_LESS(zero, R), MASK_LESS_EQUAL(R, VEC_VSET(DBL_MAX))));
#endif
    /* The extended-range branch takes finite z only: at z = inf the exponent would be
     * -inf + ln P(inf) = NaN, so those lanes keep the 0 of the blend below. */
    const vec_mask mtail = MASK_AND(MASK_GREATER(Z, zmax), MASK_LESS_EQUAL(Z, VEC_VSET(DBL_MAX)));
    v = MASK_BLEND(MASK_GREATER(Z, zmax), v, zero);        /* every z > 708.396, inf included */
    /* Extended range: e^-z is subnormal past 708.396 and the direct product loses bits,
     * so M = exp(-z + ln P(z)).  Every c_k > 0, so P(z) > 0 and the logarithm is defined. */
    if (MASK_ANY(mtail)){
        /* P(Z+ZL) ~ acc + dac*ZL.  The exponent carries -Z-ZL exactly, so only P'
         * enters, unlike the direct form, whose (dac - acc) term includes e^{-ZL}. */
        SIMD_TYPE_FLOAT pc = VEC_FMA(dac, ZL, acc);
        SIMD_TYPE_FLOAT lh, ll; dd_log(pc, &lh, &ll);
        SIMD_TYPE_FLOAT Eh, e1; dd_two_sum(lh, VEC_SUB(zero, Z), &Eh, &e1);
        SIMD_TYPE_FLOAT El = VEC_SUB(VEC_ADD(e1, ll), ZL);
        /* The exponential engine's range, handled as in the large-argument extended
         * assembly: halve and square through the subnormal band, 0 below it. */
        const SIMD_TYPE_FLOAT eLO = VEC_VSET(-708.396418532264), eZR = VEC_VSET(-745.2);
        SIMD_TYPE_FLOAT arg = VEC_SELECT_LT(VEC_MUL(Eh, VEC_VSET(0.5)), Eh, Eh, eLO);
        SIMD_TYPE_FLOAT e0  = matern_exponential_pd(arg);
        SIMD_TYPE_FLOAT vt  = VEC_SELECT_LT(VEC_MUL(e0, e0), e0, Eh, eLO);
        vt = VEC_SELECT_LT(VEC_VSET(0.0), vt, Eh, eZR);
        /* Second order in El, since dd_log's low word is not normalised. */
        vt = VEC_FMA(vt, VEC_FMA(VEC_MUL(El, VEC_VSET(0.5)), El, El), vt);
        /* M lies in (0, 1].  Once P(z) overflows, dd_log receives a non-finite value and
         * the exponent is +inf or NaN; M is far below the double range there, so a value
         * outside (0, 1] is 0. */
        vt = MASK_BLEND(MASK_AND(MASK_GREATER(vt, zero), MASK_LESS_EQUAL(vt, one)), zero, vt);
        v = MASK_BLEND(mtail, v, vt);
    }
    v = MASK_BLEND(MASK_AND(valid, MASK_NOT(MASK_GREATER(Z, zero))), v, one);
    v = MASK_BLEND(MASK_NOT(valid), v, nan_);
    return v;
}

static void matern_half_integer_run(double *restrict out,const double *restrict h,
                            double nu,const double *restrict scale,size_t n,int p)
{
    double c[HALF_INTEGER_MAX_DEGREE + 1];
    half_integer_coefficients(p, c);
    /* sqrt(2nu) as a double-double, once per run. */
    const double two_nu = nu + nu;                       /* exact */
    const double shd = sqrt(two_nu);
    const double sqh = shd * shd;                        /* rounded square; the fma recovers its error */
    const double sld = ((two_nu - sqh) - fma_cancel(shd, shd, -sqh)) / (shd + shd);
    const SIMD_TYPE_FLOAT S  = VEC_VSET(shd);
    const SIMD_TYPE_FLOAT SL = VEC_VSET(sld);
    size_t i = 0;
    for(; i + VEC_LANES <= n; i += VEC_LANES)
        VEC_VST(out+i, half_integer_cell(VEC_VLD(h+i), VEC_VLD(scale+i), S, SL, c, p));
    if(i < n){                       /* tail: same code, padded to a full vector */
        double hb[VEC_LANES], rb[VEC_LANES], ob[VEC_LANES];
        const size_t r = n - i;
        for(size_t k = 0; k < (size_t)VEC_LANES; k++){
            const size_t j = (k < r) ? k : r - 1;          /* replicate, never read OOB */
            hb[k] = h[i+j]; rb[k] = scale[i+j];
        }
        VEC_VST(ob, half_integer_cell(VEC_VLD(hb), VEC_VLD(rb), S, SL, c, p));
        for(size_t k = 0; k < r; k++) out[i+k] = ob[k];
    }
}

#endif /* MATERN_HALF_INTEGER_CLOSED_FORM_H */
