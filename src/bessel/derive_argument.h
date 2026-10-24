/* derive_argument.h -- bessel_derive_argument(): z = sqrt(2 nu) h / s as a double-double
 * zh + zl for the large-argument, Debye and Levin kernels, which feed zl back.  The
 * small-argument and continued-fraction kernels derive z in one double at their own sites. */
#ifndef MATERN_DERIVE_ARGUMENT_H
#define MATERN_DERIVE_ARGUMENT_H

#include "vector_current.h"

/* Under MATERN_ENTRY_Z, H holds z: zh = H and zl = 0, so every downstream double-double
 * correction is an exact no-op (rebuilding h from z and deriving z again would re-round). */
static inline __attribute__((always_inline))
void bessel_derive_argument(SIMD_TYPE_FLOAT H, SIMD_TYPE_FLOAT NU, SIMD_TYPE_FLOAT R,
               SIMD_TYPE_FLOAT *zh, SIMD_TYPE_FLOAT *zl){
#ifdef MATERN_ENTRY_Z
    (void)NU; (void)R;
    *zh = H;                       /* the caller's z */
    *zl = VEC_VSET(0.0);
    return;
#else
    /* Three steps, each keeping its rounding residual, formed with a fused operation.
     * M behaves like e^-z, so an absolute error in z is a relative error in M. */
    SIMD_TYPE_FLOAT t  = VEC_ADD(NU, NU);                       /* 2nu, exact */
    SIMD_TYPE_FLOAT sh = VEC_SQRT(t);
    /* sqrt(2nu) = sh + sl, sl = (t - sh^2)/(2 sh): one Newton step */
    SIMD_TYPE_FLOAT sl = VEC_DIV(VEC_FNMA_CANCEL(sh, sh, t), VEC_ADD(sh, sh));
    SIMD_TYPE_FLOAT ph = VEC_MUL(sh, H);                        /* sqrt(2nu) h = ph + pl */
    SIMD_TYPE_FLOAT pl = VEC_FMA_CANCEL(sl, H, VEC_FMS_CANCEL(sh, H, ph));
    *zh = VEC_DIV(ph, R);
    /* the quotient's residual: z_lo = (ph - zh R + pl)/R */
    *zl = VEC_DIV(VEC_ADD(VEC_FNMA_CANCEL(*zh, R, ph), pl), R);
#endif
}

#endif /* MATERN_DERIVE_ARGUMENT_H */
