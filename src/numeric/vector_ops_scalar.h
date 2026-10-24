/* vector_ops_scalar.h -- the region-kernel vocabulary over plain doubles: VEC_FMA is fma()
 * as set by fused_multiply_add_scalar.h, vec_divide_refined a plain divide as on AVX2.
 * matched_fma runs the AVX2 region formulas, but the compiler may fuse further multiply-adds
 * in this width-1 code that AVX2 keeps apart, so it agrees with AVX2 to a few ulp, not bit
 * for bit. */
#ifndef MATERN_VECTOR_OPS_SCALAR_H
#define MATERN_VECTOR_OPS_SCALAR_H

#include "simd_scalar.h"

#define VEC_LANES 1

#define VEC_VLD(P)      (*(const double *)(P))
#define VEC_VST(P, V)   (*(double *)(P) = (V))
#define VEC_VSET(X)     ((double)(X))

#define VEC_MUL(A, B)   ((A) * (B))
#define VEC_ADD(A, B)   ((A) + (B))
#define VEC_SUB(A, B)   ((A) - (B))
#define VEC_DIV(A, B)   ((A) / (B))
#define VEC_SQRT(A)     sqrt(A)
#define VEC_MIN(A, B)   fmin((A), (B))
#define VEC_MAX(A, B)   fmax((A), (B))
#define VEC_ABS(A)      fabs(A)

#define VEC_FMA(A, B, C)   fma((A), (B), (C))
#define VEC_FNMA(A, B, C)  fma(-(A), (B), (C))
#define VEC_FMS(A, B, C)   fma((A), (B), -(C))
/* Sites whose operands can cancel: the guarded form without -mfma
 * (fused_multiply_add_scalar.h). */
#define VEC_FMA_CANCEL(A, B, C)   fma_cancel((A), (B), (C))
#define VEC_FNMA_CANCEL(A, B, C)  fma_cancel(-(A), (B), (C))
#define VEC_FMS_CANCEL(A, B, C)   fma_cancel((A), (B), -(C))

/* Aliases of the simd_*.h macro set used by the kernels. */
#define MUL_FLOAT(A, B)   VEC_MUL(A, B)
#define ADD_FLOAT(A, B)   VEC_ADD(A, B)
#define SUB_FLOAT(A, B)   VEC_SUB(A, B)
#define DIV_FLOAT(A, B)   VEC_DIV(A, B)

#define FMADD_FLOAT(X, Y, Z)  VEC_FMA(X, Y, Z)

/* SQRT_ROUND ignores its mode argument (the call sites pass round-to-nearest, the default
 * mode); the _MM_FROUND_ names are defined only so that the call sites compile. */
#ifndef _MM_FROUND_TO_NEAREST_INT
#define _MM_FROUND_TO_NEAREST_INT 0
#endif
#ifndef _MM_FROUND_NO_EXC
#define _MM_FROUND_NO_EXC 0
#endif
#define SQRT_ROUND(X, MODE)   ((void)(MODE), VEC_SQRT(X))

/* AVX2 semantics: ABS is max(X, -X) as in simd_avx2.h (equal to fabs for finite input);
 * BLEND_FLOAT(M, Y, Z) is M ? Z : Y. */
#define FLOOR(X)                 floor(X)
#define FNMADD_FLOAT(X, Y, Z)    VEC_FNMA(X, Y, Z)
#define LESS_THAN_FLOAT(X, Y)    ((X) < (Y))
#define LESS_THAN_EQUAL_FLOAT(X, Y) ((X) <= (Y))
#define EQUAL_FLOAT(X, Y)        ((X) == (Y))
#define ABS(X)                   fmax((X), -1.0 * (X))
#define BLEND_FLOAT(M, Y, Z)     ((M) ? (Z) : (Y))

static inline double vec_first_lane(double v) { return v; }

/* A where cmpa < cmpb, B elsewhere. */
#define VEC_SELECT_LT(A, B, CMPA, CMPB)  (((CMPA) < (CMPB)) ? (A) : (B))

/* A plain divide, as on AVX2. */
static inline __attribute__((always_inline)) double vec_divide_refined(double x, double d)
{
    return x / d;
}

/* One lane: the horizontal maximum is that lane. */
static inline double vec_reduce_max(double d) { return d; }

#endif /* MATERN_VECTOR_OPS_SCALAR_H */
