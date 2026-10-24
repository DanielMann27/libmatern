/* levin_transform_avx2.h -- Levin transform of the large-argument partial sums, times
 * sqrt(pi/2z) e^-z; AVX2 form: independent term ratios, then a product chain.  The width-1
 * form (levin_transform_scalar.h) has the same arithmetic; the AVX-512 form rounds differently. */
#ifndef MATERN_LEVIN_TRANSFORM_AVX2_H
#define MATERN_LEVIN_TRANSFORM_AVX2_H

#include "vector_current.h"
#include "engines_current.h"

/* Must equal LEVIN_WEIGHTS in levin_transform_{avx512,scalar}.h and levin_transform_device.cuh. */
static const double LEVIN_WEIGHTS[15] = {
    3.7391005219784331e-07, -3.9260555480773548e-05, 0.0013610325900001495, -0.023137554030002543, 0.22906178489702517,
    -1.4507246376811593, 6.2173913043478262, -18.652173913043477, 39.894927536231883, -61.172222222222224,
    66.733333333333334, -50.555555555555557, 25.277777777777779, -7.5, 1,
};

/* 8i for i = 1..15, each value stored INDEX_FACTOR times so that a plain load broadcasts it. */
static const double LEVIN_INVERSE_INDEX[60] = {
    8,8,8,8, 16,16,16,16, 24,24,24,24, 32,32,32,32, 40,40,40,40,
    48,48,48,48, 56,56,56,56, 64,64,64,64, 72,72,72,72, 80,80,80,80,
    88,88,88,88, 96,96,96,96, 104,104,104,104, 112,112,112,112, 120,120,120,120,
};

static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT levin_transform(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT NU)
{
    const SIMD_TYPE_FLOAT ZERO = LOAD_FLOAT(MATERN_BROADCAST_POOL);
    const SIMD_TYPE_FLOAT ONE  = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT TWO  = LOAD_FLOAT(MATERN_BROADCAST_POOL + 2 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT FOUR = LOAD_FLOAT(MATERN_BROADCAST_POOL + 4 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT PI   = LOAD_FLOAT(MATERN_BROADCAST_PI);
    const SIMD_TYPE_FLOAT MU   = MUL_FLOAT(FOUR, MUL_FLOAT(NU, NU));

    /* Phase 1a: the independent term ratios t_i/t_(i-1). */
    SIMD_TYPE_FLOAT RHO[15];
    /* Unrolling does not reassociate. */
#pragma GCC unroll 16
    for (int i = 1; i <= 15; i++) {
        const SIMD_TYPE_FLOAT FACT = LOAD_FLOAT(MATERN_BROADCAST_FACTORS + (i - 1) * INDEX_FACTOR);   /* (2i-1)^2 */
        const SIMD_TYPE_FLOAT T8I  = LOAD_FLOAT(LEVIN_INVERSE_INDEX + (i - 1) * INDEX_FACTOR);   /* 8i      */
        RHO[i - 1] = DIV_FLOAT(SUB_FLOAT(MU, FACT), MUL_FLOAT(Z, T8I));
    }
    /* Phase 1b: cumulative product -> A_i, omega_i. */
    SIMD_TYPE_FLOAT SEQ[15], WT[15], S = ZERO, T = ONE;
#pragma GCC unroll 16
    for (int i = 1; i <= 15; i++) {
        S = ADD_FLOAT(S, T);
        T = MUL_FLOAT(T, RHO[i - 1]);
        SEQ[i - 1] = S;
        WT[i - 1]  = T;
    }
    /* Phases 2 and 3: NUM = sum C_k A_k/omega_k, DEN = sum C_k/omega_k. */
    SIMD_TYPE_FLOAT NUM = ZERO, DEN = ZERO;
#pragma GCC unroll 16
    for (int k = 0; k < 15; k++) {
        const SIMD_TYPE_FLOAT INV = DIV_FLOAT(ONE, WT[k]);
        const SIMD_TYPE_FLOAT C   = SET_FLOAT(LEVIN_WEIGHTS[k]);
        NUM = FMADD_FLOAT(C, MUL_FLOAT(SEQ[k], INV), NUM);
        DEN = FMADD_FLOAT(C, INV, DEN);
    }
    const SIMD_TYPE_FLOAT SQ = SQRT_ROUND(DIV_FLOAT(PI, MUL_FLOAT(TWO, Z)),
                                          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const SIMD_TYPE_FLOAT EZ = matern_exponential_pd(SUB_FLOAT(ZERO, Z));
    /* Half-integer nu terminates the series (factor 0 at 2nu = 2i-1) and NUM/DEN would be NaN.
     * T then stays 0, so SEQ[14] is the exact sum and WT[14] == 0 detects it; EQUAL_FLOAT also
     * matches -0.0. */
    const MASK_FLOAT LV_TERMINATED = EQUAL_FLOAT(WT[14], ZERO);
    const SIMD_TYPE_FLOAT LV_RATIO =
        BLEND_FLOAT(LV_TERMINATED, DIV_FLOAT(NUM, DEN), SEQ[14]);
    return MUL_FLOAT(MUL_FLOAT(LV_RATIO, SQ), EZ);
}

#endif /* MATERN_LEVIN_TRANSFORM_AVX2_H */
