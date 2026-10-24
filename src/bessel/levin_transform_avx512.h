/* levin_transform_avx512.h -- Levin transform of the large-argument partial sums, times
 * sqrt(pi/2z) e^-z; AVX-512 form: a running recurrence with one divide per term.  It rounds
 * differently from the AVX2 and width-1 form (levin_transform_avx2.h). */
#ifndef MATERN_LEVIN_TRANSFORM_AVX512_H
#define MATERN_LEVIN_TRANSFORM_AVX512_H

#include "vector_current.h"
#include "engines_current.h"

/* The difference-table recursion of backends/scalar/levin_transformed_asymptotic.c at fixed
 * length, collapsed to a linear functional of the partial sums.
 * Must equal LEVIN_WEIGHTS in levin_transform_{avx2,scalar}.h and levin_transform_device.cuh. */
static const double LEVIN_WEIGHTS[15] = {
    3.7391005219784331e-07, -3.9260555480773548e-05, 0.0013610325900001495, -0.023137554030002543, 0.22906178489702517,
    -1.4507246376811593, 6.2173913043478262, -18.652173913043477, 39.894927536231883, -61.172222222222224,
    66.733333333333334, -50.555555555555557, 25.277777777777779, -7.5, 1,
};

static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT levin_transform(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT NU)
{
    const SIMD_TYPE_FLOAT ZERO  = LOAD_FLOAT(MATERN_BROADCAST_POOL);
    const SIMD_TYPE_FLOAT ONE   = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT TWO   = LOAD_FLOAT(MATERN_BROADCAST_POOL + 2 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT FOUR  = LOAD_FLOAT(MATERN_BROADCAST_POOL + 4 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT EIGHT = LOAD_FLOAT(MATERN_BROADCAST_POOL + 8 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT PI    = LOAD_FLOAT(MATERN_BROADCAST_PI);

    /* Phase 1: SEQ[i] = A_i (partial sums), WT[i] = omega_i = t_{i+1}. */
    SIMD_TYPE_FLOAT SEQ[15], WT[15];
    SIMD_TYPE_FLOAT S = ZERO, T = ONE;
    const SIMD_TYPE_FLOAT MU = MUL_FLOAT(FOUR, MUL_FLOAT(NU, NU));   /* 4 nu^2 */
    /* Unrolling does not reassociate. */
#pragma GCC unroll 16
    for (int i = 1; i <= 15; i++) {
        S = ADD_FLOAT(S, T);
        const SIMD_TYPE_FLOAT FACT   = LOAD_FLOAT(MATERN_BROADCAST_FACTORS + (i - 1) * INDEX_FACTOR); /* (2i-1)^2 */
        const SIMD_TYPE_FLOAT FLOATS = LOAD_FLOAT(MATERN_BROADCAST_POOL + i * INDEX_FACTOR);        /* i        */
        const SIMD_TYPE_FLOAT TMP3 = MUL_FLOAT(EIGHT, MUL_FLOAT(Z, FLOATS));              /* 8 z i    */
        const SIMD_TYPE_FLOAT TMP6 = MUL_FLOAT(T, SUB_FLOAT(MU, FACT));
        T = DIV_FLOAT(TMP6, TMP3);
        SEQ[i - 1] = S;
        WT[i - 1]  = T;
    }

    /* Phases 2 and 3: NUM = sum C_k A_k/omega_k, DEN = sum C_k/omega_k. */
    SIMD_TYPE_FLOAT NUM = ZERO, DEN = ZERO;
#pragma GCC unroll 16
    for (int k = 0; k < 15; k++) {
        const SIMD_TYPE_FLOAT INV = DIV_FLOAT(ONE, WT[k]);          /* 1/omega_k */
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

#endif /* MATERN_LEVIN_TRANSFORM_AVX512_H */
