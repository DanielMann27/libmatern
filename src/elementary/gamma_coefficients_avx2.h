/* gamma_coefficients_avx2.h -- tables of the AVX2 gamma kernel: the Stirling pool (values
 * and layout of gamma_coefficients_avx512.h at width 4) and the log and 2^(j/8) tables
 * as 32-bit halves. */
#ifndef MATERN_GAMMA_COEFFICIENTS_AVX2_H
#define MATERN_GAMMA_COEFFICIENTS_AVX2_H

#include <stdint.h>
#include "simd_avx2.h"
#include "broadcast_pools_avx2.h"

/* Stirling series for ln Gamma, one broadcast group of four per term. */
static const DATA_TYPE_FLOAT MATERN_STIRLING_COEFFICIENTS[] = {
    4.18938533204672725052e-01,
    4.18938533204672725052e-01,
    4.18938533204672725052e-01,
    4.18938533204672725052e-01,
    8.33333333333329678849e-02,
    8.33333333333329678849e-02,
    8.33333333333329678849e-02,
    8.33333333333329678849e-02,
    -2.77777777728775536470e-03,
    -2.77777777728775536470e-03,
    -2.77777777728775536470e-03,
    -2.77777777728775536470e-03,
    7.93650558643019558500e-04,
    7.93650558643019558500e-04,
    7.93650558643019558500e-04,
    7.93650558643019558500e-04,
    -5.95187557450339963135e-04,
    -5.95187557450339963135e-04,
    -5.95187557450339963135e-04,
    -5.95187557450339963135e-04,
    8.36339918996282139126e-04,
    8.36339918996282139126e-04,
    8.36339918996282139126e-04,
    8.36339918996282139126e-04,
    -1.63092934096575273989e-03,
    -1.63092934096575273989e-03,
    -1.63092934096575273989e-03,
    -1.63092934096575273989e-03,
};





/* (c, logc) of the degree-11 logarithm: same values as the L_C/L_LC tables of
 * gamma_avx512.h, as 32-bit halves, buckets 0..7 (A) and 8..15 (B). */
static const uint32_t MATERN_GAMMA_C_LO32A[8]  = { 0xa061ead5u, 0x2f6dca63u, 0x241a584au, 0x98090bd4u, 0xdf1225d7u, 0x05b03089u, 0x8a4b8678u, 0x8f6b6c71u };
static const uint32_t MATERN_GAMMA_C_LO32B[8]  = { 0xf69ba7e6u, 0x00000000u, 0x8067ffc7u, 0xe97808d3u, 0x789f95b3u, 0xf953e58bu, 0xcafda03au, 0x5e0e4329u };
static const uint32_t MATERN_GAMMA_C_HI32A[8]  = { 0x3ff66dc1u, 0x3ff57cbcu, 0x3ff49f23u, 0x3ff3d2b4u, 0x3ff31584u, 0x3ff265efu, 0x3ff1c288u, 0x3ff12a17u };
static const uint32_t MATERN_GAMMA_C_HI32B[8]  = { 0x3ff09b8au, 0x3ff00000u, 0x3fee6de4u, 0x3fecb8b8u, 0x3feb3207u, 0x3fe9d2c5u, 0x3fe89544u, 0x3fe774dfu };
static const uint32_t MATERN_GAMMA_LOGC_LO32A[8] = { 0x0bf7fad6u, 0xfc31fea0u, 0x315b742eu, 0xf9e90967u, 0xfb941a16u, 0xa7156ba3u, 0x3721f401u, 0xf18bfbddu };
static const uint32_t MATERN_GAMMA_LOGC_LO32B[8] = { 0xd08c0289u, 0x00000000u, 0x07dbbc71u, 0x5834a389u, 0x330389bbu, 0x48880f84u, 0x6fede440u, 0xa09b7669u };
static const uint32_t MATERN_GAMMA_LOGC_HI32A[8] = { 0xbfd59dc4u, 0xbfd2df29u, 0xbfd03d78u, 0xbfcb6ccau, 0xbfc68fd6u, 0xbfc1e079u, 0xbfbab6f4u, 0xbfb1fbe7u };
static const uint32_t MATERN_GAMMA_LOGC_HI32B[8] = { 0xbfa31530u, 0x00000000u, 0x3fa9c4fcu, 0x3fbbab9eu, 0x3fc4d2eeu, 0x3fcb7357u, 0x3fd0e017u, 0x3fd3e0a7u };

/* 2^(j/8) as 32-bit halves; same values as MATERN_EXP_TH/TL in exponential_avx512.h. */
static const uint32_t MATERN_GAMMA_ETH_LO32[8] = { 0x00000000u, 0x3c7d517bu, 0x0a31b715u, 0xd5362a27u, 0x667f3bcdu, 0x422aa0dbu, 0x995ad3adu, 0xdcfba487u };
static const uint32_t MATERN_GAMMA_ETH_HI32[8] = { 0x3ff00000u, 0x3ff172b8u, 0x3ff306feu, 0x3ff4bfdau, 0x3ff6a09eu, 0x3ff8ace5u, 0x3ffae89fu, 0x3ffd5818u };
static const uint32_t MATERN_GAMMA_ETL_LO32[8] = { 0x00000000u, 0xb9d78a76u, 0xd23182e4u, 0xafec42e2u, 0x13b26456u, 0x56864b27u, 0x345dcc81u, 0xd75b3707u };
static const uint32_t MATERN_GAMMA_ETL_HI32[8] = { 0x00000000u, 0xbc819041u, 0x3c86f46au, 0x3c7d4397u, 0xbc9bdd34u, 0x3c96e9f1u, 0x3c97a1cdu, 0x3c82ed02u };

#endif /* MATERN_GAMMA_COEFFICIENTS_AVX2_H */
