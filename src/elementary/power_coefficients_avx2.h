/* power_coefficients_avx2.h -- AVX2 power tables as 32-bit halves: entry j of C and LC2
 * (-log2 c) is HI32{A,B}[j & 7]:LO32{A,B}[j & 7], B for j > 7; entry 9 is c = 1, lc2 = 0.
 * High words of C equal those of dd_log and the AVX2 gamma tables; low words differ. */
#ifndef MATERN_POWER_COEFFICIENTS_AVX2_H
#define MATERN_POWER_COEFFICIENTS_AVX2_H

#include <stdint.h>

static const uint32_t MATERN_POWER_C_LO32A[8]   = { 0xa062eb90u, 0x2f6bc72cu, 0x241e16e5u, 0x98097416u, 0xdf141d98u, 0x05af525bu, 0x8a4ea8ceu, 0x8f6b4b9cu };
static const uint32_t MATERN_POWER_C_LO32B[8]   = { 0xf69c8f68u, 0x00000000u, 0x8068223fu, 0xe97a0bc3u, 0x78a0b502u, 0xf958789bu, 0xcb005a3fu, 0x5e1019f2u };
static const uint32_t MATERN_POWER_C_HI32A[8]   = { 0x3ff66dc1u, 0x3ff57cbcu, 0x3ff49f23u, 0x3ff3d2b4u, 0x3ff31584u, 0x3ff265efu, 0x3ff1c288u, 0x3ff12a17u };
static const uint32_t MATERN_POWER_C_HI32B[8]   = { 0x3ff09b8au, 0x3ff00000u, 0x3fee6de4u, 0x3fecb8b8u, 0x3feb3207u, 0x3fe9d2c5u, 0x3fe89544u, 0x3fe774dfu };
static const uint32_t MATERN_POWER_LC2_LO32A[8] = { 0x574cf28fu, 0x77389420u, 0xec3a8f0eu, 0x78ec6ae4u, 0x406abb93u, 0x44f6d3ccu, 0x3268cc92u, 0x4b2fcf92u };
static const uint32_t MATERN_POWER_LC2_LO32B[8] = { 0x22b57afdu, 0x00000000u, 0xe9916c61u, 0x2b8568aau, 0x2d9d8f17u, 0xac0c1108u, 0x1cbb1baau, 0x85731270u };
static const uint32_t MATERN_POWER_LC2_HI32A[8] = { 0xbfdf2f89u, 0xbfdb39e5u, 0xbfd76df5u, 0xbfd3c86eu, 0xbfd04663u, 0xbfc9ca75u, 0xbfc34543u, 0xbfb9f208u };
static const uint32_t MATERN_POWER_LC2_HI32B[8] = { 0xbfab87d7u, 0x00000000u, 0x3fb296b7u, 0x3fc3f5c0u, 0x3fce0ae8u, 0x3fd3cd27u, 0x3fd85893u, 0x3fdcad60u };
/* 2^(j/8) as 32-bit halves; same values as MATERN_EXP_TH/TL in exponential_avx512.h. */
static const uint32_t MATERN_POWER_TH8_LO32[8] = { 0x00000000u, 0x3c7d517bu, 0x0a31b715u, 0xd5362a27u, 0x667f3bcdu, 0x422aa0dbu, 0x995ad3adu, 0xdcfba487u };
static const uint32_t MATERN_POWER_TH8_HI32[8] = { 0x3ff00000u, 0x3ff172b8u, 0x3ff306feu, 0x3ff4bfdau, 0x3ff6a09eu, 0x3ff8ace5u, 0x3ffae89fu, 0x3ffd5818u };
static const uint32_t MATERN_POWER_TL8_LO32[8] = { 0x00000000u, 0xb9d78a76u, 0xd23182e4u, 0xafec42e2u, 0x13b26456u, 0x56864b27u, 0x345dcc81u, 0xd75b3707u };
static const uint32_t MATERN_POWER_TL8_HI32[8] = { 0x00000000u, 0xbc819041u, 0x3c86f46au, 0x3c7d4397u, 0xbc9bdd34u, 0x3c96e9f1u, 0x3c97a1cdu, 0x3c82ed02u };

#endif /* MATERN_POWER_COEFFICIENTS_AVX2_H */
