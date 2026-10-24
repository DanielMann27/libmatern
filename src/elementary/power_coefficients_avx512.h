/* power_coefficients_avx512.h -- base-2 table of the AVX-512 power: c and lc2 = -log2(c),
 * with lc2 a multiple of 2^-42 so that e + lc2 is exact for |e| <= 1074 (53 bits). */
#ifndef MATERN_POWER_COEFFICIENTS_AVX512_H
#define MATERN_POWER_COEFFICIENTS_AVX512_H

/* Generated table; do not edit by hand.  The reduced argument satisfies |r| <= 0.0302. */
static const double MATERN_POWER_C[16] __attribute__((aligned(64))) = {
    0x1.66dc19e1cc31fp+0, 0x1.57cbc2d2ec8a9p+0, 0x1.49f23226792fbp+0, 0x1.3d2b4966053acp+0,
    0x1.31584dcd35ae3p+0, 0x1.265ef035d4516p+0, 0x1.1c288891368dcp+0, 0x1.12a178d175c98p+0,
    0x1.09b8af48045edp+0, 0x1.0000000000000p+0, 0x1.e6de47e5a3feap-1, 0x1.cb8b8e718c9b8p-1,
    0x1.b32077642e801p-1, 0x1.9d2c5f792b674p-1, 0x1.89544c920ec71p-1, 0x1.774df5bd21332p-1,
};
static const double MATERN_POWER_LC2[16] __attribute__((aligned(64))) = {
    -0x1.f2f894df06000p-2, -0x1.b39e56d9a6000p-2, -0x1.76df5e48f2000p-2, -0x1.3c86e71304000p-2,
    -0x1.0466335844000p-2, -0x1.9ca752dac4000p-3, -0x1.34543259ae000p-3, -0x1.9f20819190000p-4,
    -0x1.b87d6c4d70000p-5, 0x0.0p+0, 0x1.296b802804000p-4, 0x1.3f5c03ad2a000p-3,
    0x1.e0ae83daf2000p-3, 0x1.3cd27b2628000p-2, 0x1.8589323c3d000p-2, 0x1.cad608e46a000p-2,
};

#endif /* MATERN_POWER_COEFFICIENTS_AVX512_H */
