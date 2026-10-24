/* levin_transformed_asymptotic.h -- covariance in the Levin band (region_partition.h): a
 * fixed-length Levin transform of the large-argument series, with the first-order z_lo
 * correction (see the accompanying paper, numerical evaluation of the modified Bessel function). */
#ifndef MATERN_LEVIN_TRANSFORMED_ASYMPTOTIC_H
#define MATERN_LEVIN_TRANSFORMED_ASYMPTOTIC_H

#include <stddef.h>

#include "vector_current.h"
#include "derive_argument.h"
#include "engines_current.h"

/* levin_transform is per instruction set; at half-integer orders, where the series
 * terminates, it returns the exact terminated sum. */
#if defined(MATERN_TARGET_SCALAR)
  #include "levin_transform_scalar.h"
#elif defined(MATERN_TARGET_AVX2)
  #include "levin_transform_avx2.h"
#else
  #include "levin_transform_avx512.h"
#endif

/* The prefactor 2 (z/2)^nu / Gamma(nu), unfolded; safe in the Levin band (nu <= 15, z <= 22). */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT levin_prefactor(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT NU){
    return VEC_MUL(VEC_VSET(2.0),
        VEC_DIV(matern_power_pd(VEC_MUL(Z, VEC_VSET(0.5)), NU), matern_gamma_pd(NU)));
}

/* One vector of M with the correction F = 1 + ((nu-0.5)/z - 1) z_lo, i.e. d lnM/dz
 * without the series' own Sigma'/Sigma term. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT levin_cell(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT ZL, SIMD_TYPE_FLOAT NU){
    SIMD_TYPE_FLOAT r  = VEC_MUL(levin_prefactor(Z,NU), levin_transform(Z,NU));
    SIMD_TYPE_FLOAT w  = VEC_DIV(VEC_VSET(1.0), Z);
    SIMD_TYPE_FLOAT cf = VEC_FMA_CANCEL(VEC_SUB(NU, VEC_VSET(0.5)), w, VEC_VSET(-1.0));
    return VEC_FMA(VEC_MUL(r, cf), ZL, r);
}

/* Blocks of four vectors, then single vectors, then each remaining element broadcast to a
 * full vector, keeping lane 0: no masked load and no read past the caller's arrays. */
static void matern_levin_x4(double *res,double *h,double *nu,double *scale,int n){
    size_t blk=(size_t)VEC_LANES*(size_t)(4), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk){
        for(int q=0;q<(4);q++){
            size_t j=i+(size_t)q*VEC_LANES;
            SIMD_TYPE_FLOAT zh,zl;
            bessel_derive_argument(VEC_VLD(h+j),VEC_VLD(nu+j),VEC_VLD(scale+j),&zh,&zl);
            VEC_VST(res+j, levin_cell(zh,zl,VEC_VLD(nu+j)));
        }
    }
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES){
        SIMD_TYPE_FLOAT zh,zl;
        bessel_derive_argument(VEC_VLD(h+i),VEC_VLD(nu+i),VEC_VLD(scale+i),&zh,&zl);
        VEC_VST(res+i, levin_cell(zh,zl,VEC_VLD(nu+i)));
    }
    for(;i<(size_t)n;i++){
        SIMD_TYPE_FLOAT zh,zl;
        bessel_derive_argument(VEC_VSET(h[i]),VEC_VSET(nu[i]),VEC_VSET(scale[i]),&zh,&zl);
        res[i]=vec_first_lane(levin_cell(zh,zl,VEC_VSET(nu[i])));
    }
}

#endif /* MATERN_LEVIN_TRANSFORMED_ASYMPTOTIC_H */
