/* debye_uniform_order.h -- covariance at large order by the uniform (Debye) expansion with
 * the covariance prefactor folded into one double-double exponent, plus its extended-range
 * entry, in the vector vocabulary (see the accompanying paper, large-order formulation). */
#ifndef MATERN_DEBYE_UNIFORM_ORDER_H
#define MATERN_DEBYE_UNIFORM_ORDER_H

#include <math.h>      /* INFINITY, for the extended-range guard */
#include <stddef.h>

#include "vector_current.h"
#include "derive_argument.h"
#include "engines_current.h"

#include "debye_polynomial.h"

/* Highest u_k index (10: eleven terms); MATERN_DEBYE_TOP_INDEX_DEVICE must equal it. */
#define DEBYE_TOP_INDEX 10


static inline SIMD_TYPE_FLOAT debye_polynomial_sum_capped(SIMD_TYPE_FLOAT P1,
                                                          SIMD_TYPE_FLOAT p,
                                                          SIMD_TYPE_FLOAT NU)
{
    return debye_polynomial_sum(P1, p, NU, DEBYE_TOP_INDEX);
}


/* debye_cell -- one vector of M(h) from the double-double argument (Z, ZL):
 *   M = sqrt(p) exp(nu G - S(nu)) SUM_k (-1)^k u_k(p)/nu^k,   G = ln((1+t)/2) - (t - 1),
 * with x = z/nu, t = sqrt(1+x^2), p = 1/t and S(nu) the Stirling correction of lnGamma(nu).
 * An absolute error in nu*G is a relative error in M, so x, x^2, 1+x^2, t, 1+t, G and nu*G
 * are double-doubles: z_lo enters at the inlet, xl = (fnma(xh, nu, z) + z_lo) * rn, and
 * propagates exactly. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT debye_cell(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT ZL, SIMD_TYPE_FLOAT NU){
    const SIMD_TYPE_FLOAT one = VEC_VSET(1.0), zero = VEC_VSET(0.0);
    const SIMD_TYPE_FLOAT rn = VEC_DIV(one, NU);                         /* 1/nu          */
    SIMD_TYPE_FLOAT xh = VEC_MUL(Z, rn);                                 /* x = z/nu      */
    SIMD_TYPE_FLOAT xl = VEC_MUL(VEC_ADD(VEC_FNMA_CANCEL(xh, NU, Z), ZL), rn);  /* the double-double inlet: one add */
    SIMD_TYPE_FLOAT x2h = VEC_MUL(xh, xh);                               /* x^2           */
    SIMD_TYPE_FLOAT x2l = VEC_FMA_CANCEL(VEC_ADD(xh, xh), xl, VEC_FMS_CANCEL(xh, xh, x2h));
    SIMD_TYPE_FLOAT uh, ue; dd_two_sum(one, x2h, &uh, &ue);              /* u = 1 + x^2   */
    SIMD_TYPE_FLOAT ul = VEC_ADD(ue, x2l);
    SIMD_TYPE_FLOAT th = VEC_SQRT(uh);                                   /* t = sqrt(u)   */
    const SIMD_TYPE_FLOAT p = VEC_DIV(one, th);                          /* p = 1/t       */
    /* low word of t from the residual u - t^2, scaled by 1/(2t): one Newton step
     * for the square root in double-double */
    SIMD_TYPE_FLOAT tl = VEC_MUL(VEC_ADD(VEC_FNMA_CANCEL(th, th, uh), ul),
                                 VEC_MUL(VEC_VSET(0.5), p));
    SIMD_TYPE_FLOAT ph, pe; dd_two_sum(one, th, &ph, &pe);               /* 1 + t         */
    SIMD_TYPE_FLOAT pl = VEC_ADD(pe, tl);
    /* ln((1+t)/2) = ln(1 + d/2), d = t-1: halving is exact, and the relative tail pl/ph is
     * scale-invariant */
    SIMD_TYPE_FLOAT l1h, l1l;
    dd_log(VEC_MUL(VEC_VSET(0.5), ph), &l1h, &l1l);
    l1l = VEC_ADD(l1l, VEC_DIV(pl, ph));
    SIMD_TYPE_FLOAT dh = VEC_SUB(th, one);          /* exact (Sterbenz) for t in [1,2] */
    /* G = ln((1+t)/2) - (t-1) */
    SIMD_TYPE_FLOAT Gh, ea; dd_two_sum(l1h, VEC_SUB(zero, dh), &Gh, &ea);
    SIMD_TYPE_FLOAT Gl = VEC_ADD(VEC_SUB(ea, tl), l1l);
    SIMD_TYPE_FLOAT nGh = VEC_MUL(NU, Gh);                               /* nu G          */
    SIMD_TYPE_FLOAT nGl = VEC_FMA_CANCEL(NU, Gl, VEC_FMS_CANCEL(NU, Gh, nGh));
    /* E = nu*G - S(nu), S = 1/(12nu) - 1/(360nu^3) + 1/(1260nu^5) - 1/(1680nu^7), the
     * Stirling correction of lnGamma(nu); the ln nu and ln 2pi terms cancel.  cc
     * below is -S(nu), an odd polynomial in 1/nu evaluated in 1/nu^2. */
    SIMD_TYPE_FLOAT rn2 = VEC_MUL(rn, rn);
    SIMD_TYPE_FLOAT cc = VEC_FMA(rn2, VEC_VSET(1.0/1680.0), VEC_VSET(-1.0/1260.0));
    cc = VEC_FMA(rn2, cc, VEC_VSET(1.0/360.0));
    cc = VEC_FMA(rn2, cc, VEC_VSET(-1.0/12.0));
    cc = VEC_MUL(rn, cc);
    SIMD_TYPE_FLOAT Eh, e3; dd_two_sum(nGh, cc, &Eh, &e3);
    SIMD_TYPE_FLOAT El = VEC_ADD(e3, nGl);
    /* Renormalise before the first-order exp below: El holds nu times G's low words, which
     * dd_log does not normalise, so El grows with nu.  Fast2Sum: exact while |El| <= |Eh|;
     * beyond, Eh is still the rounded sum, so the first-order factor stays small. */
    { const SIMD_TYPE_FLOAT Es = VEC_ADD(Eh, El); El = VEC_SUB(El, VEC_SUB(Es, Eh)); Eh = Es; }
    SIMD_TYPE_FLOAT eE;
    /* The exponential engine is valid on [ln DBL_MIN, 708] and E reaches far below it, so
     * the three cases of the large-argument extended assembly apply:
     *   Eh >= -708.396        : the engine directly;
     *   -745.2 <= Eh < -708.4 : (engine(Eh/2))^2, reaching the subnormal band;
     *   Eh < -745.2           : exactly 0. */
    { const SIMD_TYPE_FLOAT eLO = VEC_VSET(-708.396418532264), eZR = VEC_VSET(-745.2);
      SIMD_TYPE_FLOAT arg = VEC_SELECT_LT(VEC_MUL(Eh, VEC_VSET(0.5)), Eh, Eh, eLO);
      SIMD_TYPE_FLOAT e0  = matern_exponential_pd(arg);
      eE = VEC_SELECT_LT(VEC_MUL(e0, e0), e0, Eh, eLO);
      eE = VEC_SELECT_LT(VEC_VSET(0.0), eE, Eh, eZR); }
    eE = VEC_FMA(eE, El, eE);              /* exp(Eh+El) = exp(Eh)(1+El) to first order */
    SIMD_TYPE_FLOAT POLY = debye_polynomial_sum_capped(VEC_MUL(p, p), p, NU);
    return VEC_MUL(VEC_MUL(VEC_SQRT(p), eE), POLY);   /* the one factor outside the exponent */
}



static void matern_debye_x4(double *res,double *h,double *nu,double *scale,int n){
    size_t blk=(size_t)VEC_LANES*(size_t)(4), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk){
        /* The unroll pragmas here and in debye_polynomial.h do not reassociate. */
        #pragma GCC unroll 4
for(int w=0;w<(4);w++){
            size_t j=i+(size_t)w*VEC_LANES;
            SIMD_TYPE_FLOAT Zh,Zl;
            bessel_derive_argument(VEC_VLD(h+j),VEC_VLD(nu+j),VEC_VLD(scale+j),&Zh,&Zl);
            VEC_VST(res+j, debye_cell(Zh,Zl,VEC_VLD(nu+j)));
        }
    }
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES){
        SIMD_TYPE_FLOAT Zh,Zl;
        bessel_derive_argument(VEC_VLD(h+i),VEC_VLD(nu+i),VEC_VLD(scale+i),&Zh,&Zl);
        VEC_VST(res+i, debye_cell(Zh,Zl,VEC_VLD(nu+i)));
    }
    for(;i<(size_t)n;i++){
        SIMD_TYPE_FLOAT Zh,Zl;
        bessel_derive_argument(VEC_VSET(h[i]),VEC_VSET(nu[i]),VEC_VSET(scale[i]),&Zh,&Zl);
        res[i]=vec_first_lane(debye_cell(Zh,Zl,VEC_VSET(nu[i])));
    }
}

/* Extended range, z > 708.396418532264 at order >= 30: the same kernel, since -nu*eta is
 * already a double-double exponent, plus one select mapping a non-finite z to 0, the
 * covariance at infinite separation.  A separate entry keeps the select off the in-range
 * path. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT debye_extended_cell(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT ZL, SIMD_TYPE_FLOAT NU){
    SIMD_TYPE_FLOAT v = debye_cell(Z, ZL, NU);
    return VEC_SELECT_LT(v, VEC_VSET(0.0), Z, VEC_VSET(INFINITY));   /* z inf/NaN -> 0 */
}

static void matern_debye_extended_x1(double *res,double *h,double *nu,double *scale,int n){
    size_t blk=(size_t)VEC_LANES*(size_t)(1), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk){
        #pragma GCC unroll 4
for(int w=0;w<(1);w++){
            size_t j=i+(size_t)w*VEC_LANES;
            SIMD_TYPE_FLOAT Zh,Zl;
            bessel_derive_argument(VEC_VLD(h+j),VEC_VLD(nu+j),VEC_VLD(scale+j),&Zh,&Zl);
            VEC_VST(res+j, debye_extended_cell(Zh,Zl,VEC_VLD(nu+j)));
        }
    }
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES){
        SIMD_TYPE_FLOAT Zh,Zl;
        bessel_derive_argument(VEC_VLD(h+i),VEC_VLD(nu+i),VEC_VLD(scale+i),&Zh,&Zl);
        VEC_VST(res+i, debye_extended_cell(Zh,Zl,VEC_VLD(nu+i)));
    }
    for(;i<(size_t)n;i++){
        SIMD_TYPE_FLOAT Zh,Zl;
        bessel_derive_argument(VEC_VSET(h[i]),VEC_VSET(nu[i]),VEC_VSET(scale[i]),&Zh,&Zl);
        res[i]=vec_first_lane(debye_extended_cell(Zh,Zl,VEC_VSET(nu[i])));
    }
}

#endif /* MATERN_DEBYE_UNIFORM_ORDER_H */
