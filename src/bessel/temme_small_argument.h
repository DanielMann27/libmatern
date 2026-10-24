/* temme_small_argument.h -- small-argument (Temme) kernel in the vector vocabulary:
 * Temme's series at orders mu and mu + 1 (nu = nf + mu, |mu| <= 1/2), then the order
 * recurrence with the Matern prefactor folded in, g_a = 2 (z/2)^a K_a / Gamma(a) (see the
 * accompanying paper, Temme formulation). */
#ifndef MATERN_TEMME_SMALL_ARGUMENT_H
#define MATERN_TEMME_SMALL_ARGUMENT_H

#include <float.h>
#include <math.h>
#include <stddef.h>

#include "vector_current.h"
#include "engines_current.h"
#include "reciprocal_gamma_coefficients.h"

/* Hard stop for the Temme series, behind its convergence test. */
#define SMALL_ARGUMENT_MAX_TERMS 20

/* The second series' term c_k (p_k - k f_k) is rounded before it is added, at every width.
 * AVX2 and AVX-512 keep it apart through their blend.  At width 1 the blend folds away
 * inside the loop, and with FMA (matched_fma) -ffp-contract=fast would fuse the product
 * into the sum; the empty asm hands the add a rounded double instead. */
#if defined(MATERN_TARGET_SCALAR) && defined(__FMA__)
static inline __attribute__((always_inline)) double small_argument_rounded(double x)
{
    __asm__("" : "+x"(x));
    return x;
}
#define SMALL_ARGUMENT_ROUNDED(X) small_argument_rounded(X)
#else
#define SMALL_ARGUMENT_ROUNDED(X) (X)
#endif


/* z = sqrt(2 nu) h / s in a single double, derived in register; at z < 1 its ~z*eps error
 * is negligible. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT small_argument_derive(SIMD_TYPE_FLOAT H, SIMD_TYPE_FLOAT NU,
                                      SIMD_TYPE_FLOAT R)
{
#ifdef MATERN_ENTRY_Z
    (void)NU; (void)R;
    return H;                      /* MATERN_ENTRY_Z: z is the input */
#else
    return VEC_DIV(VEC_MUL(VEC_SQRT(VEC_MUL(VEC_VSET(2.0), NU)), H), R);
#endif
}

/* gam1 = (1/Gamma(1-mu) - 1/Gamma(1+mu)) / (2 mu) (odd part, negated after summation) and
 * gam2 = (1/Gamma(1-mu) + 1/Gamma(1+mu)) / 2 (even part) as cancellation-free series in
 * mu^2, regular at mu = 0; 1/Gamma(1+mu) = gam2 - mu gam1 and
 * 1/Gamma(1-mu) = gam2 + mu gam1, one FMA each.  The loops hard-code the 11 terms of
 * RECIPROCAL_GAMMA_ODD_TERMS and RECIPROCAL_GAMMA_EVEN_TERMS. */
static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT reciprocal_gamma_odd_series(SIMD_TYPE_FLOAT m2){
    SIMD_TYPE_FLOAT s = VEC_VSET(RECIPROCAL_GAMMA_ODD_B[10]);
    for (int j = 9; j >= 0; j--) s = VEC_FMA(s, m2, VEC_VSET(RECIPROCAL_GAMMA_ODD_B[j]));
    return VEC_SUB(VEC_VSET(0.0), s);                 /* gam1 = -SUM b_j m^(2j) */
}

static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT reciprocal_gamma_even_series(SIMD_TYPE_FLOAT m2){
    SIMD_TYPE_FLOAT s = VEC_VSET(RECIPROCAL_GAMMA_EVEN_C[10]);
    for (int j = 9; j >= 0; j--) s = VEC_FMA(s, m2, VEC_VSET(RECIPROCAL_GAMMA_EVEN_C[j]));
    return s;
}


/* ln x for every x > 0: the vector logarithm engines take normal input only, so lanes below
 * DBL_MIN are scaled by 2^54, exactly, and 54 ln 2 is subtracted. */
static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT small_argument_log_any(SIMD_TYPE_FLOAT X){
    const MASK_FLOAT sub = LESS_THAN_FLOAT(X, VEC_VSET(DBL_MIN));
    const SIMD_TYPE_FLOAT l = matern_logarithm_pd(BLEND_FLOAT(sub, X, MUL_FLOAT(X, VEC_VSET(0x1p54))));
    return BLEND_FLOAT(sub, l, SUB_FLOAT(l, VEC_VSET(0x1.2b708872320e2p+5)));   /* 54 ln 2 */
}

/* ln z and z/2 (*X2) for every z >= 0; a vector of normal lanes takes the plain calls.  Lanes
 * below DBL_MIN: on the (z, nu) entry ln z of the input, by rescaling; on the (h, nu, s) entry
 * the derived z has lost bits or underflowed to 0, so ln z = ln h - ln s + ln(2 nu)/2 from the
 * inputs and z/2 = exp(ln z - ln 2), which is 0 in effect unless an intermediate product of
 * the derivation underflowed, as on the GPU. */
static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT small_argument_log(SIMD_TYPE_FLOAT Z,
        SIMD_TYPE_FLOAT NU, SIMD_TYPE_FLOAT H, SIMD_TYPE_FLOAT R, SIMD_TYPE_FLOAT *X2){
    const MASK_FLOAT sub = LESS_THAN_FLOAT(Z, VEC_VSET(DBL_MIN));
    *X2 = DIV_FLOAT(Z, VEC_VSET(2.0));
    if (MASK_REDUCE(sub) == 0) return matern_logarithm_pd(Z);
#ifdef MATERN_ENTRY_Z
    (void)NU; (void)H; (void)R;
    return small_argument_log_any(Z);
#else
    const SIMD_TYPE_FLOAT lz = ADD_FLOAT(SUB_FLOAT(small_argument_log_any(H), small_argument_log_any(R)),
                                         MUL_FLOAT(VEC_VSET(0.5), small_argument_log_any(ADD_FLOAT(NU, NU))));
    const SIMD_TYPE_FLOAT lx2 = VEC_MAX(SUB_FLOAT(lz, VEC_VSET(0.69314718055994530942)), VEC_VSET(-700.0));
    *X2 = BLEND_FLOAT(sub, *X2, matern_exponential_pd(lx2));
    return BLEND_FLOAT(sub, matern_logarithm_pd(Z), lz);
#endif
}

/* Order-recurrence trip count: ceil(max over lanes), clamped at 0; lanes that arrive
 * earlier are held by a mask. */
static inline int small_argument_trip_count(SIMD_TYPE_FLOAT d){
    double v = vec_reduce_max(d);
    int t = (int)ceil(v);
    return t > 0 ? t : 0;
}

/* One cell: stage 1 splits nu = nf + mu and forms the seeds f_0, p_0, q_0; stage 2 sums
 * Temme's series at orders mu and mu + 1; stage 3 carries the folded pair up to order nu. */
static inline __attribute__((always_inline)) SIMD_TYPE_FLOAT small_argument_cell(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT NU,
                                                                                 SIMD_TYPE_FLOAT H, SIMD_TYPE_FLOAT R){
    const SIMD_TYPE_FLOAT _HALF =LOAD_FLOAT(MATERN_BROADCAST_HALF);
    const SIMD_TYPE_FLOAT _ONE =LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT _ZERO =LOAD_FLOAT(MATERN_BROADCAST_POOL);
    const SIMD_TYPE_FLOAT _LOG2 =LOAD_FLOAT(MATERN_BROADCAST_LOG2);
    const SIMD_TYPE_FLOAT _TOL =LOAD_FLOAT(MATERN_BROADCAST_TOL);
    const SIMD_TYPE_FLOAT _ETHR =VEC_VSET(0.1);
    const SIMD_TYPE_FLOAT _M2 =VEC_VSET(1.0/6.0);
    const SIMD_TYPE_FLOAT _M4 =VEC_VSET(1.0/120.0);
    const SIMD_TYPE_FLOAT _M6 =VEC_VSET(1.0/5040.0);
    const SIMD_TYPE_FLOAT _M8 =VEC_VSET(1.0/362880.0);
    SIMD_TYPE_FLOAT F,PK,QK,X4,NUNU,NU2;
    /* stage 1: the seed, with NU2 = mu and NUNU = mu^2; the do { } while(0) runs once and
     * only scopes the temporaries. */
    do {
        const SIMD_TYPE_FLOAT _nf =FLOOR(ADD_FLOAT((NU),_HALF));
        const SIMD_TYPE_FLOAT _nu2=SUB_FLOAT((NU),_nf);
        const SIMD_TYPE_FLOAT _nunu=MUL_FLOAT(_nu2,_nu2);
        const SIMD_TYPE_FLOAT _gam1=reciprocal_gamma_odd_series(_nunu);
        const SIMD_TYPE_FLOAT _gam2=reciprocal_gamma_even_series(_nunu);
        const SIMD_TYPE_FLOAT _gampl=FNMADD_FLOAT(_nu2,_gam1,_gam2);
        const SIMD_TYPE_FLOAT _gammi=FMADD_FLOAT(_nu2,_gam1,_gam2);
        const SIMD_TYPE_FLOAT _g=MUL_FLOAT(_gampl,_gammi);
        const SIMD_TYPE_FLOAT _r=DIV_FLOAT(_ONE,_g);
        /* one divide gives Gamma(1+mu) Gamma(1-mu) = pi mu / sin(pi mu) and each gamma */
        const SIMD_TYPE_FLOAT _fact=_r;
        const SIMD_TYPE_FLOAT _F5=MUL_FLOAT(_gammi,_r);
        const SIMD_TYPE_FLOAT _F6=MUL_FLOAT(_gampl,_r);
        SIMD_TYPE_FLOAT _x2;
        const SIMD_TYPE_FLOAT _log=small_argument_log(Z,NU,H,R,&_x2);
        const SIMD_TYPE_FLOAT _logs=SUB_FLOAT(_LOG2,_log);
        const SIMD_TYPE_FLOAT _e=MUL_FLOAT(_nu2,_logs);
        const SIMD_TYPE_FLOAT _znu=matern_exponential_pd(SUB_FLOAT(_ZERO,_e));
        const SIMD_TYPE_FLOAT _E=DIV_FLOAT(_ONE,_znu);
        const SIMD_TYPE_FLOAT _cosh=MUL_FLOAT(_HALF,ADD_FLOAT(_znu,_E));
        const SIMD_TYPE_FLOAT _e2=MUL_FLOAT(_e,_e);
        /* sinh(e)/e: the closed form _ex subtracts two nearly equal exponentials and
         * loses accuracy as e -> 0 (e vanishes with mu), so for |e| < 0.1 the Taylor
         * series _mac = 1 + e^2/3! + e^4/5! + e^6/7! + e^8/9! is selected. */
        SIMD_TYPE_FLOAT _mac=FMADD_FLOAT(_M8,_e2,_M6);
        _mac=FMADD_FLOAT(_mac,_e2,_M4);
        _mac=FMADD_FLOAT(_mac,_e2,_M2);
        _mac=FMADD_FLOAT(_mac,_e2,_ONE);
        const SIMD_TYPE_FLOAT _ex=MUL_FLOAT(_HALF,DIV_FLOAT(SUB_FLOAT(_E,_znu),_e));
        const MASK_FLOAT _msm=LESS_THAN_FLOAT(ABS(_e),_ETHR);
        const SIMD_TYPE_FLOAT _fact2=BLEND_FLOAT(_msm,_ex,_mac);
        const SIMD_TYPE_FLOAT _x4=MUL_FLOAT(_x2,_x2);
        const SIMD_TYPE_FLOAT _pk=MUL_FLOAT(_HALF,MUL_FLOAT(_E,_F5));
        const SIMD_TYPE_FLOAT _qk=MUL_FLOAT(_HALF,MUL_FLOAT(_znu,_F6));
        /* p_0 = (2/z)^mu Gamma(1+mu)/2,  q_0 = (z/2)^mu Gamma(1-mu)/2, and
         * f_0 = Gamma(1+mu) Gamma(1-mu) [gam1 cosh(e) + gam2 (sinh(e)/e) ln(2/z)],
         * every factor finite as mu -> 0. */
        const SIMD_TYPE_FLOAT _t3=MUL_FLOAT(_gam1,_cosh);
        const SIMD_TYPE_FLOAT _t4=MUL_FLOAT(_gam2,_fact2);
        const SIMD_TYPE_FLOAT _t5=MUL_FLOAT(_t4,_logs);
        const SIMD_TYPE_FLOAT _t6=ADD_FLOAT(_t3,_t5);
        (F)=MUL_FLOAT(_fact,_t6);
        (PK)=_pk;
        (QK)=_qk;
        (X4)=_x4;
        (NUNU)=_nunu;
        (NU2)=_nu2;
    } while(0);
    /* ZG = 1/p_0 = 2 (z/2)^mu / Gamma(1+mu) turns the series sums into g; no prefactor
     * is applied later. */
    const SIMD_TYPE_FLOAT ZG = vec_divide_refined(_ONE, PK);
    SIMD_TYPE_FLOAT CK=_ONE, SUM1=_ZERO, SUM2=_ZERO, FAC1=_ZERO, FAC2=_ZERO;
    SIMD_TYPE_FLOAT FLOATS=_ZERO;
    MASK_FLOAT M3=LESS_THAN_FLOAT(_ONE,_TOL);   /* converged lanes; initially none */
    /* stage 2: SUM1 = sum c_j f_j and SUM2 = sum c_j (p_j - j f_j), so that ZG mu SUM1 = g_mu
     * and ZG SUM2 = g_(mu+1); with k = FLOATS + 1 = T10,
     *   f_k = (k f_(k-1) + p_(k-1) + q_(k-1))/(k^2 - mu^2),   c_k = c_(k-1) (z/2)^2/k,
     *   p_k = p_(k-1)/(k - mu),                               q_k = q_(k-1)/(k + mu).
     * Converged lanes add zero through the M3 blend.  Iteration j adds term j-1, so the
     * series runs terms 0 .. SMALL_ARGUMENT_MAX_TERMS-1, as the scalar and CUDA kernels. */
    for (int j=1;j<=SMALL_ARGUMENT_MAX_TERMS;j++){
        FAC1=BLEND_FLOAT(M3,MUL_FLOAT(CK,F),_ZERO);
        const SIMD_TYPE_FLOAT T9=SUB_FLOAT(PK,MUL_FLOAT(FLOATS,F));
        FAC2=BLEND_FLOAT(M3,SMALL_ARGUMENT_ROUNDED(MUL_FLOAT(CK,T9)),_ZERO);
        SUM1=ADD_FLOAT(SUM1,FAC1);
        SUM2=ADD_FLOAT(SUM2,FAC2);
        /* |term| < eps |sum| on the first series, as the scalar and CUDA kernels */
        M3=LESS_THAN_FLOAT(ABS(FAC1),MUL_FLOAT(_TOL,ABS(SUM1)));
        if (MASK_REDUCE(M3)==WHILE_FACTOR) break;
        const SIMD_TYPE_FLOAT T10=ADD_FLOAT(FLOATS,_ONE);
        const SIMD_TYPE_FLOAT T11=MUL_FLOAT(T10,T10);
        const SIMD_TYPE_FLOAT D1=SUB_FLOAT(T11,NUNU);
        const SIMD_TYPE_FLOAT N1=ADD_FLOAT(FMADD_FLOAT(T10,F,PK),QK);
        F =vec_divide_refined(N1,D1);
        PK=vec_divide_refined(PK,SUB_FLOAT(T10,NU2));
        QK=vec_divide_refined(QK,ADD_FLOAT(T10,NU2));
        CK=vec_divide_refined(MUL_FLOAT(CK,X4),T10);
        FLOATS=ADD_FLOAT(FLOATS,_ONE);
    }
    /* stage 3: from order mu + 1 to nu.  G = g_a and V = g_(a-1) (z/2)^2/(a-1) at the
     * current order a; one step is g_(a+1) = g_a + V/a, V <- g_a (z/2)^2/a.  Each lane
     * captures G into OUT until it reaches its own order (OUT stays g_mu when nf = 0). */
    SIMD_TYPE_FLOAT G = MUL_FLOAT(ZG,SUM2);                  /* g at order mu + 1 */
    SIMD_TYPE_FLOAT V = MUL_FLOAT(MUL_FLOAT(ZG,SUM1),X4);    /* g_mu (z/2)^2 / mu */
    SIMD_TYPE_FLOAT OUT = MUL_FLOAT(MUL_FLOAT(ZG,NU2),SUM1); /* g at order mu     */
    SIMD_TYPE_FLOAT NU3 = ADD_FLOAT(NU2,_ONE);               /* the current order a */
    const SIMD_TYPE_FLOAT NH=ADD_FLOAT(NU,_HALF);
    MASK_FLOAT MK=LESS_THAN_FLOAT(NU3,NH);                   /* lane still climbing */
    const int TRIP=small_argument_trip_count(SUB_FLOAT(NH,NU3));
    for (int _s=0;_s<TRIP;_s++){
        OUT=BLEND_FLOAT(MK,OUT,G);
        const SIMD_TYPE_FLOAT R=vec_divide_refined(_ONE,NU3);
        const SIMD_TYPE_FLOAT GN=FMADD_FLOAT(V,R,G);
        V = MUL_FLOAT(MUL_FLOAT(G,X4),R);
        G = GN;
        NU3=ADD_FLOAT(NU3,_ONE);
        MK=LESS_THAN_FLOAT(NU3,NH);
    }
    return OUT;                                              /* the covariance at nu */
}
static void matern_small_argument_x4(double *res,double *h,double *nu,double *scale,int n){
    size_t blk=(size_t)VEC_LANES*(size_t)(4), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk){
        for(int q=0;q<(4);q++){
            size_t j=i+(size_t)q*VEC_LANES;
            VEC_VST(res+j, small_argument_cell(small_argument_derive(VEC_VLD(h+j), VEC_VLD(nu+j), VEC_VLD(scale+j)),VEC_VLD(nu+j),VEC_VLD(h+j),VEC_VLD(scale+j)));
        }
    }
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES) VEC_VST(res+i, small_argument_cell(small_argument_derive(VEC_VLD(h+i), VEC_VLD(nu+i), VEC_VLD(scale+i)),VEC_VLD(nu+i),VEC_VLD(h+i),VEC_VLD(scale+i)));
    for(;i<(size_t)n;i++) res[i]=vec_first_lane(small_argument_cell(small_argument_derive(VEC_VSET(h[i]), VEC_VSET(nu[i]), VEC_VSET(scale[i])),VEC_VSET(nu[i]),VEC_VSET(h[i]),VEC_VSET(scale[i])));
}

#endif /* MATERN_TEMME_SMALL_ARGUMENT_H */
