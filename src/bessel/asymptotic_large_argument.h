/* asymptotic_large_argument.h -- covariance for large z in the vector vocabulary: the
 * asymptotic series of K_nu in 1/z with a first-order z_lo correction, and a single-exponent
 * extended-range assembly (see the accompanying paper, large-argument formulation). */
#ifndef MATERN_ASYMPTOTIC_LARGE_ARGUMENT_H
#define MATERN_ASYMPTOTIC_LARGE_ARGUMENT_H

#include <float.h>
#include <stddef.h>

#include "vector_current.h"
#include "derive_argument.h"
#include "engines_current.h"
#include "large_argument_coefficients.h"

/* The term budget is two-sided: too few terms truncate the series, too many walk past its
 * minimum into the divergent tail, which the test at chunk boundaries cannot stop. */
#define LARGE_ARGUMENT_CHUNKS    7    /* 8-term chunks, truncated at LARGE_ARGUMENT_MAX_TERM */
#define LARGE_ARGUMENT_MAX_TERM 50    /* hard ceiling on the term index */

/* Converged when |t| <= tol |s| in every lane.  The test is relative to the partial sum of
 * Sigma = sqrt(2z/pi) e^z K_nu(z), since an absolute test cannot fire where Sigma is large.
 * Converged lanes keep accumulating; above the region border their further terms are below eps. */
static inline int large_argument_converged(const SIMD_TYPE_FLOAT *t,
                                           const SIMD_TYPE_FLOAT *s,
                                           SIMD_TYPE_FLOAT tol, int width)
{
    for (int q = 0; q < width; q++)
        if (MASK_ANY(MASK_GREATER(VEC_ABS(t[q]), VEC_MUL(tol, VEC_ABS(s[q])))))
            return 0;

    return 1;
}



/* M = 2 (z/2)^nu / Gamma(nu) * sqrt(pi/2z) e^-z * S, each factor formed separately: safe
 * because the router sends only nu <= 30 and z <= 708.396 here.  The term budget
 * (LARGE_ARGUMENT_MAX_TERM) also holds only up to nu = 30 (LARGE_ARGUMENT_ORDER_CAP). */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT large_argument_assemble(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT NU,
                              SIMD_TYPE_FLOAT S){
    SIMD_TYPE_FLOAT pref = VEC_MUL(VEC_VSET(2.0),
        VEC_DIV(matern_power_pd(VEC_MUL(Z, VEC_VSET(0.5)), NU), matern_gamma_pd(NU)));
    SIMD_TYPE_FLOAT kn = VEC_MUL(VEC_MUL(
        VEC_DIV(VEC_VSET(SQRT_PI_OVER_2), VEC_SQRT(Z)), S),
        matern_exponential_pd(VEC_SUB(VEC_VSET(0.0), Z)));
    return VEC_MUL(pref, kn);
}

/* With the first-order z_lo correction F = 1 + ((nu-0.5)/z - 1) z_lo, reusing W = 1/z.
 * F omits the Sigma'/Sigma term of d lnM/dz, which would need dSigma/dz; near nu = 30,
 * z = 22 that term exceeds the kept ones with opposite sign, and F enlarges the z_lo error. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT large_argument_assemble_corrected(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT ZL, SIMD_TYPE_FLOAT NU,
                                  SIMD_TYPE_FLOAT S, SIMD_TYPE_FLOAT W){
    SIMD_TYPE_FLOAT r  = large_argument_assemble(Z, NU, S);
    SIMD_TYPE_FLOAT cf = VEC_FMA_CANCEL(VEC_SUB(NU, VEC_VSET(0.5)), W, VEC_VSET(-1.0)); /* (nu-1/2)/z - 1 */
    return VEC_FMA(VEC_MUL(r, cf), ZL, r);                                              /* r (1 + cf z_lo) */
}

/* One cell: term ratio fq = (4 nu^2 A[k] + B[k]) / z, term t_ and partial sum s_ from a_0 = 1.
 * The do { } while(0) is the target of the chunk-boundary breaks; terms run
 * k = 1 .. LARGE_ARGUMENT_MAX_TERM - 1, the guard in the last chunk stopping it early. */
static inline __attribute__((always_inline)) void large_argument_cell(const SIMD_TYPE_FLOAT *Hv, const SIMD_TYPE_FLOAT *NUv, const SIMD_TYPE_FLOAT *Rv, SIMD_TYPE_FLOAT *out){
    const SIMD_TYPE_FLOAT one=VEC_VSET(1.0), tol=VEC_VSET(DBL_EPSILON);
    SIMD_TYPE_FLOAT Zh[1], Zl[1], w_[1], mu_[1], t_[1], s_[1];
    for(int q=0;q<(1);q++){
        bessel_derive_argument(Hv[q], NUv[q], Rv[q], &Zh[q], &Zl[q]);
        w_[q]=VEC_DIV(one, Zh[q]);
        mu_[q]=VEC_MUL(VEC_MUL(NUv[q],NUv[q]),VEC_VSET(4.0));
        t_[q]=one;
        s_[q]=one;
    }
    do{
        for(int kk=0; kk<8; kk++){
            const int k=(1)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=1 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(9)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=2 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(17)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=3 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(25)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=4 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(33)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=5 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(41)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=6 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(49)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
    }while(0);
    for(int q=0;q<(1);q++) out[q] = large_argument_assemble_corrected(Zh[q], Zl[q], NUv[q], s_[q], w_[q]);
}
static void matern_large_argument_x4(double *res,double *h,double *nu,double *scale,int n){
    size_t blk=(size_t)VEC_LANES*(size_t)(4), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk){
        for(int q=0;q<(4);q++){
            size_t j=i+(size_t)q*VEC_LANES;
            SIMD_TYPE_FLOAT h1[1],n1[1],r1[1],o1[1];
            h1[0]=VEC_VLD(h+j);
            n1[0]=VEC_VLD(nu+j);
            r1[0]=VEC_VLD(scale+j);
            large_argument_cell(h1,n1,r1,o1);
            VEC_VST(res+j, o1[0]);
        }
    }
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES){
        SIMD_TYPE_FLOAT h1[1],n1[1],r1[1],o1[1];
        h1[0]=VEC_VLD(h+i);
        n1[0]=VEC_VLD(nu+i);
        r1[0]=VEC_VLD(scale+i);
        large_argument_cell(h1,n1,r1,o1);
        VEC_VST(res+i, o1[0]);
    }
    for(;i<(size_t)n;i++){
        SIMD_TYPE_FLOAT h1[1],n1[1],r1[1],o1[1];
        h1[0]=VEC_VSET(h[i]);
        n1[0]=VEC_VSET(nu[i]);
        r1[0]=VEC_VSET(scale[i]);
        large_argument_cell(h1,n1,r1,o1);
        res[i]=vec_first_lane(o1[0]);
    }
}

/* Extended range, z > 708.396418532264, where e^-z leaves the normal range: every prefactor
 * is folded into one exponent, M = e^E Sigma with
 *   E = -z + (nu-0.5) ln z - lnGamma(nu) + (1-nu) ln2 + 0.5 ln(pi/2),
 * carried as a double-double since an absolute error in E is a relative error in M.
 * lnGamma is the dd log of the gamma engine's result, so a relative error in Gamma becomes
 * an absolute error of the same size in E, not amplified by nu.
 * The router sends only nu < 30 here. */
static inline __attribute__((always_inline))
SIMD_TYPE_FLOAT large_argument_extended_assemble(SIMD_TYPE_FLOAT Z, SIMD_TYPE_FLOAT ZL, SIMD_TYPE_FLOAT NU,
                                  SIMD_TYPE_FLOAT S, SIMD_TYPE_FLOAT W){
    const SIMD_TYPE_FLOAT zero = VEC_VSET(0.0);
    /* ln z in dd, z_lo absorbed: ln(zh+zl) = ln zh + zl/zh, with W = 1/z from the series. */
    SIMD_TYPE_FLOAT lzh, lzl; dd_log(Z, &lzh, &lzl);
    lzl = VEC_FMA(ZL, W, lzl);
    /* (nu-0.5) * ln z, dd */
    SIMD_TYPE_FLOAT a  = VEC_SUB(NU, VEC_VSET(0.5));
    SIMD_TYPE_FLOAT ph = VEC_MUL(a, lzh);
    SIMD_TYPE_FLOAT pl = VEC_FMA_CANCEL(a, lzl, VEC_FMS_CANCEL(a, lzh, ph));
    /* - z, exact in dd */
    SIMD_TYPE_FLOAT e1, x1; dd_two_sum(ph, VEC_SUB(zero, Z), &e1, &x1);
    SIMD_TYPE_FLOAT l1 = VEC_SUB(VEC_ADD(x1, pl), ZL);
    /* - lnGamma(nu) = - dd log(Gamma(nu)) */
    SIMD_TYPE_FLOAT gh, gl; dd_log(matern_gamma_pd(NU), &gh, &gl);
    SIMD_TYPE_FLOAT e2, x2; dd_two_sum(e1, VEC_SUB(zero, gh), &e2, &x2);
    SIMD_TYPE_FLOAT l2 = VEC_SUB(VEC_ADD(x2, l1), gl);
    /* + (1-nu) ln2, dd */
    SIMD_TYPE_FLOAT b  = VEC_SUB(VEC_VSET(1.0), NU);
    SIMD_TYPE_FLOAT qh = VEC_MUL(b, VEC_VSET(LN2_HI));
    SIMD_TYPE_FLOAT ql = VEC_FMA_CANCEL(b, VEC_VSET(LN2_LO),
                                 VEC_FMS_CANCEL(b, VEC_VSET(LN2_HI), qh));
    SIMD_TYPE_FLOAT e3, x3; dd_two_sum(e2, qh, &e3, &x3);
    SIMD_TYPE_FLOAT l3 = VEC_ADD(VEC_ADD(x3, l2), ql);
    /* + 0.5 ln(pi/2) */
    SIMD_TYPE_FLOAT Eh, x4; dd_two_sum(e3, VEC_VSET(HALF_LN_PI_OVER_2_HI), &Eh, &x4);
    SIMD_TYPE_FLOAT El = VEC_ADD(VEC_ADD(x4, l3), VEC_VSET(HALF_LN_PI_OVER_2_LO));
    /* The exponential engine is valid on [ln DBL_MIN, 708]; three branchless cases keep it there:
     *   Eh >= -708.396418532264 (ln DBL_MIN)   : the engine directly;
     *   -745.2 <= Eh < -708.396418532264       : engine(Eh/2)^2, reaching the subnormals;
     *   Eh < -745.2 (exp underflows to 0)      : exactly 0. */
    const SIMD_TYPE_FLOAT eLO = VEC_VSET(-708.396418532264);
    const SIMD_TYPE_FLOAT eZR = VEC_VSET(-745.2);
    SIMD_TYPE_FLOAT Eh2 = VEC_MUL(Eh, VEC_VSET(0.5));
    SIMD_TYPE_FLOAT arg = VEC_SELECT_LT(Eh2, Eh, Eh, eLO);              /* Eh<eLO ? Eh/2 : Eh */
    SIMD_TYPE_FLOAT e0  = matern_exponential_pd(arg);
    SIMD_TYPE_FLOAT eE  = VEC_SELECT_LT(VEC_MUL(e0,e0), e0, Eh, eLO);   /* ... ? e0^2 : e0   */
    eE = VEC_SELECT_LT(VEC_VSET(0.0), eE, Eh, eZR);                     /* Eh<eZR ? 0 : eE   */
    /* Folding extends the assembly's range, not the series' domain.  Two guards below return 0:
     * nu > 50 with z < nu^2/18 (the term budget does not reach the series minimum) and
     * nu > 171 (Gamma(nu) overflows).  Neither fires for the orders the router sends here. */
    /* Second order in El, since the dd_log low words are not normalised. */
    eE = VEC_FMA(eE, VEC_FMA(VEC_MUL(El, VEC_VSET(0.5)), El, El), eE);   /* exp(Eh+El) = exp(Eh)(1 + El + El^2/2) */
    SIMD_TYPE_FLOAT lzv = VEC_MUL(eE, S);
    { SIMD_TYPE_FLOAT nn = VEC_MUL(NU, NU);
      SIMD_TYPE_FLOAT zneed = VEC_MUL(nn, VEC_VSET(1.0/18.0));
      SIMD_TYPE_FLOAT z0 = VEC_VSET(0.0);
      SIMD_TYPE_FLOAT f2 = VEC_SELECT_LT(z0, lzv, Z, zneed);
      lzv = VEC_SELECT_LT(f2, lzv, VEC_VSET(50.0), NU);
      lzv = VEC_SELECT_LT(z0, lzv, VEC_VSET(171.0), NU); }
    /* Non-finite z (h = inf) returns 0; ln(inf) would make the exponent NaN. */
    lzv = VEC_SELECT_LT(lzv, VEC_VSET(0.0), Z, VEC_VSET(INFINITY));
    return lzv;
}

/* The series and its chunked exit are those of large_argument_cell; only the assembly differs. */
static inline __attribute__((always_inline)) void large_argument_extended_cell(const SIMD_TYPE_FLOAT *Hv, const SIMD_TYPE_FLOAT *NUv, const SIMD_TYPE_FLOAT *Rv, SIMD_TYPE_FLOAT *out){
    const SIMD_TYPE_FLOAT one=VEC_VSET(1.0), tol=VEC_VSET(DBL_EPSILON);
    SIMD_TYPE_FLOAT Zh[1], Zl[1], w_[1], mu_[1], t_[1], s_[1];
    for(int q=0;q<(1);q++){
        bessel_derive_argument(Hv[q], NUv[q], Rv[q], &Zh[q], &Zl[q]);
        w_[q]=VEC_DIV(one, Zh[q]);
        mu_[q]=VEC_MUL(VEC_MUL(NUv[q],NUv[q]),VEC_VSET(4.0));
        t_[q]=one;
        s_[q]=one;
    }
    do{
        for(int kk=0; kk<8; kk++){
            const int k=(1)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=1 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(9)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=2 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(17)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=3 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(25)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=4 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(33)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=5 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(41)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
        if((LARGE_ARGUMENT_CHUNKS)<=6 || large_argument_converged(t_, s_, tol, 1)) break;
        for(int kk=0; kk<8; kk++){
            const int k=(49)+kk;
            if(k>=LARGE_ARGUMENT_MAX_TERM) break;
            const double A=LARGE_ARGUMENT_RECURRENCE_A[k], B=LARGE_ARGUMENT_RECURRENCE_B[k];
            for(int q=0;q<(1);q++){
                {
                    SIMD_TYPE_FLOAT fq = VEC_FMA_CANCEL(mu_[q], VEC_VSET(A), VEC_VSET(B));
                    fq = VEC_MUL(fq, w_[q]);
                    t_[q] = VEC_MUL(t_[q], fq);
                    s_[q] = VEC_ADD(s_[q], t_[q]);
                }
            }
        }
    }while(0);
    for(int q=0;q<(1);q++) out[q] = large_argument_extended_assemble(Zh[q], Zl[q], NUv[q], s_[q], w_[q]);
}
/* The router calls this per element (n = 1) from region_map.h. */
static void matern_large_argument_extended_x1(double *res,double *h,double *nu,double *scale,int n){
    size_t blk=(size_t)VEC_LANES*(size_t)(1), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk){
        for(int q=0;q<(1);q++){
            size_t j=i+(size_t)q*VEC_LANES;
            SIMD_TYPE_FLOAT h1[1],n1[1],r1[1],o1[1];
            h1[0]=VEC_VLD(h+j);
            n1[0]=VEC_VLD(nu+j);
            r1[0]=VEC_VLD(scale+j);
            large_argument_extended_cell(h1,n1,r1,o1);
            VEC_VST(res+j, o1[0]);
        }
    }
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES){
        SIMD_TYPE_FLOAT h1[1],n1[1],r1[1],o1[1];
        h1[0]=VEC_VLD(h+i);
        n1[0]=VEC_VLD(nu+i);
        r1[0]=VEC_VLD(scale+i);
        large_argument_extended_cell(h1,n1,r1,o1);
        VEC_VST(res+i, o1[0]);
    }
    for(;i<(size_t)n;i++){
        SIMD_TYPE_FLOAT h1[1],n1[1],r1[1],o1[1];
        h1[0]=VEC_VSET(h[i]);
        n1[0]=VEC_VSET(nu[i]);
        r1[0]=VEC_VSET(scale[i]);
        large_argument_extended_cell(h1,n1,r1,o1);
        res[i]=vec_first_lane(o1[0]);
    }
}


#endif /* MATERN_ASYMPTOTIC_LARGE_ARGUMENT_H */
