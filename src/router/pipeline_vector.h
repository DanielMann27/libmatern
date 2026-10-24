/* pipeline_vector.h -- tiled vector router: per tile, A classify, B bucket into per-region
 * runs, C run each region's kernel over its run, D scatter back with a [0, 1] clamp (see
 * the accompanying paper, CPU implementation and SIMD strategy). */
#ifndef MATERN_PIPELINE_VECTOR_H
#define MATERN_PIPELINE_VECTOR_H

#include <float.h>
#include <string.h>

#include "internal.h"          /* struct matern_workspace */
#include "vector_current.h"
#include "region_map.h"
#include "asymptotic_large_argument.h"
#include "debye_uniform_order.h"
#include "temme_small_argument.h"
#include "levin_transformed_asymptotic.h"
#include "continued_fraction_wronskian.h"
#include "half_integer_closed_form.h"

/* Phase C dispatch table, one driver per region.  Both large-argument rows run the same
 * kernel.  REGION_TRIVIAL has no entry: matern_trivial_run (region_map.h) serves it. */
typedef void (*matern_region_fn)(double *, double *, double *, double *, int);

static const matern_region_fn MATERN_REGION_KERNELS[REGION_COUNT] = {
    [REGION_LARGE_ARGUMENT_LOW_ORDER] = matern_large_argument_x4,
    [REGION_LARGE_ARGUMENT]           = matern_large_argument_x4,
    [REGION_LEVIN]                    = matern_levin_x4,
    [REGION_CONTINUED_FRACTION]       = matern_continued_fraction_x4,
    [REGION_DEBYE]                    = matern_debye_x4,
    [REGION_SMALL_ARGUMENT]           = matern_small_argument_x4,
};

/* Phase A: matern_region_of (region_partition.h) without branches; the two must agree
 * (up to the rounding of the border curves), and the scalar one classifies the tile
 * remainder.  Each test is a mask and
 * each assignment a blend, and a later blend overrides an earlier one, so the tests run
 * in the reverse of the scalar cascade's order:
 *
 *   code = continued fraction            the scalar fall-through
 *     -> Levin                           nu <= 15 and z > L(nu)
 *     -> Debye                           nu >= 30
 *     -> small argument or Debye         z < 1, split at nu = 26
 *     -> large argument                  z > C(nu) and nu <= 30
 *     -> trivial                         past the exponential limit, or invalid */
static inline __attribute__((always_inline))
size_t matern_classify(const double *restrict h,const double *restrict nu,
                      const double *restrict scale,unsigned char *restrict rg,
                      size_t n,size_t *cnt)
{
    const SIMD_TYPE_FLOAT c36=VEC_VSET(1.0/36.0), c18=VEC_VSET(18.0);
#ifndef MATERN_ENTRY_Z
    const SIMD_TYPE_FLOAT cTWO=VEC_VSET(2.0);
#else
    const SIMD_TYPE_FLOAT dmax=VEC_VSET(DBL_MAX);   /* nu must be finite */
#endif
#ifndef MATERN_WITHOUT_LEVIN
    /* the Levin band: nu <= 15 and z above nu^4/2401 + 1.5 */
    const SIMD_TYPE_FLOAT c2401=VEC_VSET(1.0/2401.0), c15v=VEC_VSET(1.5), n15=VEC_VSET(15.0),
                          vLEV=VEC_VSET((double)REGION_LEVIN);
#endif
    const SIMD_TYPE_FLOAT zsmall=VEC_VSET(SMALL_ARGUMENT_Z_CEILING), ncap=VEC_VSET(LARGE_ARGUMENT_ORDER_CAP),
                          nsz=VEC_VSET(SMALL_ARGUMENT_ORDER_MAX);
    const SIMD_TYPE_FLOAT zceil=VEC_VSET(LARGE_ARGUMENT_Z_FLOOR), n8=VEC_VSET(GAMMA_RECURRENCE_ORDER);
    const SIMD_TYPE_FLOAT zmax=VEC_VSET(MATERN_EXP_RANGE_LIMIT), zero=VEC_VSET(0.0);
    const SIMD_TYPE_FLOAT vLZA=VEC_VSET((double)REGION_LARGE_ARGUMENT_LOW_ORDER), vLZ=VEC_VSET((double)REGION_LARGE_ARGUMENT),
                          vFR=VEC_VSET((double)REGION_CONTINUED_FRACTION),
                          vDB=VEC_VSET((double)REGION_DEBYE), vSZ=VEC_VSET((double)REGION_SMALL_ARGUMENT),
                          vTRIV=VEC_VSET((double)REGION_TRIVIAL);
    size_t i=0;
    /* Per-region counts by compare and lane count, in registers: a per-lane cnt[r]++ is a
     * store-to-load chain at a data-dependent address, on which AMD Zen 4 cores can settle,
     * at random per process and depending on code layout, into flushing and re-executing
     * (measured about 1 % of processes, 1.4-1.7x slower), as in phase B below.  At width 1
     * the compares would cost a few per cent, so matched counts per point as before. */
#if VEC_LANES > 1
    size_t lc[REGION_TOTAL]={0};
#endif
    for(; i+VEC_LANES<=n; i+=VEC_LANES){
        SIMD_TYPE_FLOAT H=VEC_VLD(h+i), NU=VEC_VLD(nu+i), R=VEC_VLD(scale+i);
#ifdef MATERN_ENTRY_Z
        SIMD_TYPE_FLOAT Z=H;       /* the (z, nu) entry: z is the input */
#else
        SIMD_TYPE_FLOAT Z=VEC_DIV(VEC_MUL(VEC_SQRT(VEC_MUL(cTWO,NU)),H),R);
#endif
        SIMD_TYPE_FLOAT NN=VEC_MUL(NU,NU);
        SIMD_TYPE_FLOAT C =VEC_FMA(NN,c36,c18);
        C = MASK_BLEND(MASK_LESS_EQUAL(NU,ncap), C, VEC_MIN(C, zceil));
        SIMD_TYPE_FLOAT code=vFR;
#ifndef MATERN_WITHOUT_LEVIN          /* must match matern_region_of */
        { SIMD_TYPE_FLOAT L =VEC_FMA(VEC_MUL(NN,NN),c2401,c15v);
          vec_mask m=MASK_AND(MASK_LESS_EQUAL(NU,n15),MASK_GREATER(Z,L));                code=MASK_BLEND(m,code,vLEV); }
#endif
        { vec_mask m=MASK_NOT(MASK_LESS(NU,ncap)); code=MASK_BLEND(m,code,vDB); }
        { vec_mask m=MASK_LESS(Z,zsmall);
          SIMD_TYPE_FLOAT s=MASK_BLEND(MASK_LESS_EQUAL(NU,nsz),vDB,vSZ);         code=MASK_BLEND(m,code,s); }
        { vec_mask m_lz=MASK_AND(MASK_GREATER(Z,C),MASK_LESS_EQUAL(NU,ncap));
          code=MASK_BLEND(m_lz,code,vLZ);
          { vec_mask ma=MASK_AND(m_lz,MASK_LESS(NU,n8)); code=MASK_BLEND(ma,code,vLZA); } }
        /* Trivial: z outside (0, limit] or invalid input, last so it overrides every
         * region.  Validity is tested on the inputs, not on a derived z: h < 0 with s < 0
         * gives a valid-looking z that matern_region_of cannot see.  The scalar tail
         * applies the same guard.  On the (h, nu, s) entry an infinite nu or s makes
         * z = inf, 0 or NaN, so the lane reaches the trivial bucket, whose guard rejects
         * it; on the (z, nu) entry nu is tested for finiteness here. */
#ifdef MATERN_ENTRY_Z
        { vec_mask m_tv=MASK_OR(MASK_OR(MASK_NOT(MASK_GREATER(Z,zero)),MASK_NOT(MASK_LESS_EQUAL(Z,zmax))),
                                MASK_NOT(MASK_AND(MASK_GREATER(NU,zero),MASK_LESS_EQUAL(NU,dmax))));
          code=MASK_BLEND(m_tv,code,vTRIV); }
        (void)R; (void)H;
#else
        { vec_mask m_tv=MASK_OR(MASK_OR(MASK_NOT(MASK_GREATER(Z,zero)),MASK_NOT(MASK_LESS_EQUAL(Z,zmax))),
                             MASK_OR(MASK_NOT(MASK_GREATER(NU,zero)),
                                   MASK_OR(MASK_NOT(MASK_GREATER(R,zero)),
                                         MASK_NOT(MASK_LESS_EQUAL(zero,H)))));   /* !(h>=0), NaN too */
          code=MASK_BLEND(m_tv,code,vTRIV); }
#endif
        double tmp[VEC_LANES]; VEC_VST(tmp,code);
#if VEC_LANES > 1
        for(int k=0;k<VEC_LANES;k++) rg[i+k]=(unsigned char)(int)tmp[k];
        #pragma GCC unroll 8
        for(int r=0;r<REGION_TOTAL;r++) lc[r]+=MASK_COUNT(MASK_EQUAL(code,VEC_VSET((double)r)));
#else
        for(int k=0;k<VEC_LANES;k++){ unsigned char r=(unsigned char)(int)tmp[k];
                                  rg[i+k]=r; cnt[r]++; }
#endif
    }
#if VEC_LANES > 1
    for(int r=0;r<REGION_TOTAL;r++) cnt[r]+=lc[r];
#endif
    return i;
}

/* The tiled pipeline over one caller-owned workspace. */
static inline __attribute__((always_inline))
void matern_route(double *restrict out,const double *restrict h,const double *restrict nu,
                 const double *restrict scale,size_t n,struct matern_workspace *ws)
{
    /* The continued-fraction workspace is published, not passed, because the dispatch
     * table's function type is shared.  Its buffers are tile-sized; that region never
     * receives more than one tile. */
    fraction_publish(ws->fraction_argument, ws->fraction_order,
                     ws->fraction_result,   ws->fraction_index, ws->tile);

    unsigned char *restrict rg = ws->region_tag;
    size_t        *restrict idx= ws->source_index;
    double        *restrict th = ws->sorted_h, *restrict tn = ws->sorted_nu,
                  *restrict tr = ws->sorted_scale, *restrict to = ws->sorted_out;
    const size_t T = ws->tile;

    for(size_t base=0; base<n; base+=T){
        const size_t m = (n-base < T) ? (n-base) : T;
        size_t cnt[REGION_TOTAL]={0}, off[REGION_TOTAL]={0};
        const double *hb=h+base, *nb=nu+base, *rb=scale+base;

        size_t i = matern_classify(hb,nb,rb,rg,m,cnt);                       /* A */
        /* tile remainder, with the same validity guard as the vector body; on the
         * (h, nu, s) entry matern_region_of sends an infinite nu or s to the trivial
         * bucket through z, as the vector body does */
        for(; i<m; i++){
#ifdef MATERN_ENTRY_Z
                         double z = hb[i];            /* h is z on this entry */
                         unsigned char r = (!(nb[i]>0.0 && nb[i]<=DBL_MAX) || !(hb[i]>=0.0))
#else
                         double z=sqrt(2.0*nb[i])*hb[i]/rb[i];
                         unsigned char r = (!(nb[i]>0.0) || !(rb[i]>0.0) || !(hb[i]>=0.0))
#endif
                                         ? (unsigned char)REGION_TRIVIAL
                                         : (unsigned char)matern_region_of(z,nb[i]);
                         rg[i]=r; cnt[r]++; }

        { size_t run=0; for(int r=0;r<REGION_TOTAL;r++){ off[r]=run; run+=cnt[r]; } }

        /* B on AVX-512: the run cursors are kept in one register (vpermq reads one, a
         * masked add advances it).  Kept in memory, the per-element pos[r]++ is a
         * store-to-load chain at a data-dependent address, on which AMD Zen 4 cores can
         * settle, at random per thread and depending on code layout, into flushing and
         * re-executing (measured up to 1.6x slower).  The positions are the same, so the
         * results are identical. */
#ifdef __AVX512F__
        { long long p8[8]={0}; for(int r=0;r<REGION_TOTAL;r++) p8[r]=(long long)off[r];
          __m512i posv=_mm512_loadu_si512((const void *)p8);
          const __m512i one=_mm512_set1_epi64(1), eight=_mm512_set1_epi64(8);
          size_t j=0;
          for(; j+8<=m; j+=8){                                               /* B */
            unsigned long long tw; memcpy(&tw,rg+j,sizeof tw);
            unsigned char r0=rg[j];
            if (tw==0x0101010101010101ull*(unsigned long long)r0){   /* whole vector, one region */
                size_t p=(size_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(
                             _mm512_permutexvar_epi64(_mm512_set1_epi64((long long)r0),posv)));
                for(int k=0;k<8;k++){ th[p+k]=hb[j+k]; tn[p+k]=nb[j+k];
                                      tr[p+k]=rb[j+k]; idx[p+k]=j+k; }
                posv=_mm512_mask_add_epi64(posv,(__mmask8)(1u<<r0),posv,eight);
            } else {
                for(int k=0;k<8;k++){ const unsigned r=rg[j+k];
                    size_t p=(size_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(
                                 _mm512_permutexvar_epi64(_mm512_set1_epi64((long long)r),posv)));
                    th[p]=hb[j+k]; tn[p]=nb[j+k]; tr[p]=rb[j+k]; idx[p]=j+k;
                    posv=_mm512_mask_add_epi64(posv,(__mmask8)(1u<<r),posv,one); }
            } }
          for(; j<m; j++){ const unsigned r=rg[j];
              size_t p=(size_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(
                           _mm512_permutexvar_epi64(_mm512_set1_epi64((long long)r),posv)));
              th[p]=hb[j]; tn[p]=nb[j]; tr[p]=rb[j]; idx[p]=j;
              posv=_mm512_mask_add_epi64(posv,(__mmask8)(1u<<r),posv,one); } }
#else
        { size_t pos[REGION_TOTAL]; for(int r=0;r<REGION_TOTAL;r++) pos[r]=off[r];
          size_t j=0;
          for(; j+8<=m; j+=8){                                               /* B */
            unsigned long long tw; memcpy(&tw,rg+j,sizeof tw);
            unsigned char r0=rg[j];
            if (tw==0x0101010101010101ull*(unsigned long long)r0){   /* whole vector, one region */
                size_t p=pos[r0];
                for(int k=0;k<8;k++){ th[p+k]=hb[j+k]; tn[p+k]=nb[j+k];
                                      tr[p+k]=rb[j+k]; idx[p+k]=j+k; }
                pos[r0]+=8;
            } else {
                for(int k=0;k<8;k++){ size_t p=pos[rg[j+k]]++;
                    th[p]=hb[j+k]; tn[p]=nb[j+k]; tr[p]=rb[j+k]; idx[p]=j+k; }
            } }
          for(; j<m; j++){ size_t p=pos[rg[j]]++;
              th[p]=hb[j]; tn[p]=nb[j]; tr[p]=rb[j]; idx[p]=j; } }
#endif

        for(int r=0;r<REGION_TOTAL;r++){                                          /* C */
            if(!cnt[r]) continue;
            if(r==REGION_TRIVIAL){
                matern_trivial_run(to+off[r], th+off[r], tn+off[r], tr+off[r], cnt[r]);
                continue;
            }
            MATERN_REGION_KERNELS[r](to+off[r], th+off[r], tn+off[r], tr+off[r], (int)cnt[r]);
        }
        /* D: scatter with a clamp to [0, 1].  Negative or NaN kernel output (an
         * underflowed tail, including the Debye fold's NaN below the subnormal range)
         * becomes 0.  Above 1 becomes 1: z is below the exponential limit here, so that
         * is rounding as z -> 0.  The trivial run, whose NaN marks invalid input, is
         * exempt: as the last region it is the tail of `to`, where the loop splits. */
        { const size_t nclamp = m - cnt[REGION_TRIVIAL];
          double *restrict ob=out+base;
          for(size_t j=0;j<nclamp;j++){ double v=to[j];
              if(!(v>0.0)) v=0.0; else if(v>1.0) v=1.0;   /* !(v>0) also catches NaN */
              ob[idx[j]]=v; }
          for(size_t j=nclamp;j<m;j++) ob[idx[j]]=to[j]; }
    }
}

#endif /* MATERN_PIPELINE_VECTOR_H */
