/* continued_fraction_wronskian.h -- continued-fraction kernel in the vector vocabulary (I
 * series, CF2 by modified Lentz, Wronskian, prefactor cancelled; see the accompanying
 * paper, continued-fraction formulation).  vec_divide_refined is per instruction set.  Each
 * lane stops accumulating at its own convergence test, so a point's result does not depend
 * on the points that share its block; the sort by z only groups similar trip counts. */
#ifndef MATERN_CONTINUED_FRACTION_WRONSKIAN_H
#define MATERN_CONTINUED_FRACTION_WRONSKIAN_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "vector_current.h"

/* Hard stops behind the convergence tests of the I series and the Lentz iteration. */
#define CONTINUED_FRACTION_SERIES_TERMS 50
#define CONTINUED_FRACTION_LENTZ_STEPS  50



/* Sort buffers from the caller's workspace, published by the router with fraction_publish.
 * The pointer is static in a header included only by the vector backend's single
 * translation unit, so the router and this kernel see the same object; it is
 * _Thread_local because threads route tiles concurrently. */
typedef struct {
    double   *argument;
    double   *order;
    double   *result;
    uint32_t *index;
} fraction_working_set;

static _Thread_local fraction_working_set fraction_published = { NULL, NULL, NULL, NULL };
static _Thread_local size_t               fraction_published_capacity = 0;

/* Called by the router once per routing call. */
static inline void fraction_publish(double *argument, double *order,
                                    double *result, uint32_t *index,
                                    size_t capacity)
{
    fraction_published.argument = argument;
    fraction_published.order    = order;
    fraction_published.result   = result;
    fraction_published.index    = index;
    fraction_published_capacity = capacity;
}

static inline fraction_working_set continued_fraction_scratch(size_t n)
{
    if (n <= fraction_published_capacity && fraction_published.argument)
        return fraction_published;
    const fraction_working_set none = { NULL, NULL, NULL, NULL };
    return none;
}

/* z = sqrt(2 nu) h / s for a whole batch, in caller order and in a single double (this
 * kernel uses no low word).  Under MATERN_ENTRY_Z the caller supplies z in h. */
static inline void continued_fraction_derive_arguments(double *z, const double *h,
                                                       const double *nu,
                                                       const double *scale, int n)
{
    const SIMD_TYPE_FLOAT two = VEC_VSET(2.0);
    int i = 0;

    for (; i + VEC_LANES <= n; i += VEC_LANES) {
        const SIMD_TYPE_FLOAT NU = VEC_VLD(nu + i);
        const SIMD_TYPE_FLOAT H  = VEC_VLD(h + i);
        const SIMD_TYPE_FLOAT R  = VEC_VLD(scale + i);
#ifdef MATERN_ENTRY_Z
        (void)NU; (void)R; (void)two;
        VEC_VST(z + i, H);         /* the (z, nu) entry: store the caller's z */
#else
        VEC_VST(z + i, VEC_DIV(VEC_MUL(VEC_SQRT(VEC_MUL(two, NU)), H), R));
#endif
    }

    /* Scalar remainder: same entry convention as the vector loop. */
    for (; i < n; i++)
#ifdef MATERN_ENTRY_Z
        z[i] = h[i];               /* the (z, nu) entry: h carries z */
#else
        z[i] = sqrt(2.0 * nu[i]) * h[i] / scale[i];
#endif
}

/* Sort key e = 4 floor(log2 z) + the two leading mantissa bits, read from the bit pattern.
 * The clamps fold z < 1 and z >= 256 into the end buckets, giving the 8 << 2 = 32 buckets
 * of the driver's cnt[] and off[].  Only the grouping depends on the key, not correctness. */
static inline int continued_fraction_bucket_of(double z) {
    uint64_t b;
    memcpy(&b, &z, sizeof b);
    int e = (int)((b >> (52 - (2))) & ((0x7FFULL << (2)) | ((1ULL << (2)) - 1))) - (1023 << (2));
    if (e < 0) e = 0;                        /* z < 1    -> bucket 0  */
    if (e >= (8 << (2))) e = (8 << (2)) - 1; /* z >= 256 -> bucket 31 */
    return e;
}
/* One cell: the covariance for VEC_LANES points.  Phase 1 seeds the series; phase 2 sums
 * O1 = G I_nu(z) and O2 = G I_(nu-1)(z), G = Gamma(nu) (z/2)^(-nu); phase 3 runs CF2 by
 * modified Lentz, LOW1 = CF and D = nu - 1 + z + 1/2 + CF = z K_nu/K_(nu-1); phase 4 closes
 * the Wronskian I_(nu-1) K_nu + I_nu K_(nu-1) = 1/z as M = 2 / (z (O2 + z O1/D)).
 * `w` indexes the vectors in flight and only unrolls: every vector gets the same arithmetic. */
static inline __attribute__((always_inline)) void continued_fraction_cell_x1(const double *restrict zp, const double *restrict np, double *restrict op) {
    const SIMD_TYPE_FLOAT TWO = LOAD_FLOAT(MATERN_BROADCAST_POOL + 2 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT ONE = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT ZERO = LOAD_FLOAT(MATERN_BROADCAST_POOL);
    const SIMD_TYPE_FLOAT TOL = LOAD_FLOAT(MATERN_BROADCAST_TOL);
    const SIMD_TYPE_FLOAT HALF = LOAD_FLOAT(MATERN_BROADCAST_HALF);
    const SIMD_TYPE_FLOAT QUARTER = LOAD_FLOAT(MATERN_BROADCAST_QUARTER);
    SIMD_TYPE_FLOAT Z[1], NU[1], Z4[1], S[1], T[1], O1[1], O2[1];
    vec_mask DS[1], DL[1];                           /* lanes past the series / Lentz test */
    for (int w = 0; w < 1; w++) {
        DS[w] = LESS_THAN_FLOAT(ONE, ZERO);          /* no lane has passed either test */
        DL[w] = DS[w];
    }
    /* phase 1: S_0 = 1/nu, T_0 = 2/z */
    #pragma GCC unroll 4
for (int w = 0; w < 1; w++){
        Z[w] = VEC_VLD(zp + (size_t)w * VEC_LANES);
        NU[w] = VEC_VLD(np + (size_t)w * VEC_LANES);
        const SIMD_TYPE_FLOAT ZH = MUL_FLOAT(Z[w], HALF);
        Z4[w] = MUL_FLOAT(ZH, ZH);
        S[w] = vec_divide_refined(ONE, NU[w]);
        T[w] = vec_divide_refined(ONE, ZH);
        O1[w] = ZERO;
        O2[w] = ZERO;
    }
    /* phase 2: with k = i + 1 the term ratios are (z/2)^2/(k (nu+k)) for S and
     * (z/2)^2/(k (nu+k-1)) for T.  A lane whose S term has become negligible adds 0 from
     * then on, which leaves its sums exactly as they were; the loop exits once every lane
     * has. */
    for (int i = 0; i < CONTINUED_FRACTION_SERIES_TERMS; i++){
        unsigned all = 1u;
        #pragma GCC unroll 4
for (int w = 0; w < 1; w++){
            O1[w] = ADD_FLOAT(O1[w], BLEND_FLOAT(DS[w], S[w], ZERO));
            O2[w] = ADD_FLOAT(O2[w], BLEND_FLOAT(DS[w], T[w], ZERO));
            DS[w] = MASK_OR(DS[w], LESS_THAN_EQUAL_FLOAT(ABS(S[w]), MUL_FLOAT(TOL, ABS(O1[w]))));
            if (MASK_REDUCE(DS[w]) != WHILE_FACTOR) all = 0u;
        }
        if (all) break;
        const SIMD_TYPE_FLOAT FAC1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + (i + 1) * INDEX_FACTOR);
        const SIMD_TYPE_FLOAT TMP2 = LOAD_FLOAT(MATERN_BROADCAST_POOL + i * INDEX_FACTOR);
        #pragma GCC unroll 4
for (int w = 0; w < 1; w++){
            const SIMD_TYPE_FLOAT FAC2 = ADD_FLOAT(NU[w], TMP2);
            const SIMD_TYPE_FLOAT FAC3 = ADD_FLOAT(NU[w], FAC1);
            S[w] = MUL_FLOAT(vec_divide_refined(S[w], MUL_FLOAT(FAC3, FAC1)), Z4[w]);
            T[w] = MUL_FLOAT(vec_divide_refined(T[w], MUL_FLOAT(FAC2, FAC1)), Z4[w]);
        }
    }
    SIMD_TYPE_FLOAT NUNU[1], LOW1[1], LOW2[1], H[1];
    /* phase 3 seed: MATERN_BROADCAST_LOW = 1e-50, the modified-Lentz starting value */
    #pragma GCC unroll 4
for (int w = 0; w < 1; w++){
        const SIMD_TYPE_FLOAT NUNEW = SUB_FLOAT(NU[w], ONE);
        NUNU[w] = MUL_FLOAT(NUNEW, NUNEW);
        LOW1[w] = LOAD_FLOAT(MATERN_BROADCAST_LOW);
        LOW2[w] = LOAD_FLOAT(MATERN_BROADCAST_LOW);
        H[w] = ZERO;
    }
    SIMD_TYPE_FLOAT C = ONE;
    /* phase 3: step k has A = (nu-1)^2 - (2k-1)^2/4 and B = 2(z+k).  A lane whose PROD
     * has reached 1 within TOL multiplies LOW1 by 1 from then on; the loop exits once every
     * lane has. */
    for (int i = 0; i < CONTINUED_FRACTION_LENTZ_STEPS; i++){
        const SIMD_TYPE_FLOAT TMP_T2 = SUB_FLOAT(MUL_FLOAT(TWO, C), ONE);
        const SIMD_TYPE_FLOAT TMP5 = MUL_FLOAT(QUARTER, MUL_FLOAT(TMP_T2, TMP_T2));
        unsigned all = 1u;
        #pragma GCC unroll 4
for (int w = 0; w < 1; w++){
            const SIMD_TYPE_FLOAT A = SUB_FLOAT(NUNU[w], TMP5);
            const SIMD_TYPE_FLOAT B = MUL_FLOAT(TWO, ADD_FLOAT(Z[w], C));
            LOW2[w] = ADD_FLOAT(B, vec_divide_refined(A, LOW2[w]));
            H[w] = vec_divide_refined(ONE, FMADD_FLOAT(A, H[w], B));
            const SIMD_TYPE_FLOAT PROD = MUL_FLOAT(H[w], LOW2[w]);
            LOW1[w] = MUL_FLOAT(LOW1[w], BLEND_FLOAT(DL[w], PROD, ONE));
            DL[w] = MASK_OR(DL[w], LESS_THAN_EQUAL_FLOAT(ABS(SUB_FLOAT(PROD, ONE)), TOL));
            if (MASK_REDUCE(DL[w]) != WHILE_FACTOR) all = 0u;
        }
        if (all) break;
        C = ADD_FLOAT(C, ONE);
    }
    /* phase 4: close the Wronskian and store the covariance */
    #pragma GCC unroll 4
for (int w = 0; w < 1; w++){
        const SIMD_TYPE_FLOAT D = ADD_FLOAT(ADD_FLOAT(ADD_FLOAT(SUB_FLOAT(NU[w], ONE), Z[w]), HALF), LOW1[w]);
        const SIMD_TYPE_FLOAT T15 = ADD_FLOAT(O2[w], vec_divide_refined(MUL_FLOAT(O1[w], Z[w]), D));
        VEC_VST(op + (size_t)w * VEC_LANES, vec_divide_refined(TWO, MUL_FLOAT(Z[w], T15)));
    }
}
/* continued_fraction_cell_x1 with four vectors in flight. */
static inline __attribute__((always_inline)) void continued_fraction_cell_x4(const double *restrict zp, const double *restrict np, double *restrict op) {
    const SIMD_TYPE_FLOAT TWO = LOAD_FLOAT(MATERN_BROADCAST_POOL + 2 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT ONE = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT ZERO = LOAD_FLOAT(MATERN_BROADCAST_POOL);
    const SIMD_TYPE_FLOAT TOL = LOAD_FLOAT(MATERN_BROADCAST_TOL);
    const SIMD_TYPE_FLOAT HALF = LOAD_FLOAT(MATERN_BROADCAST_HALF);
    const SIMD_TYPE_FLOAT QUARTER = LOAD_FLOAT(MATERN_BROADCAST_QUARTER);
    SIMD_TYPE_FLOAT Z[4], NU[4], Z4[4], S[4], T[4], O1[4], O2[4];
    vec_mask DS[4], DL[4];                           /* lanes past the series / Lentz test */
    for (int w = 0; w < 4; w++) {
        DS[w] = LESS_THAN_FLOAT(ONE, ZERO);          /* no lane has passed either test */
        DL[w] = DS[w];
    }
    #pragma GCC unroll 4
for (int w = 0; w < 4; w++){
        Z[w] = VEC_VLD(zp + (size_t)w * VEC_LANES);
        NU[w] = VEC_VLD(np + (size_t)w * VEC_LANES);
        const SIMD_TYPE_FLOAT ZH = MUL_FLOAT(Z[w], HALF);
        Z4[w] = MUL_FLOAT(ZH, ZH);
        S[w] = vec_divide_refined(ONE, NU[w]);
        T[w] = vec_divide_refined(ONE, ZH);
        O1[w] = ZERO;
        O2[w] = ZERO;
    }
    for (int i = 0; i < CONTINUED_FRACTION_SERIES_TERMS; i++){
        unsigned all = 1u;
        #pragma GCC unroll 4
for (int w = 0; w < 4; w++){
            O1[w] = ADD_FLOAT(O1[w], BLEND_FLOAT(DS[w], S[w], ZERO));
            O2[w] = ADD_FLOAT(O2[w], BLEND_FLOAT(DS[w], T[w], ZERO));
            DS[w] = MASK_OR(DS[w], LESS_THAN_EQUAL_FLOAT(ABS(S[w]), MUL_FLOAT(TOL, ABS(O1[w]))));
            if (MASK_REDUCE(DS[w]) != WHILE_FACTOR) all = 0u;
        }
        if (all) break;
        const SIMD_TYPE_FLOAT FAC1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + (i + 1) * INDEX_FACTOR);
        const SIMD_TYPE_FLOAT TMP2 = LOAD_FLOAT(MATERN_BROADCAST_POOL + i * INDEX_FACTOR);
        #pragma GCC unroll 4
for (int w = 0; w < 4; w++){
            const SIMD_TYPE_FLOAT FAC2 = ADD_FLOAT(NU[w], TMP2);
            const SIMD_TYPE_FLOAT FAC3 = ADD_FLOAT(NU[w], FAC1);
            S[w] = MUL_FLOAT(vec_divide_refined(S[w], MUL_FLOAT(FAC3, FAC1)), Z4[w]);
            T[w] = MUL_FLOAT(vec_divide_refined(T[w], MUL_FLOAT(FAC2, FAC1)), Z4[w]);
        }
    }
    SIMD_TYPE_FLOAT NUNU[4], LOW1[4], LOW2[4], H[4];
    #pragma GCC unroll 4
for (int w = 0; w < 4; w++){
        const SIMD_TYPE_FLOAT NUNEW = SUB_FLOAT(NU[w], ONE);
        NUNU[w] = MUL_FLOAT(NUNEW, NUNEW);
        LOW1[w] = LOAD_FLOAT(MATERN_BROADCAST_LOW);
        LOW2[w] = LOAD_FLOAT(MATERN_BROADCAST_LOW);
        H[w] = ZERO;
    }
    SIMD_TYPE_FLOAT C = ONE;
    for (int i = 0; i < CONTINUED_FRACTION_LENTZ_STEPS; i++){
        const SIMD_TYPE_FLOAT TMP_T2 = SUB_FLOAT(MUL_FLOAT(TWO, C), ONE);
        const SIMD_TYPE_FLOAT TMP5 = MUL_FLOAT(QUARTER, MUL_FLOAT(TMP_T2, TMP_T2));
        unsigned all = 1u;
        #pragma GCC unroll 4
for (int w = 0; w < 4; w++){
            const SIMD_TYPE_FLOAT A = SUB_FLOAT(NUNU[w], TMP5);
            const SIMD_TYPE_FLOAT B = MUL_FLOAT(TWO, ADD_FLOAT(Z[w], C));
            LOW2[w] = ADD_FLOAT(B, vec_divide_refined(A, LOW2[w]));
            H[w] = vec_divide_refined(ONE, FMADD_FLOAT(A, H[w], B));
            const SIMD_TYPE_FLOAT PROD = MUL_FLOAT(H[w], LOW2[w]);
            LOW1[w] = MUL_FLOAT(LOW1[w], BLEND_FLOAT(DL[w], PROD, ONE));
            DL[w] = MASK_OR(DL[w], LESS_THAN_EQUAL_FLOAT(ABS(SUB_FLOAT(PROD, ONE)), TOL));
            if (MASK_REDUCE(DL[w]) != WHILE_FACTOR) all = 0u;
        }
        if (all) break;
        C = ADD_FLOAT(C, ONE);
    }
    #pragma GCC unroll 4
for (int w = 0; w < 4; w++){
        const SIMD_TYPE_FLOAT D = ADD_FLOAT(ADD_FLOAT(ADD_FLOAT(SUB_FLOAT(NU[w], ONE), Z[w]), HALF), LOW1[w]);
        const SIMD_TYPE_FLOAT T15 = ADD_FLOAT(O2[w], vec_divide_refined(MUL_FLOAT(O1[w], Z[w]), D));
        VEC_VST(op + (size_t)w * VEC_LANES, vec_divide_refined(TWO, MUL_FLOAT(Z[w], T15)));
    }
}
static void matern_continued_fraction_x4(double *res,double *h,double *nu,double *scale,int n){
    if(n<=0) return;
    /* Obtain the scratch first: the derivation below writes into sorted_result. */
    const fraction_working_set scratch = continued_fraction_scratch((size_t)n);
    if (!scratch.argument) {                 /* no workspace published, or the run exceeds its capacity: no result */
        for (int k = 0; k < n; k++) res[k] = NAN;
        return;
    }
    double   *const sorted_argument = scratch.argument;
    double   *const sorted_order    = scratch.order;
    double   *const sorted_result   = scratch.result;
    uint32_t *const source_index    = scratch.index;
    /* z for the whole run, held in sorted_result until the scatter below. */
    continued_fraction_derive_arguments(sorted_result,h,nu,scale,n);
    /* Counting sort: tally, running-sum offsets, scatter; source_index records each
     * element's caller position for the unpermute. */
    size_t cnt[32], off[32];
    memset(cnt,0,sizeof cnt);
    for(size_t i=0;i<(size_t)n;i++) cnt[continued_fraction_bucket_of(sorted_result[i])]++;
    size_t acc=0;
    for(int b=0;b<(32);b++){
        off[b]=acc;
        acc+=cnt[b];
    }
    for(size_t i=0;i<(size_t)n;i++){
        size_t p=off[continued_fraction_bucket_of(sorted_result[i])]++;
        sorted_argument[p]=sorted_result[i];
        sorted_order[p]=nu[i];
        source_index[p]=(uint32_t)i;
    }
    /* Blocks of four vectors, single vectors, then one padded vector: the pad repeats
     * element r-1 instead of reading past the end, and its spare lanes are not stored. */
    size_t blk=(size_t)VEC_LANES*(size_t)(4), m=(size_t)n/blk*blk, i=0;
    for(;i<m;i+=blk) continued_fraction_cell_x4(sorted_argument+i, sorted_order+i, sorted_result+i);
    for(;i+VEC_LANES<=(size_t)n;i+=VEC_LANES) continued_fraction_cell_x1(sorted_argument+i,sorted_order+i,sorted_result+i);
    if(i<(size_t)n){
        double zb[VEC_LANES],nb[VEC_LANES],ob[VEC_LANES];
        size_t r=(size_t)n-i;
        for(size_t k=0;k<(size_t)VEC_LANES;k++){
            zb[k]=sorted_argument[i+(k<r?k:r-1)];
            nb[k]=sorted_order[i+(k<r?k:r-1)];
        }
        continued_fraction_cell_x1(zb,nb,ob);
        for(size_t k=0;k<r;k++) sorted_result[i+k]=ob[k];
    }
    /* Unpermute: each result returns to its caller position. */
    for(size_t k=0;k<(size_t)n;k++) res[source_index[k]]=sorted_result[k];
}

#endif /* MATERN_CONTINUED_FRACTION_WRONSKIAN_H */
