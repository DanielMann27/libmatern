/* gamma_avx2.h -- Gamma(x) for four doubles, x in (0, 171]: the method of gamma_avx512.h
 * with AVX2 table reads (matern_log_table_16) and the borrow logarithm reduction. */
#ifndef MATERN_GAMMA_AVX2_H
#define MATERN_GAMMA_AVX2_H

#include "simd_avx2.h"
#include "gamma_coefficients_avx2.h"
#include "logarithm_reduction_avx2.h"

/* 2^(j/8) lookup from 32-bit halves, as in exponential_avx2.h. */
static inline void matern_gamma_exp_table(__m256i j, __m256d *th, __m256d *tl)
{
    const __m256i idxLo = j;
    const __m256i idxHi = _mm256_slli_epi64(j, 32);
    __m256i th_lo = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)MATERN_GAMMA_ETH_LO32), idxLo);
    __m256i th_hi = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)MATERN_GAMMA_ETH_HI32), idxHi);
    *th = _mm256_castsi256_pd(_mm256_blend_epi32(th_lo, th_hi, 0xAA));
    __m256i tl_lo = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)MATERN_GAMMA_ETL_LO32), idxLo);
    __m256i tl_hi = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((const __m256i *)MATERN_GAMMA_ETL_HI32), idxHi);
    *tl = _mm256_castsi256_pd(_mm256_blend_epi32(tl_lo, tl_hi, 0xAA));
}

static inline __attribute__((always_inline)) __m256d matern_gamma_pd(__m256d X)
{
    /* 1. Double-double upward recurrence into [8, 9). */
    SIMD_TYPE_FLOAT XW_1 = X;
    SIMD_TYPE_FLOAT XL_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL);
    SIMD_TYPE_FLOAT PH_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    SIMD_TYPE_FLOAT PL_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL);
    {
        const SIMD_TYPE_FLOAT EIGHT_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + 8 * INDEX_FACTOR);
        const SIMD_TYPE_FLOAT ONE_1   = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
        for (int rs = 0; rs < 8; rs++) {
            const MASK_FLOAT       mm  = LESS_THAN_FLOAT(XW_1, EIGHT_1);
            if (!_mm256_movemask_pd(mm)) break;
            const SIMD_TYPE_FLOAT  ph  = MUL_FLOAT(PH_1, XW_1);
            const SIMD_TYPE_FLOAT  plo = FMADD_FLOAT(PH_1, XL_1, FMADD_FLOAT(PL_1, XW_1, FMSUB_FLOAT(PH_1, XW_1, ph)));
            const SIMD_TYPE_FLOAT  sh  = ADD_FLOAT(ph, plo);
            const SIMD_TYPE_FLOAT  sl  = ADD_FLOAT(SUB_FLOAT(ph, sh), plo);
            const SIMD_TYPE_FLOAT  ts  = ADD_FLOAT(XW_1, ONE_1);
            const SIMD_TYPE_FLOAT  bb  = SUB_FLOAT(ts, XW_1);
            const SIMD_TYPE_FLOAT  cc  = ADD_FLOAT(SUB_FLOAT(XW_1, SUB_FLOAT(ts, bb)), SUB_FLOAT(ONE_1, bb));
            const SIMD_TYPE_FLOAT  nl  = ADD_FLOAT(XL_1, cc);
            const SIMD_TYPE_FLOAT  xh  = ADD_FLOAT(ts, nl);
            const SIMD_TYPE_FLOAT  xll = ADD_FLOAT(SUB_FLOAT(ts, xh), nl);
            PH_1 = BLEND_FLOAT(mm, PH_1, sh);
            PL_1 = BLEND_FLOAT(mm, PL_1, sl);
            XW_1 = BLEND_FLOAT(mm, XW_1, xh);
            XL_1 = BLEND_FLOAT(mm, XL_1, xll);
        }
    }

    /* 2. ln XW as a double-double: accurate table, relaxed recomposition. */
    const __m256d L_one   = _mm256_set1_pd(1.0);
    const __m256d L_ln2hi = _mm256_set1_pd(0x1.62e42fee00000p-1);
    const __m256d L_ln2lo = _mm256_set1_pd(0x1.a39ef35793c76p-33);
    const __m256i L_base  = _mm256_set1_epi64x(0x3FE6555555555556LL);
    /* Borrow reduction, SUBC = frac(THRESH), THRESH = 0x1.6555555555556p+0: m in
     * [THRESH/2, THRESH); centre and index base are shared with the power reduction
     * and dd_log. */
    __m256d Lm, Le;
    matern_log_reduce_borrow(XW_1, 0x0006555555555556LL, &Lm, &Le);
    __m256i Lj    = _mm256_srli_epi64(_mm256_sub_epi64(_mm256_castpd_si256(Lm), L_base), 48);
    __m256i LidxLo = Lj;
    __m256i LidxHi = _mm256_slli_epi64(Lj, 32);
    __m256i Lm8    = _mm256_cmpgt_epi64(Lj, _mm256_set1_epi64x(7));
    __m256d Lc    = matern_log_table_16(MATERN_GAMMA_C_LO32A,    MATERN_GAMMA_C_LO32B,    MATERN_GAMMA_C_HI32A,    MATERN_GAMMA_C_HI32B,    LidxLo, LidxHi, Lm8);
    __m256d Llogc = matern_log_table_16(MATERN_GAMMA_LOGC_LO32A, MATERN_GAMMA_LOGC_LO32B, MATERN_GAMMA_LOGC_HI32A, MATERN_GAMMA_LOGC_HI32B, LidxLo, LidxHi, Lm8);
    __m256d Lph_ = _mm256_mul_pd(Lm, Lc);
    __m256d Lrh  = _mm256_sub_pd(Lph_, L_one);
    __m256d Lpl  = _mm256_fmsub_pd(Lm, Lc, Lph_);
    __m256d Lr   = _mm256_add_pd(Lrh, Lpl);
    __m256d Lr2  = _mm256_mul_pd(Lr, Lr);
    __m256d LB = _mm256_set1_pd( 0x1.7514b8968cd04p-4);
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd(-0x1.9a6087a41cb9bp-4));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd( 0x1.c71c4c5a59011p-4));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd(-0x1.ffffd77508470p-4));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd( 0x1.2492492620b04p-3));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd(-0x1.5555555704ed7p-3));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd( 0x1.99999999998e0p-3));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd(-0x1.fffffffffff37p-3));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd( 0x1.5555555555555p-2));
    LB = _mm256_fmadd_pd(LB, Lr, _mm256_set1_pd(-0x1.0000000000000p-1));
    __m256d Lqv  = _mm256_mul_pd(Lr2, LB);
    /* As in gamma_avx512.h: exact FastTwoSum chain A (+logc) (+Lrh), the small parts
     * summed into one low word. */
    __m256d Lplo = _mm256_add_pd(Lpl, Lqv);                            /* P(r) low */
    __m256d LA   = _mm256_mul_pd(Le, L_ln2hi);                         /* e*ln2hi exact */
    __m256d Lh1  = _mm256_add_pd(LA, Llogc);
    __m256d Ll1  = _mm256_sub_pd(Llogc, _mm256_sub_pd(Lh1, LA));
    __m256d Lh2  = _mm256_add_pd(Lh1, Lrh);
    __m256d Ll2  = _mm256_sub_pd(Lrh, _mm256_sub_pd(Lh2, Lh1));
    __m256d Lelo = _mm256_mul_pd(Le, L_ln2lo);
    __m256d Llo  = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(Ll1, Ll2), Lplo), Lelo);
    const SIMD_TYPE_FLOAT SUMLOG_1 = _mm256_add_pd(Lh2, Llo);
    const SIMD_TYPE_FLOAT LOG_LL_1 = _mm256_sub_pd(Llo, _mm256_sub_pd(SUMLOG_1, Lh2));

    /* 3. Stirling: W = ln Gamma(XW) as a double-double. */
    const SIMD_TYPE_FLOAT HALF1   = LOAD_FLOAT(MATERN_BROADCAST_HALF);
    const SIMD_TYPE_FLOAT LEFT_1  = SUB_FLOAT(XW_1, HALF1);
    const SIMD_TYPE_FLOAT ONE_F   = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT DIV_1   = DIV_FLOAT(ONE_F, XW_1);
    const SIMD_TYPE_FLOAT XNEW0_1 = MUL_FLOAT(DIV_1, DIV_1);
    const SIMD_TYPE_FLOAT COEF1 = LOAD_FLOAT(MATERN_STIRLING_COEFFICIENTS + 6 * INDEX_FACTOR);
    const SIMD_TYPE_FLOAT FACT1 = LOAD_FLOAT(MATERN_STIRLING_COEFFICIENTS + 5 * INDEX_FACTOR);
    SIMD_TYPE_FLOAT SUM1_1 = FMADD_FLOAT(COEF1, XNEW0_1, FACT1);
    for (int i = 4 * INDEX_FACTOR; i >= INDEX_FACTOR; i = i - INDEX_FACTOR) {
        SIMD_TYPE_FLOAT COEF2 = LOAD_FLOAT(MATERN_STIRLING_COEFFICIENTS + i);
        SUM1_1 = FMADD_FLOAT(SUM1_1, XNEW0_1, COEF2);
    }
    const SIMD_TYPE_FLOAT COEF3 = LOAD_FLOAT(MATERN_STIRLING_COEFFICIENTS);
    SUM1_1 = MUL_FLOAT(SUM1_1, DIV_1);
    SUM1_1 = ADD_FLOAT(SUM1_1, COEF3);
    const SIMD_TYPE_FLOAT SS_1 = SUM1_1;

    const SIMD_TYPE_FLOAT DD_ONE_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
    const SIMD_TYPE_FLOAT DD_RH_1  = SUB_FLOAT(SUMLOG_1, DD_ONE_1);
    const SIMD_TYPE_FLOAT DD_RL_1  = ADD_FLOAT(SUB_FLOAT(SUB_FLOAT(SUMLOG_1, DD_RH_1), DD_ONE_1), LOG_LL_1);
    const SIMD_TYPE_FLOAT DD_ADH_1 = MUL_FLOAT(LEFT_1, DD_RH_1);
    const SIMD_TYPE_FLOAT DD_ADL_1 = FMADD_FLOAT(LEFT_1, DD_RL_1, FMSUB_FLOAT(LEFT_1, DD_RH_1, DD_ADH_1));
    const SIMD_TYPE_FLOAT DD_WH_1  = ADD_FLOAT(DD_ADH_1, SS_1);
    const SIMD_TYPE_FLOAT DD_WL_1  = ADD_FLOAT(SUB_FLOAT(SS_1, SUB_FLOAT(DD_WH_1, DD_ADH_1)), DD_ADL_1);

    /* 4. Low-word correction psi(XW) XL; the 1/XW estimate only scales XL.  It is an IEEE
     *    single-precision division, not VRCPPS, whose result the x86 instruction set leaves
     *    to the implementation, so the result does not depend on the processor vendor. */
    const SIMD_TYPE_FLOAT RES3H_1 = DD_WH_1;
    const SIMD_TYPE_FLOAT INVXW   = _mm256_cvtps_pd(_mm_div_ps(_mm_set1_ps(1.0f), _mm256_cvtpd_ps(XW_1)));
    const SIMD_TYPE_FLOAT RES3L_1 = FMADD_FLOAT(SUB_FLOAT(SUMLOG_1, MUL_FLOAT(LOAD_FLOAT(MATERN_BROADCAST_HALF), INVXW)), XL_1, DD_WL_1);

    /* 5. exp(W) as a double-double (TH, TL), with the constants of exponential_avx2.h;
     *    (8k) << 49 == k << 52. */
    const __m256d E_SHIFT  = _mm256_set1_pd(0x1.8p52);
    const __m256i E_BIAS   = _mm256_set1_epi64x(0x4338000000000000LL);
    __m256d Es  = _mm256_fmadd_pd(RES3H_1, _mm256_set1_pd(0x1.71547652b82fep+3), E_SHIFT);
    __m256i Eqi = _mm256_castpd_si256(Es);
    __m256d Eqd = _mm256_sub_pd(Es, E_SHIFT);
    __m256i Eq  = _mm256_sub_epi64(Eqi, E_BIAS);
    __m256i Ej  = _mm256_and_si256(Eq, _mm256_set1_epi64x(7));
    __m256i Ek8 = _mm256_and_si256(Eq, _mm256_set1_epi64x(~7LL));
    __m256d Er  = _mm256_fnmadd_pd(Eqd, _mm256_set1_pd(0x1.62e42f8000000p-4), RES3H_1);
    Er          = _mm256_fnmadd_pd(Eqd, _mm256_set1_pd(0x1.be8e7bcd5e4f2p-30), Er);
    Er          = _mm256_add_pd(Er, RES3L_1);
    __m256d Eth, Etl;
    matern_gamma_exp_table(Ej, &Eth, &Etl);
    __m256d Ew = _mm256_set1_pd(0x1.a00d960e1cab3p-13);
    Ew = _mm256_fmadd_pd(Ew, Er, _mm256_set1_pd(0x1.6c1cbae98c662p-10));
    Ew = _mm256_fmadd_pd(Ew, Er, _mm256_set1_pd(0x1.11111125558f7p-7));
    Ew = _mm256_fmadd_pd(Ew, Er, _mm256_set1_pd(0x1.55555547c6b2ap-5));
    Ew = _mm256_fmadd_pd(Ew, Er, _mm256_set1_pd(0x1.55555555552ccp-3));
    Ew = _mm256_fmadd_pd(Ew, Er, _mm256_set1_pd(0x1.0000000000138p-1));
    Ew = _mm256_fmadd_pd(Ew, Er, _mm256_set1_pd(1.0));
    __m256d Eu = _mm256_mul_pd(Er, Ew);
    __m256d Ev = _mm256_fmadd_pd(Eth, Eu, Etl);
    __m256d Eyh = _mm256_add_pd(Eth, Ev);
    __m256d Eyl = _mm256_add_pd(_mm256_sub_pd(Eth, Eyh), Ev);
    __m256i Eksh = _mm256_slli_epi64(Ek8, 49);
    const SIMD_TYPE_FLOAT TH_1 = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(Eyh), Eksh));
    const SIMD_TYPE_FLOAT TL_1 = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(Eyl), Eksh));

    /* 6. Double-double division (TH, TL) / (PH, PL). */
    const SIMD_TYPE_FLOAT qd  = DIV_FLOAT(TH_1, PH_1);
    const SIMD_TYPE_FLOAT e1  = FMSUB_FLOAT(qd, PH_1, TH_1);
    const SIMD_TYPE_FLOAT e2  = FMADD_FLOAT(qd, PL_1, e1);
    const SIMD_TYPE_FLOAT e3  = SUB_FLOAT(e2, TL_1);
    const SIMD_TYPE_FLOAT cor = DIV_FLOAT(e3, PH_1);
    return SUB_FLOAT(qd, cor);
}


#endif /* MATERN_GAMMA_AVX2_H */
