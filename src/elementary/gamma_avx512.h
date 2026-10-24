/* gamma_avx512.h -- Gamma(x) for eight doubles, x in (0, 171]: double-double upward
 * recurrence into [8, 9), ln x and Stirling ln Gamma in double-double, the inlined
 * exponential, double-double division. */
#ifndef MATERN_GAMMA_AVX512_H
#define MATERN_GAMMA_AVX512_H

#include "simd_avx512.h"
#include "gamma_coefficients_avx512.h"


/* 2^(j/8) double-double; same values as MATERN_EXP_TH/TL in exponential_avx512.h. */
static const double MATERN_GAMMA_EXP_TABLE_HIGH[8] __attribute__((aligned(64))) = {
    0x1.0000000000000p+0, 0x1.172b83c7d517bp+0, 0x1.306fe0a31b715p+0, 0x1.4bfdad5362a27p+0,
    0x1.6a09e667f3bcdp+0, 0x1.8ace5422aa0dbp+0, 0x1.ae89f995ad3adp+0, 0x1.d5818dcfba487p+0
};
static const double MATERN_GAMMA_EXP_TABLE_LOW[8] __attribute__((aligned(64))) = {
    0x0.0p+0, -0x1.19041b9d78a76p-55, 0x1.6f46ad23182e4p-55, 0x1.d4397afec42e2p-56,
    -0x1.bdd3413b26456p-54, 0x1.6e9f156864b27p-54, 0x1.7a1cd345dcc81p-54, 0x1.2ed02d75b3707p-55
};

/* ln and exp are inlined because the chain needs ln XW and the exponent W as
 * double-doubles; every quantity feeding the exponent is a double-double, and the low
 * word XL of XW enters W as psi(XW) XL.  The recurrence exit is horizontal, so the
 * order of inputs affects speed, not results. */
static inline __attribute__((always_inline)) __m512d matern_gamma_pd(__m512d X)
{
    /* 1. Double-double upward recurrence into [8, 9): product (PH, PL), shifted
     *    argument (XW, XL). */
    SIMD_TYPE_FLOAT XW_1 = X;
    SIMD_TYPE_FLOAT XL_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL);                 /* 0.0 */
    SIMD_TYPE_FLOAT PH_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);  /* 1.0 */
    SIMD_TYPE_FLOAT PL_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL);                 /* 0.0 */
    {
        const SIMD_TYPE_FLOAT EIGHT_1 = LOAD_FLOAT(MATERN_BROADCAST_POOL + 8 * INDEX_FACTOR);
        const SIMD_TYPE_FLOAT ONE_1   = LOAD_FLOAT(MATERN_BROADCAST_POOL + INDEX_FACTOR);
        for (int rs = 0; rs < 8; rs++) {
            const MASK_FLOAT       mm  = LESS_THAN_FLOAT(XW_1, EIGHT_1);
            if (!mm) break;
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

    /* 2. ln XW = (SUMLOG_1, LOG_LL_1): 16-entry accurate table, degree-11 polynomial,
     *    relaxed recomposition.  The tables equal MATERN_GAMMA_C/LOGC of
     *    gamma_coefficients_avx2.h. */
    const __m512d L_thresh = _mm512_set1_pd(0x1.6555555555556p+0);
    const __m512d L_one    = _mm512_set1_pd(1.0);
    const __m512d L_half   = _mm512_set1_pd(0.5);
    const __m512d L_ln2hi  = _mm512_set1_pd(0x1.62e42fee00000p-1);
    const __m512d L_ln2lo  = _mm512_set1_pd(0x1.a39ef35793c76p-33);
    const __m512i L_base   = _mm512_set1_epi64(0x3FE6555555555556LL);
    const __m512d L_C_lo  = _mm512_setr_pd(0x1.66dc1a061ead5p+0, 0x1.57cbc2f6dca63p+0, 0x1.49f23241a584ap+0, 0x1.3d2b498090bd4p+0, 0x1.31584df1225d7p+0, 0x1.265ef05b03089p+0, 0x1.1c2888a4b8678p+0, 0x1.12a178f6b6c71p+0);
    const __m512d L_C_hi  = _mm512_setr_pd(0x1.09b8af69ba7e6p+0, 0x1.0000000000000p+0, 0x1.e6de48067ffc7p-1, 0x1.cb8b8e97808d3p-1, 0x1.b3207789f95b3p-1, 0x1.9d2c5f953e58bp-1, 0x1.89544cafda03ap-1, 0x1.774df5e0e4329p-1);
    const __m512d L_LC_lo = _mm512_setr_pd(-0x1.59dc40bf7fad6p-2, -0x1.2df29fc31fea0p-2, -0x1.03d78315b742ep-2, -0x1.b6ccaf9e90967p-3, -0x1.68fd6fb941a16p-3, -0x1.1e079a7156ba3p-3, -0x1.ab6f43721f401p-4, -0x1.1fbe7f18bfbddp-4);
    const __m512d L_LC_hi = _mm512_setr_pd(-0x1.31530d08c0289p-5, 0x0.0p+0, 0x1.9c4fc07dbbc71p-5, 0x1.bab9e5834a389p-4, 0x1.4d2ee330389bbp-3, 0x1.b735748880f84p-3, 0x1.0e0176fede440p-2, 0x1.3e0a7a09b7669p-2);

    __m512d Lm = _mm512_getmant_pd(XW_1, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    __m512d Le = _mm512_getexp_pd(XW_1);
    const __mmask8 Lhi = _mm512_cmp_pd_mask(Lm, L_thresh, _CMP_GE_OQ);
    Lm = _mm512_mask_mul_pd(Lm, Lhi, Lm, L_half);
    Le = _mm512_mask_add_pd(Le, Lhi, Le, L_one);
    __m512i Lmb   = _mm512_castpd_si512(Lm);
    __m512i Lidx  = _mm512_srli_epi64(_mm512_sub_epi64(Lmb, L_base), 48);
    __m512d Lc    = _mm512_permutex2var_pd(L_C_lo,  Lidx, L_C_hi);
    __m512d Llogc = _mm512_permutex2var_pd(L_LC_lo, Lidx, L_LC_hi);
    __m512d Lph_ = _mm512_mul_pd(Lm, Lc);
    __m512d Lrh  = _mm512_sub_pd(Lph_, L_one);            /* P(r) hi: exact (Sterbenz) */
    __m512d Lpl  = _mm512_fmsub_pd(Lm, Lc, Lph_);
    __m512d Lr   = _mm512_add_pd(Lrh, Lpl);
    __m512d Lr2 = _mm512_mul_pd(Lr, Lr);
    __m512d LB = _mm512_set1_pd( 0x1.7514b8968cd04p-4);
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd(-0x1.9a6087a41cb9bp-4));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd( 0x1.c71c4c5a59011p-4));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd(-0x1.ffffd77508470p-4));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd( 0x1.2492492620b04p-3));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd(-0x1.5555555704ed7p-3));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd( 0x1.99999999998e0p-3));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd(-0x1.fffffffffff37p-3));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd( 0x1.5555555555555p-2));
    LB = _mm512_fmadd_pd(LB, Lr, _mm512_set1_pd(-0x1.0000000000000p-1));
    __m512d Lqv  = _mm512_mul_pd(Lr2, LB);               /* r^2*B (P low, nonlinear) */
    /* log = e*ln2 + logc + P(r), P = Lrh + Lplo.  The FastTwoSum chain A (+logc) (+Lrh)
     * is exact since the magnitudes decrease (|A| >= 2.08, |logc| <= 0.59,
     * |Lrh| <= 0.03); the roundoffs, Lplo and e*ln2lo are summed into one low word. */
    __m512d Lplo = _mm512_add_pd(Lpl, Lqv);                            /* P(r) low */
    __m512d LA   = _mm512_mul_pd(Le, L_ln2hi);                         /* e*ln2hi exact (small e) */
    __m512d Lh1  = _mm512_add_pd(LA, Llogc);
    __m512d Ll1  = _mm512_sub_pd(Llogc, _mm512_sub_pd(Lh1, LA));       /* FastTwoSum lo (exact) */
    __m512d Lh2  = _mm512_add_pd(Lh1, Lrh);
    __m512d Ll2  = _mm512_sub_pd(Lrh, _mm512_sub_pd(Lh2, Lh1));        /* FastTwoSum lo (exact) */
    __m512d Lelo = _mm512_mul_pd(Le, L_ln2lo);
    __m512d Llo  = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(Ll1, Ll2), Lplo), Lelo);
    const SIMD_TYPE_FLOAT SUMLOG_1 = _mm512_add_pd(Lh2, Llo);
    const SIMD_TYPE_FLOAT LOG_LL_1 = _mm512_sub_pd(Llo, _mm512_sub_pd(SUMLOG_1, Lh2));

    /* 3. Stirling: W = ln Gamma(XW) = (XW - 0.5)(ln XW - 1) + S(1/XW) as a double-double
     *    (DD_WH, DD_WL); S is the coefficient pool of gamma_coefficients_avx512.h. */
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

    /* 4. Low-word correction psi(XW) XL, psi(XW) ~ ln XW - 0.5/XW; the 14-bit rcp14 only
     *    scales XL. */
    const SIMD_TYPE_FLOAT RES3H_1 = DD_WH_1;
    const SIMD_TYPE_FLOAT RES3L_1 = FMADD_FLOAT(SUB_FLOAT(SUMLOG_1, MUL_FLOAT(LOAD_FLOAT(MATERN_BROADCAST_HALF), _mm512_rcp14_pd(XW_1))), XL_1, DD_WL_1);

    /* 5. exp(W) as a double-double (TH, TL), with the reduction constants, 2^(j/8) table
     *    and polynomial of exponential_avx512.h. */
    const __m512d E_SHIFT  = _mm512_set1_pd(0x1.8p52);
    const __m512i E_BIAS   = _mm512_set1_epi64(0x4338000000000000LL);
    const __m512d E_8byln2 = _mm512_set1_pd(0x1.71547652b82fep+3);
    const __m512d E_ln2_8h = _mm512_set1_pd(0x1.62e42f8000000p-4);
    const __m512d E_ln2_8l = _mm512_set1_pd(0x1.be8e7bcd5e4f2p-30);
    __m512d Es  = _mm512_fmadd_pd(RES3H_1, E_8byln2, E_SHIFT);
    __m512i Eqi = _mm512_castpd_si512(Es);
    __m512d Eqd = _mm512_sub_pd(Es, E_SHIFT);
    __m512i Ej  = _mm512_and_si512(Eqi, _mm512_set1_epi64(7));
    __m512i Ek  = _mm512_srai_epi64(_mm512_sub_epi64(Eqi, E_BIAS), 3);
    __m512d Er  = _mm512_fnmadd_pd(Eqd, E_ln2_8h, RES3H_1);
    Er          = _mm512_fnmadd_pd(Eqd, E_ln2_8l, Er);
    Er          = _mm512_add_pd(Er, RES3L_1);
    __m512d Eth = _mm512_permutexvar_pd(Ej, _mm512_load_pd(MATERN_GAMMA_EXP_TABLE_HIGH));
    __m512d Etl = _mm512_permutexvar_pd(Ej, _mm512_load_pd(MATERN_GAMMA_EXP_TABLE_LOW));
    __m512d Ew = _mm512_set1_pd(0x1.a00d960e1cab3p-13);
    Ew = _mm512_fmadd_pd(Ew, Er, _mm512_set1_pd(0x1.6c1cbae98c662p-10));
    Ew = _mm512_fmadd_pd(Ew, Er, _mm512_set1_pd(0x1.11111125558f7p-7));
    Ew = _mm512_fmadd_pd(Ew, Er, _mm512_set1_pd(0x1.55555547c6b2ap-5));
    Ew = _mm512_fmadd_pd(Ew, Er, _mm512_set1_pd(0x1.55555555552ccp-3));
    Ew = _mm512_fmadd_pd(Ew, Er, _mm512_set1_pd(0x1.0000000000138p-1));
    Ew = _mm512_fmadd_pd(Ew, Er, _mm512_set1_pd(1.0));
    __m512d Eu = _mm512_mul_pd(Er, Ew);
    __m512d Ev  = _mm512_fmadd_pd(Eth, Eu, Etl);
    __m512d Eyh = _mm512_add_pd(Eth, Ev);
    __m512d Eyl = _mm512_add_pd(_mm512_sub_pd(Eth, Eyh), Ev);
    __m512i Eksh = _mm512_slli_epi64(Ek, 52);
    const SIMD_TYPE_FLOAT TH_1 = _mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(Eyh), Eksh));
    const SIMD_TYPE_FLOAT TL_1 = _mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(Eyl), Eksh));

    /* 6. (TH, TL) / (PH, PL): q = TH/PH, the residual q PH - TH exact by the fma, the low
     *    words added, and a second divide for the correction. */
    const SIMD_TYPE_FLOAT q   = DIV_FLOAT(TH_1, PH_1);
    const SIMD_TYPE_FLOAT e1  = FMSUB_FLOAT(q, PH_1, TH_1);
    const SIMD_TYPE_FLOAT e2  = FMADD_FLOAT(q, PL_1, e1);
    const SIMD_TYPE_FLOAT e3  = SUB_FLOAT(e2, TL_1);
    const SIMD_TYPE_FLOAT cor = DIV_FLOAT(e3, PH_1);
    return SUB_FLOAT(q, cor);
}

#endif /* MATERN_GAMMA_AVX512_H */
