/* reciprocal_gamma_coefficients.h -- even and odd Taylor halves of 1/Gamma(1+x), 11 terms
 * each (the length the scalar and vector small-argument kernels evaluate), giving gam1 and
 * gam2 as series in mu^2 free of cancellation; the consumers negate the odd sum, so
 * gam1(0) = -EulerGamma and gam2(0) = 1. */
#ifndef MATERN_RECIPROCAL_GAMMA_COEFFICIENTS_H
#define MATERN_RECIPROCAL_GAMMA_COEFFICIENTS_H

/* temme_small_argument_device.cuh holds a copy (MATERN_TMWS_G1S, MATERN_TMWS_G2S) that must
 * stay equal. */
#define RECIPROCAL_GAMMA_EVEN_TERMS 11
static const double RECIPROCAL_GAMMA_EVEN_C[RECIPROCAL_GAMMA_EVEN_TERMS] = {
    1.00000000000000000e+00,
    -6.55878071520253902e-01,
    1.66538611382291479e-01,
    -9.62197152787697303e-03,
    -1.16516759185906517e-03,
    1.28050282388116196e-04,
    -1.25049348214267063e-06,
    -2.05633841697760707e-07,
    5.00200764446922295e-09,
    1.04342671169110054e-10,
    -3.69680561864220598e-12
};

#define RECIPROCAL_GAMMA_ODD_TERMS 11
static const double RECIPROCAL_GAMMA_ODD_B[RECIPROCAL_GAMMA_ODD_TERMS] = {
    5.77215664901532866e-01,
    -4.20026350340952370e-02,
    -4.21977345555443334e-02,
    7.21894324666309990e-03,
    -2.15241674114950975e-04,
    -2.01348547807882387e-05,
    1.13302723198169593e-06,
    6.11609510448141609e-09,
    -1.18127457048702004e-09,
    7.78226343990507081e-12,
    5.10037028745447575e-13
};

#endif /* MATERN_RECIPROCAL_GAMMA_COEFFICIENTS_H */
