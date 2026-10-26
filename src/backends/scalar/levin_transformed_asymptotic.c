/* levin_transformed_asymptotic.c -- scalar Levin transform of the large-argument series by
 * the difference-table recursion; compiled, but routed only under -DMATERN_SCALAR_WITH_LEVIN
 * (see the accompanying paper, numerical evaluation of the modified Bessel function). */
#include <math.h>

#include "scalar_internal.h"

/* Hidden under -std=c11; the same tokens as glibc's M_PI, so a redefinition is benign. */
#define M_PI    3.14159265358979323846

/* K ~ (SUM_i c_i A_i / omega_i) / (SUM_i c_i / omega_i), A_i the partial sums and omega_i
 * the next term; levin_rescale is the per-step factor of the difference-table recursion. */
static double levin_rescale(double x, double y, double zz)
{
    return -(x + y + zz) * (x + y + zz - 1) / ((x + y + 2 * zz) * (x + y + 2 * zz - 1));
}

double matern_scalar_levin(double z, double nu)
{
    double sequence[SCALAR_LEVIN_TERMS];
    double weights[SCALAR_LEVIN_TERMS];
    double s = 0.0, t = 1.0;
    const double nu2 = nu * nu;

    /* s: partial sums; t: current term, ratio (4nu^2 - (2i-1)^2)/(8 i z).  The next term is
     * stored as the remainder estimate of the partial sum it follows. */
    for (int i = 1; i < SCALAR_LEVIN_TERMS + 1; i++){
        s = s + t;
        t = t * (4 * nu2 - (2 * i - 1) * (2 * i - 1)) / (8 * z * i);
        sequence[i - 1] = s;
        weights[i - 1] = t;
    }

    double knu = 0.0;
    int terminated = 0;
    for (int i = 0; i < SCALAR_LEVIN_TERMS; i++){
        /* At half-integer nu, 4nu^2 - (2i-1)^2 = 0 terminates the series: the partial sum in
         * hand is exact, and taking it avoids dividing by a zero remainder estimate. */
        if (weights[i] == 0.0){
            knu = sequence[i];
            terminated = 1;
            break;
        }
        sequence[i] = sequence[i] / weights[i];   /* A_i / omega_i */
        weights[i]  = 1.0 / weights[i];           /*   1 / omega_i */
    }
    if (!terminated){
        /* Each pass shortens both rows by one; entry 0 then holds numerator and denominator. */
        for (int i = 0; i < SCALAR_LEVIN_TERMS; i++)
            for (int j = 0; j < SCALAR_LEVIN_TERMS - 2 - i; j++){
                sequence[j] = sequence[j] * levin_rescale(1.0, j + 1, i) + sequence[j + 1];
                weights[j]  = weights[j]  * levin_rescale(1.0, j + 1, i) + weights[j + 1];
            }
        knu = sequence[0] / weights[0];
    }
    knu *= sqrt(M_PI / (2.0 * z)) * exp(-z);      /* K_nu(z) */
    return pow(2.0, 1.0 - nu) / tgamma(nu) * pow(z, nu) * knu;
}
