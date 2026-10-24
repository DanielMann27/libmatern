/* continued_fraction_wronskian_device.cuh -- device continued-fraction kernel: I-series
 * pair, CF2 by modified Lentz at mu = nu - 1 and the Wronskian, prefactor cancelled, with
 * the low-word correction from the exact d log M/dz = -1/H (see the accompanying paper,
 * continued-fraction formulation).  Host counterpart: continued_fraction_wronskian.h. */
#ifndef MATERN_CONTINUED_FRACTION_WRONSKIAN_DEVICE_CUH
#define MATERN_CONTINUED_FRACTION_WRONSKIAN_DEVICE_CUH

/* max1 and max2, supplied by the caller, cap the series and the Lentz loop. */
static __device__ double matern_continued_fraction_dd_argument_device(double z, double zl, double nu, int max1, int max2, double tol)
{
    double out1 = 0.0, out2 = 0.0;
    double z2 = z / 2.0;
    double s = 1.0 / nu;
    double t = 1.0 / z2;
    double z4 = z2 * z2;

    /* loop 1: out1 = Gamma(nu) (2/z)^nu I_nu and out2 with out2/out1 = I_(nu-1)/I_nu share
     * one loop, their term ratios one index apart. */
    for (int i = 0; i < max1; i++)
    {
        out1 = out1 + s;
        out2 = out2 + t;
        if (fabs(s) < tol * fabs(out1)) break;
        s = s * 1.0 / ((nu + i + 1.0) * (i + 1.0)) * z4;
        t = t * 1.0 / ((nu + i) * (i + 1.0)) * z4;
    }

    nu = nu - 1.0;
    double low1 = 1e-50, low2 = 1e-50;
    double h = 0.0, c = 1.0;
    double nu2 = nu * nu;
    double a, b, prod;

    /* loop 2: modified Lentz with a = mu^2 - (2c-1)^2/4 and b = 2(z+c); low1, seeded with
     * the starting value 1e-50, accumulates the fraction. */
    for (int i = 0; i < max2; i++)
    {
        a = (nu2 - ((2 * c - 1) * (2 * c - 1)) * 0.25);
        b = 2.0 * (z + c);
        low2 = a / low2 + b;
        h = 1.0 / (a * h + b);
        prod = h * low2;
        low1 = low1 * prod;
        if (fabs(prod - 1.0) < tol) break;
        c = c + 1.0;
    }
    double inv = 1.0 / z;
    double H = inv * (nu + z + 0.5) + inv * low1;   /* K_(mu+1)/K_mu; the variable nu holds mu here */
    double iH = 1.0 / H;                            /* one divide serves both uses */
    double v  = 2.0 / (z * (out2 + out1 * iH));
    return fma(v * zl, -iH, v);                     /* M(z+zl) = M(z)(1 + zl*d log M/dz) */
}

#endif /* MATERN_CONTINUED_FRACTION_WRONSKIAN_DEVICE_CUH */
