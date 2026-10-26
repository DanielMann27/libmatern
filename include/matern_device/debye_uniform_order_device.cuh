/* debye_uniform_order_device.cuh -- device uniform-order (Debye) kernel in plain double
 * arithmetic, the Matern prefactor folded into a double-double exponent (see the
 * accompanying paper, large-order formulation).  Host counterpart: debye_uniform_order.h. */
#ifndef MATERN_DEBYE_UNIFORM_ORDER_DEVICE_CUH
#define MATERN_DEBYE_UNIFORM_ORDER_DEVICE_CUH

#include "dd_arithmetic_device.cuh"   /* two_sum, dd_log */

/* Highest u_k index (eleven terms); equal to DEBYE_TOP_INDEX in
 * debye_uniform_order.h. */
#define MATERN_DEBYE_TOP_INDEX_DEVICE 10

static __device__ double matern_debye_polynomial_device(double nu, double p)
{
    double p1 = p * p;         /* = 1/t^2; nu*nu/(nu*nu+z*z) overflows at large nu */
    double p2 = p1 * p1, p3 = p2 * p1, p4 = p3 * p1, p5 = p4 * p1;
    double p6 = p5 * p1, p7 = p6 * p1, p8 = p7 * p1, p9 = p8 * p1, p10 = p9 * p1;

    /* Same values as the host Debye tables (debye_coefficients_scalar.h and the lane-padded
     * copies), signs written out; keep all five copies equal. */
    double poly[11];
    poly[0] = 1.0;
    poly[1] = 0.125 - 0.20833333333333334 * p1;
    poly[2] = 0.0703125 - 0.4010416666666667 * p1 + 0.3342013888888889 * p2;
    poly[3] = 0.0732421875 - 0.8912109375 * p1 + 1.8464626736111112 * p2 - 1.0258125964506173 * p3;
    poly[4] = 0.112152099609375 - 2.3640869140625 * p1 + 8.78912353515625 * p2 - 11.2070026162229946 * p3 + 4.669584423426247 * p4;
    poly[5] = 0.22710800170898438 - 7.368794359479632 * p1 + 42.53499874538846 * p2 - 91.81824154324002 * p3 + 84.63621767460073 * p4 - 28.212072558200244 * p5;
    poly[6] = 0.5725014209747314 - 26.491430486951554 * p1 + 218.1905117442116 * p2 - 699.5796273761325 * p3 + 1059.9904525279999 * p4 - 765.2524681411817 * p5 + 212.57013003921713 * p6;
    poly[7] = 1.7277275025844574 - 108.09091978839466 * p1 + 1200.9029132163525 * p2 - 5305.646978613403 * p3 + 11655.393336864534 * p4 - 13586.550006434136 * p5 + 8061.722181737309 * p6 - 1919.457662318407 * p7;
    poly[8] = 6.074042001273483 - 493.91530477308805 * p1 + 7109.514302489364 * p2 - 41192.65496889755 * p3 + 122200.46498301747 * p4 - 203400.17728041555 * p5 + 192547.00123253153 * p6 - 96980.59838863752 * p7 + 20204.29133096615 * p8;
    poly[9] = 24.380529699556064 - 2499.8304818112097 * p1 + 45218.76898136273 * p2 - 331645.17248456355 * p3 + 1.2683652733216248e6 * p4 - 2.8135632265865337e6 * p5 + 3.763271297656404e6 * p6 - 2.998015918538107e6 * p7 + 1.3117636146629772e6 * p8 - 242919.18790055133 * p9;
    poly[10] = 110.01714026924674 - 13886.08975371704 * p1 + 308186.40461266245 * p2 - 2.7856181280864547e6 * p3 + 1.3288767166421818e7 * p4 - 3.7567176660763346e7 * p5 + 6.634451227472903e7 * p6 - 7.410514821153265e7 * p7 + 5.095260249266464e7 * p8 - 1.9706819118432228e7 * p9 + 3.284469853072038e6 * p10;

    double in = -p / nu;
    double in2 = in * in;
    /* Even/odd Horner in in2 = (p/nu)^2, split by the parity of the index so the
     * even chain ends on poly[0] and the odd chain on poly[1] for any term count. */
    const int te = (MATERN_DEBYE_TOP_INDEX_DEVICE & 1) ? MATERN_DEBYE_TOP_INDEX_DEVICE - 1 : MATERN_DEBYE_TOP_INDEX_DEVICE;
    const int to = (MATERN_DEBYE_TOP_INDEX_DEVICE & 1) ? MATERN_DEBYE_TOP_INDEX_DEVICE     : MATERN_DEBYE_TOP_INDEX_DEVICE - 1;
    double out1 = poly[te];
    for (int i = te - 2; i >= 0; i -= 2) out1 = in2 * out1 + poly[i];
    double out2 = poly[to];
    for (int i = to - 2; i >= 1; i -= 2) out2 = in2 * out2 + poly[i];
    out2 = out2 * in;
    return out1 + out2;
}

/* M(h) with the Matern prefactor folded into the double-double exponent -nu*eta;
 * zl is the argument's low word.  With zl = 0.0 the body equals the single-double form:
 * z - xh*nu is never -0.0, so adding 0.0 is the identity, and zl enters before the divide.
 *   G = log1p(d/2) - d, d = t-1: (1+t)/2 = 1 + d/2 exactly, so ln 2 is never formed and
 *       its cancellation against -t is avoided; d = th-1 is exact by Sterbenz for t in [1,2].
 *   The ln nu and ln 2pi terms of Stirling's Gamma(nu) cancel against sqrt(2pi/(nu t)),
 *       so sqrt(1/th) is the only factor outside the exponent.
 *   G is no more accurate than matern_dd_log_device (about 2^-58 absolute), so the error
 *       of M grows about linearly with nu. */
static __device__ double matern_debye_fold_dd_argument_device(double zh, double zl, double nu)
{
    double z = zh;
    /* x = z/nu (dd) */
    double xh = z / nu;
    double xl = (z - xh * nu + zl) / nu;       /* (fnma(xh,nu,z) + zl)/nu */
    /* x^2 (dd) */
    double x2h = xh * xh;
    double x2l = fma(xh + xh, xl, fma(xh, xh, -x2h));
    /* 1 + x^2 (dd) */
    double uh, ue; matern_two_sum_device(1.0, x2h, &uh, &ue);
    double ul = ue + x2l;
    /* t = sqrt(1+x^2) (dd) */
    double th = sqrt(uh);
    double tl = (fma(-th, th, uh) + ul) / (th + th);
    /* 1 + t (dd) */
    double ph, pe; matern_two_sum_device(1.0, th, &ph, &pe);
    double pl = pe + tl;
    /* G = log1p(d/2) - d, d = t-1 (see above); the tail pl/ph is scale-invariant */
    double l1h, l1l; matern_dd_log_device(0.5 * ph, &l1h, &l1l);
    l1l = l1l + pl / ph;
    double dh = th - 1.0;
    double Gh, ea; matern_two_sum_device(l1h, -dh, &Gh, &ea);
    double Gl = (ea - tl) + l1l;
    /* nu*G (dd, two_prod) */
    double nGh = nu * Gh;
    double nGl = fma(nu, Gl, fma(nu, Gh, -nGh));
    /* - S(nu), the Stirling correction of lnGamma(nu), in single double */
    double rn = 1.0 / nu, rn2 = rn * rn;
    double cc = fma(rn2, 1.0 / 1680.0, -1.0 / 1260.0);
    cc = fma(rn2, cc, 1.0 / 360.0);
    cc = fma(rn2, cc, -1.0 / 12.0);
    cc = rn * cc;
    double Eh, e3; matern_two_sum_device(nGh, cc, &Eh, &e3);
    double El = e3 + nGl;
    /* Renormalise before the first-order exp (Fast2Sum, as the CPU vector Debye): El carries
     * nu times dd_log's unnormalised low word.  Exact while |El| <= |Eh|; beyond, Eh is still
     * the rounded sum, so the first-order factor stays small. */
    { const double Es = Eh + El; El = El - (Es - Eh); Eh = Es; }
    /* exp(Eh + El) = exp(Eh)*(1+El) */
    double eE = exp(Eh);
    eE = fma(eE, El, eE);
    /* POLY = the u_k sum, in single double */
    double POLY = matern_debye_polynomial_device(nu, 1.0 / th);      /* p = 1/t */
    double rest = sqrt(1.0 / th);                  /* the one factor outside the exponent */
    return rest * eE * POLY;
}

#endif /* MATERN_DEBYE_UNIFORM_ORDER_DEVICE_CUH */
