/* derive_argument_device.cuh -- device form of derive_argument.h: z = sqrt(2 nu) h / s as a
 * double-double Zh + Zl, with h passed in zc.  The (z, nu) entry (matern_covariance_cuda_z)
 * bypasses it and passes z with Zl = 0. */
#ifndef MATERN_DERIVE_ARGUMENT_DEVICE_CUH
#define MATERN_DERIVE_ARGUMENT_DEVICE_CUH

static __device__ inline void matern_derive_argument_device(double nuc, double zc, double scalec,
                                      double *Zh, double *Zl)
{
    double a   = 2.0 * nuc;                        /* exact */
    double sq  = sqrt(a);
    double sql = fma(-sq, sq, a) / (2.0 * sq);     /* sqrt(a) = sq + sql */
    double th  = sq * zc;
    double tl  = fma(sq, zc, -th) + sql * zc;      /* sq*zc = th + tl */
    double zh  = th / scalec;
    *Zh = zh;
    *Zl = (fma(-zh, scalec, th) + tl) / scalec;
}

#endif /* MATERN_DERIVE_ARGUMENT_DEVICE_CUH */
