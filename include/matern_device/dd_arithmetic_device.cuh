/* dd_arithmetic_device.cuh -- device double-double primitives: TwoSum, and log x = lh + ll for
 * positive normal x with the method, coefficients and operation order of dd_log in
 * double_double_avx512.h, accurate to about 2^-58 absolute, not to double-double precision.
 * The __constant__ tables are statically initialised (no upload). */
#ifndef MATERN_DD_ARITHMETIC_DEVICE_CUH
#define MATERN_DD_ARITHMETIC_DEVICE_CUH

static __device__ __constant__ double MATERN_DDLOG_C[16] = {
  0x1.66dc1a064aef0p+0, 0x1.57cbc2f6cb044p+0, 0x1.49f23241c41ecp+0, 0x1.3d2b498096841p+0,
  0x1.31584df11803ep+0, 0x1.265ef05ad41adp+0, 0x1.1c2888a4bd60ep+0, 0x1.12a178f6bb705p+0,
  0x1.09b8af69ba7e6p+0, 0x1.0000000000000p+0, 0x1.e6de48066a58bp-1, 0x1.cb8b8e97829fap-1,
  0x1.b3207789e8ce7p-1, 0x1.9d2c5f955aceap-1, 0x1.89544cafebaa1p-1, 0x1.774df5e0f8dcdp-1 };
static __device__ __constant__ double MATERN_DDLOG_LC[16] = {
 -0x1.59dc40bffdf67p-2,-0x1.2df29fc2eb653p-2,-0x1.03d78316163c6p-2,-0x1.b6ccaf9eb5e35p-3,
 -0x1.68fd6fb8fc365p-3,-0x1.1e079a70103c0p-3,-0x1.ab6f437266f32p-4,-0x1.1fbe7f190542ap-4,
 -0x1.31530d08c0289p-5, 0x0.0p+0,            0x1.9c4fc07f27e26p-5, 0x1.bab9e58337c0bp-4,
  0x1.4d2ee3308680bp-3, 0x1.b7357487f3e5ap-3, 0x1.0e0176feb050ap-2, 0x1.3e0a7a097f046p-2 };

/* (s,e) = TwoSum(a,b): s = fl(a+b), a + b = s + e exactly.  The six-operation form, valid
 * for either ordering of |a| and |b|; the operations must not be reassociated. */
static __device__ inline void matern_two_sum_device(double a, double b, double *s, double *e)
{
    double x = a + b;
    double bb = x - a;
    *e = (a - (x - bb)) + (b - bb);
    *s = x;
}

/* log(x) as lh + ll, for positive normal x.  Not a full double-double logarithm:
 * log(1+r) is a double polynomial and r is rounded, so lh + ll is accurate to about 2^-58
 * absolute; near x = 1, where log x is small, that is about 2^-52.6 relative, no better
 * than one double. */
static __device__ inline void matern_dd_log_device(double x, double *lh, double *ll)
{
    const double thresh = 0x1.6555555555556p+0;
    const double ln2_hi = 0x1.62e42fee00000p-1;   /* 32 significant bits -> e*ln2_hi exact */
    const double ln2_lo = 0x1.a39ef35793c76p-33;
    /* x = 2^e m with m in [1,2), recentred to [thresh/2, thresh); halving m is exact. */
    int k; double f = frexp(x, &k);
    double m = 2.0 * f; double e = (double)(k - 1);
    if (m >= thresh){ m *= 0.5; e += 1.0; }
    /* index from the leading mantissa bits, biased so that bucket 0 starts at thresh/2;
     * entry 9 is c = 1, log c = 0 */
    long long b = __double_as_longlong(m) - 0x3FE6555555555556LL;
    int idx = (int)(b >> 48); idx = idx < 0 ? 0 : (idx > 15 ? 15 : idx);
    double c = MATERN_DDLOG_C[idx], logc = MATERN_DDLOG_LC[idx];
    double r  = fma(m, c, -1.0);
    double r2 = r * r;
    double B =            0x1.c7ca050ba3963p-4;
    B = fma(B, r, -0x1.005f77015466cp-3);
    B = fma(B, r,  0x1.24923c827ddc1p-3);
    B = fma(B, r, -0x1.5555476feb961p-3);
    B = fma(B, r,  0x1.9999999a30370p-3);
    B = fma(B, r, -0x1.0000000052d61p-2);
    B = fma(B, r,  0x1.555555555554dp-2);
    B = fma(B, r, -0x1.ffffffffffff6p-2);
    double P = fma(r2, B, r);                     /* ~log(1+r) */
    /* recomposition log x = e ln2 + log c + log(1+r), smallest terms first so that
     * each TwoSum captures the rounding the following addition would discard */
    double s, es;  matern_two_sum_device(logc, P, &s, &es);
    double lo = fma(e, ln2_lo, es);
    double eh2 = e * ln2_hi;
    double hh, ee2; matern_two_sum_device(eh2, s, &hh, &ee2);
    *lh = hh; *ll = ee2 + lo;
}

#endif /* MATERN_DD_ARITHMETIC_DEVICE_CUH */
