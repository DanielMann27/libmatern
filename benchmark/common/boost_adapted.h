/* boost_adapted.h -- the `boost-...-adapted` K_nu rule, shared by suppliers_gpu.cu and
 * boost_shortcut_gpu.cu so that the stage and the supplier row run the same code. */
#ifndef BENCH_BOOST_ADAPTED_H
#define BENCH_BOOST_ADAPTED_H

#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

#ifdef __CUDACC__
#  define BENCH_BOOST_HOST_DEVICE __host__ __device__
#else
#  define BENCH_BOOST_HOST_DEVICE
#endif

/* Boost >= 1.86 replaces K_v by a 4-term Hankel series when |4v^2 - 25|/(8 x) < eps^(1/4),
 * exact only at v = 5/2.  Wherever that test could fire (with a margin of 2, sized on
 * double's eps, which also covers a long double build), K_v comes from the stable upward
 * recurrence, whose own calls reach the test only where K underflows.  v == 2.5 is
 * excluded (the short-cut is exact there); elsewhere the result is the plain call's. */
BENCH_BOOST_HOST_DEVICE inline double bench_boost_kv_adapted(double v, double x)
{
    const double fre = 1.220703125e-4;               /* 2^-13 = eps^(1/4), double */
    if (v > 2.0 && v != 2.5 && fabs(4.0 * v * v - 25.0) < 2.0 * 8.0 * x * fre)
        return boost::math::cyl_bessel_k(v - 2.0, x)
             + 2.0 * (v - 1.0) / x * boost::math::cyl_bessel_k(v - 1.0, x);
    return boost::math::cyl_bessel_k(v, x);
}

#endif /* BENCH_BOOST_ADAPTED_H */
