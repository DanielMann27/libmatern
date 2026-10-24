/* boost_shortcut_host.cpp -- the cpu row of stage `gpu-boost-shortcut`.  Compiled by g++,
 * not nvcc, whose `__CUDACC__` disables Boost's promote_double.  It evaluates as the CPU
 * supplier row does (references/cpp_stdlib/cpu_cxx.cpp): long double, errors returned, not
 * thrown; it is built against the device rows' Boost. */
#include <cmath>
#include <cstdio>
#include <limits>
#include <boost/version.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/policies/policy.hpp>
#include "../references/common/build_info.h"

namespace bmp = boost::math::policies;
typedef bmp::policy<
    bmp::domain_error    <bmp::ignore_error>,
    bmp::overflow_error  <bmp::ignore_error>,
    bmp::underflow_error <bmp::ignore_error>,
    bmp::evaluation_error<bmp::ignore_error>,
    bmp::pole_error      <bmp::ignore_error>
> host_policy;

extern "C" void wm_boost_shortcut_host(const double *nu, const double *z,
                                       double *out, int n)
{
    for (int i = 0; i < n; i++) {
        try { out[i] = boost::math::cyl_bessel_k(nu[i], z[i], host_policy()); }
        catch (...) { out[i] = std::numeric_limits<double>::quiet_NaN(); }
    }
}

/* 1 when this object evaluates in a type wider than double (the `precision` column) */
extern "C" int wm_boost_shortcut_host_promoted(void)
{
    typedef bmp::evaluation<double, bmp::policy<> >::type eval_t;
    return sizeof(eval_t) > sizeof(double);
}

extern "C" int wm_boost_shortcut_host_version(void) { return BOOST_VERSION; }

/* This object's build record (build_gpu_boost_shortcut_cpu_row_*), Boost version and
 * evaluation type included. */
extern "C" void wm_boost_shortcut_host_build_info(FILE *f)
{
    const char *unit = "gpu_boost_shortcut_cpu_row";
    char v[32];
    WM_BUILD_INFO_COMMON(f, unit);
    snprintf(v, sizeof v, "%d.%d.%d", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000,
             BOOST_VERSION % 100);
    wm_build_info_kv(f, unit, "boost", v);
    wm_build_info_kv(f, unit, "boost_eval_type",
                     wm_boost_shortcut_host_promoted() ? "long double" : "double");
}
