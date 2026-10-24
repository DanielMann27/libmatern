/* cpu_cxx.cpp -- std::cyl_bessel_k (C++17 <cmath>; in libstdc++ the TR1 implementation,
 * tr1/modified_bessel_func.tcc) and boost::math::cyl_bessel_k, Boost.Math's routine
 * (BSL-1.0), as K_nu suppliers; neither has a scaled or log form. */

#include <cmath>
#include <cstdio>
#include <limits>
#include "cpu_wm.h"
#include "build_info.h"

/* WMC_BOOST_EXPECT is the BOOST_VERSION of the directory benchmark/tools/boost_detect.sh
 * selected.  The row label carries the version, so a different Boost that comes first on
 * the include path (CPATH, a module, another -I) stops the build. */
#ifdef WMC_WITH_BOOST
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/policies/policy.hpp>
#if defined(WMC_BOOST_EXPECT) && BOOST_VERSION != WMC_BOOST_EXPECT
#error "Boost on the include path is not the Boost selected for the CPU row; check CPATH and CPLUS_INCLUDE_PATH, or set BOOST_CPU_ROOT"
#endif

/* Boost error policy: return infinity/NaN instead of throwing.  The std entry
 * is wrapped in a catch that returns NaN. */
namespace bmp = boost::math::policies;
typedef bmp::policy<
    bmp::domain_error    <bmp::ignore_error>,
    bmp::overflow_error  <bmp::ignore_error>,
    bmp::underflow_error <bmp::ignore_error>,
    bmp::evaluation_error<bmp::ignore_error>,
    bmp::pole_error      <bmp::ignore_error>
> wmc_boost_policy;
#endif

extern "C" {

static double cxx_std_kv(double nu, double z)
{
    try {
        return std::cyl_bessel_k(nu, z);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

const wmc_supplier wmc_cxx_std = {
    "libstdc++ C++17 <cmath> (TR1 modified_bessel_func.tcc)",
    WMC_LOG_NONE,
    cxx_std_kv, NULL, NULL
};

#ifdef WMC_WITH_BOOST
static double cxx_boost_kv(double nu, double z)
{
    try {
        return boost::math::cyl_bessel_k(nu, z, wmc_boost_policy());
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

const wmc_supplier wmc_cxx_boost = {
    "boost.math special_functions/bessel.hpp",
    WMC_LOG_NONE,
    cxx_boost_kv, NULL, NULL
};

/* boost version, for the run header */
const char *cpu_cxx_boost_version(void)
{
    static char buf[64];
    std::snprintf(buf, sizeof buf, "%d.%d.%d",
                  BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000, BOOST_VERSION % 100);
    return buf;
}
#endif /* WMC_WITH_BOOST */

/* Build record of this object (build_info.h), with the Boost version and the type in
 * which Boost evaluates a double argument under the row's policy: long double with g++
 * (default promote_double policy), double under nvcc. */
void wm_build_info_cpu_supplier_cxx(FILE *f)
{
    WM_BUILD_INFO_COMMON(f, "cpu_supplier_cxx");
#ifdef WMC_WITH_BOOST
    wm_build_info_kv(f, "cpu_supplier_cxx", "boost", cpu_cxx_boost_version());
    typedef bmp::evaluation<double, wmc_boost_policy>::type eval_t;
    char ev[96];
    std::snprintf(ev, sizeof ev, "%s (%d-bit significand)",
                  sizeof(eval_t) > sizeof(double) ? "long double" : "double",
                  std::numeric_limits<eval_t>::digits);
    wm_build_info_kv(f, "cpu_supplier_cxx", "boost_eval_type", ev);
#else
    wm_build_info_kv(f, "cpu_supplier_cxx", "boost",
                     "none (no Boost selected at build time; std:: row only)");
#endif
}

} /* extern "C" */
