/* cpu_gsl.c -- the GNU Scientific Library as a K_nu supplier (CPU only), with K_nu,
 * K_nu e^x and ln K_nu all native.  GSL (GPL-3.0) is linked, not distributed: no GSL
 * source is in this tree.  The library is the static GSL that `make externals` builds,
 * else a system GSL found through pkg-config (benchmark/tools/gsl_flags.sh).  This
 * wrapper is this project's code (MIT, ../../LICENSE). */

#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_version.h>
#include "build_info.h"
#include "cpu_supplier.h"

/* GSL's default error handler calls abort() (e.g. on the overflow in
 * gsl_sf_bessel_Knu(1000, 0.1)), so it is disabled before any call into GSL.  The
 * object is linked through the supplier struct below, so the constructor runs before
 * main.  With the handler off, the three entries below return the computed value (0 on
 * underflow, inf on overflow). */
__attribute__((constructor))
static void cpu_supplier_gsl_init(void) { gsl_set_error_handler_off(); }

/* The three forms, passed through unmodified; the harness's non-finite
 * accounting decides whether a point is scored. */
static double gsl_kv   (double nu, double z) { return gsl_sf_bessel_Knu(nu, z); }
static double gsl_kxv  (double nu, double z) { return gsl_sf_bessel_Knu_scaled(nu, z); }
static double gsl_logkv(double nu, double z) { return gsl_sf_bessel_lnKnu(nu, z); }

const cpu_supplier cpu_supplier_gsl_specfunc = {
    "GNU Scientific Library, specfunc/bessel_Knu.c (GPL-3.0)",
    CPU_SUPPLIER_LOG_NATIVE,
    gsl_kv, gsl_kxv, gsl_logkv
};

/* Version for the row label and the run header, from the headers this object is
 * compiled against (GSL_VERSION), not the shared library found at run time. */
const char *cpu_gsl_version(void)
{
    static char buf[64];
    snprintf(buf, sizeof buf, "%s", GSL_VERSION);
    return buf;
}

/* Build record of this object (build_info.h): the GSL headers it is compiled against
 * (GSL_VERSION), the loaded library (gsl_version), which computes the row, the copy the
 * build selected (CPU_SUPPLIER_GSL_SOURCE, from gsl_flags.sh --where) and the library file
 * the build linked, with its package or build flags and the FMA instructions in it
 * (CPU_SUPPLIER_GSL_LIBRARY, from gsl_flags.sh --library); the headers and the loaded
 * library can come from different installations.  The keys of BENCH_BUILD_INFO_COMMON
 * describe this object, not the library. */
void bench_build_info_cpu_supplier_gsl(FILE *f)
{
    BENCH_BUILD_INFO_COMMON(f, "cpu_supplier_gsl");
    bench_build_info_kv(f, "cpu_supplier_gsl", "headers", GSL_VERSION);
    bench_build_info_kv(f, "cpu_supplier_gsl", "runtime", gsl_version);
#ifdef CPU_SUPPLIER_GSL_SOURCE
    bench_build_info_kv(f, "cpu_supplier_gsl", "source", CPU_SUPPLIER_GSL_SOURCE);
#endif
#ifdef CPU_SUPPLIER_GSL_LIBRARY
    bench_build_info_kv(f, "cpu_supplier_gsl", "library", CPU_SUPPLIER_GSL_LIBRARY);
#endif
}
