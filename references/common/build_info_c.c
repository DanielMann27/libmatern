/* build_info_c.c -- build record of the C supplier objects (Bessels.jl ports, KAUST
 * port, GSL wrapper).  Compiled by the same rule as those objects (references/Makefile),
 * so the compiler, standard and ISA it reports are theirs. */
#include "build_info.h"

void wm_build_info_cpu_supplier_c(FILE *f)
{
    WM_BUILD_INFO_COMMON(f, "cpu_supplier_c");
}
