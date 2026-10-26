/* build_info_c.c -- build record of the C supplier objects (Bessels.jl ports, KAUST
 * port, GSL wrapper).  Compiled with the same $(CC) and CFLAGS as those objects
 * (references/Makefile), so the compiler, standard and ISA it reports are theirs. */
#include "build_info.h"

void bench_build_info_cpu_supplier_c(FILE *f)
{
    BENCH_BUILD_INFO_COMMON(f, "cpu_supplier_c");
}
