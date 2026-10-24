/* cpu_wm_registry.h -- the external suppliers in this build, one definition for every
 * harness; optional ones under WMC_WITH_CXX, WMC_WITH_BOOST, WMC_WITH_GSL and WMC_WITH_KAUST
 * (references/build/defines.txt).  Boost and GSL labels carry the library version. */
#ifndef CPU_WM_REGISTRY_H
#define CPU_WM_REGISTRY_H

#include <stdio.h>
#include <string.h>
#include "cpu_wm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { const wmc_supplier *s; const char *label; } wmc_entry;

#ifdef WMC_WITH_BOOST
const char *cpu_cxx_boost_version(void);
#endif
#ifdef WMC_WITH_GSL
const char *cpu_gsl_version(void);
#endif

/* Fills `e` with at most `cap` entries and returns how many.  The order is
 * fixed, and is the row order of every table. */
static inline int wmc_build_list(wmc_entry *e, int cap)
{
    int n = 0;
    if (n < cap) e[n++] = (wmc_entry){ &wmc_bessels_jl,  "bessels.jl" };
    if (n < cap) e[n++] = (wmc_entry){ &wmc_bessels_old, "bessels.jl-old" };
#ifdef WMC_WITH_CXX
    if (n < cap) e[n++] = (wmc_entry){ &wmc_cxx_std,     "std-cyl_bessel_k" };
#endif
#ifdef WMC_WITH_BOOST
    static char boostlab[64];
    if (n < cap) {
        snprintf(boostlab, sizeof boostlab, "boost-%s", cpu_cxx_boost_version());
        e[n++] = (wmc_entry){ &wmc_cxx_boost, boostlab };
    }
#endif
#ifdef WMC_WITH_GSL
    static char gsllab[64];
    if (n < cap) {
        snprintf(gsllab, sizeof gsllab, "gsl-%s", cpu_gsl_version());
        e[n++] = (wmc_entry){ &wmc_gsl, gsllab };
    }
#endif
#ifdef WMC_WITH_KAUST
    if (n < cap) e[n++] = (wmc_entry){ &wmc_kaust,   "kaust-128" };
    if (n < cap) e[n++] = (wmc_entry){ &wmc_kaust40, "kaust-40" };
    /* `kaust-adaptive`: 128 panels, `e10` applied, non-overflowing log(cosh),
     * and the integration range of Geng et al.'s Eq. (7) instead of t in [0, 9]. */
    if (n < cap) e[n++] = (wmc_entry){ &wmc_kaust4,  "kaust-adaptive" };
#endif
    return n;
}

/* One `  (<row> absent: ...)` line per optional supplier this build lacks, for
 * --list.  The lines start with "  (", so the row parsers of the run scripts
 * (`^  [a-z]`) skip them; make requirements gives the command that adds each,
 * and `make kaust` states the copyright situation of CuBesselK. */
static inline void wmc_print_absent(void)
{
#ifndef WMC_WITH_CXX
    puts("  (std-cyl_bessel_k absent: no C++17 compiler in this build; make requirements)");
#endif
#ifndef WMC_WITH_BOOST
    puts("  (boost-1.75.0 absent: no Boost 1.75.0 in this build; make requirements)");
#endif
#ifndef WMC_WITH_GSL
    puts("  (gsl absent: no GSL in this build; make requirements)");
#endif
#ifndef WMC_WITH_KAUST
    puts("  (KAUST absent: CuBesselK not ported; make kaust)");
#endif
}

/* Resolve a label to a supplier, or NULL.  Callers report an unknown label
 * rather than falling back to a default. */
static inline const wmc_entry *wmc_find(const wmc_entry *e, int n,
                                        const char *label)
{
    for (int i = 0; i < n; i++)
        if (label && e[i].label && !strcmp(e[i].label, label)) return &e[i];
    return NULL;
}

#ifdef __cplusplus
}
#endif
#endif /* CPU_WM_REGISTRY_H */
