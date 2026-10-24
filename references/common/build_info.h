/* build_info.h -- `build_<unit>_<key>  <value>` lines recording, at compile time, the
 * compiler, standard, ISA extensions and C/C++ runtime headers of the including C, C++ or
 * CUDA unit; wm_build_info_kv() adds library keys, `_runtime` marks load-time versions. */
#ifndef MATERN_BUILD_INFO_H
#define MATERN_BUILD_INFO_H

#include <stdio.h>

#define WM_BI_STR2(x) #x
#define WM_BI_STR(x)  WM_BI_STR2(x)

#if defined(__CUDACC__)
#  define WM_BI_COMPILER "nvcc " WM_BI_STR(__CUDACC_VER_MAJOR__) "." \
                         WM_BI_STR(__CUDACC_VER_MINOR__) "." WM_BI_STR(__CUDACC_VER_BUILD__)
#elif defined(__INTEL_LLVM_COMPILER)
#  define WM_BI_COMPILER "icx " WM_BI_STR(__INTEL_LLVM_COMPILER)
#elif defined(__clang__)
#  define WM_BI_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#  define WM_BI_COMPILER "gcc " __VERSION__
#elif defined(_MSC_VER)
#  define WM_BI_COMPILER "msvc " WM_BI_STR(_MSC_FULL_VER)
#else
#  define WM_BI_COMPILER "unknown"
#endif

/* Under nvcc, the host compiler of the host pass. */
#if defined(__CUDACC__) && defined(__clang__)
#  define WM_BI_HOST_COMPILER "clang " __clang_version__
#elif defined(__CUDACC__) && defined(__GNUC__)
#  define WM_BI_HOST_COMPILER "gcc " __VERSION__
#endif

#if defined(__cplusplus)
#  define WM_BI_STD "c++ " WM_BI_STR(__cplusplus)
#elif defined(__STDC_VERSION__)
#  define WM_BI_STD "c " WM_BI_STR(__STDC_VERSION__)
#else
#  define WM_BI_STD "c89"
#endif

#if defined(__AVX512F__)
#  define WM_BI_ISA_512 " avx512f"
#else
#  define WM_BI_ISA_512 ""
#endif
#if defined(__AVX2__)
#  define WM_BI_ISA_AVX2 " avx2"
#else
#  define WM_BI_ISA_AVX2 ""
#endif
#if defined(__FMA__)
#  define WM_BI_ISA_FMA " fma"
#else
#  define WM_BI_ISA_FMA ""
#endif
#if defined(__SSE4_1__)
#  define WM_BI_ISA_SSE4 " sse4.1"
#else
#  define WM_BI_ISA_SSE4 ""
#endif
#if defined(__x86_64__) || defined(_M_X64)
#  define WM_BI_ISA_BASE "x86-64"
#elif defined(__aarch64__)
#  define WM_BI_ISA_BASE "aarch64"
#else
#  define WM_BI_ISA_BASE "other"
#endif
#define WM_BI_ISA WM_BI_ISA_BASE WM_BI_ISA_SSE4 WM_BI_ISA_AVX2 WM_BI_ISA_FMA WM_BI_ISA_512

static inline void wm_build_info_kv(FILE *f, const char *unit, const char *key, const char *val)
{
    char k[96];
    snprintf(k, sizeof k, "build_%s_%s", unit, key);
    fprintf(f, "%-42s %s\n", k, val);
}

#if defined(__GLIBC__)
#  define WM_BI_GLIBC_HEADERS(f, unit) \
       wm_build_info_kv(f, unit, "glibc_headers", WM_BI_STR(__GLIBC__) "." WM_BI_STR(__GLIBC_MINOR__))
#else
#  define WM_BI_GLIBC_HEADERS(f, unit) ((void)0)
#endif
#if defined(__cplusplus) && defined(_GLIBCXX_RELEASE)
#  define WM_BI_CXXLIB(f, unit) \
       wm_build_info_kv(f, unit, "libstdcxx", WM_BI_STR(_GLIBCXX_RELEASE) " (" WM_BI_STR(__GLIBCXX__) ")")
#elif defined(__cplusplus) && defined(_LIBCPP_VERSION)
#  define WM_BI_CXXLIB(f, unit) wm_build_info_kv(f, unit, "libcxx", WM_BI_STR(_LIBCPP_VERSION))
#else
#  define WM_BI_CXXLIB(f, unit) ((void)0)
#endif

#if defined(WM_BI_HOST_COMPILER)
#  define WM_BI_HOST(f, unit) wm_build_info_kv(f, unit, "host_compiler", WM_BI_HOST_COMPILER)
#else
#  define WM_BI_HOST(f, unit) ((void)0)
#endif

#define WM_BUILD_INFO_COMMON(f, unit) do {                         \
        wm_build_info_kv(f, unit, "compiler", WM_BI_COMPILER);     \
        WM_BI_HOST(f, unit);                                       \
        wm_build_info_kv(f, unit, "std", WM_BI_STD);               \
        wm_build_info_kv(f, unit, "isa", WM_BI_ISA);               \
        WM_BI_GLIBC_HEADERS(f, unit);                              \
        WM_BI_CXXLIB(f, unit);                                     \
    } while (0)

/* Per-object records of the CPU suppliers, each defined in the object it describes:
 * the C one always, the C++ one under WMC_WITH_CXX, the GSL one under WMC_WITH_GSL. */
#ifdef __cplusplus
extern "C" {
#endif
void wm_build_info_cpu_supplier_c(FILE *f);     /* common/build_info_c.c   */
void wm_build_info_cpu_supplier_cxx(FILE *f);   /* cpp_stdlib/cpu_cxx.cpp  */
void wm_build_info_cpu_supplier_gsl(FILE *f);   /* gsl/cpu_gsl.c           */
#ifdef __cplusplus
}
#endif

#endif /* MATERN_BUILD_INFO_H */
