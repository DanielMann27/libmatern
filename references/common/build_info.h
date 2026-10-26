/* build_info.h -- `build_<unit>_<key>  <value>` lines recording, at compile time, the
 * compiler, standard, ISA extensions and C/C++ runtime headers of the including C, C++ or
 * CUDA unit; bench_build_info_kv() adds library keys, `_runtime` marks load-time versions. */
#ifndef MATERN_BUILD_INFO_H
#define MATERN_BUILD_INFO_H

#include <stdio.h>

#define BENCH_BUILD_INFO_STR2(x) #x
#define BENCH_BUILD_INFO_STR(x)  BENCH_BUILD_INFO_STR2(x)

#if defined(__CUDACC__)
#  define BENCH_BUILD_INFO_COMPILER "nvcc " BENCH_BUILD_INFO_STR(__CUDACC_VER_MAJOR__) "." \
                         BENCH_BUILD_INFO_STR(__CUDACC_VER_MINOR__) "." BENCH_BUILD_INFO_STR(__CUDACC_VER_BUILD__)
#elif defined(__INTEL_LLVM_COMPILER)
#  define BENCH_BUILD_INFO_COMPILER "icx " BENCH_BUILD_INFO_STR(__INTEL_LLVM_COMPILER)
#elif defined(__clang__)
#  define BENCH_BUILD_INFO_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#  define BENCH_BUILD_INFO_COMPILER "gcc " __VERSION__
#elif defined(_MSC_VER)
#  define BENCH_BUILD_INFO_COMPILER "msvc " BENCH_BUILD_INFO_STR(_MSC_FULL_VER)
#else
#  define BENCH_BUILD_INFO_COMPILER "unknown"
#endif

/* Under nvcc, the host compiler of the host pass. */
#if defined(__CUDACC__) && defined(__clang__)
#  define BENCH_BUILD_INFO_HOST_COMPILER "clang " __clang_version__
#elif defined(__CUDACC__) && defined(__GNUC__)
#  define BENCH_BUILD_INFO_HOST_COMPILER "gcc " __VERSION__
#endif

#if defined(__cplusplus)
#  define BENCH_BUILD_INFO_STD "c++ " BENCH_BUILD_INFO_STR(__cplusplus)
#elif defined(__STDC_VERSION__)
#  define BENCH_BUILD_INFO_STD "c " BENCH_BUILD_INFO_STR(__STDC_VERSION__)
#else
#  define BENCH_BUILD_INFO_STD "c89"
#endif

#if defined(__AVX512F__)
#  define BENCH_BUILD_INFO_ISA_512 " avx512f"
#else
#  define BENCH_BUILD_INFO_ISA_512 ""
#endif
#if defined(__AVX2__)
#  define BENCH_BUILD_INFO_ISA_AVX2 " avx2"
#else
#  define BENCH_BUILD_INFO_ISA_AVX2 ""
#endif
#if defined(__FMA__)
#  define BENCH_BUILD_INFO_ISA_FMA " fma"
#else
#  define BENCH_BUILD_INFO_ISA_FMA ""
#endif
#if defined(__SSE4_1__)
#  define BENCH_BUILD_INFO_ISA_SSE4 " sse4.1"
#else
#  define BENCH_BUILD_INFO_ISA_SSE4 ""
#endif
#if defined(__x86_64__) || defined(_M_X64)
#  define BENCH_BUILD_INFO_ISA_BASE "x86-64"
#elif defined(__aarch64__)
#  define BENCH_BUILD_INFO_ISA_BASE "aarch64"
#else
#  define BENCH_BUILD_INFO_ISA_BASE "other"
#endif
#define BENCH_BUILD_INFO_ISA BENCH_BUILD_INFO_ISA_BASE BENCH_BUILD_INFO_ISA_SSE4 BENCH_BUILD_INFO_ISA_AVX2 \
                             BENCH_BUILD_INFO_ISA_FMA BENCH_BUILD_INFO_ISA_512

static inline void bench_build_info_kv(FILE *f, const char *unit, const char *key, const char *val)
{
    char k[96];
    snprintf(k, sizeof k, "build_%s_%s", unit, key);
    fprintf(f, "%-42s %s\n", k, val);
}

#if defined(__GLIBC__)
#  define BENCH_BUILD_INFO_GLIBC_HEADERS(f, unit) \
       bench_build_info_kv(f, unit, "glibc_headers", BENCH_BUILD_INFO_STR(__GLIBC__) "." BENCH_BUILD_INFO_STR(__GLIBC_MINOR__))
#else
#  define BENCH_BUILD_INFO_GLIBC_HEADERS(f, unit) ((void)0)
#endif
#if defined(__cplusplus) && defined(_GLIBCXX_RELEASE)
#  define BENCH_BUILD_INFO_CXXLIB(f, unit) \
       bench_build_info_kv(f, unit, "libstdcxx", BENCH_BUILD_INFO_STR(_GLIBCXX_RELEASE) " (" BENCH_BUILD_INFO_STR(__GLIBCXX__) ")")
#elif defined(__cplusplus) && defined(_LIBCPP_VERSION)
#  define BENCH_BUILD_INFO_CXXLIB(f, unit) bench_build_info_kv(f, unit, "libcxx", BENCH_BUILD_INFO_STR(_LIBCPP_VERSION))
#else
#  define BENCH_BUILD_INFO_CXXLIB(f, unit) ((void)0)
#endif

#if defined(BENCH_BUILD_INFO_HOST_COMPILER)
#  define BENCH_BUILD_INFO_HOST(f, unit) bench_build_info_kv(f, unit, "host_compiler", BENCH_BUILD_INFO_HOST_COMPILER)
#else
#  define BENCH_BUILD_INFO_HOST(f, unit) ((void)0)
#endif

#define BENCH_BUILD_INFO_COMMON(f, unit) do {                                    \
        bench_build_info_kv(f, unit, "compiler", BENCH_BUILD_INFO_COMPILER);     \
        BENCH_BUILD_INFO_HOST(f, unit);                                          \
        bench_build_info_kv(f, unit, "std", BENCH_BUILD_INFO_STD);               \
        bench_build_info_kv(f, unit, "isa", BENCH_BUILD_INFO_ISA);               \
        BENCH_BUILD_INFO_GLIBC_HEADERS(f, unit);                                 \
        BENCH_BUILD_INFO_CXXLIB(f, unit);                                        \
    } while (0)

/* Per-object records of the CPU suppliers, each defined in the object it describes: the C
 * one always, the C++ one under CPU_SUPPLIER_WITH_CXX, the GSL one under CPU_SUPPLIER_WITH_GSL. */
#ifdef __cplusplus
extern "C" {
#endif
void bench_build_info_cpu_supplier_c(FILE *f);     /* common/build_info_c.c   */
void bench_build_info_cpu_supplier_cxx(FILE *f);   /* cpp_stdlib/cpu_cxx.cpp  */
void bench_build_info_cpu_supplier_gsl(FILE *f);   /* gsl/cpu_gsl.c           */
#ifdef __cplusplus
}
#endif

#endif /* MATERN_BUILD_INFO_H */
