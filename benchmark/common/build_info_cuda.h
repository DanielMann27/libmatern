/* build_info_cuda.h -- the `--build-info` record of a CUDA driver: compilers, standard,
 * host ISA, CUDA headers, and the loaded runtime and driver API versions.  The version
 * queries need no context, so the record is available without a GPU. */
#ifndef BENCH_BUILD_INFO_CUDA_H
#define BENCH_BUILD_INFO_CUDA_H

#include <stdio.h>
#include <cuda_runtime.h>
#include "../../references/common/build_info.h"

/* The device target architectures are not recorded here: run_benchmark.sh reads them
 * from the binaries with cuobjdump. */
static inline void bench_build_info_cuda(const char *unit)
{
    char v[96];
    BENCH_BUILD_INFO_COMMON(stdout, unit);
    snprintf(v, sizeof v, "%d.%d", CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10);
    bench_build_info_kv(stdout, unit, "cuda_headers", v);
    int rt = 0, drv = 0;
    if (cudaRuntimeGetVersion(&rt) == cudaSuccess) {
        snprintf(v, sizeof v, "%d.%d", rt / 1000, rt % 1000 / 10);
        bench_build_info_kv(stdout, unit, "cuda_runtime", v);
    } else {
        bench_build_info_kv(stdout, unit, "cuda_runtime", "unavailable");
    }
    if (cudaDriverGetVersion(&drv) == cudaSuccess && drv > 0) {
        snprintf(v, sizeof v, "%d.%d", drv / 1000, drv % 1000 / 10);
        bench_build_info_kv(stdout, unit, "cuda_driver_api", v);
    } else {
        bench_build_info_kv(stdout, unit, "cuda_driver_api", "none (no driver on this host)");
    }
}

#endif /* BENCH_BUILD_INFO_CUDA_H */
