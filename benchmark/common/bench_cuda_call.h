/* bench_cuda_call.h -- the timing loop of the public CUDA entry, shared by benchmark_gpu and
 * suppliers_gpu so that both time the same thing: BENCH_WARMUP_REPS untimed calls, then per
 * repetition a device drain outside the timer and one call inside it, in ns per element.
 * The entry allocates and frees its device memory on every call, and those calls take less
 * time while the process holds device allocations of its own (13-24 % of the call at 10^5
 * points), so a caller holds none while this runs. */
#ifndef MATERN_BENCH_CUDA_CALL_H
#define MATERN_BENCH_CUDA_CALL_H

#include <stddef.h>
#include <cuda_runtime.h>
#include "bench_timing.h"

/* call(ctx) makes one call over n points and returns non-zero on failure.  Fills t[0..reps)
 * and returns 0, or returns 1 at the first failed call. */
static int bench_time_cuda_call(int (*call)(void *), void *ctx, size_t n, int reps,
                                double *t)
{
    for (int w = 0; w < BENCH_WARMUP_REPS; w++)
        if (call(ctx)) return 1;
    for (int r = 0; r < reps; r++) {
        cudaDeviceSynchronize();
        const double t0 = bench_now_s();
        if (call(ctx)) return 1;
        t[r] = (bench_now_s() - t0) * 1e9 / (double)n;
    }
    return 0;
}

#endif
