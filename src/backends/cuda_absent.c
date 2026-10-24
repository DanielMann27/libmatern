/* cuda_absent.c -- weak definitions of the public CUDA entry points for a library
 * built without CUDA (no device; a valid evaluation call returns MATERN_UNSUPPORTED_CPU).
 * The strong definitions in build/cuda.o take precedence in either link order. */
#include <stddef.h>

#include "matern.h"

#define MATERN_WEAK __attribute__((weak))

static int mode_ok(int mode)
{
    return mode == MATERN_CUDA_SYNC
        || mode == MATERN_CUDA_ASYNC_DIRECT
        || mode == MATERN_CUDA_ASYNC_RING
        || mode == MATERN_CUDA_ASYNC_MAPPED;
}

MATERN_WEAK int matern_cuda_available(void)
{
    return 0;
}

MATERN_WEAK const char *matern_cuda_device_name(void)
{
    return "";
}

/* The argument checks follow cuda.cu in the same order (mode, n = 0, null pointers,
 * device), so the status matches a CUDA build wherever the device does not decide it. */
MATERN_WEAK int matern_covariance_cuda_tuned(double *out, const double *h, const double *nu,
                                             const double *scale, size_t n, int mode,
                                             const matern_cuda_options *options)
{
    (void)options;
    if (!mode_ok(mode)) return MATERN_INVALID_ARGUMENT;
    if (n == 0) return MATERN_OK;
    if (!out || !h || !nu || !scale) return MATERN_INVALID_ARGUMENT;
    return MATERN_UNSUPPORTED_CPU;
}

MATERN_WEAK int matern_covariance_cuda_z_tuned(double *out, const double *z, const double *nu,
                                               size_t n, int mode,
                                               const matern_cuda_options *options)
{
    (void)options;
    if (!mode_ok(mode)) return MATERN_INVALID_ARGUMENT;
    if (n == 0) return MATERN_OK;
    if (!out || !z || !nu) return MATERN_INVALID_ARGUMENT;
    return MATERN_UNSUPPORTED_CPU;
}

MATERN_WEAK int matern_covariance_cuda(double *out, const double *h, const double *nu,
                                       const double *scale, size_t n, int mode)
{
    return matern_covariance_cuda_tuned(out, h, nu, scale, n, mode, NULL);
}

MATERN_WEAK int matern_covariance_cuda_z(double *out, const double *z, const double *nu,
                                         size_t n, int mode)
{
    return matern_covariance_cuda_z_tuned(out, z, nu, n, mode, NULL);
}

MATERN_WEAK int matern_covariance_cuda_device(double *out, const double *h, const double *nu,
                                              const double *scale, size_t n, void *stream)
{
    (void)stream;
    if (n == 0) return MATERN_OK;
    if (!out || !h || !nu || !scale) return MATERN_INVALID_ARGUMENT;
    return MATERN_UNSUPPORTED_CPU;
}

MATERN_WEAK int matern_covariance_cuda_device_z(double *out, const double *z, const double *nu,
                                                size_t n, void *stream)
{
    (void)stream;
    if (n == 0) return MATERN_OK;
    if (!out || !z || !nu) return MATERN_INVALID_ARGUMENT;
    return MATERN_UNSUPPORTED_CPU;
}

MATERN_WEAK void matern_cuda_release(void)
{
}
