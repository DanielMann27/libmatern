/* matern_device.cuh -- the normalised Matern covariance M(h) of matern.h as __device__
 * functions for the caller's own CUDA kernels; header-only: no library object, host
 * setup or -rdc.
 *
 * USAGE
 *      #include "matern_device.cuh"           // nvcc -I<prefix>/include
 *
 *      __global__ void covariance_row(const double *x, double xi, double nu, double s,
 *                                     double *out, int n)
 *      {
 *          int j = blockIdx.x * blockDim.x + threadIdx.x;
 *          if (j < n) out[j] = matern_covariance_device(fabs(x[j] - xi), nu, s);
 *      }
 *
 * With the library's default build, and this header compiled with the nvcc flags of
 * build/cuda.o, a point gives the bits of matern_covariance_cuda() and
 * matern_covariance_cuda_z() for the same input; input rules and domain as in matern.h.
 * For arrays already in device memory, see matern_covariance_cuda_device() in matern.h.
 *
 * COMPILATION
 * The library is built with nvcc's default -fmad=true; -fmad=false changes the
 * arithmetic and so the results.  Each translation unit that includes this header holds
 * its own copy of the __constant__ tables.  Neighbouring threads whose points fall in
 * different regions of the (nu, z) plane diverge, as in the library's kernel.
 */
#ifndef MATERN_DEVICE_CUH
#define MATERN_DEVICE_CUH

#include "matern_device/region_router_device.cuh"

/* M(h) at separation h >= 0, order nu > 0 and scale s > 0. */
static __device__ __forceinline__ double matern_covariance_device(double h, double nu,
                                                                  double scale)
{
    return matern_route_device<MATERN_ROUTE_FLAGS_DEFAULT>(h, nu, scale);
}

/* M_nu(z), given z = sqrt(2 nu) h / s directly. */
static __device__ __forceinline__ double matern_covariance_z_device(double z, double nu)
{
    return matern_route_device<MATERN_ROUTE_FLAGS_DEFAULT_Z>(z, nu, 1.0);
}

#endif /* MATERN_DEVICE_CUH */
