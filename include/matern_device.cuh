/* matern_device.cuh -- the normalised Matern covariance M(h) of matern.h as __device__
 * functions, for use inside the caller's own CUDA kernels.  Header-only: it needs no
 * library object, no host setup and no -rdc, and runs on whatever stream launches the
 * caller's kernel.
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
 * A point gives the bits of matern_covariance_cuda() and matern_covariance_cuda_z() for
 * the same input, whose kernels run the same per-point code, with the input rules and
 * domain stated in matern.h.  For arrays already in device memory, see
 * matern_covariance_cuda_device() in matern.h.
 *
 * COMPILATION
 * The library is built with nvcc's defaults for the arithmetic (-fmad=true); -fmad=false
 * changes the compiled arithmetic, so results are then no longer the library's.  Every
 * translation unit that includes this header holds its own copy of the constant tables,
 * about 5 KB of the 64 KB of __constant__ memory per module.  The evaluation is inlined
 * and takes about 50 registers, which the calling kernel needs at least.  Neighbouring
 * threads whose points fall in different regions of the (nu, z) plane diverge, as in
 * the library's own kernel.
 */
#ifndef MATERN_DEVICE_CUH
#define MATERN_DEVICE_CUH

#include "matern_device/region_router_device.cuh"

/* M(h) at separation h >= 0, order nu > 0 and scale s > 0. */
static __device__ __forceinline__ double matern_covariance_device(double h, double nu,
                                                                  double scale)
{
    return matern_route_device<MATERN_ROUTE_FLAGS_PRIMEA>(h, nu, scale);
}

/* M_nu(z), given z = sqrt(2 nu) h / s directly. */
static __device__ __forceinline__ double matern_covariance_z_device(double z, double nu)
{
    return matern_route_device<MATERN_ROUTE_FLAGS_PRIMEAZ>(z, nu, 1.0);
}

#endif /* MATERN_DEVICE_CUH */
