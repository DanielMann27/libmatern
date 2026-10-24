/* cuda_device_second.cu -- the second translation unit of make cuda-device: a kernel of its
 * own on matern_device.cuh, so that the header's static tables and functions are linked
 * twice into one program. */
#include "matern_device.cuh"

static __global__ void second_kernel_z(const double *z, const double *nu, double *out, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = matern_covariance_z_device(z[i], nu[i]);
}

/* Device arrays; returns the launch's error code. */
extern "C" int cuda_device_second_z(const double *z, const double *nu, double *out, long n)
{
    second_kernel_z<<<(unsigned)((n + 255) / 256), 256>>>(z, nu, out, n);
    return (int)cudaGetLastError();
}
