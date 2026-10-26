/* cuda_device.cu -- matern_device.cuh in a kernel here and one in cuda_device_second.cu, both
 * before any library call, and the device-pointer entries on a non-blocking stream, each bit
 * for bit against matern_covariance_cuda() / _z() in SYNC mode; then argument checks. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matern.h"
#include "matern_device.cuh"

extern "C" int cuda_device_second_z(const double *z, const double *nu, double *out, long n);

static __global__ void first_kernel(const double *h, const double *nu, const double *s,
                                    double *out, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = matern_covariance_device(h[i], nu[i], s[i]);
}

/* points that differ in their bits, NaNs included */
static size_t differ(const double *a, const double *b, size_t n)
{
    size_t k = 0;
    for (size_t i = 0; i < n; i++) if (memcmp(a + i, b + i, sizeof *a)) k++;
    return k;
}

static double *to_device(const double *x, size_t n)
{
    double *d = NULL;
    if (cudaMalloc((void **)&d, n * sizeof *d) != cudaSuccess) return NULL;
    cudaMemcpy(d, x, n * sizeof *d, cudaMemcpyHostToDevice);
    return d;
}

int main(void)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        fprintf(stderr, "cuda_device: no CUDA device; this check cannot be skipped.\n");
        return 1;
    }

    /* Every region and the special inputs: orders log-uniform in [1e-3, 60] with every
     * fourth point a half-integer (closed form up to 29.5, the router above), arguments
     * log-uniform in [1e-4, 2000] (past the exponential limit too), then h = 0, h = inf,
     * invalid orders and scales, NaN and a subnormal separation. */
    const size_t n = 200000;
    double *h = (double *)malloc(n * sizeof *h), *nu = (double *)malloc(n * sizeof *nu);
    double *s = (double *)malloc(n * sizeof *s), *z  = (double *)malloc(n * sizeof *z);
    unsigned long long st = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i++) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        const double u = (double)(st >> 11) * 0x1.0p-53;
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        const double v = (double)(st >> 11) * 0x1.0p-53;
        nu[i] = (i % 4 == 0) ? floor(40.0 * v) + 0.5 : exp(log(1e-3) + v * log(6e4));
        z[i]  = exp(log(1e-4) + u * log(2e7));
        s[i]  = 0.5 + v;
        h[i]  = z[i] * s[i] / sqrt(2.0 * nu[i]);
    }
    const double special[][3] = { { 0.0, 1.5, 1.0 }, { INFINITY, 2.0, 1.0 }, { 1.0, 0.0, 1.0 },
                                  { 1.0, -1.0, 1.0 }, { 1.0, 1.5, 0.0 }, { -1.0, 1.5, 1.0 },
                                  { NAN, 1.5, 1.0 }, { 1.0, NAN, 1.0 }, { 4.9e-324, 0.7, 1.0 },
                                  { 1.0, INFINITY, 1.0 }, { 1.0, 1.5, INFINITY } };
    const size_t ns = sizeof special / sizeof *special;
    for (size_t k = 0; k < ns; k++) {
        h[k] = special[k][0]; nu[k] = special[k][1]; s[k] = special[k][2];
        z[k] = special[k][0];
    }
    const size_t kc = ns;                     /* nu = 3/2 at z = 2.5, against its closed form */
    nu[kc] = 1.5; z[kc] = 2.5; s[kc] = 1.0; h[kc] = 2.5 / sqrt(3.0);

    double *dh = to_device(h, n), *dnu = to_device(nu, n), *ds = to_device(s, n), *dz = to_device(z, n);
    double *dout = NULL;
    if (!dh || !dnu || !ds || !dz || cudaMalloc((void **)&dout, n * sizeof *dout) != cudaSuccess) {
        fprintf(stderr, "cuda_device: device allocation failed\n");
        return 1;
    }
    double *dev1 = (double *)malloc(n * sizeof(double)), *dev2 = (double *)malloc(n * sizeof(double));
    double *ent1 = (double *)malloc(n * sizeof(double)), *ent2 = (double *)malloc(n * sizeof(double));
    double *ref1 = (double *)malloc(n * sizeof(double)), *ref2 = (double *)malloc(n * sizeof(double));
    int failures = 0;

    /* 1, 2: the header in two translation units, before any library call */
    first_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dh, dnu, ds, dout, (long)n);
    if (cudaGetLastError() != cudaSuccess) { printf("  first kernel FAILED to launch\n"); failures++; }
    cudaMemcpy(dev1, dout, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_device_second_z(dz, dnu, dout, (long)n) != 0) { printf("  second kernel FAILED to launch\n"); failures++; }
    cudaMemcpy(dev2, dout, n * sizeof(double), cudaMemcpyDeviceToHost);

    /* the references */
    if (matern_covariance_cuda(ref1, h, nu, s, n, MATERN_CUDA_SYNC) != MATERN_OK
     || matern_covariance_cuda_z(ref2, z, nu, n, MATERN_CUDA_SYNC) != MATERN_OK) {
        printf("  reference call FAILED\n");
        return 1;
    }

    /* 3: the device-pointer entries on a stream of their own */
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaMemsetAsync(dout, 0xff, n * sizeof(double), stream);
    if (matern_covariance_cuda_device(dout, dh, dnu, ds, n, (void *)stream) != MATERN_OK) {
        printf("  matern_covariance_cuda_device FAILED\n"); failures++;
    }
    cudaMemcpyAsync(ent1, dout, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemsetAsync(dout, 0xff, n * sizeof(double), stream);
    if (matern_covariance_cuda_device_z(dout, dz, dnu, n, (void *)stream) != MATERN_OK) {
        printf("  matern_covariance_cuda_device_z FAILED\n"); failures++;
    }
    cudaMemcpyAsync(ent2, dout, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const size_t d1 = differ(dev1, ref1, n), d2 = differ(dev2, ref2, n);
    const size_t e1 = differ(ent1, ref1, n), e2 = differ(ent2, ref2, n);
    printf("  matern_covariance_device,   first unit:   %zu of %zu points differ from matern_covariance_cuda\n", d1, n);
    printf("  matern_covariance_z_device, second unit:  %zu of %zu points differ from matern_covariance_cuda_z\n", d2, n);
    printf("  matern_covariance_cuda_device,   stream:  %zu of %zu points differ\n", e1, n);
    printf("  matern_covariance_cuda_device_z, stream:  %zu of %zu points differ\n", e2, n);
    failures += (d1 != 0) + (d2 != 0) + (e1 != 0) + (e2 != 0);

    /* the closed form at nu = 3/2: M = (1 + z) e^-z; the special point h = 0 gives 1 */
    const double want = 3.5 * exp(-2.5);
    if (!(fabs(dev2[kc] - want) <= 1e-14 * want)) {
        printf("  nu = 3/2 at z = 2.5: %.17g, closed form %.17g\n", dev2[kc], want); failures++;
    }
    if (dev1[0] != 1.0) { printf("  h = 0 gives %.17g, not 1\n", dev1[0]); failures++; }

    /* 4: argument checks, as the stand-ins in src/backends/cuda_absent.c */
    if (matern_covariance_cuda_device(dout, dh, dnu, ds, 0, NULL) != MATERN_OK
     || matern_covariance_cuda_device(NULL, dh, dnu, ds, n, NULL) != MATERN_INVALID_ARGUMENT
     || matern_covariance_cuda_device_z(dout, NULL, dnu, n, NULL) != MATERN_INVALID_ARGUMENT) {
        printf("  argument checks FAILED\n"); failures++;
    }

    cudaStreamDestroy(stream);
    cudaFree(dh); cudaFree(dnu); cudaFree(ds); cudaFree(dz); cudaFree(dout);
    free(h); free(nu); free(s); free(z);
    free(dev1); free(dev2); free(ent1); free(ent2); free(ref1); free(ref2);
    printf(failures ? "\ncuda_device: %d FAILED\n" : "\ncuda_device: passed\n", failures);
    return failures ? 1 : 0;
}
