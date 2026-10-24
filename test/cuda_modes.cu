/* cuda_modes.cu -- the CUDA transfer modes differ only in how data moves: an asynchronous
 * call leaves page registrations as the caller had them (1), every asynchronous strategy
 * returns the bits of the synchronous mode (2), the backend works after release (3). */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include <string.h>
#include <math.h>
#include "matern.h"

/* The pinning probe: cudaHostGetFlags fails on a pointer that is not page-locked. */
static int is_pinned(const void *p)
{
    unsigned int flags = 0;
    cudaError_t e = cudaHostGetFlags(&flags, (void *)(uintptr_t)p);
    cudaGetLastError();
    return e == cudaSuccess;
}

int main(void)
{
    if (!matern_cuda_available()) {
        /* Without a device the test fails rather than skips. */
        fprintf(stderr, "cuda_modes: no CUDA device; this check cannot be skipped.\n");
        return 1;
    }
    printf("cuda_modes: %s\n\n", matern_cuda_device_name());

    const size_t n = 100000;
    double *h     = (double *)malloc(n * sizeof *h);
    double *nu    = (double *)malloc(n * sizeof *nu);
    double *scale = (double *)malloc(n * sizeof *scale);
    double *a     = (double *)malloc(n * sizeof *a);
    double *b     = (double *)malloc(n * sizeof *b);

    unsigned long long st = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i++) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        double u = (double)(st >> 11) * 0x1.0p-53;
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        double v = (double)(st >> 11) * 0x1.0p-53;
        nu[i]    = 0.25 + 30.0 * v;
        scale[i] = 1.0;
        h[i]     = (1e-3 + 30.0 * u) * scale[i] / sqrt(2.0 * nu[i]);
    }

    int failures = 0;

    /* 1a. buffers the caller did not pin must come back unpinned */
    const int pinned_before = is_pinned(h) || is_pinned(nu) || is_pinned(scale) || is_pinned(a);
    if (pinned_before) {
        printf("  unexpected: a fresh malloc was already page-locked\n");
        failures++;
    }
    if (matern_covariance_cuda(a, h, nu, scale, n, MATERN_CUDA_ASYNC) != MATERN_OK) {
        printf("  async call FAILED\n"); failures++;
    }
    const int leaked = is_pinned(h) || is_pinned(nu) || is_pinned(scale) || is_pinned(a);
    printf("  %-46s %s\n", "async leaves unpinned buffers unpinned",
           leaked ? "FAIL -- the call kept pages registered" : "PASS");
    if (leaked) failures++;

    /* 1b. buffers the caller did pin must still be pinned */
    if (cudaHostRegister(h, n * sizeof *h, cudaHostRegisterDefault) != cudaSuccess) {
        cudaGetLastError();
        printf("  (could not pin the caller's buffer; 1b not exercised)\n");
    } else {
        if (matern_covariance_cuda(a, h, nu, scale, n, MATERN_CUDA_ASYNC) != MATERN_OK) {
            printf("  async call over a caller-pinned buffer FAILED\n"); failures++;
        }
        const int still = is_pinned(h);
        printf("  %-46s %s\n", "async leaves caller-pinned buffers pinned",
               still ? "PASS" : "FAIL -- the call released a page it did not lock");
        if (!still) failures++;
        cudaHostUnregister(h); cudaGetLastError();
    }

    /* 2. every asynchronous strategy agrees with sync, bit for bit.
     * DIRECT transfers from the caller's registered pages, RING stages through
     * library-owned pinned buffers, MAPPED uses mapped host memory. */
    memset(a, 0, n * sizeof *a);
    if (matern_covariance_cuda(a, h, nu, scale, n, MATERN_CUDA_SYNC) != MATERN_OK) {
        printf("  sync returned an error\n"); failures++;
    }
    size_t usable = 0;
    for (size_t i = 0; i < n; i++)
        if (a[i] == a[i] && a[i] > 0.0 && a[i] <= 1.0) usable++;
    /* The comparison is refused unless most sync results lie in (0, 1]: two
     * modes that both return zeros would agree trivially. */
    const int sound = (usable * 2 > n);

    static const struct { int mode; const char *name; } ASYNCS[] = {
        { MATERN_CUDA_ASYNC_DIRECT, "direct" },
        { MATERN_CUDA_ASYNC_RING,   "ring"   },
        { MATERN_CUDA_ASYNC_MAPPED, "mapped" },
    };
    for (size_t k = 0; k < sizeof ASYNCS / sizeof ASYNCS[0]; k++) {
        memset(b, 0, n * sizeof *b);
        char label[64];
        snprintf(label, sizeof label, "%s agrees with sync bit for bit", ASYNCS[k].name);
        if (matern_covariance_cuda(b, h, nu, scale, n, ASYNCS[k].mode) != MATERN_OK) {
            printf("  %-46s FAIL -- returned an error\n", label); failures++; continue;
        }
        size_t differing = 0;
        for (size_t i = 0; i < n; i++) {
            unsigned long long x, y;
            memcpy(&x, &a[i], 8); memcpy(&y, &b[i], 8);
            if (x != y) differing++;
        }
        printf("  %-46s %s (%zu differing)\n", label,
               !sound ? "REFUSED -- the reference is degenerate"
                      : (differing == 0 ? "PASS" : "FAIL"), differing);
        if (differing != 0 || !sound) failures++;
    }

    /* 2b. the ring strategy on memory that cannot be registered.
     * A read-only mapping cannot be page-locked by cudaHostRegister. */
    {
        const size_t bytes = ((n * sizeof(double)) + 4095) & ~(size_t)4095;
        void *ro = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ro == MAP_FAILED) {
            printf("  (could not map a read-only buffer; 2b not exercised)\n");
        } else {
            memcpy(ro, h, n * sizeof(double));
            mprotect(ro, bytes, PROT_READ);
            memset(b, 0, n * sizeof *b);
            /* DIRECT and MAPPED require registrable pages; RING is the
             * asynchronous strategy for this case. */
            const int rrc = matern_covariance_cuda(b, (const double *)ro, nu, scale, n,
                                                   MATERN_CUDA_ASYNC_RING);
            size_t differing = 0;
            for (size_t i = 0; i < n; i++) {
                unsigned long long x, y;
                memcpy(&x, &a[i], 8); memcpy(&y, &b[i], 8);
                if (x != y) differing++;
            }
            printf("  %-46s %s\n", "ring works on unregisterable memory",
                   (rrc == MATERN_OK && differing == 0) ? "PASS" : "FAIL");
            if (rrc != MATERN_OK || differing != 0) failures++;

            munmap(ro, bytes);
        }
    }

    /* 3. the backend is usable after release */
    matern_cuda_release();
    const int rc3 = matern_covariance_cuda(a, h, nu, scale, n, MATERN_CUDA_ASYNC);
    printf("  %-46s %s\n", "backend still works after release",
           rc3 == MATERN_OK ? "PASS" : "FAIL -- release poisoned the backend");
    if (rc3 != MATERN_OK) failures++;

    free(h); free(nu); free(scale); free(a); free(b);
    printf("\n");
    if (failures) { printf("cuda-modes: %d FAILED\n", failures); return 1; }
    printf("cuda-modes: the modes differ in transfer strategy and nothing else.\n");
    return 0;
}
