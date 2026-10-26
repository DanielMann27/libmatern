/* cuda.cu -- CUDA backend (see the accompanying paper, GPU implementation and divergence
 * strategy): the matern_covariance_cuda* entries (modes SYNC, ASYNC_DIRECT, ASYNC_RING and
 * ASYNC_MAPPED) and the device-pointer entries, one thread per point (matern_device.cuh). */
#include <cuda_runtime.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <mutex>

#include "../../include/matern.h"
#include "../../include/matern_device.cuh"

/* Routing for the (h, nu, s) and (z, nu) entries; override both together.  Levin variant:
 *   -DMATERN_CUDA_FLAGS=MATERN_ROUTE_FLAGS_DEFAULT_LEVIN
 *   -DMATERN_CUDA_FLAGS_Z=MATERN_ROUTE_FLAGS_DEFAULT_LEVIN_Z */
#ifndef MATERN_CUDA_FLAGS
#define MATERN_CUDA_FLAGS   MATERN_ROUTE_FLAGS_DEFAULT
#endif
#ifndef MATERN_CUDA_FLAGS_Z
#define MATERN_CUDA_FLAGS_Z MATERN_ROUTE_FLAGS_DEFAULT_Z
#endif

#ifndef MATERN_CUDA_BLOCK
#define MATERN_CUDA_BLOCK 256
#endif

/* Streams for the async paths; the pipeline depth is streams * chunks per stream. */
#ifndef MATERN_CUDA_STREAMS
#define MATERN_CUDA_STREAMS 4
#endif

/* Allocation flags for the ring's input staging; write-combined memory is not used. */
#define MATERN_CUDA_STAGING_FLAGS cudaHostAllocDefault

/* Internal sentinel: the device cannot map host memory, so another strategy is used.
 * Never returned to a caller. */
#define MATERN_CUDA_MAPPED_UNAVAILABLE (-999)

/* Default chunks per stream: DIRECT on a chunk count, the ring on a target footprint. */
static int matern_cuda_chunks_per_stream(size_t n)
{
    return ((double)n * (double)sizeof(double) >= 100.0e6) ? 16 : 4;
}

/* Target footprint of the pinned ring in bytes, small so that the staging destinations stay
 * cache-resident; the footprint, streams * arrays * chunk, is 32n/cps bytes independent of
 * the stream count. */
#define MATERN_CUDA_RING_TARGET_BYTES 2.0e6

static int matern_cuda_ring_chunks_per_stream(size_t n)
{
    double want = 32.0 * (double)n / MATERN_CUDA_RING_TARGET_BYTES;
    if (want < 4.0)  want = 4.0;
    if (want > 64.0) want = 64.0;
    return (int)want;
}

/* An option of zero selects the default resolved here. */
typedef struct { int block, streams, cps, host_threads; } matern_cuda_resolved;

static matern_cuda_resolved matern_cuda_resolve(const matern_cuda_options *o, size_t n,
                                                int mode)
{
    matern_cuda_resolved r;
    r.block        = (o && o->block   > 0) ? o->block   : MATERN_CUDA_BLOCK;
    r.streams      = (o && o->streams > 0) ? o->streams : MATERN_CUDA_STREAMS;
    r.cps          = (o && o->chunks_per_stream > 0)
                   ? o->chunks_per_stream
                   : (mode == MATERN_CUDA_ASYNC_RING ? matern_cuda_ring_chunks_per_stream(n)
                                                     : matern_cuda_chunks_per_stream(n));
    /* host threads for the ring's staging memcpy */
    r.host_threads = (o && o->host_threads > 0) ? o->host_threads : 4;
    if (r.streams > 32) r.streams = 32;
    return r;
}

/* One point per thread; `base` is the offset of the launch's chunk. */
static __global__ void matern_cuda_kernel(const double *h, const double *nu,
                                          const double *scale, double *out, long n, long base)
{
    long i = base + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = matern_route_device<MATERN_CUDA_FLAGS>(h[i], nu[i], scale[i]);
}

/* The direct-argument kernel: under MATERN_ROUTE_DIRECT_ARGUMENT_BIT the first argument is z,
 * with no scale. */
static __global__ void matern_cuda_kernel_z(const double *z, const double *nu,
                                            double *out, long n, long base)
{
    long i = base + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = matern_route_device<MATERN_CUDA_FLAGS_Z>(z[i], nu[i], 1.0);
}

/* matern_cuda_launch() is the only way to start the kernels (they are static).
 * MATERN_CUDA_BENCHMARK_HOOKS (build/cuda_bench.o, which the benchmark drivers link) adds
 * matern_cuda_set_kernel_override() (a substitute kernel with the launcher's signature;
 * NULL selects this library's kernel), matern_cuda_launch_external() and
 * matern_cuda_route_flags(); they are extern "C" for the benchmark drivers and absent from
 * build/cuda.o, the object a program links. */
#ifdef MATERN_CUDA_BENCHMARK_HOOKS
typedef void (*matern_cuda_kernel_fn)(int direct, const double *a, const double *b,
                                      const double *c, double *out,
                                      long n, long base,
                                      long blocks, int threads, cudaStream_t stream);
static matern_cuda_kernel_fn matern_cuda_kernel_override = NULL;

extern "C" void matern_cuda_set_kernel_override(void *fn)
{ matern_cuda_kernel_override = (matern_cuda_kernel_fn)fn; }
#endif

static inline void matern_cuda_launch(int direct,
                                      const double *a, const double *b,
                                      const double *c, double *out,
                                      long n, long base,
                                      long blocks, int threads,
                                      cudaStream_t stream)
{
#ifdef MATERN_CUDA_BENCHMARK_HOOKS
    if (matern_cuda_kernel_override) {
        matern_cuda_kernel_override(direct, a, b, c, out, n, base,
                                    blocks, threads, stream);
        return;
    }
#endif
    if (direct) matern_cuda_kernel_z<<<blocks, threads, 0, stream>>>(a, b, out, n, base);
    else        matern_cuda_kernel  <<<blocks, threads, 0, stream>>>(a, b, c, out, n, base);
}

#ifdef MATERN_CUDA_BENCHMARK_HOOKS
/* The launcher for kernel-only timing, without the entry's transfers. */
extern "C" void matern_cuda_launch_external(int direct,
                                            const double *a, const double *b,
                                            const double *c, double *out,
                                            long n, long base,
                                            long blocks, int threads)
{
    matern_cuda_launch(direct, a, b, c, out, n, base, blocks, threads, 0);
}
#endif

static int  matern_cuda_probed = 0;
static int  matern_cuda_ok     = 0;
static char matern_cuda_name[256];
static std::mutex matern_cuda_probe_lock;   /* serialises probe and release */

static void matern_cuda_probe(void)
{
    std::lock_guard<std::mutex> guard(matern_cuda_probe_lock);
    if (matern_cuda_probed) return;
    matern_cuda_probed = 1;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        cudaGetLastError();
        matern_cuda_ok = 0;
        return;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { cudaGetLastError(); matern_cuda_ok = 0; return; }
    strncpy(matern_cuda_name, prop.name, sizeof matern_cuda_name - 1);
    matern_cuda_ok = 1;
}

extern "C" int matern_cuda_available(void)
{
    matern_cuda_probe();
    return matern_cuda_ok;
}

extern "C" const char *matern_cuda_device_name(void)
{
    matern_cuda_probe();
    return matern_cuda_ok ? matern_cuda_name : "";
}

#ifdef MATERN_CUDA_BENCHMARK_HOOKS
/* The compiled-in routing; no device probe, so it answers without a GPU.  The benchmark
 * reads the Levin band as bit 32, MATERN_ROUTE_LEVIN_BIT. */
extern "C" int matern_cuda_route_flags(void)
{
    static_assert(MATERN_ROUTE_LEVIN_BIT == 32, "benchmark/benchmark_gpu.cu reads the Levin band as bit 32");
    return MATERN_CUDA_FLAGS;
}
#endif

/* Per-call registration: `owned` records whether this call locked the page; a range the
 * caller already registered is not an error and is left as found. */
struct matern_cuda_pin {
    void  *p[4];
    int    owned[4];
    int    n;
};

static void matern_cuda_pin_set(matern_cuda_pin *pin, void **ps, int count, size_t nb)
{
    pin->n = count;
    for (int i = 0; i < count; i++) {
        pin->p[i] = ps[i];
        cudaError_t e = cudaHostRegister(ps[i], nb, cudaHostRegisterDefault);
        if (e == cudaSuccess) {
            pin->owned[i] = 1;
        } else {
            /* already locked, or refused: the copy stays correct through the driver's
             * bounce buffer */
            cudaGetLastError();
            pin->owned[i] = 0;
        }
    }
}

static void matern_cuda_unpin_set(matern_cuda_pin *pin)
{
    for (int i = 0; i < pin->n; i++)
        if (pin->owned[i]) { cudaHostUnregister(pin->p[i]); cudaGetLastError(); }
    pin->n = 0;
}

/* Pinned staging ring: chunks are copied through page-locked slots and the caller's pages
 * are never registered, so it serves memory that cannot be registered (read-only mappings,
 * .rodata, pages shared with other data). */
static int matern_cuda_run_ring(double *out, const double *a, const double *b,
                                const double *c, size_t n, int direct,
                                matern_cuda_resolved R,
                                double *ad, double *bd, double *cd, double *od)
{
    const int arrays = direct ? 3 : 4;          /* a, b, (c), out */
    long total = (long)R.streams * (long)R.cps;
    if (total < 1) total = 1;
    size_t chunk = (n + (size_t)total - 1) / (size_t)total;
    if (chunk < 1) chunk = 1;

    cudaStream_t st[32];
    double *stage[32];      /* inputs:  one slice of a single allocation */
    double *sres[32];       /* results: one slice of a single allocation */
    int made = 0;
    for (; made < R.streams; made++)
        if (cudaStreamCreateWithFlags(&st[made], cudaStreamNonBlocking) != cudaSuccess) break;
    if (made == 0) { cudaGetLastError(); return MATERN_OUT_OF_MEMORY; }

    /* one allocation for all input slots (MATERN_CUDA_STAGING_FLAGS), one for all results
     * (cudaHostAllocDefault, since the host reads them back) */
    double *inputs = NULL, *results = NULL;
    if (cudaHostAlloc((void **)&inputs, (size_t)made * (arrays - 1) * chunk * sizeof(double),
                      MATERN_CUDA_STAGING_FLAGS) != cudaSuccess ||
        cudaHostAlloc((void **)&results, (size_t)made * chunk * sizeof(double),
                      cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        if (inputs) cudaFreeHost(inputs);
        for (int i = 0; i < made; i++) cudaStreamDestroy(st[i]);
        return MATERN_OUT_OF_MEMORY;
    }
    for (int i = 0; i < made; i++) {
        stage[i] = inputs  + (size_t)i * (arrays - 1) * chunk;
        sres[i]  = results + (size_t)i * chunk;
    }

    /* what each slot currently holds, so its result can be harvested before reuse */
    long soff[32]; size_t slen[32];
    for (int i = 0; i < made; i++) { soff[i] = -1; slen[i] = 0; }

    for (long k = 0; ; k++) {
        const size_t off = (size_t)k * chunk;
        if (off >= n) break;
        const size_t len = (off + chunk > n) ? (n - off) : chunk;
        const int    s   = (int)(k % made);

        /* back-pressure: this slot's previous chunk must land before it is overwritten */
        if (soff[s] >= 0) {
            cudaStreamSynchronize(st[s]);
            memcpy(out + soff[s], sres[s], slen[s] * sizeof(double));
        }

        double *sa = stage[s];
        double *sb = stage[s] + chunk;
        double *sc = direct ? NULL : stage[s] + 2 * chunk;
#ifdef _OPENMP
        #pragma omp parallel num_threads(R.host_threads)
        {
            #pragma omp sections
            {
                #pragma omp section
                memcpy(sa, a + off, len * sizeof(double));
                #pragma omp section
                memcpy(sb, b + off, len * sizeof(double));
                #pragma omp section
                { if (!direct) memcpy(sc, c + off, len * sizeof(double)); }
            }
        }
#else
        memcpy(sa, a + off, len * sizeof(double));
        memcpy(sb, b + off, len * sizeof(double));
        if (!direct) memcpy(sc, c + off, len * sizeof(double));
#endif
        const size_t lb = len * sizeof(double);
        cudaMemcpyAsync(ad + off, sa, lb, cudaMemcpyHostToDevice, st[s]);
        cudaMemcpyAsync(bd + off, sb, lb, cudaMemcpyHostToDevice, st[s]);
        if (!direct) cudaMemcpyAsync(cd + off, sc, lb, cudaMemcpyHostToDevice, st[s]);
        const long blocks = (long)((len + R.block - 1) / R.block);
        matern_cuda_launch(direct, ad, bd, cd, od, (long)(off + len), (long)off,
                           blocks, R.block, st[s]);
        cudaMemcpyAsync(sres[s], od + off, lb, cudaMemcpyDeviceToHost, st[s]);
        soff[s] = (long)off; slen[s] = len;
    }

    for (int i = 0; i < made; i++) {
        cudaStreamSynchronize(st[i]);
        if (soff[i] >= 0)
            memcpy(out + soff[i], sres[i], slen[i] * sizeof(double));
        cudaStreamDestroy(st[i]);
    }
    cudaFreeHost(inputs);
    cudaFreeHost(results);
    return (cudaGetLastError() == cudaSuccess) ? MATERN_OK : MATERN_OUT_OF_MEMORY;
}

/* Zero-copy: the kernel accesses the caller's pages through mapped device aliases.  It needs
 * registration as DIRECT does; a device without canMapHostMemory returns
 * MATERN_CUDA_MAPPED_UNAVAILABLE and the caller falls back to DIRECT. */
static int matern_cuda_run_mapped(double *out, const double *a, const double *b,
                                  const double *c, size_t n, int direct,
                                  matern_cuda_resolved R)
{
    int can_map = 0, dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, dev) != cudaSuccess
        || !can_map) {
        cudaGetLastError();
        return MATERN_CUDA_MAPPED_UNAVAILABLE;
    }

    const size_t nb = n * sizeof(double);
    void  *ps[4]; int np = 0;
    ps[np++] = (void *)(uintptr_t)a;
    ps[np++] = (void *)(uintptr_t)b;
    if (!direct) ps[np++] = (void *)(uintptr_t)c;
    ps[np++] = (void *)out;

    int owned[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < np; i++) {
        const cudaError_t e = cudaHostRegister(ps[i], nb,
                                               cudaHostRegisterMapped);
        if (e == cudaSuccess) owned[i] = 1;
        else {
            cudaGetLastError();
            /* an existing registration is usable only if mapped, which cannot be
             * queried; a missing device pointer fails below */
        }
    }

    double *da = NULL, *db = NULL, *dc = NULL, *dout = NULL;
    int rc = MATERN_OK;
    if (cudaHostGetDevicePointer((void **)&da, ps[0], 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void **)&db, ps[1], 0) != cudaSuccess ||
        (!direct && cudaHostGetDevicePointer((void **)&dc, ps[2], 0) != cudaSuccess) ||
        cudaHostGetDevicePointer((void **)&dout, ps[np - 1], 0) != cudaSuccess) {
        cudaGetLastError();
        rc = MATERN_OUT_OF_MEMORY;
        goto unpin;
    }

    {
        const long blocks = (long)((n + R.block - 1) / R.block);
        matern_cuda_launch(direct, da, db, dc, dout, (long)n, 0, blocks, R.block, 0);
        if (cudaDeviceSynchronize() != cudaSuccess) rc = MATERN_OUT_OF_MEMORY;
    }

unpin:
    for (int i = 0; i < np; i++)
        if (owned[i]) { cudaHostUnregister(ps[i]); cudaGetLastError(); }
    cudaGetLastError();
    return rc;
}

static int matern_cuda_run(double *out, const double *a, const double *b,
                           const double *c, size_t n, int mode, int direct,
                           const matern_cuda_options *options)
{
    const matern_cuda_resolved R = matern_cuda_resolve(options, n, mode);
    const size_t nb = n * sizeof(double);
    double *ad = NULL, *bd = NULL, *cd = NULL, *od = NULL;
    int rc = MATERN_OK;

    /* zero-copy allocates no device memory, so it runs before the cudaMallocs */
    if (mode == MATERN_CUDA_ASYNC_MAPPED) {
        const int mrc = matern_cuda_run_mapped(out, a, b, c, n, direct, R);
        if (mrc != MATERN_CUDA_MAPPED_UNAVAILABLE) return mrc;
        mode = MATERN_CUDA_ASYNC_DIRECT;
    }

    if (cudaMalloc((void **)&ad, nb) != cudaSuccess) { rc = MATERN_OUT_OF_MEMORY; goto done; }
    if (cudaMalloc((void **)&bd, nb) != cudaSuccess) { rc = MATERN_OUT_OF_MEMORY; goto done; }
    if (!direct && cudaMalloc((void **)&cd, nb) != cudaSuccess) { rc = MATERN_OUT_OF_MEMORY; goto done; }
    if (cudaMalloc((void **)&od, nb) != cudaSuccess) { rc = MATERN_OUT_OF_MEMORY; goto done; }

    if (mode == MATERN_CUDA_SYNC) {
        if (cudaMemcpy(ad, a, nb, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(bd, b, nb, cudaMemcpyHostToDevice) != cudaSuccess ||
            (!direct && cudaMemcpy(cd, c, nb, cudaMemcpyHostToDevice) != cudaSuccess)) {
            rc = MATERN_OUT_OF_MEMORY; goto done;
        }
        const long blocks = (long)((n + R.block - 1) / R.block);
        matern_cuda_launch(direct, ad, bd, cd, od, (long)n, 0, blocks, R.block, 0);
        if (cudaDeviceSynchronize() != cudaSuccess) { rc = MATERN_OUT_OF_MEMORY; goto done; }
        if (cudaMemcpy(out, od, nb, cudaMemcpyDeviceToHost) != cudaSuccess) rc = MATERN_OUT_OF_MEMORY;
        goto done;
    }

    if (mode == MATERN_CUDA_ASYNC_RING) {
        rc = matern_cuda_run_ring(out, a, b, c, n, direct, R, ad, bd, cd, od);
        goto done;
    }

    /* async DIRECT: pin, overlap, unpin, on every call */
    {
        matern_cuda_pin pin;
        void *ps[4]; int np = 0;
        ps[np++] = (void *)(uintptr_t)a;
        ps[np++] = (void *)(uintptr_t)b;
        if (!direct) ps[np++] = (void *)(uintptr_t)c;
        ps[np++] = (void *)out;
        matern_cuda_pin_set(&pin, ps, np, nb);

        cudaStream_t st[32];
        int made = 0;
        for (; made < R.streams; made++)
            if (cudaStreamCreateWithFlags(&st[made], cudaStreamNonBlocking) != cudaSuccess) break;
        if (made == 0) { cudaGetLastError(); matern_cuda_unpin_set(&pin); rc = MATERN_OUT_OF_MEMORY; goto done; }

        /* streams * cps chunks, dealt round-robin across the streams */
        const long   total  = (long)made * (long)R.cps;
        const size_t chunk  = (n + (size_t)total - 1) / (size_t)total;
        for (long k = 0; k < total; k++) {
            const size_t off = (size_t)k * chunk;
            if (off >= n) break;
            const size_t len = (off + chunk > n) ? (n - off) : chunk;
            const size_t lb  = len * sizeof(double);
            cudaStream_t s = st[k % made];
            cudaMemcpyAsync(ad + off, a + off, lb, cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(bd + off, b + off, lb, cudaMemcpyHostToDevice, s);
            if (!direct) cudaMemcpyAsync(cd + off, c + off, lb, cudaMemcpyHostToDevice, s);
            const long blocks = (long)((len + R.block - 1) / R.block);
            matern_cuda_launch(direct, ad, bd, cd, od, (long)(off + len),
                               (long)off, blocks, R.block, s);
            cudaMemcpyAsync(out + off, od + off, lb, cudaMemcpyDeviceToHost, s);
        }
        for (int s = 0; s < made; s++) { cudaStreamSynchronize(st[s]); cudaStreamDestroy(st[s]); }
        matern_cuda_unpin_set(&pin);          /* always; nothing is kept */
        if (cudaGetLastError() != cudaSuccess) rc = MATERN_OUT_OF_MEMORY;
    }

done:
    if (ad) cudaFree(ad);
    if (bd) cudaFree(bd);
    if (cd) cudaFree(cd);
    if (od) cudaFree(od);
    cudaGetLastError();
    return rc;
}

static int matern_cuda_mode_ok(int mode)
{
    return mode == MATERN_CUDA_SYNC
        || mode == MATERN_CUDA_ASYNC_DIRECT
        || mode == MATERN_CUDA_ASYNC_RING
        || mode == MATERN_CUDA_ASYNC_MAPPED;
}

extern "C" int matern_covariance_cuda_tuned(double *out, const double *h, const double *nu,
                                            const double *scale, size_t n, int mode,
                                            const matern_cuda_options *options)
{
    if (!matern_cuda_mode_ok(mode)) return MATERN_INVALID_ARGUMENT;
    if (n == 0) return MATERN_OK;
    if (!out || !h || !nu || !scale) return MATERN_INVALID_ARGUMENT;
    if (!matern_cuda_available()) return MATERN_UNSUPPORTED_CPU;
    return matern_cuda_run(out, h, nu, scale, n, mode, 0, options);
}

extern "C" int matern_covariance_cuda_z_tuned(double *out, const double *z, const double *nu,
                                              size_t n, int mode,
                                              const matern_cuda_options *options)
{
    if (!matern_cuda_mode_ok(mode)) return MATERN_INVALID_ARGUMENT;
    if (n == 0) return MATERN_OK;
    if (!out || !z || !nu) return MATERN_INVALID_ARGUMENT;
    if (!matern_cuda_available()) return MATERN_UNSUPPORTED_CPU;
    return matern_cuda_run(out, z, nu, NULL, n, mode, 1, options);
}

extern "C" int matern_covariance_cuda(double *out, const double *h, const double *nu,
                                      const double *scale, size_t n, int mode)
{
    return matern_covariance_cuda_tuned(out, h, nu, scale, n, mode, NULL);
}

extern "C" int matern_covariance_cuda_z(double *out, const double *z, const double *nu,
                                        size_t n, int mode)
{
    return matern_covariance_cuda_z_tuned(out, z, nu, n, mode, NULL);
}

/* Device pointers, the caller's stream: no probe, allocation, copy or synchronisation.
 * One launch per 2^30 points keeps the grid within its limit. */
static int matern_cuda_device_run(double *out, const double *a, const double *b,
                                  const double *c, size_t n, int direct, void *stream)
{
    if (n > (size_t)LONG_MAX) return MATERN_INVALID_ARGUMENT;
    const long step = 1L << 30;
    for (long base = 0; base < (long)n; base += step) {
        const long m      = ((long)n - base < step) ? (long)n - base : step;
        const long blocks = (m + MATERN_CUDA_BLOCK - 1) / MATERN_CUDA_BLOCK;
        matern_cuda_launch(direct, a, b, c, out, base + m, base, blocks,
                           MATERN_CUDA_BLOCK, (cudaStream_t)stream);
    }
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return MATERN_OK;
    if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) return MATERN_UNSUPPORTED_CPU;
    return MATERN_OUT_OF_MEMORY;
}

extern "C" int matern_covariance_cuda_device(double *out, const double *h, const double *nu,
                                             const double *scale, size_t n, void *stream)
{
    if (n == 0) return MATERN_OK;
    if (!out || !h || !nu || !scale) return MATERN_INVALID_ARGUMENT;
    return matern_cuda_device_run(out, h, nu, scale, n, 0, stream);
}

extern "C" int matern_covariance_cuda_device_z(double *out, const double *z, const double *nu,
                                               size_t n, void *stream)
{
    if (n == 0) return MATERN_OK;
    if (!out || !z || !nu) return MATERN_INVALID_ARGUMENT;
    return matern_cuda_device_run(out, z, nu, NULL, n, 1, stream);
}

/* Clears the device probe, so the next call probes again.  It does not call
 * cudaDeviceReset(). */
extern "C" void matern_cuda_release(void)
{
    std::lock_guard<std::mutex> guard(matern_cuda_probe_lock);
    matern_cuda_probed = 0;
    matern_cuda_ok     = 0;
    cudaGetLastError();
}
