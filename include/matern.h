/* matern.h -- libmatern public API: the normalised Matern covariance in double precision.
 *
 * For a separation h >= 0, an order nu > 0 and a scale parameter s > 0,
 *
 *                 2^(1-nu)                                  sqrt(2 nu) h
 *      M(h)  =  ------------ * z^nu * K_nu(z) ,      z  =  --------------
 *                 Gamma(nu)                                      s
 *
 * where K_nu is the modified Bessel function of the second kind.  M is the normalised
 * covariance, M_nu(z) in the accompanying paper: M(0) = 1, and a process with marginal
 * variance sigma^2 has the covariance sigma^2 M(h).
 *
 * USAGE
 *      matern_workspace *work;
 *      matern_workspace_create(&work, 0);              // once
 *
 *      double nu[column_count], scale[column_count];   // one value per column,
 *                                                      // filled even when constant
 *
 *      for (int row = 0; row < row_count; row++)
 *          matern_covariance(out[row], h[row], nu, scale, column_count, work);
 *
 *      matern_workspace_destroy(work);
 *
 * With `work = NULL` a workspace is created on first use and cached per thread; a
 * thread that does so calls matern_release_thread_workspace() before it ends.  The
 * workspace holds the routing scratch across calls, so an evaluation does not
 * allocate.  scalar, scalar_portable, matched and matched_fma use no workspace and
 * allocate nothing.
 *
 * DOMAIN AND VALIDITY
 * Valid input is h >= 0 (+infinity included) and finite nu > 0 and s > 0.  Anything
 * else (nu <= 0, s <= 0, nu or s = +infinity, h < 0, or a NaN anywhere) returns NaN.
 * h = +infinity returns 0; h = 0 returns exactly 1.  Validity is tested on (h, nu, s),
 * not on the derived z, because z is not injective in the signs of h and s.
 * The properties under ACCURACY hold for 1e-3 <= nu <= 1e3 and
 * 1e-3 <= z <= 1e5 (the contract domain), with sqrt(2 nu) h / s representable;
 * outside that domain results are not guaranteed.  make check samples that range of
 * nu with z <= 700, and points past the exponential limit.  Known classes: nu < 1e-3,
 * nu > DBL_MAX/2, subnormal z, very large nu and very small z on the scalar backends,
 * h and s near the ends of the double range (where the derived z loses bits, underflows
 * or overflows); there a valid input can return an inaccurate value, 0 or 1.
 *
 * ACCURACY
 * In the contract domain every backend satisfies:
 *      M(0) = 1 exactly, at every order
 *      0 <= M <= 1 and finite for every valid input; M = 0 only for h = +infinity
 *        or where M is below the double range
 *      non-increasing in h up to rounding: between arguments a few ulp apart
 *        the computed M can rise in its last bits
 *      invalid input returns NaN
 *      a nonzero value past the exponential limit z = 708.4 wherever M >= DBL_MIN
 * No error bound is guaranteed.  Accuracy is evaluated against arbitrary-precision
 * references by the benchmark and in the accompanying paper.  Results can differ in
 * the last bits with the composition of a batch (THREAD SAFETY) and, on avx2,
 * between processor vendors (BACKENDS).
 *
 * BACKENDS
 * The backend is resolved once, at the first call of matern_covariance(),
 * matern_covariance_z() or matern_backend_name(): avx512 when the CPU has AVX-512
 * F + DQ + FMA, else avx2 (AVX2 + FMA), else matched_fma (FMA3 + AVX), else matched.
 * MATERN_BACKEND = avx512 | avx2 | matched_fma | matched | scalar | scalar_portable
 * forces one, for testing; it is read at that first call only and matched exactly in
 * lower case.  An unknown name, or a backend the CPU cannot run, gives the automatic
 * choice without a message.  The CUDA entries are separate functions, not a value of
 * MATERN_BACKEND.
 *      scalar       direct implementation: plain doubles, libm, no double-double
 *                   arithmetic or argument corrections; never selected
 *                   automatically.  `scalar` runs the best build the CPU supports,
 *                   `scalar_portable` the x86-64 baseline build.
 *      matched      the vector kernels at width 1 for the x86-64 baseline, with
 *                   libm's elementary functions, evaluated point by point except
 *                   the continued fraction (four points at a time).
 *      matched_fma  the same source compiled with -mfma.
 *      avx2         the vectorised kernels and elementary functions, 4 lanes.
 *      avx512       the same at 8 lanes.
 * A build with FMA=yes compiles the x86-64 baseline builds (matched, scalar_portable)
 * with -mfma, which changes their results and requires FMA3 on every machine
 * (docs/benchmark.md, "Make parameters").  scalar and the CUDA entries route
 * differently from the other CPU backends, and the results of all backends differ;
 * the errors of scalar are larger than those of the other backends.  avx2 results
 * can differ in the last bits between processor vendors: its gamma function uses the
 * single-precision reciprocal estimate VRCPPS, whose result the x86 instruction set
 * leaves to the implementation.
 *
 * THREAD SAFETY
 * The backend choice is published atomically at the first call; the CPU library
 * shares no other mutable state between threads.  A workspace must not be used by
 * two threads at the same time (concurrent use is undefined behaviour): give each
 * thread its own, or pass NULL so that each thread caches its own.
 * Each call is deterministic, but the composition of a batch can change the last
 * bits: on avx512 and avx2 the same point evaluated alone and inside a batch can
 * differ, because the router tiles and sorts, some kernels stop on a test across a
 * whole vector, and whether a half-integer order takes the closed form depends on the
 * runs in the batch (see the CUDA entries below); on matched_fma and matched only a
 * continued-fraction point can, since those run four at a time.  scalar routes each
 * point on its own and is exactly batch-invariant.  Consequently a covariance matrix is
 * not exactly symmetric on those backends: compare the two triangles with a tolerance,
 * or compute one and mirror it.
 *
 * RESOURCES
 * The CPU library has no global constructors, atexit handlers, signal handlers or
 * I/O, allocates only in matern_workspace_create and the per-thread cache, and
 * depends on libm only.  The CUDA entries (build/cuda.o) need the CUDA runtime, OpenMP
 * and the C++ standard library.  Within each call, depending on the transfer mode, the
 * entries on host arrays allocate device memory, page-lock host memory and start OpenMP
 * threads; matern_covariance_cuda_device() and _device_z() do none of these.
 */
#ifndef MATERN_H
#define MATERN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The release this header belongs to. */
#define MATERN_VERSION_MAJOR  0
#define MATERN_VERSION_MINOR  9
#define MATERN_VERSION_PATCH  0
#define MATERN_VERSION_STRING "0.9.0"

/* The release of the linked library (the MATERN_VERSION_STRING it is built with);
 * comparing the two detects a header that does not belong to the library. */
const char *matern_version(void);

/* Status codes.  MATERN_OK: success.  MATERN_INVALID_ARGUMENT also covers
 * tile > INT_MAX and an unknown CUDA mode.  MATERN_OUT_OF_MEMORY: an allocation
 * failed (on the CUDA entries, also a device operation).  MATERN_UNSUPPORTED_CPU is
 * also returned by the CUDA entries when no device is present. */
enum {
    MATERN_OK               =  0,
    MATERN_INVALID_ARGUMENT = -1,   /* a null pointer, or an impossible tile   */
    MATERN_OUT_OF_MEMORY    = -2,
    MATERN_UNSUPPORTED_CPU  = -3    /* no backend can run on this machine      */
};

/* Invalid mathematical input is not a status code: it produces NaN at exactly the
 * offending points and leaves the rest of the batch alone. */

/* Opaque routing scratch for the CPU entries (see THREAD SAFETY). */
typedef struct matern_workspace matern_workspace;

/* Default elements per tile; a workspace holds 69 bytes per tile element. */
#define MATERN_TILE_DEFAULT 8192

/* Creates a workspace.  `tile = 0` selects MATERN_TILE_DEFAULT; tile > INT_MAX
 * returns MATERN_INVALID_ARGUMENT. */
int  matern_workspace_create(matern_workspace **workspace, size_t tile);
/* Frees a workspace; NULL is a no-op. */
void matern_workspace_destroy(matern_workspace *workspace);

/* Releases the workspace this thread cached by calling with `workspace = NULL`.
 * C has no portable thread-exit hook, so every such thread calls this before it
 * ends, or its workspace leaks; the main thread can leave it to process exit. */
void matern_release_thread_workspace(void);

/* M(h) for n points.  `out` must not overlap any input.  n = 0 is a no-op; a null
 * `out`, `h`, `nu` or `scale` returns MATERN_INVALID_ARGUMENT. */
int matern_covariance(double *out,
                      const double *h, const double *nu, const double *scale,
                      size_t n, matern_workspace *workspace);

/* M for n points, given z directly; there is no `scale` argument.  Passing z avoids
 * the second rounding of reconstructing h = z / sqrt(2 nu) and re-deriving z, which
 * matters because M behaves like e^-z.  Validity is tested on (z, nu): nu <= 0,
 * nu = +infinity, z < 0 and NaN return NaN; z = 0 returns exactly 1; z = +infinity
 * returns 0.  Same argument checks, backends and kernels as matern_covariance(). */
int matern_covariance_z(double *out,
                        const double *z, const double *nu,
                        size_t n, matern_workspace *workspace);

/* The backend resolved for this process: "avx512", "avx2", "matched_fma",
 * "matched", "scalar" or "scalar_portable"; the last two only on request through
 * MATERN_BACKEND (see BACKENDS). */
const char *matern_backend_name(void);

/* CUDA entries, separate from matern_covariance().  The device code is the object
 * build/cuda.o (make cuda); a library built without it defines every entry below as
 * a weak stand-in with the same argument checks (matern_cuda_available() 0,
 * matern_cuda_device_name() "", evaluation entries MATERN_UNSUPPORTED_CPU).  Linking
 * build/cuda.o replaces them and adds the CUDA runtime, OpenMP and the C++ standard
 * library (README.md, "CUDA").  Inside a CUDA kernel, include/matern_device.cuh gives
 * the per-point evaluation of the default build as __device__ functions, without this
 * library.
 *
 * Transfer modes, which differ only in the transfer strategy.  The ASYNC modes
 * release every page lock before returning and own nothing across calls; pin the
 * buffers (cudaHostRegister) beforehand to avoid the per-call registration. */
enum {
    /* Copy in, run, copy back. */
    MATERN_CUDA_SYNC         = 0,

    /* Page-locks the caller's arrays for the call and transfers from them, no staging;
     * MAPPED falls back to this.  An already-locked or unlockable buffer is not an
     * error. */
    MATERN_CUDA_ASYNC_DIRECT = 1,

    /* Host copy through a pinned staging ring; never touches the caller's pages, so
     * it works where registration cannot (a read-only mapping, or pages shared
     * with data the library does not own). */
    MATERN_CUDA_ASYNC_RING   = 2,

    /* Zero-copy: the kernel reads and writes the caller's mapped pages directly.
     * Without canMapHostMemory the call falls back to DIRECT; if no device pointer
     * can be obtained for the buffers (for example a buffer registered without
     * mapping) it returns MATERN_OUT_OF_MEMORY. */
    MATERN_CUDA_ASYNC_MAPPED = 3
};

/* Alias of MATERN_CUDA_ASYNC_DIRECT. */
#define MATERN_CUDA_ASYNC MATERN_CUDA_ASYNC_DIRECT

/* Tuning options.  Zero in any field means "the library decides", which is what the
 * plain entries pass. */
typedef struct {
    int block;              /* CUDA threads per block; 0 -> 256                  */
    int streams;            /* concurrent streams; 0 -> 4                        */
    int chunks_per_stream;  /* pipeline depth per stream; 0 -> size-adaptive.  RING:
                               pinned footprint 32n/chunks_per_stream bytes (24n for
                               the _z entries) */
    int host_threads;       /* RING: threads for the staging memcpy; 0 -> 4        */
} matern_cuda_options;

/* Nonzero when a CUDA device is usable; without one the evaluation entries return
 * MATERN_UNSUPPORTED_CPU. */
int         matern_cuda_available(void);
/* The device's name, or "" without a device. */
const char *matern_cuda_device_name(void);

/* M(h) and M_nu(z) on the device, with the interface and input rules of the CPU
 * entries.  Unlike avx512, avx2, matched_fma and matched, the routing sends the Levin
 * band to the continued fraction.  As on matched and matched_fma, the half-integer
 * closed form is selected per point.  avx512 and avx2 select it for a run of constant
 * half-integer order, or a mixed half-integer segment, of at least 256 points
 * (HALF_INTEGER_MIN_RUN) or reaching the end of the batch.  The scan runs from the start
 * of the batch; from the first run of fewer than 256 points that the closed form does
 * not take, the rest of the batch goes through the router. */
int matern_covariance_cuda(double *out, const double *h, const double *nu,
                           const double *scale, size_t n, int mode);
int matern_covariance_cuda_z(double *out, const double *z, const double *nu,
                             size_t n, int mode);

/* The same two entries with tuning options; `options` may be NULL. */
int matern_covariance_cuda_tuned(double *out, const double *h, const double *nu,
                                 const double *scale, size_t n, int mode,
                                 const matern_cuda_options *options);
int matern_covariance_cuda_z_tuned(double *out, const double *z, const double *nu,
                                   size_t n, int mode,
                                   const matern_cuda_options *options);

/* M(h) and M_nu(z) for arrays in device-accessible memory (cudaMalloc, managed, or
 * mapped host memory), launched on `stream` (a cudaStream_t; NULL is the default
 * stream).  The call returns after the launch: it allocates, copies and synchronises
 * nothing, so `out` is ready once the stream has run, and the input arrays must stay
 * valid until then.  `out` must not overlap an input.  Same per-point results and
 * input rules as the entries above.  A launch failure, or a CUDA error left pending by
 * earlier calls, returns MATERN_OUT_OF_MEMORY, or MATERN_UNSUPPORTED_CPU without a
 * usable device. */
int matern_covariance_cuda_device(double *out, const double *h, const double *nu,
                                  const double *scale, size_t n, void *stream);
int matern_covariance_cuda_device_z(double *out, const double *z, const double *nu,
                                    size_t n, void *stream);

/* Clears the cached device probe; a later call probes again.  It does not reset the
 * device. */
void matern_cuda_release(void);

#ifdef __cplusplus
}
#endif

#endif /* MATERN_H */
