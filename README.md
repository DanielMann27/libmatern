# libmatern

The normalised Matérn covariance function in double precision, with vectorised backends for
AVX-512 and AVX2, a width-1 build of the same kernels for CPUs without AVX2, an independent
scalar implementation to test against, and a CUDA backend. The `(ν, z)` plane is partitioned
into regions, each evaluated by an approximation of `K_ν`, in most regions with the covariance
prefactor folded in; the formulations and the domain decomposition are described in the
accompanying paper.

```
             2^(1-ν)                             √(2ν)·h
    M(h)  =  ───────── · z^ν · K_ν(z) ,    z  =  ───────
             Γ(ν)                                   s
```

Here `h` is the separation (the distance between two points), `ν` the smoothness (order) and
`s` the scale parameter; `K_ν` is the modified Bessel function of the second kind. `M` is
the normalised covariance, `M_ν(z)` in the accompanying paper: `M(0) = 1`, and a process with
marginal variance `σ²` has the covariance `σ²·M(h)`. In the Whittle–Matérn parameterisation
`κ = √(2ν)/s` of the accompanying paper, `z = κ·h`. A parameterisation that writes the
argument as `h/φ`, without the factor `√(2ν)`, is this `M` with `s = √(2ν)·φ`, or
`matern_covariance_z()` with `z = h/φ`; at `ν = 1/2` the two coincide (`s = φ`) and
`M(h) = exp(-h/s)`.

## Quick start

```sh
make
make check
make install PREFIX=/usr/local     # DESTDIR=<dir> stages the files, e.g. for a package
```

A complete program:

```c
#include <matern.h>
#include <stdio.h>

int main(void)
{
    enum { N = 4 };
    double h[N] = { 0.0, 0.5, 1.0, 2.0 }, out[N];
    double nu[N], scale[N];

    /* nu and scale are PER-ELEMENT arrays, even when they are constant */
    for (int i = 0; i < N; i++) { nu[i] = 1.5; scale[i] = 1.0; }

    if (matern_covariance(out, h, nu, scale, N, NULL) != 0)
        return 1;

    printf("backend %s\n", matern_backend_name());
    for (int i = 0; i < N; i++)
        printf("  M(%.1f) = %.17g\n", h[i], out[i]);

    matern_release_thread_workspace();   /* frees this thread's cached workspace */
    return 0;
}
```

prints, on a machine with AVX-512,

```
backend avx512
  M(0.0) = 1
  M(0.5) = 0.78488765395745064
  M(1.0) = 0.48335772459650772
  M(2.0) = 0.13973135019231464
```

The first line names the backend, which the library selects from the CPU at the first call
([Backends](#backends)). The program is compiled with

```sh
PREFIX=/usr/local                  # the PREFIX given to make install
cc -O2 -I"$PREFIX/include" app.c -L"$PREFIX/lib" -lmatern -lm
```

## Building

libmatern builds and runs on x86-64 Linux; other architectures and operating systems are not
supported. The build needs a GCC-compatible C11 compiler (GCC, Clang or Intel icx), GNU make
and GNU binutils (`ld`, `objcopy`, `ar`, `nm`); the library itself needs only libm.
`make check` also needs Python 3, and the CUDA backend needs nvcc and, to run, an NVIDIA GPU.

```sh
make                 # build/libmatern.a
make shared          # build/libmatern.so.0 (soname libmatern.so.0) and the link build/libmatern.so
make example         # build/example, a program that builds a covariance matrix and factors it
make install         # headers and libraries into PREFIX (default /usr/local); DESTDIR=<dir> stages them
make cuda            # build/cuda.o (nvcc; GPU_ARCH=sm_80 by default)
make help            # the targets and a summary of the make parameters
```

`make install` also installs the shared library and the CUDA object when they have been
built. `CC` selects the compiler, `CFLAGS` replaces the default flags (`-std=c11 -Wall
-Wextra -fno-math-errno`) and `XCF` adds flags. The flags the results depend on, the
optimisation level (`OPTIMISE`, default `-O2`), `-ffp-contract=fast` and, for icx,
`-fp-model=precise`, are added outside `CFLAGS`, so `CFLAGS` cannot change them (`XCF`,
appended after them, can); contraction
is set explicitly because the compiler's default depends on the language dialect. An
environment that exports `CFLAGS`, as some compiler modules do, replaces the defaults;
`env -u CFLAGS make` restores them. `OPTIMISE=-O0` gives a build for debugging whose results
differ from the default build's. The default build (`FMA=no`) runs on any x86-64 CPU and
selects its backend at run time, so a calling program needs no `-march` flag. A build prints
one line per compiled file and per product; `V=yes` prints the commands. Switches take `yes`
or `no`. All make parameters are listed in
[`docs/benchmark.md`, "Make parameters"](docs/benchmark.md#make-parameters).

## Backends

The backend is selected at the first call from the CPU's capabilities, in the order
`avx512` → `avx2` → `matched_fma` → `matched`; `matern_backend_name()` reports the backend in
use. The environment variable `MATERN_BACKEND`, set to one of the names below in lower case,
selects that backend for the process; it is read once, at the first call, and any other
value, or a backend the CPU cannot run, leaves the automatic choice.

| backend | description |
|---|---|
| `avx512` | the vectorised kernels, 8 doubles per vector; needs AVX-512 (F and DQ) and FMA |
| `avx2` | the vectorised kernels, 4 doubles per vector; needs AVX2 and FMA |
| `matched_fma` | the vector kernels at width 1 with the FMA instruction and the C library's `exp`, `log`, `pow` and `tgamma`, evaluated point by point without a workspace, except continued-fraction points, which run four at a time; selected on a CPU with FMA and AVX but without AVX2 |
| `matched` | the same with only baseline x86-64 instructions (default build, `FMA=no`); the fallback |
| `scalar` | an independent direct implementation in plain double arithmetic that shares no region kernel with the others, for cross-checking; never selected automatically; `scalar_portable` selects its baseline x86-64 build |

The first four carry selected intermediate values, among them the `z` derived from `h` and
`s`, in double-double arithmetic (a value held as the unevaluated sum of two doubles). The
CUDA backend has separate entry points and is not a value of `MATERN_BACKEND` ([CUDA](#cuda)).

## Using it from your own project

`-lmatern` in the compile line of the Quick start links the shared library where it is
installed (`make shared` before `make install`), else `libmatern.a`; `"$PREFIX/lib/libmatern.a"`
in place of `-lmatern` links the static library. A program linked against the shared library
finds `libmatern.so.0` at run time through `-Wl,-rpath,"$PREFIX/lib"`, `LD_LIBRARY_PATH` or the
system's library path. The interface and its contract are in `include/matern.h`.

* `nu` and `scale` are arrays of length `n`; a constant order is passed as a filled array.
* With `workspace = NULL` the library creates a workspace at a thread's first call and keeps
  it for that thread, which releases it with `matern_release_thread_workspace()` before it
  ends (the main thread may leave this to process exit); `scalar`, `scalar_portable`,
  `matched` and `matched_fma` use no workspace. A workspace from
  `matern_workspace_create()` lives until `matern_workspace_destroy()` and serves one thread
  at a time. Neither form allocates on every call.
* On the CPU backends other than `scalar`, the same point can differ in the last bits
  depending on the other points of the batch, so a covariance matrix need not be exactly
  symmetric; where exact symmetry matters, compute one triangle and mirror it.
* `matern_version()` returns the release of the library linked, `MATERN_VERSION_STRING`
  that of the header compiled against.
* `matern_covariance_z()` takes `z` directly, for a model that works in `z`. A caller who
  has `h` and `s` passes them to `matern_covariance()`, whose vector backends carry the
  derived `z` as a double-double; computing `z = √(2ν)·h/s` in double first adds a relative
  error of about `z·ε` to the result.

### CUDA

```c
if (matern_cuda_available())
    matern_covariance_cuda(out, h, nu, scale, n, MATERN_CUDA_ASYNC);
```

The CUDA entry points are separate from `matern_covariance()`. The device code is the
object `build/cuda.o`, built by `make cuda` (nvcc; `GPU_ARCH=sm_80` by default), and a
program that uses the GPU links it together with the library:

```sh
cc -O2 -Iinclude app.c build/cuda.o -Lbuild -lmatern \
   -L$CUDA_HOME/lib64 -lcudart -lstdc++ -fopenmp -lm
```

After `make shared`, `-Lbuild -lmatern` links `build/libmatern.so`, found at run time for
example with `LD_LIBRARY_PATH=build`; `build/libmatern.a` in its place links the static
library. `make install` copies `build/cuda.o`, when it has been built, to
`$PREFIX/lib/matern_cuda.o`.

Without the object the same program links against the library alone, which carries weak
stand-ins: `matern_cuda_available()` returns 0 and the entries return
`MATERN_UNSUPPORTED_CPU`, as they do with the object on a machine without a device. The
transfer modes (`MATERN_CUDA_SYNC`, `MATERN_CUDA_ASYNC_DIRECT` =
`MATERN_CUDA_ASYNC`, `MATERN_CUDA_ASYNC_RING`, `MATERN_CUDA_ASYNC_MAPPED`) differ only in how
data is moved and return bit-identical results; no mode keeps the caller's memory
page-locked after it returns.

For data already on the GPU, `matern_covariance_cuda_device()` and
`matern_covariance_cuda_device_z()` take device pointers and a stream and only launch: no
allocation, copy or synchronisation. Inside a kernel of your own, the header
`include/matern_device.cuh` provides the same evaluation per point as `__device__`
functions, needing neither the object nor any host setup:

```cuda
#include "matern_device.cuh"          // nvcc -Iinclude

__global__ void covariance(const double *h, double nu, double s, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = matern_covariance_device(h[i], nu, s);
}
```

Both return the bits of `matern_covariance_cuda()` for the same point. The header's
conditions (nvcc's default `-fmad=true`, about 5 KB of constant memory per translation
unit, about 50 registers) are stated in it. `make install` installs it with its
`include/matern_device/` headers.

## Testing

`make check` builds the library and runs the verification suite. The tests check properties
of the output stated in `include/matern.h` without a high-precision reference, so they need
only libm and pthreads, plus Python 3 for `audit` and `readme-example`. The per-backend tests
run once for each backend this CPU runs; a backend it cannot run is reported as `skipped`.
Each part can also be run on its own:

| target | checks |
|---|---|
| `make contract` | per backend: `0 < M <= 1` and finite on a sample of `10⁻³ ≤ ν ≤ 10³`, `10⁻³ ≤ z ≤ 700`, `M` non-increasing in `h`, NaN for invalid input (also infinite `ν` or `s`), 0 at `h = +∞`, `M > 0` past the exponential limit, `matern_version()` |
| `make h-zero` | per backend: `M(0)` is exactly 1 in single, batched and mixed calls |
| `make crosscheck` | `scalar` and `scalar_portable` against `matched`, which share no region kernel; reports the largest relative difference |
| `make positive-definite` | per backend: covariance matrices factor by Cholesky; symmetry and batch invariance are reported |
| `make direct-argument` | per backend: the `(z, ν)` entry `matern_covariance_z()` |
| `make threads` | per backend: 8 threads reproduce the single-thread result bit for bit; a positive control that shares one workspace must fail |
| `make exports` | `build/libmatern.a` defines exactly the names in `matern.map`, the CUDA stand-ins weak; `build/cuda.o`, when built, no other name |
| `make audit` | static checks of the tree (`tools/audit.py`); outside a git checkout one check prints `skip`, which is not a failure |
| `make neutral` | every region kernel compiles at width 1 |
| `make readme-example` | the Quick start program compiles without warnings and prints the values shown |

`make tsan` runs the thread test under ThreadSanitizer on every backend this CPU runs
(`TSAN_CC` names the compiler, default `gcc`), `make strict` is a pedantic warning pass over
the portable sources, `make cuda-modes` checks the CUDA transfer modes and `make cuda-device`
checks `include/matern_device.cuh` and the device-pointer entries against the CUDA entries
(both need a GPU; neither is part of `make check`).

## Accuracy

In the contract domain, `10⁻³ ≤ ν ≤ 10³` and `10⁻³ ≤ z ≤ 10⁵`, every backend returns
`M(0) = 1` exactly, values in `[0, 1]` that are non-increasing in `h` up to rounding in the
last bits, a nonzero value past the exponential limit `z ≈ 708.4` wherever `M ≥ DBL_MIN`, and
NaN for invalid input (`ν ≤ 0`, `s ≤ 0`, `ν` or `s` infinite, `h < 0`, any NaN); the exact
statements are in `include/matern.h`. The benchmark also measures points below this domain
(`data/DATASETS.md`). No error bound is guaranteed: accuracy is measured by the benchmark
against arbitrary-precision references (FLINT/arb) and reported in the accompanying paper,
and the `scalar` backend has larger errors than the others. The `avx2` backend can differ in
the last bit between processor vendors, because its gamma function uses the processor's
single-precision reciprocal estimate (`VRCPPS`), whose result the instruction set leaves to
the implementation.

## Benchmark

`benchmark/` measures the accuracy and throughput of every backend on the evaluation corpus,
compares the library with external implementations of `K_ν` (Bessels.jl, the C++ standard
library, Boost, GSL and KAUST CuBesselK) and generates the figures and tables of the paper.
`make requirements` lists the external software it can use and how to add it, and
`make pipeline` builds and runs everything in one command; on a terminal it first asks
whether to switch to the full settings and to download the missing externals, KAUST
CuBesselK and the corpus. The guide is
[`docs/benchmark.md`](docs/benchmark.md), including the full run behind the published results
([Replicating the published results](docs/benchmark.md#replicating-the-published-results)).

## Data

The evaluation corpus, arguments with arbitrary-precision references, is published on Zenodo
as [doi:10.5281/zenodo.23393956](https://doi.org/10.5281/zenodo.23393956) under CC-BY-4.0;
cite the DOI when using the data. `data/` holds its documentation and manifest:
`make corpus` reports which files are present, `make corpus DOWNLOAD=yes` downloads the
missing or damaged ones into `data/`, and `make corpus-check` checks every file against the
sha256 list in `data/MANIFEST.md`. The sets are specified in
[`data/DATASETS.md`](data/DATASETS.md); [`data/README.md`](data/README.md) describes the sets
and file formats.

## Repository layout

| path | contents |
|---|---|
| `include/` | the public headers: `matern.h`, and `matern_device.cuh` with the device code in `include/matern_device/` |
| `src/` | the library: SIMD and double-double arithmetic, elementary functions, region kernels, router and backends |
| `test/` | the verification suite run by `make check` |
| `example/` | a covariance matrix and its Cholesky factorisation |
| `data/` | documentation and manifest of the evaluation corpus |
| `benchmark/` | benchmark drivers, run scripts and the figure and table generators |
| `references/` | adapters for the external `K_ν` implementations used in the comparison |
| `tools/` | scripts behind the corpus, externals, requirements and check targets |
| `docs/` | the benchmark guide, [`docs/benchmark.md`](docs/benchmark.md) |

## Licence and citation

libmatern 0.9.0 is released under the MIT licence ([`LICENSE`](LICENSE)), copyright Daniel
Mann, Matthis Ehrhardt and Martin Schlather. Third-party code under `references/` keeps its
own licence ([`docs/benchmark.md`, "External suppliers"](docs/benchmark.md#external-suppliers)).
KAUST
CuBesselK is not distributed: `make kaust` prints its copyright notice, and
`make kaust DOWNLOAD=yes`, or a yes to the question of `make pipeline`, downloads and ports
it on the user's machine. The evaluation
corpus is CC-BY-4.0 ([Data](#data)). To cite the software, use the metadata in
[`CITATION.cff`](CITATION.cff) together with the accompanying paper: D. Mann, M. Ehrhardt
and M. Schlather, *libmatern: Fast and Accurate Matérn Covariance Evaluation on SIMD CPUs and
CUDA GPUs*.
