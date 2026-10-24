# `cpp_stdlib` — the two `K_ν` implementations that ship with a C++ toolchain

No third-party source is vendored here. Both suppliers come with a C++ toolchain;
`cpu_cxx.cpp` is this project's wrapper around them.

| supplier | where it comes from | licence |
|---|---|---|
| `std::cyl_bessel_k` | the C++17 `<cmath>` special functions of the standard library | that of the standard library (libstdc++: GPL-3.0 with the GCC Runtime Library Exception) |
| `boost::math::cyl_bessel_k` | Boost.Math, header-only (`boost/math/special_functions/bessel.hpp`) | Boost Software License 1.0 |

`bessel.hpp` carries the BSL-1.0 notice in its header: copyright John Maddock (2007, 2013)
and Christopher Kormanyos (2013), with most of the implementation copyright Xiaogang Zhang.

## Versions

| | version used for the published measurements |
|---|---|
| Boost | 1.75.0 (`BOOST_VERSION 107500`; `build_cpu_supplier_cxx_boost 1.75.0`) |
| libstdc++ | 14 (GCC 14.2.0; `build_cpu_supplier_cxx_libstdcxx 14 (20240801)`, `build_cpu_supplier_cxx_compiler gcc 14.2.0`) |

The keys in parentheses are those of `PROVENANCE.txt` in every published run that built the
CPU suppliers.

`std::cyl_bessel_k` is whatever the standard library provides. In libstdc++ it is the TR1
implementation (`tr1/modified_bessel_func.tcc`): Temme's series for small x and
Steed's continued fraction otherwise, with no separate large-order method. Another standard library
gives a different row, so a number quoted from this row should state the library version.
The Boost row label carries the Boost version (`boost-<version>`). The row is built
against Boost 1.75.0 only (`../../docs/benchmark.md`, Boost versions); `make boost-headers`
fetches it.

## Upstream

* **Boost 1.75.0:** <https://archives.boost.io/release/1.75.0/source/boost_1_75_0.tar.bz2>,
  sha256 `953db31e016db7bb207f11432bef7df100516eeb746843fa0486a222e3fd49cb`, the archive
  `make boost-headers` fetches and checks (`../fetch_boost.sh`). `K_ν` is in
  `boost/math/special_functions/bessel.hpp` and the headers it uses for it under
  `boost/math/special_functions/detail/`: `bessel_ik.hpp` (non-integer order), `bessel_kn.hpp`,
  `bessel_k0.hpp` and `bessel_k1.hpp` (integer order). The GPU Boost row uses Boost 1.87.0,
  from the same server (`../fetch_boost.sh` holds its digest).
* **libstdc++ 14:** part of GCC 14.2.0, whose source release is at
  <https://ftp.gnu.org/gnu/gcc/gcc-14.2.0/>; `std::cyl_bessel_k` is in
  `libstdc++-v3/include/tr1/modified_bessel_func.tcc` of that source, installed as
  `<include>/c++/14.2.0/tr1/modified_bessel_func.tcc`. Neither library is distributed here.

## Wrapper

`cpu_cxx.cpp` calls the two functions and registers them behind the shared supplier
interface (`../common/cpu_wm.h`), so the covariance assembly around `K_ν` is the same as
for every other supplier.

The only configuration change: both return a value instead of throwing, so an overflow
appears in the tables as `inf`/`nan`. For Boost this is an error policy that ignores
domain, overflow, underflow, evaluation and pole errors; the `std::` call is wrapped in a
`catch` that returns NaN. The numerics are untouched.

Neither library exposes a scaled or a log-space form, so the log/exp assembly for these
rows is `log(K_ν)`, which cannot recover a `K_ν` that already overflowed inside the
library. By contrast, KAUST's quadrature branch produces `ln K_ν` natively, and Bessels.jl
and GSL publish a scaled `K_ν e^z`.
