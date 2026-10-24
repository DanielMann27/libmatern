# `kaust_logbesselk` — KAUST CuBesselK

## Upstream files

CuBesselK (Geng et al., KAUST; see References) is published at
<https://github.com/stsds/CuBesselK>. The benchmark uses two files of commit
`94ec9ca4bb9d7524c196f044503519ab30d8eb39` (2025-03-17), which are downloaded into
`upstream/` under their repository names only on the user's explicit yes, after their
copyright notice: `make kaust DOWNLOAD=yes` (the notice: `make kaust`) or the question that
`make pipeline` asks at its start on a terminal. They are not distributed with this tree;
`references/fetch.sh`, `make -C references verify-upstream` and `kaust.sh` check them
against these digests, and `derive_port.py` refuses any other file:

| file | repository path | lines | sha256 |
|---|---|---|---|
| `upstream/logbesselk.cu` | `src/logbesselk.cu` | 283 | `5a51e1a5c7c08ecc31696db58d3a69e9fc9e3fe6d8071e57af26dcaa666ee5ed` |
| `upstream/logbesselk.h` | `include/logbesselk.h` | 55 | `650bafdad9c4444992fbf1b5435bc2d16db0e70fff3135066d79f08eeea30550` |

`logbesselk.cu` defines the device function `modified_besselk(nu, x)`, the kernel
`modified_besselkKernel` and the host entry `BesselK_CUDA`, with the panel count in the host
variable `intervals = 128` (`:27`), `setIntervals()` (`:33`) and `initBesselK()` (`:39`).
`logbesselk.h` defines `cheb_series` and the constants `DBL_EPSILON` and `SQRT_DBL_MAX` that
`logbesselk.cu` uses.

### Licence

The repository has no licence file. `logbesselk.cu` and `logbesselk.h` carry, verbatim:

```
* Copyright (c) 2025  King Abdullah University of Science and Technology
* All rights reserved.
*
* CuBesselK is provided by KAUST
```

The repository README says that its CUDA functions may be copied into a program; that is
not a licence grant. "All rights reserved" without a grant does not permit
redistribution, so neither the files nor anything derived from their text is distributed
here, apart from the short fragments that `derive_port.py` matches to find the code it
rewrites (declarations, function signatures, single statements): the port is generated on
the user's machine by `make kaust DOWNLOAD=yes` (or after a yes to the question of
`make pipeline`) into `references/build/kaust/`, which is not distributed, and deleted (with
the objects and programs built from it) at the end of `make pipeline` and by
`make kaust CLEAN=yes`.

The small-argument branch also carries material of a second rights holder: the Chebyshev
tables of Temme's series, `g1_dat` (14 entries) and `g2_dat` (15 entries) of `logbesselk.cu`,
are those of the GNU Scientific Library (GPL-3.0). Packed as doubles, each occurs once and
contiguously in the GSL 2.6 library (`libgsl.so.25.0.0`). The downloaded files, and the port
`derive_port.py` generates from them (it copies both tables), therefore contain GSL data as
well.

## Algorithm

* **x ≤ 0.1** — Temme's series for `|μ| ≤ 1/2` (GSL's Chebyshev `g1`/`g2` tables), then the
  forward recurrence `K_{ν+1} = 2(μ+n+1)/x · K_ν + K_{ν−1}` up to the requested order, on the
  `e^{+x}`-scaled values.
* **x > 0.1** — Takekawa's trapezoid rule (References) on
  `K_ν(x) = ∫₀^∞ cosh(νt) e^{−x cosh t} dt` over `t ∈ [0, 9]` with `intervals` panels,
  summed with the log-sum-exp trick.

The trapezoid branch forms `res = log(h) + max_term + log(sum)` and exponentiates only at
the end (`logbesselk.cu:245-246`), so `ln K_ν` is available natively; the port returns it
as the supplier's log form (`WMC_LOG_NATIVE`).

## Upstream defects

| # | defect | where |
|---|---|---|
| 1 | the small-`x` forward recurrence accumulates a decimal exponent `e10` and does not apply it to the returned value (`return K_nu * exp(-x)`) | `:199`, `:216`, `:220` |
| 2 | the integrand forms `log(cosh(νt))` by evaluating `cosh` first; with the quadrature running to `t = 9`, `cosh(9ν)` overflows for ν ≳ 78.9 and the result is `inf`/`nan` for every `z` | `f`, `:182-186` |

A third property is a design parameter rather than a defect: the fixed range `t ∈ [0, 9]`
with a fixed panel count. The integrand's support contracts toward `t = 0` as `x` grows (to
`t ~ sqrt(72/x)`, 0.32 at x = 700) while the panel width `h = 9/128` is fixed, so accuracy
falls at large `x`: at 128 panels the relative error is 9.4e-06 at (x, ν) = (700, 0.5),
9.1e-09 at (300, 1.5) and 1.0e-08 at (300, 30).

## The port

Two files of this folder hold the project's code for the port:

* `derive_port.py`, which holds its rewriting rules and the short fragments they match,
  checks the two digests and rewrites `logbesselk.cu` mechanically into
  `references/build/kaust/kaust_derived.h`: the host code and the kernel are dropped,
  `__constant__`/`__device__` become qualifiers that `kaust_port.c` defines for the host
  compiler and for nvcc, every name gains the prefix `ka_`, the panel count becomes a
  parameter, and `modified_besselk` is split into its small-argument branch (returning the
  `e^{+x}`-scaled `K` and `e10`) and three forms of its quadrature (as released; with the
  non-overflowing integrand; with a given integration range), each returning `res` before
  the final `exp`. The header is read for its two constants and not included.
* `kaust_port.c` adds the non-overflowing integrand, the integration range of Geng et al.'s
  Eq. (7), the three configurations below, the host supplier table and the device entry
  points `wmk_dev_kv` / `wmk_dev_logkv`. It is compiled as C for the CPU rows and as CUDA
  for the GPU rows, so a variant name means the same configuration on both devices.

The benchmark registers three configurations (`../common/cpu_wm_registry.h`):

| variant | symbol | configuration |
|---|---|---|
| `kaust-128` | `wmc_kaust` | 128 panels (upstream's default), both defects as released |
| `kaust-40` | `wmc_kaust40` | `setIntervals(40)`, the panel count Geng et al. recommend, both defects as released |
| `kaust-adaptive` | `wmc_kaust4` | 128 panels, `e10` applied, `log(cosh y) = \|y\| − ln2 + log1p(e^{−2\|y\|})`, and the adaptive integration range of Geng et al.'s Eq. (7) in place of `t ∈ [0, 9]` |

The corrected configuration and the log-space assembly of `../common/cpu_wm.h` (on the GPU
`../../benchmark/suppliers_gpu.cu`; the exponent is summed and exponentiated once) make
KAUST more accurate than the released code; a comparison quoting the `kaust-adaptive` row
should state that it is an adjusted configuration.

## References

* Z. Geng, S. Abdulah, Y. Sun, H. Ltaief, D. E. Keyes, M. G. Genton. *GPU-accelerated
  modified Bessel function of the second kind for Gaussian processes.* arXiv:2502.00356,
  version 2 (April 2025), <https://arxiv.org/abs/2502.00356v2>. Equation numbers cited in
  this tree refer to that version: Eq. (7), in Section IV.B, is the integration range
  `[t0, t1] = {t : g(t) ≥ log(ε) + g(t_max)}` with `g(t) = log cosh(νt) − x cosh t`
  (version 1 numbers it Eq. (8)); Section IV.C recommends 40 panels.
* T. Takekawa. *Fast parallel calculation of modified Bessel function of the second kind and
  its derivatives.* SoftwareX 17 (2022), 100923. doi:10.1016/j.softx.2021.100923. The
  trapezoid rule on the integral representation; Section IV.B of Geng et al. describes it,
  including that integration range.
