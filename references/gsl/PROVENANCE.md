# `gsl` — the GNU Scientific Library as a supplier

## Upstream

[GSL](https://www.gnu.org/software/gsl/), version 2.6, the version of the published rows.
The supplier calls `gsl_sf_bessel_Knu`, `gsl_sf_bessel_Knu_scaled` and `gsl_sf_bessel_lnKnu`
(`specfunc/bessel_Knu.c`).

**Licence: GPL-3.0.** GSL is linked into the benchmark programs `build/suppliers` and
`build/threads_cpu`, not distributed: the repository holds no GSL source or binary.
`cpu_gsl.c` is this project's code (MIT, `../../LICENSE`).

## How it is obtained

`make externals` runs `references/fetch_gsl.sh`, which downloads `gsl-2.6.tar.gz` from
ftp.gnu.org or a mirror, checks its sha256 against the pinned digest
`b782339fc7a38fe17689cb39966c4d821236c28018b6593ddb6fd59ee40786a8`, and builds a static
library into `references/gsl/2.6/`. Without it, `benchmark/tools/gsl_flags.sh` uses a system
GSL found by pkg-config; its version is recorded in each run's `PROVENANCE.txt`, and the
published rows used 2.6.

## Use

`cpu_gsl.c` disables GSL's default error handler, which calls `abort()` on an error such as
an overflow, and passes the three forms through unmodified; the benchmark's accounting of
non-finite results decides whether a point is scored. GSL has no device implementation, so
it is a CPU row only.
