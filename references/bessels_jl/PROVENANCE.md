# `bessels_jl` — Bessels.jl and two C ports of its `besselk`

## Upstream

[Bessels.jl](https://github.com/JuliaMath/Bessels.jl), a pure-Julia implementation of
Bessel functions.

**Licence: MIT.** The full text is in `LICENSE`, copied verbatim from the upstream `LICENSE`
file at the pinned commit (below); `make references` fetches that file and checks that the
two are identical. Copyright (c) 2021-2023 Michael Helton, Oscar Smith, Chris Geoga, and
contributors. The ports in `port/` are derived work and are covered by that notice, and for
the K0/K1 coefficients also by the Boost.Math notice in `LICENSE_BOOST_1_0.txt` (below).

### Boost.Math origin of the K0/K1 coefficients

The pinned `besselk.jl` states that its K0/K1 polynomial coefficients "are taken from boost
math library" (its comment at line 32, with the Boost.Math `detail` directory as reference
[3], line 66). They are: every literal of the eleven K0/K1 tables in
`port/cpu_bessels_data.h` (`P1_k0` to `Q3_k1`, read from `constants.jl`) occurs, in the same
order, in `boost/math/special_functions/detail/bessel_k0.hpp` and `bessel_k1.hpp` of Boost
1.75.0. Those headers carry "Copyright (c) 2006 Xiaogang Zhang" and "Copyright (c) 2017 John
Maddock" under the Boost Software License 1.0 (BSL-1.0), which asks that its notice
accompany copies "in whole or in part". `LICENSE_BOOST_1_0.txt` holds the two headers'
notice and the licence text, beside the MIT notice in `LICENSE`. The Debye tables
(`U_polynomials.jl`) are not involved; upstream cites DLMF 10.41 for them.

## Fetched sources

The upstream sources are not distributed with this tree. `make references`
(`references/fetch.sh`) downloads them into `upstream/` from
<https://github.com/JuliaMath/Bessels.jl> and verifies each against its sha256 digest. Every
file but one comes from the pinned commit **`009b615347d91d1dd69962641647521901362af3`**
(committed 2026-06-30); `besselk_old.jl` comes from commit
**`1475e9e57464dbb7b8ba2d812766e34d0f822c4d`**, release 0.2.8 (see Earlier revision).

| file | commit | upstream path | sha256 |
|---|---|---|---|
| `besselk.jl` | `009b615` | `src/BesselFunctions/besselk.jl` | `3560c95f769af60cfba49d932d4ce6e6e041b01128cb00bfd4b6a480066695b1` |
| `besseli.jl` | `009b615` | `src/BesselFunctions/besseli.jl` | `99df3dcd86ccba44186279da13659e66db6e4d205e6f99290f3ce2de86e18fb6` |
| `constants.jl` | `009b615` | `src/BesselFunctions/constants.jl` | `92936551ce9563e112ff538d0b5de40014a695d9f7cec23ce1924dd43e58fa07` |
| `U_polynomials.jl` | `009b615` | `src/BesselFunctions/U_polynomials.jl` | `87e3cee08cf2cda5d763b4c10f11d5a0dce112179736913f1388485ce6eafb4c` |
| `recurrence.jl` | `009b615` | `src/BesselFunctions/recurrence.jl` | `71536a72dd8f752203d7cf4a66fba4cbdee795fd44e1a75c719d483d2e385322` |
| `sphericalbessel.jl` | `009b615` | `src/BesselFunctions/sphericalbessel.jl` | `a8e55b8c7729c08bf7c8d7e700e5a0ba86b0e94951cfab9361d55cb1eeb18a30` |
| `modifiedsphericalbessel.jl` | `009b615` | `src/BesselFunctions/modifiedsphericalbessel.jl` | `7dcd41ef5c5350cdf712dda6c1eeae6168fc4854f9bc20158e640af29eaa51b5` |
| `Math.jl` | `009b615` | `src/Math/Math.jl` | `0541b470fecdff0de96462c653568c6055e9738ac77091f349ec0423906dc982` |
| `math_constants.jl` | `009b615` | `src/Math/math_constants.jl` | `39a9baae6be2bbc0b99a5d28202d3c1775a667e21df82e561975094d7e5c7e82` |
| `Bessels.jl` | `009b615` | `src/Bessels.jl` | `e1ea3129c05e27586ba99d20a695607a1fcbf6bb3c7ad26e31306d8b2e0f9426` |
| `BF.jl` | `009b615` | `src/BesselFunctions/BesselFunctions.jl` | `6a634c1d66e8f297acfec67b43bbeaf6ebe625482c4bef683288282ffd6ed372` |
| `LICENSE` | `009b615` | `LICENSE` | `1ab0da1b8fd834885da485f90a5093fd1ce7430a44459472721589d10eb2a384` |
| `besselk_old.jl` | `1475e9e` | `src/besselk.jl` | `e684931f3283258d74b4b8661836ee4cf0ab99a14f13c2acbdf735ce55292b5f` |

`besselk.jl` is the file ported by `cpu_bessels_jl.c`, and `besselk_old.jl` the file ported
by `cpu_bessels_old.c`; the other `.jl` files hold the code and constants the pinned
`besselk.jl` references. `Bessels.jl` and `BF.jl` only establish where the pinned commit's
`gamma` comes from (the separate Gamma.jl package); release 0.2.8 defines its own in
`src/gamma.jl`, which is not fetched. Both are replaced by libm `tgamma` in the ports
(Substitutions, below); neither `Bessels.jl` nor `BF.jl` is ported.
`LICENSE` is fetched to check `bessels_jl/LICENSE` against it.

### Earlier revision

`cpu_bessels_old.c` ports `besselk.jl` of Bessels.jl release 0.2.8: tag `v0.2.8`, commit
`1475e9e57464dbb7b8ba2d812766e34d0f822c4d`, path `src/besselk.jl`. The tree of that commit,
`4435559dc39793d53a9e3d278e185e920b4619ef`, is the `git-tree-sha1` that Julia's General
registry records for Bessels 0.2.8. `make references` fetches the file as
`upstream/besselk_old.jl` and verifies it (table above). The release's `LICENSE` is the same
MIT text with the copyright line "Copyright (c) 2021-2022 Michael Helton, Oscar Smith, and
contributors".

Three definitions that this `besselk.jl` uses but does not define are taken in the port from
the pinned commit, each marked in the port: the one-line cutoffs
`besseli_large_argument_cutoff` (`besseli.jl`) and `sphericalbesselk_cutoff`, and the
function `sphericalbesselk_int`, the half-integer closed form (these two in
`modifiedsphericalbessel.jl`). Release 0.2.8 defines all three, in files of the same
names. Its `src/modifiedsphericalbessel.jl` is byte-identical to the pinned commit's (the
same sha256 as in the table above), so `sphericalbesselk_cutoff` (`nu < 41.5`) and
`sphericalbesselk_int` are the release's own text. `besseli_large_argument_cutoff` is not:
release 0.2.8 has `x > 23.0 && x > nu^2 / 1.8 + 23.0`, the pinned commit
`x > nu^2 / 2 + 19.0`, which the port uses. The port therefore turns to the large-argument
expansion where the pinned commit does, not where release 0.2.8 does.

## Ports

`port/` holds C written for this project from the upstream Julia (a translation, not upstream
files); as derived work it carries the Bessels.jl notice in `LICENSE`, and the Boost.Math
notice in `LICENSE_BOOST_1_0.txt` for the K0/K1 coefficients.

| file | contents |
|---|---|
| `cpu_bessels_jl.c` | port of `besselk.jl` at the pinned commit (`besselk`, `besselkx`) |
| `cpu_bessels_old.c` | port of the earlier revision, `besselk.jl` of release 0.2.8 |
| `cpu_bessels_shared.h` | code the two revisions have in common (K0/K1, Debye, forward recurrence): the same text and tables in both, apart from the spacing of one line (end of Substitutions in the ports) |
| `cpu_bessels_data.h` | coefficient tables, generated by `extract_upstream.sh` |
| `extract_upstream.sh` | generator for `cpu_bessels_data.h` |

### Coefficients

`cpu_bessels_data.h` is generated by `extract_upstream.sh`, which reads the literals from
`upstream/constants.jl` and `upstream/U_polynomials.jl` and rewrites Julia tuple syntax as C
arrays; every literal is the upstream text unchanged. The script first checks both sources
against their sha256 digests in the table above and stops, writing nothing, if either is
missing or differs; it fails if a table is missing and rejects a `Float32` literal. To
regenerate (after `make references`):

```sh
make -C references extract     # or: cd port && ./extract_upstream.sh
```

### Substitutions in the ports

1. **Julia `muladd` → C `fma`.** Julia lowers `muladd` to a hardware FMA wherever one exists.
2. **`gamma` → libm `tgamma`.** Upstream's `gamma` is the Gamma.jl package's at the pinned
   commit and Bessels.jl's own (`src/gamma.jl`) in release 0.2.8. Used by the power series:
   `besselk_power_series` (ν ≤ 25 and not near an integer, x ≤ 1.5) and, in
   `cpu_bessels_old.c`, also the `I_ν` power series. This changes the arithmetic of those
   paths, and the rows depend on the libm: `tgamma` is not correctly rounded. Against mpmath
   at 160 bits, glibc 2.34's `tgamma` was more than 0.5 ulp off at 47 % of 50 000 points
   sampled over ν ∈ [1e-6, 25], by up to 4.2 ulp (largest measured on that sample; not a
   bound).
3. **Julia `sinpi` → exact reduction and libm `sin`.** Julia's `sinpi` reduces its argument
   exactly to |r| ≤ 1/4 and evaluates its own polynomial kernels for sin(πr) and cos(πr).
   `bjl_sinpi` (`cpu_bessels_shared.h`) also reduces exactly (`x − nearbyint(x)`), but then
   evaluates libm `sin(π·f)` on |f| ≤ 1/2, so its result can differ from Julia's in the last
   bits. Used for Γ(−ν), by reflection, in the power series of both ports.
4. **Julia's elementary functions → the C library's.** `exp`, `log`, `^`, `cosh`, `hypot` and
   `log1p` are Julia's own implementations upstream and the C library's (`exp`, `log`, `pow`,
   `cosh`, `hypot`, `log1p`) in the ports, so the rows depend on the libm here as well (the
   published runs were built against glibc 2.34). `sqrt` is correctly rounded in both.

The Levin transform follows Julia exactly: upstream carries `(numerator, denominator)` as a
2-vector and applies the same FMA to both lanes, which here is two scalar `fma`s in the same
order.

`cpu_bessels_old.c` takes the K0/K1, Debye and forward-recurrence code, with its tables and
`SQPIO2`, from `cpu_bessels_shared.h` and `cpu_bessels_data.h`, which are ported from the
pinned commit. Release 0.2.8's own files (commit `1475e9e`) have the same code: its
`constants.jl` and `recurrence.jl` are the pinned commit's files (the same git blob);
`besselk0`, `besselk1`, their scaled forms, `besselk_large_orders_scaled` and
`besselik_debye_cutoff` have the same text in both `besselk.jl`, and `besselk_large_orders`
differs only in the spacing of one line; its `U_polynomials.jl` differs only in the `return`
lines of `besseljy_debye` and `hankel_debye`, and its `math_constants.jl` only in the
`export` lines and the constants `PIPOW3O2` and `ONEOSQPI`, none of which the ports use, so
the Debye tables and `SQPIO2` are the same. The rest of `cpu_bessels_old.c` is ported from
release 0.2.8's `besselk.jl` itself. Apart from the substitutions above, it therefore departs
from release 0.2.8 only through the three definitions it takes from the pinned commit
(Earlier revision), of which only `besseli_large_argument_cutoff` differs in text.

## The earlier revision

The two revisions cover the `(ν, z)` plane differently. The earlier one contains two methods
the pinned revision does not: a half-integer closed form (`ν − 1/2` integral, `ν < 41.5`, via
the modified spherical Bessel recurrence; Matérn ν = 1/2, 3/2, 5/2, ...) and a continued
fraction (`I_ν` power series plus the CF2 ratio by modified Lentz, closed with the
Wronskian). Its `besselkx` has no half-integer branch, so `besselk` and `besselkx` take
different routes at half-integer ν.
