# Dataset specification

This file specifies the evaluation corpus of libmatern: the datasets in `z_nu/` and `h_nu_s/`
and the `boost_shortcut` dataset. The census of each family and the sha256 of every file are
in `MANIFEST.md`.

The arguments and references are IEEE-754 doubles (the `bits` column a 16-bit integer; the
seam labels text), stored in binary files (`benchmark/corpus_bin.h` in libmatern).
`build/corpus_convert` (`make corpus-tools`) prints them as the `%.13a` text of the generator
(the program that drew the points and computed the references; not published) and converts
that text back, byte for byte.

## 1. Two families

The normalised Matérn covariance

    M(h) = 2^(1-nu) / Gamma(nu) * z^nu * K_nu(z),    z = sqrt(2 nu) h / s

takes `(h, nu, s)` and derives `z`. Deriving `z` in double precision adds a relative error of
about `z * eps` to `M`, independent of the internal precision of the Bessel kernel:
`M ~ e^-z`, so an absolute error in `z` is a relative error in `M`.

| family | stored arguments | what it scores |
|---|---|---|
| `z_nu` | `(z, nu)` | the kernel alone, through the direct-argument entry `matern_covariance_z` |
| `h_nu_s` | `(h, nu, s)` | the public entry: kernel plus the derivation of `z` |

The two families are a matched pair. Both use the same `(z, nu)` draws; `h_nu_s` sets
`h = z` exactly and `s = fl(sqrt(2 nu))`. The files `h_nu_s-nu_<dataset>_<N>.bin` and
`h_nu_s-h_<dataset>_<N>.bin` are therefore byte-identical to `z_nu-nu_<dataset>_<N>.bin` and
`z_nu-z_<dataset>_<N>.bin`, for all ten datasets. The scale parameter `s` is stored in the
`h_nu_s-s_<dataset>_<N>.bin` files.

The `h_nu_s` reference derives `z = sqrt(2 nu) h / s` in arb, inside the precision loop
(Section 4), so it is the covariance at the exact stored `(h, nu, s)`. That argument differs
from the `z_nu` argument by the rounding of `s`, about 0.5 ulp of `z`, so the two families do
not share references.

Consequences of the tie `s = sqrt(2 nu)`:

* The difference between the double-precision and the exact argument is the rounding of the
  multiplication and the division, about 2 eps, whether or not `s` equals `sqrt(2 nu)`
  exactly; the tie does not reduce the derivation error being measured.
* `s` is never an independent input, and `h` and `z` are numerically equal at every point.
  The corpus therefore cannot detect a reader that confuses `h` with `z`; the distinct file
  prefixes `h_` and `z_` turn such a confusion into a missing-file error. Independent `s`
  ranges and invalid inputs (for example `h < 0` together with `s < 0`) are covered by
  `test/contract.c` in libmatern, not by the corpus.
* Compare the two families by error distributions (maximum, median, upper quantiles), not by
  a per-index difference: index `i` names points whose arguments differ by about 0.5 ulp in
  `z`, which is worth `z * eps` in `M`, more than the effect being measured.

## 2. Layout, naming and formats

```
z_nu/      z_nu-nu_<dataset>_<N>.bin    z_nu-z_<dataset>_<N>.bin    z_nu-ref_<dataset>_<N>.bin
h_nu_s/    h_nu_s-nu_<dataset>_<N>.bin  h_nu_s-h_<dataset>_<N>.bin  h_nu_s-s_<dataset>_<N>.bin
           h_nu_s-ref_<dataset>_<N>.bin
           seam also has <family>-label_seam_100000.txt
```

The family prefix keeps every file name unique, so the files may also lie directly in the
corpus root (`README.md`, Layout). Record `i` of every file of a dataset describes point `i`.

| file | content |
|---|---|
| `<family>-nu_ z_ h_ s_` | one double per point |
| `<family>-ref_` | `hi`, `lo` (doubles) and `bits` (16-bit integer) per point, stored as three arrays |
| `<family>-label_seam_` | text, `class axis sign k` per line (Section 5) |

* A binary file is a 32-byte header (magic `WMCORP01`, version, kind, point count, the text
  format the generator printed) followed by little-endian arrays; its size follows from the
  header, so a truncated file is refused (`benchmark/corpus_bin.h`).
* The generator writes `%.13a` text, the 52 fraction bits as 13 hexadecimal digits;
  `build/corpus_convert to-bin` stores the doubles and refuses any line that does not print
  back identically, and `verify` prints the binary file as text and compares it with the text
  file byte for byte.
* The reference value is the double-double `hi + lo`. `hi` is the arb midpoint rounded to the
  nearest double (a single rounding, also for subnormal results); `lo` is the remainder,
  formed in arb and rounded to a double. The pair carries about 106 bits, 53 more than the
  doubles being scored, except near the bottom of the double range: where `hi` is subnormal,
  `lo` is 0, where `hi` is close to `DBL_MIN`, `lo` is 0 or subnormal, and where `hi` is 0
  (true `M` at most 2^-1075, half the smallest subnormal) the value is not stored at all.
  `lo` is often -0.0.
* `bits` is `arb_rel_accuracy_bits` of the final ball. The sentinel values 32767 (exact ball)
  and -32768 (zero midpoint) are defined but occur nowhere in the corpus.

## 3. The datasets

A dataset is a fixed list of points, drawn from the distribution in the `sampling` column.
Ten datasets per family, 3 259 496 points per family: eight in files of their own (1 162 344
points; 25 files in `z_nu/`, 33 in `h_nu_s/`), and the two size datasets, read as the first
2^20 points of the 2^25-point files (`size_random` and `size_half` below).

| dataset | N | sampling | seeds |
|---|---|---|---|
| `grid` | 262144 | `512 x 512` linear tensor, nu [1e-3, 20] x z [1e-3, 140] | none |
| `random` | 250000 | nu, z independent uniform on [1e-3, 40] | 101/102 |
| `small_z` | 100000 | nu uniform [1e-3, 40], z log-uniform [1e-12, 1] | 201/202 |
| `near_int` | 100200 | nu = n + d for n = 1..40 and 15 offsets d in {0, +-1e-2, +-1e-4, ..., +-1e-14}, each at least `4 ulp(n)` in magnitude; 167 z log-spaced over [1e-3, 40] | none |
| `half_int` | 100000 | nu = p + 0.5 for p = 0..39 in blocks of 2500 consecutive points per order; z log-uniform [1e-3, z_max(p)] | 301 |
| `large_z` | 100000 | nu uniform [1e-3, 100], z uniform [20, 800] | 401/402 |
| `large_nu` | 150000 | nu log-uniform [20, 1000], r log-uniform [1e-4, 10], z = nu r (up to 1e4) | 501/502 |
| `seam` | 100000 | 4000 anchors on the router borders x 25 points (Section 5) | none |
| `size_random` | 2^20 | region-balanced mixture, nu, z in [1e-10, 40] | 601-604 |
| `size_half` | 2^20 | constant nu = 2.5, z log-uniform [1e-10, 40] | 701 |

A point depends only on `(dataset, i)`: its random draws `unif(i, seed)` are the splitmix64
output (mixing) function of `i + 0x9E3779B97F4A7C15 * (seed + 1)`, independent of thread
order and of the range of indices generated together. `log-uniform` and `uniform` map that
draw to the stated interval. `grid`, `near_int` and `seam` are deterministic layouts.

### `half_int`

`p + 0.5` is exactly representable and is stored exactly. Every backend evaluates a
half-integer point with the half-integer closed form `M = e^-z P_p(z)` where the order is
within the backend's cap, whatever its neighbours. The block length 2500 exceeds
`HALF_INTEGER_MIN_RUN = 256` (`src/bessel/half_integer_common.h`), so on the vector backends
every block also takes the direct path, which skips the router:

| backend | closed form | Debye |
|---|---|---|
| scalar (`HALF_INTEGER_SCALAR_CAP` = 40) | p = 0..39, all 100000 points | none |
| AVX-512, AVX2, CUDA (`HALF_INTEGER_VECTOR_CAP` = 29) | p = 0..29, 75000 points | p = 30..39, 25000 points |

At p = 30..39 the same points are evaluated by the closed form on one backend and by the
uniform-order expansion on the others. The dataset does not reach the large-argument, Levin,
continued-fraction or small-argument regions.

The log-uniform z axis puts about half of each block below z = 1. Its upper limit `z_max(p)`
is the z at which M falls to `DBL_MIN`: 708.396 at p = 0, where M = e^-z and the limit
equals `MATERN_EXP_RANGE_LIMIT = 708.396418532264`, rising to 840.7 at p = 39. For p >= 1
the limit lies above that constant: `e^-z` leaves the normal range at 708.396, but the
product `e^-z P_p(z)` remains representable while `P_p(z)` grows. The largest z drawn in a
block is 705.8 at p = 0 and 837.2 at p = 39. From p = 1 on, 708 points, in 38 of the 40
blocks (none at p = 0 and p = 2), lie above 708.396 and exercise the exp-log evaluation
`M = exp(-z + ln P_p(z))` of the closed form: all 708 on the scalar backend, and the 431 at
p <= 29 on the others, which evaluate p = 30..39 with the Debye expansion.

### `size_random` and `size_half`

Each is the first `N = 2^20` points of a `2^25`-point dataset, `size_random_large` and
`size_half_large`, with the same sampling and seeds; a ladder of sizes `2^k` is read as
prefixes. The 2^20 datasets have no files of their own: the corpus reader reads them as that
prefix. Both samplings are independent and identically distributed per point, so every
prefix is a sample of the whole distribution.

`size_random` draws each axis from a three-piece mixture, the piece chosen from the point's
own random stream:

| axis | weight | piece |
|---|---|---|
| nu | 0.50 | log-uniform [1e-10, 15) |
| | 0.30 | uniform [15, 30) |
| | 0.20 | uniform [30, 40] |
| z | 0.25 | log-uniform [1e-10, 1) |
| | 0.50 | log-uniform [1, 22) |
| | 0.25 | log-uniform [22, 40] |

With the borders of `src/router/region_partition.h`, each of the five regions (small
argument, continued fraction, Levin, large argument, Debye) receives between 17.9 % and
22.0 % of the points; a single log-uniform draw over the same range would put about 85 % in
the small-argument region. The breakpoints are fixed numbers, not router constants, so the
workload does not change when a border moves. The mixture is a design choice: the ladder
reports throughput on a region-balanced workload.

`size_half` has constant order, so the whole array is one run and the half-integer closed form
serves every prefix length.

`LADDER_SET=small` (default of `benchmark/run_benchmark.sh`) ladders over these two datasets,
up to `2^20`; `LADDER_SET=large` (default of `benchmark/run_gpu.sh`) over `size_random_large`
and `size_half_large`, up to `2^25`. The `2^25` files (`*_size_*_large_33554432.bin`, 5.1 GB
for both families) are part of the corpus like the others.

## 4. Reference computation

The reference is `M = 2^(1-nu) / Gamma(nu) * z^nu * K_nu(z)`, evaluated in arb (FLINT) with
adaptive precision: from 512 bits, doubling up to a ceiling of 65536, until the ball holds at
least 110 accurate bits. A point still short of 110 bits at the ceiling fails the generation;
no stored point is short (see the `bits` column below). For `h_nu_s`, `z = sqrt(2 nu) h / s`
is formed in arb at the working precision of each attempt; `h`, `nu` and `s` enter exactly.

The 110-bit bar makes the low word meaningful: with only 60 accurate bits, `lo` (of order
`2^-54 |M|`) would carry about seven correct bits. Cancellation in arb's evaluation costs
about `2 z log2(e)` bits, which the adaptive precision absorbs.

The stored references were checked by the authors' recomputation program (not published),
which recomputes `M` from each file's own axes to a 250-bit target: on every point of
`large_z` and `large_nu`, the datasets with results below the smallest subnormal, and on
every 10th to 50th point of the other datasets. The first three rows are measurements on the
points checked, not bounds:

| property | result |
|---|---|
| `\|hi + lo - M\|` | largest measured 4.4e-16 ulp(hi) where `hi` >= 2^-972 (about 1.6e-293). Below that, a property of the format: `lo` is a multiple of the smallest subnormal, 2^-1074, so `hi + lo` resolves `M` only to that step; half a step, 2^-1075, is 0.5 ulp(hi) where `hi` is subnormal and `lo` therefore 0 (the `subnormal` column of the census in `MANIFEST.md`) |
| `\|hi - M\|` | largest measured 0.5000 ulp; no checked high word is incorrectly rounded |
| incorrectly rounded high words | 0 among the points checked |
| minimum accurate bits (`bits` column) | at least 110 in every dataset (the stored column, every point) |

Relative error is meaningful only where `M` is representable as a nonzero double. Points with
`hi == 0` (`M` at most 2^-1075, half the smallest subnormal) occur only in `large_z` and
`large_nu`; their references are still accurate (the `bits` column holds), only their double
image is 0. The `bits` column, not the magnitude of `M`, identifies an unscorable reference;
every point in the corpus has at least 110.

## 5. `seam`

`seam` places points on both sides of every border of the region router
(`src/router/region_partition.h`) and of the half-integer dispatch caps. Anchors are derived
from the router constants. For each anchor the dataset holds the anchor itself and the points
at `+-k ulp` of the border coordinate for `k` in {2, 4, 8, ..., 4096}: 25 points per anchor,
4000 anchors. Offsets of 0 and 1 ulp on the off-border sides are excluded, because the
`h_nu_s` argument lies about 0.5 ulp from the `z_nu` argument and such a point could fall on
different sides of the border in the two families. The anchor itself (k = 0) lies on the
border and may be classified to either side.

Each region-border anchor was validated with `matern_region_of`: a step of 2 ulp changes the
region at all 3700 of them. The remaining 300 anchors (border class 9) lie on the
half-integer caps, which select between the closed form and the router rather than between
two regions.

`<family>-label_seam_100000.txt` has one line per point: `class axis sign k`, where `class` is
the border class of the table below, `axis` is the coordinate offset (`z` or `nu`), `sign` is
-1, 0 or 1 and `k` is the offset in ulp.

| class | border | offset axis | anchors |
|---|---|---|---|
| 0 | `z = SMALL_ARGUMENT_Z_CEILING = 1`, nu in [1e-3, 26] | z | 500 |
| 1 | `z = LARGE_ARGUMENT_Z_FLOOR = 22`, nu in [12, 30] | z | 500 |
| 2 | `z = nu^2/36 + 18`, nu in [1e-3, 12] | z | 500 |
| 3 | Levin lower border `z = nu^4/2401 + 1.5`, nu in [1e-3, 14.85] | z | 500 |
| 4 | `nu = LARGE_ARGUMENT_ORDER_CAP = 30`, z in [1, 700] | nu | 400 |
| 5 | `nu = SMALL_ARGUMENT_ORDER_MAX = 26`, z in [1e-3, 1) | nu | 300 |
| 6 | `nu = GAMMA_RECURRENCE_ORDER = 8`, z in [22, 700] | nu | 300 |
| 7 | Levin lower border, z in [2, 21.9] | nu | 300 |
| 8 | `z = MATERN_EXP_RANGE_LIMIT = 708.396418532264`, nu in [1e-3, 40] | z | 400 |
| 9 | half-integer caps nu = 29.5 and nu = 40.5, z in [1e-3, 700] | nu | 300 |

The Levin band closes at nu = 14.8949, where `nu^4/2401 + 1.5` reaches 22; the router's
`nu <= 15` limit on the band therefore never binds, and classes 3 and 7 stop below it.
Class 4 starts at z = 1 and class 0 stops at nu = 26 because below z = 1 and above nu = 26
both sides of those lines are served by the Debye expansion.

## 6. Router constants the dataset design depends on

| constant | value | consequence |
|---|---|---|
| `LARGE_ARGUMENT_ORDER_CAP` | 30 | above it the router serves every z with the Debye expansion |
| large-argument border | `min(nu^2/36 + 18, LARGE_ARGUMENT_Z_FLOOR = 22)` for nu <= 30 | a curve on (0, 12], the line z = 22 on [12, 30] |
| `SMALL_ARGUMENT_Z_CEILING` | 1 | below it, Temme's series |
| `SMALL_ARGUMENT_ORDER_MAX` | 26 | above this order, Debye instead of Temme |
| `GAMMA_RECURRENCE_ORDER` | 8 | boundary of the low-order sort class |
| `EXTENDED_RANGE_ORDER_SPLIT` | 30 | selects the assembly used past the exponent range |
| `MATERN_EXP_RANGE_LIMIT` | 708.396418532264 | `-log(DBL_MIN)`; past it, the extended-range bucket |
| Levin band | nu <= 15 and `z > nu^4/2401 + 1.5` | closes at nu = 14.8949 |
| `HALF_INTEGER_MIN_RUN` | 256 | vector backends: shorter runs go through the router, which serves their half-integer points with the same closed form (speed only; the scalar backend has no threshold) |
| `HALF_INTEGER_VECTOR_CAP` / `HALF_INTEGER_SCALAR_CAP` | 29 / 40 | closed form for nu <= 29.5 (AVX-512, AVX2, CUDA) or nu <= 40.5 (scalar) |
