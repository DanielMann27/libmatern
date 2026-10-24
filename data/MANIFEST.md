# Manifest of the evaluation corpus

Every file of the corpus, with the census, the precision verification and the sha256 of
each. Sampling, formats and the reference
computation are specified in `DATASETS.md`.

File names carry their family (`z_nu-`, `h_nu_s-`; the `boost_shortcut` files carry the
dataset name), so they are unique across the folders: the corpus can be kept in the folders
`z_nu/`, `h_nu_s/`, `boost_shortcut/` or with all files directly in the corpus root. The
readers look in the folder first, then in the root.

## `z_nu/`: the `(z, nu)` direct-argument family

Ten datasets, 3 259 496 points: eight in 25 files (38.7 MiB), and `size_random` and
`size_half`, the first 2^20 points of the 2^25-point datasets `size_random_large` and
`size_half_large` (6 files, 2.28 GB). The arguments are stored directly, so a kernel is
scored without the `z * eps` error of deriving `z`.

Files per dataset: `z_nu-nu_<dataset>_<N>.bin`, `z_nu-z_<dataset>_<N>.bin`,
`z_nu-ref_<dataset>_<N>.bin`; `seam` also has `z_nu-label_seam_100000.txt`. `size_random` and
`size_half` are read from the files of their `_large` datasets.

### Census

| dataset | N | sampling | min acc bits | M == 0 | subnormal | 0 < \|M\| < 1e-40 | seeds |
|---|---|---|---|---|---|---|---|
| `grid` | 262144 | `512 x 512` linear tensor, nu [1e-3,20] x z [1e-3,140] | 110 | 0 | 0 | 42187 (16.1 %) | none |
| `random` | 250000 | uniform x uniform, nu, z in [1e-3,40] | 379 | 0 | 0 | 0 | 101/102 |
| `small_z` | 100000 | nu uniform [1e-3,40], z log [1e-12,1] | 497 | 0 | 0 | 0 | 201/202 |
| `near_int` | 100200 | 40 integers x 15 offsets (0, +-1e-2..+-1e-14, at least 4 ulp) x 167 z log [1e-3,40] | 344 | 0 | 0 | 0 | none |
| `half_int` | 100000 | nu = p+0.5, p = 0..39, blocks of 2500 per order, z log [1e-3, z_max(p)] | 255 | 0 | 0 | 13057 (13.1 %) | 301 |
| `large_z` | 100000 | uniform, nu [1e-3,100] x z [20,800] | 116 | 432 | 901 | 80260 (80.3 %) | 401/402 |
| `large_nu` | 150000 | nu log [20,1000], r = z/nu log [1e-4,10], z = nu r | 255 | 6483 | 257 | 15353 (10.2 %) | 501/502 |
| `seam` | 100000 | 4000 anchors x 25 points on the router borders | 255 | 0 | 5875 | 17075 (17.1 %) | none |
| `size_random` | 1048576 | ladder, region-balanced mixture, nu, z in [1e-10,40] | 358 | 0 | 0 | 0 | 601-604 |
| `size_half` | 1048576 | ladder, constant nu = 2.5, z log [1e-10,40] | 390 | 0 | 0 | 0 | 701 |

`M == 0` counts points whose reference high word is 0 (true `M` at most 2^-1075, half the
smallest subnormal); `subnormal` counts nonzero high words below `DBL_MIN`. The last magnitude
column counts nonzero references, subnormals included, below 1e-40; it is a census only.
Scoring should select points by the `bits` column, which is at least 110 everywhere, and by
whether `M` is representable, not by a magnitude cut.

### Formats

| file | content |
|---|---|
| `z_nu-nu_`, `z_nu-z_` | one double per point |
| `z_nu-ref_` | `hi`, `lo` (doubles) and `bits` (16-bit integer), three arrays |
| `z_nu-label_seam_` | text, `class axis sign k` per line |

The reference value is `hi + lo` (about 106 bits); `bits` is `arb_rel_accuracy_bits`. The
sentinels 32767 (exact ball) and -32768 (zero midpoint) do not occur.

### Precision

Checked by the authors' recomputation program (not published), which recomputes `M` from
the stored axes to a 250-bit target, on every point of `large_z` and `large_nu` and on every
10th to 50th point of the other datasets; the first three rows are measurements on the points
checked, not bounds:

| property | result |
|---|---|
| `\|hi + lo - M\|` | largest measured 4.4e-16 ulp(hi) where `hi` >= 2^-972 (about 1.6e-293). Below that, a property of the format: `lo` is a multiple of the smallest subnormal, 2^-1074, so `hi + lo` resolves `M` only to that step; half a step, 2^-1075, is 0.5 ulp(hi) where `hi` is subnormal and `lo` therefore 0 (the `subnormal` column of the census) |
| `\|hi - M\|` | largest measured 0.5000 ulp; no checked high word is incorrectly rounded |
| incorrectly rounded high words | 0 among the points checked |
| minimum accurate bits | at least 110 (the stored column, every point) |

### `seam`

Anchors are derived from the constants in `src/router/region_partition.h`. At all 3700
region-border anchors a step of 2 ulp changes the region returned by `matern_region_of`; the
other 300 anchors lie on the half-integer dispatch caps nu = 29.5 and 40.5. The border
classes of the labels are listed in `DATASETS.md`, Section 5.

### Matched pair

`z_nu-nu_<dataset>_<N>.bin` and `z_nu-z_<dataset>_<N>.bin` are byte-identical to
`h_nu_s-nu_<dataset>_<N>.bin` and `h_nu_s-h_<dataset>_<N>.bin`, for all ten datasets.

## `h_nu_s/`: the `(h, nu, s)` public-API family

Ten datasets, 3 259 496 points: eight in 33 files (47.5 MiB), and `size_random` and
`size_half`, the first 2^20 points of the 2^25-point datasets `size_random_large` and
`size_half_large` (8 files, 2.82 GB). The arguments are those of the public entry, which
derives `z = sqrt(2 nu) h / s`, so a score includes the derivation error.

Files per dataset: `h_nu_s-nu_<dataset>_<N>.bin`, `h_nu_s-h_<dataset>_<N>.bin`,
`h_nu_s-s_<dataset>_<N>.bin`, `h_nu_s-ref_<dataset>_<N>.bin`; `seam` also has
`h_nu_s-label_seam_100000.txt`. `size_random` and `size_half` are read from the files of
their `_large` datasets. The scale parameter `s` is stored in the `h_nu_s-s_` files.

### Matched pair

`s = fl(sqrt(2 nu))` and `h = z` exactly, so `h_nu_s-nu_<dataset>_<N>.bin` and
`h_nu_s-h_<dataset>_<N>.bin` are byte-identical to `z_nu-nu_<dataset>_<N>.bin` and
`z_nu-z_<dataset>_<N>.bin`, for all ten datasets. The reference derives `z` from the stored
`(h, nu, s)` in arb at the working precision of each attempt, so it is the covariance at
exactly these inputs; it differs from the `z_nu` reference because `z_ref` differs from the
stored `z` by the rounding of `s`.

### Census

| dataset | N | sampling | min acc bits | M == 0 | subnormal | 0 < \|M\| < 1e-40 | seeds |
|---|---|---|---|---|---|---|---|
| `grid` | 262144 | `512 x 512` linear tensor, nu [1e-3,20] x z [1e-3,140] | 110 | 0 | 0 | 42187 (16.1 %) | none |
| `random` | 250000 | uniform x uniform, nu, z in [1e-3,40] | 377 | 0 | 0 | 0 | 101/102 |
| `small_z` | 100000 | nu uniform [1e-3,40], z log [1e-12,1] | 497 | 0 | 0 | 0 | 201/202 |
| `near_int` | 100200 | 40 integers x 15 offsets (0, +-1e-2..+-1e-14, at least 4 ulp) x 167 z log [1e-3,40] | 342 | 0 | 0 | 0 | none |
| `half_int` | 100000 | nu = p+0.5, p = 0..39, blocks of 2500 per order, z log [1e-3, z_max(p)] | 254 | 0 | 0 | 13057 (13.1 %) | 301 |
| `large_z` | 100000 | uniform, nu [1e-3,100] x z [20,800] | 113 | 432 | 901 | 80260 (80.3 %) | 401/402 |
| `large_nu` | 150000 | nu log [20,1000], r = z/nu log [1e-4,10], z = nu r | 254 | 6483 | 257 | 15353 (10.2 %) | 501/502 |
| `seam` | 100000 | 4000 anchors x 25 points on the router borders | 254 | 0 | 5875 | 17075 (17.1 %) | none |
| `size_random` | 1048576 | ladder, region-balanced mixture, nu, z in [1e-10,40] | 356 | 0 | 0 | 0 | 601-604 |
| `size_half` | 1048576 | ladder, constant nu = 2.5, z log [1e-10,40] | 388 | 0 | 0 | 0 | 701 |

`M == 0` counts points whose reference high word is 0 (true `M` at most 2^-1075, half the
smallest subnormal); `subnormal` counts nonzero high words below `DBL_MIN`. The last magnitude
column counts nonzero references, subnormals included, below 1e-40; it is a census only.
Scoring should select points by the `bits` column, which is at least 110 everywhere, and by
whether `M` is representable, not by a magnitude cut.

### Formats

| file | content |
|---|---|
| `h_nu_s-nu_`, `h_nu_s-h_`, `h_nu_s-s_` | one double per point |
| `h_nu_s-ref_` | `hi`, `lo` (doubles) and `bits` (16-bit integer), three arrays |
| `h_nu_s-label_seam_` | text, `class axis sign k` per line |

The reference value is `hi + lo` (about 106 bits); `bits` is `arb_rel_accuracy_bits`. The
sentinels 32767 (exact ball) and -32768 (zero midpoint) do not occur.

### Precision

Checked by the authors' recomputation program (not published), which recomputes `M` from
the stored axes to a 250-bit target, on every point of `large_z` and `large_nu` and on every
10th to 50th point of the other datasets; the first three rows are measurements on the points
checked, not bounds:

| property | result |
|---|---|
| `\|hi + lo - M\|` | largest measured 4.4e-16 ulp(hi) where `hi` >= 2^-972 (about 1.6e-293). Below that, a property of the format: `lo` is a multiple of the smallest subnormal, 2^-1074, so `hi + lo` resolves `M` only to that step; half a step, 2^-1075, is 0.5 ulp(hi) where `hi` is subnormal and `lo` therefore 0 (the `subnormal` column of the census) |
| `\|hi - M\|` | largest measured 0.5000 ulp; no checked high word is incorrectly rounded |
| incorrectly rounded high words | 0 among the points checked |
| minimum accurate bits | at least 110 (the stored column, every point) |

### `seam`

The seam points and labels are those of `z_nu/`, with `h = z` and `s = sqrt(2 nu)`; the
border classes are listed in `DATASETS.md`, Section 5.

## `boost_shortcut/`: K_nu at nu = 2.5 + delta

One dataset of 873 points: nine orders nu = 2.5 + delta, delta in {0, 1 ulp, 2 ulp, 1e-12, 1e-9,
1e-6, 1e-4, 1e-3, 1e-2} (the realised delta is `nu - 2.5`, exact), crossed with 97 values of z
log-spaced over [1e-10, 300].

Boost.Math 1.86 and later evaluates K_v with a four-term Hankel series when
`|4v^2 - 25| / (8x) < eps^(1/4)`. The series is exact at v = 5/2 and inaccurate beside it, by
up to about 4e-5 in double precision. The other datasets either draw nu = 2.5 exactly or reach
this neighbourhood only by chance, so this dataset sweeps delta explicitly. It is read by
`benchmark/boost_shortcut_gpu.cu` (`make build/boost_shortcut_gpu`, stage
`gpu-boost-shortcut` of `benchmark/run_gpu.sh`).

| file | content |
|---|---|
| `nu_boost_shortcut_873.bin` | nu, one double per point |
| `z_boost_shortcut_873.bin` | z, one double per point |
| `refK_boost_shortcut_873.bin` | `hi`, `lo`, `bits` per point |

The binary format is `benchmark/corpus_bin.h` (libmatern); the generator (the program that drew the
points and computed the references; not published) printed these values with `%a`, which
the header records, so `build/corpus_convert to-text` (built by `make corpus-tools`)
reproduces that text.

The reference is K_nu(z) itself, not the covariance. `hi` is the arb midpoint rounded to the
nearest double, `lo` the double nearest `K - hi`, and `bits` is `arb_rel_accuracy_bits` at
the precision that reached at least 110 bits (minimum 110). Every value is a normal double,
from about 1e25 at z = 1e-10 to about 1e-131 at z = 300.

## sha256

`make corpus-check` (`tools/corpus_check.sh` in libmatern) checks every corpus file present
against this list, in either layout; every benchmark run does the same and records the
result in its `MANIFEST.txt`.

```
b867fc90e680a6bb196f73bbbf3906c77680c6138623d1d88a2c51bf639d19f2  z_nu-nu_grid_262144.bin
f1a41c1b36639a933af0d6824a57caec0d9caac5f69829b035cd55027906f2a4  z_nu-nu_half_int_100000.bin
e902cb50c43e7bb6e80f44fa4793ca7722ced2dd82f7e1ada0526b62615198f3  z_nu-nu_large_nu_150000.bin
0007407fa8c73ee774bb8ad3a6ea99a57b18f296ec85512d23a6d3b9538a31ab  z_nu-nu_large_z_100000.bin
c602a8ba58aa03b811ad53ef9836dcedbbaf29dea68562c39f2c2175878cb850  z_nu-nu_near_int_100200.bin
fb9e893c7c312dd6a1a6e287b51d60569ace40b8d38c424f9b0e254630148d72  z_nu-nu_random_250000.bin
1a3753e945cd780a75702462ef77eacf2bbffc5d74b165b69f0391e1d709c88b  z_nu-nu_seam_100000.bin
df2c99fa84c7712fc35da3724d3dddd8c46cd7290cf44d98e2157f5d08869d09  z_nu-nu_size_half_large_33554432.bin
7db85bf9f338e37e9df5c4db71f89db55ce2ab3f14d3ec6a135f85d6df4e3a95  z_nu-nu_size_random_large_33554432.bin
b8d7a8cf23587f29a33d2a9f0c33c0166b729ea97088b080be1b03e489852c7e  z_nu-nu_small_z_100000.bin
47b69bf2a36ed1df0360de736cabd0f770237dc9e3a9099d60c1d696a31c6ce3  z_nu-ref_grid_262144.bin
f5feb0fafad52eb2b5a14f8138699b2a498f7efa45a913f0fa77156320fbef42  z_nu-ref_half_int_100000.bin
967404a8c8aafaee1649b87f5b238eaf9b11b1b9136d1d0604e2e45cff8989da  z_nu-ref_large_nu_150000.bin
5f795ff0599565ecc3c165e5b2accce16fd73d68d7a2565f45ad251592841bad  z_nu-ref_large_z_100000.bin
090489808d1bba78f014d791f3d0a50b60755fbe0370dd6b79ef76a4e791b88c  z_nu-ref_near_int_100200.bin
19eb6b52aee618ad754fffbdbbbefaeb80e67aa5c6d06815e6fe81d67ec60c6c  z_nu-ref_random_250000.bin
258584effaefd7beb717a3e4817fb1ea2104931c638e431c54656a4803902c7c  z_nu-ref_seam_100000.bin
1d97658c0cc8137ecbd0d67125b67e49b7d2bf2a72b33de084b30aa6ab905afe  z_nu-ref_size_half_large_33554432.bin
1e3d1fe2f429acdf9afee43f0cb64c6c94c03a639d75528c953c9be1ec7c4d0a  z_nu-ref_size_random_large_33554432.bin
a30eda7c892c6bde9e090263bfd74c4b33e8a13426eff5e213a94f9ed860bf6e  z_nu-ref_small_z_100000.bin
ff6d59a289008f6fe695b91cbf0d76f43cb99fb08ecb04fa85d5892a3d0bf1e4  z_nu-z_grid_262144.bin
8dd2a1f0ebea7d3fcf6d66a6e453d40088e7e2c348eaafe7a6bc4a5bbc64e50c  z_nu-z_half_int_100000.bin
19fc947d9629f13ec109128b07e15cca938b78121369da5fe439df58bcf6bc10  z_nu-z_large_nu_150000.bin
fe246f73d03096962a8f382790c9f73998a66b3ed27ca7be20ed6563a5048a50  z_nu-z_large_z_100000.bin
7e510756fd45bd3a3abe6a31fb27f75cdf8243b1ea3200a955e69b96df97f648  z_nu-z_near_int_100200.bin
c9173cfd697000ef2d6e37ebde7e732ca0445a8e4106135f6a1e0a756604abd6  z_nu-z_random_250000.bin
44dd2d5de4e030e9c6f14ea1df54ae4802b8e1e62497bd17c673bb8550fbee51  z_nu-z_seam_100000.bin
55e75fc655068c4b8452e7a3ba49b33408a0b55e1f0d332c9854a35afe470b78  z_nu-z_size_half_large_33554432.bin
44741d6f200f9a6beace8e0277b4cc829aa4620103046cd85e074418eec4ef26  z_nu-z_size_random_large_33554432.bin
98b618f8a51c563bd1ac8843fe54a0158aad22a44dbea23f39987a940827abce  z_nu-z_small_z_100000.bin
559bc02031074986c66a2b59c5c12f39ac820d43cf60cdba26811efed07efef0  z_nu-label_seam_100000.txt
ff6d59a289008f6fe695b91cbf0d76f43cb99fb08ecb04fa85d5892a3d0bf1e4  h_nu_s-h_grid_262144.bin
8dd2a1f0ebea7d3fcf6d66a6e453d40088e7e2c348eaafe7a6bc4a5bbc64e50c  h_nu_s-h_half_int_100000.bin
19fc947d9629f13ec109128b07e15cca938b78121369da5fe439df58bcf6bc10  h_nu_s-h_large_nu_150000.bin
fe246f73d03096962a8f382790c9f73998a66b3ed27ca7be20ed6563a5048a50  h_nu_s-h_large_z_100000.bin
7e510756fd45bd3a3abe6a31fb27f75cdf8243b1ea3200a955e69b96df97f648  h_nu_s-h_near_int_100200.bin
c9173cfd697000ef2d6e37ebde7e732ca0445a8e4106135f6a1e0a756604abd6  h_nu_s-h_random_250000.bin
44dd2d5de4e030e9c6f14ea1df54ae4802b8e1e62497bd17c673bb8550fbee51  h_nu_s-h_seam_100000.bin
55e75fc655068c4b8452e7a3ba49b33408a0b55e1f0d332c9854a35afe470b78  h_nu_s-h_size_half_large_33554432.bin
44741d6f200f9a6beace8e0277b4cc829aa4620103046cd85e074418eec4ef26  h_nu_s-h_size_random_large_33554432.bin
98b618f8a51c563bd1ac8843fe54a0158aad22a44dbea23f39987a940827abce  h_nu_s-h_small_z_100000.bin
b867fc90e680a6bb196f73bbbf3906c77680c6138623d1d88a2c51bf639d19f2  h_nu_s-nu_grid_262144.bin
f1a41c1b36639a933af0d6824a57caec0d9caac5f69829b035cd55027906f2a4  h_nu_s-nu_half_int_100000.bin
e902cb50c43e7bb6e80f44fa4793ca7722ced2dd82f7e1ada0526b62615198f3  h_nu_s-nu_large_nu_150000.bin
0007407fa8c73ee774bb8ad3a6ea99a57b18f296ec85512d23a6d3b9538a31ab  h_nu_s-nu_large_z_100000.bin
c602a8ba58aa03b811ad53ef9836dcedbbaf29dea68562c39f2c2175878cb850  h_nu_s-nu_near_int_100200.bin
fb9e893c7c312dd6a1a6e287b51d60569ace40b8d38c424f9b0e254630148d72  h_nu_s-nu_random_250000.bin
1a3753e945cd780a75702462ef77eacf2bbffc5d74b165b69f0391e1d709c88b  h_nu_s-nu_seam_100000.bin
df2c99fa84c7712fc35da3724d3dddd8c46cd7290cf44d98e2157f5d08869d09  h_nu_s-nu_size_half_large_33554432.bin
7db85bf9f338e37e9df5c4db71f89db55ce2ab3f14d3ec6a135f85d6df4e3a95  h_nu_s-nu_size_random_large_33554432.bin
b8d7a8cf23587f29a33d2a9f0c33c0166b729ea97088b080be1b03e489852c7e  h_nu_s-nu_small_z_100000.bin
6151f3058e98774056082c55cb3fee7529738585ab0216ca40f9660919bcc795  h_nu_s-ref_grid_262144.bin
dc29bb8985c7ec0f98ea2bb38436ff042de5ac6a3f4ff7d21f20fbe75e7e2c63  h_nu_s-ref_half_int_100000.bin
1f3ee9657289432980eca631a3017b1fc23b9eb6c2eb985150ce3884b3243905  h_nu_s-ref_large_nu_150000.bin
59a6468635aee3e2313321f95df6efd5da91e81fd24fc12297eceb144cbd5104  h_nu_s-ref_large_z_100000.bin
d6868da3f4a65127ce55c4cc9c7e75e19ef76a4a428fbb22ccbff436e9bc5153  h_nu_s-ref_near_int_100200.bin
aa6e90b9d7df82b54925f7aa59e3a9a17808de04493397c9a84c7e703228d08c  h_nu_s-ref_random_250000.bin
6357d331e0d50cea4853773b8479274c5bfbd70fd03439613905765beb826bd7  h_nu_s-ref_seam_100000.bin
903e5af638c84175dacdd325da14694697e2c126120e3fcc2dca9491cc9d8241  h_nu_s-ref_size_half_large_33554432.bin
760d8c1225000943970c992ce259fab7a0dfa595d75c86416a70536db8f525d3  h_nu_s-ref_size_random_large_33554432.bin
a3bd63c8df128c1d3c6d9615abf22ece0102e8411517c67eed136fa73e881ca1  h_nu_s-ref_small_z_100000.bin
06620303f22b74ae96584d98ff405a22efb7dffcfce8ecfe4d42dfc358de0aa7  h_nu_s-s_grid_262144.bin
e486ca16cb91a456462c75afcea807a971284aa6762f1ccf98d9b009b86162e3  h_nu_s-s_half_int_100000.bin
6da28c916c26f454edf363708954d9ba88e713b1fff987ba8e82f452c5f18762  h_nu_s-s_large_nu_150000.bin
75e24e48c60da9028e46c2146f893ff880a9467383d1098eb5a2c7ce393059e5  h_nu_s-s_large_z_100000.bin
749afca1ae9f4122956146fe11d366271c0524858969d49d5ee970fa4ab400a1  h_nu_s-s_near_int_100200.bin
f51dbcec4cc9ea66ac16bbafa7a78e7843e808734cb20f3389718f08e86724c7  h_nu_s-s_random_250000.bin
5a8c92abe0e21be070f9486921dcc066b8df7d03b40253cd9425e6415183899f  h_nu_s-s_seam_100000.bin
4ec836a5d9277b40491cb9938ce43e2e456df0de5ebb039aa9a28e9f96c9b4ff  h_nu_s-s_size_half_large_33554432.bin
7bcff7017bc940759e92891d1d79c4eeb49dd859ab9a34e75b78c7c4bbd9c2fc  h_nu_s-s_size_random_large_33554432.bin
b902d0b4ccaa0867c53f8ad7f26801f9934d79fb746e2f5be76474a40ee38346  h_nu_s-s_small_z_100000.bin
559bc02031074986c66a2b59c5c12f39ac820d43cf60cdba26811efed07efef0  h_nu_s-label_seam_100000.txt
77dc087873249e889960afeb8f06314248a5d1eed8a39498cd86a306a1ce0682  nu_boost_shortcut_873.bin
d854ec4f883db160fb8355c26ef43b2e464a2b8b429abb699ece9008a34cd037  refK_boost_shortcut_873.bin
e997ec04707361bf9568da70bd4fad0ce5a0f6da9e790d4f8275b11af5c11376  z_boost_shortcut_873.bin
```
