# Evaluation corpus of libmatern

Arguments and high-precision references for the normalised Matérn covariance
`M(h) = 2^(1-nu)/Gamma(nu) z^nu K_nu(z)`, `z = sqrt(2 nu) h / s`, used by the benchmark of
libmatern (https://github.com/DanielMann27/libmatern) and the accompanying paper. The corpus
is published on Zenodo, doi:10.5281/zenodo.23393956, under CC-BY-4.0; cite the
DOI when you use the data.

The record holds the 75 data files (5.19 GB) individually under their file names (the record
has no folders; the family prefix of a name gives its folder below), and the documents
`README.md` (this description), `DATASETS.md` (the specification: sampling, seeds, reference
computation, seam construction) and `MANIFEST.md` (census and precision of every dataset,
and the sha256 of every file).

In libmatern, `make corpus DOWNLOAD=yes` downloads the missing files into their folders of
`data/` and checks each against `MANIFEST.md`; `make corpus-check` checks the files present.
Without libmatern, compare `sha256sum` of each downloaded file with `MANIFEST.md`.

## Datasets

| folder | content | size |
|---|---|---|
| `z_nu/` | the `(z, nu)` family, files `z_nu-*`: 8 datasets, 1 162 344 points, 25 files | 38.7 MiB |
| `h_nu_s/` | the `(h, nu, s)` family, files `h_nu_s-*`: the same 8 datasets, 33 files | 47.5 MiB |
| `z_nu/`, `h_nu_s/`: `*_size_*_large_33554432.bin` | the 2^25-point datasets `size_random_large` and `size_half_large` of both families, 14 files (6 `z_nu`, 8 `h_nu_s`); their first 2^20 points are the datasets `size_random` and `size_half` | 2.28 + 2.82 GB |
| `boost_shortcut/` | K_nu(z) at nu = 2.5 + delta, 873 points, for a check of Boost's four-term Hankel short cut for K_nu near nu = 5/2 (`MANIFEST.md`, `boost_shortcut`) | 30 kB |

The corpus has two families because deriving `z` from `(h, nu, s)` in double precision adds
about `z * eps` relative error to `M`: `z_nu` scores a kernel alone, `h_nu_s` the public
entry including that derivation. The families are a matched pair: the `h_nu_s-nu_` and
`h_nu_s-h_` files are byte-identical to the `z_nu-nu_` and `z_nu-z_` files, and
`s = sqrt(2 nu)`, stored in the `h_nu_s-s_` files.

## Values

The arguments and references are IEEE-754 doubles (the `bits` column a 16-bit integer; the
seam labels text), exactly the doubles the generator (the program that drew the points and
computed the references; not published) printed in hexadecimal. The references were
computed with FLINT/arb to at least 110 accurate bits and stored as double-doubles: the high
word is the arb midpoint rounded to the nearest double, the low word the remainder rounded
to a double (`DATASETS.md`, Sections 2 and 4).

## Layout

Every file name carries its family (`z_nu-`, `h_nu_s-`; the `boost_shortcut` files carry the
dataset name), so no two files share a name. The readers of libmatern look for a file in its
folder first and then directly in the corpus root, so the files may also be kept flat; both
layouts may be mixed, and a file present in its folder is the one read.

## Formats

| files | content |
|---|---|
| `<family>-nu_`, `-z_`, `-h_`, `-s_<dataset>_<N>.bin` | one double per point |
| `<family>-ref_<dataset>_<N>.bin` | `hi`, `lo`, `bits` per point: reference `hi + lo`, `bits` = accurate bits of the arb ball |
| `<family>-label_seam_100000.txt` | text, `class axis sign k` of each seam point: its border class and offset (`DATASETS.md`, Section 5) |
| `boost_shortcut/`: `nu_`, `z_`, `refK_boost_shortcut_873.bin` | `nu_`, `z_` one double per point; `refK_` as `hi`, `lo`, `bits` |

Each `.bin` file is a 32-byte header and little-endian arrays (`benchmark/corpus_bin.h`;
libmatern's `docs/benchmark.md`, The corpus, gives the byte layout and a reader).
`build/corpus_convert`, built by `make corpus-tools`, converts between the binary files and
the generator's text: `to-text` prints a file, one record per line, `to-bin` converts the text
and refuses any line that would not print back identically, and `verify` compares a binary
file with a text file byte for byte. The generator writes `%.13a` (`boost_shortcut/`: `%a`).

## Use in libmatern

`benchmark/corpus_dataset.c` reads the corpus for the benchmark programs (`make benchmark`,
`benchmark/run_benchmark.sh`); `benchmark/boost_shortcut_gpu.cu` reads `boost_shortcut`.

The throughput ladder times an implementation at sizes 2^k, read as prefixes of one dataset;
by default it reads `size_random_large` and `size_half_large`, and `LADDER_LAWS` names
another dataset. `size_random` and `size_half` have no files of their own: the corpus reader
reads them as the 2^20 prefix of their `_large` datasets. `LADDER_SET=small`, the default of
`benchmark/run_benchmark.sh` and `make benchmark`, ladders up to 2^20; `LADDER_SET=large`,
the default of `benchmark/run_gpu.sh`, up to 2^25; `LADDER_MAX=<N>` caps either lower.
Without the 2^25-point files the size datasets, the default ladders and the thread-scaling
stage at its default dataset (`THREADS_LAW`) cannot run.
