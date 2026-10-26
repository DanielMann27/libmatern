# Benchmark

`benchmark/` measures the accuracy and throughput of the library and compares it with
external implementations of `K_ν`. A run records its results in `benchmark/runs/<run id>/`;
the figure and table generators only parse recorded runs. The experiments, the error
metrics and the timing protocol are described in the accompanying paper (experimental
setup; accuracy and stability metrics; timing methodology).

```sh
make requirements        # what is installed, what it enables, how to add the rest
make reproduce           # the full benchmark behind the published results, with its downloads
make pipeline            # builds, runs, figures and tables in one command
make benchmark-build     # build/detect, build/benchmark_cpu
make build/suppliers     # the external CPU suppliers
make benchmark           # a CPU run with the drivers built, then the figures and tables
make figures             # the paper's figures and captions.tex -> benchmark/plots/paper-final/
make tables              # the paper's tables -> benchmark/tables/paper/
make tables-check        # compiles every generated table with pdflatex
```

Terms used below:

* **backend**: one of the library's CPU implementations `avx512`, `avx2`, `matched_fma`,
  `matched`, `scalar` (`README.md`, Backends); the GPU stages measure the CUDA backend.
* **supplier**: an external implementation of `K_ν`; its **variants** are the values of the
  CSV column `supplier` ([External suppliers](#external-suppliers)).
* **assembly**: how `M` is formed from a supplier's `K_ν`, by the same code for every
  supplier: `direct` (`2^(1-ν)/Γ(ν) · z^ν · K_ν(z)`), `logtrick` (the exp-log assembly of
  the accompanying paper: the logarithms of the factors summed and exponentiated once) or
  `scaled` (from `e^z K_ν(z)`).
* **dataset**: a set of the corpus (`data/DATASETS.md`), named by the option `--law` and the
  CSV column `law`: the accuracy datasets `grid`, `half_int`, `large_nu`, `large_z`,
  `near_int`, `random`, `seam`, `small_z`, and the size datasets `size_random` and
  `size_half`, the first 2^20 points of the 2^25-point datasets `size_random_large` and
  `size_half_large`.
* **family**: the argument convention, `z_nu` for `(z, ν)` or `h_nu_s` for `(h, ν, s)`.
* **ladder**, **rung**: an implementation timed on the first N points of one dataset for
  N = 2^8, 2^9, ... up to the size of the dataset; each N is a rung.
* **stage**: one measurement of a run (`STAGES`, `GPU_STAGES`); step 0 checks the corpus.
* **arm**: one build of the Levin A/B, which measures whether the Levin band (see the
  accompanying paper, domain decomposition) is routed to the Levin-transformed series:
  `base` (the default build), `novec` (vector backends with `-DMATERN_WITHOUT_LEVIN`),
  `scal` (`scalar` with `-DMATERN_SCALAR_WITH_LEVIN`); on the GPU `nolevin` and `levin`.
* **`ship`**, **`all`**: supplier selections; `ship` is Bessels.jl, `std::cyl_bessel_k`,
  Boost and GSL, `all` adds the KAUST rows.

## Requirements

x86-64 Linux, bash 4.2 or later, GNU coreutils and `sed`, `gzip`, a C compiler, libm and
the evaluation corpus ([The corpus](#the-corpus)). The rows of a backend need a CPU with its
instructions (`avx512`: AVX-512 F and DQ, and FMA; `avx2`: AVX2 and FMA; `matched_fma`: FMA
and AVX); the default `BACKENDS` leaves out a backend this CPU cannot run. Downloads need
`curl` or `wget`; the corpus scripts need `sha256sum`, `shasum` or `openssl`, the other
downloads `sha256sum`, and `make boost-headers` needs `bzip2`. Every other component is
optional; its absence removes only what it enables:

| component | enables | how to add |
|---|---|---|
| Python 3 with NumPy and Matplotlib | the figures and tables; without them the generators stop with exit status 2, and `make benchmark` and `make pipeline` record the run and fail | `make externals` (into `.venv/`), or the system packages |
| pdflatex with geometry, amsmath, amssymb, booktabs, array, float and rotating | `make tables` (exit status 2 without pdflatex, 1 on a LaTeX error; with `ARGS=--no-preview` it needs no pdflatex and leaves the tables unfitted) and `make tables-check` | a TeX distribution |
| a C++17 compiler with libstdc++ | the `std::cyl_bessel_k` row (libc++ lacks the special functions) and the Boost CPU row | the system compiler (`CXX`) |
| Boost 1.75.0; Boost 1.87 or later | the Boost CPU row; the Boost GPU rows | `make boost-headers` |
| GSL | the GSL row | `make externals`, or a system GSL found by pkg-config |
| the KAUST port | the KAUST rows | `make kaust DOWNLOAD=yes` |
| `nvcc` and an NVIDIA GPU | the GPU stages | the CUDA toolkit |

`make requirements` reports each component, its version, what it enables and the command or
package that adds it; `MISSING` marks an absence that stops a target, `absent` one that only
removes rows. It changes nothing and exits 0. The generators use the first of
`PYTHON_CANDIDATES` with NumPy and Matplotlib at or above the tested minimum, NumPy 2.0.2
and Matplotlib 3.9.4 (`make python-info`); `PYTHON` overrides it.

## Running the benchmark

`make pipeline` builds what the selected stages need, runs them, writes the figures and
tables and notifies once at the end. It first prints its plan. On a terminal it then asks,
in this order:

1. whether to switch to the settings of the full benchmark, when they differ; it lists the
   differences and the duration of the full run, and where `nvcc` or an NVIDIA GPU is
   missing, or with `GPU=no`, the switch sets the full CPU part;
2. whether to run `make externals`, when an external this run uses is missing and can be
   downloaded; it lists what a yes downloads: Boost 1.75.0 and GSL 2.6 when neither
   `references/` nor the system has that version, Boost 1.87.0 (with a GPU part) when neither
   `references/` nor the system has a Boost with GPU support and `BOOST_ROOT` is not `none`
   ([Boost versions](#boost-versions)), NumPy and Matplotlib when no Python has them at or
   above the tested minimum. A GSL of another version found by pkg-config counts as present, so it does not by
   itself lead to this question, and the gsl row uses it (row `gsl-<version>`,
   `build_cpu_supplier_gsl_*` in `PROVENANCE.txt`) unless GSL 2.6 is built: by a yes to this
   question asked for another external, by `EXTERNALS=yes` or by `make reproduce`;
3. whether to download and port KAUST CuBesselK, after its copyright notice, when a stage
   runs a KAUST row and the port is absent;
4. whether to download the missing or damaged corpus files; without them it stops.

It prints the answers to the questions it asked and then runs unattended: the downloads first, then the builds, whose
commands go to `build/pipeline.<build>.log` (the Levin arms to `build-ab/`; `V=yes` prints
them), then the stages.

A switch on the command line answers a question in advance, and that question is not
asked: `SETTINGS=full` the first, `EXTERNALS=yes` the second, `KAUST=yes` or `KAUST=no`
the third, `CORPUS=yes` the fourth ([Make parameters](#make-parameters)). Without a
terminal on standard input and output, as in a batch job, under `nohup` or behind `| tee`,
it asks nothing and only these switches answer: without them the settings stay as given
(by default a reduced run), no external is downloaded, the KAUST rows run only when the
port exists (`make kaust DOWNLOAD=yes`), and an incomplete corpus stops the pipeline before
anything runs (exit status 2).
`KAUST=no` while the port exists stops the pipeline before anything runs (exit status 2)
when a stage runs a KAUST row, since a run uses the port while it exists; `make kaust
CLEAN=yes` deletes it.
`make reproduce` is `make pipeline SETTINGS=full EXTERNALS=yes CORPUS=yes`; without a
terminal it needs `KAUST=yes` or `KAUST=no` when the port is absent
([Replicating the published results](#replicating-the-published-results)).

`DRY=yes` prints the plan, the full-run notice and the four questions, each as a terminal
would be asked it (for KAUST, also as asked after a switch to the full settings) or as
`not asked` with the answer a switch or the machine gives. It probes the machine (test
compiles in temporary files, the Python, the size and header of the corpus files),
downloads, builds and runs nothing, and leaves the tree unchanged. Its exit status is 2 for
an invalid switch or parameter, for `KAUST=no` while the port exists and a stage runs a
KAUST row, and for `make reproduce` without a terminal when the KAUST port is absent and
neither `KAUST=yes` nor `KAUST=no` is given; else 0. An incomplete corpus, which stops a run
without a terminal, shows in the plan, not in the exit status.

```sh
make pipeline                                       # CPU run, figures, tables
make pipeline LEVIN=yes THREADS=yes THREADS_MAX=8   # with the Levin and thread stages
make pipeline FOLD=yes                              # with the prefactor-fold stage
make pipeline GPU=yes LADDER_SET=small REPS=10      # with the GPU pipeline
make pipeline SETTINGS=full DRY=yes                 # the plan of the full benchmark; runs nothing
```

The pipeline runs `STAGES=all` plus the stages its switches add, and refuses another
`STAGES` and the reduced-run caps. A failed build of the Levin arms, the library or the
benchmark drivers stops it; a failed `build/suppliers` skips the CPU supplier stages, and a
failed `build/fold_cpu` or `build/threads_cpu` skips its stage. A failed stage is recorded
and the others run; a failed download of an external is a warning. Its exit status is
non-zero if a build other than `build/suppliers`, a run, the figures or the tables failed.

`make benchmark` builds the drivers of `make benchmark-build`, not `build/suppliers`, whose
absence it records as skipped supplier stages. `benchmark/run_benchmark.sh` runs the stages
`STAGES` selects, `benchmark/run_gpu.sh` the GPU stages (it builds the GPU drivers). Every
driver (`build/benchmark_cpu`, `build/suppliers`, `build/threads_cpu`, `build/fold_cpu`,
`build/detect` and the GPU drivers) prints its options with `--help`. A reduced run scores
and times part of each dataset (after `make benchmark-build` and `make build/suppliers`):

```sh
# every stage of STAGES=all, ladders up to 2^14 points (needs the 2^25-point files)
REPS=3 LADDER_MAX=16384 SUP_MAX_POINTS=20000 TIME_MAX_POINTS=16384 \
    bash benchmark/run_benchmark.sh
# with only the 61 small corpus files: accuracy and suppliers, ending with STATUS partial
STAGES=rest REPS=3 SUP_MAX_POINTS=20000 TIME_MAX_POINTS=16384 \
    bash benchmark/run_benchmark.sh
```

`make benchmark-test` runs the benchmark's test suite (`benchmark/tests/run_tests.sh`): unit
tests of the harness, reduced runs of `benchmark/run_benchmark.sh` and checks of their
records, and `DRY=yes` runs of `make pipeline` and `make reproduce` (20 minutes or more); it
needs the whole corpus, the 2^25-point files included.
`GPU_STAGES=call-large bash benchmark/run_gpu.sh` times every device kernel as a whole call
through the library's CUDA entry, under the four transfer strategies, on the 2^25-point sets of
both families (timing only, `gpu_call_large.csv`; not part of `GPU_STAGES=all`).

### The prefactor-fold stage

`STAGES=fold` (after `make build/fold_cpu`; `make pipeline FOLD=yes` builds it) compares the
scalar backend, which folds the covariance prefactor into its kernels, with its unfolded
twins, the same sources built with `-DMATERN_SCALAR_UNFOLDED=1` (`scalar-unfolded-direct`:
`2 (z/2)^ν / Γ(ν) · K_ν(z)`) and `-DMATERN_SCALAR_UNFOLDED=2` (`scalar-unfolded-explog`: the
logarithms summed and exponentiated once), and with `scalar-control`, the shipped source
built as the twins are. These are linked only into `build/fold_cpu`, never into
`libmatern.a`. The rows go to `fold.csv`, in the layout of `accuracy.csv`.

## Make parameters

Every parameter is given as `make <target> NAME=value`; most are also read from the
environment. Switches take `yes` or `no` (also `1`/`0`, `on`/`off`, `true`/`false`, `y`/`n`,
in lower or upper case or capitalised); another value stops with an error. `make help`
prints the targets and a summary of this section.

**Build** (the library, `make check`, `make tsan`, `make install`, `make cuda`, the drivers)

| parameter | default | meaning |
|---|---|---|
| `CC`, `CXX` | `cc`, `g++` | the C compiler; the C++17 compiler of the `std::cyl_bessel_k` and Boost rows and of the C++ pass of `make headers` (part of `make` and `make check`; skipped without a C++17 compiler) |
| `CFLAGS` | `-std=c11 -Wall -Wextra -fno-math-errno` | replaces the default flags; `OPTIMISE`, `-ffp-contract=fast` and, for icx, `-fp-model=precise` are added outside it (`README.md`, Building) |
| `OPTIMISE` | `-O2` | the optimisation level of everything built with the library's flags; part of the arithmetic, since it decides whether `a*b + c` is fused into one FMA (GCC at `-O2`, `-O3`, `-Os`; Clang from `-O1`) |
| `XCF` | empty | flags appended after `CFLAGS` and the flags added outside it, which they can override, e.g. `XCF=-DMATERN_WITHOUT_LEVIN` |
| `FMA` | `no` | `yes` builds `scalar_portable` and both entries of `matched` with `-mfma`, which changes their results; the library then needs FMA3 on every machine it runs on (recorded as `fma_knob` in `PROVENANCE.txt`) |
| `TSAN_CC` | `gcc` | the compiler of `make tsan`; its libtsan must be installed |
| `AR` | `ar` | the archiver |
| `PREFIX`, `DESTDIR` | `/usr/local`, empty | the destination of `make install`, and a staging directory put in front of it |
| `NVCC` | `nvcc` on `PATH`, else `/usr/local/cuda/bin/nvcc` | the CUDA compiler |
| `GPU_ARCH` | `sm_80` | the architecture of `make cuda`; the GPU drivers use the one `nvidia-smi` reports, and `GPU_ARCH` only without it |
| `NVCCFLAGS`, `XNVCF` | `-O2 -arch=$(GPU_ARCH) --expt-relaxed-constexpr -Xcompiler -fopenmp`, empty | replaces, or adds to, the flags of `make cuda` and of `build/cuda_bench.o`, the drivers' copy of it with the benchmark hooks |
| `V` | `no` | `yes` prints every build command; otherwise a build prints one line per compiled file and per product, the Levin arms write their commands to `build-ab/build-<arm>.log`, and `make pipeline` its builds to `build/pipeline.<build>.log` |

**Pipeline** (`make pipeline` and `make reproduce`; the run parameters are passed through).
`make reproduce` sets `SETTINGS=full`, `EXTERNALS=yes` and `CORPUS=yes`; another value of
these on the command line is not used, and it says so. `SETTINGS=full` in turn sets
`LEVIN`, `FOLD`, `THREADS` and the run parameters of the full run over values given on the
command line, and adds the GPU part where one can run unless `GPU=no` is given, as a yes to
the first question does; the other switches act as for `make pipeline`.

| switch | default | meaning |
|---|---|---|
| `SETTINGS` | unset | `full` applies the settings of the full benchmark without asking, as a yes to the first question: `LEVIN`, `FOLD` and `THREADS` on, the GPU part where `nvcc` and an NVIDIA GPU are found unless `GPU=no` is given (`GPU=no` is kept, and so is `GPU=yes` where no GPU part can run, which then fails in the GPU part), every run parameter at its value in the full run (`LADDER_SET=large`, `SUPPLIERS=all`, `SUP_LADDER_BUDGET=1500`, the caps unset); `DATA` is kept |
| `CORPUS` | `no` | `yes` downloads the missing or damaged corpus files at the start of the run without asking, as a yes to the fourth question |
| `KAUST` | unset | the answer to the third question, used when a stage runs a KAUST row (`SUPPLIERS=all`, or the GPU core stage): with the port absent, `yes` prints the copyright notice of CuBesselK in short, then downloads its two files and generates the port at the start of the run (the port is deleted at the end; where the files cannot be downloaded or ported, the run goes on without the KAUST rows), and `no` leaves the KAUST rows out; with the port in place, `no` stops the pipeline before anything runs (exit status 2; `make kaust CLEAN=yes` deletes the port). Unset, the question is asked on a terminal; without one `make pipeline` runs the KAUST rows only when the port exists, and `make reproduce` refuses to start (exit status 2) when it is absent. No other switch and no target implies it |
| `EXTERNALS` | `no` | `make externals` first, for what the run uses: Boost 1.75.0 and GSL 2.6 when neither `references/` nor the system has that version (where the system has it, the row uses that copy: the Boost 1.75.0 that `benchmark/tools/boost_detect.sh` selects, the GSL 2.6 that pkg-config finds; `PROVENANCE.txt` records it; a system GSL of another version does not count: GSL 2.6 is built, and the gsl row uses it; a GSL in `references/gsl/` is preferred to a system one, and of several there the highest version; with `BOOST_CPU_ROOT` set, Boost 1.75.0 is not downloaded, since the row uses that directory only), Boost 1.87.0 (only with a GPU part) when neither `references/` nor the system has a Boost with GPU support and `BOOST_ROOT` is not `none` (the GPU rows use the Boost that `benchmark/tools/boost_detect.sh` selects, of any version with GPU support; [Boost versions](#boost-versions)), NumPy and Matplotlib when the selected Python lacks them; no Bessels.jl sources and no closing `make requirements`; the plan lists what it downloads (on a terminal the pipeline asks when a missing external can be downloaded; a GSL of another version is not missing there) |
| `LEVIN` | `no` | the Levin A/B: builds the three arms (`make levin-ab-build`, logs in `build-ab/`), runs `STAGES=levin` |
| `FOLD` | `no` | the prefactor-fold stage: builds `build/fold_cpu`, runs `STAGES=fold` (CPU only) |
| `THREADS` | `no` | the thread-scaling stage, `STAGES=threads` |
| `GPU` | `auto` | the GPU pipeline (`benchmark/run_gpu.sh`); `auto` runs it when `nvcc` and an NVIDIA GPU are found, `no` leaves it out (also with `SETTINGS=full` and in `make reproduce`); with `yes` and no `nvcc` or no visible GPU, `benchmark/run_gpu.sh` stops before its runs (exit status 3) and the pipeline ends with failures |
| `GPU_STAGES` | `core`, plus `levin` and `threads` with `LEVIN` and `THREADS` | a comma list of `core`, `levin`, `threads`, or `all`, and `call-large`, which `all` does not include |
| `CHECK` | `no` | `make check` after the builds, before the runs; a failure stops the pipeline |
| `FIGURES` | `yes` | `make figures` and `make tables` at the end |
| `TABLES_CHECK` | `no` | `make tables-check` after the tables (needs `FIGURES=yes`) |
| `DRY` | `no` | `yes` prints the plan, the full-run notice and the questions with the answers the switches give, and downloads, builds and runs nothing; exit status 2 where the run would refuse to start ([Running the benchmark](#running-the-benchmark)) |

**Corpus, externals, Python**

| parameter | target | meaning |
|---|---|---|
| `DOWNLOAD=yes` | `make corpus`, `make kaust` | download; without it, the notice and the state |
| `CLEAN=yes` | `make kaust` | deletes the KAUST port, keeps the downloaded files |
| `DATA` | `make corpus`, `make corpus-check`, the runs | the corpus root (default `data/`) |
| `CORPUS_URL` | `make corpus` | a mirror URL or a local directory in place of the Zenodo record |
| `BOOST_CPU_ROOT`, `BOOST_ROOT` | the supplier builds | the Boost of the CPU row and of the GPU rows; `none` disables the row ([Boost versions](#boost-versions)) |
| `BOOST_VERSIONS` | `make boost-headers` | the versions fetched, of `1.75.0` and `1.87.0` (default both) |
| `PYTHON` | the generators, `make check` | the interpreter (default: the first of `PYTHON_CANDIDATES` with NumPy and Matplotlib at or above the tested minimum, `tools/py_stack.py`, else `python3`) |
| `PYTHON_CANDIDATES` | the generators, `make check`, `make requirements` | the interpreters searched (default `python3 /usr/bin/python3 python .venv/bin/python3`) |

**Figures and tables**: `ARGS` passes options to the generators: `--list` (the runs a
generator would use; writes nothing), `--strict` (refuses a run not marked complete; without
it such a run is read, and the generators name it on the console and in the tables'
`README.md`), `--runs DIR`, and for the tables
`--only=<text>` (the stems containing the text), `--out DIR` and `--no-preview`. The
generators list the items they skipped for missing inputs, and the table generator the files
in its output tree that it did not write.

## Run parameters

Environment variables of `benchmark/run_benchmark.sh`, passed through by `make pipeline` and
`make benchmark`. `benchmark/run_gpu.sh` passes them to its CPU reference run, which measures
one backend (the best this CPU runs, unless `BACKENDS` is set); its device stages read only
`REPS`, `DATA`, `LADDER_SET`, `LADDER_LAWS`, `LADDER_MAX` and `BENCH_RUNS_DIR`, always measure
both families, and it refuses `SUP_MAX_POINTS` and `TIME_MAX_POINTS`.

| variable | default | meaning |
|---|---|---|
| `FAMILIES` | `z_nu,h_nu_s` | the argument conventions measured |
| `BACKENDS` | every backend of `avx512,avx2,matched_fma,matched,scalar` this CPU runs | the backends measured; one given explicitly that this CPU cannot run fails the accuracy stage |
| `REPS` | `65` | timed repetitions per measurement |
| `STAGES` | `all` | `all` (accuracy, ladder, suppliers, supplier ladder), `ladder` (the two ladders), `rest` (accuracy and suppliers), `levin` (needs `make levin-ab-build`), `threads` (needs `make build/threads_cpu`), `fold` (needs `make build/fold_cpu`), `record` (the run record alone) |
| `LADDER_SET` | `small` (`large` in `benchmark/run_gpu.sh` run on its own) | ladders up to 2^20 (`small`) or 2^25 (`large`) points; both read the 2^25-point sets |
| `LADDER_MAX` | unset | caps the top rung of every ladder |
| `LADDER_PASS_FROM` | `4194304` (2^22) | `off`, or `S`, a power of two of at least 256: the CPU ladder rungs `N >= S` are read from one cumulative pass per repetition over the chunks `[0,S)`, `[S,2S)`, `[2S,4S)`, ..., with the clock read after each chunk; `off` times every rung on its own; `S` should be a rung whose arrays exceed the last-level cache (`ws_bytes` and `llc_bytes` in `ladder.csv`) |
| `LADDER_LAWS` | from `LADDER_SET` | the laws of the ladders |
| `SUPPLIERS` | `ship` | the suppliers of the supplier stage, `ship` or `all` |
| `SUP_LADDER`, `SUP_LADDER_BUDGET` | `ship`, `20` | the suppliers of the supplier ladder, and the seconds of one measurement after which a supplier's ladder stops |
| `SUP_MAX_POINTS`, `TIME_MAX_POINTS` | unset | reduced runs only: the supplier stage scores at most N points of each dataset (every k-th point); only the first N points of each dataset are timed |
| `DATA` | `data` | the corpus root |
| `BENCH_PIN` | `cpu` | `cpu` pins each timing process to one CPU of the NUMA node it selects, `node` to all CPUs of that node |
| `BUILD`, `AB_DIR` | `build`, `build-ab` | the drivers and the Levin A/B builds |
| `BENCH_RUNS_DIR` | `benchmark/runs` | the root of the run records, from which `make benchmark` also draws its figures and tables (not used by `make pipeline`) |
| `NOTIFY` | `auto` | the end-of-run notice: `auto` adds a desktop notification where possible, `none` prints only, any other value is run as a command with `BENCH_NOTIFY_TITLE` and `BENCH_NOTIFY_MESSAGE` set |
| `BENCH_UNATTENDED` | unset | when non-empty, no script asks a question, as without a terminal; `make pipeline` sets it after its own questions |
| `SCOPE` | derived | the label in the run id and `PROVENANCE.txt` (`run_label`): `default` when every run parameter has its default, else `custom`; `make pipeline` and `make reproduce` with the full settings set `cpu` for their CPU runs and `run_gpu.sh` sets `gpu`, as in the published runs |
| `THREADS_LAW`, `THREADS_N`, `THREADS_REPS`, `THREADS_ASSEMBLY`, `THREADS_PIN_MODES`, `THREADS_SUPPLIERS`, `THREADS_MAX` | `size_random`, `262144`, `REPS`, `direct`, `dense`, `ship`, the physical core count | the thread-scaling stage: law, points per thread, repetitions, supplier assembly, pinning (`dense`, `spread`), suppliers, largest thread count |
| `THREADS_WS_N`, `THREADS_WS_THREADS`, `THREADS_WS_BACKENDS` | unset, the online CPU count (at most `THREADS_MAX`), `BACKENDS` | the working-set sweep, run when `THREADS_WS_N` (points per thread, a comma list) is set |

## External suppliers

The suppliers are the external implementations of `K_ν` compared with the library (see the
accompanying paper, reference implementations). On the CPU each is wrapped behind one
interface (`references/common/cpu_supplier.h`), so the assembly of `M` is the same code for all.

| supplier | variants | source | licence | built when |
|---|---|---|---|---|
| Bessels.jl `besselk`, pinned commit and release 0.2.8 | `bessels.jl`, `bessels.jl-old` | C ports in `references/bessels_jl/port/` | MIT (`references/bessels_jl/LICENSE`); the K0/K1 coefficients originate in Boost.Math, BSL-1.0 (`references/bessels_jl/LICENSE_BOOST_1_0.txt`) | always |
| `std::cyl_bessel_k` | `std-cyl_bessel_k` | the C++17 standard library | libstdc++: GPL-3.0 with the GCC Runtime Library Exception | a C++17 compiler whose standard library provides the special functions (libstdc++; libc++ does not) |
| `boost::math::cyl_bessel_k` | `boost-1.75.0`; GPU `boost-gpu-<version>`, `boost-gpu-<version>-adapted` | Boost.Math | BSL-1.0 | a C++ compiler and the Boost of the row |
| GSL `gsl_sf_bessel_Knu` | `gsl-<version>` | the GNU Scientific Library | GPL-3.0 | built by `make externals`, or a system GSL found by pkg-config |
| KAUST CuBesselK | `kaust-128`, `kaust-40`, `kaust-adaptive` (CPU and GPU) | [github.com/stsds/CuBesselK](https://github.com/stsds/CuBesselK) at commit `94ec9ca`, not distributed | "All rights reserved", no licence | the port exists |

`libmatern` does not link GSL; `build/suppliers` and `build/threads_cpu` link it when a GSL
is found, so a distributed binary linked against GSL falls under GPL-3.0. No GSL source is
vendored. The files in `references/` are this project's (MIT, `LICENSE`), except the Bessels.jl
licence files and ports, which carry the Bessels.jl and Boost.Math notices, and
`references/kaust_logbesselk/derive_port.py`, which holds short CuBesselK fragments. The
`PROVENANCE.md` files in `references/bessels_jl/`, `references/kaust_logbesselk/`,
`references/cpp_stdlib/` and `references/gsl/` record the upstream source and licence, and
for the ports every change. The KAUST corrections and the log-space assembly make the external rows more accurate than
the released upstream code; a comparison quoting those rows should say so.

`make externals` downloads, without root and without changing the system, what is freely
available and not yet present: the Bessels.jl upstream sources at the pinned commit
(`references/bessels_jl/upstream/`, to verify the ports), the Boost headers 1.75.0 and
1.87.0 (`references/boost/<version>/`), GSL 2.6 as a static library
(`references/gsl/<version>/`, preferred to a system GSL), and NumPy and Matplotlib into
`.venv/` (pip; the versions of the published figures, NumPy 2.2.5 and Matplotlib 3.11.0,
else versions at or above NumPy 2.0.2 and Matplotlib 3.9.4) when no Python has them at or
above that tested minimum; an older NumPy or Matplotlib is not used (`tools/py_stack.py`
decides, for the Makefile's search of `PYTHON` as well). It checks every other
archive against a pinned SHA-256 digest and ends with `make requirements`.

```sh
make references                     # fetch the upstream sources at pinned commits, verify, report
make -C references                  # build every supplier whose prerequisites are present
make -C references check-flags      # count the fused multiply-adds: GNU dialect vs -std=c11
make -C references extract          # regenerate the Bessels.jl coefficient header
make -C references verify-upstream  # re-hash the KAUST upstream files, if present
make benchmark-suppliers            # list the supplier variants in the build
```

The C supplier objects are built with `-march=native -O2` and no `-std` (the C++ object
with `-std=c++17`), the flags of the published measurements; the header of
`references/Makefile` states the rule. The flags are part of the suppliers' arithmetic: in
GCC's GNU dialect a written `a*b + c` is fused into one FMA, under `-std=c11` it is not;
`make -C references check-flags` counts the fused instructions of the C ports under both
dialects.

### Boost versions

`benchmark/tools/boost_detect.sh` selects the Boost of each row from every Boost it finds
(system paths, conda, Homebrew, `BOOST_ROOT`, `references/boost/`):

| row | version | override |
|---|---|---|
| CPU (`boost-1.75.0`) | Boost 1.75.0 only; without it the row is left out | `BOOST_CPU_ROOT=<dir>` (must hold 1.75.0), `none` disables it |
| GPU (`boost-gpu-<version>`) | the highest Boost with `BOOST_MATH_GPU_ENABLED` (1.87 or later; published: 1.87.0) | `BOOST_ROOT=<dir>`, `none` disables it; a `BOOST_ROOT` without GPU support is passed over with a note |

A compile of the CPU row whose `boost/version.hpp` differs from the selected version (for
example through `CPATH`) stops with an error. `make boost-headers` fetches the header-only
Boost.Math versions of `BOOST_VERSIONS` from archives.boost.io into
`references/boost/<version>/`, checked against pinned SHA-256 digests; it prints nothing
while a download runs. A version already present is not fetched again, and one whose
`BOOST_VERSION` differs makes the command fail. A change of the Boost selected for the CPU row rebuilds its object.

Under the pipeline (`EXTERNALS=yes`, `make reproduce` or a yes to the externals question),
`make externals` downloads Boost 1.87.0 only where neither `references/` nor the system has
a Boost with GPU support and `BOOST_ROOT` is not `none`; otherwise its output names the
Boost the GPU rows use (none with `BOOST_ROOT=none`). `PROVENANCE.txt` records the version
of that Boost (`build_gpu_suppliers_boost`) and its directory with its origin
(`build_gpu_suppliers_boost_source`). The published GPU rows used
the Boost 1.87.0 of an environment module of the run machine, through `BOOST_ROOT`; its
`boost/` directory is identical to that of the archive `make boost-headers` fetches. A Boost
with GPU support of another version gives rows `boost-gpu-<version>` whose values can differ
from the published ones. A Boost found outside `references/` is used where it is and not
copied, so a later build in a shell without it (for example without the environment module)
needs its own: the module loaded again, `BOOST_ROOT` (GPU rows) or `BOOST_CPU_ROOT` (CPU
row), or `make boost-headers`. A change of the Boost selected for the GPU rows, or of its
origin, rebuilds `build/suppliers_gpu` and `build/boost_shortcut_gpu`; with none selected
the first is built without Boost rows, and building the second stops and removes it.

### KAUST CuBesselK

CuBesselK is public but carries "All rights reserved" without a licence, so its sources and
anything derived from them are not distributed. Its small-argument branch carries GSL's
Chebyshev tables (GPL-3.0), so the downloaded files and the generated port contain GSL data
as well. The port is generated on the user's machine, on request, and is not for
redistribution.

```sh
make kaust                 # the copyright notice, the state of the files, the next command
make kaust DOWNLOAD=yes    # download the two files, check their sha256, generate the port
make kaust CLEAN=yes       # delete the port; the downloaded files are kept
```

`make kaust DOWNLOAD=yes` downloads `src/logbesselk.cu` and `include/logbesselk.h` of commit
`94ec9ca4bb9d7524c196f044503519ab30d8eb39` into `references/kaust_logbesselk/upstream/`,
checks their sha256 against `references/kaust_logbesselk/PROVENANCE.md`, and generates the
port `references/build/kaust/kaust_derived.h` (`derive_port.py`, Python 3). Files placed in
`upstream/` by hand, for example from a clone at that commit, are used as they are when their
sha256 matches, and the same command ports them; a file with another sha256 is reported and
left, and nothing is ported. The files are downloaded only on an explicit yes: this command
(after the notice), or `KAUST=yes` for `make pipeline` or `make reproduce` and the question
these ask on a terminal, both after a short form of the copyright notice.

While the port exists, the KAUST rows are built: `-DCPU_SUPPLIER_WITH_KAUST` (recorded in
`references/build/defines.txt`) for `build/suppliers` and `build/threads_cpu`, and
`build/kaust_dev.o` with `-DGPU_SUPPLIER_WITH_KAUST` for `build/suppliers_gpu`. On the CPU they run
only with `SUPPLIERS=all` (`SUP_LADDER=all` and `THREADS_SUPPLIERS=all` for the supplier
ladder and the thread stage), on the GPU whenever the port exists. `make kaust CLEAN=yes`
and the end of `make pipeline` and of `make reproduce` delete the port, its objects and the
supplier programs linked with them; `upstream/` is kept. `make -C references verify-upstream` checks `upstream/` and
neither downloads nor ports.

`kaust-128` and `kaust-40` run the released code with 128 and 40 quadrature panels (the
bins of Geng et al.). `kaust-adaptive` uses 128 panels, applies the decimal exponent `e10` that the released
small-argument recurrence accumulates, forms `log cosh` without the intermediate `cosh`,
and integrates over the adaptive range of Geng et al. in place of the released fixed range
(`references/kaust_logbesselk/PROVENANCE.md`).

## The corpus

Every stage scores and times against the evaluation corpus, published on Zenodo as
[doi:10.5281/zenodo.23393956](https://doi.org/10.5281/zenodo.23393956) under CC-BY-4.0
(cite the DOI when using the data; `CITATION.cff` carries the reference): 75 files,
5.19 GB, of which 61 small files take 90 MB and 14 files of 2^25 points 5.10 GB.
`data/DATASETS.md` specifies the datasets, `data/MANIFEST.md` gives the census and the sha256
of every file, and `data/README.md` describes the folders (see also the accompanying paper,
benchmark domain and sampling).

```sh
make corpus                # what the corpus is, what of it is here, how long a download takes
make corpus DOWNLOAD=yes   # download the missing or damaged files, check them, move them into place
make corpus-check          # check every file against the sha256 list of data/MANIFEST.md
```

The download checks each file against `data/MANIFEST.md` and moves it into its folder
(`data/z_nu/`, `data/h_nu_s/`, `data/boost_shortcut/`); run again, it continues an
interrupted download, and it never overwrites a file. `DATA=<dir>` keeps the corpus
elsewhere (download, check and runs); `CORPUS_URL` takes the files from a mirror or a local
copy. The scripts (`tools/corpus.sh`, `tools/corpus_check.sh`) are POSIX `sh`, so the corpus
can be downloaded and checked on Linux, macOS and WSL. Every measuring command checks the
corpus before its first stage and, on a terminal, offers to download missing or damaged
files; a `benchmark/run_benchmark.sh` run that reads no 2^25-point file (`STAGES=rest`, or
`LADDER_LAWS` without a size law) needs only the 61 small files.

This check reads the size of every file and the 32-byte header of every `.bin` file, not
the sha256. A file that fails it, for example an interrupted copy or an HTML page
saved in place of the file, counts as missing: the download moves it to `<file>.bad`, checks
each new file against `data/MANIFEST.md` before moving it into place, and deletes the `.bad`
copy once its replacement is verified. When a download fails, it names the files still
missing or damaged and compares them with the sizes and md5 sums of the record's file list
(`sh tools/corpus.sh zenodo`). `make corpus-check` computes the sha256 of every file and
fails when a file differs or is absent; the exit status of `tools/corpus_check.sh` is 0 when
every file is present and matches, 1 for a mismatch or a file not in the list, 3 when every
file present matches but some are absent, and 2 when it cannot check. Every run records the
same sums (`make pipeline` computes them once, before its first stage, for all its runs); a
mismatch is a `warning` line in the run's `STATUS`.

### Corpus files

A file is named `<family>-<prefix>_<set>_<N>.bin`, with the prefix `nu`, `z`, `h`, `s` (the
scale parameter) or `ref`; `boost_shortcut/` holds `<prefix>_boost_shortcut_873.bin`
with the prefix `nu`, `z` or `refK`. Record `i` of every file of a set describes point `i`.
The seam labels, `<family>-label_seam_100000.txt`, are text (`data/DATASETS.md`). Every
`.bin` file is a 32-byte header followed by little-endian arrays:

| offset | bytes | type | field |
|---|---|---|---|
| 0 | 8 | `char[8]` | magic, `WMCORP01` |
| 8 | 4 | `uint32` | format version, 1 |
| 12 | 4 | `uint32` | kind: 1 = axis, 2 = reference |
| 16 | 8 | `uint64` | `N`, the number of points |
| 24 | 4 | `uint32` | the text format of the values: 0 = `%.13a`, 1 = `%a` (`boost_shortcut`) |
| 28 | 4 | `uint32` | reserved, 0 |
| 32 | | | the arrays |

An axis file (`nu_`, `z_`, `h_`, `s_`) holds `double v[N]` (`32 + 8 N` bytes). A
reference file holds `double hi[N]`, then `double lo[N]`, then `int16 bits[N]` (signed), not
interleaved (`32 + 18 N` bytes): the reference is `hi[i] + lo[i]`, `bits[i]` the accurate
bits of the arb ball it comes from (the sentinels 32767 and -32768 occur in no file).
`ref_` holds the covariance `M`, `refK_` holds `K_ν(z)`. `benchmark/corpus_bin.h` is a
header-only C99 reader that checks the header against the file size and reads an array, or
a prefix of it at a stride. In Python:

```python
import struct
import numpy as np

def read_corpus(path):
    """An axis file as one array; a reference file as (hi, lo, bits)."""
    raw = open(path, 'rb').read()
    magic, version, kind, n, fmt, _ = struct.unpack_from('<8sIIQII', raw)
    if magic != b'WMCORP01' or version != 1 or len(raw) != 32 + (8 if kind == 1 else 18) * n:
        raise ValueError(path + ': not a corpus file of this format')
    if kind == 1:
        return np.frombuffer(raw, '<f8', n, 32)
    return (np.frombuffer(raw, '<f8', n, 32), np.frombuffer(raw, '<f8', n, 32 + 8 * n),
            np.frombuffer(raw, '<i2', n, 32 + 16 * n))
```

`build/corpus_convert` (`make corpus-tools`) converts between the binary files and their
text form, every double in C99 hexadecimal, which `strtod` and `float.fromhex` read back to
the same double, signed zeros included:

```sh
build/corpus_convert to-text <in.bin> [out.txt] # one record per line: a value, or "hi lo bits"
build/corpus_convert to-bin <in.txt> <out.bin>  # refuses a line that does not print back identically
build/corpus_convert verify <in.txt> <in.bin>   # prints the binary file as text, compares byte for byte
```

### The documents in `data/` and the Zenodo record

The data files are not part of the repository; `data/README.md`, `data/DATASETS.md` and
`data/MANIFEST.md` are the documents of the record. An earlier record,
doi:10.5281/zenodo.23219731, holds the same 75 files (same sha256), with the `h_nu_s-*`
files named `h_nu_rho-*` (`h_nu_s-s_*` as `h_nu_rho-rho_*`); `make corpus DOWNLOAD=yes`
moves such files to their names here without downloading them.

## Replicating the published results

The published results come from the full benchmark on exclusive nodes: its CPU part once on
an Intel Xeon Platinum 8358 and once on an AMD EPYC 9454, its GPU part on an NVIDIA A100
80GB PCIe in a machine with the Intel CPU, each part as a separate batch job. One command
runs the full benchmark on one machine:

```sh
make reproduce
```

It is `make pipeline` with the settings of the full benchmark (`SETTINGS=full`), the
externals the run uses (`EXTERNALS=yes`) and the download of the missing or damaged corpus
files (`CORPUS=yes`, 5.19 GB from Zenodo). The GPU part runs where `nvcc` and an NVIDIA GPU
are found and is left out elsewhere; `make reproduce GPU=no` runs the full CPU part only.
`SETTINGS=full` sets `LEVIN`, `FOLD`, `THREADS` and the run parameters over values given on
the command line. The published CPU rows used the system Boost 1.75.0 and GSL 2.6 of the
run machines (packages of Red Hat Enterprise Linux 9.6; GSL as a shared library found by
pkg-config). `make reproduce` likewise uses a system Boost 1.75.0 or GSL 2.6 where it finds
one (a GSL copy in `references/gsl/` is preferred to it) and downloads the version only where
the system lacks it; `PROVENANCE.txt` records the copy a run used
(`build_cpu_supplier_cxx_boost_source`, `build_cpu_supplier_gsl_source`). A GSL compiled
with fused multiply-adds (an `-march` with FMA) gives different gsl rows; that can be a
system copy or the copy `make externals` builds, which takes the compiler's default `-march`
unless `CFLAGS` sets one. `build_cpu_supplier_gsl_library` records the library file the
build linked and the number of fused multiply-adds in its Bessel functions (most of the K_nu
code).

KAUST CuBesselK is not distributed. When its port is absent, `make reproduce` on a terminal
shows the copyright notice of CuBesselK in short and asks whether to download its two files
and generate the port. Without a terminal (a batch job, `nohup`, `| tee`) it asks nothing
and needs `KAUST=yes` (the notice in short, then the download and the port; where these
cannot be done, the run goes on without the KAUST rows) or `KAUST=no` (no KAUST rows) on
the command line; without either it stops before anything runs. A port in place is used,
and `KAUST=no` then stops the run before anything runs (`make kaust CLEAN=yes` deletes the
port). The port is deleted at the end of the run and the two files are kept, so a later
`KAUST=yes` or `make kaust DOWNLOAD=yes` only generates the port again.

The figure `cross-machine/fig-paper-machine-independence` compares the CPU runs of an Intel
and an AMD CPU (two models of one vendor do not count), so one machine cannot write it; a
machine with the GPU part writes every other figure and table of the paper set (with the
KAUST rows after a yes to the KAUST question or `KAUST=yes`). As in the published run, run
`make reproduce` on one Intel machine with an NVIDIA GPU and on one AMD machine, copy the
run directories of the second (`benchmark/runs/<run id>/`, not `latest`) into
`benchmark/runs/` of the first, and write the figures and tables again there with
`make figures` and `make tables`. The generators read every run in
`benchmark/runs/` and take each kind of result from the newest run of the machine that has
it, so an earlier run there can supply a stage; the CPU results come from a machine without
GPU data, here the AMD one (`ARGS=--cpu-machine NAME` selects another).

`make reproduce` installs no system software; `make requirements` reports what is present.
Besides the [requirements](#requirements) it needs pdflatex with geometry, amsmath, amssymb,
booktabs, array, float and rotating; a C++17 compiler with libstdc++; `curl` or `wget`,
`bzip2` and `sha256sum`; Python 3; and for the GPU part `nvcc` and an NVIDIA GPU. Without
pdflatex the tables are not written and the run ends with failures; a CPU supplier row that
is not built (`std::cyl_bessel_k`, Boost, GSL, KAUST) is a `warning` in the CPU run's
`STATUS`.

```sh
make reproduce DRY=yes                   # the plan and the questions; runs nothing
make reproduce                           # on a terminal
make reproduce KAUST=no > run.log 2>&1   # without a terminal: KAUST=yes or KAUST=no
```

Without a terminal, `make reproduce DRY=yes` exits with status 2 where the run would refuse
to start, as without `KAUST=yes` or `KAUST=no` while the port is absent.

The full run takes about 18 hours on one machine with an A100 GPU, most of it the CPU
ladders. The ladders time every rung up to 2^21 points on its own and the rungs from 2^22 up
in one cumulative pass per repetition (`LADDER_PASS_FROM`, default 2^22);
`LADDER_PASS_FROM=off` times every rung on its own and about doubles the ladder time.
Durations of the published run, as the wall time of each batch job (build included):

| part | time |
|---|---|
| ladders (up to 2^25 points) | 12.2 h on the AMD EPYC 9454, 14.3 h on the Intel Xeon 8358 |
| accuracy and suppliers | 2.0 h (AMD), 2.3 h (Intel); the KAUST rows 1.7 h and 1.8 h of it |
| Levin A/B | 10 min (AMD), 13 min (Intel) |
| thread scaling | 14 min (AMD), 15 min (Intel) |
| prefactor fold | 5 min (AMD), 6 min (Intel), a separate run |
| GPU stages (A100) | 46 min (core, Levin A/B and thread settings) |

The published rows come from this toolchain (the `build_*` keys of each run's
`PROVENANCE.txt`); a different compiler or libm can change the last bits of the rows that
depend on it:

| part | toolchain |
|---|---|
| the library, the C drivers, the Bessels.jl ports, the KAUST CPU port and the GSL wrapper | the system `cc`, GCC 11.5.0; glibc 2.34 |
| `std::cyl_bessel_k` and the Boost CPU row | g++ 14.2.0 (libstdc++ 14), C++17; Boost 1.75.0, evaluated in `long double` |
| GSL | GSL 2.6, the Red Hat Enterprise Linux 9.6 package (a shared library, found by pkg-config) |
| the device code and the GPU drivers | nvcc 12.8.61 with host compiler GCC 14.2.0; CUDA runtime 12.8; the Boost GPU rows use Boost 1.87.0, evaluated in `double` |
| the figures and tables | Python 3.12.9 with NumPy 2.2.5 and Matplotlib 3.11.0 (the two versions `make externals` installs into `.venv/`); pdfTeX of TeX Live 2020 for the tables |

`tools/published_env.sh` holds these versions. `make requirements` ends with a `published
environment` section that compares them with the tools found here, and `make pipeline` and
`make reproduce` show the comparison for the tools the run uses in their plan and warn on
standard error where one differs. A different compiler or nvcc changes the values (in the
last bits) and the timings of the rows it builds; a different Python, NumPy, Matplotlib or
pdflatex changes only the rendering of the figures and the fitted column widths of the
tables, since the generators compute every value in plain Python. The comparison covers
these tools, not the processor, the libm or the other system libraries.

Every measuring command prints the full-run notice at its start (`benchmark/tools/full_run.sh`),
with the settings in which it differs from the full run.

## Output

A run is written to `benchmark/runs/<run id>/` (`BENCH_RUNS_DIR` sets another root), and
`latest` links to the newest run. The run id is `<stamp>_<host>_<SCOPE>`, with `-<STAGES>`
appended when `STAGES` is not `all`, and `.2`, `.3`, ... when the name is taken. A run
directory holds the files of the stages it ran:

| file | contents |
|---|---|
| `accuracy.csv`, `ladder.csv` | the library's accuracy and timing per family, dataset and backend; its throughput against N |
| `suppliers.csv`, `suppliers_ladder.csv` | the same for the suppliers |
| `levin_<arm>.csv`, `levin.csv`, `exp_source.csv` | the Levin A/B: each arm, the arms combined, the default-routing rows behind `EXPECTED.txt` |
| `fold.csv` | the prefactor-fold stage |
| `threads.csv`, `threads_*.csv` | thread scaling: every row, each invocation, the working-set sweep (`threads_ws*.csv`) |
| `gpu*.csv`, `levin_build.log` | the GPU stages, added by `benchmark/run_gpu.sh` |
| `PROVENANCE.txt` | machine, operating system, clock, compilers, build and run parameters |
| `MANIFEST.txt` | the sha256 of every input: the corpus files (a `# corpus:` line names any mismatch and the absent files), the benchmark, supplier and library sources, the KAUST files when present, the library archive |
| `detect.txt` | the report of `build/detect`: CPU, load, NUMA placement, timer, ULP definition, instruction sets, installed components, corpus |
| `STAGES.txt` | per stage, supplier and missing dataset: exit status, row count, `NO DATA` or `SKIPPED` |
| `STATUS` | `complete`, `partial` or `failed`, what is missing or skipped, warnings (from the sha256 check of the corpus against `data/MANIFEST.md`), the load before and after |
| `EXPECTED.txt` | `u_max` of `avx512` on four `z_nu` datasets beside a reference value |
| `timing_raw.<x>` | every timed repetition (`n`, `rep`, `ns_per_elem`) |
| `timing_raw.<x>.sched.csv` | per timed row of `benchmark_cpu` and `suppliers`: thread count, migrations, involuntary context switches, and the clock of the timing CPU right after the row (`mhz_end`, MHz; `cpu_mhz_timed` in `PROVENANCE.txt` summarises it) |
| `accuracy_detail.<x>[.gz]` | per accuracy row: the signed ULP histogram (`hist`), the 64 worst points (`worst`), every excluded point with its reason (`excluded`) |
| `*.err` | the standard error of each stage |

`EXPECTED.txt` lists `u_max` of `avx512` on `grid`, `large_z`, `large_nu` and `seam`
(`z_nu`) beside the reference value hard-coded in `benchmark/run_benchmark.sh`: `unchanged`
within 10^-3 ulp, `MOVED` otherwise. It is a report, not a test. The reference values come
from the default build with GCC 11.5.0; another compiler or a changed `OPTIMISE`, `CFLAGS`
or `XCF` can move the measured values. With that build
on a CPU with AVX-512, a `MOVED` row leaves two causes to check: changed library sources, or
a corpus file that does not match its sha256 (`MANIFEST.txt`). `avx2` is the one backend
whose results can depend on the processor vendor: its gamma engine uses `_mm_rcp_ps`, whose
result is implementation-defined.

In `PROVENANCE.txt`, `release` is `MATERN_VERSION_STRING`; `source_commit` is taken from
git when the tree is the top level of a git checkout, else from `BENCH_SOURCE_COMMIT`, else
from the `RELEASE` file that `git archive` fills in (`make dist`). The `build_*` keys name
what each artefact was compiled with, read back from it (`build/detect --build-info`), the
`make_*` keys the make variables of the build, `ulp_*` the ULP definition and `timer_*` the
clock. `build_cpu_supplier_cxx_boost_source` and `build_cpu_supplier_gsl_source` record the
Boost and GSL copy behind the CPU rows, and `build_gpu_suppliers_boost_source` the Boost
behind the GPU rows, as `<directory> (<origin>)`, the directory being the include directory,
or the install prefix for a GSL in `references/gsl/`. For GSL the origin is `make externals`
(a copy in `references/gsl/`) or `pkg-config`; for Boost it is the override where it selects
the directory (`BOOST_CPU_ROOT` for the CPU row, `BOOST_ROOT` for the GPU rows), else the
first way the directory is found: `system` (the compiler's search path, which `CPATH`
extends, Homebrew or a standard directory), the variable that names it, or `make externals`
(a copy in `references/boost/`, which `make boost-headers` also fetches). `HARDWARE.md`
shows these three; the Boost short-cut check records its copy in the same way
(`build_gpu_boost_shortcut_boost_source`, `build_gpu_boost_shortcut_cpu_row_boost_source`).
The `build_cpu_supplier_gsl_compiler`, `_std`, `_isa` and `_glibc_headers` keys describe the
wrapper `cpu_gsl.c`, not the GSL library. `build_cpu_supplier_gsl_library` describes the
library the build linked, as `<file> (<origin>); <build>; <fused multiply-adds>`. For a GSL
from `pkg-config` the file is the one the linker opens for `-lgsl`, symbolic links resolved;
for a copy in `references/gsl/` it is its `lib/libgsl.a`. `<build>` names the compiler and
options that compiled `bessel_Knu.c` where the file's debugging information records them
(`DW_AT_producer`, cut to the compiler and its `-march`, `-mtune`, `-mfma`, `-mno-fma`, `-O`
and `-f` options; a distribution package usually strips it), and for a GSL from `pkg-config`
the package that owns the file (`rpm` or `dpkg`). Without that information, for a copy in
`references/gsl/` `<build>` is the compile command of `bessel_Knu.c` from the build log with
the compiler version, which `references/fetch_gsl.sh` writes into
`references/gsl/<version>/build-flags.txt`. The last field counts the fused multiply-add
instructions `objdump` finds in the Bessel functions (names starting `gsl_sf_bessel_` or
containing `temme` or `steed`, compiler clones included; most of the K_nu code is among
them) and in the whole library, or names why they were not counted (no `objdump`, not
x86-64, no machine code in the file, as in an LTO-only archive). An instruction counts for
the symbol before it, so for a library without a symbol table (a stripped shared library)
the first count also takes in the local functions that follow a Bessel function and compares
only with that of another such library. `HARDWARE.md` shows the key after the GSL copy.
Progress lines go to descriptor 3 (`BENCH_PROGRESS_FD`), not into the record.

### Figures and tables

The figure and table targets, `make benchmark` and `make pipeline` write the **paper set**
and nothing else: the figures the paper includes and the tables its own tables are taken from
(`PAPER_FIGURES` and `PAPER_TABLES` in `benchmark/plots/make_paper_figures.py`). A result
that depends on the argument convention is written for both, `(z,ν)` and `(h,ν,s)`, except
the ladder figure, which is drawn for `(z,ν)` only. Each CPU result, the machine-independent
tables included, comes from one machine: the first that has the stage it is drawn from,
machines without GPU data first (`ARGS=--cpu-machine NAME` puts one machine first); the
one-backend CPU stage of a GPU run is used only when no CPU run has the stage, and not for
the machine-independent accuracy tables, which need a CPU run with the accuracy stage. The
GPU figures and tables come from the GPU host, the thread figure from every machine; the
generators print where each stage was taken from.

`make figures` writes the figures as `.pdf`, with `captions.tex`, to
`benchmark/plots/paper-final/`; nothing is drawn on the canvas but the panels and their keys.
`HARDWARE.md` there names the Python, NumPy and Matplotlib and the machines and software of
the runs drawn, and lists per figure its runs and the details its caption does not carry.
`make tables` writes to `benchmark/tables/paper/` the machine-independent accuracy
tables at the root and every table with a cost column in `<machine>/`, with a `README.md`
naming the software that made them and, per table, its runs and the details its caption and
note do not carry. Each table is written as `.tex` (a booktabs float; a `% requires:` line
names its LaTeX packages), `.md` and `.csv` (full precision, including values the `.tex` does
not print); each directory has a `preview.tex` and `preview.pdf`, and a table that overflows
the column is set smaller, down to a rotated `sidewaystable`.

A machine is a CPU model as recorded in `PROVENANCE.txt` (`xeon-platinum-8358`,
`epyc-9454`), so runs with the same CPU model are joined; the GPU outputs go to the
directory of the GPU host's CPU model. Each kind of result is taken from the newest run of
the machine that has it; the one-backend CPU rows of a GPU-pipeline run are used only when
no other run of the machine has that kind.

The outputs by stem (`<m>`: a CPU model; `<g>`: the GPU host; `<law>`: the ladder dataset,
`size_random` or `size_half`, or with `LADDER_SET=large`, as published, `size_random_large` or
`size_half_large`); *accuracy* is a CPU run with the accuracy stage, *suppliers* adds the
supplier stages:

| output file | shows | needs |
|---|---|---|
| `tab-datasets`, `tab-references`, `tab-exclusions-<family>` | the corpus sets, the implementations, the excluded points | accuracy |
| `<m>/fig-paper-assembly-<family>` | the assemblies of each supplier | suppliers |
| `<m>/fig-paper-cost-accuracy-<family>-call` | cost against accuracy | suppliers |
| `<m>/fig-paper-accuracy-distribution-<family>`, `tab-medmax-<family>` | accuracy distributions; median and maximum ULP error | suppliers |
| `<m>/fig-paper-ladder-<law>-z_nu`, `<m>/tab-throughput-<family>` | throughput against N | ladders |
| `<m>/fig-paper-workspace-<law>-<family>` | cost with a new and a reused workspace against N | ladders |
| `<m>/fig-paper-levin-cpu-<backend>-<family>` | Levin A/B of the vector router on the CPU, `<backend>` the best vector backend present | `LEVIN=yes` |
| `<m>/fig-paper-fold-<family>` | prefactor fold | `FOLD=yes` |
| `<m>/fig-paper-threads-<family>` | thread scaling | `THREADS=yes` |
| `cross-machine/fig-paper-machine-independence` | the rows of two CPU architectures against each other | CPU runs on an Intel and on an AMD CPU (two models of one vendor do not count) |
| `<g>/fig-paper-gpu-assembly-<family>` | assemblies of the device kernels | `GPU=yes` |
| `<g>/fig-paper-gpu-plane-<family>`, `<g>/tab-gpu-cost-<family>`, `<g>/tab-gpu-medmax-<family>`, `<g>/tab-gpu-exclusions-<family>` | cost and accuracy of the device kernels; their excluded points | `GPU=yes` |
| `<g>/fig-paper-gpu-accuracy-distribution-<family>` | accuracy distributions of the device kernels | `GPU=yes` |
| `<g>/tab-gpu-call-<family>` | whole calls, transfers included | `GPU=yes` |
| `<g>/fig-paper-gpu-modes-<law>-<family>` | transfer strategies against N, on the random size dataset | `GPU=yes` |
| `<g>/fig-paper-gpu-boost-shortcut` | Boost's `K_ν` near ν = 5/2 | `GPU=yes`, Boost 1.87 or later |
| `<g>/fig-paper-levin-cuda-async-direct-<family>` | Levin A/B on the GPU | `GPU=yes LEVIN=yes` |
| `<g>/fig-paper-gpu-knobs` | CUDA block size and host staging threads | `GPU=yes THREADS=yes` |
| `tab-environment` | machines and software of the runs | any run |

### Columns of the CSV files

Each CSV file starts with a header line, and the generators read its columns by name. The
error of a value `v` against the reference `r_hi + r_lo` is `u = ((v - r_hi) - r_lo) / s`
ulp, with `s` taken at the reference: the spacing above `|r_hi|` by default (`ulp_mode`
`up`); `benchmark_cpu --ulp-mode down|muller|straddle` selects another definition, and
`build/detect --ulp` prints the default definition and where the others differ from it.
Times are in ns per element; every cost in the figures and tables is the median over the
repetitions.

**Per dataset**: `accuracy.csv`, `suppliers.csv`, `gpu.csv`, `gpu_kernels.csv`,
`levin_<arm>.csv`, `gpu_levin_<arm>.csv`, `exp_source.csv` and `fold.csv` have one row per
family, dataset and backend, or supplier variant and assembly; `levin.csv` and `gpu_levin.csv`
add `variant` (`levin_on`, `levin_off`) and `shipped` (`yes` for the default routing) in
front. The columns, in the order of `accuracy.csv`:

| columns | meaning |
|---|---|
| `schema` | the layout version, 11 (the generators accept 10 and 11) |
| `device`, `supplier` | `cpu` or `cuda`; `-` for the library, else the supplier variant (`libmatern-cuda` for the library's kernel in `gpu_kernels.csv`) |
| `family`, `law` | the argument convention and the corpus set |
| `backend` | the library's backend; for a supplier the assembly; in `fold.csv` the scalar backend, a twin or the control; in `gpu.csv` and `gpu_levin_<arm>.csv` the transfer strategy; in `gpu_kernels.csv` the assembly, with `/<strategy>` on `call` rows |
| `ulp_mode` | the definition of `s` |
| `n`, `n_scored` | points evaluated (the dataset, or every k-th point of it); points with a finite result, over which the errors are formed |
| `n_excluded`, `nonfinite` | points left out of the errors, and those with a non-finite result (the only reason, so equal) |
| `ref_zero`, `ref_subnormal`, `exact` | scored points with a zero or a subnormal reference, and with `u = 0` |
| `min_bits` | the fewest accurate bits of a reference among the `n` points |
| `u_min`, `u_q1`, `u_med`, `u_q3`, `u_p99`, `u_p999`, `u_max`, `u_mean` | the distribution of `\|u\|`: minimum, quartiles, 99th and 99.9th percentiles (nearest rank), maximum, mean |
| `u_iqr`, `u_fence`, `beyond_fence` | `u_q3 - u_q1`; `u_q3 + 1.5 u_iqr`; scored points above it |
| `u_med_nontrivial` | the median of `\|u\|` over the points with a nonzero reference |
| `rel_max`, `abs_max` | the largest relative error of `M` (nonzero references) and the largest absolute error |
| `worst_nu`, `worst_arg` | ν and the argument (`z` or `h`) of the point of `u_max` |
| `ns_min`, `ns_q1`, `ns_med`, `ns_q3`, `ns_p95`, `ns_max`, `ns_mean`, `ns_iqr`, `ns_spread`, `reps` | the time over the repetitions: minimum, quartiles, 95th percentile, maximum, mean, `ns_q3 - ns_q1`, `100 (ns_max - ns_min) / ns_min` (percent); the repetitions |
| `ns_points` | points timed: `n`, or the first `TIME_MAX_POINTS` |
| `timing_scope` | `call`: the whole call (on the CPU with a workspace built in each repetition on `avx512` and `avx2`, the backends that use one; on the GPU with the transfers); `kernel`: the device kernel alone |
| `n_file` | points of the dataset on disk |

**Ladders**: `ladder.csv` has one row per family, dataset, backend and rung `n`, timed with a
workspace built in every call (`ns_ws_per_call_*`, the library's timing) and with one
reused workspace (`ns_ws_reused_*`), each with the `ns_*` statistics (on the backends that use
no workspace both series time the same call); `ns_setup_per_elem` is the difference of their
minima; `ns_workspace` and `ns_workspace_med` time one workspace
create and destroy; `nu_med`, `arg_med` are medians of the rung's inputs; `llc_bytes`,
`llc_shared_cpus` the last-level cache of CPU 0 and its sharing; `ws_bytes` the rung's
arrays; `ladder_mode` is `rung` (timed alone) or `pass` (from the cumulative pass) and
`ladder_pass_from` the `S` of `LADDER_PASS_FROM` (0 for `off`). `suppliers_ladder.csv` has
`schema` to `backend`, `n`, the `ns_*` block, `reps`, `ns_points`, `timing_scope`,
`ladder_mode`, `ladder_pass_from`; a supplier's ladder stops after the first rung over
`SUP_LADDER_BUDGET`. `gpu_kernels_ladder.csv` and `gpu_call_large.csv` have the same
columns without the last two (`backend`: the assembly, or `<assembly>/<strategy>`).
`gpu_ladder.csv` (`backend`: the transfer strategy) has `nu_med` and `arg_med` after `reps`,
then `timing_scope` and `ns_points`.

**Thread scaling** (`build/threads_cpu`): `threads.csv` and every `threads_*.csv`,
`threads_ws*.csv` included, have one row per thread count with `schema` to `backend` (the
assembly for a supplier), `threads`, `pinned`, `cores_used`, `physical_cores`,
`siblings_per_core`, `n_per_thread`, `n_total`, `reps`; `wall_s` and `ns_per_elem`, from
the slowest thread's minimum, workspace creation included on `avx512` and `avx2`; `speedup`
(ns per element at the invocation's first thread count over that at `threads`) and
`efficiency` (`speedup / threads`); `identical` (every thread's results bit-identical; `n/a`
at one thread); `load1` (the one-minute load at the start), `status`, `values_per_s`,
`u_max`, `u_scored`; `pin_mode`, `mhz_end` (the clock of the pinned CPUs at the end),
`llc_bytes`, `llc_shared_cpus`, `ws_bytes`; `ns_min` to `ns_spread` over the slowest
thread's time per repetition; then `ulp_mode`, `acc_points` (the first `n_per_thread`
points) and `n_excluded` to `worst_arg` as above, without `u_max`.

**GPU options** (`build/threads_gpu`): `gpu_threads.csv` and `gpu_threads_<family>.csv` have
one row per option value and transfer strategy: `schema` to `law`, `backend` (`cuda`),
`knob` (`block`, threads per block, or `host_threads`, the host staging threads, under
`ring` only), `knob_value`, `transfer_mode` (`sync`, `direct`, `ring`, `mapped`), `n`,
`reps`, `ns_per_elem` (the minimum), `ns_med`, `ns_spread`, `identical` (bit-identical to
the call with the default options), `status`, `ns_min`, `ns_q1`, `ns_q3`, `ns_p95`,
`ns_max`, `ns_mean`, `ns_iqr`, `ns_points`, `timing_scope` (`call`), then `ulp_mode`,
`n_scored` to `worst_arg` as in `accuracy.csv`.

**Boost short cut** (`build/boost_shortcut_gpu`): `gpu_boost_shortcut.csv` has one row per
point of the `boost_shortcut` set and per implementation: `schema`, `device` (`cpu`,
`cuda`), `supplier` (`boost-<version>`, `boost-gpu-<version>`,
`boost-gpu-<version>-adapted`), `precision` (`long-double`, `double`), `law`
(`boost_shortcut`), `nu`, `delta` (`nu - 2.5`), `z`, `k` (the computed `K_ν(z)`; these four
in C99 hexadecimal), `rel` (`|(k - hi) - lo| / |hi|`), `ref_bits`, `ulp_mode` and `ulp`
(the signed error of `K_ν` in ulp).
