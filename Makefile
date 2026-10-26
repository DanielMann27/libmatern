# ===========================================================================
# libmatern -- the normalised Matern covariance in double precision.
#
# The library (README.md)
#   make                  static library build/libmatern.a
#   make shared           build/libmatern.so.0 (soname libmatern.so.0) and the link
#                         build/libmatern.so; make install then installs both
#   make example          build/example, a program that builds a covariance matrix
#                         and factors it
#   make check            verification suite (needs a C compiler, libm and Python 3)
#   make install          headers and library into PREFIX (default /usr/local);
#                         DESTDIR=<dir> stages the files under <dir>
#   make cuda             CUDA backend object build/cuda.o (needs nvcc)
#   make cuda-modes       CUDA transfer-mode check (needs a GPU)
#   make cuda-device      check of include/matern_device.cuh and the device-pointer
#                         entries against the CUDA entries (needs a GPU)
#   make clean
#   parameters: CC, CXX, CFLAGS (replaces the defaults), OPTIMISE (default -O2), XCF (adds
#     flags), FMA=yes|no (default no; yes is for measurements and needs FMA3 wherever
#     the library runs), V=yes (prints every build command; by default each build
#     prints one short line), AR, PREFIX, DESTDIR, NVCC, GPU_ARCH (default sm_80),
#     NVCCFLAGS, XNVCF
#
# The parts of make check, one at a time
#   make headers          every kernel header compiled on its own (also part of make)
#   make contract | h-zero | crosscheck | positive-definite | direct-argument | threads
#   make exports | audit | neutral | readme-example
#   make strict           pedantic warning pass over the portable sources
#   make tsan             ThreadSanitizer build of the thread test, run on every backend
#                         this CPU runs (TSAN_CC: a compiler whose libtsan is installed,
#                         default gcc)
#
# The benchmark (docs/benchmark.md)
#   make requirements     external software: present, what it enables, how to add it
#   make reproduce        the full benchmark behind the published results on this machine:
#                         make pipeline with SETTINGS=full, EXTERNALS=yes and CORPUS=yes
#                         (durations: benchmark/durations.tsv; downloads not included;
#                         GPU=no runs the CPU part only, CPU=no the GPU part only); on a
#                         terminal it asks about KAUST after its copyright notice, without
#                         one it needs KAUST=yes or KAUST=no while the port is absent; the
#                         cross-machine figure needs a second run on a CPU of the other
#                         vendor
#                         (docs/benchmark.md, "Replicating the published results")
#   make pipeline         everything in one command: builds, CPU run, optional Levin,
#                         prefactor-fold, thread and GPU stages, figures, tables; on a
#                         terminal it first asks whether to switch to the full run
#                         behind the published results (durations: benchmark/durations.tsv)
#                         and whether to download the missing externals, KAUST and the
#                         corpus; without a terminal it asks nothing, and the switches
#                         give the answers
#   make benchmark-build  the drivers build/detect and build/benchmark_cpu
#   make build/suppliers  the driver of the external CPU suppliers; without it make
#                         benchmark skips the supplier stages
#   make benchmark        the CPU run, then the figures and tables; it builds the drivers
#                         of benchmark-build, not build/suppliers
#   make benchmark-test   the benchmark's test suite: unit tests and reduced runs of
#                         run_benchmark.sh (20+ min; needs the whole corpus)
#   make figures | tables the paper's figures (benchmark/plots/paper-final/, with
#                         captions.tex) and the tables its own tables are taken from
#                         (benchmark/tables/paper/)
#   make tables-check     compiles every generated table in one document (needs pdflatex)
#   make durations        benchmark/durations.tsv from the run records (TIMES.txt), and the
#                         full-run notice, docs/benchmark.md and README.md rendered from it
#   make python-info      the Python the generators use
#   make build/threads_cpu                     the thread-scaling driver (STAGES=threads)
#   make build/fold_cpu                        the prefactor-fold driver (STAGES=fold)
#   make levin-ab-build | levin-ab-build-gpu   the Levin A/B arms (STAGES=levin)
#   make benchmark-suppliers | benchmark-gpu   list the built suppliers | a short GPU run
#   make build/benchmark_gpu | build/suppliers_gpu | build/threads_gpu
#   make build/boost_shortcut_gpu              the GPU drivers (benchmark/run_gpu.sh)
#   make references       fetch the supplier sources, then report every supplier
#   make boost-headers    Boost 1.75.0 (CPU row) and 1.87.0 (GPU row) headers
#   make externals        download every downloadable external (Boost, GSL, Bessels.jl,
#                         NumPy and Matplotlib)
#   make kaust            the KAUST CuBesselK supplier: notice; DOWNLOAD=yes ports it,
#                         CLEAN=yes deletes the port
#   make corpus           the evaluation corpus (Zenodo): notice and state;
#                         DOWNLOAD=yes downloads the missing or damaged files
#   make corpus-check     every corpus file against the sha256 list of data/MANIFEST.md
#   make corpus-tools     build/corpus_convert, the corpus converter (docs/benchmark.md, The corpus)
#   parameters (switches take yes or no; also 1/0, on/off, true/false, y/n):
#     drivers    CXX, V=yes (prints every build command)
#     pipeline   SETTINGS=full, EXTERNALS=yes, KAUST=yes|no, CORPUS=yes (the answers
#                to the four questions, also without a terminal), LEVIN FOLD THREADS
#                CHECK FIGURES TABLES_CHECK=yes|no, GPU=auto|yes|no,
#                GPU_STAGES=core,levin,threads|all (+ call-large, never in all),
#                DRY=yes (prints the plan, the notice and the questions; downloads,
#                builds and runs nothing; exit status 2 where the run would refuse to
#                start)
#     runs       REPS FAMILIES BACKENDS STAGES LADDER_SET LADDER_PASS_FROM SUPPLIERS
#                SUP_LADDER SUP_LADDER_BUDGET THREADS_MAX NOTIFY ... ("Run parameters")
#     corpus     DATA=<dir> (the corpus root, default data/), CORPUS_URL=<mirror or
#                directory>
#     externals  BOOST_CPU_ROOT, BOOST_ROOT (=none disables a Boost row), BOOST_VERSIONS
#     python     PYTHON (the interpreter), PYTHON_CANDIDATES (the ones searched for it)
#     figures    ARGS=--list | --strict | --runs DIR | --cpu-machine NAME (the figures
#                and tables targets); tables also --only=<text> and --no-preview
#                (without pdflatex: the tables unfitted, no previews)
#
# For the maintainers, from a git checkout of libmatern (a release tarball has no git)
#   make dist             the code release: build/dist/libmatern-<version>.tar.gz at a
#                         v<version> tag, libmatern-<describe>.tar.gz at another commit
#   make clean-clone      build and check a copy of the committed tree (git archive HEAD)
#
# Used by other targets and scripts: supplier-objects, externals-missing.  make help
# prints this.
# Parameters are given as `make <target> NAME=value`; in full, with their defaults:
# docs/benchmark.md, "Make parameters".
# ===========================================================================

CC       ?= cc
CXX      ?= g++
AR       ?= ar
PREFIX   ?= /usr/local

# Switches take yes or no (also 1/0, on/off, true/false, y/n; in lower or upper case or
# capitalised).
YES_WORDS = yes y 1 on true Yes Y On True YES ON TRUE
NO_WORDS  = no n 0 off false No N Off False NO OFF FALSE
# $(call yes_no,NAME): yes or no for the switch NAME (empty is no); another value stops make.
yes_no = $(strip $(if $(filter $(YES_WORDS),$(strip $($(1)))),yes,$(if $(filter $(NO_WORDS),$(strip $($(1)))),no,$(if $(strip $($(1))),$(error $(1)=$($(1)): use yes or no (also 1/0, on/off, true/false, y/n)),no))))

# V=yes prints every build command; by default a build prints one short line (say).
Q    := $(if $(filter yes,$(call yes_no,V)),,@)
say   = $(if $(Q),@printf '  %-5s %s\n' '$(1)' '$@')
# For recipes that are shell loops: with V=yes the shell echoes each command.
V_SH := $(if $(Q),,set -x;)

WARNINGS        = -Wall -Wextra
WARNINGS_STRICT = $(WARNINGS) -Wconversion -Wshadow -Wcast-qual -Wpointer-arith

STANDARD  = -std=c11

# The optimisation level is part of the arithmetic: GCC fuses a written a*b + c
# into one FMA only at -O2, -O3 and -Os (Clang from -O1).  Appended outside CFLAGS.
OPTIMISE ?= -O2

# Contraction is part of the arithmetic and stated explicitly, since GCC's default
# follows the dialect (-std=c11 does not fuse) and _mm*_mul_pd/_mm*_add_pd are plain
# vector operators.  Any other build of these sources must pass it as well.
CONTRACTION = -ffp-contract=fast

CFLAGS   ?= $(STANDARD) $(WARNINGS) -fno-math-errno

# icx defaults to -fp-model=fast, which breaks the double-double arithmetic;
# -fp-model=precise must precede CONTRACTION, which it would otherwise reset.
FP_MODEL = $(eval FP_MODEL := $(if $(shell $(CC) -dM -E -x c /dev/null 2>/dev/null \
    | grep -q __INTEL_LLVM_COMPILER && echo icx),-fp-model=precise))$(FP_MODEL)

# A header change rebuilds the objects that include it.
DEPFLAGS = -MMD -MP

# For rules that compile to $@.tmp before objcopy.
DEPFIX     = -MT $@ -MF $(@:.o=.d)

# The static library links into PIE executables and shared objects.
PIC = -fPIC

# XCF adds flags without replacing CFLAGS, e.g. `make XCF=-DMATERN_WITHOUT_LEVIN`.
XCF =

# Appended outside CFLAGS, so overriding CFLAGS cannot remove them.
ALL_CFLAGS = $(CFLAGS) $(OPTIMISE) $(FP_MODEL) $(CONTRACTION) $(PIC) $(XCF) $(DEPFLAGS)

# Scalar objects only: loop heads aligned to 64 bytes, so that unrelated code edits do
# not move their short inner loops across a 64-byte line; no result changes.  Dropped
# for a compiler without the option.
LOOP_ALIGN = $(eval LOOP_ALIGN := $(shell $(CC) -falign-loops=64 -x c -c /dev/null \
    -o /dev/null >/dev/null 2>&1 && echo -falign-loops=64))$(LOOP_ALIGN)

# For -fsyntax-only checks, which write no dependency file.
SYNTAX_CFLAGS = $(CFLAGS) $(CONTRACTION)
INCLUDES  = -Iinclude -Isrc/backends -Isrc/backends/scalar -Isrc/numeric -Isrc/elementary -Isrc/bessel -Isrc/router

# -mno-avx512f keeps AVX-512 encodings out of the AVX2 object.
FLAGS_AVX512 = -mavx512f -mavx512dq -mfma
FLAGS_AVX2   = -mavx2 -mfma -mno-avx512f -DMATERN_TARGET_AVX2

# Objects that must run without AVX2 (dispatcher, workspace, CUDA stand-ins,
# version query, scalar and matched backends) target the x86-64 baseline.
PORTABLE_ARCH = -march=x86-64 -mtune=generic

# FMA=yes adds -mfma to scalar_portable and both matched objects, which changes their
# results and requires FMA3 on every machine.  dispatch.c and workspace.c never get
# -mfma: the dispatcher runs before the CPU feature check.  FMA_STATE (on/off) is what
# build/BUILD_KNOBS records.
FMA ?= no
ifeq ($(call yes_no,FMA),yes)
FLAGS_FMA = -mfma
FMA_STATE = on
else
FLAGS_FMA =
FMA_STATE = off
endif

SOURCES_PORTABLE = \
	src/backends/dispatch.c \
	src/backends/workspace.c \
	src/backends/cuda_absent.c \
	src/backends/version.c \
	src/backends/scalar/asymptotic_large_argument.c \
	src/backends/scalar/debye_uniform_order.c \
	src/backends/scalar/continued_fraction_wronskian.c \
	src/backends/scalar/temme_small_argument.c \
	src/backends/scalar/levin_transformed_asymptotic.c \
	src/backends/scalar/half_integer_closed_form.c \
	src/backends/scalar/region_router.c

OBJECTS_PORTABLE = $(SOURCES_PORTABLE:src/%.c=build/%.o)

# vector.c compiled per {avx512, avx2, matched, matched_fma} x {(h, nu, s), (z, nu)}
# entry; objcopy localises all but the entry, so the objects coexist in one binary.
OBJECTS_VECTOR = build/backends/vector_avx512.o build/backends/vector_avx2.o build/backends/vector_matched.o \
                 build/backends/vector_matched_fma.o \
                 build/backends/vector_avx512_z.o build/backends/vector_avx2_z.o \
                 build/backends/vector_matched_z.o build/backends/vector_matched_fma_z.o

# One more scalar build per instruction set (MATERN_BACKEND=scalar picks the best,
# dispatch.c): one ld -r object per set in which only its two entries stay global.
SOURCES_SCALAR = $(filter src/backends/scalar/%.c,$(SOURCES_PORTABLE))
SCALAR_BUILDS  = avx512 avx2 fma
OBJECTS_SCALAR = $(SCALAR_BUILDS:%=build/backends/scalar_%.o)
FLAGS_SCALAR_avx512 = $(FLAGS_AVX512)
FLAGS_SCALAR_avx2   = -mavx2 -mfma -mno-avx512f
FLAGS_SCALAR_fma    = $(PORTABLE_ARCH) -mfma

# Every library object, in the order the archives hold them.
OBJECTS_LIBRARY = $(OBJECTS_PORTABLE) $(OBJECTS_VECTOR) $(OBJECTS_SCALAR)

# libmatern.a: OBJECTS_LIBRARY linked into one object (ld -r) keeping only the names of
# matern.map global.  LIBRARY_INTERNAL keeps internal symbols global and is never
# installed (test/crosscheck.c, build/detect, benchmark/tests/run_tests.sh).
LIBRARY          = build/libmatern.a
LIBRARY_INTERNAL = build/libmatern_internal.a
LIBRARY_SYMBOLS  = build/matern.syms

# SOVERSION is the ABI version, not the release: raised only when programs linked
# against the previous library cannot run against the new one.
SOVERSION = 0
SONAME    = libmatern.so.$(SOVERSION)

# The unfolded twin of the scalar backend (src/backends/scalar/scalar_internal.h),
# -DMATERN_SCALAR_UNFOLDED=1 (direct) or =2 (exp-log), and `control`, the shipped source built
# the same way: objects for build/fold_cpu only, never in a library or a check.
FOLD_ISAS     = avx512 avx2 fma portable
FOLD_VARIANTS = control unfolded_direct unfolded_explog
FOLD_DEF_control         =
FOLD_DEF_unfolded_direct = -DMATERN_SCALAR_UNFOLDED=1
FOLD_DEF_unfolded_explog = -DMATERN_SCALAR_UNFOLDED=2
FLAGS_SCALAR_portable    = $(PORTABLE_ARCH) $(FLAGS_FMA)
$(foreach v,$(FOLD_VARIANTS),$(foreach i,$(FOLD_ISAS),\
    $(eval FLAGS_SCALAR_$(v)_$(i) = $$(FLAGS_SCALAR_$(i)) $$(FOLD_DEF_$(v)))))
OBJECTS_FOLD = $(foreach v,$(FOLD_VARIANTS),$(foreach i,$(FOLD_ISAS),build/backends/scalar_$(v)_$(i).o))

# In XCF or CFLAGS the twin's switches would reach the shipped objects.
ifneq ($(findstring MATERN_SCALAR_UNFOLDED,$(XCF) $(CFLAGS))$(findstring BENCH_FOLD,$(XCF) $(CFLAGS)),)
$(error MATERN_SCALAR_UNFOLDED and BENCH_FOLD belong to the build/fold_cpu rules; remove them from XCF and CFLAGS)
endif

# The backends every check runs on (MATERN_BACKEND); not named BACKENDS, the run
# parameter that make pipeline passes through.
CHECK_BACKENDS = avx512 avx2 scalar scalar_portable matched matched_fma

# PYTHON: the first candidate whose NumPy and Matplotlib are at or above the tested minimum
# (tools/py_stack.py), else python3; searched once, on first use.  The .venv path is
# absolute for scripts run elsewhere.
PYTHON_CANDIDATES ?= python3 /usr/bin/python3 python $(CURDIR)/.venv/bin/python3
ifeq ($(origin PYTHON),undefined)
PYTHON = $(eval PYTHON := $(or $(shell for p in $(PYTHON_CANDIDATES); do \
    command -v $$p >/dev/null 2>&1 && $$p $(CURDIR)/tools/py_stack.py \
    >/dev/null 2>&1 && { echo "$$p -B"; break; }; done),python3 -B))$(PYTHON)
endif

.PHONY: all headers shared example check contract h-zero crosscheck positive-definite \
        direct-argument threads neutral audit exports readme-example strict tsan install clean \
        clean-clone cuda cuda-modes references benchmark-build benchmark-suppliers \
        benchmark-gpu benchmark levin-ab-build levin-ab-build-gpu \
        figures tables tables-check durations \
        benchmark-test python-info requirements supplier-objects boost-headers pipeline externals \
        corpus-tools corpus-check kaust externals-missing corpus dist help reproduce

all: headers $(LIBRARY)

# Every kernel header compiled on its own (as C, the ISA-specific ones also as C++);
# -Wno-unused-function since a lone header leaves its static functions uncalled.  The C++
# pass is one shell line, so that its skip ends the recipe where $(CXX) does not compile a
# minimal C++17 program.
HEADER_WARNINGS = $(WARNINGS) -Wno-unused-function
HEADERS_AVX512 = $(wildcard src/*/*_avx512.h)
HEADERS_AVX2   = $(wildcard src/*/*_avx2.h)
HEADERS_NEUTRAL = $(filter-out $(HEADERS_AVX512) $(HEADERS_AVX2), $(wildcard src/*/*.h))

headers:
	@mkdir -p build
	@for header in $(HEADERS_AVX512) ; do \
	    printf '#include "%s"\nint main(void){return 0;}\n' "$$header" > build/.header_check.c ; \
	    $(CC) $(STANDARD) $(HEADER_WARNINGS) $(FLAGS_AVX512) $(INCLUDES) -I. \
	        -fsyntax-only build/.header_check.c || exit 1 ; \
	done
	@for header in $(HEADERS_AVX2) ; do \
	    printf '#include "%s"\nint main(void){return 0;}\n' "$$header" > build/.header_check.c ; \
	    $(CC) $(STANDARD) $(HEADER_WARNINGS) $(FLAGS_AVX2) $(INCLUDES) -I. \
	        -fsyntax-only build/.header_check.c || exit 1 ; \
	done
	@for header in $(HEADERS_NEUTRAL) ; do \
	    printf '#include "%s"\nint main(void){return 0;}\n' "$$header" > build/.header_check.c ; \
	    $(CC) $(STANDARD) $(HEADER_WARNINGS) $(FLAGS_AVX512) $(INCLUDES) -I. \
	        -fsyntax-only build/.header_check.c || exit 1 ; \
	    $(CC) $(STANDARD) $(HEADER_WARNINGS) $(FLAGS_AVX2) $(INCLUDES) -I. \
	        -fsyntax-only build/.header_check.c || exit 1 ; \
	done
	@rm -f build/.header_check.c
	@printf 'int main(){return 0;}\n' > build/.header_check.cpp; \
	if ! $(CXX) -std=c++17 -fsyntax-only build/.header_check.cpp >/dev/null 2>&1; then \
	    rm -f build/.header_check.cpp; \
	    echo "headers ok (no C++17 compiler for the second pass)"; exit 0; fi; \
	for header in $(HEADERS_AVX512) ; do \
	    printf '#include "%s"\nint main(){return 0;}\n' "$$header" > build/.header_check.cpp ; \
	    $(CXX) -std=c++17 $(HEADER_WARNINGS) $(FLAGS_AVX512) $(INCLUDES) -I. \
	        -fsyntax-only build/.header_check.cpp || exit 1 ; \
	done; \
	for header in $(HEADERS_AVX2) ; do \
	    printf '#include "%s"\nint main(){return 0;}\n' "$$header" > build/.header_check.cpp ; \
	    $(CXX) -std=c++17 $(HEADER_WARNINGS) $(FLAGS_AVX2) $(INCLUDES) -I. \
	        -fsyntax-only build/.header_check.cpp || exit 1 ; \
	done; \
	rm -f build/.header_check.cpp; \
	echo "headers ok: $(words $(HEADERS_AVX512)) AVX-512, $(words $(HEADERS_AVX2)) AVX2, $(words $(HEADERS_NEUTRAL)) ISA-neutral (each at both), as C; the ISA-specific headers also as C++"

# Archives are written from scratch with `ar ...D` (zero dates and owners), so
# identical objects give an identical archive.
$(LIBRARY): $(OBJECTS_LIBRARY) $(LIBRARY_SYMBOLS)
	@mkdir -p $(dir $@)
	$(Q)ld -r -o build/matern.o.tmp $(OBJECTS_LIBRARY)
	$(Q)objcopy --keep-global-symbols=$(LIBRARY_SYMBOLS) build/matern.o.tmp build/matern.o
	@rm -f build/matern.o.tmp $@
	$(Q)$(AR) rcsD $@ build/matern.o
	@echo "built $@"

$(LIBRARY_INTERNAL): $(OBJECTS_LIBRARY)
	@mkdir -p $(dir $@)
	@rm -f $@
	@$(AR) rcsD $@ $^
	@echo "built $@"

# The global names of matern.map, one per line.
$(LIBRARY_SYMBOLS): matern.map
	@mkdir -p $(dir $@)
	$(Q)sed -n 's/^[[:space:]]*\(matern_[A-Za-z0-9_]*\);[[:space:]]*$$/\1/p' $< > $@

# Scalar kernels: portable floor, optionally with FMA.
build/backends/scalar/%.o: src/backends/scalar/%.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(LOOP_ALIGN) $(PORTABLE_ARCH) $(FLAGS_FMA) $(INCLUDES) -c $< -o $@

# One scalar build per instruction set: entries named per set through
# MATERN_SCALAR_ENTRY/_Z, linked with ld -r; every other symbol is made local.
build/backends/scalar_%.o: $(SOURCES_SCALAR)
	@mkdir -p build/backends/scalar_$*
	$(call say,CC)
	@$(V_SH) for src in $(SOURCES_SCALAR); do \
	    obj=build/backends/scalar_$*/$$(basename $$src .c).o; \
	    $(CC) $(ALL_CFLAGS) $(LOOP_ALIGN) $(FLAGS_SCALAR_$*) -MT $@ -MF $${obj%.o}.d \
	        -DMATERN_SCALAR_ENTRY=matern_evaluate_scalar_$* \
	        -DMATERN_SCALAR_ENTRY_Z=matern_evaluate_scalar_$*_z \
	        $(INCLUDES) -c $$src -o $$obj || exit 1; \
	done
	$(Q)ld -r -o $@.tmp $(SOURCES_SCALAR:src/backends/scalar/%.c=build/backends/scalar_$*/%.o)
	$(Q)objcopy --keep-global-symbol=matern_evaluate_scalar_$* \
	        --keep-global-symbol=matern_evaluate_scalar_$*_z $@.tmp $@
	@rm -f $@.tmp

# Other portable objects (dispatch.c, workspace.c, cuda_absent.c, version.c).
build/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(PORTABLE_ARCH) $(INCLUDES) -c $< -o $@

build/backends/vector_avx512.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(DEPFIX) $(FLAGS_AVX512) $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_avx512 $@.tmp $@
	@rm -f $@.tmp

# `matched`: the vector source at width 1 (see src/backends/internal.h).
build/backends/vector_matched.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(PORTABLE_ARCH) $(FLAGS_FMA) $(DEPFIX) -DMATERN_TARGET_SCALAR $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_matched $@.tmp $@
	@rm -f $@.tmp

build/backends/vector_avx2.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(DEPFIX) $(FLAGS_AVX2) $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_avx2 $@.tmp $@
	@rm -f $@.tmp

# `matched_fma`: width 1 with hardware FMA, selected only when cpuid reports FMA.
build/backends/vector_matched_fma.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(PORTABLE_ARCH) -mfma $(DEPFIX) -DMATERN_TARGET_SCALAR -DMATERN_TARGET_FMA $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_matched_fma $@.tmp $@
	@rm -f $@.tmp

# The (z, nu) entry: -DMATERN_ENTRY_Z makes the argument derivation an identity.
build/backends/vector_avx512_z.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(DEPFIX) $(FLAGS_AVX512) -DMATERN_ENTRY_Z $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_avx512_z $@.tmp $@
	@rm -f $@.tmp

build/backends/vector_matched_z.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(PORTABLE_ARCH) $(FLAGS_FMA) $(DEPFIX) -DMATERN_TARGET_SCALAR -DMATERN_ENTRY_Z $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_matched_z $@.tmp $@
	@rm -f $@.tmp

build/backends/vector_matched_fma_z.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(PORTABLE_ARCH) -mfma $(DEPFIX) -DMATERN_TARGET_SCALAR -DMATERN_TARGET_FMA -DMATERN_ENTRY_Z $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_matched_fma_z $@.tmp $@
	@rm -f $@.tmp

build/backends/vector_avx2_z.o: src/backends/vector.c
	@mkdir -p $(dir $@)
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) $(DEPFIX) $(FLAGS_AVX2) -DMATERN_ENTRY_Z $(INCLUDES) -c $< -o $@.tmp
	$(Q)objcopy --localize-hidden --keep-global-symbol=matern_evaluate_avx2_z $@.tmp $@
	@rm -f $@.tmp

# The shared library: the objects of the static one, exporting only matern.map.
shared: build/$(SONAME)

build/$(SONAME): $(OBJECTS_LIBRARY) matern.map
	@mkdir -p build
	$(Q)$(CC) -shared -Wl,-soname,$(SONAME) -Wl,--version-script=matern.map \
	    -o $@ $(OBJECTS_LIBRARY) -lm
	$(Q)ln -sf $(SONAME) build/libmatern.so
	@echo "built $@ (soname $(SONAME)) and the link build/libmatern.so"

example: $(LIBRARY)
	$(Q)$(CC) $(ALL_CFLAGS) $(INCLUDES) -o build/example \
	    example/covariance_matrix.c $(LIBRARY) -lm
	@echo "run ./build/example"

# Per-backend tests run on each backend of CHECK_BACKENDS this CPU runs.
check: $(LIBRARY)
	@echo "== kernel headers ==========================================="
	@$(MAKE) --no-print-directory headers
	@echo "== contract ================================================="
	@$(MAKE) --no-print-directory contract
	@echo "== h = 0 ====================================================="
	@$(MAKE) --no-print-directory h-zero
	@echo "== the scalar builds against matched ========================"
	@$(MAKE) --no-print-directory crosscheck
	@echo "== positive definiteness ===================================="
	@$(MAKE) --no-print-directory positive-definite
	@echo "== the (z, nu) entry ========================================"
	@$(MAKE) --no-print-directory direct-argument
	@echo "== thread safety ============================================"
	@$(MAKE) --no-print-directory threads
	@echo "== static checks ============================================"
	@$(MAKE) --no-print-directory exports
	@$(MAKE) --no-print-directory audit
	@$(MAKE) --no-print-directory neutral
	@echo "== README example ==========================================="
	@$(MAKE) --no-print-directory readme-example
	@echo ""
	@echo "check passed."

# Prints the backend the dispatcher selects in its process.
build/backend_name: test/backend_name.c $(LIBRARY)
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT $@ -MF $@.d $(INCLUDES) -o $@ test/backend_name.c $(LIBRARY) -lm

# $(call per_backend,<program>): a backend the dispatcher replaces with another
# (the CPU cannot run it) is reported as skipped rather than tested twice.
per_backend = for backend in $(CHECK_BACKENDS); do \
	    live=$$(MATERN_BACKEND=$$backend ./build/backend_name) || exit 1; \
	    if [ "$$live" != "$$backend" ]; then \
	        echo "  skipped $$backend: this CPU cannot run it (the dispatcher chose $$live)"; \
	        continue; \
	    fi; \
	    MATERN_BACKEND=$$backend $(1) || exit 1; \
	done

contract: $(LIBRARY) build/backend_name
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT build/contract -MF build/contract.d \
	    $(INCLUDES) -o build/contract test/contract.c $(LIBRARY) -lm
	@$(call per_backend,./build/contract)

h-zero: $(LIBRARY) build/backend_name
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT build/degenerate_h_zero -MF build/degenerate_h_zero.d \
	    $(INCLUDES) -o build/degenerate_h_zero test/degenerate_h_zero.c $(LIBRARY) -lm
	@$(call per_backend,./build/degenerate_h_zero)
	@echo "value checks pass on every backend this CPU runs: h = 0 is exactly 1.0"

# One process per scalar build (one backend per process); `matched` is an
# internal entry, hence LIBRARY_INTERNAL.
crosscheck: $(LIBRARY_INTERNAL)
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT build/crosscheck -MF build/crosscheck.d $(INCLUDES) \
	    -o build/crosscheck test/crosscheck.c $(LIBRARY_INTERNAL) -lm
	@MATERN_BACKEND=scalar ./build/crosscheck
	@echo ""
	@MATERN_BACKEND=scalar_portable ./build/crosscheck

positive-definite: $(LIBRARY) build/backend_name
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT build/positive_definite -MF build/positive_definite.d \
	    $(INCLUDES) -o build/positive_definite test/positive_definite.c $(LIBRARY) -lm
	@$(call per_backend,./build/positive_definite)

direct-argument: $(LIBRARY) build/backend_name
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT build/direct_argument -MF build/direct_argument.d \
	    $(INCLUDES) -o build/direct_argument test/direct_argument.c $(LIBRARY) -lm
	@$(call per_backend,./build/direct_argument)

# The positive control (one workspace shared by all threads) must fail, by differing
# results or a crash.  It runs on the automatic backend and is skipped where that is
# matched or matched_fma, which use no workspace.
threads: $(LIBRARY) build/backend_name
	@mkdir -p build
	@$(CC) $(ALL_CFLAGS) -MT build/thread_safety -MF build/thread_safety.d \
	    $(INCLUDES) -pthread -o build/thread_safety test/thread_safety.c $(LIBRARY) -lm
	@$(call per_backend,./build/thread_safety)
	@live=$$(MATERN_BACKEND= ./build/backend_name); \
	 case "$$live" in matched*) echo "control: skipped, the automatic backend ($$live) uses no workspace"; exit 0;; esac; \
	 if MATERN_BACKEND= MATERN_THREAD_CONTROL=1 ./build/thread_safety > build/thread_control.log 2>&1; \
	 then echo "control: with one workspace shared by all threads, which matern.h forbids, the test passed;"; \
	      echo "  it cannot detect a shared workspace (build/thread_control.log)"; exit 1; \
	 else rc=$$?; say="control: with one workspace shared by all threads, which matern.h forbids, the test"; \
	      if grep -q '^control fired' build/thread_control.log; then echo "$$say reported differing results, as it must"; \
	      elif [ $$rc -gt 128 ]; then echo "$$say crashed (signal $$((rc - 128))), as it must"; \
	      else echo "$$say stopped with exit status $$rc without reporting differences;"; \
	           echo "  it did not run (build/thread_control.log)"; exit 1; fi; fi

# Every region kernel must compile at width 1; -Werror=implicit-function-declaration
# makes a missing vocabulary entry an error.
NEUTRAL_KERNELS = asymptotic_large_argument temme_small_argument \
                  continued_fraction_wronskian half_integer_closed_form \
                  derive_argument debye_uniform_order levin_transformed_asymptotic

neutral:
	@mkdir -p build
	@fail=0; \
	for k in $(NEUTRAL_KERNELS); do \
	    printf '#include "%s.h"\nint main(void){return 0;}\n' $$k > build/neutral.c; \
	    if $(CC) $(SYNTAX_CFLAGS) -Werror=implicit-function-declaration \
	         -DMATERN_TARGET_SCALAR $(INCLUDES) -fsyntax-only build/neutral.c 2>/dev/null; \
	    then echo "  ok    $$k compiles at width 1"; \
	    else echo "  FAIL  $$k does not compile at width 1"; fail=1; fi; \
	done; \
	exit $$fail

audit:
	@$(PYTHON) tools/audit.py

# libmatern.a defines exactly the names of matern.map, the CUDA ones weak.  Read
# from the map, not LIBRARY_SYMBOLS, so a line the extraction misses is reported.
exports: $(LIBRARY)
	@nm -g --defined-only $(LIBRARY) | awk 'NF == 3 { print $$3 }' | LC_ALL=C sort > build/exports_library.txt
	@awk '/^[[:space:]]*global:/ { g = 1; next } /^[[:space:]]*local:/ { g = 0 } g' matern.map \
	    | tr -d ' \t' | tr ';' '\n' | grep -v '^$$' | LC_ALL=C sort > build/exports_map.txt
	@extra=$$(LC_ALL=C comm -13 build/exports_map.txt build/exports_library.txt); \
	 missing=$$(LC_ALL=C comm -23 build/exports_map.txt build/exports_library.txt); \
	 strong=$$(nm -g --defined-only $(LIBRARY) | awk 'NF == 3 && $$3 ~ /cuda/ && $$2 != "W" { print $$3 }'); \
	 weak=$$(nm -g --defined-only $(LIBRARY) | awk 'NF == 3 && $$3 ~ /cuda/ && $$2 == "W"' | wc -l); \
	 [ -n "$$extra" ] && { echo "  FAIL  $(LIBRARY) exports names that matern.map does not list:"; echo "$$extra" | sed 's/^/          /'; }; \
	 [ -n "$$missing" ] && { echo "  FAIL  $(LIBRARY) does not define names that matern.map lists:"; echo "$$missing" | sed 's/^/          /'; }; \
	 [ -n "$$strong" ] && { echo "  FAIL  CUDA entries that are not weak in $(LIBRARY):"; echo "$$strong" | sed 's/^/          /'; }; \
	 if [ -n "$$extra$$missing$$strong" ]; then exit 1; fi; \
	 echo "  ok    $(LIBRARY) defines the $$(wc -l < build/exports_map.txt) names of matern.map and no other global symbol; the $$weak CUDA entries are weak"
	@if [ -f build/cuda.o ]; then \
	     nm -g --defined-only build/cuda.o | awk 'NF == 3 { print $$3 }' | LC_ALL=C sort > build/exports_cuda.txt; \
	     extra=$$(LC_ALL=C comm -13 build/exports_map.txt build/exports_cuda.txt); \
	     if [ -n "$$extra" ]; then echo "  FAIL  build/cuda.o exports names that matern.map does not list:"; \
	         echo "$$extra" | sed 's/^/          /'; exit 1; fi; \
	     echo "  ok    build/cuda.o exports $$(wc -l < build/exports_cuda.txt) names, each listed in matern.map"; \
	 fi

# The README.md example must compile and print the output shown there.
readme-example: $(LIBRARY)
	@$(PYTHON) tools/readme_example.py

strict:
	@for source in $(SOURCES_PORTABLE); do \
	    echo "== $$source"; \
	    $(CC) $(STANDARD) $(WARNINGS_STRICT) $(INCLUDES) -fsyntax-only $$source \
	        || exit 1; \
	done
	@for v in 1 2; do for source in $(SOURCES_SCALAR); do \
	    echo "== $$source (MATERN_SCALAR_UNFOLDED=$$v)"; \
	    $(CC) $(STANDARD) $(WARNINGS_STRICT) $(INCLUDES) -DMATERN_SCALAR_UNFOLDED=$$v \
	        -DMATERN_SCALAR_ENTRY=matern_evaluate_scalar_strict -DMATERN_SCALAR_ENTRY_Z=matern_evaluate_scalar_strict_z \
	        -fsyntax-only $$source || exit 1; \
	done; done
	@echo "strict warning pass clean"

# Prints the comment block at the top of this file (lines 2 to the first fence).
help:
	@sed -n '2,/^# =====/p' $(firstword $(MAKEFILE_LIST)) | sed '$$d' | sed 's/^# \{0,1\}//'

# dist and clean-clone need a git checkout whose top level is this directory (inside
# another repository git answers for that one); -ef also accepts a symlinked path.
GIT_TOP_CHECK = top=$$(git rev-parse --show-toplevel 2>/dev/null) && [ "$$top" -ef . ] || { \
	    echo "make $@: needs a git checkout of libmatern with this directory as its top level;"; \
	    echo "  a release tarball has none -- this target is for the maintainers"; exit 2; }

# Builds and checks a copy of the committed tree (git archive HEAD).
clean-clone:
	@$(GIT_TOP_CHECK)
	@set -e; \
	tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	git archive HEAD | tar -x -C "$$tmp"; \
	$(MAKE) -C "$$tmp" --no-print-directory >/dev/null; echo "  built"; \
	if $(MAKE) -C "$$tmp" --no-print-directory check > "$$tmp/_check.log" 2>&1; then \
	    tail -1 "$$tmp/_check.log"; \
	else \
	    echo "  clean clone failed make check:"; tail -25 "$$tmp/_check.log"; exit 1; \
	fi; \
	$(MAKE) -C "$$tmp" --no-print-directory example >/dev/null && \
	    ( cd "$$tmp" && ./build/example >/dev/null ) && echo "  example ok"

# HEAD as build/dist/libmatern-<git describe without v>.tar.gz and its sha256.
# Refuses uncommitted changes; warns when the tag differs from MATERN_VERSION_STRING.
dist:
	@$(GIT_TOP_CHECK)
	@[ -z "$$(git status --porcelain --untracked-files=no)" ] || { \
	    echo "make dist: uncommitted changes -- commit them first (the tarball holds HEAD only)"; exit 2; }
	@d=$$(git describe --tags --always --match 'v[0-9]*'); v=$${d#v}; n=libmatern-$$v; \
	 t=$$(git describe --tags --exact-match --match 'v[0-9]*' HEAD 2>/dev/null); \
	 r=$$(sed -n 's/^#define MATERN_VERSION_STRING[[:space:]]*"\([^"]*\)".*/\1/p' include/matern.h); \
	 if [ -n "$$t" ] && [ "$${t#v}" != "$$r" ]; then \
	     echo "make dist: warning: HEAD is tagged $$t, but include/matern.h states the version $$r" >&2; fi; \
	 mkdir -p build/dist && \
	 git archive --format=tar --prefix=$$n/ HEAD | gzip -9n > build/dist/$$n.tar.gz && \
	 ( cd build/dist && { sha256sum $$n.tar.gz 2>/dev/null || shasum -a 256 $$n.tar.gz; } > $$n.tar.gz.sha256 ) && \
	 echo "built build/dist/$$n.tar.gz ($$(du -h build/dist/$$n.tar.gz | cut -f1)), sha256 $$(cut -d' ' -f1 build/dist/$$n.tar.gz.sha256)" && \
	 echo "  the corpus is not in it: make corpus DOWNLOAD=yes (doi:10.5281/zenodo.23393956)"

# ThreadSanitizer build of the library and the thread test, probed first with an
# empty program.  The sed below rewrites three defines of test/thread_safety.c and
# depends on their exact spacing.
TSAN_CC   ?= gcc
TSAN_FLAGS = -std=c11 -O1 -g -fsanitize=thread $(CONTRACTION) -fno-math-errno
tsan:
	@mkdir -p build/tsan
	@printf 'int main(void){return 0;}\n' > build/tsan/probe.c
	@$(TSAN_CC) -fsanitize=thread build/tsan/probe.c -o build/tsan/probe >/dev/null 2>&1 || { \
	    echo "make tsan: $(TSAN_CC) cannot link -fsanitize=thread: its ThreadSanitizer runtime (libtsan) is not installed,"; \
	    echo "  or $(TSAN_CC) is not found.  Install libtsan, or name another compiler: make tsan TSAN_CC=clang"; exit 2; }
	@./build/tsan/probe >/dev/null 2>&1 || { \
	    echo "make tsan: a program built with $(TSAN_CC) -fsanitize=thread does not run here"; \
	    echo "  (./build/tsan/probe shows why); try another compiler: make tsan TSAN_CC=clang"; exit 2; }
	@sed -e 's/THREAD_COUNT   8/THREAD_COUNT   4/' \
	     -e 's/POINT_COUNT    200000/POINT_COUNT    20000/' \
	     -e 's/REPEAT_COUNT   6/REPEAT_COUNT   2/' \
	     test/thread_safety.c > build/tsan/thread_safety.c
	@for v in "avx512:$(FLAGS_AVX512)" "avx2:$(FLAGS_AVX2)" \
	         "matched:-DMATERN_TARGET_SCALAR" \
	         "matched_fma:-mfma -DMATERN_TARGET_SCALAR -DMATERN_TARGET_FMA" ; do \
	    name=$${v%%:*}; flags=$${v#*:}; \
	    for entry in "" _z; do \
	        extra=; [ -n "$$entry" ] && extra=-DMATERN_ENTRY_Z; \
	        $(TSAN_CC) $(TSAN_FLAGS) $$flags $$extra $(INCLUDES) -c src/backends/vector.c \
	            -o build/tsan/vector_$$name$$entry.o.tmp || exit 1; \
	        objcopy --localize-hidden --keep-global-symbol=matern_evaluate_$$name$$entry \
	            build/tsan/vector_$$name$$entry.o.tmp build/tsan/vector_$$name$$entry.o || exit 1; \
	    done; \
	done
	@for name in $(SCALAR_BUILDS); do \
	    case $$name in avx512) flags="$(FLAGS_SCALAR_avx512)";; avx2) flags="$(FLAGS_SCALAR_avx2)";; \
	                   *) flags="$(FLAGS_SCALAR_fma)";; esac; \
	    mkdir -p build/tsan/scalar_$$name; \
	    for src in $(SOURCES_SCALAR); do \
	        $(TSAN_CC) $(TSAN_FLAGS) $$flags -DMATERN_SCALAR_ENTRY=matern_evaluate_scalar_$$name \
	            -DMATERN_SCALAR_ENTRY_Z=matern_evaluate_scalar_$${name}_z $(INCLUDES) \
	            -c $$src -o build/tsan/scalar_$$name/$$(basename $$src .c).o || exit 1; \
	    done; \
	    ld -r -o build/tsan/scalar_$$name.o.tmp build/tsan/scalar_$$name/*.o || exit 1; \
	    objcopy --keep-global-symbol=matern_evaluate_scalar_$$name \
	        --keep-global-symbol=matern_evaluate_scalar_$${name}_z \
	        build/tsan/scalar_$$name.o.tmp build/tsan/scalar_$$name.o || exit 1; \
	done
	@$(TSAN_CC) $(TSAN_FLAGS) $(INCLUDES) -pthread -o build/tsan/thread_safety \
	    build/tsan/thread_safety.c $(SOURCES_PORTABLE) build/tsan/vector_*.o \
	    $(SCALAR_BUILDS:%=build/tsan/scalar_%.o) -lm
	@for backend in $(CHECK_BACKENDS); do \
	    report=build/tsan/report_$$backend.txt; \
	    MATERN_BACKEND=$$backend ./build/tsan/thread_safety > $$report 2>&1; status=$$?; \
	    live=$$(sed -n 's/^thread_safety: backend \([a-z0-9_]*\).*/\1/p' $$report | head -1); \
	    if [ -n "$$live" ] && [ "$$live" != "$$backend" ]; then \
	        echo "  skipped $$backend: this CPU cannot run it (the dispatcher chose $$live)"; \
	        continue; \
	    fi; \
	    echo "backend $$backend"; \
	    if grep -q "data race" $$report; then \
	        echo "ThreadSanitizer found a race on $$backend ($$report):"; \
	        grep -A 4 "WARNING: ThreadSanitizer" $$report | head -20; \
	        exit 1; \
	    fi; \
	    if [ $$status -ne 0 ]; then \
	        echo "the thread test failed on $$backend (exit $$status, $$report):"; \
	        tail -20 $$report; \
	        exit 1; \
	    fi; \
	    grep -E "ok:|holds" $$report; \
	done
	@echo "ThreadSanitizer clean on every backend this CPU runs"

# Installs under $(DESTDIR)$(PREFIX); build/$(SONAME) and build/cuda.o, when they
# exist, are brought up to date and installed too (the CUDA object as matern_cuda.o).
install: $(LIBRARY) $(if $(wildcard build/cuda.o),build/cuda.o) $(if $(wildcard build/$(SONAME)),build/$(SONAME))
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib
	install -m 644 include/matern.h include/matern_device.cuh $(DESTDIR)$(PREFIX)/include/
	install -d $(DESTDIR)$(PREFIX)/include/matern_device
	install -m 644 include/matern_device/*.cuh $(DESTDIR)$(PREFIX)/include/matern_device/
	install -m 644 $(LIBRARY) $(DESTDIR)$(PREFIX)/lib/
	@if [ -f build/$(SONAME) ]; then install -m 755 build/$(SONAME) $(DESTDIR)$(PREFIX)/lib/ && \
	    ln -sf $(SONAME) $(DESTDIR)$(PREFIX)/lib/libmatern.so && \
	    echo "installed $(DESTDIR)$(PREFIX)/lib/$(SONAME) and the link libmatern.so (see README.md, Using it from your own project)"; fi
	@if [ -f build/cuda.o ]; then install -m 644 build/cuda.o $(DESTDIR)$(PREFIX)/lib/matern_cuda.o && \
	    echo "installed $(DESTDIR)$(PREFIX)/lib/matern_cuda.o (see README.md, CUDA)"; fi

-include $(shell find build -name '*.d' 2>/dev/null)
# Dependency files are only read: without this rule make tries to remake them, and
# the scalar-build pattern rule matches through the built-in `%: %.o` rule.
%.d: ;

clean:
	rm -rf build

# CUDA backend, not part of the default build.  nvcc from PATH, else the toolkit's
# default location (sudo resets PATH).
NVCC       ?= $(or $(shell command -v nvcc 2>/dev/null),$(wildcard /usr/local/cuda/bin/nvcc),nvcc)
# Every rule that runs nvcc checks for it first, so a missing toolkit is named.
NVCC_CHECK = command -v $(firstword $(NVCC)) >/dev/null 2>&1 || { \
	    echo "nvcc not found ($(firstword $(NVCC))): install the CUDA toolkit or put nvcc on PATH,"; \
	    echo "  or set NVCC=<path to nvcc>.  Do not build with sudo: sudo resets PATH."; exit 1; }
GPU_ARCH   ?= sm_80
# -fopenmp: the RING transfer mode threads its staging copy.
NVCCFLAGS  ?= -O2 -arch=$(GPU_ARCH) --expt-relaxed-constexpr -Xcompiler -fopenmp

# XNVCF adds flags to the device compile without replacing NVCCFLAGS.
XNVCF =

DEVICE_HEADERS = include/matern_device.cuh $(wildcard include/matern_device/*.cuh)

build/cuda.o: src/backends/cuda.cu $(DEVICE_HEADERS) include/matern.h
	@mkdir -p build
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) $(NVCCFLAGS) $(XNVCF) -c $< -o $@

# The benchmark drivers' copy: the same kernels plus the hooks of
# MATERN_CUDA_BENCHMARK_HOOKS (src/backends/cuda.cu), which build/cuda.o does not export.
build/cuda_bench.o: src/backends/cuda.cu $(DEVICE_HEADERS) include/matern.h
	@mkdir -p build
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) $(NVCCFLAGS) $(XNVCF) -DMATERN_CUDA_BENCHMARK_HOOKS -c $< -o $@

cuda: build/cuda.o
	@echo "built build/cuda.o"

# Every transfer mode returns the bits of SYNC and leaves the caller's buffers
# registered as before.
cuda-modes: build/cuda.o test/cuda_modes.cu
	@mkdir -p build
	@$(NVCC_CHECK)
	$(Q)$(NVCC) $(NVCCFLAGS) -Iinclude -o build/cuda_modes test/cuda_modes.cu build/cuda.o
	@./build/cuda_modes

# The device header in two translation units of one program (no duplicate symbols, its
# own constant tables in each), against the CUDA entries bit for bit; the device-pointer
# entries on a stream of their own.
cuda-device: build/cuda.o $(LIBRARY) test/cuda_device.cu test/cuda_device_second.cu $(DEVICE_HEADERS)
	@mkdir -p build
	@$(NVCC_CHECK)
	$(Q)$(NVCC) $(NVCCFLAGS) -Iinclude -o build/cuda_device test/cuda_device.cu \
	    test/cuda_device_second.cu build/cuda.o $(LIBRARY) -lm
	@./build/cuda_device

# Benchmark (docs/benchmark.md).  Every optional component (C++, GSL, Boost, KAUST,
# nvcc, a GPU) is detected; an absent one removes its rows instead of failing.

# `make references` fetches the public upstream sources and reports every supplier;
# its status is fetch.sh's (a digest mismatch fails, an absent supplier does not).
REQ_ENV = PYTHON="$(PYTHON)" PYTHON_CANDIDATES="$(PYTHON_CANDIDATES)" CC="$(CC)" CXX="$(CXX)" NVCC="$(NVCC)" \
          NVIDIA_SMI="$(NVIDIA_SMI)" \
          BOOST_CPU_ROOT="$(BOOST_CPU_ROOT)" BOOST_ROOT="$(BOOST_ROOT)" \
          PYTHON_ORIGIN="$(origin PYTHON)" CC_ORIGIN="$(origin CC)" CXX_ORIGIN="$(origin CXX)" \
          NVCC_ORIGIN="$(origin NVCC)" DATA="$(DATA)"

references:
	@bash references/fetch.sh; status=$$?; \
	$(REQ_ENV) bash tools/requirements.sh suppliers; exit $$status

# One line `key|name|add|detail` per absent component (benchmark/tools/missing.sh reads it).
externals-missing:
	@$(REQ_ENV) bash tools/requirements.sh missing

# Boost headers into references/boost/<version>/: 1.75.0 for the CPU row and 1.87.0
# for the GPU row, the versions of the published rows.
BOOST_VERSIONS ?= 1.75.0 1.87.0
boost-headers:
	@bash references/fetch_boost.sh $(BOOST_VERSIONS)
	@CXX="$(CXX)" bash benchmark/tools/boost_detect.sh | sed -n '/^cpu row/,$$p'

# Every downloadable external, without root (tools/externals.sh).
externals:
	@$(REQ_ENV) MAKE="$(MAKE)" bash tools/externals.sh

# Everything external the build, the benchmark and its generators can use.
requirements:
	@$(REQ_ENV) bash tools/requirements.sh all

# KAUST CuBesselK ("All rights reserved", not distributed): prints the copyright
# notice; DOWNLOAD=yes downloads and ports it, CLEAN=yes deletes the port.  It is
# downloaded only on an explicit yes: DOWNLOAD=yes here, or KAUST=yes or the question of
# make pipeline and make reproduce.
kaust:
	@case "$(call yes_no,DOWNLOAD)$(call yes_no,CLEAN)" in \
	   yesyes) echo "make kaust: DOWNLOAD=yes and CLEAN=yes exclude each other"; exit 2 ;; \
	   yesno)  PYTHON="$(PYTHON)" bash references/kaust_logbesselk/kaust.sh download ;; \
	   noyes)  bash references/kaust_logbesselk/kaust.sh clean ;; \
	   *)      bash references/kaust_logbesselk/kaust.sh notice ;; \
	 esac

# The evaluation corpus (data/MANIFEST.md, doi:10.5281/zenodo.23393956): the notice and
# state; DOWNLOAD=yes downloads the missing or damaged files (tools/corpus.sh).
corpus:
	@if [ "$(call yes_no,DOWNLOAD)" = yes ]; then \
	     DATA="$(DATA)" sh tools/corpus.sh download; \
	 else \
	     DATA="$(DATA)" sh tools/corpus.sh notice; \
	 fi

# CPU suppliers are detected only in references/Makefile (build/defines.txt, link.txt);
# supplier-objects is phony so the drivers relink when the configuration changes.
# KAUST is built only when its port exists; no build rule downloads or ports it.
KAUST_DERIVED = references/build/kaust/kaust_derived.h
KAUST_PORT   := $(wildcard $(KAUST_DERIVED))

supplier-objects:
	@$(MAKE) --no-print-directory -C references all

BENCH_CFLAGS = $(ALL_CFLAGS)
BENCH_INC    = -Ibenchmark -Ibenchmark/common $(INCLUDES)
BENCH_HDRS  := $(wildcard benchmark/common/*.h) benchmark/corpus_dataset.h benchmark/corpus_bin.h \
               references/common/build_info.h

# The make variables as resolved here (value, origin, compiler), read into
# PROVENANCE.txt by run_benchmark.sh.  Rewritten only when its content changes and
# order-only, so a change does not relink the drivers.
build/BUILD_KNOBS: build-knobs-force
	@mkdir -p build
	@{ printf '%-18s %s (origin %s) -> %s: %s\n' make_cc '$(CC)' '$(origin CC)' \
	       "$$(command -v $(firstword $(CC)) 2>/dev/null || echo 'not found')" \
	       "$$($(CC) --version 2>/dev/null | head -1)"; \
	   printf '%-18s %s (origin %s) -> %s: %s\n' make_cxx '$(CXX)' '$(origin CXX)' \
	       "$$(command -v $(firstword $(CXX)) 2>/dev/null || echo 'not found')" \
	       "$$($(CXX) --version 2>/dev/null | head -1)"; \
	   printf '%-18s %s (origin %s): %s\n' make_nvcc '$(NVCC)' '$(origin NVCC)' \
	       "$$($(NVCC) --version 2>/dev/null | sed -n 's/.*release /release /p' | head -1)"; \
	   printf '%-18s %s (origin %s)\n' make_fma '$(FMA_STATE)' '$(origin FMA)'; \
	   printf '%-18s %s\n' make_cflags '$(strip $(CFLAGS) $(OPTIMISE) $(CONTRACTION) $(XCF))'; \
	   printf '%-18s %s\n' make_loop_align '$(or $(LOOP_ALIGN),none) -- scalar objects only'; \
	   printf '%-18s %s -- build/cuda.o, build/cuda_bench.o and build/kaust_dev.o (NVCCFLAGS: %s)\n' \
	       make_gpu_arch '$(GPU_ARCH)' '$(NVCCFLAGS)'; \
	   printf '%-18s %s -- the GPU drivers (from nvidia-smi at build time, else GPU_ARCH)\n' \
	       make_gpu_arch_use '$(GPU_ARCH_USE)'; \
	 } > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

.PHONY: build-knobs-force
build-knobs-force:

# The corpus converter: text <-> binary and the byte-exact round-trip check.
corpus-tools: build/corpus_convert

# Every corpus file against the sha256 list of data/MANIFEST.md; an absent file fails it.
corpus-check:
	@bash tools/corpus_check.sh "$(or $(DATA),data)"
build/corpus_convert: tools/corpus_convert.c benchmark/corpus_bin.h
	@mkdir -p build
	$(call say,CC)
	$(Q)$(CC) $(ALL_CFLAGS) -Ibenchmark -o $@ tools/corpus_convert.c

# detect calls the internal matern_scalar_isa() (LIBRARY_INTERNAL) and reads the
# compiler stamps of LIBRARY, the archive the drivers link.
build/detect: benchmark/detect.c benchmark/corpus_dataset.c $(LIBRARY) $(LIBRARY_INTERNAL) $(BENCH_HDRS) | build/BUILD_KNOBS
	@mkdir -p build
	$(call say,CC)
	$(Q)$(CC) $(BENCH_CFLAGS) $(BENCH_INC) -o $@ benchmark/detect.c \
	    benchmark/corpus_dataset.c $(LIBRARY_INTERNAL) -lm

build/benchmark_cpu: benchmark/benchmark_cpu.c benchmark/corpus_dataset.c $(LIBRARY) $(BENCH_HDRS) | build/BUILD_KNOBS
	@mkdir -p build
	$(call say,CC)
	$(Q)$(CC) $(BENCH_CFLAGS) $(BENCH_INC) -o $@ benchmark/benchmark_cpu.c \
	    benchmark/corpus_dataset.c $(LIBRARY) -lm

# STAGES=fold: benchmark_cpu.c with -DBENCH_FOLD, so the twin backends are scored and
# timed by the accuracy stage's own code; links LIBRARY_INTERNAL, as detect does.
build/fold_cpu: benchmark/benchmark_cpu.c benchmark/corpus_dataset.c $(LIBRARY) $(LIBRARY_INTERNAL) $(OBJECTS_FOLD) $(BENCH_HDRS) | build/BUILD_KNOBS
	@mkdir -p build
	$(call say,CC)
	$(Q)$(CC) $(BENCH_CFLAGS) -DBENCH_FOLD $(BENCH_INC) -o $@ benchmark/benchmark_cpu.c \
	    benchmark/corpus_dataset.c $(OBJECTS_FOLD) $(LIBRARY_INTERNAL) -lm
	@echo "built $@"

# The supplier drivers print a summary line; V=yes prints their compile commands.
SUP_WITH = Bessels.jl $$(sed -e 's/-DCPU_SUPPLIER_WITH_CXX/std/' -e 's/-DCPU_SUPPLIER_WITH_BOOST/Boost/' -e 's/-DCPU_SUPPLIER_WITH_//g' references/build/defines.txt)

build/threads_cpu: benchmark/threads_cpu.c benchmark/corpus_dataset.c $(LIBRARY) $(BENCH_HDRS) supplier-objects
	@mkdir -p build
	$(Q)$(CC) $(BENCH_CFLAGS) $$(cat references/build/defines.txt) $(BENCH_INC) -Ireferences/common \
	    -o $@ benchmark/threads_cpu.c \
	    benchmark/corpus_dataset.c $(LIBRARY) $$(cat references/build/link.txt) -lpthread -lm
	@echo built build/threads_cpu "(suppliers:" $(SUP_WITH)")"

build/suppliers: benchmark/suppliers.c benchmark/corpus_dataset.c $(LIBRARY) $(BENCH_HDRS) supplier-objects | build/BUILD_KNOBS
	@mkdir -p build
	$(Q)$(CC) $(BENCH_CFLAGS) $$(cat references/build/defines.txt) $(BENCH_INC) -Ireferences/common \
	    -o $@ benchmark/suppliers.c benchmark/corpus_dataset.c \
	    $(LIBRARY) $$(cat references/build/link.txt) -lm
	@echo built build/suppliers "(suppliers:" $(SUP_WITH)")"

benchmark-suppliers: build/suppliers
	@./build/suppliers --list

benchmark-build: build/detect build/benchmark_cpu
	@echo "built build/detect and build/benchmark_cpu"

# GPU drivers target the architecture nvidia-smi reports, else GPU_ARCH; nvidia-smi
# from PATH, else WSL's location (absent from sudo's PATH).
NVIDIA_SMI = $(or $(shell command -v nvidia-smi 2>/dev/null),$(wildcard /usr/lib/wsl/lib/nvidia-smi),nvidia-smi)
GPU_ARCH_DETECTED = $(shell $(NVIDIA_SMI) --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
GPU_ARCH_USE = $(if $(GPU_ARCH_DETECTED),sm_$(GPU_ARCH_DETECTED),$(GPU_ARCH))

# Device suppliers: KAUST (when ported) and Boost >= 1.87 (BOOST_MATH_GPU_ENABLED).
ifneq ($(KAUST_PORT),)
KAUST_GPU_DEF := -DGPU_SUPPLIER_WITH_KAUST
endif
# The device Boost (boost_detect.sh --select gpu): BOOST_ROOT when GPU-capable (none
# disables the row), else the highest Boost with BOOST_MATH_GPU_ENABLED; resolved once.
# BOOST_GPU_WHERE (--where gpu), the directory and its origin, goes into the build records.
BOOST_GPU_DIR = $(eval BOOST_GPU_DIR := $(shell CXX="$(CXX)" BOOST_ROOT="$(BOOST_ROOT)" bash benchmark/tools/boost_detect.sh --select gpu))$(BOOST_GPU_DIR)
BOOST_GPU_WHERE = $(eval BOOST_GPU_WHERE := $(shell CXX="$(CXX)" BOOST_ROOT="$(BOOST_ROOT)" bash benchmark/tools/boost_detect.sh --where gpu))$(BOOST_GPU_WHERE)
BOOST_GPU_DEF = $(if $(BOOST_GPU_DIR),-DGPU_SUPPLIER_WITH_BOOST -I$(BOOST_GPU_DIR))
BOOST_GPU_SOURCE_DEF = $(if $(BOOST_GPU_DIR),-DGPU_SUPPLIER_BOOST_SOURCE='"$(BOOST_GPU_WHERE)"')

# Configuration stamp of the device Boost, rewritten only on change: another Boost, another
# origin of the same directory, or none rebuilds the drivers compiled against it.  make
# rebuilds only on a strictly newer prerequisite and some file systems store whole seconds,
# so stamp_newer makes a rewritten stamp newer than a listed file written in the same second.
stamp_newer = for f in $(1); do [ ! -e $$f ] || [ $@ -nt $$f ] || { sleep 1; touch $@; break; }; done
build/boost_gpu.config: build-knobs-force
	@mkdir -p build
	@echo "$(BOOST_GPU_WHERE)" | cmp -s - $@ || { echo "$(BOOST_GPU_WHERE)" > $@; \
	    $(call stamp_newer,build/suppliers_gpu build/boost_shortcut_host.o build/boost_shortcut_gpu); }

# The KAUST port compiled as CUDA, so CPU and GPU rows run the same source.
build/kaust_dev.o: references/kaust_logbesselk/kaust_port.c $(KAUST_DERIVED)
	@mkdir -p build
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) $(NVCCFLAGS) -rdc=true -dc -x cu \
	    -Iinclude -Ibenchmark/common -Ireferences/common \
	    -Ireferences/build/kaust \
	    $< -o $@

build/suppliers_gpu: benchmark/suppliers_gpu.cu benchmark/corpus_dataset.c build/cuda_bench.o $(LIBRARY) $(BENCH_HDRS) $(if $(KAUST_GPU_DEF),build/kaust_dev.o) \
                     build/boost_gpu.config | build/BUILD_KNOBS
	@mkdir -p build
	@echo "  gpu suppliers: kaust=$(if $(KAUST_GPU_DEF),yes,no) boost=$(if $(BOOST_GPU_DEF),yes,no)"
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) -O2 -arch=$(GPU_ARCH_USE) --expt-relaxed-constexpr \
	    -Xcompiler -fopenmp,-Wall,-Wextra,-Wformat=2 \
	    $(KAUST_GPU_DEF) $(BOOST_GPU_DEF) $(BOOST_GPU_SOURCE_DEF) \
	    -Ibenchmark -Ibenchmark/common $(INCLUDES) \
	    -Ireferences/common -rdc=true \
	    -o $@ benchmark/suppliers_gpu.cu benchmark/corpus_dataset.c \
	    $(if $(KAUST_GPU_DEF),build/kaust_dev.o) build/cuda_bench.o $(LIBRARY) -lm

# Boost's K_nu at nu = 2.5 + delta, host and device, against data/boost_shortcut/.
# The host half is a g++ object with the supplier flags of references/Makefile:
# under nvcc, __CUDACC__ disables Boost's long-double promotion on the host too.
# Without a device Boost both rules stop and remove the driver built against another one.
build/boost_shortcut_host.o: benchmark/boost_shortcut_host.cpp references/common/build_info.h build/boost_gpu.config
	@mkdir -p build
	@test -n "$(BOOST_GPU_DIR)" || { rm -f $@ build/boost_shortcut_gpu; echo "boost_shortcut_host: needs Boost >= 1.87 (BOOST_ROOT, or make boost-headers)"; exit 1; }
	$(call say,CXX)
	$(Q)$(CXX) -std=c++17 -march=native -O2 -Wall -I$(BOOST_GPU_DIR) $(BOOST_GPU_SOURCE_DEF) -c $< -o $@

build/boost_shortcut_gpu: benchmark/boost_shortcut_gpu.cu benchmark/common/boost_adapted.h benchmark/corpus_bin.h \
                          benchmark/common/build_info_cuda.h build/boost_shortcut_host.o build/boost_gpu.config
	@mkdir -p build
	@test -n "$(BOOST_GPU_DEF)" || { rm -f $@; echo "boost_shortcut_gpu: needs Boost >= 1.87 (BOOST_ROOT, or make boost-headers)"; exit 1; }
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) -O2 -arch=$(GPU_ARCH_USE) --expt-relaxed-constexpr -I$(BOOST_GPU_DIR) $(BOOST_GPU_SOURCE_DEF) \
	    -Ibenchmark -Ibenchmark/common -o $@ benchmark/boost_shortcut_gpu.cu build/boost_shortcut_host.o -lm

build/threads_gpu: benchmark/threads_gpu.cu benchmark/corpus_dataset.c build/cuda_bench.o $(LIBRARY) $(BENCH_HDRS)
	@mkdir -p build
	@echo "  targeting $(GPU_ARCH_USE)"
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) -O2 -arch=$(GPU_ARCH_USE) --expt-relaxed-constexpr \
	    -Xcompiler -fopenmp,-Wall,-Wextra,-Wformat=2 \
	    -Ibenchmark -Ibenchmark/common $(INCLUDES) \
	    -o $@ benchmark/threads_gpu.cu benchmark/corpus_dataset.c \
	    build/cuda_bench.o $(LIBRARY) -lm

# -Wformat=2 on the host pass: nvcc does not format-check printf by default.
build/benchmark_gpu: benchmark/benchmark_gpu.cu benchmark/corpus_dataset.c build/cuda_bench.o $(LIBRARY) $(BENCH_HDRS) | build/BUILD_KNOBS
	@mkdir -p build
	@echo "  targeting $(GPU_ARCH_USE)"
	@$(NVCC_CHECK)
	$(call say,NVCC)
	$(Q)$(NVCC) -O2 -arch=$(GPU_ARCH_USE) --expt-relaxed-constexpr \
	    -Xcompiler -fopenmp,-Wall,-Wextra,-Wformat=2 \
	    -Ibenchmark -Ibenchmark/common $(INCLUDES) \
	    -o $@ benchmark/benchmark_gpu.cu benchmark/corpus_dataset.c \
	    build/cuda_bench.o $(LIBRARY) -lm

benchmark-gpu: build/benchmark_gpu
	@./build/benchmark_gpu --law grid --max-points 20000

# Levin band A/B: novec removes Levin from the vector backends, scal adds it to scalar,
# the device arm adds it.  Arms share object paths, so they build serially with a
# clean in between; the target fails if two arms report the same routing.
AB_DIR = build-ab
# Each arm logs its build commands to $(AB_DIR)/build-<arm>.log; V=yes prints them instead.
AB_LOG   = $(if $(Q),V=yes >$(AB_DIR)/build-$(1).log 2>&1 || { echo "  arm $(1) FAILED; end of $(AB_DIR)/build-$(1).log:"; \
           tail -25 $(AB_DIR)/build-$(1).log; exit 1; })
AB_CLEAN = $(MAKE) --no-print-directory clean $(if $(Q),>/dev/null)
LEVIN_GPU_FLAGS = -DMATERN_CUDA_FLAGS=MATERN_ROUTE_FLAGS_DEFAULT_LEVIN \
                  -DMATERN_CUDA_FLAGS_Z=MATERN_ROUTE_FLAGS_DEFAULT_LEVIN_Z

levin-ab-build:
	+@mkdir -p $(AB_DIR)
	@echo "levin-ab-build: arm base  (shipped routing)"
	@$(AB_CLEAN)
	@$(MAKE) --no-print-directory build/benchmark_cpu $(call AB_LOG,base)
	@cp build/benchmark_cpu $(AB_DIR)/benchmark_cpu_base
	@echo "levin-ab-build: arm novec (-DMATERN_WITHOUT_LEVIN)"
	@$(AB_CLEAN)
	@$(MAKE) --no-print-directory build/benchmark_cpu XCF=-DMATERN_WITHOUT_LEVIN $(call AB_LOG,novec)
	@cp build/benchmark_cpu $(AB_DIR)/benchmark_cpu_novec
	@echo "levin-ab-build: arm scal  (-DMATERN_SCALAR_WITH_LEVIN)"
	@$(AB_CLEAN)
	@$(MAKE) --no-print-directory build/benchmark_cpu XCF=-DMATERN_SCALAR_WITH_LEVIN $(call AB_LOG,scal)
	@cp build/benchmark_cpu $(AB_DIR)/benchmark_cpu_scal
	@echo "levin-ab-build: restoring the shipped build"
	@$(AB_CLEAN)
	@$(MAKE) --no-print-directory benchmark-build $(call AB_LOG,shipped)
	@a=`$(AB_DIR)/benchmark_cpu_base  --print-build | tr '\n' ' '`; \
	 b=`$(AB_DIR)/benchmark_cpu_novec --print-build | tr '\n' ' '`; \
	 c=`$(AB_DIR)/benchmark_cpu_scal  --print-build | tr '\n' ' '`; \
	 if [ "$$a" = "$$b" ] || [ "$$a" = "$$c" ] || [ "$$b" = "$$c" ]; then \
	   echo "levin-ab-build: two arms report the same routing; a rebuild did not take."; \
	   echo "  base  $$a"; echo "  novec $$b"; echo "  scal  $$c"; \
	   exit 1; fi; \
	 echo "levin-ab-build: base [$$a]"; \
	 echo "                novec [$$b]"; \
	 echo "                scal  [$$c]"

# Only the device objects carry the routing: both are removed, and build/cuda_bench.o and
# the binary are rebuilt.
levin-ab-build-gpu:
	+@mkdir -p $(AB_DIR)
	@echo "levin-ab-build-gpu: arm nolevin (shipped routing)"
	@rm -f build/cuda.o build/cuda_bench.o build/benchmark_gpu
	@$(MAKE) --no-print-directory build/benchmark_gpu $(call AB_LOG,gpu-nolevin)
	@cp build/benchmark_gpu $(AB_DIR)/benchmark_gpu_nolevin
	@echo "levin-ab-build-gpu: arm levin"
	@rm -f build/cuda.o build/cuda_bench.o build/benchmark_gpu
	@$(MAKE) --no-print-directory build/benchmark_gpu XNVCF='$(LEVIN_GPU_FLAGS)' $(call AB_LOG,gpu-levin)
	@cp build/benchmark_gpu $(AB_DIR)/benchmark_gpu_levin
	@echo "levin-ab-build-gpu: restoring the shipped build"
	@rm -f build/cuda.o build/cuda_bench.o build/benchmark_gpu
	@$(MAKE) --no-print-directory build/benchmark_gpu $(call AB_LOG,gpu-shipped)
	@a=`$(AB_DIR)/benchmark_gpu_levin --print-build | head -1`; \
	 b=`$(AB_DIR)/benchmark_gpu_nolevin --print-build | head -1`; \
	 if [ "$$a" = "$$b" ]; then \
	   echo "levin-ab-build-gpu: both arms report $$a; the device rebuild did not take."; \
	   exit 1; fi; \
	 echo "levin-ab-build-gpu: $(AB_DIR)/benchmark_gpu_levin [$$a], $(AB_DIR)/benchmark_gpu_nolevin [$$b]"

# make benchmark notifies once (benchmark/tools/notify.sh), after the tables or on a
# failed run; a failed figure or table step fails the target, the run stays recorded.
RUNS_DIR = $(or $(BENCH_RUNS_DIR),benchmark/runs)
# Non-empty under make -n, where a $(MAKE) line still runs; it is tested before notifying.
DRY_RUN = $(findstring n,$(firstword -$(MAKEFLAGS)))
# Passes every run parameter through except STAGES and the reduced-run caps.
pipeline:
	@NVCC="$(NVCC)" NVIDIA_SMI="$(NVIDIA_SMI)" MAKE="$(MAKE)" bash benchmark/pipeline.sh

# The published results on this machine: make pipeline with the full settings, the externals
# and the corpus download; GPU is auto unless given (GPU=no: the CPU part only).  KAUST is
# asked on a terminal (after the copyright notice); without one KAUST=yes or KAUST=no is
# required while the port is absent.  The recipe is recursive (+, so that make -j reaches the
# nested builds) and runs nothing under make -n, -q or -t.
REPRODUCE_NOEXEC = $(strip $(foreach f,n q t,$(findstring $(f),$(firstword -$(MAKEFLAGS)))))
# SETTINGS, EXTERNALS and CORPUS given on the command line with another value than the one
# make reproduce sets.
REPRODUCE_IGNORED = $(strip \
    $(if $(filter command line,$(origin SETTINGS)),$(if $(filter full FULL Full,$(SETTINGS)),,SETTINGS=$(SETTINGS))) \
    $(if $(filter command line,$(origin EXTERNALS)),$(if $(filter $(YES_WORDS),$(EXTERNALS)),,EXTERNALS=$(EXTERNALS))) \
    $(if $(filter command line,$(origin CORPUS)),$(if $(filter $(YES_WORDS),$(CORPUS)),,CORPUS=$(CORPUS))))
reproduce:
	+@if [ -n "$(REPRODUCE_NOEXEC)" ]; then \
	     echo "make reproduce: not run under make -n, -q or -t (make reproduce DRY=yes prints the plan)"; \
	     exit 0; fi; \
	 ign='$(REPRODUCE_IGNORED)'; [ -z "$$ign" ] || echo "make reproduce: $$ign not used: it sets" \
	     "SETTINGS=full, EXTERNALS=yes and CORPUS=yes (make pipeline takes them)" >&2; \
	 NVCC="$(NVCC)" NVIDIA_SMI="$(NVIDIA_SMI)" MAKE="$(MAKE)" SETTINGS=full EXTERNALS=yes CORPUS=yes \
	     bash benchmark/pipeline.sh reproduce

benchmark: benchmark-build
	@BENCH_NOTIFY_SKIP=1 BENCH_FULL_RUN_ENTRY="make benchmark" bash benchmark/run_benchmark.sh; rc=$$?; \
	 if [ $$rc -ne 0 ]; then \
	     bash benchmark/tools/notify.sh "make benchmark" \
	         "run $$(sed -n 's/^STATUS *//p' $(RUNS_DIR)/latest/STATUS 2>/dev/null | head -1) -- figures and tables not made; see $(RUNS_DIR)/latest/STATUS"; \
	     exit $$rc; \
	 fi
	@made=; notmade=; \
	 if $(MAKE) --no-print-directory figures ARGS="--runs $(RUNS_DIR) $(ARGS)"; \
	 then made=figures; else notmade=figures; fi; \
	 if $(MAKE) --no-print-directory tables ARGS="--runs $(RUNS_DIR) $(ARGS)"; \
	 then made="$${made:+$$made and }tables"; else notmade="$${notmade:+$$notmade and }tables"; fi; \
	 [ -z "$$notmade" ] || \
	     echo "make benchmark: $$notmade not made (above); the run itself is recorded under $(RUNS_DIR)/"; \
	 msg="run $$(sed -n 's/^STATUS *//p' $(RUNS_DIR)/latest/STATUS 2>/dev/null | head -1)"; \
	 msg="$$msg$${made:+; $$made written}$${notmade:+; $$notmade not made (see above)}"; \
	 [ -n "$(DRY_RUN)" ] || bash benchmark/tools/notify.sh "make benchmark" "$$msg"; \
	 [ -z "$$notmade" ]

# The interpreter, its version, whether it is this tree's .venv/, and its NumPy/Matplotlib.
python-info:
	@echo "PYTHON = $(PYTHON)   ($(if $(filter file,$(origin PYTHON)),searched: the first of PYTHON_CANDIDATES with NumPy and Matplotlib at or above the tested minimum; python3 if none has them,set on the $(origin PYTHON)))"
	@if command -v $(firstword $(PYTHON)) >/dev/null 2>&1; then \
	     $(firstword $(PYTHON)) -c 'import os, sys; v = os.path.realpath(sys.prefix) == os.path.realpath(".venv"); print("  interpreter  %s, Python %s, %s" % (sys.executable, sys.version.split()[0], "the .venv/ of this tree" if v else "not the .venv/ of this tree"))'; \
	     _pk=$$($(firstword $(PYTHON)) tools/py_stack.py 2>/dev/null); \
	     case $$? in \
	         0) echo "  packages     $$_pk" ;; \
	         1) echo "  packages     $$_pk -- make externals installs the published versions into .venv/$(if $(filter file,$(origin PYTHON)),; the Makefile then selects it,; used once PYTHON is unset)" ;; \
	         *) echo "  packages     NumPy/Matplotlib not importable -- make externals (installs NumPy and Matplotlib into .venv/$(if $(filter file,$(origin PYTHON)),; the Makefile then selects it,; used once PYTHON is unset)), or make requirements for the system packages" ;; \
	     esac; \
	 else \
	     echo "  interpreter  $(firstword $(PYTHON)) not found -- $(if $(filter file,$(origin PYTHON)),install Python 3 (make requirements names the package) then make externals (installs NumPy and Matplotlib into .venv/),unset PYTHON or set it to an installed interpreter)"; \
	 fi

# The check and the generator share one shell line, so its `exit 2` ends the target first.
PY_FOUND_CHECK = command -v $(firstword $(PYTHON)) >/dev/null 2>&1 || { \
	    echo "$@: $(firstword $(PYTHON)) not found; not made -- $(if $(filter file,$(origin PYTHON)),install Python 3 (make requirements names the package),unset PYTHON (set on the $(origin PYTHON)) or set it to an installed interpreter)"; exit 2; }
# PY_GEN_CHECK: NumPy and Matplotlib at or above the tested minimum (tools/py_stack.py).
PY_GEN_CHECK = $(PY_FOUND_CHECK); _pk=$$($(PYTHON) tools/py_stack.py 2>/dev/null); case $$? in \
	  0) ;; \
	  1) echo "$@: $(firstword $(PYTHON)) has $$_pk; not made -- make externals (installs the published versions into .venv/$(if $(filter file,$(origin PYTHON)),; the Makefile then selects it,; used once PYTHON is unset))"; exit 2 ;; \
	  *) echo "$@: NumPy/Matplotlib not importable by $(firstword $(PYTHON)); not made -- make externals (installs NumPy and Matplotlib into .venv/$(if $(filter file,$(origin PYTHON)),; the Makefile then selects it,; used once PYTHON is unset)), or make requirements for the system packages"; exit 2 ;; \
	esac

# The paper's figures and tables, drawn from the recorded runs only: the figures into
# benchmark/plots/paper-final/ (PDF, captions.tex, HARDWARE.md), the tables into
# benchmark/tables/paper/, where pdflatex fits them to the column unless ARGS=--no-preview.
figures:
	@$(PY_GEN_CHECK); $(PYTHON) benchmark/plots/make_paper_figures.py $(ARGS)

tables:
	@$(PY_GEN_CHECK); $(PYTHON) benchmark/tables/make_paper_tables.py $(ARGS)

# Compiles every generated table in one document (needs pdflatex).
tables-check:
	@$(PY_FOUND_CHECK); $(PYTHON) benchmark/tables/check_tex.py $(ARGS)

# The durations of the published run: benchmark/durations.tsv from the TIMES.txt of the runs
# under RUNS_DIR, then the three places that state them rendered from it
# (benchmark/tools/durations.py; make audit checks that they agree).
durations:
	@$(PY_FOUND_CHECK); $(PYTHON) benchmark/tools/durations.py --runs $(RUNS_DIR) $(ARGS)

benchmark-test: benchmark-build
	@bash benchmark/tests/run_tests.sh
