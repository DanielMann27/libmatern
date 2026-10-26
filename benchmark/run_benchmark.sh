#!/bin/bash
# run_benchmark.sh -- runs the CPU benchmark stages (STAGES) and writes one run record (CSV files,
# PROVENANCE.txt, MANIFEST.txt, STAGES.txt, STATUS) under benchmark/runs/ (see the accompanying
# paper, experimental setup).  Exits non-zero if any stage failed.
set -u
cd "$(dirname "$0")/.." || exit 1

# The run parameters; the caller's values are captured before any default, for the run label.
RUN_PARAMS="FAMILIES BACKENDS REPS DATA LADDER_SET LADDER_LAWS LADDER_MAX LADDER_PASS_FROM SUPPLIERS SUP_MAX_POINTS TIME_MAX_POINTS
            SUP_LADDER SUP_LADDER_BUDGET THREADS_LAW THREADS_N THREADS_REPS THREADS_ASSEMBLY
            THREADS_PIN_MODES THREADS_SUPPLIERS THREADS_WS_N THREADS_WS_THREADS THREADS_WS_BACKENDS
            THREADS_MAX"
declare -A GIVEN=()
for _p in $RUN_PARAMS; do [ -n "${!_p:-}" ] && GIVEN[$_p]="${!_p}"; done

# TIME_MAX_POINTS times only the first N points of each law (accuracy still scores every
# point); for reduced runs only, make pipeline and benchmark/run_gpu.sh refuse it.
case "${TIME_MAX_POINTS:-}" in
    "") time_cap="" ;;
    *[!0-9]*|0) echo "run_benchmark.sh: TIME_MAX_POINTS must be a positive integer (got '$TIME_MAX_POINTS')" >&2; exit 2 ;;
    *) time_cap="--time-points $TIME_MAX_POINTS" ;;
esac

# matched_fma: the matched source built with -mfma, selected on FMA3 machines without AVX2.
# BACKENDS unset: every backend this CPU runs (filtered below); set: used as given, and a
# backend the CPU cannot run fails the accuracy stage.
BACKENDS_GIVEN="${BACKENDS:+yes}"
BACKENDS="${BACKENDS:-avx512,avx2,matched_fma,matched,scalar}"
FAMILIES="${FAMILIES:-z_nu,h_nu_s}"
# The corpus families: z_nu ((z, nu) supplied) and h_nu_s ((h, nu, s), z computed).
for _f in $(printf '%s' "$FAMILIES" | tr ',' ' '); do
    case "$_f" in z_nu|h_nu_s) ;; *)
        echo "run_benchmark.sh: FAMILIES: unknown family '$_f'; the families are z_nu and h_nu_s" >&2
        exit 2 ;;
    esac
done
# A fixed repetition count, not a floor, for every timing stage (threads through THREADS_REPS,
# the GPU stages through benchmark/run_gpu.sh), so the estimators are comparable.
REPS="${REPS:-65}"
DATA="${DATA:-data}"
# LADDER_SET: small (size_random/size_half, the first 2^20 points of the 2^25 sets) or large
# (size_random_large/size_half_large, all 2^25; the default of benchmark/run_gpu.sh).  The top
# rung comes from the corpus; LADDER_MAX=<N> lowers it for every ladder of the run.
LADDER_SET="${LADDER_SET:-small}"
case "$LADDER_SET" in
    large) _LADDER_DEFAULT="size_random_large,size_half_large" ;;
    small) _LADDER_DEFAULT="size_random,size_half" ;;
    *) echo "LADDER_SET must be large|small (got '$LADDER_SET')" >&2; exit 2 ;;
esac
LADDER_LAWS="${LADDER_LAWS:-$_LADDER_DEFAULT}"
LADDER_MAX="${LADDER_MAX:-}"
case "$LADDER_MAX" in
    "") _LADDER_MAX_ARG=() ;;
    *[!0-9]*|0) echo "LADDER_MAX must be a positive number of points (got '$LADDER_MAX')" >&2; exit 2 ;;
    *) _LADDER_MAX_ARG=(--ladder-max "$LADDER_MAX") ;;
esac
# LADDER_PASS_FROM=<S> (a power of two >= 256) times the CPU ladder rungs N >= S in one
# cumulative pass per repetition (--ladder-pass-from); off times every rung on its own.
LADDER_PASS_FROM="${LADDER_PASS_FROM:-4194304}"
case "$LADDER_PASS_FROM" in
    off) _LADDER_PASS_ARG=() ;;
    *[!0-9]*|"") echo "LADDER_PASS_FROM must be off or a power of two >= 256 (got '$LADDER_PASS_FROM')" >&2; exit 2 ;;
    *) if [ "$LADDER_PASS_FROM" -lt 256 ] || [ $(( LADDER_PASS_FROM & (LADDER_PASS_FROM - 1) )) -ne 0 ]; then
           echo "LADDER_PASS_FROM must be off or a power of two >= 256 (got '$LADDER_PASS_FROM')" >&2; exit 2
       fi
       _LADDER_PASS_ARG=(--ladder-pass-from "$LADDER_PASS_FROM") ;;
esac
# Step 0: the corpus in $DATA; missing files are asked for on a terminal, else the run stops
# before anything is recorded.  A run that reads no 2^25-point file needs only the 61 small
# files (`tools/corpus.sh need ... tracked`); its accuracy or fold stage then reports the size laws missing (STATUS partial).
case "${STAGES:-all}" in
    rest|record|fold) _corpus_scope=tracked ;;
    all|ladder) case ",$LADDER_LAWS," in *,size_*) _corpus_scope="" ;; *) _corpus_scope=tracked ;; esac ;;
    *)          _corpus_scope="" ;;
esac
DATA="$DATA" sh tools/corpus.sh need "${BENCH_FULL_RUN_ENTRY:-benchmark/run_benchmark.sh}" $_corpus_scope || exit 2
# A ladder stage refuses a missing 2^25 set here, not at its first rung (the 2^20 laws are its
# prefixes); files are looked for as the readers look: data/z_nu/, then data/.
_has_law(){ compgen -G "$DATA/z_nu/z_nu-nu_$1_[0-9]*.bin" >/dev/null ||
            compgen -G "$DATA/z_nu-nu_$1_[0-9]*.bin" >/dev/null; }
case "${STAGES:-all}" in all|ladder) _ladder_laws=$LADDER_LAWS ;; *) _ladder_laws="" ;; esac
for _l in $(printf '%s' "$_ladder_laws" | tr ',' ' '); do
    case "$_l" in size_*)
        _base="${_l%_large}"
        _has_law "${_base}_large" || _has_law "$_l" || {
            echo "LADDER_SET=$LADDER_SET needs the 2^25 set ${_base}_large, and $DATA has none." >&2
            echo "  the size datasets are read from <family>-*_size_*_large_33554432.bin; see data/README.md;" >&2
            echo "  make corpus DOWNLOAD=yes downloads the corpus into data/" >&2
            exit 2; }
    esac
done
# The thread stage: size_random (unstructured, so no thread's slice is a special case).
# THREADS_N is per thread and must not exceed the law's size: every thread evaluates the
# same first N points.
THREADS_LAW="${THREADS_LAW:-size_random}"
THREADS_N="${THREADS_N:-262144}"
# The repetition count of the other stages, whose rows the thread rows are compared with.
THREADS_REPS="${THREADS_REPS:-$REPS}"
# THREADS_MAX caps the thread ladder (powers of two, THREADS_MAX itself included); unset, it
# runs to the physical core count.
THREADS_MAX="${THREADS_MAX:-}"
THREADS_CAP="${THREADS_MAX:+--max-threads $THREADS_MAX}"
# STAGES: all (= rest + ladder), ladder (the N sweeps), rest (all but the ladders); the opt-in
# levin (needs the build-ab arms), fold (needs build/fold_cpu) and threads (loads every core);
# record (the run record alone, no CPU timing; benchmark/run_gpu.sh GPU_STAGES=call-large).
STAGES="${STAGES:-all}"
case "$STAGES" in all|ladder|rest|levin|threads|record|fold) ;; *)
    echo "run_benchmark: STAGES must be all, ladder, rest, levin, threads, record or fold (got '$STAGES')" >&2
    exit 2 ;;
esac
AB="${AB_DIR:-build-ab}"
BUILD="${BUILD:-build}"
# The A/B stages' prerequisites are checked before a run directory exists, so that a refused
# run leaves no record without a STATUS.
if [ "$STAGES" = levin ]; then
    for arm in base novec scal; do
        [ -x "$AB/benchmark_cpu_$arm" ] || {
            echo "levin: $AB/benchmark_cpu_$arm is missing -- run 'make levin-ab-build' first" >&2
            exit 2; }
    done
    _a=$("$AB"/benchmark_cpu_base  --print-build 2>/dev/null | tr '\n' ' ')
    _b=$("$AB"/benchmark_cpu_novec --print-build 2>/dev/null | tr '\n' ' ')
    _c=$("$AB"/benchmark_cpu_scal  --print-build 2>/dev/null | tr '\n' ' ')
    if [ "$_a" = "$_b" ] || [ "$_a" = "$_c" ] || [ "$_b" = "$_c" ]; then
        echo "levin: two arms report the same routing -- these are the same build," >&2
        echo "       refusing to measure. base[$_a] novec[$_b] scal[$_c]" >&2
        exit 2
    fi
fi
if [ "$STAGES" = threads ] && ! [ -x "$BUILD/threads_cpu" ]; then
    echo "threads: $BUILD/threads_cpu is missing -- run 'make build/threads_cpu' first" >&2
    exit 2
fi
if [ "$STAGES" = fold ]; then
    [ -x "$BUILD/fold_cpu" ] || {
        echo "fold: $BUILD/fold_cpu is missing -- run 'make build/fold_cpu' first" >&2
        exit 2; }
    # fold_cpu is benchmark_cpu.c built with -DBENCH_FOLD; without it the library's backends run.
    "$BUILD"/fold_cpu --print-build 2>/dev/null | grep -qx 'fold=on' || {
        echo "fold: $BUILD/fold_cpu does not report fold=on -- it is not the fold build," >&2
        echo "      refusing to measure; rebuild it with 'make build/fold_cpu'" >&2
        exit 2; }
fi

# The default list keeps the backends the library runs on this CPU (build/detect --live-backend).
if [ -z "$BACKENDS_GIVEN" ] && [ -x "$BUILD/detect" ]; then
    _keep=""; _drop=""
    for _b in $(printf '%s' "$BACKENDS" | tr ',' ' '); do
        if [ "$(./"$BUILD"/detect --live-backend "$_b" 2>/dev/null)" = "$_b" ]; then
            _keep="${_keep:+$_keep,}$_b"
        else
            _drop="${_drop:+$_drop,}$_b"
        fi
    done
    [ -n "$_drop" ] && echo "run_benchmark: this CPU cannot run backend(s) $_drop; measuring $_keep (set BACKENDS to choose)" >&2
    BACKENDS="$_keep"
fi

# Run id <UTC time>_<host>_<label>[-<STAGES>][.<n>], e.g. 2026-09-14T12-42Z_node01_custom-rest.
# label: default when every RUN_PARAMS value is its default (else custom, and run_label in
# PROVENANCE.txt names those that differ), or SCOPE=<name>.  STAGES other than all is appended,
# except record with SCOPE set; .<n> resolves collisions.
# default_of <parameter>: the default this script applies, read from its own
# first `${<parameter>:-...}`, so the label cannot disagree with the defaults.
# full_run.sh's rb_default() reads it the same way; comments count, so none may hold that form.
default_of() {
    local d
    d=$(grep -o -m1 "\${$1:-[^}]*}" benchmark/run_benchmark.sh | head -1)
    d=${d#\$\{$1:-}; d=${d%\}}
    # A default that does not parse to one expression counts as changed, not as an error.
    eval "printf '%s' \"$d\"" 2>/dev/null || printf '%s' "<unparsed default>"
}
LABEL_REASONS=""
for _p in $RUN_PARAMS; do
    [ -n "${GIVEN[$_p]+x}" ] || continue
    [ "${GIVEN[$_p]}" = "$(default_of "$_p")" ] && continue
    LABEL_REASONS="${LABEL_REASONS:+$LABEL_REASONS }$_p=${GIVEN[$_p]}"
done
SCOPE="${SCOPE:-}"
_scope_given="$SCOPE"
if [ -z "$SCOPE" ]; then
    if [ -z "$LABEL_REASONS" ]; then SCOPE=default; else SCOPE=custom; fi
fi
BASE="$(date -u +%Y-%m-%dT%H-%MZ)_$(hostname -s)_$SCOPE"
[ "$STAGES" = all ] || { [ "$STAGES" = record ] && [ -n "$_scope_given" ]; } \
    || BASE="${BASE}-${STAGES}"
# BENCH_RUNS_DIR overrides the record root, so that the test suite does not write into benchmark/runs/.
RUNS_ROOT="${BENCH_RUNS_DIR:-benchmark/runs}"
mkdir -p "$RUNS_ROOT" || exit 1
RUNID="$BASE"; n=2
while [ -e "$RUNS_ROOT/$RUNID" ]; do RUNID="$BASE.$n"; n=$((n+1)); done
OUT="$RUNS_ROOT/$RUNID"
mkdir -p "$OUT" || exit 1
# `latest` points at the most recent run.
ln -sfn "$RUNID" "$RUNS_ROOT/latest" 2>/dev/null

say(){ printf '%s\n' "$*"; }
# progress: a timestamped line before each timing process; the drivers add one per (family,
# law, backend) block on descriptor 3 (BENCH_PROGRESS_FD), before the block's untimed warm-up.
progress(){ printf '%(%H:%M:%S)T  %s\n' -1 "$*"; }
# ext_note <stage> <cpu|threads> <parameter> <program --list output>: the supplier rows the build
# lacks, with the commands that add them, and whether built KAUST rows run (not under ship).
ext_note(){
    bash benchmark/tools/missing.sh rows "$1" "$2" "$3" "${!3:-ship}" "$4"
}
exec 3>&1
export BENCH_PROGRESS_FD=3
stage_rc=0
declare -a MISSING=()
declare -a SKIPPED=()
# WARNINGS: what may make the results differ from those of the published inputs; listed in STATUS.
declare -a WARNINGS=()
warn(){ WARNINGS+=("$1"); say "WARNING: $1"; }

# TIMES.txt: the wall time of each recorded stage, as the seconds since the previous record,
# and run_total at the end; benchmark/tools/durations.py reads it (make durations).
_record_t=$SECONDS
record(){  # record <name> <rc>
    if [ "$2" -ne 0 ]; then MISSING+=("$1"); stage_rc=1; fi
    printf '%-22s rc=%d\n' "$1" "$2" >> "$OUT/STAGES.txt"
    printf '%-40s %d\n' "$1" "$((SECONDS - _record_t))" >> "$OUT/TIMES.txt"
    _record_t=$SECONDS
}
# record_laws <stage> <stderr file>: each `corpus: no data for <family>/<law>` line of a driver
# is a missing part of <stage>, so a run without every dataset is not `complete`.
record_laws(){
    local l
    for l in $(sed -n 's/^corpus: no data for \([^ ]*\) .*/\1/p' "$2" 2>/dev/null | sort -u); do
        MISSING+=("$1:$l"); stage_rc=1
        printf '%-22s NO DATA\n' "$1:$l" >> "$OUT/STAGES.txt"
    done
}
# record_rows <name> <rc> <rows before> <csv>: a supplier that exits 0 but adds no row fails.
record_rows(){
    local after; after=$(grep -c '^[0-9]' "$4" 2>/dev/null); after=${after:-0}
    if [ "$2" -eq 0 ] && [ "$after" -le "$3" ]; then record "$1:no-rows" 1; else record "$1" "$2"; fi
}

say "run id : $RUNID"
say "output : $OUT"
# The full-run notice (benchmark/tools/full_run.sh); silent when an outer command printed it.
case "$STAGES" in all) _fr_stages=rest,ladder ;; *) _fr_stages=$STAGES ;; esac
_fr_entry="${BENCH_FULL_RUN_ENTRY:-bash benchmark/run_benchmark.sh}"
[ "$STAGES" = all ] || _fr_entry="$_fr_entry, STAGES=$STAGES"
# One probe of the missing software before any timing; an outer command passes its own.
if [ -z "${BENCH_MISSING_CACHE:-}" ]; then
    BENCH_MISSING_CACHE=$(bash benchmark/tools/missing.sh probe); export BENCH_MISSING_CACHE
    trap 'rm -f "$BENCH_MISSING_CACHE"' EXIT
fi
bash benchmark/tools/full_run.sh --entry "$_fr_entry" --stages "$_fr_stages"
export BENCH_FULL_RUN_NOTICE=shown

# Provenance.  A probe that fails is recorded by pv() as `unavailable` with the reason.
pv(){ # pv <key> <value> [why it is missing]
  if [ -n "${2:-}" ]; then printf '%-18s %s\n' "$1" "$2"
  else printf '%-18s unavailable (%s)\n' "$1" "${3:-not exposed on this platform}"; fi
}
# pfirst: the first of several probes that answers (/proc or /sys on Linux, sysctl on macOS);
# empty and non-zero when none does.
pfirst(){ for _c in "$@"; do
    _v=$(eval "$_c" 2>/dev/null | head -1 | sed 's/^[ \t]*//;s/[ \t]*$//')
    [ -n "$_v" ] && { printf '%s' "$_v"; return 0; }
  done; return 1; }

LOAD_BEFORE="$(pfirst \
    'cut -d" " -f1-3 /proc/loadavg' \
    'sysctl -n vm.loadavg | tr -d "{}" | sed "s/^ *//"' \
    'uptime | sed "s/.*average[s]*: //" | tr -d ","')"
NCPU_ONLN="$(pfirst 'getconf _NPROCESSORS_ONLN' 'sysctl -n hw.logicalcpu' 'nproc')"
# Load divided by the number of online CPUs, which is comparable between machines.
_loadpc(){ awk -v l="$(printf '%s' "$1" | awk '{print $1}')" -v n="${NCPU_ONLN:-}" \
    'BEGIN{ if (l != "" && n != "" && n+0 > 0) printf "%.3f", l/n }' 2>/dev/null; }
LOAD_BEFORE_PC="$(_loadpc "$LOAD_BEFORE")"

# One nvidia-smi field, trimmed; empty on any failure.
_gpuq() {
  nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null \
    | head -1 | sed 's/^[ \t]*//;s/[ \t]*$//'
}

# Source identity: release from include/matern.h; the commit from git only when this directory
# is the checkout's top level (not a tree inside another repository), else BENCH_SOURCE_COMMIT,
# else RELEASE as git archive fills it in (a $Format or %(...) placeholder names nothing).
RELEASE_VERSION="$(sed -n 's/^#define MATERN_VERSION_STRING[[:space:]]*"\([^"]*\)".*/\1/p' include/matern.h 2>/dev/null | head -1)"
_git_top="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -n "$_git_top" ] && [ "$_git_top" -ef . ]; then OWN_CHECKOUT=yes; else OWN_CHECKOUT=""; fi
_release_line(){ # the value of one RELEASE line; empty when git archive did not fill it in
  _v="$(sed -n "s/^$1[[:space:]]*//p" RELEASE 2>/dev/null | head -1)"
  case "$_v" in *'$Format'*|*'%('*) _v="" ;; esac
  printf '%s' "$_v"
}
if [ -n "$OWN_CHECKOUT" ]; then
  SOURCE_COMMIT="$(git rev-parse HEAD 2>/dev/null) (this git checkout; git_dirty below)"
elif [ -n "${BENCH_SOURCE_COMMIT:-}" ]; then
  SOURCE_COMMIT="$BENCH_SOURCE_COMMIT (BENCH_SOURCE_COMMIT: the checkout this tree was copied from)"
elif [ -n "$(_release_line commit)" ]; then
  _described="$(_release_line describe)"
  SOURCE_COMMIT="$(_release_line commit)${_described:+ = $_described} (RELEASE, filled in by git archive)"
else
  SOURCE_COMMIT=""
fi

{
  echo "run_id            $RUNID"
  echo "generated         $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "rep_policy        FIXED $REPS reps per rung and per row (comparable estimators)"
  # --max-points is a stride over the scored points; every row carries n_file, so a capped row
  # is identifiable.
  echo "sup_max_points    ${SUP_MAX_POINTS:-none (every supplier scores the whole dataset)}"
  echo "time_max_points   ${TIME_MAX_POINTS:-none (every point of every dataset is timed)}"
  case "${BENCH_PIN:-cpu}" in
    node) echo "cpu_pinning       node -- BENCH_PIN=node: each timing process is bound to all CPUs of one NUMA node" ;;
    *)    echo "cpu_pinning       cpu -- each timing process is pinned to one CPU of the NUMA node it chose; per-row migrations, involuntary switches and threads in timing_raw.*.sched.csv" ;;
  esac
  echo "run_label         $SCOPE${LABEL_REASONS:+  (differs from the defaults: $LABEL_REASONS)}"
  # A benchmark/run_gpu.sh run: one-backend CPU rows, used by the figures only where the machine
  # has no other CPU run.
  [ -n "${BENCH_GPU_PIPELINE:-}" ] && echo "pipeline          gpu"
  pv release "$RELEASE_VERSION" "include/matern.h states no MATERN_VERSION_STRING"
  pv source_commit "$SOURCE_COMMIT" \
     "this tree is not the top level of a git checkout, BENCH_SOURCE_COMMIT is unset and RELEASE was not filled in by git archive"
  if [ -n "$OWN_CHECKOUT" ]; then
    echo "git_commit        $(git rev-parse --short HEAD 2>/dev/null)"
    echo "git_dirty         $(test -n "$(git --no-optional-locks status --porcelain --untracked-files=no 2>/dev/null)" && echo "yes (uncommitted changes to tracked files)" || echo no)"
  else
    echo "git_commit        none (this tree is not the top level of a git checkout)"
    echo "git_dirty         unknown (this tree is not the top level of a git checkout)"
  fi
  echo "host              $(hostname)"
  pv cpu "$(pfirst "sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo" \
                   'sysctl -n machdep.cpu.brand_string' \
                   "lscpu | sed -n 's/^Model name: *//p'" \
                   'uname -p')" "no /proc/cpuinfo, no machdep.cpu sysctl, no lscpu"
  pv cpus_online "${NCPU_ONLN:-}" "no _NPROCESSORS_ONLN, hw.logicalcpu or nproc"
  pv cpus_physical "$(pfirst 'sysctl -n hw.physicalcpu' \
                   "lscpu | awk -F: '/^Core\(s\) per socket/{c=\$2} /^Socket\(s\)/{s=\$2} END{if(c&&s) print c*s}'" \
                   "grep -c '^core id' /proc/cpuinfo")" "no lscpu and no hw.physicalcpu"
  pv sockets "$(pfirst "lscpu | sed -n 's/^Socket(s): *//p'" \
                   'sysctl -n hw.packages')" "no lscpu and no hw.packages"
  pv os "$(pfirst "sed -n 's/^PRETTY_NAME=\"\\(.*\\)\"/\\1/p' /etc/os-release" \
                   'sw_vers -productName' 'uname -s')" "no /etc/os-release and no sw_vers"
  pv kernel "$(pfirst 'uname -sr')" "uname(1) is absent"
  pv arch   "$(pfirst 'uname -m')"  "uname(1) is absent"
  pv affinity_mask "$(pfirst "taskset -pc $$ | sed 's/.*: //'")" \
     "no taskset(1); affinity is not queryable from the shell here"
  pv load_before    "$LOAD_BEFORE"    "no /proc/loadavg, no vm.loadavg sysctl and no uptime(1)"
  pv load_before_pc "$LOAD_BEFORE_PC" "needs both a load average and a cpu count"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "slurm_job         $SLURM_JOB_ID"
    # Dedicated when all online CPUs are allocated to the job; OverSubscribe is recorded beside it.
    _alloc="${SLURM_CPUS_ON_NODE:-}"
    _ovs="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'OverSubscribe=[A-Z_]*' | head -1)"
    if [ -n "$_alloc" ] && [ -n "${NCPU_ONLN:-}" ] && [ "$_alloc" -ge "$NCPU_ONLN" ]; then
      echo "exclusive         yes -- all $NCPU_ONLN cpus allocated (${_ovs:-OverSubscribe=?})"
    elif [ -n "$_alloc" ]; then
      echo "exclusive         no -- $_alloc of ${NCPU_ONLN:-?} cpus allocated (${_ovs:-OverSubscribe=?}); see load_before and load_after"
    else
      echo "exclusive         unknown -- SLURM_CPUS_ON_NODE unset (${_ovs:-OverSubscribe=?})"
    fi
  else
    echo "slurm_job         none"
    echo "exclusive         no -- not a dedicated allocation; timings can include load from other processes (see load_before and load_after)"
  fi
  # The compiler $CC names in this shell, cc when unset as for make (make predefines CC=cc);
  # what built the library is in the build_* keys below, read from the artefacts.
  pv cc_on_path "$(pfirst "${CC:-cc} --version" "${CC:-gcc} --version")" \
     "the compiler named by \$CC does not answer --version"
  # The C library provides the libm of the scalar backend; libms differ in the last ulp, so it
  # is part of an accuracy result.
  pv libc "$(pfirst 'ldd --version | head -1' \
                    'getconf GNU_LIBC_VERSION' \
                    "ls /usr/lib/libSystem.B.dylib >/dev/null && sw_vers -productVersion | sed 's/^/Apple libSystem, macOS /'")" \
     "not a glibc/musl system and not macOS"
  echo "numa_nodes        $(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)"
  for d in /sys/devices/system/node/node[0-9]*; do
    [ -d "$d" ] || continue
    echo "numa_$(basename "$d")     cpus=$(cat "$d/cpulist" 2>/dev/null) distance=$(cat "$d/distance" 2>/dev/null)"
  done
  # Cache sizes and sharing, to decide from the record whether a working set fits a level.
  for _i in /sys/devices/system/cpu/cpu0/cache/index[0-9]*; do
    [ -d "$_i" ] || continue
    echo "cache_L$(cat "$_i/level" 2>/dev/null)$(cat "$_i/type" 2>/dev/null | cut -c1)        size=$(cat "$_i/size" 2>/dev/null) shared_by=$(cat "$_i/shared_cpu_list" 2>/dev/null)"
  done
  # lscpu's "L3 cache": every L3 instance of the machine (the per-instance size is cache_L3U).
  echo "l3_total          $(lscpu 2>/dev/null | sed -n 's/^L3 cache: *//p' | head -1)"
  # The clock state: a powersave or schedutil governor caps every timing in the record without
  # showing in the load average.
  _cf=/sys/devices/system/cpu/cpu0/cpufreq
  pv cpufreq_driver   "$(pfirst "cat $_cf/scaling_driver")" \
     "no cpufreq sysfs (a VM, a container, or a non-Linux kernel)"
  pv cpufreq_governor "$(pfirst "cat $_cf/scaling_governor")" \
     "no cpufreq sysfs; on macOS the OS owns the clock and exposes no governor"
  pv cpufreq_range_kHz "$(pfirst "paste -d' ' $_cf/scaling_min_freq $_cf/scaling_max_freq | sed 's/ / .. /'" \
                                 "awk 'BEGIN{}' /dev/null; sysctl -n hw.cpufrequency_min hw.cpufrequency_max 2>/dev/null | paste -sd' ' - | sed 's/ / .. /'")" \
     "no cpufreq sysfs and no hw.cpufrequency sysctl"
  if [ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "turbo             intel_pstate no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  elif [ -r /sys/devices/system/cpu/cpufreq/boost ]; then
    echo "turbo             cpufreq boost=$(cat /sys/devices/system/cpu/cpufreq/boost)"
  else
    echo "turbo             not exposed"
  fi
  # The FP environment: a non-default rounding mode or flush-to-zero changes the ULP figures.
  if [ -x "$BUILD/detect" ]; then
      ./"$BUILD"/detect --fpenv 2>/dev/null
  else
      pv fp_environment "" "build/detect is not built yet; run 'make benchmark-build'"
  fi
  # Each of the following changes a timing without showing in the load average.
  pv cgroup_cpu_max "$(pfirst 'cat /sys/fs/cgroup/cpu.max' \
        'echo "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"')" \
     "no cgroup v2 cpu.max and no v1 cfs quota -- not container-limited"
  pv cgroup_cpuset "$(pfirst 'cat /sys/fs/cgroup/cpuset.cpus.effective' \
        'cat /sys/fs/cgroup/cpuset/cpuset.cpus')" "no cgroup cpuset"
  pv power_source "$(pfirst \
        "for _b in /sys/class/power_supply/A[CD]*/online; do [ -r \$_b ] && { [ \$(cat \$_b) = 1 ] && echo 'AC (mains)' || echo 'BATTERY -- expect aggressive downclocking'; break; }; done" \
        "pmset -g batt | head -1 | sed 's/Now drawing from //;s/'\\''//g'")" \
     "no power_supply class and no pmset -- a desktop, a server or a VM"
  # Thermal throttling: counters, so a change over the run indicates throttling during it.
  pv thermal_throttle_core "$(pfirst \
        "cat /sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count")" \
     "no thermal_throttle sysfs (AMD, ARM, a VM, or a non-Linux kernel)"
  pv thermal_pressure "$(pfirst \
        "awk '{print \$1}' /proc/pressure/cpu 2>/dev/null | head -1" \
        "sysctl -n machdep.xcpm.cpu_thermal_level")" \
     "no PSI (/proc/pressure) and no thermal-level sysctl"
  pv hypervisor "$(pfirst "grep -qm1 '^flags.*hypervisor' /proc/cpuinfo && systemd-detect-virt 2>/dev/null || systemd-detect-virt" \
        "grep -qm1 '^flags.*hypervisor' /proc/cpuinfo && echo 'present (flag set; no systemd-detect-virt to name it)'" \
        "sysctl -n kern.hv_vmm_present | sed 's/^1$/present/;s/^0$/none/'")" \
     "no hypervisor flag and no detector -- bare metal, or undetectable"
  pv cpu_mitigations "$(pfirst \
        "grep -r . /sys/devices/system/cpu/vulnerabilities 2>/dev/null | sed 's|.*vulnerabilities/||' | awk -F: '{printf \"%s=%s; \", \$1, \$2}'")" \
     "no vulnerabilities sysfs"
  # Page size and transparent hugepages affect TLB behaviour for the workspace.
  pv page_size_bytes "$(pfirst 'getconf PAGESIZE' 'sysctl -n hw.pagesize')" "no getconf and no hw.pagesize"
  pv transparent_hugepage "$(pfirst "cat /sys/kernel/mm/transparent_hugepage/enabled")" \
     "no THP sysfs -- not a Linux kernel with THP"
  pv memory_total "$(pfirst "awk '/^MemTotal/{print \$2\" kB\"}' /proc/meminfo" \
        "sysctl -n hw.memsize | awk '{print \$1/1024\" kB\"}'")" "no /proc/meminfo and no hw.memsize"
  pv cpu_flags_simd "$(pfirst "sed -n 's/^flags[ \t]*: //p' /proc/cpuinfo | head -1 | tr ' ' '\n' | grep -E '^(avx512f|avx512dq|avx512vl|avx2|fma|avx|sse2)$' | sort | tr '\n' ' '" \
                             "sysctl -n machdep.cpu.leaf7_features machdep.cpu.features | tr ' ' '\n' | grep -iE '^(avx512f|avx512dq|avx2|fma|avx1\.0|sse2)$' | sort | tr '\n' ' '" \
                             "sysctl -n hw.optional.arm64 >/dev/null && echo 'arm64 (no x86 SIMD; libmatern builds for x86-64 only)'")" \
     "no /proc/cpuinfo flags and no machdep.cpu sysctl"
  # FMA contraction is a per-object build choice: fused-instruction counts per object follow
  # at the end of this block.
  # _gpu_part: the run has a GPU part (run_gpu.sh) or device objects are built; only then are
  # the nvcc and device-architecture keys recorded, which otherwise hold Makefile defaults.
  _gpu_part=no
  { [ -n "${BENCH_GPU_PIPELINE:-}" ] || [ -f "$BUILD/cuda_bench.o" ]; } && _gpu_part=yes
  # The device: model, clocks, driver and throttle state; a missing GPU is noted, not a failure.
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu               $(_gpuq name)"
    echo "gpu_uuid          $(_gpuq uuid)"
    echo "gpu_compute_cap   $(_gpuq compute_cap)"
    echo "gpu_memory_total  $(_gpuq memory.total)"
    echo "gpu_driver        $(_gpuq driver_version)"
    echo "gpu_ecc           $(_gpuq ecc.mode.current)"
    echo "gpu_persistence   $(_gpuq persistence_mode)"
    echo "gpu_clock_sm_max  $(_gpuq clocks.max.sm)"
    echo "gpu_clock_mem_max $(_gpuq clocks.max.memory)"
    echo "gpu_clock_sm_now  $(_gpuq clocks.current.sm)"
    echo "gpu_power_limit   $(_gpuq power.limit)"
    echo "gpu_power_default $(_gpuq power.default_limit)"
    # The throttle reasons, decoded; bit 0 (GpuIdle) is what an idle card reports before the
    # run, not a cap.
    _thr=$(_gpuq clocks_throttle_reasons.active)
    _thrn=$(printf '%d' "$_thr" 2>/dev/null || echo 0)
    _why=""
    [ $(( _thrn & 0x1   )) -ne 0 ] && _why="$_why GpuIdle"
    [ $(( _thrn & 0x2   )) -ne 0 ] && _why="$_why ApplicationsClocksSetting"
    [ $(( _thrn & 0x4   )) -ne 0 ] && _why="$_why SwPowerCap"
    [ $(( _thrn & 0x8   )) -ne 0 ] && _why="$_why HwSlowdown"
    [ $(( _thrn & 0x10  )) -ne 0 ] && _why="$_why SyncBoost"
    [ $(( _thrn & 0x20  )) -ne 0 ] && _why="$_why SwThermalSlowdown"
    [ $(( _thrn & 0x40  )) -ne 0 ] && _why="$_why HwThermalSlowdown"
    [ $(( _thrn & 0x80  )) -ne 0 ] && _why="$_why HwPowerBrakeSlowdown"
    [ $(( _thrn & 0x100 )) -ne 0 ] && _why="$_why DisplayClockSetting"
    [ -z "$_why" ] && _why=" none"
    echo "gpu_throttle      $_thr --$_why"
    # The capping subset (power and thermal reasons).
    if [ $(( _thrn & 0xEC )) -ne 0 ]; then
      echo "gpu_throttle_cap  YES -- a power or thermal cap was active AT SAMPLE TIME"
    else
      echo "gpu_throttle_cap  no power or thermal cap at sample time"
    fi
    # nvidia-smi names the header field "CUDA Version" or "CUDA UMD Version", depending on the driver.
    echo "gpu_cuda_driver   $(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9.]*\).*/\2/p' | head -1)"
  else
    echo "gpu               none -- no nvidia-smi on this machine (CPU run, no device rows)"
  fi
  [ $_gpu_part = yes ] && echo "nvcc_on_path      $(nvcc --version 2>/dev/null | sed -n 's/.*release /release /p' | head -1)"
  # What built the artefacts, read from the artefacts (they may come prebuilt): detect
  # --build-info prints `build_<artefact>_<key>` lines from each object's .comment stamp and
  # each driver's compiled-in record.
  if [ -x "$BUILD/detect" ]; then
    ./"$BUILD"/detect --build-info "$BUILD" 2>/dev/null
  else
    pv build_info "" "build/detect is not built yet; run 'make benchmark-build'"
  fi
  # The make variables as the build resolved them (build/BUILD_KNOBS, written with each driver).
  if [ -f "$BUILD/BUILD_KNOBS" ] && [ $_gpu_part = yes ]; then
    cat "$BUILD/BUILD_KNOBS"
  elif [ -f "$BUILD/BUILD_KNOBS" ]; then
    grep -Ev '^(make_nvcc|make_gpu_arch|make_gpu_arch_use) ' "$BUILD/BUILD_KNOBS"
  else
    pv make_knobs "" "no $BUILD/BUILD_KNOBS; the drivers were built by a Makefile that does not write it"
  fi
  # FMA read from the objects (FMA=on adds -mfma to the baseline scalar objects and to matched);
  # a disagreement with build/BUILD_KNOBS is named.
  _fobj="$BUILD/backends/vector_matched.o"
  _mk_fma=$(sed -n 's/^make_fma  *\([a-z]*\) .*/\1/p' "$BUILD/BUILD_KNOBS" 2>/dev/null | head -1)
  if [ -f "$_fobj" ] && command -v objdump >/dev/null 2>&1; then
    _nf=$(objdump -d "$_fobj" 2>/dev/null | grep -c 'vfmadd\|vfmsub\|vfnmadd\|vfnmsub')
    if [ "${_nf:-0}" -gt 0 ]; then _fma=on; else _fma=off; fi
    if [ -n "$_mk_fma" ] && [ "$_mk_fma" != "$_fma" ]; then
      echo "fma_knob          FMA=$_fma -- from the objects ($_nf fused instructions in vector_matched.o); build/BUILD_KNOBS says FMA=$_mk_fma, so the library predates the last make"
    else
      echo "fma_knob          FMA=$_fma -- from the objects ($_nf fused instructions in vector_matched.o)${_mk_fma:+; make FMA=$_mk_fma}"
    fi
  else
    pv fma_knob "${_mk_fma:+FMA=$_mk_fma -- from build/BUILD_KNOBS; the objects could not be inspected}" \
       "no $_fobj or no objdump"
  fi
  # The device architectures, read with cuobjdump: the drivers' device object
  # build/cuda_bench.o (build/cuda.o with the benchmark hooks) is built for GPU_ARCH, the
  # drivers for the architecture detected at build time; both are recorded.
  _cuobj=$(command -v cuobjdump 2>/dev/null)
  if [ -z "$_cuobj" ]; then
    _nv=$(sed -n 's/^make_nvcc  *\([^ ]*\) .*/\1/p' "$BUILD/BUILD_KNOBS" 2>/dev/null | head -1)
    [ -n "$_nv" ] && [ -x "$(dirname "$_nv")/cuobjdump" ] && _cuobj="$(dirname "$_nv")/cuobjdump"
  fi
  # `sm_80 (ptx sm_80)`: the SASS architectures, then the PTX embedded for JIT.
  _archs(){
    _e=$("$_cuobj" --list-elf "$1" 2>/dev/null | grep -o 'sm_[0-9]*[a-z]*' | sort -u | tr '\n' ' ' | sed 's/ $//')
    _p=$("$_cuobj" --list-ptx "$1" 2>/dev/null | grep -o 'sm_[0-9]*[a-z]*' | sort -u | tr '\n' ' ' | sed 's/ $//')
    printf '%s%s' "${_e:-no sass}" "${_p:+ (ptx $_p)}"; }
  if [ $_gpu_part = no ]; then
    :
  elif [ -n "$_cuobj" ] && [ -f "$BUILD/cuda_bench.o" ]; then
    _al=$(_archs "$BUILD/cuda_bench.o"); _ad=""
    for _d in benchmark_gpu suppliers_gpu; do
      [ -x "$BUILD/$_d" ] && _ad="${_ad:+$_ad; }$_d $(_archs "$BUILD/$_d")"
    done
    echo "gpu_arch_built    library ${_al:-none}${_ad:+; $_ad} -- read with cuobjdump from build/cuda_bench.o and the drivers"
  elif [ -f "$BUILD/cuda_bench.o" ]; then
    echo "gpu_arch_built    unknown (no cuobjdump to read build/cuda_bench.o; see make_gpu_arch)"
  else
    echo "gpu_arch_built    unknown (no device objects built)"
  fi
  # The scalar build selected on this CPU for MATERN_BACKEND=scalar (scalar_portable: x86-64).
  if [ -x "$BUILD/detect" ]; then
    pv scalar_isa "$(./"$BUILD"/detect --scalar-isa 2>/dev/null)" "detect --scalar-isa printed nothing"
  fi
  # The ULP metric of the accuracy columns (text in benchmark/common/ulpmodes.h).
  if [ -x "$BUILD/detect" ]; then
    ./"$BUILD"/detect --ulp 2>/dev/null
  fi
  # detect --timer warns on stderr, which is not redirected, so the terminal sees it too.
  if [ -x "$BUILD/detect" ]; then
    ./"$BUILD"/detect --timer
  fi
  for o in backends/vector_avx512 backends/vector_avx2 backends/vector_matched \
           backends/scalar/debye_uniform_order backends/scalar_avx512 backends/scalar_avx2 \
           backends/scalar_fma backends/dispatch \
           backends/scalar_unfolded_direct_avx512 backends/scalar_unfolded_explog_avx512 \
           backends/scalar_unfolded_direct_avx2 backends/scalar_unfolded_explog_avx2 \
           backends/scalar_unfolded_direct_fma backends/scalar_unfolded_explog_fma; do
    f="$BUILD/$o.o"
    [ -f "$f" ] && echo "fma_insns_$(basename "$o")  $(objdump -d "$f" 2>/dev/null | grep -c 'vfmadd\|vfmsub\|vfnmadd')"
  done
} > "$OUT/PROVENANCE.txt"
record provenance 0

# The corpus files against the sha256 list of data/MANIFEST.md (tools/corpus_check.sh: 0 all
# match, 3 the present ones match, 1 a mismatch or an unlisted file).  make pipeline checks
# them once for all its runs and passes the result in the file BENCH_CORPUS_CHECK (first line
# `<exit status> <DATA>`, then the output of corpus_check.sh --list); it is taken only for the
# same DATA, else the files are checked here.
_cc_rc=""
if [ -n "${BENCH_CORPUS_CHECK:-}" ] && [ -r "$BENCH_CORPUS_CHECK" ]; then
    read -r _cc_rc _cc_data < "$BENCH_CORPUS_CHECK"
    case "$_cc_rc" in [0-9]|[0-9][0-9]|[0-9][0-9][0-9]) ;; *) _cc_rc="" ;; esac
    [ "$_cc_data" = "$DATA" ] || _cc_rc=""
fi
if [ -n "$_cc_rc" ]; then
    progress "manifest: sha256 of the sources; the corpus check made before the runs"
    _cc_out=$(tail -n +2 "$BENCH_CORPUS_CHECK")
else
    progress "manifest: sha256 of the corpus and the sources"
    _cc_out=$(bash tools/corpus_check.sh "$DATA" --list 2>/dev/null); _cc_rc=$?
fi
case $_cc_rc in
    0|3) ;;
    1) warn "corpus: sha256 differs from data/MANIFEST.md for $(printf '%s\n' "$_cc_out" | tail -n 1 \
            | sed -n 's/.*MISMATCH: *\([^;]*\).*/\1/p') -- results read from them are not those of the published corpus" ;;
    *) warn "corpus not checked (tools/corpus_check.sh exited $_cc_rc)" ;;
esac
{
  echo "# sha256 of every input and every source this run depended on."
  echo "# A run that read a modified corpus is then detectable from the record alone."
  for f in data/MANIFEST.md data/DATASETS.md; do
    [ -f "$f" ] && sha256sum "$f"
  done
  # The corpus files, checked against the sha256 list of data/MANIFEST.md; every file present
  # is hashed, since every stage reads the 2^25 sets.
  if [ -n "$_cc_out" ]; then
    printf '%s\n' "$_cc_out" | sed '$s/^/# /'
  else
    echo "# corpus not checked (tools/corpus_check.sh exited $_cc_rc)"
  fi
  for f in benchmark/*.c benchmark/*.h benchmark/common/*.c benchmark/common/*.h; do
    [ -f "$f" ] && sha256sum "$f"
  done
  for f in benchmark/*.cu benchmark/*.cpp; do
    [ -f "$f" ] && sha256sum "$f"
  done
  # The library sources and the build description: equal digests mean the same library source
  # built by the same rules.
  find src include -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
      | LC_ALL=C sort | while IFS= read -r f; do sha256sum "$f"; done
  for f in Makefile matern.map; do
    [ -f "$f" ] && sha256sum "$f"
  done
  # The supplier sources.  GSL is linked as a built library, so its versions and location are
  # PROVENANCE keys (build_cpu_supplier_gsl_headers, _runtime, _source and _library), not a
  # digest.
  for f in references/*/*.c references/*/*.cpp references/*/port/*.c references/*/port/*.h \
           references/common/*.h references/kaust_logbesselk/derive_port.py references/Makefile; do
    [ -f "$f" ] && sha256sum "$f"
  done
  # KAUST: the downloaded upstream files and the generated port, when present.
  for f in references/kaust_logbesselk/upstream/logbesselk.cu \
           references/kaust_logbesselk/upstream/logbesselk.h \
           references/build/kaust/kaust_derived.h; do
    [ -f "$f" ] && sha256sum "$f"
  done
  [ -f references/build/kaust/kaust_derived.h ] || echo "kaust not ported (make kaust)"
  [ -f "$BUILD/libmatern.a" ] && sha256sum "$BUILD/libmatern.a"
  # The fold stage's unfolded twin is not in libmatern.a: its sources, driver and objects.
  if [ "$STAGES" = fold ]; then
    for f in src/backends/scalar/*.c src/backends/scalar/*.h "$BUILD"/fold_cpu \
             "$BUILD"/backends/scalar_control_*.o "$BUILD"/backends/scalar_unfolded_*.o; do
      [ -f "$f" ] && sha256sum "$f"
    done
  fi
} > "$OUT/MANIFEST.txt" 2>/dev/null
record manifest 0

progress "detect: CPU, timer and placement"
./"$BUILD"/detect "$DATA" > "$OUT/detect.txt" 2>&1
record detect $?
# detect's warnings (placement, timer) are repeated on the terminal.
grep '^WARNING' "$OUT/detect.txt" 2>/dev/null | while IFS= read -r _w; do say "$_w"; done
# The placement after NUMA binding, which the timings ran on (affinity_mask above is the
# shell's, before binding).
grep -m1 '^  affinity_after_bind' "$OUT/detect.txt" 2>/dev/null \
  | sed 's/^  affinity_after_bind */affinity_after_bind /' >> "$OUT/PROVENANCE.txt" || true
grep -m1 '^  placement' "$OUT/detect.txt" 2>/dev/null \
  | sed 's/^  placement  */placement         /' >> "$OUT/PROVENANCE.txt" || true

# Raw repetitions are recorded: summary statistics cannot show bimodality.
export BENCH_TIMING_RAW="$OUT/timing_raw"
# Stages are selected by inclusion, so the A/B stages do not run the accuracy sweep.
if [ "$STAGES" = all ] || [ "$STAGES" = rest ]; then
# accuracy_detail.<backend>: histogram, the 64 worst points, excluded points with reason.
progress "accuracy: library, backends $BACKENDS"
BENCH_ACCURACY_DETAIL="$OUT/accuracy_detail" \
./"$BUILD"/benchmark_cpu --family "$FAMILIES" --law all --backend "$BACKENDS" \
    --metric accuracy --reps "$REPS" --format csv --data-dir "$DATA" $time_cap \
    > "$OUT/accuracy.csv" 2>"$OUT/accuracy.err"
record accuracy $?
record_laws accuracy "$OUT/accuracy.err"
fi

if [ "$STAGES" = all ] || [ "$STAGES" = ladder ]; then
# Own raw prefix, so the accuracy stage's timing_raw.<backend> files are not overwritten.
progress "ladder: library, backends $BACKENDS"
BENCH_TIMING_RAW="$OUT/timing_raw.ladder" \
./"$BUILD"/benchmark_cpu --family "$FAMILIES" --law "$LADDER_LAWS" --backend "$BACKENDS" \
    --metric speed --ladder ${_LADDER_MAX_ARG[@]+"${_LADDER_MAX_ARG[@]}"} ${_LADDER_PASS_ARG[@]+"${_LADDER_PASS_ARG[@]}"} --reps "$REPS" --format csv --data-dir "$DATA" \
    > "$OUT/ladder.csv" 2>"$OUT/ladder.err"
record ladder $?
record_laws ladder "$OUT/ladder.err"
fi

# The external suppliers; skipped in benchmark/run_gpu.sh's CPU reference run and without
# build/suppliers.  A C++ compiler, Boost, GSL and the KAUST port each add rows.
sup_note(){
    printf '%-22s %s\n' "$1" "$2" >> "$OUT/STAGES.txt"
    case "$2" in SKIPPED*) SKIPPED+=("$1 ${2#SKIPPED }") ;; esac
}
if [ -n "${BENCH_GPU_PIPELINE:-}" ]; then
    say "suppliers: the CPU suppliers are not part of the GPU pipeline (a CPU run,"
    say "           run_benchmark.sh, measures them; the GPU suppliers are run by"
    say "           benchmark/run_gpu.sh from build/suppliers_gpu) -- skipping both CPU supplier stages."
    sup_note suppliers       "SKIPPED (GPU pipeline: CPU suppliers belong to a CPU run)"
    sup_note supplier-ladder "SKIPPED (GPU pipeline: CPU suppliers belong to a CPU run)"
elif [ ! -x "$BUILD/suppliers" ]; then
    say "suppliers: $BUILD/suppliers is not built -- skipping both supplier stages."
    say "           build them with 'make build/suppliers' in this tree ($PWD);"
    say "           without them the comparison tables and figures have nothing to"
    say "           compare against and will say so by name."
    sup_note suppliers       "SKIPPED (no $BUILD/suppliers)"
    sup_note supplier-ladder "SKIPPED (no $BUILD/suppliers)"
    # SUPPLIERS=all asks for every supplier row of the suppliers stage (STAGES all or rest).
    if [ "${SUPPLIERS:-ship}" = all ] && { [ "$STAGES" = all ] || [ "$STAGES" = rest ]; }; then
        warn "suppliers: SUPPLIERS=all, but $BUILD/suppliers is not built; no supplier rows (make build/suppliers)"
    fi
elif [ "$STAGES" = levin ] || [ "$STAGES" = threads ] || [ "$STAGES" = fold ] || [ "$STAGES" = record ]; then
    # The A/B stages have no supplier side; `record` times nothing.
    sup_note suppliers       "SKIPPED (STAGES=$STAGES)"
    sup_note supplier-ladder "SKIPPED (STAGES=$STAGES)"
else
    # The redistributable set by name: a supplier added to the binary must be added here.
    # Versioned labels (boost-<version>, gsl-<version>) and the rows present come from the binary.
    sup_ship="$(./"$BUILD"/suppliers --list | awk '/^  (bessels\.jl|bessels\.jl-old|std-cyl_bessel_k|boost-|gsl-)/{print $1}' | tr '\n' ' ')"
    sup_all="$(./"$BUILD"/suppliers --list | awk '/^  [a-z]/{print $1}' | tr '\n' ' ')"
    # SUPPLIERS=ship is the set a reader can reproduce; all adds KAUST (not redistributable).
    if [ "${SUPPLIERS:-ship}" = all ]; then sup_list="$sup_all"; else sup_list="$sup_ship"; fi
    # SUP_MAX_POINTS caps the corpus each supplier reads; unset means the whole law.
    sup_cap=""
    [ -n "${SUP_MAX_POINTS:-}" ] && sup_cap="--max-points ${SUP_MAX_POINTS}"
    sup_cap="$sup_cap $time_cap"

    if [ "$STAGES" = ladder ]; then
        sup_note suppliers "SKIPPED (STAGES=ladder)"
    else
        # One header: the first supplier writes it, later ones append their rows only.
        : > "$OUT/suppliers.csv"
        # Each supplier has its own STAGES.txt entry (suppliers:<label>).
        ext_note "suppliers (accuracy and timing per dataset)" cpu SUPPLIERS "$(./"$BUILD"/suppliers --list)"
        # SUPPLIERS=all asks for every row: a row the build lacks is a warning in STATUS.  Labels
        # and row prefixes as in benchmark/tools/missing.sh rows.
        if [ "${SUPPLIERS:-ship}" = all ]; then
            _absent=""
            for _k in "std::cyl_bessel_k|std-cyl_bessel_k" "Boost 1.75.0|boost-" "GSL|gsl-" "KAUST CuBesselK|kaust"; do
                case " $sup_all" in *" ${_k#*|}"*) ;; *) _absent="${_absent:+$_absent, }${_k%%|*}" ;; esac
            done
            [ -z "$_absent" ] ||
                warn "suppliers: SUPPLIERS=all, but $BUILD/suppliers has no rows of $_absent (make requirements)"
        fi
        _pi=0; _pn=$(echo $sup_list | wc -w)
        for sp in $sup_list; do
            _pi=$((_pi+1)); progress "suppliers: [$_pi/$_pn] $sp"
            export BENCH_TIMING_RAW="$OUT/timing_raw.suppliers.$sp"
            _before=$(grep -c '^[0-9]' "$OUT/suppliers.csv")
            if [ -s "$OUT/suppliers.csv" ]; then
                BENCH_ACCURACY_DETAIL="$OUT/accuracy_detail.suppliers.$sp" \
                ./"$BUILD"/suppliers --family "$FAMILIES" --law all --supplier "$sp" \
                    --reps "$REPS" --data-dir "$DATA" $sup_cap --format csv 2>>"$OUT/suppliers.err" \
                    | tail -n +2 >> "$OUT/suppliers.csv"
                _rc=${PIPESTATUS[0]}
            else
                BENCH_ACCURACY_DETAIL="$OUT/accuracy_detail.suppliers.$sp" \
                ./"$BUILD"/suppliers --family "$FAMILIES" --law all --supplier "$sp" \
                    --reps "$REPS" --data-dir "$DATA" $sup_cap --format csv >> "$OUT/suppliers.csv" \
                    2>>"$OUT/suppliers.err"
                _rc=$?
            fi
            unset BENCH_TIMING_RAW
            record_rows "suppliers:$sp" "$_rc" "$_before" "$OUT/suppliers.csv"
        done
        record_laws suppliers "$OUT/suppliers.err"
        sup_note suppliers "$(grep -c '^[0-9]' "$OUT/suppliers.csv") rows"
        say "suppliers (accuracy and timing per dataset): $(grep -c '^[0-9]' "$OUT/suppliers.csv") rows from: $sup_list"
    fi

    # The supplier ladder (time against N): a row walks the rungs while one fits
    # SUP_LADDER_BUDGET (the seconds one rung may take, per supplier and family) and runs one
# rung beyond.  SUP_LADDER=all
    # adds the non-redistributable suppliers.
    if [ "$STAGES" = rest ]; then
        sup_note supplier-ladder "SKIPPED (STAGES=rest)"
    else
        if [ "${SUP_LADDER:-ship}" = all ]; then sup_lad="$sup_all"; else sup_lad="$sup_ship"; fi
        : > "$OUT/suppliers_ladder.csv"
        ext_note "supplier ladder (time against N)" cpu SUP_LADDER "$(./"$BUILD"/suppliers --list)"
        _pi=0; _pn=$(echo $sup_lad | wc -w)
        for sp in $sup_lad; do
            _pi=$((_pi+1)); progress "supplier ladder: [$_pi/$_pn] $sp"
            export BENCH_TIMING_RAW="$OUT/timing_raw.suppliers_ladder.$sp"
            _before=$(grep -c '^[0-9]' "$OUT/suppliers_ladder.csv")
            if [ -s "$OUT/suppliers_ladder.csv" ]; then
                ./"$BUILD"/suppliers --family "$FAMILIES" --law "$LADDER_LAWS" --ladder ${_LADDER_MAX_ARG[@]+"${_LADDER_MAX_ARG[@]}"} ${_LADDER_PASS_ARG[@]+"${_LADDER_PASS_ARG[@]}"} \
                    --ladder-budget "${SUP_LADDER_BUDGET:-20}" --supplier "$sp" --reps "$REPS" \
                    --data-dir "$DATA" --format csv 2>>"$OUT/suppliers_ladder.err" \
                    | tail -n +2 >> "$OUT/suppliers_ladder.csv"
                _rc=${PIPESTATUS[0]}
            else
                ./"$BUILD"/suppliers --family "$FAMILIES" --law "$LADDER_LAWS" --ladder ${_LADDER_MAX_ARG[@]+"${_LADDER_MAX_ARG[@]}"} ${_LADDER_PASS_ARG[@]+"${_LADDER_PASS_ARG[@]}"} \
                    --ladder-budget "${SUP_LADDER_BUDGET:-20}" --supplier "$sp" --reps "$REPS" \
                    --data-dir "$DATA" --format csv >> "$OUT/suppliers_ladder.csv" \
                    2>>"$OUT/suppliers_ladder.err"
                _rc=$?
            fi
            unset BENCH_TIMING_RAW
            record_rows "supplier-ladder:$sp" "$_rc" "$_before" "$OUT/suppliers_ladder.csv"
        done
        record_laws supplier-ladder "$OUT/suppliers_ladder.err"
        sup_note supplier-ladder "$(grep -c '^[0-9]' "$OUT/suppliers_ladder.csv") rows"
        say "supplier ladder (time against N): $(grep -c '^[0-9]' "$OUT/suppliers_ladder.csv") rows from: $sup_lad"
    fi
fi

# Levin A/B: the Levin band included and excluded on one machine, so the difference is the band
# alone.  `variant` is what an arm routes (levin_on/levin_off), `shipped` whether that is the
# backend's default.  In the vector-macro arm scalar is a control: its router ignores that
# macro, so its rows must be identical in both arms.
if [ "$STAGES" = levin ]; then
    # The arms are checked above, before the run directory exists.
    say "levin: base[$_a] novec[$_b] scal[$_c]"

    : > "$OUT/levin.err"
    for arm in base novec scal; do
        progress "levin: arm $arm, backends $BACKENDS"
        # One raw prefix per arm, since the three arms time the same (law, backend) rows.
        BENCH_TIMING_RAW="$OUT/timing_raw.levin_$arm" \
        BENCH_ACCURACY_DETAIL="$OUT/accuracy_detail.levin_$arm" \
        ./"$AB"/benchmark_cpu_"$arm" --family "$FAMILIES" --law all --backend "$BACKENDS" \
            --metric accuracy --reps "$REPS" --format csv --data-dir "$DATA" $time_cap \
            > "$OUT/levin_$arm.csv" 2>>"$OUT/levin.err"
        record "levin_$arm" $?
    done

    # One levin_on and one levin_off row per (backend, family, law), from the arm in which that
    # backend's router changed: base (every backend: vector levin_on, scalar levin_off), novec
    # (vector backends, band removed), scal (scalar, band added).  Duplicates are checked below.
    : > "$OUT/levin.csv"
    awk -F, 'NR==1 { print "variant,shipped," $0; next }
                   { print ($6 == "scalar" ? "levin_off" : "levin_on") ",yes," $0 }' \
        "$OUT/levin_base.csv"  >> "$OUT/levin.csv"
    awk -F, 'NR==1 { next } $6 != "scalar" { print "levin_off,no," $0 }' \
        "$OUT/levin_novec.csv" >> "$OUT/levin.csv"
    awk -F, 'NR==1 { next } $6 == "scalar" { print "levin_on,no," $0 }' \
        "$OUT/levin_scal.csv"  >> "$OUT/levin.csv"

    # The two macros must be independent: neither may move the other router's rows.  Fields 1..31
    # (schema..worst_arg, must match the CSV schema) are the accuracy columns, compared byte for
    # byte; no rows of a class on either side skips the check, rows on one side only fail it.
    _xcheck() {   # _xcheck <label> <fileA> <fileB> <awk-test-on-$6>
        local xa="$OUT/levin_xcheck_$1.a" xb="$OUT/levin_xcheck_$1.b"
        awk -F, "NR>1 && $4 { for(i=1;i<=31;i++) printf \"%s,\", \$i; print \"\" }" \
            "$2" > "$xa"
        awk -F, "NR>1 && $4 { for(i=1;i<=31;i++) printf \"%s,\", \$i; print \"\" }" \
            "$3" > "$xb"
        if [ ! -s "$xa" ] && [ ! -s "$xb" ]; then
            say "levin: cross-check skipped -- $1 (no rows of this class in either arm; BACKENDS=$BACKENDS)"
            sup_note "levin_xcheck_$1" "SKIPPED (no rows of this class in either arm)"
        elif [ ! -s "$xa" ] || [ ! -s "$xb" ]; then
            say "levin: CROSS-CHECK FAILED -- $1 (rows on one side only)"
            record "levin_xcheck_$1" 1
        elif cmp -s "$xa" "$xb"; then
            say "levin: cross-check OK -- $1"
            printf '%-22s ok\n' "levin_xcheck_$1" >> "$OUT/STAGES.txt"
        else
            say "levin: CROSS-CHECK FAILED -- $1 (the two macros are not independent)"
            record "levin_xcheck_$1" 1
        fi
        rm -f "$xa" "$xb"
    }
    _xcheck scalar_unmoved_by_vector_macro \
            "$OUT/levin_base.csv" "$OUT/levin_novec.csv" '$6 == "scalar"'
    _xcheck vector_unmoved_by_scalar_macro \
            "$OUT/levin_base.csv" "$OUT/levin_scal.csv"  '$6 != "scalar"'

    record levin 0
fi

# Thread scaling (weak): each thread evaluates its own THREADS_N points (benchmark/threads_cpu.c).
# One process per (family, backend), since the dispatch caches the backend per process.
if [ "$STAGES" = threads ]; then
    # threads_cpu is checked above, before the run directory exists.
    : > "$OUT/threads.csv"; : > "$OUT/threads.err"; _first=1
    ext_note "threads (supplier rows)" threads THREADS_SUPPLIERS "$(BENCH_TIMING_RAW= ./"$BUILD"/threads_cpu --list)"
    for _fam in $(echo "$FAMILIES" | tr ',' ' '); do
      for _bk in $(echo "$BACKENDS" | tr ',' ' '); do
        # Pin modes: dense fills one socket first; spread deals the threads round-robin over the
        # NUMA nodes, so it recovers a cache-capacity loss but not a clock cap.
        for _pm in $(echo "${THREADS_PIN_MODES:-dense}" | tr ',' ' '); do
          _tag="${_fam}_${_bk}"; [ "$_pm" = dense ] || _tag="${_tag}_${_pm}"
          progress "threads: $_fam $_bk ($_pm)"
          # One raw file per invocation, so processes do not truncate each other's.
          BENCH_TIMING_RAW="$OUT/timing_raw.threads_${_tag}" \
          ./"$BUILD"/threads_cpu $THREADS_CAP --family "$_fam" --law "$THREADS_LAW" --backend "$_bk" \
              --reps "$THREADS_REPS" --n "$THREADS_N" --pin-mode "$_pm" \
              --format csv --data-dir "$DATA" \
              > "$OUT/threads_${_tag}.csv" 2>>"$OUT/threads.err"
          record "threads_${_tag}" $?
          awk -v f="$_first" 'NR==1 { if (f==1) print; next } { print }' \
              "$OUT/threads_${_tag}.csv" >> "$OUT/threads.csv"
          _first=0
        done
      done
      # Suppliers: one process per row, since threads_cpu takes its scaling baseline from the
      # first thread count of an invocation; timed in the assembly of their accuracy rows
      # (THREADS_ASSEMBLY overrides it).  KAUST only with THREADS_SUPPLIERS=all.
      if [ "${THREADS_SUPPLIERS:-ship}" = all ]; then
        _splist=$(BENCH_TIMING_RAW= ./"$BUILD"/threads_cpu --list | awk '/^  [a-z]/{print $1}')
      else
        _splist=$(BENCH_TIMING_RAW= ./"$BUILD"/threads_cpu --list | awk '/^  [a-z]/{print $1}' | grep -v '^kaust')
      fi
      for _sp in $_splist; do
        progress "threads: $_fam $_sp"
        BENCH_TIMING_RAW="$OUT/timing_raw.threads_${_fam}_${_sp}" \
        ./"$BUILD"/threads_cpu $THREADS_CAP --family "$_fam" --law "$THREADS_LAW" \
            --supplier "$_sp" --assembly "${THREADS_ASSEMBLY:-direct}" \
            --reps "$THREADS_REPS" --n "$THREADS_N" --format csv --data-dir "$DATA" \
            > "$OUT/threads_${_fam}_${_sp}.csv" 2>>"$OUT/threads.err"
        record "threads_${_fam}_${_sp}" $?
        awk -v f="$_first" 'NR==1 { if (f==1) print; next } { print }' \
            "$OUT/threads_${_fam}_${_sp}.csv" >> "$OUT/threads.csv"
        _first=0
      done
    done
    # Working-set sweep (opt-in, THREADS_WS_N=<list>): thread count fixed, --n swept; a cache
    # capacity effect shows as a cliff where ws_bytes crosses llc_bytes, a clock cap as none.
    if [ -n "${THREADS_WS_N:-}" ]; then
      : > "$OUT/threads_ws.csv"; _wfirst=1
      _wt="${THREADS_WS_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 32)}"
      # THREADS_MAX caps the default count as well.
      [ -z "${THREADS_WS_THREADS:-}" ] && [ -n "$THREADS_MAX" ] && [ "$THREADS_MAX" -lt "$_wt" ] && _wt=$THREADS_MAX
      for _fam in $(echo "$FAMILIES" | tr ',' ' '); do
        for _bk in $(echo "${THREADS_WS_BACKENDS:-$BACKENDS}" | tr ',' ' '); do
          for _wn in $(echo "$THREADS_WS_N" | tr ',' ' '); do
            progress "threads working set: $_fam $_bk n=$_wn"
            BENCH_TIMING_RAW="$OUT/timing_raw.threads_ws_${_fam}_${_bk}_${_wn}" \
            ./"$BUILD"/threads_cpu $THREADS_CAP --family "$_fam" --law "$THREADS_LAW" \
                --backend "$_bk" --threads "$_wt" --reps "$THREADS_REPS" \
                --n "$_wn" --format csv --data-dir "$DATA" \
                > "$OUT/threads_ws_${_fam}_${_bk}_${_wn}.csv" 2>>"$OUT/threads.err"
            record "threads_ws_${_fam}_${_bk}_${_wn}" $?
            awk -v f="$_wfirst" 'NR==1 { if (f==1) print; next } { print }' \
                "$OUT/threads_ws_${_fam}_${_bk}_${_wn}.csv" >> "$OUT/threads_ws.csv"
            _wfirst=0
          done
        done
      done
      record threads_ws 0
    fi
    record threads 0
fi

# Prefactor fold: the scalar backend against its unfolded twin (src/backends/scalar/
# scalar_internal.h), whose regions return K_nu and apply the Matern prefactor outside.
# fold_cpu is benchmark_cpu.c with -DBENCH_FOLD, so fold.csv has accuracy.csv's layout; backends:
#   scalar                  the shipped scalar backend
#   scalar-control          the shipped source built as the twin is
#   scalar-unfolded-direct  the twin, M = 2 (z/2)^nu / Gamma(nu) * K_nu
#   scalar-unfolded-explog  the twin, M = exp((1-nu) ln2 - lnGamma(nu) + nu ln z + ln K_nu)
if [ "$STAGES" = fold ]; then
    progress "fold: the scalar backend against its unfolded twin"
    BENCH_TIMING_RAW="$OUT/timing_raw.fold" BENCH_ACCURACY_DETAIL="$OUT/accuracy_detail.fold" \
        ./"$BUILD"/fold_cpu --family "$FAMILIES" --law all --metric accuracy \
        --reps "$REPS" --format csv --data-dir "$DATA" $time_cap \
        > "$OUT/fold.csv" 2>>"$OUT/fold.err"
    record fold $?
    record_laws fold "$OUT/fold.err"
    # Cross-checks on fields 1-31 (backend aside): the control equals scalar (the twin's build
    # route moves no result), and the direct twin differs from it on some law.
    _fold_rows(){ awk -F, -v b="$1" 'NR>1 && $6==b' "$OUT/fold.csv" | cut -d, -f1-5,7-31; }
    if [ "$(_fold_rows scalar | md5sum)" = "$(_fold_rows scalar-control | md5sum)" ] &&
       [ -n "$(_fold_rows scalar)" ]; then
        printf '%-22s ok\n' fold_xcheck_control_equals_scalar >> "$OUT/STAGES.txt"
    else
        record fold_xcheck_control_equals_scalar 1
    fi
    if [ -n "$(_fold_rows scalar-unfolded-direct)" ] &&
       [ "$(_fold_rows scalar | md5sum)" != "$(_fold_rows scalar-unfolded-direct | md5sum)" ]; then
        printf '%-22s ok\n' fold_xcheck_twin_differs >> "$OUT/STAGES.txt"
    else
        record fold_xcheck_twin_differs 1
    fi
    say "fold: $(grep -c '^[0-9]' "$OUT/fold.csv") rows (scalar, control, direct and exp-log twins)"
fi

# Expectations: against accuracy.csv or, for STAGES=levin, its default-routing rows (which also
# verifies that side of the A/B is the default build).
EXP_SRC="$OUT/accuracy.csv"
if [ ! -f "$EXP_SRC" ] && [ -f "$OUT/levin.csv" ]; then
    { head -1 "$OUT/levin.csv"; awk -F, 'NR>1 && $2=="yes"' "$OUT/levin.csv"; } \
        | cut -d, -f3- > "$OUT/exp_source.csv"
    EXP_SRC="$OUT/exp_source.csv"
fi

if [ -f "$EXP_SRC" ]; then
# The expected figures are reference values, not limits; nothing fails on them.
{
  echo "# expected vs measured; a difference is reported, not treated as a failure."
  printf '%-10s %-12s %-10s %-12s %-12s %s\n' family law backend expected measured note
  awk -F, -v OFS=' ' '
    NR==1 { for (i=1;i<=NF;i++) col[$i]=i; next }
    {
      key = $col["family"] "/" $col["law"] "/" $col["backend"]
      want = ""
      if (key == "z_nu/grid/avx512")     want = 12.4813
      if (key == "z_nu/large_nu/avx512") want = 25.1282
      if (key == "z_nu/large_z/avx512")  want = 14.6549
      if (key == "z_nu/seam/avx512")     want = 22.4065
      if (want == "") next
      m = $col["u_max"] + 0
      d = m - want; if (d < 0) d = -d
      printf "%-10s %-12s %-10s %-12s %-14.4f %s\n", $col["family"], $col["law"],
             $col["backend"], want, m,
             (d < 1e-3 ? "unchanged" : "MOVED (docs/benchmark.md, Output)")
    }' "$EXP_SRC"
} > "$OUT/EXPECTED.txt" 2>"$OUT/expected.err"
record expectations 0
fi

LOAD_AFTER="$(pfirst \
    'cut -d" " -f1-3 /proc/loadavg' \
    'sysctl -n vm.loadavg | tr -d "{}" | sed "s/^ *//"' \
    'uptime | sed "s/.*average[s]*: //" | tr -d ","')"
# The clock of the timing CPU right after each timed row (mhz_end of the .sched.csv sidecars,
# cpufreq's scaling_cur_freq): the median and range over the rows of this run.
CPU_MHZ_TIMED=$(cat "$OUT"/timing_raw.*.sched.csv 2>/dev/null | awk -F, '
    $1 == "device" { c = 0; for (i = 1; i <= NF; i++) if ($i == "mhz_end") c = i; next }
    c && $c > 0 { print $c + 0 }' | sort -n | awk '
    { v[++n] = $1 }
    END { if (!n) exit
          m = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
          printf "%d MHz median (%d..%d MHz) -- over %d timed rows: scaling_cur_freq of the timing CPU after each row (timing_raw.*.sched.csv)", m, v[1], v[n], n }')
{
  pv load_after    "$LOAD_AFTER"              "no /proc/loadavg, no vm.loadavg sysctl and no uptime(1)"
  pv load_after_pc "$(_loadpc "$LOAD_AFTER")" "needs both a load average and a cpu count"
  # absent where no row recorded one: no cpufreq, or the threads stage (mhz_end in threads.csv)
  [ -z "$CPU_MHZ_TIMED" ] || pv cpu_mhz_timed "$CPU_MHZ_TIMED"
} >> "$OUT/PROVENANCE.txt"

# The accuracy detail files are compressed: a supplier that fails widely lists many excluded
# points, whose repeated label columns compress well.
for _d in "$OUT"/accuracy_detail.*; do
    case "$_d" in *.gz) ;; *) [ -f "$_d" ] && gzip -f "$_d" ;; esac
done

# `grep -c` prints 0 and exits 1 when nothing matches, so `|| echo 0` would yield "0\n0";
# the count is defaulted instead.  The threads and fold stages write no accuracy rows.
ROW_SRC="$EXP_SRC"; ROW_KIND="accuracy"
if [ ! -f "$ROW_SRC" ] && [ -f "$OUT/threads.csv" ]; then
    ROW_SRC="$OUT/threads.csv"; ROW_KIND="thread-scaling"
elif [ ! -f "$ROW_SRC" ] && [ -f "$OUT/fold.csv" ]; then
    ROW_SRC="$OUT/fold.csv"; ROW_KIND="fold"
fi
# STAGES=record times nothing on the CPU: complete when the record is, CPU stages skipped.
if [ "$STAGES" = record ]; then
    ROW_KIND="CPU"
    SKIPPED+=("accuracy (STAGES=record: no CPU measurement)")
    SKIPPED+=("ladder (STAGES=record: no CPU measurement)")
fi
rows=$(grep -c '^[0-9]' "$ROW_SRC" 2>/dev/null); rows=${rows:-0}
printf '%-40s %d\n' run_total "$SECONDS" >> "$OUT/TIMES.txt"
{
  if [ "$stage_rc" -eq 0 ]; then
      echo "STATUS   complete"
  elif [ "$rows" -gt 0 ]; then
      echo "STATUS   partial"
  else
      echo "STATUS   failed"
  fi
  echo "run_id   $RUNID"
  echo "rows     $rows $ROW_KIND rows"
  if [ ${#MISSING[@]} -gt 0 ]; then
      echo "missing  ${MISSING[*]}"
      echo "         these stages, suppliers or datasets did not complete; anything"
      echo "         derived from this run must say so."
  else
      echo "missing  nothing"
  fi
  # Skipped is not failed: `complete` with a skipped line means everything that ran completed.
  if [ ${#SKIPPED[@]} -gt 0 ]; then
      _first=1
      for _s in "${SKIPPED[@]}"; do
          if [ $_first -eq 1 ]; then echo "skipped  $_s"; _first=0; else echo "         $_s"; fi
      done
  else
      echo "skipped  nothing"
  fi
  if [ ${#WARNINGS[@]} -gt 0 ]; then
      for _w in "${WARNINGS[@]}"; do echo "warning  $_w"; done
  fi
  echo "load     before=$LOAD_BEFORE after=$LOAD_AFTER"
  echo "note     accuracy does not depend on the load, but on the build (compiler, flags, libm), and avx2's on the processor vendor; timings on a machine shared with other processes can be slower."
} > "$OUT/STATUS"

cat "$OUT/STATUS"

# End-of-run notification (benchmark/tools/notify.sh); make benchmark and make pipeline set
# BENCH_NOTIFY_SKIP and notify once.
if [ -z "${BENCH_NOTIFY_SKIP:-}" ]; then
    _el=$SECONDS
    _msg="$(sed -n 's/^STATUS *//p' "$OUT/STATUS" | head -1) after $(printf '%d:%02d:%02d' $((_el/3600)) $((_el%3600/60)) $((_el%60))) -- $OUT"
    [ ${#MISSING[@]} -gt 0 ] && _msg="$_msg; missing: ${MISSING[*]}"
    [ ${#SKIPPED[@]} -gt 0 ] && _msg="$_msg; skipped: $(printf '%s\n' "${SKIPPED[@]}" | cut -d' ' -f1 | tr '\n' ' ' | sed 's/ $//')"
    [ ${#WARNINGS[@]} -gt 0 ] && _msg="$_msg; warning: $(printf '%s; ' "${WARNINGS[@]}" | sed 's/; $//')"
    bash ./benchmark/tools/notify.sh "run_benchmark" "$_msg"
fi
exit "$stage_rc"
