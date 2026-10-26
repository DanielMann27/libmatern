#!/bin/bash
# run_gpu.sh -- the CUDA part of the benchmark: builds the device drivers, records a one-backend
# CPU reference run, then runs the device stages of GPU_STAGES (core, levin, threads, all; default
# core; call-large only by name) into the same run directory.
set -u

# The reduced-run caps are refused; benchmark/run_benchmark.sh accepts them.
for _cap in TIME_MAX_POINTS SUP_MAX_POINTS; do
    if [ -n "${!_cap:-}" ]; then
        echo "run_gpu.sh: $_cap=${!_cap} is set; the complete benchmark times and scores every" >&2
        echo "run_gpu.sh: point -- unset it, or use benchmark/run_benchmark.sh for a reduced run" >&2
        exit 2
    fi
done
# Step 0: the corpus in DATA, exported to the CPU reference run and passed to every device
# driver as --data-dir; missing files are asked for on a terminal, else the script stops here.
DATA="${DATA:-data}"; export DATA
sh tools/corpus.sh need "benchmark/run_gpu.sh" || exit 2
GPU_STAGES="${GPU_STAGES:-core}"
for _s in $(printf '%s' "$GPU_STAGES" | tr ',' ' '); do
    case "$_s" in core|levin|threads|all|call-large) ;; *)
        echo "run_gpu.sh: GPU_STAGES must be a comma list of core, levin, threads, all, call-large (got '$GPU_STAGES')" >&2
        exit 2 ;;
    esac
done
# call-large as the only stage times the device only: no CPU reference run.
CALL_LARGE_ONLY=0
for _s in $(printf '%s' "$GPU_STAGES" | tr ',' ' '); do
    [ "$_s" = call-large ] || { CALL_LARGE_ONLY=0; break; }
    CALL_LARGE_ONLY=1
done
# Resolved as in run_benchmark.sh, since all ladders of a run share one N axis.
LADDER_SET="${LADDER_SET:-large}"
case "$LADDER_SET" in
    large) LADDER_LAWS="${LADDER_LAWS:-size_random_large,size_half_large}" ;;
    small) LADDER_LAWS="${LADDER_LAWS:-size_random,size_half}" ;;
    *) echo "LADDER_SET must be large|small (got '$LADDER_SET')" >&2; exit 2 ;;
esac
# LADDER_MAX=<N> caps every ladder's top rung, as in run_benchmark.sh.
_LADDER_MAX_ARG=(); [ -n "${LADDER_MAX:-}" ] && _LADDER_MAX_ARG=(--ladder-max "$LADDER_MAX")
export LADDER_SET LADDER_LAWS
RUNS_ROOT="${BENCH_RUNS_DIR:-benchmark/runs}"

echo "=============================================================="
echo "DEVICE    cuda"
echo "cpu       $(sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo | head -1)"
echo "gpu       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "load      $(cut -d' ' -f1-3 /proc/loadavg)"
echo "=============================================================="
t0=$SECONDS

# build <what> <make target...>: output to build/run_gpu.<what>.log, its end printed on failure.
build(){
    local what=$1 log="build/run_gpu.$1.log"; shift
    mkdir -p build
    make "$@" >"$log" 2>&1 && return 0
    echo "!! $what build failed (make $*) -- the end of $log:"
    tail -n 20 "$log" | sed 's/^/   /'
    return 1
}
build library build/libmatern.a || exit 1
build bench   benchmark-build   || exit 1
# nvcc as the Makefile resolves it: NVCC, else PATH, else the CUDA default location.
_nvcc="${NVCC:-$(command -v nvcc 2>/dev/null || ls /usr/local/cuda/bin/nvcc 2>/dev/null)}"
if [ -z "$_nvcc" ] || ! command -v "${_nvcc%% *}" >/dev/null 2>&1; then
  echo "!! no nvcc on this machine -- a GPU run cannot be produced here"
  _add=$(bash benchmark/tools/missing.sh lines nvcc | cut -f2)
  echo "   nvcc: ${_add:-install the NVIDIA CUDA toolkit, or pass NVCC=<path> (make requirements)}"
  exit 3
fi
# nvidia-smi as the Makefile resolves it: NVIDIA_SMI, else PATH, else the WSL location.  A GPU
# is visible when `nvidia-smi -L` succeeds, the test of make pipeline.
_smi="${NVIDIA_SMI:-$(command -v nvidia-smi 2>/dev/null || ls /usr/lib/wsl/lib/nvidia-smi 2>/dev/null)}"
if [ -z "$_smi" ] || ! "$_smi" -L >/dev/null 2>&1; then
  echo "!! nvcc found, but no NVIDIA GPU visible: a GPU run cannot be produced here"
  echo "   nvidia-smi: ${_smi:+$_smi -L fails}${_smi:-not found} (NVIDIA_SMI=<path> names another)"
  echo "   make pipeline or make reproduce runs the CPU part alone with GPU=auto or GPU=no"
  exit 3
fi
# `all` selects the stages of the published run; call-large is selected by name only.
want_gpu(){
    case ",$GPU_STAGES," in *",$1,"*) return 0 ;; esac
    [ "$1" != call-large ] && case ",$GPU_STAGES," in *",all,"*) return 0 ;; esac
    return 1
}
echo "[$((SECONDS-t0))s] gpu stages requested: $GPU_STAGES"
# One probe of the missing software before any timing (make pipeline passes its own), then
# the full-run notice, silent under make pipeline and not repeated by the CPU reference run.
if [ -z "${BENCH_MISSING_CACHE:-}" ]; then
    BENCH_MISSING_CACHE=$(bash benchmark/tools/missing.sh probe); export BENCH_MISSING_CACHE
    trap 'rm -f "$BENCH_MISSING_CACHE"' EXIT
fi
bash benchmark/tools/full_run.sh --entry "bash benchmark/run_gpu.sh" --gpu "$GPU_STAGES"
export BENCH_FULL_RUN_NOTICE=shown

# The pipeline's fixed repetition count, forwarded to every device stage.
REPS="${REPS:-65}"
build gpu-stage build/benchmark_gpu || exit 1
# build/suppliers_gpu builds without Boost and KAUST (this library's kernel only), so a
# failure is a build error and the stage counts as failed.
SUP_GPU_BUILT=0
if build gpu-suppliers build/suppliers_gpu; then
    SUP_GPU_BUILT=1; echo "[$((SECONDS-t0))s] gpu suppliers built"
else
    echo "[$((SECONDS-t0))s] gpu suppliers NOT built (above)"
fi
if make build/boost_shortcut_gpu >/dev/null 2>&1; then
  echo "[$((SECONDS-t0))s] check of Boost's K_nu short cut near nu = 5/2 built"
else
  _add=$(bash benchmark/tools/missing.sh lines boost-gpu | cut -f2)
  echo "[$((SECONDS-t0))s] check of Boost's K_nu short cut near nu = 5/2 NOT built: it needs Boost >= 1.87 with GPU support"
  [ -n "$_add" ] && echo "    Boost >= 1.87 (GPU): $_add"
fi
echo "[$((SECONDS-t0))s] library + gpu stage built"
# The device drivers are built before run_benchmark.sh, whose provenance stage reads their
# build records (build_gpu_*) and device architectures (gpu_arch_built) from the binaries.

# The CPU reference: one backend, accuracy and ladder, so the CPU and GPU ladders come from one
# machine; marked `pipeline gpu` in PROVENANCE.txt.  The backend is the best this CPU runs
# (build/detect --live-backend); BACKENDS overrides it.
if [ -z "${BACKENDS:-}" ]; then
    for _b in avx512 avx2 matched_fma matched scalar; do
        if [ "$(./build/detect --live-backend "$_b" 2>/dev/null)" = "$_b" ]; then BACKENDS=$_b; break; fi
    done
fi
FAILED=""
if [ "$CALL_LARGE_ONLY" = 1 ]; then
    # The run record alone (provenance, manifest, detect), no CPU measurement.
    echo "[$((SECONDS-t0))s] cpu reference: not made (GPU_STAGES=call-large times the device only); run record alone"
    SCOPE="${SCOPE:-gpu-call-large}" STAGES=record BACKENDS="${BACKENDS:-avx512}" BENCH_GPU_PIPELINE=1 \
        bash benchmark/run_benchmark.sh
    _rc=$?
    [ "$_rc" -eq 0 ] || FAILED="$FAILED run-record(rc=$_rc)"
    echo "[$((SECONDS-t0))s] run record done (exit status $_rc)"
else
    echo "[$((SECONDS-t0))s] cpu reference backend: ${BACKENDS:-avx512}"
    SCOPE="${SCOPE:-gpu}" BACKENDS="${BACKENDS:-avx512}" BENCH_GPU_PIPELINE=1 \
        bash benchmark/run_benchmark.sh
    _rc=$?
    [ "$_rc" -eq 0 ] || FAILED="$FAILED cpu-reference(rc=$_rc)"
    echo "[$((SECONDS-t0))s] cpu reference row done (exit status $_rc)"
fi
_latest=$(readlink "$RUNS_ROOT/latest" 2>/dev/null)
[ -n "$_latest" ] && [ -d "$RUNS_ROOT/$_latest" ] || { echo "!! no run directory under $RUNS_ROOT"; exit 1; }
R="$RUNS_ROOT/$_latest"

# snapshot: with BENCH_COPY_RUNS_TO set, the run directory (with the load log BENCH_LOAD_LOG) is
# copied there after each stage, so an interrupted job loses at most the stage in progress.
snapshot() {
    [ -n "${BENCH_COPY_RUNS_TO:-}" ] || return 0
    mkdir -p "$BENCH_COPY_RUNS_TO" 2>/dev/null
    [ -n "${BENCH_LOAD_LOG:-}" ] && cp "$BENCH_LOAD_LOG" "$R/loadavg.tsv" 2>/dev/null
    cp -r "$R" "$BENCH_COPY_RUNS_TO/" 2>/dev/null && return 0
    echo "!! snapshot failed after stage ${1:-?}"
    return 1
}

# note_stage records a stage's row count in STAGES.txt.  A stage that ran and produced no rows
# is recorded as ATTEMPTED, 0 ROWS and marks STATUS (written by run_benchmark.sh) partial.
# Each note_stage also adds the stage's wall time to TIMES.txt (seconds since the previous
# stage), as run_benchmark.sh's record does; run_total at the end covers the whole run.
_note_t=$SECONDS
note_stage() {   # note_stage <name> <csv> <attempted 0|1>
    local name="$1" csv="$2" tried="$3" rows=0
    printf '%-40s %d\n' "$name" "$((SECONDS - _note_t))" >> "$R/TIMES.txt"
    _note_t=$SECONDS
    # Rows are the non-empty lines after the header; gpu_levin.csv rows start
    # with the variant name, not the schema number.
    [ -s "$csv" ] && rows=$(tail -n +2 "$csv" | grep -c . 2>/dev/null); rows=${rows:-0}
    if [ "$tried" -eq 1 ] && [ "$rows" -eq 0 ]; then
        FAILED="$FAILED $name"
        echo "!! STAGE PRODUCED NO ROWS: $name -- see $(basename "${csv%.csv}").err"
        printf '%-24s ATTEMPTED, 0 ROWS\n' "$name" >> "$R/STAGES.txt"
        sed -i 's/^STATUS   complete/STATUS   partial/' "$R/STATUS" 2>/dev/null
        grep -q '^missing  nothing' "$R/STATUS" 2>/dev/null \
            && sed -i "s/^missing  nothing/missing  $name(0 rows)/" "$R/STATUS"
    else
        printf '%-24s %s rows\n' "$name" "$rows" >> "$R/STAGES.txt"
    fi
}


if want_gpu core; then
# core: the library on the device as a whole call; without --time-points every point is timed.
BENCH_TIMING_RAW="$R/timing_raw.gpu" BENCH_ACCURACY_DETAIL="$R/accuracy_detail.gpu" \
./build/benchmark_gpu --family z_nu,h_nu_s --law all --format csv --reps "$REPS" \
    --data-dir "$DATA" > "$R/gpu.csv" 2>"$R/gpu.err"
note_stage gpu "$R/gpu.csv" 1
snapshot gpu
  echo "[$((SECONDS-t0))s] gpu: $(grep -c '^[0-9]' "$R/gpu.csv" 2>/dev/null || true) rows"

# The device ladder (time against N).
BENCH_TIMING_RAW="$R/timing_raw.gpu_ladder" \
./build/benchmark_gpu --family z_nu,h_nu_s --law "$LADDER_LAWS" \
    --ladder ${_LADDER_MAX_ARG[@]+"${_LADDER_MAX_ARG[@]}"} --reps "$REPS" --format csv \
    --data-dir "$DATA" > "$R/gpu_ladder.csv" 2>"$R/gpu_ladder.err"
note_stage gpu-ladder "$R/gpu_ladder.csv" 1
snapshot gpu-ladder
  echo "[$((SECONDS-t0))s] gpu ladder: $(grep -c '^[0-9]' "$R/gpu_ladder.csv" 2>/dev/null || true) rows"

# The device kernels, external and this library's, timed kernel-only; `timing_scope` tells
# these rows from gpu.csv's whole-call rows.
[ "$SUP_GPU_BUILT" = 1 ] || note_stage gpu-kernels "$R/gpu_kernels.csv" 1
if [ "$SUP_GPU_BUILT" = 1 ] && [ -x build/suppliers_gpu ]; then
  bash benchmark/tools/missing.sh rows "[$((SECONDS-t0))s] gpu suppliers (kernels compared kernel-only)" \
      gpu "" "" "$(./build/suppliers_gpu --list 2>/dev/null)"
  BENCH_TIMING_RAW="$R/timing_raw.gpu_kernels" BENCH_ACCURACY_DETAIL="$R/accuracy_detail.gpu_kernels" \
  ./build/suppliers_gpu --family z_nu,h_nu_s --law all --format csv --reps "$REPS" \
      --data-dir "$DATA" > "$R/gpu_kernels.csv" 2>"$R/gpu_kernels.err"
  note_stage gpu-kernels "$R/gpu_kernels.csv" 1
  snapshot gpu-kernels
  echo "[$((SECONDS-t0))s] gpu kernels: $(grep -c '^[0-9]' "$R/gpu_kernels.csv" 2>/dev/null || true) rows"
  # Boost >= 1.86 takes a Hankel short cut that is inaccurate near nu = 5/2; this stage sweeps
  # delta = nu - 2.5 against z (data/boost_shortcut/).  It runs before the kernel ladder, the
  # stage most likely to run out of time, and is skipped without Boost >= 1.87.
  if [ -x build/boost_shortcut_gpu ]; then
    ./build/boost_shortcut_gpu --data-dir "$DATA" > "$R/gpu_boost_shortcut.csv" 2>"$R/gpu_boost_shortcut.err"
    note_stage gpu-boost-shortcut "$R/gpu_boost_shortcut.csv" 1
    snapshot gpu-boost-shortcut
    echo "[$((SECONDS-t0))s] gpu check of Boost's K_nu short cut: $(grep -c '^[0-9]' "$R/gpu_boost_shortcut.csv" 2>/dev/null || true) rows"
  else
    echo "[$((SECONDS-t0))s] gpu check of Boost's K_nu short cut: skipped (needs Boost >= 1.87 with GPU support; see above)"
  fi
  BENCH_TIMING_RAW="$R/timing_raw.gpu_kernels_ladder" \
  ./build/suppliers_gpu --family z_nu,h_nu_s --law "$LADDER_LAWS" --ladder ${_LADDER_MAX_ARG[@]+"${_LADDER_MAX_ARG[@]}"} \
      --reps "$REPS" --format csv --data-dir "$DATA" > "$R/gpu_kernels_ladder.csv" 2>"$R/gpu_kernels_ladder.err"
  note_stage gpu-kernels-ladder "$R/gpu_kernels_ladder.csv" 1
  snapshot gpu-kernels-ladder
  echo "[$((SECONDS-t0))s] gpu kernel ladder: $(grep -c '^[0-9]' "$R/gpu_kernels_ladder.csv" 2>/dev/null || true) rows"
fi
fi   # want_gpu core

# levin: the device Levin-band A/B.  The device default excludes the band (the CPU vector
# backends include it), so the variant arm adds it; rows carry `variant` (what the arm routes)
# and `shipped` (whether that is the device's default routing).
if want_gpu levin; then
  if make levin-ab-build-gpu >"$R/levin_build.log" 2>&1; then
    # The two arms are built after the provenance stage; their build records are
    # appended under their own keys (build_gpu_levin_<arm>_*).
    for arm in levin nolevin; do
      [ -x "build-ab/benchmark_gpu_$arm" ] && ./build-ab/benchmark_gpu_"$arm" --build-info 2>/dev/null \
        | sed "s/^build_gpu_benchmark_\([^ ]*\)/build_gpu_levin_${arm}_\1/" >> "$R/PROVENANCE.txt"
    done
    : > "$R/gpu_levin.csv"; : > "$R/gpu_levin.err"; first=1
    for arm in levin nolevin; do
      case "$arm" in
        levin)   v=levin_on;  sh=no  ;;   # the variant: the device does not ship it
        nolevin) v=levin_off; sh=yes ;;   # the shipped device routing
      esac
      BENCH_TIMING_RAW="$R/timing_raw.gpu_levin_$arm" BENCH_ACCURACY_DETAIL="$R/accuracy_detail.gpu_levin_$arm" \
      ./build-ab/benchmark_gpu_"$arm" --family z_nu,h_nu_s --law all --format csv --reps "$REPS" \
          --data-dir "$DATA" > "$R/gpu_levin_$arm.csv" 2>>"$R/gpu_levin.err"
      awk -v v="$v" -v sh="$sh" -v f="$first" '
          NR==1 { if (f==1) print "variant,shipped," $0; next }
                { print v "," sh "," $0 }' "$R/gpu_levin_$arm.csv" >> "$R/gpu_levin.csv"
      first=0
    done
    note_stage gpu-levin "$R/gpu_levin.csv" 1
    snapshot gpu-levin
    echo "[$((SECONDS-t0))s] gpu levin: $(grep -c '^levin_o' "$R/gpu_levin.csv" 2>/dev/null || true) rows"
  else
    echo "!! gpu levin A/B build failed -- see levin_build.log"
    note_stage gpu-levin "$R/gpu_levin.csv" 1
  fi
fi

# threads: `block` under all four transfer strategies, `host_threads` under the ring only, the
# one strategy that copies on the host (benchmark/threads_gpu.cu).
if want_gpu threads; then
  if build gpu-threads build/threads_gpu; then
    : > "$R/gpu_threads.csv"; : > "$R/gpu_threads.err"; first=1
    for fam in z_nu h_nu_s; do
      BENCH_TIMING_RAW="$R/timing_raw.gpu_threads_$fam" \
      ./build/threads_gpu --family "$fam" --law size_random --format csv --reps "$REPS" \
          --data-dir "$DATA" > "$R/gpu_threads_$fam.csv" 2>>"$R/gpu_threads.err"
      awk -v f="$first" 'NR==1 { if (f==1) print; next } { print }' \
          "$R/gpu_threads_$fam.csv" >> "$R/gpu_threads.csv"
      first=0
    done
    note_stage gpu-threads "$R/gpu_threads.csv" 1
    snapshot gpu-threads
    echo "[$((SECONDS-t0))s] gpu threads: $(grep -c '^[0-9]' "$R/gpu_threads.csv" 2>/dev/null || true) rows"
  else
    echo "!! threads_gpu build failed (above)"
    note_stage gpu-threads "$R/gpu_threads.csv" 1
  fi
fi

# call-large: every device kernel (this library's and the external ones, both assemblies) timed
# as a whole call through the public CUDA entry, transfers included, under the four transfer
# strategies; nothing is scored (ladder layout, timing_scope call).  The laws are fixed to the
# 2^25-point sets, not LADDER_LAWS.
if want_gpu call-large; then
  if [ "$SUP_GPU_BUILT" = 1 ] && [ -x build/suppliers_gpu ]; then
    bash benchmark/tools/missing.sh rows "[$((SECONDS-t0))s] gpu call-large (whole calls at 2^25)" \
        gpu "" "" "$(./build/suppliers_gpu --list 2>/dev/null)"
    BENCH_TIMING_RAW="$R/timing_raw.gpu_call_large" \
    ./build/suppliers_gpu --family z_nu,h_nu_s --law size_random_large,size_half_large \
        --call-only --format csv --reps "$REPS" --data-dir "$DATA" \
        > "$R/gpu_call_large.csv" 2>"$R/gpu_call_large.err"
    note_stage gpu-call-large "$R/gpu_call_large.csv" 1
    snapshot gpu-call-large
    echo "[$((SECONDS-t0))s] gpu call-large: $(grep -c '^[0-9]' "$R/gpu_call_large.csv" 2>/dev/null || true) rows"
  else
    note_stage gpu-call-large "$R/gpu_call_large.csv" 1
  fi
fi

printf '%-40s %d\n' run_total "$SECONDS" >> "$R/TIMES.txt"
echo "[$((SECONDS-t0))s] recorded: $R"
if [ -n "${BENCH_COPY_RUNS_TO:-}" ]; then
  snapshot final || { echo "!! COPY-BACK FAILED -- the run exists only in $R"; exit 2; }
  if [ "$CALL_LARGE_ONLY" = 1 ]; then
    echo "copied back: $BENCH_COPY_RUNS_TO/$_latest"
  else
    echo "copied back: $(grep -c '^[0-9]' "$BENCH_COPY_RUNS_TO/$_latest/gpu.csv" 2>/dev/null || true) gpu rows"
  fi
fi
echo "TOTAL ${SECONDS}s"
cat "$R/STATUS"
# The exit status covers the GPU stages: STATUS, written by the CPU reference run, is only
# marked partial here.
if [ -n "$FAILED" ]; then
    echo "run_gpu.sh: failed:$FAILED"
    exit 1
fi
