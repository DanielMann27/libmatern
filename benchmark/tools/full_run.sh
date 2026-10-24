#!/bin/bash
# full_run.sh [--brief|--durations|--diff|--assign] [--entry E] [--stages S] [--gpu G] [--note N]
#             [P=v ...] -- prints the notice of the full benchmark (published results, command,
# duration) and how the calling invocation differs from it; the notice is silent once
# WM_FULLRUN_NOTICE=shown.
# The notice repeats docs/benchmark.md, "Replicating the published results" -- keep the two in step.
set -u

# The other modes (make pipeline uses --brief, --durations and --assign):
#   --brief      the lines on the parts and the differences alone
#   --durations  the paragraph on the duration alone
#   --diff       the differences, tab-separated: part<TAB>name, param<TAB>P<TAB>value<TAB>full
#                (full: unset for a cap), absent<TAB>label<TAB>add, note<TAB>text
#   --assign     `set P v` / `unset P` lines that turn the invocation into the full benchmark,
#                over every part: LEVIN, FOLD, THREADS, GPU, GPU_STAGES and the run
#                parameters (never DATA)
MODE=notice; ENTRY="this command"; STAGES=""; GPU=""; NOTES=()
declare -A ARG=()
while [ $# -gt 0 ]; do
    case "$1" in
        --brief|--durations|--diff|--assign) MODE=${1#--} ;;
        --entry)  ENTRY="${2:-}"; shift ;;
        --stages) STAGES="${2:-}"; shift ;;
        --gpu)    GPU="${2:-}"; shift ;;
        --note)   NOTES+=("${2:-}"); shift ;;
        *=*)      ARG[${1%%=*}]="${1#*=}" ;;
    esac
    shift
done
[ $MODE = notice ] && [ "${WM_FULLRUN_NOTICE:-}" = shown ] && exit 0
cd "$(dirname "$0")/../.." || exit 0
RB=benchmark/run_benchmark.sh

text_commands() {
    cat <<'EOF'
== the full benchmark (the published results)
  The published results come from the full benchmark on exclusive nodes: its CPU part
  once on an Intel Xeon Platinum 8358 and once on an AMD EPYC 9454, its GPU part on an
  NVIDIA A100 80GB PCIe.  The first four commands put the corpus, the external
  suppliers and the KAUST port in place; the last runs all of it on one machine:

      make corpus DOWNLOAD=yes      (the corpus from Zenodo, 5.19 GB; make corpus first)
      make externals
      make kaust                    (the copyright notice of CuBesselK: read it first)
      make kaust DOWNLOAD=yes       (if you decide to: downloads the files, ports them)
      make pipeline GPU=yes LEVIN=yes FOLD=yes THREADS=yes LADDER_SET=large SUPPLIERS=all \
                    SUP_LADDER_BUDGET=1500

  On a terminal, make pipeline asks for all of this at its start (the full settings,
  the externals, KAUST and the corpus) and then runs unattended.

  The port is deleted at the end of make pipeline, so run make kaust DOWNLOAD=yes
  again before the next make pipeline.  The downloaded files are kept, so the
  repeated command only generates the port again.

  GPU=no runs the CPU part alone, on a machine without an NVIDIA GPU.  The published
  run ran the parts as separate batch jobs.
EOF
}
text_durations() {
    cat <<'EOF'
  The full run takes about 18 hours on one machine with an A100 GPU, most of it the
  CPU ladders.  The ladders time every rung up to 2^21 points on its own and the
  rungs from 2^22 up in one cumulative pass per repetition (LADDER_PASS_FROM,
  default 2^22); LADDER_PASS_FROM=off times every rung on its own and about doubles
  the ladder time.  Durations of the published run, as the wall time of each batch
  job (build included):
      ladders (up to 2^25 points)   12.2 h on the AMD EPYC 9454, 14.3 h on the Intel
                                    Xeon 8358 (benchmark/tools/ladder_estimate.sh
                                    estimates 11.8 h and 14.5 h)
      accuracy and suppliers        2.0 h (AMD), 2.3 h (Intel); the KAUST rows 1.7 h
                                    and 1.8 h of it
      Levin A/B                     10 min (AMD), 13 min (Intel)
      thread scaling                14 min (AMD), 15 min (Intel)
      prefactor fold                5 min (AMD), 6 min (Intel), a separate run
      GPU stages (A100)             46 min (core, Levin A/B and thread settings)
EOF
}
if [ $MODE = durations ]; then text_durations; exit 0; fi
if [ $MODE = notice ]; then echo; text_commands; echo; text_durations; echo; fi

# The parts this invocation runs.
has() { case ",$1," in *",$2,"*) return 0 ;; esac; return 1; }
part_name() {
    case "$1" in
        rest) echo "accuracy and suppliers" ;;   ladder) echo "ladders" ;;
        levin) echo "Levin A/B" ;;               threads) echo "thread scaling" ;;
        gpu-core) echo "GPU core stages" ;;      gpu-levin) echo "GPU Levin A/B" ;;
        gpu-threads) echo "GPU thread settings" ;;
        fold) echo "prefactor fold" ;;
        record) echo "the run record alone (no CPU measurement)" ;;
        gpu-call-large) echo "GPU whole calls at 2^25, every device kernel (not in the published run)" ;;
    esac
}
RUNS=""; MISSES=""; MISS=()
for p in rest ladder levin fold threads gpu-core gpu-levin gpu-threads; do
    case "$p" in
        gpu-*) on=no; { has "$GPU" "${p#gpu-}" || has "$GPU" all; } && on=yes ;;
        *)     on=no; has "$STAGES" "$p" && on=yes ;;
    esac
    if [ $on = yes ]; then RUNS="$RUNS,$p"; else MISSES="$MISSES, $(part_name "$p")"; MISS+=("$p"); fi
done
# Parts outside the published run: named when they run, never listed as missing.
has "$STAGES" record && RUNS="$RUNS,record"
has "$GPU" call-large && RUNS="$RUNS,gpu-call-large"
runs() { has "$RUNS" "$1"; }
cpu_runs() { runs rest || runs ladder || runs levin || runs threads || runs fold; }
names=""; for p in $(echo "$RUNS" | tr ',' ' '); do names="$names, $(part_name "$p")"; done

# The settings that differ from the full run.
# rb_default <parameter>: run_benchmark.sh's default, as its default_of() reads it;
# evaluated here, where the parameters before it hold their effective values.
rb_default() {
    local d
    d=$(grep -o -m1 "\${$1:-[^}]*}" "$RB" | head -1)
    d=${d#\$\{$1:-}; d=${d%\}}
    eval "printf '%s' \"$d\"" 2>/dev/null
}
# ladder_set_laws <set>: the default LADDER_LAWS of a LADDER_SET (run_benchmark.sh's case)
ladder_set_laws() {
    sed -n "s/^ *$1) _LADDER_DEFAULT=\"\(.*\)\" ;;.*/\1/p" "$RB" | head -1
}
# ladder_top: the largest ladder_top_n of benchmark/tools/ladder_estimate.sh over the effective
# LADDER_LAWS, rounded down to a power of two as the rungs are; 0 when none is found.
ladder_top() {
    local t=0 n r l
    for l in $(printf '%s' "${LADDER_LAWS:-}" | tr ',' ' '); do
        n=$(LADDER_LAWS=$l LADDER_MAX=${LADDER_MAX:-} DATA=${DATA:-} \
            bash benchmark/tools/ladder_estimate.sh 2>/dev/null | sed -n 's/^ladder_top_n *//p')
        case $n in ''|*[!0-9]*) continue ;; esac
        [ "$n" -ge 256 ] || continue
        r=256; while [ $((r * 2)) -le "$n" ]; do r=$((r * 2)); done
        [ "$r" -gt "$t" ] && t=$r
    done
    echo "$t"
}
# The full run's values of the run parameters it sets, and the caps, which it leaves
# unset.
declare -A FULL=([REPS]=65 [LADDER_SET]=large [LADDER_PASS_FROM]=4194304 [SUPPLIERS]=all
                 [SUP_LADDER]=ship [SUP_LADDER_BUDGET]=1500)
CAPS=" TIME_MAX_POINTS SUP_MAX_POINTS LADDER_MAX THREADS_MAX "
is_cap() { [ "${CAPS/ $1 /}" != "$CAPS" ]; }
# <parameter> <the parts it belongs to>, in run_benchmark.sh's RUN_PARAMS order
PARAMS='FAMILIES any
BACKENDS cpu
REPS any
DATA any
LADDER_SET ladder gpu-core
LADDER_LAWS ladder gpu-core
LADDER_MAX ladder gpu-core
LADDER_PASS_FROM ladder
SUPPLIERS rest
SUP_MAX_POINTS rest
TIME_MAX_POINTS any
SUP_LADDER ladder
SUP_LADDER_BUDGET ladder
THREADS_LAW threads
THREADS_N threads
THREADS_REPS threads
THREADS_ASSEMBLY threads
THREADS_PIN_MODES threads
THREADS_SUPPLIERS threads
THREADS_WS_N threads
THREADS_WS_THREADS threads
THREADS_WS_BACKENDS threads
THREADS_MAX threads'
# The differences, one record each: P, its value, the full run's value and a note.
D_P=(); D_V=(); D_F=(); D_N=()
diff_add() { D_P+=("$1"); D_V+=("$2"); D_F+=("$3"); D_N+=("${4:-}"); }
declare -A GV=()
while read -r P parts; do
    [ -n "$P" ] || continue
    if [ -n "${ARG[$P]+x}" ]; then v="${ARG[$P]}"; else v="${!P:-}"; fi
    given=no; [ -n "$v" ] && given=yes && GV[$P]=$v
    dfull=""
    if [ "$P" = LADDER_LAWS ]; then
        # the default is the law pair of the effective LADDER_SET (run_benchmark.sh's
        # case), the full run's that of the full run's LADDER_SET
        d=$(ladder_set_laws "${LADDER_SET:-}")
        dfull=$(ladder_set_laws "${FULL[LADDER_SET]}")
    else
        d=$(rb_default "$P")
    fi
    [ $given = yes ] || v="$d"
    printf -v "$P" '%s' "$v"
    [ $MODE = assign ] && continue
    on=no
    for q in $parts; do
        case "$q" in any) [ -n "${RUNS#,}" ] && on=yes ;; cpu) cpu_runs && on=yes ;; *) runs "$q" && on=yes ;; esac
    done
    [ $on = yes ] || continue
    if [ -n "${FULL[$P]+x}" ]; then
        [ "$v" = "${FULL[$P]}" ] && continue
        # LADDER_PASS_FROM acts only on the rungs N >= its value
        note=""
        if [ "$P" = LADDER_PASS_FROM ] && [ $MODE != diff ]; then
            case "$v" in
                ''|*[!0-9]*) ;;
                *) top=$(ladder_top)
                   [ "$top" -gt 0 ] && [ "$top" -lt "$v" ] 2>/dev/null &&
                       note=" -- no effect: the top rung is $top" ;;
            esac
        fi
        diff_add "$P" "$v" "${FULL[$P]}" "$note"
    elif is_cap "$P"; then
        [ $given = no ] || diff_add "$P" "$v" unset
    else
        [ $given = no ] || [ "$v" = "$d" ] || [ "$v" = "${dfull:-$d}" ] ||
            diff_add "$P" "$v" "${dfull:-$d}"
    fi
done <<< "$PARAMS"

# --assign: the switches from the parts, then each run parameter against its value in the
# full run (FULL's, unset for a cap, else the default) and the default it takes there when
# unset (run_benchmark.sh's, with the parameters before it at their full values).
if [ $MODE = assign ]; then
    runs levin   || echo "set LEVIN yes"
    runs fold    || echo "set FOLD yes"
    runs threads || echo "set THREADS yes"
    { runs gpu-core || runs gpu-levin || runs gpu-threads; } || echo "set GPU yes"
    # GPU_STAGES replaces the stages LEVIN and THREADS select; the full run leaves it unset
    gs=${ARG[GPU_STAGES]-${GPU_STAGES:-}}
    if [ -n "$gs" ] && { has "$gs" call-large ||
            ! { has "$gs" all || { has "$gs" core && has "$gs" levin && has "$gs" threads; }; }; }; then
        echo "unset GPU_STAGES"
    fi
    while IFS='|' read -r P fv fd; do
        [ "$P" = DATA ] && continue
        if [ -n "${FULL[$P]+x}" ]; then
            [ "${GV[$P]-$fd}" = "$fv" ] || echo "set $P $fv"
        elif is_cap "$P"; then
            [ -z "${GV[$P]+x}" ] || echo "unset $P"
        elif [ -n "${GV[$P]+x}" ] && [ "${GV[$P]}" != "$fv" ]; then
            echo "unset $P"
        fi
    done < <(
        while read -r P parts; do
            [ -n "$P" ] || continue
            if [ "$P" = LADDER_LAWS ]; then fd=$(ladder_set_laws "${FULL[LADDER_SET]}")
            else fd=$(rb_default "$P"); fi
            if [ -n "${FULL[$P]+x}" ]; then fv=${FULL[$P]}
            elif is_cap "$P"; then fv=""
            else fv=$fd; fi
            printf -v "$P" '%s' "$fv"
            printf '%s|%s|%s\n' "$P" "$fv" "$fd"
        done <<< "$PARAMS")
    exit 0
fi

# The external software the running parts would use and this machine lacks; the supplier
# ladder and the thread rows run the redistributable suppliers, as in the full run.
ABS_L=(); ABS_T=()
mk=""
{ runs rest || runs ladder || runs threads; } && mk="$mk cxx boost-cpu gsl"
runs rest && mk="$mk kaust"
{ runs gpu-core || runs gpu-levin || runs gpu-threads || runs gpu-call-large; } && mk="$mk nvcc gpu"
{ runs gpu-core || runs gpu-call-large; } && mk="$mk boost-gpu kaust-dev"
if [ -n "$mk" ]; then
    while IFS=$'\t' read -r ml mt; do
        [ -n "$ml" ] && { ABS_L+=("$ml"); ABS_T+=("$mt"); }
    done <<< "$(bash benchmark/tools/missing.sh lines $mk)"
fi

if [ $MODE = diff ]; then
    for p in ${MISS[@]+"${MISS[@]}"}; do printf 'part\t%s\n' "$(part_name "$p")"; done
    for i in ${D_P[@]+"${!D_P[@]}"}; do printf 'param\t%s\t%s\t%s\n' "${D_P[$i]}" "${D_V[$i]}" "${D_F[$i]}"; done
    for i in ${ABS_L[@]+"${!ABS_L[@]}"}; do printf 'absent\t%s\t%s\n' "${ABS_L[$i]}" "${ABS_T[$i]}"; done
    for n in ${NOTES[@]+"${NOTES[@]}"}; do printf 'note\t%s\n' "$n"; done
    exit 0
fi

echo "  this run ($ENTRY): ${names#, }"
[ -n "$MISSES" ] && echo "  not in this run: ${MISSES#, }"
DIFF=()
for i in ${D_P[@]+"${!D_P[@]}"}; do DIFF+=("${D_P[$i]}=${D_V[$i]} (full: ${D_F[$i]})${D_N[$i]}"); done
for i in ${ABS_L[@]+"${!ABS_L[@]}"}; do DIFF+=("${ABS_L[$i]} absent (full: present) -- ${ABS_T[$i]}"); done
for n in ${NOTES[@]+"${NOTES[@]}"}; do DIFF+=("$n"); done
if [ ${#DIFF[@]} -eq 0 ]; then
    echo "  settings: those of the full run"
else
    echo "  settings that differ from the full run:"
    for d in "${DIFF[@]}"; do echo "      $d"; done
fi
[ $MODE = brief ] || echo
exit 0
