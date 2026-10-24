#!/bin/bash
# pipeline.sh -- the whole benchmark in one command (make pipeline): builds, the selected stages,
# figures, tables and one notification at the end; on a terminal it first asks its questions (full
# settings, externals, KAUST, corpus).  Switches and run parameters: make help, docs/benchmark.md.
set -u
cd "$(dirname "$0")/.." || exit 1
. ./tools/ask.sh

# The reduced-run caps are refused here; benchmark/run_benchmark.sh accepts them.
for _cap in TIME_MAX_POINTS SUP_MAX_POINTS; do
    if [ -n "${!_cap:-}" ]; then
        echo "pipeline: $_cap=${!_cap} is set; the complete benchmark times and scores every" >&2
        echo "pipeline: point -- unset it, or use benchmark/run_benchmark.sh for a reduced run" >&2
        exit 2
    fi
done
# A STAGES other than all is refused rather than ignored (LEVIN, FOLD and THREADS add stages).
case "${STAGES:-all}" in all) ;; *)
    echo "pipeline: STAGES=$STAGES is set; make pipeline runs STAGES=all, and LEVIN=yes / FOLD=yes /" >&2
    echo "pipeline: THREADS=yes add the Levin, prefactor-fold and thread stages -- unset it, or use" >&2
    echo "pipeline: benchmark/run_benchmark.sh for one stage" >&2
    exit 2 ;;
esac

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
yes_no() {   # yes_no <name> <value> -> yes|no, or exit on anything else
    case "$(lower "$2")" in yes|y|1|on|true) echo yes ;; no|n|0|off|false|"") echo no ;;
        *) echo "pipeline: $1=$2: use yes or no (also 1/0, on/off, true/false, y/n)" >&2; exit 2 ;; esac
}
EXTERNALS=$(yes_no EXTERNALS "${EXTERNALS:-no}") || exit 2
LEVIN=$(yes_no LEVIN "${LEVIN:-no}") || exit 2
FOLD=$(yes_no FOLD "${FOLD:-no}") || exit 2
THREADS=$(yes_no THREADS "${THREADS:-no}") || exit 2
CHECK=$(yes_no CHECK "${CHECK:-no}") || exit 2
FIGURES=$(yes_no FIGURES "${FIGURES:-yes}") || exit 2
FULL=$(yes_no FULL "${FULL:-no}") || exit 2
TABLES_CHECK=$(yes_no TABLES_CHECK "${TABLES_CHECK:-no}") || exit 2
DRY=$(yes_no DRY "${DRY:-no}") || exit 2
VERBOSE=$(yes_no V "${V:-no}") || exit 2
if [ $TABLES_CHECK = yes ] && [ $FIGURES = no ]; then
    echo "pipeline: TABLES_CHECK=yes checks the tables this pipeline writes, and FIGURES=no writes none --" >&2
    echo "pipeline: set FIGURES=yes, or run make paper-tables-check on its own" >&2
    exit 2
fi
_g=$(lower "${GPU:-auto}")
case "$_g" in auto) GPU=auto ;; *)
    GPU=$(yes_no GPU "$_g" 2>/dev/null) || { echo "pipeline: GPU=$_g: use auto, yes or no" >&2; exit 2; } ;;
esac
MAKE="${MAKE:-make}"
DATA_DIR="${DATA:-data}"

# One ladder set for every stage (run_gpu.sh alone defaults to large).
export LADDER_SET="${LADDER_SET:-small}"
# The inner scripts do not notify; the pipeline notifies once at the end.
export WM_NOTIFY_SKIP=1
# Runs go to benchmark/runs/, which the figures and tables at the end read.
if [ -n "${WM_RUNS_DIR:-}" ]; then
    echo "pipeline: WM_RUNS_DIR=$WM_RUNS_DIR is ignored; the pipeline writes to benchmark/runs/" >&2
    unset WM_RUNS_DIR
fi

# GPU detection, resolving nvcc and nvidia-smi as the Makefile does.
nvcc_bin="${NVCC:-nvcc}"; smi_bin="${NVIDIA_SMI:-nvidia-smi}"
have_nvcc=no; have_gpu=no
command -v "${nvcc_bin%% *}" >/dev/null 2>&1 && have_nvcc=yes
"$smi_bin" -L >/dev/null 2>&1 && have_gpu=yes

# The parts and components that follow from the switches; derived again after a switch to
# the full settings.
derive() {
    case "$GPU" in
        auto) if [ $have_nvcc = yes ] && [ $have_gpu = yes ]; then RUN_GPU=yes; else RUN_GPU=no; fi ;;
        yes)  RUN_GPU=yes ;;
        no)   RUN_GPU=no ;;
    esac
    GPU_STAGES_PLAN="${GPU_STAGES:-core$([ $LEVIN = yes ] && echo ,levin)$([ $THREADS = yes ] && echo ,threads)}"
    CPU_PARTS="rest,ladder$([ $LEVIN = yes ] && echo ,levin)$([ $FOLD = yes ] && echo ,fold)$([ $THREADS = yes ] && echo ,threads)"
    GPU_PARTS="$([ $RUN_GPU = yes ] && echo "$GPU_STAGES_PLAN")"
    # The components to report; nvcc and the GPU also when GPU=auto finds none.
    MX_KEYS="cxx boost-cpu gsl kaust"
    if [ $RUN_GPU = yes ]; then MX_KEYS="$MX_KEYS nvcc gpu boost-gpu kaust-dev"
    elif [ "$GPU" = auto ]; then MX_KEYS="$MX_KEYS nvcc gpu"; fi
    [ $FIGURES = yes ] && MX_KEYS="$MX_KEYS python pdflatex"
    return 0
}
derive
full_run() {
    bash benchmark/tools/full_run.sh --entry "make pipeline" --stages "$CPU_PARTS" --gpu "$GPU_PARTS" "$@"
}

# CuBesselK is not distributed ("All rights reserved"); `make kaust` states the copyright
# situation.  A port that exists is used and deleted at the end (downloaded files are kept).
KAUST_HDR=references/build/kaust/kaust_derived.h
if [ -f "$KAUST_HDR" ]; then KAUST_STATE=ported; else KAUST_STATE=absent; fi
# A KAUST row runs with SUPPLIERS=all (CPU) and in the GPU core stage.
kaust_runs() {
    [ "${SUPPLIERS:-ship}" = all ] && return 0
    [ $RUN_GPU = yes ] && case ",$GPU_STAGES_PLAN," in *,core,*|*,all,*) return 0 ;; esac
    return 1
}

# The external software this machine lacks, probed before anything runs and again after the
# downloads; every step reads the probe through WM_MISSING_CACHE.
WM_MISSING_CACHE=$(bash benchmark/tools/missing.sh probe); export WM_MISSING_CACHE
trap 'rm -f "$WM_MISSING_CACHE"' EXIT
# python_line: the interpreter the generators use and its NumPy/Matplotlib (make python-info)
python_line() {
    "$MAKE" -s --no-print-directory python-info 2>/dev/null |
        sed -n 's/^  interpreter  //p; s/^  packages     //p' | paste -sd';' - | sed 's/;/; /'
}

# The answers; on a terminal the questions below set them.
SET_ANSWER=""; EXT_ANSWER=""; KAUST_ANSWER=""; CORPUS_ANSWER=""
SWITCHED=no; KAUST_GET=no; CORPUS_GET=no; CORPUS_STOP=no

show() { printf '  %-22s %s\n' "$1" "$2"; }
plan() {
    echo "== benchmark pipeline"
    show "download externals" "$EXTERNALS$([ $EXTERNALS = yes ] && echo ' (Boost, GSL, Bessels.jl, Python packages)')"
    show "builds" "library, benchmark drivers, CPU suppliers$([ $LEVIN = yes ] && echo ', Levin arms (first)')$([ $FOLD = yes ] && echo ', fold_cpu')$([ $THREADS = yes ] && echo ', threads_cpu')"
    show "make check" "$CHECK"
    show "CPU run" "yes (STAGES=all: accuracy, ladder, suppliers, supplier ladder)"
    show "Levin A/B" "$LEVIN"
    show "prefactor fold" "$FOLD"
    show "thread scaling" "$THREADS${THREADS_MAX:+ (THREADS_MAX=$THREADS_MAX)}"
    show "GPU pipeline" "$RUN_GPU  (GPU=$GPU; nvcc: $have_nvcc, GPU visible: $have_gpu)$([ $RUN_GPU = yes ] && echo "; GPU_STAGES=$GPU_STAGES_PLAN")"
    local _figs _py
    _figs="$FIGURES$([ $FIGURES = yes ] && { [ $FULL = yes ] && echo ' (every figure and table, FULL=yes)' || echo ' (the paper set; FULL=yes for every figure and table)'; })$([ $TABLES_CHECK = yes ] && echo ', paper-tables-check')"
    if [ $FIGURES = yes ]; then
        _py=$(bash benchmark/tools/missing.sh lines python)
        if [ -n "$_py" ] && [ $EXTERNALS = yes ] && [ -n "$(bash benchmark/tools/missing.sh fetchable python)" ]; then
            _figs="$_figs -- made after make externals installs NumPy and Matplotlib into .venv/"
        elif [ -n "$_py" ]; then
            _figs="$_figs -- not made without ${_py%%$'\t'*} (missing externals)"
        elif [ -n "$(bash benchmark/tools/missing.sh lines pdflatex)" ]; then
            _figs="$_figs -- the tables not made without pdflatex (missing externals)"
        fi
        show "Python" "$(python_line)"
    fi
    show "figures and tables" "$_figs"
    show "LADDER_SET" "$LADDER_SET${LADDER_LAWS:+ (LADDER_LAWS=$LADDER_LAWS)}${LADDER_MAX:+ (LADDER_MAX=$LADDER_MAX)}"
    # The ladders dominate the CPU run's time; their estimate is a model (ladder_estimate.sh).
    local _lest _cs
    _lest=$(bash benchmark/tools/ladder_estimate.sh 2>/dev/null | awk '$1=="ladder_estimate_s"{print $2}')
    [ -n "$_lest" ] && show "ladder estimate" "$(awk -v s="$_lest" 'BEGIN{
        if (s < 60) printf "under a minute"; else if (s < 7200) printf "about %d min", (s + 30) / 60;
        else printf "about %.1f h", s / 3600 }') for the CPU ladders (a model of the published machines: benchmark/tools/ladder_estimate.sh)"
    if ! _cs=$(DATA="$DATA_DIR" sh tools/corpus.sh missing); then
        if [ $CORPUS_GET = yes ]; then _cs="$_cs -- downloaded at the start of the run"
        elif ask_tty && [ $DRY = no ]; then _cs="$_cs -- asked for before anything runs"
        else _cs="$_cs -- the pipeline stops before anything runs (make corpus DOWNLOAD=yes)"; fi
    fi
    show "corpus" "$DATA_DIR/: $_cs"
    if [ $KAUST_STATE = ported ]; then
        show "KAUST" "ported (the generated port is deleted at the end; the downloaded files are kept)"
    elif [ $KAUST_GET = yes ]; then
        show "KAUST" "downloaded and ported first (make kaust DOWNLOAD=yes); the port is deleted at the end"
    fi
    local _mxl _mxh _ml _mt
    _mxl=$(bash benchmark/tools/missing.sh lines $MX_KEYS)
    if [ -z "$_mxl" ]; then
        show "missing externals" "none"
    else
        _mxh="missing externals"
        while IFS=$'\t' read -r _ml _mt; do
            show "$_mxh" "$(printf '%-22s %s' "$_ml" "$_mt")"; _mxh=""
        done <<< "$_mxl"
        [ $EXTERNALS = yes ] && show "" "(make externals runs first; the summary says what is still missing)"
    fi
    local _given="" _p
    for _p in BACKENDS FAMILIES REPS SUPPLIERS SUP_LADDER SUP_LADDER_BUDGET LADDER_PASS_FROM \
              THREADS_N THREADS_REPS THREADS_LAW THREADS_PIN_MODES THREADS_SUPPLIERS NOTIFY; do
        [ -n "${!_p:-}" ] && _given="$_given $_p=${!_p}"
    done
    show "run parameters" "${_given:- (defaults)}"
    if [ "$GPU" = yes ] && { [ $have_nvcc = no ] || [ $have_gpu = no ]; }; then
        echo "pipeline: GPU=yes, but nvcc: $have_nvcc, GPU visible: $have_gpu -- the GPU pipeline will fail" >&2
    fi
    return 0
}

# The questions, asked in this order on a terminal; `list` prints them without asking
# (DRY=yes).
gpu_possible() { [ $have_nvcc = yes ] && [ $have_gpu = yes ]; }
# full_assign: full_run.sh --assign, without the GPU settings where no GPU part can run.
full_assign() {
    if gpu_possible; then full_run --assign
    else full_run --assign | grep -Ev '^(set|unset) GPU(_STAGES)?( |$)'; fi
}
q_settings() {
    local assign cmd name value
    assign=$(full_assign)
    if [ -z "$assign" ]; then
        SET_ANSWER="those of the full benchmark$(gpu_possible || echo '; the GPU part cannot run here')"
        [ "$1" = list ] && show "[1/4] settings" "not asked: $SET_ANSWER"
        return 0
    fi
    if [ "$1" = list ]; then
        show "[1/4] settings" "Switch to the full benchmark settings? (sets $(echo "$assign" |
            awk '$1=="set"{s=s" "$2"="$3} $1=="unset"{s=s" "$2"=(unset)"} END{print substr(s,2)}'))"
        return 0
    fi
    echo; echo "== [1/4] the settings of this run differ from the full benchmark"
    full_run --brief
    echo
    full_run --durations
    gpu_possible || echo "  The GPU part cannot run here (nvcc: $have_nvcc, GPU visible: $have_gpu); the switch sets the full CPU part."
    if ! ask_yn "Switch to the full benchmark settings?"; then SET_ANSWER="kept as given"; return 0; fi
    while read -r cmd name value; do
        case "$name" in ''|DATA|*[!A-Z0-9_]*) continue ;; esac
        case "$cmd" in
            set)   printf -v "$name" '%s' "$value"; export "$name" ;;
            unset) unset "$name" ;;
        esac
    done <<< "$assign"
    LEVIN=$(yes_no LEVIN "${LEVIN:-no}") || exit 2
    FOLD=$(yes_no FOLD "${FOLD:-no}") || exit 2
    THREADS=$(yes_no THREADS "${THREADS:-no}") || exit 2
    GPU=$(lower "${GPU:-auto}"); case "$GPU" in auto) ;; *) GPU=$(yes_no GPU "$GPU") || exit 2 ;; esac
    export LADDER_SET="${LADDER_SET:-small}"
    derive
    SWITCHED=yes
    SET_ANSWER="switched to the full benchmark$(gpu_possible || echo '; the GPU part cannot run here')"
}
q_externals() {
    local keys fetch sys k l
    if [ $EXTERNALS = yes ]; then
        EXT_ANSWER="make externals runs first (EXTERNALS=yes)"
        [ "$1" = list ] && show "[2/4] externals" "not asked: $EXT_ANSWER"
        return 0
    fi
    # KAUST has a question of its own.
    keys=$(echo " $MX_KEYS bessels-upstream " | sed 's/ kaust-dev / /; s/ kaust / /')
    fetch=$(bash benchmark/tools/missing.sh fetchable $keys)
    if [ -z "$fetch" ]; then
        EXT_ANSWER="nothing to download"
        [ "$1" = list ] && show "[2/4] externals" "not asked: $EXT_ANSWER"
        return 0
    fi
    if [ -n "$(bash benchmark/tools/missing.sh lines download)" ]; then
        EXT_ANSWER="not downloadable here: neither curl nor wget"
        [ "$1" = list ] && show "[2/4] externals" "not asked: $EXT_ANSWER"
        return 0
    fi
    if [ "$1" = list ]; then
        show "[2/4] externals" "Download them with make externals? ($(echo "$fetch" | cut -d'|' -f2 | paste -sd, - | sed 's/,/, /g'))"
        return 0
    fi
    echo; echo "== [2/4] externals"
    echo "  make externals downloads, without root (into references/ and .venv/):"
    echo "$fetch" | awk -F'|' '{printf "      %-34s %s\n", $2, $3}'
    sys=""
    for k in $keys bzip2 sha256; do
        case "$fetch" in "$k|"*|*$'\n'"$k|"*) continue ;; esac
        l=$(bash benchmark/tools/missing.sh lines "$k")
        [ -n "$l" ] && sys="$sys$l"$'\n'
    done
    if [ -n "$sys" ]; then
        echo "  Also absent, not downloaded by make externals:"
        printf '%s' "$sys" | awk -F'\t' '{printf "      %-34s %s\n", $1, $2}'
    fi
    if ask_yn "Download them with make externals at the start of the run?"; then
        EXTERNALS=yes; EXT_ANSWER="make externals runs first"
    else
        EXT_ANSWER="not downloaded; the rows that need them are left out"
    fi
}
q_kaust() {
    if [ $KAUST_STATE = ported ]; then
        KAUST_ANSWER="ported (the port is deleted at the end)"
    elif ! kaust_runs; then
        KAUST_ANSWER="no stage runs a KAUST row (SUPPLIERS=all or the GPU core stage)"
        if [ "$1" = list ] && full_assign | grep -q '^set SUPPLIERS all$'; then
            show "[3/4] KAUST" "asked after a switch to the full settings (SUPPLIERS=all): the copyright notice of CuBesselK, then: Download the two CuBesselK files and generate the port?"
            return 0
        fi
    elif [ "$1" = list ]; then
        show "[3/4] KAUST" "the copyright notice of CuBesselK (make kaust), then: Download the two CuBesselK files and generate the port?"
        return 0
    else
        echo; echo "== [3/4] KAUST"
        bash references/kaust_logbesselk/kaust.sh ask
        case $? in
            0) KAUST_GET=yes; KAUST_ANSWER="downloaded and ported first; the port is deleted at the end" ;;
            1) KAUST_ANSWER="not downloaded; the KAUST rows are left out" ;;
            *) KAUST_ANSWER="cannot be downloaded here (above); the KAUST rows are left out" ;;
        esac
        return 0
    fi
    [ "$1" = list ] && show "[3/4] KAUST" "not asked: $KAUST_ANSWER"
    return 0
}
q_corpus() {
    local cs
    if cs=$(DATA="$DATA_DIR" sh tools/corpus.sh missing); then
        CORPUS_ANSWER="complete"
        [ "$1" = list ] && show "[4/4] corpus" "not asked: $CORPUS_ANSWER"
        return 0
    fi
    if [ "$1" = list ]; then
        show "[4/4] corpus" "Download the missing files from Zenodo? ($cs; a no stops the pipeline)"
        return 0
    fi
    echo; echo "== [4/4] corpus"
    if DATA="$DATA_DIR" sh tools/corpus.sh ask "make pipeline"; then
        CORPUS_GET=yes; CORPUS_ANSWER="downloaded at the start of the run"
    else
        CORPUS_STOP=yes; CORPUS_ANSWER="not downloaded: $cs -- the pipeline stops here"
    fi
}

plan
if [ $DRY = yes ]; then
    full_run
    echo "== the questions of make pipeline on a terminal (DRY=yes: none asked, nothing run)"
    q_settings list; q_externals list; q_kaust list; q_corpus list
    exit 0
fi
if ask_tty; then
    q_settings ask; q_externals ask; q_kaust ask; q_corpus ask
    echo; echo "== the answers"
    show "settings" "$SET_ANSWER"
    show "externals" "$EXT_ANSWER"
    show "KAUST" "$KAUST_ANSWER"
    show "corpus" "$CORPUS_ANSWER"
    if [ $CORPUS_STOP = yes ]; then
        echo "pipeline: the corpus is incomplete; nothing was run (make corpus DOWNLOAD=yes downloads it)" >&2
        exit 2
    fi
    # test hook (WM_PIPELINE_STOP=answers|downloads|builds): stop after that step
    [ "${WM_PIPELINE_STOP:-}" = answers ] && exit 0
    FR_SHOW=brief
else
    FR_SHOW=notice
fi
# fr_show: on a terminal the plan after a switch and the differences from the full run, else the
# full-run notice, printed once; WM_FULLRUN_NOTICE then silences the notice in the stages.  Both
# read the probe, so a run that downloads externals or KAUST prints them after the downloads
# (fr_show after), where the differences need a heading of their own.
_fr_notice=${WM_FULLRUN_NOTICE:-}
fr_show() {
    if [ $FR_SHOW = brief ]; then
        if [ $SWITCHED = yes ]; then echo; plan
        elif [ "${1:-}" = after ]; then echo; echo "== this run against the full benchmark"; fi
        full_run --brief
    else
        WM_FULLRUN_NOTICE=$_fr_notice full_run
    fi
}
# Now when nothing is downloaded, or when an incomplete corpus stops the run at step 0 (below).
if { [ $EXTERNALS = no ] && [ $KAUST_GET = no ]; } ||
   { [ $CORPUS_GET = no ] && ! DATA="$DATA_DIR" sh tools/corpus.sh missing >/dev/null; }; then
    fr_show; FR_SHOW=shown
fi
# From here on nothing asks.
export WM_FULLRUN_NOTICE=shown WM_UNATTENDED=1
exec </dev/null

# Step 0: without a download the corpus is checked before anything runs.
if [ $CORPUS_GET = no ]; then DATA="$DATA_DIR" sh tools/corpus.sh need "make pipeline" || exit 2; fi

declare -a SUMMARY=()
failed=0
warned=0
note() { SUMMARY+=("$(printf '%-22s %s' "$1" "$2")"); }
runs_now() { ls -1 benchmark/runs 2>/dev/null | grep -v '^latest$' | sort; }
# stage <name> <command...>: run a stage and summarise the run(s) it recorded
stage() {
    local name=$1; shift
    local before after new st rc
    before=$(runs_now)
    echo; echo "== $name"
    "$@"; rc=$?
    after=$(runs_now)
    new=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tr '\n' ' ')
    st=""
    for _r in $new; do
        st="$st $(sed -n 's/^STATUS *//p' "benchmark/runs/$_r/STATUS" 2>/dev/null | head -1) ($_r)"
        if grep -q '^warning ' "benchmark/runs/$_r/STATUS" 2>/dev/null; then
            st="$st, with warnings (STATUS)"; warned=1
        fi
    done
    [ $rc -eq 0 ] || failed=1
    note "$name" "rc=$rc${st:+ --$st}"
}
# build <name> <hard|soft> <make args...>: the commands go to a log (V=yes prints them), the
# end of which is shown when the build fails.
build() {
    local name=$1 kind=$2 log rc; shift 2
    echo; echo "== build: $name"
    if [ $VERBOSE = yes ]; then
        "$MAKE" --no-print-directory "$@"; rc=$?
    else
        # levin-ab-build deletes build/, so its log is in build-ab/
        case "$1" in levin-ab-build) log=build-ab/pipeline.levin-arms.log ;;
                     *) log=build/pipeline.$(echo "$name" | tr 'A-Z ' 'a-z-').log ;; esac
        mkdir -p "${log%/*}"
        "$MAKE" --no-print-directory "$@" >"$log" 2>&1; rc=$?
        if [ $rc -eq 0 ]; then
            grep -Ev '^  [A-Z]{2,5} +[^ ]+$' "$log"
        else
            echo "  FAILED; the end of $log:"; tail -25 "$log"
        fi
    fi
    if [ $rc -eq 0 ]; then note "build $name" "ok"; return 0; fi
    note "build $name" "FAILED"
    if [ "$kind" = hard ]; then
        failed=1; finish; exit 1
    fi
    return 1
}
# The generated KAUST port is deleted once: in finish(), so that the summary shows it, or by
# the EXIT trap when the pipeline is interrupted.
_kaust_done=0
kaust_cleanup() {
    [ $_kaust_done = 1 ] && return 0
    _kaust_done=1
    if [ -f "$KAUST_HDR" ] || [ -f references/build/kaust_port.o ]; then
        echo; echo "== KAUST: deleting the generated port"
        if bash references/kaust_logbesselk/kaust.sh clean; then
            note "KAUST" "generated port deleted (downloaded files kept)"
        else
            note "KAUST" "FAILED to delete the generated port -- make kaust CLEAN=yes"; failed=1
        fi
    fi
}
trap 'kaust_cleanup; rm -f "$WM_MISSING_CACHE"' EXIT
trap 'exit 130' INT TERM
finish() {
    kaust_cleanup
    # what the run went without, each with the command that adds it
    local _ml _mt
    while IFS=$'\t' read -r _ml _mt; do
        [ -n "$_ml" ] && note "missing" "$_ml -- $_mt"
    done <<< "$(bash benchmark/tools/missing.sh lines $MX_KEYS)"
    echo; echo "== pipeline summary ($(printf '%d:%02d:%02d' $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60))))"
    printf '  %s\n' "${SUMMARY[@]}"
    local verdict=complete; [ $warned -eq 0 ] || verdict="complete, with warnings"
    [ $failed -eq 0 ] || verdict="finished with failures"
    WM_NOTIFY_SKIP= bash benchmark/tools/notify.sh "make pipeline" \
        "$verdict after $(printf '%d:%02d:%02d' $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60))); see the summary above and benchmark/runs/"
}

# 1. Downloads: externals, KAUST, corpus.  A failed external or KAUST is reported and the
# stages that need it skip; an incomplete corpus stops the pipeline.
if [ $EXTERNALS = yes ]; then
    echo; echo "== download externals"
    _xs=$(mktemp) || exit 1
    EXTERNALS_STATUS="$_xs" "$MAKE" --no-print-directory externals
    if [ -s "$_xs" ]; then
        while IFS= read -r _l; do note "externals" "WARNING $(echo $_l)"; warned=1; done < "$_xs"
    else note "externals" "ok"; fi
    rm -f "$_xs"
fi
if [ $KAUST_GET = yes ]; then
    echo; echo "== KAUST: download and port"
    if "$MAKE" --no-print-directory kaust DOWNLOAD=yes; then
        KAUST_STATE=ported; note "KAUST" "downloaded and ported"
    else
        note "KAUST" "WARNING download or port failed; the KAUST rows are left out"; warned=1
    fi
fi
if [ $CORPUS_GET = yes ]; then
    echo; echo "== corpus: download"
    # without the notice: the corpus question shows it
    DATA="$DATA_DIR" sh tools/corpus.sh download --no-notice
    if DATA="$DATA_DIR" sh tools/corpus.sh need "make pipeline"; then
        note "corpus" "downloaded"
    else
        note "corpus" "incomplete after the download -- pipeline stopped"; failed=1; finish; exit 2
    fi
fi
if [ $EXTERNALS = yes ] || [ $KAUST_GET = yes ]; then
    # probe again for the stages and the summary
    _mxn=$(bash benchmark/tools/missing.sh probe) && mv -f "$_mxn" "$WM_MISSING_CACHE"
    [ $FIGURES = yes ] && echo "pipeline: Python for the figures and tables: $(python_line)"
fi
[ $FR_SHOW = shown ] || fr_show after
[ "${WM_PIPELINE_STOP:-}" = downloads ] && exit 0

# 2. Builds.  The Levin arms first, since levin-ab-build cleans build/.
[ $LEVIN = yes ] && build "Levin arms" hard levin-ab-build
build "library" hard all
build "benchmark drivers" hard benchmark-build
build "CPU suppliers" soft build/suppliers || {
    warned=1
    echo "pipeline: build/suppliers did not build; the CPU supplier stages are skipped (make requirements)"; }
if [ $FOLD = yes ]; then
    build "fold_cpu" soft build/fold_cpu || FOLD=failed
fi
if [ $THREADS = yes ]; then
    build "threads_cpu" soft build/threads_cpu || THREADS=failed
fi
[ "${WM_PIPELINE_STOP:-}" = builds ] && { finish; exit $failed; }

# 3. make check
if [ $CHECK = yes ]; then
    echo; echo "== make check"
    if "$MAKE" --no-print-directory check; then note "make check" "passed"
    else note "make check" "FAILED -- pipeline stopped"; failed=1; finish; exit 1; fi
fi

# 4. Runs.
stage "CPU run" env STAGES=all bash benchmark/run_benchmark.sh
[ $LEVIN = yes ] && stage "Levin A/B" env STAGES=levin bash benchmark/run_benchmark.sh
if [ "$FOLD" = yes ]; then
    stage "prefactor fold" env STAGES=fold bash benchmark/run_benchmark.sh
elif [ "$FOLD" = failed ]; then
    note "prefactor fold" "not run: build/fold_cpu did not build"; failed=1
fi
if [ "$THREADS" = yes ]; then
    stage "thread scaling" env STAGES=threads bash benchmark/run_benchmark.sh
elif [ "$THREADS" = failed ]; then
    note "thread scaling" "not run: build/threads_cpu did not build"; failed=1
fi
[ $RUN_GPU = yes ] && stage "GPU pipeline" env GPU_STAGES="$GPU_STAGES_PLAN" NVCC="$nvcc_bin" \
    bash benchmark/run_gpu.sh

# 5. Figures and tables from every run in benchmark/runs/ (the tables need pdflatex).  The
# tables check runs only on tables this pipeline wrote, not on an earlier generation.
if [ $FIGURES = yes ]; then
    echo; echo "== figures and tables"
    _py=$(bash benchmark/tools/missing.sh lines python)
    _tx=$(bash benchmark/tools/missing.sh lines pdflatex)
    _tables=no
    if [ -n "$_py" ]; then
        note "figures" "not made -- ${_py%%$'\t'*} missing: ${_py#*$'\t'}"; failed=1
        note "tables" "not made -- ${_py%%$'\t'*} missing (as for the figures)"
    else
        if "$MAKE" --no-print-directory paper-figures-final FULL=$FULL; then note "figures" "benchmark/plots/paper-final/"
        else note "figures" "FAILED"; failed=1; fi
        if [ -n "$_tx" ]; then
            note "tables" "not made -- ${_tx%%$'\t'*} missing: ${_tx#*$'\t'}"; failed=1
        elif "$MAKE" --no-print-directory paper-tables FULL=$FULL; then note "tables" "benchmark/tables/paper/"; _tables=yes
        else note "tables" "FAILED"; failed=1; fi
    fi
    if [ $TABLES_CHECK = yes ]; then
        if [ -z "$_py" ] && [ -n "$_tx" ]; then note "tables check" "not run -- pdflatex missing: ${_tx#*$'\t'}"; failed=1
        elif [ $_tables = no ]; then note "tables check" "not run -- the tables were not made"; failed=1
        elif "$MAKE" --no-print-directory paper-tables-check; then note "tables check" "passed"
        else note "tables check" "FAILED"; failed=1; fi
    fi
fi

finish
exit $failed
