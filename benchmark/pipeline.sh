#!/bin/bash
# pipeline.sh [reproduce] -- the whole benchmark in one command (make pipeline, make reproduce): builds,
# the selected stages, figures, tables, one notification at the end; on a terminal it first asks its
# questions (full settings, externals, KAUST, corpus).  Switches: make help, docs/benchmark.md.
set -u
cd "$(dirname "$0")/.." || exit 1
. ./tools/ask.sh

# make reproduce passes `reproduce`: without a terminal it needs KAUST=yes or KAUST=no.
case "${1:-}" in
    "")        ENTRY="make pipeline"; REPRODUCE=no ;;
    reproduce) ENTRY="make reproduce"; REPRODUCE=yes ;;
    *)         echo "usage: pipeline.sh [reproduce]" >&2; exit 2 ;;
esac

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
TABLES_CHECK=$(yes_no TABLES_CHECK "${TABLES_CHECK:-no}") || exit 2
DRY=$(yes_no DRY "${DRY:-no}") || exit 2
VERBOSE=$(yes_no V "${V:-no}") || exit 2
# Answers given in advance: SETTINGS=full ([1/4]), KAUST=yes|no ([3/4]; unset, it is asked),
# CORPUS=yes ([4/4]).  EXTERNALS=yes answers [2/4].
CORPUS=$(yes_no CORPUS "${CORPUS:-no}") || exit 2
KAUST_SW=""
[ -z "${KAUST:-}" ] || KAUST_SW=$(yes_no KAUST "$KAUST") || exit 2
case "$(lower "${SETTINGS:-}")" in
    "")   SETTINGS="" ;;
    full) SETTINGS=full ;;
    *)    echo "pipeline: SETTINGS=$SETTINGS: use full, or leave it unset" >&2; exit 2 ;;
esac
if [ $TABLES_CHECK = yes ] && [ $FIGURES = no ]; then
    echo "pipeline: TABLES_CHECK=yes checks the tables this pipeline writes, and FIGURES=no writes none --" >&2
    echo "pipeline: set FIGURES=yes, or run make tables-check on its own" >&2
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
export BENCH_NOTIFY_SKIP=1
# Runs go to benchmark/runs/, which the figures and tables at the end read.
if [ -n "${BENCH_RUNS_DIR:-}" ]; then
    echo "pipeline: BENCH_RUNS_DIR=$BENCH_RUNS_DIR is ignored; the pipeline writes to benchmark/runs/" >&2
    unset BENCH_RUNS_DIR
fi
# The corpus result the stages read; set below, after the builds.
unset BENCH_CORPUS_CHECK

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
    bash benchmark/tools/full_run.sh --entry "$ENTRY" --stages "$CPU_PARTS" --gpu "$GPU_PARTS" "$@"
}
# make externals under the pipeline fetches what this run uses (tools/externals.sh,
# EXTERNALS_PIPELINE): Boost 1.75.0 and GSL 2.6 unless the system has these versions, Boost
# 1.87.0 only with a GPU part, where no Boost with GPU support is found and BOOST_ROOT is not
# none, NumPy and Matplotlib.  ext_plan: what it would fetch, one `name|size` line each;
# ext_names: the names.
ext_mode() { if [ $RUN_GPU = yes ]; then echo gpu; else echo cpu; fi; }
ext_plan() {
    EXTERNALS_PIPELINE=$(ext_mode) EXTERNALS_PLAN=yes "$MAKE" -s --no-print-directory externals 2>/dev/null
}
ext_names() {
    local p; p=$(ext_plan)
    if [ -n "$p" ]; then echo "$p" | awk -F'|' '{printf "%s%s", (NR > 1 ? ", " : ""), $1} END{print ""}'
    else echo "nothing to download"; fi
}
# CuBesselK is not distributed ("All rights reserved"); `make kaust` states the copyright
# situation.  A port that exists is used (KAUST=no is then refused) and deleted at the end
# (downloaded files are kept).
KAUST_HDR=references/build/kaust/kaust_derived.h
if [ -f "$KAUST_HDR" ]; then KAUST_STATE=ported; else KAUST_STATE=absent; fi
# A KAUST row runs with SUPPLIERS=all (CPU) and in the GPU core stage.
kaust_runs() {
    [ "${SUPPLIERS:-ship}" = all ] && return 0
    [ $RUN_GPU = yes ] && case ",$GPU_STAGES_PLAN," in *,core,*|*,all,*) return 0 ;; esac
    return 1
}
# kaust_no_ported: KAUST=no while the port exists and a stage would run its rows (refused).
kaust_no_ported() { [ "$KAUST_SW" = no ] && [ $KAUST_STATE = ported ] && kaust_runs; }
# kaust_to_port: the port is generated at the start of the run (KAUST=yes or a yes).
kaust_to_port() {
    [ $KAUST_STATE = absent ] && { [ $KAUST_GET = yes ] || { [ "$KAUST_SW" = yes ] && kaust_runs; }; }
}

# The external software this machine lacks, probed before anything runs and again after the
# downloads; every step reads the probe through BENCH_MISSING_CACHE.
BENCH_MISSING_CACHE=$(bash benchmark/tools/missing.sh probe); export BENCH_MISSING_CACHE
trap 'rm -f "$BENCH_MISSING_CACHE"' EXIT
# GSL_OTHER: set where the GSL that gsl_flags.sh takes is not 2.6; the gsl row uses it, and it
# counts as present (it does not open [2/4]).  For a GSL from pkg-config, make externals builds
# GSL 2.6, which the row then uses, where references/gsl/2.6 is not complete (the test of
# tools/externals.sh) and curl or wget is found.
GSL_OTHER=""
_gv=$(bash benchmark/tools/gsl_flags.sh --modversion 2>/dev/null)
if [ -n "$_gv" ] && [ "$_gv" != 2.6 ]; then
    _gw=$(bash benchmark/tools/gsl_flags.sh --where 2>/dev/null)
    case "$_gw" in
        *"(pkg-config)")
            GSL_OTHER="the gsl row uses the system GSL $_gv (row gsl-$_gv)"
            { [ -f references/gsl/2.6/lib/pkgconfig/gsl.pc ] && [ -f references/gsl/2.6/lib/libgsl.a ]; } ||
                [ -n "$(bash benchmark/tools/missing.sh lines download)" ] ||
                GSL_OTHER="$GSL_OTHER; EXTERNALS=yes or make reproduce build GSL 2.6" ;;
        *)  _gw=${_gw% (*}
            GSL_OTHER="the gsl row uses GSL $_gv from references/gsl/${_gw##*/} (row gsl-$_gv)" ;;
    esac
fi
# python_line: the interpreter the generators use and its NumPy/Matplotlib (make python-info)
python_line() {
    "$MAKE" -s --no-print-directory python-info 2>/dev/null |
        sed -n 's/^  interpreter  //p; s/^  packages     //p' | paste -sd';' - | sed 's/;/; /'
}

# The answers; on a terminal the questions below set them.
SET_ANSWER=""; EXT_ANSWER=""; KAUST_ANSWER=""; CORPUS_ANSWER=""; PY_PLAN=""
# ASKED: the questions this run asked (settings externals KAUST corpus), which "== the answers" lists
ASKED=""
SWITCHED=no; KAUST_GET=no; CORPUS_GET=no; CORPUS_STOP=no

show() { printf '  %-22s %s\n' "$1" "$2"; }
plan() {
    echo "== benchmark pipeline"
    show "download externals" "$EXTERNALS$([ $EXTERNALS = yes ] && echo " ($(ext_names))")$([ $EXTERNALS = no ] && [ -n "$GSL_OTHER" ] && echo "; $GSL_OTHER")"
    show "builds" "library, benchmark drivers, CPU suppliers$([ $LEVIN = yes ] && echo ', Levin arms (first)')$([ $FOLD = yes ] && echo ', fold_cpu')$([ $THREADS = yes ] && echo ', threads_cpu')"
    show "make check" "$CHECK"
    show "CPU run" "yes (STAGES=all: accuracy, ladder, suppliers, supplier ladder)"
    show "Levin A/B" "$LEVIN"
    show "prefactor fold" "$FOLD"
    show "thread scaling" "$THREADS${THREADS_MAX:+ (THREADS_MAX=$THREADS_MAX)}"
    show "GPU pipeline" "$RUN_GPU  (GPU=$GPU; nvcc: $have_nvcc, GPU visible: $have_gpu)$([ $RUN_GPU = yes ] && echo "; GPU_STAGES=$GPU_STAGES_PLAN")"
    local _figs _py
    _figs="$FIGURES$([ $FIGURES = yes ] && echo ' (the paper set)')$([ $TABLES_CHECK = yes ] && echo ', tables-check')"
    if [ $FIGURES = yes ]; then
        _py=$(bash benchmark/tools/missing.sh lines python)
        if [ -n "$_py" ] && [ $EXTERNALS = yes ] && [ -n "$(bash benchmark/tools/missing.sh fetchable python)" ]; then
            _figs="$_figs -- made after make externals installs NumPy and Matplotlib into .venv/"
        elif [ -n "$_py" ]; then
            _figs="$_figs -- not made without ${_py%%$'\t'*} (missing externals)"
        elif [ -n "$(bash benchmark/tools/missing.sh lines pdflatex)" ]; then
            _figs="$_figs -- the tables not made without pdflatex (missing externals)"
        fi
        PY_PLAN=$(python_line)
        show "Python" "$PY_PLAN"
    fi
    show "figures and tables" "$_figs"
    show "LADDER_SET" "$LADDER_SET${LADDER_LAWS:+ (LADDER_LAWS=$LADDER_LAWS)}${LADDER_MAX:+ (LADDER_MAX=$LADDER_MAX)}"
    local _cs
    if ! _cs=$(DATA="$DATA_DIR" sh tools/corpus.sh missing); then
        if [ $CORPUS_GET = yes ]; then _cs="$_cs -- downloaded at the start of the run"
        elif ask_tty; then _cs="$_cs -- asked for before anything runs"
        else _cs="$_cs -- the pipeline stops before anything runs (CORPUS=yes or make corpus DOWNLOAD=yes)"; fi
    fi
    show "corpus" "$DATA_DIR/: $_cs"
    if kaust_no_ported; then
        show "KAUST" "KAUST=no, but the port exists: the pipeline stops before anything runs (make kaust CLEAN=yes)"
    elif [ $KAUST_STATE = ported ]; then
        show "KAUST" "ported (the generated port is deleted at the end; the downloaded files are kept)"
    elif [ $KAUST_GET = yes ]; then
        show "KAUST" "downloaded and ported first (make kaust DOWNLOAD=yes); the port is deleted at the end"
    elif kaust_runs && [ "$KAUST_SW" = yes ]; then
        show "KAUST" "KAUST=yes: after the copyright notice of CuBesselK, downloaded and ported first (make kaust DOWNLOAD=yes); the port is deleted at the end"
    elif kaust_runs && [ "$KAUST_SW" = no ]; then
        show "KAUST" "KAUST=no: not downloaded; the KAUST rows are left out"
    elif kaust_runs && ! ask_tty && [ $REPRODUCE = yes ]; then
        show "KAUST" "not asked without a terminal: the pipeline stops before anything runs (KAUST=yes or KAUST=no)"
    elif kaust_runs && ! ask_tty; then
        show "KAUST" "not asked without a terminal; the KAUST rows are left out (KAUST=yes adds them)"
    elif [ "$KAUST_SW" = yes ]; then
        show "KAUST" "KAUST=yes: no stage of these settings runs a KAUST row (SUPPLIERS=all or the GPU core stage)"
    fi
    local _mxl _mxh _ml _mt
    _mxl=$(bash benchmark/tools/missing.sh lines $MX_KEYS)
    if [ -z "$_mxl" ]; then
        show "missing externals" "none"
    else
        _mxh="missing externals"
        while IFS=$'\t' read -r _ml _mt; do
            case "$_ml" in "KAUST CuBesselK"*) kaust_to_port && _mt="downloaded and ported first (make kaust DOWNLOAD=yes)" ;; esac
            show "$_mxh" "$(printf '%-22s %s' "$_ml" "$_mt")"; _mxh=""
        done <<< "$_mxl"
        [ $EXTERNALS = yes ] && [ -n "$(bash benchmark/tools/missing.sh fetchable $MX_KEYS)" ] &&
            show "" "(make externals runs first; the summary says what is still missing)"
    fi
    local _given="" _p
    for _p in BACKENDS FAMILIES REPS SUPPLIERS SUP_LADDER SUP_LADDER_BUDGET LADDER_PASS_FROM \
              THREADS_N THREADS_REPS THREADS_LAW THREADS_PIN_MODES THREADS_SUPPLIERS NOTIFY; do
        [ -n "${!_p:-}" ] && _given="$_given $_p=${!_p}"
    done
    _given=${_given# }
    show "run parameters" "${_given:-(defaults)}"
    if [ "$GPU" = yes ] && { [ $have_nvcc = no ] || [ $have_gpu = no ]; }; then
        echo "pipeline: GPU=yes, but nvcc: $have_nvcc, GPU visible: $have_gpu -- the GPU pipeline will fail" >&2
    fi
    return 0
}

# The questions, asked in this order on a terminal; `list` prints them without asking
# (DRY=yes).
gpu_possible() { [ $have_nvcc = yes ] && [ $have_gpu = yes ]; }
# gpu_note: what the full settings leave of the GPU part under GPU=no or where no GPU part can
# run (GPU=no and GPU=yes are kept).
gpu_note() {
    if [ "$GPU" = no ]; then echo "; GPU=no is kept, so no GPU part runs"; return 0; fi
    gpu_possible && return 0
    if [ "$GPU" = yes ]; then echo "; GPU=yes is kept, and the GPU part fails here"
    else echo "; the GPU part cannot run here"; fi
}
# full_assign: full_run.sh --assign, without the GPU settings under GPU=no or where no GPU part
# can run.
full_assign() {
    if gpu_possible && [ "$GPU" != no ]; then full_run --assign
    else full_run --assign | grep -Ev '^(set|unset) GPU(_STAGES)?( |$)'; fi
}
# apply_full <assign>: the lines of full_assign applied, the switches derived again (a yes to
# [1/4], or SETTINGS=full).
apply_full() {
    local cmd name value
    while read -r cmd name value; do
        case "$name" in ''|DATA|*[!A-Z0-9_]*) continue ;; esac
        case "$cmd" in
            set)   printf -v "$name" '%s' "$value"; export "$name" ;;
            unset) unset "$name" ;;
        esac
    done <<< "$1"
    LEVIN=$(yes_no LEVIN "${LEVIN:-no}") || exit 2
    FOLD=$(yes_no FOLD "${FOLD:-no}") || exit 2
    THREADS=$(yes_no THREADS "${THREADS:-no}") || exit 2
    GPU=$(lower "${GPU:-auto}"); case "$GPU" in auto) ;; *) GPU=$(yes_no GPU "$GPU") || exit 2 ;; esac
    export LADDER_SET="${LADDER_SET:-small}"
    derive
}
q_settings() {
    local assign
    assign=$(full_assign)
    if [ -z "$assign" ]; then
        SET_ANSWER="those of the full benchmark$([ "$SETTINGS" = full ] && echo ' (SETTINGS=full)')$(gpu_note)"
        [ "$1" = list ] && show "[1/4] settings" "not asked: $SET_ANSWER"
        return 0
    fi
    if [ "$1" = list ]; then
        show "[1/4] settings" "Switch to the full benchmark settings? (sets $(echo "$assign" |
            awk '$1=="set"{s=s" "$2"="$3} $1=="unset"{s=s" "$2"=(unset)"} END{print substr(s,2)}')$([ "$GPU" = no ] && echo '; GPU=no is kept'))"
        return 0
    fi
    echo; echo "== question: the settings of this run differ from the full benchmark"
    ASKED="$ASKED settings"
    full_run --brief
    echo
    full_run --durations
    if [ "$GPU" = no ]; then echo "  GPU=no is kept: the switch sets the full CPU part."
    else gpu_possible || echo "  The GPU part cannot run here (nvcc: $have_nvcc, GPU visible: $have_gpu); the switch sets the full CPU part$([ "$GPU" = yes ] && echo '; GPU=yes is kept, and the GPU part fails')."; fi
    if ! ask_yn "Switch to the full benchmark settings?"; then SET_ANSWER="kept as given"; return 0; fi
    apply_full "$assign"
    SWITCHED=yes
    SET_ANSWER="switched to the full benchmark$(gpu_note)"
}
q_externals() {
    local keys fetch plan="" sys k l
    if [ $EXTERNALS = yes ]; then
        EXT_ANSWER="make externals runs first (EXTERNALS=yes): $(ext_names)"
        [ "$1" = list ] && show "[2/4] externals" "not asked: $EXT_ANSWER"
        return 0
    fi
    # Asked when a component this run uses is missing and make externals fetches it; the
    # question lists what a yes fetches (ext_plan).  KAUST has a question of its own.
    keys=$(echo " $MX_KEYS " | sed 's/ kaust-dev / /; s/ kaust / /')
    fetch=$(bash benchmark/tools/missing.sh fetchable $keys)
    [ -z "$fetch" ] || plan=$(ext_plan)
    if [ -z "$fetch" ] || [ -z "$plan" ]; then
        EXT_ANSWER="nothing missing to download${GSL_OTHER:+; $GSL_OTHER}"
        [ "$1" = list ] && show "[2/4] externals" "not asked: $EXT_ANSWER"
        return 0
    fi
    if [ -n "$(bash benchmark/tools/missing.sh lines download)" ]; then
        EXT_ANSWER="not downloadable here: neither curl nor wget${GSL_OTHER:+; $GSL_OTHER}"
        [ "$1" = list ] && show "[2/4] externals" "not asked: $EXT_ANSWER"
        return 0
    fi
    if [ "$1" = list ]; then
        show "[2/4] externals" "Download them with make externals? ($(echo "$plan" | awk -F'|' '{printf "%s%s", (NR > 1 ? ", " : ""), $1}'))"
        return 0
    fi
    echo; echo "== question: externals"
    ASKED="$ASKED externals"
    echo "  make externals downloads, without root (into references/ and .venv/):"
    echo "$plan" | awk -F'|' '{printf "      %-34s %s\n", $1, $2}'
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
        EXT_ANSWER="not downloaded; the rows, figures and tables that need an absent external are left out${GSL_OTHER:+; $GSL_OTHER}"
    fi
}
q_kaust() {
    if kaust_no_ported; then
        # refused before the questions, or here on a terminal after a switch to the full settings
        KAUST_ANSWER="KAUST=no, but the port exists: the pipeline stops (make kaust CLEAN=yes)"
        [ "$1" = list ] || kaust_ported_refuse
    elif [ $KAUST_STATE = ported ]; then
        KAUST_ANSWER="ported (the port is deleted at the end)"
        if [ "$1" = list ] && [ "$KAUST_SW" = no ] && full_assign | grep -q '^set SUPPLIERS all$'; then
            show "[3/4] KAUST" "after a switch to the full settings (SUPPLIERS=all), the pipeline stops: KAUST=no, but the port exists (make kaust CLEAN=yes)"
            return 0
        fi
    elif ! kaust_runs; then
        KAUST_ANSWER="no stage runs a KAUST row (SUPPLIERS=all or the GPU core stage)"
        if [ "$1" = list ] && full_assign | grep -q '^set SUPPLIERS all$'; then
            case "$KAUST_SW" in
                yes) show "[3/4] KAUST" "after a switch to the full settings (SUPPLIERS=all), not asked (KAUST=yes): the copyright notice of CuBesselK, then the download and the port" ;;
                no)  show "[3/4] KAUST" "after a switch to the full settings (SUPPLIERS=all), not asked (KAUST=no): not downloaded" ;;
                *)   show "[3/4] KAUST" "asked after a switch to the full settings (SUPPLIERS=all): the copyright notice of CuBesselK, then: Download the two CuBesselK files and generate the port?" ;;
            esac
            return 0
        fi
    elif [ "$KAUST_SW" = no ]; then
        KAUST_ANSWER="not downloaded (KAUST=no); the KAUST rows are left out"
    elif [ "$KAUST_SW" = yes ] && [ "$1" = list ]; then
        KAUST_ANSWER="KAUST=yes: the copyright notice of CuBesselK, then the two files are downloaded and the port is generated first"
    elif [ "$KAUST_SW" = yes ]; then
        # the answer is given: the box and the check of the question, without the question
        echo; echo "== KAUST (KAUST=yes)"
        if bash references/kaust_logbesselk/kaust.sh given; then
            KAUST_GET=yes; KAUST_ANSWER="downloaded and ported first (KAUST=yes); the port is deleted at the end"
        else
            KAUST_ANSWER="cannot be downloaded here (above); the KAUST rows are left out"
        fi
        return 0
    elif [ "$1" = list ]; then
        show "[3/4] KAUST" "the copyright notice of CuBesselK (make kaust), then: Download the two CuBesselK files and generate the port?"
        return 0
    else
        echo; echo "== question: KAUST"
        ASKED="$ASKED KAUST"
        bash references/kaust_logbesselk/kaust.sh ask
        case $? in
            0) KAUST_GET=yes; KAUST_ANSWER="downloaded and ported first" ;;
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
    if [ $CORPUS = yes ]; then
        CORPUS_ANSWER="$cs; downloaded at the start of the run (CORPUS=yes)"
        [ "$1" = list ] && show "[4/4] corpus" "not asked: $CORPUS_ANSWER"
        return 0
    fi
    if [ "$1" = list ]; then
        show "[4/4] corpus" "Download the missing files from Zenodo? ($cs; a no stops the pipeline)"
        return 0
    fi
    echo; echo "== question: corpus"
    ASKED="$ASKED corpus"
    if DATA="$DATA_DIR" sh tools/corpus.sh ask "$ENTRY"; then
        CORPUS_GET=yes; CORPUS_ANSWER="downloaded at the start of the run"
    else
        CORPUS_STOP=yes; CORPUS_ANSWER="not downloaded: $cs -- the pipeline stops here"
    fi
}

# SETTINGS=full and CORPUS=yes take effect before the plan; KAUST=yes|no at the KAUST question.
if [ "$SETTINGS" = full ]; then
    _assign=$(full_assign)
    [ -z "$_assign" ] || apply_full "$_assign"
fi
if [ $CORPUS = yes ] && ! DATA="$DATA_DIR" sh tools/corpus.sh missing --quiet; then CORPUS_GET=yes; fi
# make reproduce without a terminal: the KAUST question needs KAUST=yes or KAUST=no.
kaust_unanswered() {
    [ $REPRODUCE = yes ] && [ -z "$KAUST_SW" ] && [ $KAUST_STATE = absent ] && kaust_runs && ! ask_tty
}
kaust_refuse() {
    echo "$ENTRY: without a terminal the KAUST question is not asked; give KAUST=yes or KAUST=no." >&2
    echo "  make kaust prints the copyright notice of CuBesselK.  KAUST=yes downloads its two files and" >&2
    echo "  generates the port, which is deleted at the end; KAUST=no runs without the KAUST rows." >&2
    echo "$ENTRY: nothing was run" >&2
    exit 2
}
# KAUST=no while the port exists: the stages would run the KAUST rows.
kaust_ported_refuse() {
    echo "$ENTRY: KAUST=no, but the KAUST port exists ($KAUST_HDR), and a run uses it." >&2
    echo "  make kaust CLEAN=yes deletes the port and keeps the downloaded files; without KAUST=no the" >&2
    echo "  run uses the port and deletes it at the end." >&2
    echo "$ENTRY: nothing was run" >&2
    exit 2
}

plan
if [ $DRY = yes ]; then
    if kaust_to_port; then
        # the notice as the run prints it, after the port is generated
        _mc=$(mktemp "${TMPDIR:-/tmp}/bench_missing.XXXXXX") &&
            grep -v '^kaust\(-dev\)\?|' "$BENCH_MISSING_CACHE" > "$_mc"
        BENCH_MISSING_CACHE=${_mc:-} full_run; rm -f ${_mc:+"$_mc"}
    else
        full_run
    fi
    echo "== the questions of $ENTRY on a terminal (DRY=yes: none asked, nothing run)"
    q_settings list; q_externals list; q_kaust list; q_corpus list
fi
# The refusals, with DRY=yes as without.
kaust_unanswered && kaust_refuse
kaust_no_ported && kaust_ported_refuse
[ $DRY = yes ] && exit 0
if ask_tty; then
    q_settings ask; q_externals ask; q_kaust ask; q_corpus ask
    # the answers to the questions asked; the plan above states the rest
    if [ -n "$ASKED" ]; then
        echo; echo "== the answers"
        case " $ASKED " in *" settings "*)  show "settings" "$SET_ANSWER" ;; esac
        case " $ASKED " in *" externals "*) show "externals" "$EXT_ANSWER" ;; esac
        case " $ASKED " in *" KAUST "*)     show "KAUST" "$KAUST_ANSWER" ;; esac
        case " $ASKED " in *" corpus "*)    show "corpus" "$CORPUS_ANSWER" ;; esac
    fi
    if [ $CORPUS_STOP = yes ]; then
        echo "pipeline: the corpus is incomplete; nothing was run (make corpus DOWNLOAD=yes downloads it)" >&2
        exit 2
    fi
    # test hook (BENCH_PIPELINE_STOP=answers|downloads|builds): stop after that step
    [ "${BENCH_PIPELINE_STOP:-}" = answers ] && exit 0
    FR_SHOW=brief
else
    # without a terminal: the KAUST question, answered by KAUST=yes|no (KAUST=yes prints the
    # box); the other switches are applied above
    [ -z "$KAUST_SW" ] || q_kaust ask
    FR_SHOW=notice
fi
# fr_show: on a terminal the plan after a switch and the differences from the full run, else the
# full-run notice, printed once (BENCH_FULL_RUN_NOTICE then silences it in the stages).  Both read the
# probe, so a run that downloads externals or KAUST prints them after the downloads (fr_show after).
_fr_notice=${BENCH_FULL_RUN_NOTICE:-}
fr_show() {
    if [ $FR_SHOW = brief ]; then
        # nothing to say when this run is the full benchmark and the plan was not changed
        [ $SWITCHED = no ] && [ -z "$(full_run --diff)" ] && return 0
        if [ $SWITCHED = yes ]; then echo; plan
        elif [ "${1:-}" = after ]; then echo; echo "== this run against the full benchmark"; fi
        full_run --brief
    else
        BENCH_FULL_RUN_NOTICE=$_fr_notice full_run
    fi
}
# Here when neither externals nor KAUST are downloaded, or when an incomplete corpus stops the
# run at step 0 (below).
if { [ $EXTERNALS = no ] && [ $KAUST_GET = no ]; } ||
   { [ $CORPUS_GET = no ] && ! DATA="$DATA_DIR" sh tools/corpus.sh missing >/dev/null; }; then
    fr_show; FR_SHOW=shown
fi
# From here on nothing asks.
export BENCH_FULL_RUN_NOTICE=shown BENCH_UNATTENDED=1
exec </dev/null

# Step 0: without a download the corpus is checked before anything runs.
if [ $CORPUS_GET = no ]; then DATA="$DATA_DIR" sh tools/corpus.sh need "$ENTRY" || exit 2; fi

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
            # the build's own lines: not make's per-file ones, and its "is up to date" in short
            grep -Ev '^  [A-Z]{2,5} +[^ ]+$' "$log" |
                sed -E 's/^make(\[[0-9]+\])?: (.* is up to date\.|Nothing to be done for .*)$/  up to date/'
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
trap 'kaust_cleanup; rm -f "$BENCH_MISSING_CACHE" ${BENCH_CORPUS_CHECK:+"$BENCH_CORPUS_CHECK"}' EXIT
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
    BENCH_NOTIFY_SKIP= bash benchmark/tools/notify.sh "$ENTRY" \
        "$verdict after $(printf '%d:%02d:%02d' $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60))); see the summary above and benchmark/runs/"
}

# 1. Downloads: externals, KAUST, corpus.  A failed external or KAUST is reported and the
# stages that need it skip; an incomplete corpus stops the pipeline.
if [ $EXTERNALS = yes ] && [ -z "$(ext_plan)" ]; then
    # the plan said "nothing to download"
    note "externals" "nothing to download"
elif [ $EXTERNALS = yes ]; then
    echo; echo "== download externals"
    _xs=$(mktemp) || exit 1
    EXTERNALS_STATUS="$_xs" EXTERNALS_PIPELINE=$(ext_mode) "$MAKE" --no-print-directory externals
    if [ -s "$_xs" ]; then
        while IFS= read -r _l; do note "externals" "WARNING $(echo $_l)"; warned=1; done < "$_xs"
    else note "externals" "ok"; fi
    rm -f "$_xs"
fi
if [ $KAUST_GET = yes ]; then
    echo; echo "== KAUST: download and port"
    # the box of the KAUST question was the notice; KAUST_PIPELINE leaves out the long one
    if KAUST_PIPELINE=1 "$MAKE" --no-print-directory kaust DOWNLOAD=yes; then
        KAUST_STATE=ported; note "KAUST" "downloaded and ported"
    else
        note "KAUST" "WARNING download or port failed; the KAUST rows are left out"; warned=1
    fi
fi
if [ $CORPUS_GET = yes ]; then
    echo; echo "== corpus: download"
    # the notice once: the corpus question shows it, else (CORPUS=yes) the download
    if [ $CORPUS = yes ]; then DATA="$DATA_DIR" sh tools/corpus.sh download
    else DATA="$DATA_DIR" sh tools/corpus.sh download --no-notice; fi
    if DATA="$DATA_DIR" sh tools/corpus.sh need "$ENTRY"; then
        note "corpus" "downloaded"
    else
        note "corpus" "incomplete after the download -- pipeline stopped"; failed=1; finish; exit 2
    fi
fi
if [ $EXTERNALS = yes ] || [ $KAUST_GET = yes ]; then
    # probe again for the stages and the summary
    _mxn=$(bash benchmark/tools/missing.sh probe) && mv -f "$_mxn" "$BENCH_MISSING_CACHE"
    # the interpreter again only where the downloads changed it from the plan's
    if [ $FIGURES = yes ]; then
        _pl=$(python_line)
        [ "$_pl" = "$PY_PLAN" ] || echo "pipeline: Python for the figures and tables now: $_pl"
    fi
fi
[ $FR_SHOW = shown ] || fr_show after
[ "${BENCH_PIPELINE_STOP:-}" = downloads ] && exit 0

# 2. Builds.  The Levin arms first, since levin-ab-build cleans build/.
[ $LEVIN = yes ] && build "Levin arms" hard levin-ab-build
# The library archive alone; the header self-check (make headers) is part of make all and
# make check, and under the pipeline runs only with CHECK=yes.
build "library" hard build/libmatern.a
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
[ "${BENCH_PIPELINE_STOP:-}" = builds ] && { finish; exit $failed; }

# 3. make check
if [ $CHECK = yes ]; then
    echo; echo "== make check"
    if "$MAKE" --no-print-directory check; then note "make check" "passed"
    else note "make check" "FAILED -- pipeline stopped"; failed=1; finish; exit 1; fi
fi

# 4. Runs.  The corpus is checked against data/MANIFEST.md once, for every run record: each
# run_benchmark.sh call (also the one of run_gpu.sh) writes the result from BENCH_CORPUS_CHECK
# into its MANIFEST.txt and, on a mismatch, a STATUS warning.  The file holds `<exit status>
# <DATA>`, then the output of tools/corpus_check.sh --list.
echo; echo "== corpus: sha256 check against data/MANIFEST.md (recorded in every run)"
if BENCH_CORPUS_CHECK=$(mktemp "${TMPDIR:-/tmp}/bench_corpus.XXXXXX"); then
    _cc=$(bash tools/corpus_check.sh "$DATA_DIR" --list 2>/dev/null); _ccrc=$?
    printf '%s %s\n%s\n' "$_ccrc" "$DATA_DIR" "$_cc" > "$BENCH_CORPUS_CHECK"
    export BENCH_CORPUS_CHECK
    case $_ccrc in
        0|1|3) echo "  $(printf '%s\n' "$_cc" | tail -n 1)" ;;
        *)     echo "  not checked (tools/corpus_check.sh exited $_ccrc)" ;;
    esac
else
    BENCH_CORPUS_CHECK=""; echo "  no temporary file: each run checks the corpus itself"
fi
# With the full settings the CPU runs carry the label of the published ones (run ids
# <stamp>_<host>_cpu[-<stage>]); other settings keep run_benchmark.sh's default or custom, and
# a SCOPE given to the pipeline names every run as before.
CPU_SCOPE=""
[ -z "${SCOPE:-}" ] && [ -z "$(full_assign)" ] && CPU_SCOPE=cpu
stage "CPU run" env STAGES=all ${CPU_SCOPE:+SCOPE=$CPU_SCOPE} bash benchmark/run_benchmark.sh
[ $LEVIN = yes ] && stage "Levin A/B" env STAGES=levin ${CPU_SCOPE:+SCOPE=$CPU_SCOPE} bash benchmark/run_benchmark.sh
if [ "$FOLD" = yes ]; then
    stage "prefactor fold" env STAGES=fold ${CPU_SCOPE:+SCOPE=$CPU_SCOPE} bash benchmark/run_benchmark.sh
elif [ "$FOLD" = failed ]; then
    note "prefactor fold" "not run: build/fold_cpu did not build"; failed=1
fi
if [ "$THREADS" = yes ]; then
    stage "thread scaling" env STAGES=threads ${CPU_SCOPE:+SCOPE=$CPU_SCOPE} bash benchmark/run_benchmark.sh
elif [ "$THREADS" = failed ]; then
    note "thread scaling" "not run: build/threads_cpu did not build"; failed=1
fi
[ $RUN_GPU = yes ] && stage "GPU pipeline" env GPU_STAGES="$GPU_STAGES_PLAN" NVCC="$nvcc_bin" \
    NVIDIA_SMI="$smi_bin" bash benchmark/run_gpu.sh

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
        if "$MAKE" --no-print-directory figures; then note "figures" "benchmark/plots/paper-final/"
        else note "figures" "FAILED"; failed=1; fi
        if [ -n "$_tx" ]; then
            note "tables" "not made -- ${_tx%%$'\t'*} missing: ${_tx#*$'\t'}"; failed=1
        elif "$MAKE" --no-print-directory tables; then note "tables" "benchmark/tables/paper/"; _tables=yes
        else note "tables" "FAILED"; failed=1; fi
    fi
    if [ $TABLES_CHECK = yes ]; then
        if [ -z "$_py" ] && [ -n "$_tx" ]; then note "tables check" "not run -- pdflatex missing: ${_tx#*$'\t'}"; failed=1
        elif [ $_tables = no ]; then note "tables check" "not run -- the tables were not made"; failed=1
        elif "$MAKE" --no-print-directory tables-check; then note "tables check" "passed"
        else note "tables check" "FAILED"; failed=1; fi
    fi
fi

finish
exit $failed
