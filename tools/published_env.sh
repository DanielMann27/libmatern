#!/bin/bash
# published_env.sh <report|lines> <component...> -- this machine's tools against the tools of
# the published results, and what a difference changes (make requirements, make pipeline,
# make reproduce).  Components: cc cxx nvcc boost-gpu python tex.
#   report  one row per component, in the layout of make requirements;
#   lines   `name<TAB>detail` for each component that differs; nothing for one that is the
#           same or absent (make requirements and the plan's missing externals name those).
# Reads CC, CXX, NVCC, PYTHON (its first word is the interpreter) and BOOST_ROOT as the
# Makefile resolves them.  Measures and changes nothing; exits 1 if the repository root
# cannot be entered, 2 for an unknown mode or component, otherwise 0.
set -u
cd "$(dirname "$0")/.." || exit 1

# The tools of the published results, as the PROVENANCE.txt of each published run records
# them.  NumPy and Matplotlib are PY_PINS of tools/py_stack.py.  Boost 1.75.0 (CPU row), GSL
# 2.6, the Bessels.jl and KAUST sources are pinned by the build itself and are not listed.
PUB_CC="GCC 11.5.0"
PUB_CXX="GCC 14.2.0"
PUB_NVCC="12.8.61"
PUB_BOOST_GPU="1.87.0"
PUB_PYTHON="3.12.9"
PUB_TEX="TeX Live 2020"
PY_PINS=$(sed -n 's/^PY_PINS = "\(.*\)"$/\1/p' tools/py_stack.py)
PUB_NUMPY=$(printf '%s\n' $PY_PINS | sed -n 's/^numpy==//p')
PUB_MPL=$(printf '%s\n' $PY_PINS | sed -n 's/^matplotlib==//p')

MODE="${1:-}"; shift
case "$MODE" in report|lines) ;; *)
    echo "published_env.sh: mode must be report or lines (got '$MODE')" >&2; exit 2 ;;
esac
for c in "$@"; do
    case "$c" in cc|cxx|nvcc|boost-gpu|python|tex) ;; *)
        echo "published_env.sh: unknown component '$c' (cc cxx nvcc boost-gpu python tex)" >&2; exit 2 ;;
    esac
done

CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-g++}"
NVCC_BIN="${NVCC:-nvcc}"
PY_BIN="${PYTHON:-}"; PY_BIN=${PY_BIN%% *}

# compiler_id <compiler [args]> <language>: `GCC 11.5.0`, `clang 18.1.8` or `icx <version>`
# from the compiler's predefined macros; empty when it does not run.
compiler_id() {
    local m
    m=$($1 -x "$2" -E -dM - </dev/null 2>/dev/null) || return 0
    printf '%s\n' "$m" | awk '$1 == "#define" { d[$2] = $3 }
        END {
            if ("__INTEL_LLVM_COMPILER" in d) print "icx " d["__INTEL_LLVM_COMPILER"]
            else if ("__clang__" in d) print "clang " d["__clang_major__"] "." d["__clang_minor__"] "." d["__clang_patchlevel__"]
            else if ("__GNUC__" in d) print "GCC " d["__GNUC__"] "." d["__GNUC_MINOR__"] "." d["__GNUC_PATCHLEVEL__"]
        }'
}

# One component: NAME, HERE (empty when absent), PUB, WHERE (the tool as found), RUNS (no:
# the tool exists but does not run), CHANGES (what a difference changes) and MATCH (how to use
# the published version).
probe() {
    NAME=""; HERE=""; PUB=""; WHERE=""; RUNS=yes; CHANGES=""; MATCH=""
    case "$1" in
    cc)
        NAME="C compiler"; PUB=$PUB_CC; WHERE="CC=$CC_BIN"
        HERE=$(compiler_id "$CC_BIN" c)
        [ -z "$HERE" ] && command -v "${CC_BIN%% *}" >/dev/null 2>&1 && RUNS=no
        CHANGES="the library, the benchmark drivers and the C supplier rows: values can differ in the last bits (code generation, FMA contraction); timings differ"
        MATCH="CC=<GCC 11.5.0>" ;;
    cxx)
        NAME="C++ compiler"; PUB=$PUB_CXX; WHERE="CXX=$CXX_BIN"
        HERE=$(compiler_id "$CXX_BIN" c++)
        [ -z "$HERE" ] && command -v "${CXX_BIN%% *}" >/dev/null 2>&1 && RUNS=no
        CHANGES="the std::cyl_bessel_k row (the libstdc++ of this compiler) and the Boost CPU row: values and timings"
        MATCH="CXX=<g++ 14.2.0>" ;;
    nvcc)
        NAME="nvcc"; PUB=$PUB_NVCC; WHERE="NVCC=$NVCC_BIN"
        HERE=$($NVCC_BIN --version 2>/dev/null | sed -n 's/.*, V\([0-9][0-9.]*\).*/\1/p' | tail -1)
        CHANGES="every GPU row (the CUDA backend and the device suppliers): values and timings"
        MATCH="NVCC=<nvcc 12.8.61>" ;;
    boost-gpu)
        NAME="Boost (GPU row)"; PUB=$PUB_BOOST_GPU; WHERE="BOOST_ROOT or the highest Boost >= 1.87"
        HERE=$(CXX="$CXX_BIN" bash benchmark/tools/boost_detect.sh 2>/dev/null |
               sed -n 's/^gpu row *: [^(]*(\([0-9][0-9.]*\)).*/\1/p' | head -1)
        CHANGES="the Boost GPU row: values and timings"
        MATCH="BOOST_ROOT=<Boost 1.87.0> (make boost-headers BOOST_VERSIONS=1.87.0 fetches it)" ;;
    python)
        NAME="Python, NumPy, Matplotlib"; WHERE="PYTHON=${PY_BIN:-(none)}"
        PUB="Python $PUB_PYTHON, NumPy $PUB_NUMPY, Matplotlib $PUB_MPL"
        local v s
        # py_stack.py prints `numpy <v> matplotlib <v>` (and exits 1 below the tested minimum).
        if [ -n "$PY_BIN" ] && v=$("$PY_BIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null); then
            s=$("$PY_BIN" tools/py_stack.py 2>/dev/null)
            case "$s" in "numpy "*)
                s=${s%%,*}
                HERE=$(printf '%s\n' "$s" | awk -v v="$v" '{ print "Python " v ", NumPy " $2 ", Matplotlib " $4 }') ;;
            esac
        fi
        CHANGES="the rendering of the figures (layout, text placement); no value in a figure or table (the generators compute in plain Python)"
        MATCH="PYTHON=<python3 with $PY_PINS>" ;;
    tex)
        NAME="pdflatex"; PUB=$PUB_TEX; WHERE="pdflatex on PATH"
        local l
        l=$(pdflatex --version 2>/dev/null | head -1)
        HERE=$(printf '%s\n' "$l" | sed -n 's/.*(\(TeX Live [0-9]*\).*/\1/p')
        [ -n "$l" ] && [ -z "$HERE" ] && HERE=$l
        CHANGES="the fitted column widths of the tables and their preview PDFs; no value"
        MATCH="pdflatex of TeX Live 2020 first on PATH" ;;
    esac
}

n=0; ndiff=0
for c in "$@"; do
    probe "$c"; n=$((n + 1))
    if [ "$MODE" = lines ]; then
        [ -n "$HERE" ] && [ "$HERE" != "$PUB" ] && printf '%s\t%s here, published %s: %s\n' "$NAME" "$HERE" "$PUB" "$CHANGES"
        continue
    fi
    if [ -z "$HERE" ] && [ $RUNS = no ]; then
        printf "  %-8s %-26s %s\n" "absent" "$NAME" "found, but it does not compile ($WHERE); published: $PUB"
    elif [ -z "$HERE" ]; then
        printf "  %-8s %-26s %s\n" "absent" "$NAME" "not found ($WHERE); published: $PUB"
    elif [ "$HERE" = "$PUB" ]; then
        printf "  %-8s %-26s %s\n" "same" "$NAME" "$HERE ($WHERE)"
    else
        ndiff=$((ndiff + 1))
        printf "  %-8s %-26s %s\n" "differs" "$NAME" "$HERE ($WHERE); published: $PUB"
        printf "  %-8s %-26s %s\n" "" "" "-> changes: $CHANGES"
        printf "  %-8s %-26s %s\n" "" "" "match: $MATCH"
    fi
done
if [ "$MODE" = report ]; then
    if [ $ndiff -eq 0 ]; then echo "  no tool found here differs from the published ones"
    else echo "  $ndiff of $n differ: results made here can differ from the published ones where marked"; fi
fi
exit 0
