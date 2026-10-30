#!/bin/bash
# externals.sh -- downloads the freely available externals without root (make externals):
# Bessels.jl sources, Boost 1.75.0/1.87.0 headers, GSL 2.6, NumPy/Matplotlib into .venv/.
# KAUST (all rights reserved) is not among them; `make kaust` states the copyright situation.
set -u
cd "$(dirname "$0")/.." || exit 1

# EXTERNALS_PIPELINE=cpu|gpu (make pipeline): what its run uses, without the Bessels.jl sources
# and the closing report, Boost 1.75.0 and GSL 2.6 only where the system lacks that version, and
# Boost 1.87.0 only for gpu, where boost_detect.sh selects no Boost for the GPU rows and
# BOOST_ROOT is not none.  EXTERNALS_PLAN=yes then prints what would be fetched, one `name|size`
# line each, and fetches nothing.
case "${EXTERNALS_PIPELINE:-}" in
    "")  LEAN=no;  BOOST_GET="1.75.0 1.87.0" ;;
    cpu) LEAN=yes; BOOST_GET="1.75.0" ;;
    gpu) LEAN=yes; BOOST_GET="1.75.0 1.87.0" ;;
    *)   echo "externals.sh: EXTERNALS_PIPELINE=$EXTERNALS_PIPELINE: use cpu or gpu" >&2; exit 2 ;;
esac
# Under the pipeline, without a copy in references/: SYS_BOOST, the Boost 1.75.0 that
# boost_detect.sh selects for the CPU row (it selects that version only); SYS_GSL, a GSL 2.6
# that gsl_flags.sh, which the build of the gsl row reads, takes from pkg-config (its --where
# ends in "(pkg-config)").  Another version does not count.  ROOT_ONLY: BOOST_CPU_ROOT where it
# holds no Boost 1.75.0; the CPU row uses that directory only, so Boost 1.75.0 is not fetched.
# For gpu, without references/boost/1.87.0: SYS_GPU, the Boost with GPU support of any version
# that boost_detect.sh selects for the GPU rows, as the build does (Makefile BOOST_GPU_DIR, same
# CXX and BOOST_ROOT); GPU_NONE, BOOST_ROOT=none, which selects none.  Either way Boost 1.87.0
# is not fetched.  SYS_BOOST and SYS_GPU are "<dir> (<origin>)" (boost_detect.sh --where).
SYS_BOOST=""; SYS_GSL=""; ROOT_ONLY=""; SYS_GPU=""; GPU_NONE=""
if [ $LEAN = yes ]; then
    if [ ! -f references/boost/1.75.0/boost/version.hpp ]; then
        SYS_BOOST=$(bash benchmark/tools/boost_detect.sh --where cpu 2>/dev/null)
        [ -n "$SYS_BOOST" ] || ROOT_ONLY=${BOOST_CPU_ROOT:-}
        [ -z "$SYS_BOOST$ROOT_ONLY" ] || BOOST_GET=$(echo $BOOST_GET | sed 's/1\.75\.0 *//')
    fi
    case " $BOOST_GET " in *" 1.87.0 "*)
        if [ ! -f references/boost/1.87.0/boost/version.hpp ]; then
            if [ "${BOOST_ROOT:-}" = none ]; then GPU_NONE=yes
            else SYS_GPU=$(bash benchmark/tools/boost_detect.sh --where gpu 2>/dev/null); fi
            [ -z "$SYS_GPU$GPU_NONE" ] || BOOST_GET=$(echo $BOOST_GET | sed 's/ *1\.87\.0//')
        fi ;;
    esac
    _w=$(bash benchmark/tools/gsl_flags.sh --where 2>/dev/null)
    case "$_w" in *"(pkg-config)")
        [ "$(bash benchmark/tools/gsl_flags.sh --modversion 2>/dev/null)" = 2.6 ] && SYS_GSL=$_w ;;
    esac
fi
# The plan applies the existence tests of fetch_boost.sh and fetch_gsl.sh for a copy in place
# (not the BOOST_VERSION check of fetch_boost.sh) and the first two tests of python_packages
# below.
if [ "${EXTERNALS_PLAN:-}" = yes ]; then
    [ $LEAN = yes ] || { echo "externals.sh: EXTERNALS_PLAN=yes needs EXTERNALS_PIPELINE=cpu|gpu" >&2; exit 2; }
    for _v in $BOOST_GET; do
        [ -f "references/boost/$_v/boost/version.hpp" ] || echo "Boost $_v headers|about 120-130 MB download"
    done
    [ -n "$SYS_GSL" ] || { [ -f references/gsl/2.6/lib/pkgconfig/gsl.pc ] && [ -f references/gsl/2.6/lib/libgsl.a ]; } ||
        echo "GSL 2.6|about 8 MB download, then a build of a few minutes"
    # python_packages below: nothing where PYTHON or .venv/ has the published versions; a
    # download where a Python 3.11+ can take them; else nothing where one meets the minimum.
    _p="${PYTHON:-python3 -B}"; _p=${_p%% *}
    if { command -v "$_p" >/dev/null 2>&1 && "$_p" tools/py_stack.py --exact >/dev/null 2>&1; } ||
       { [ -x .venv/bin/python3 ] && .venv/bin/python3 tools/py_stack.py --exact >/dev/null 2>&1; }; then :
    elif for _r in "$_p" python3 /usr/bin/python3; do command -v "$_r" >/dev/null 2>&1 && break; done &&
         "$_r" tools/py_stack.py --base >/dev/null 2>&1; then
        echo "NumPy and Matplotlib|about 40 MB download into .venv/"
    elif { command -v "$_p" >/dev/null 2>&1 && "$_p" tools/py_stack.py >/dev/null 2>&1; } ||
         { [ -x .venv/bin/python3 ] && .venv/bin/python3 tools/py_stack.py >/dev/null 2>&1; }; then :
    else
        echo "NumPy and Matplotlib|about 40 MB download into .venv/"
    fi
    exit 0
fi

# PYTHON as the user set it (PYTHON_ORIGIN: its make origin), not as the Makefile searched it.
python_user_set() {
    case "${PYTHON_ORIGIN:-}" in "command line"|environment|"environment override"|override) return 0 ;; esac
    return 1
}

declare -a SUMMARY=()
fail=0
item() {     # item <name> <command...>
    local name=$1; shift
    echo; echo "== $name"
    if "$@"; then SUMMARY+=("$(printf '%-22s ok' "$name")")
    else SUMMARY+=("$(printf '%-22s FAILED' "$name")"); fail=1; fi
}

# A copy found (SYS_BOOST, SYS_GPU, SYS_GSL), ROOT_ONLY and GPU_NONE are named, not fetched;
# fetch_boost.sh without a version fetches both.
boost_headers() {
    [ -z "$SYS_BOOST" ] || echo "  found     Boost 1.75.0 at $SYS_BOOST, for the boost-1.75.0 row; not fetched"
    case "$ROOT_ONLY" in
        "")   ;;
        none) echo "  skipped   Boost 1.75.0: BOOST_CPU_ROOT=none leaves out the boost-1.75.0 row" ;;
        *)    echo "  skipped   Boost 1.75.0: the boost-1.75.0 row uses BOOST_CPU_ROOT=$ROOT_ONLY only, which holds no Boost 1.75.0" ;;
    esac
    if [ -n "$SYS_GPU" ]; then
        local v
        v=$(sed -n 's/^#define BOOST_VERSION  *\([0-9]*\).*/\1/p' "${SYS_GPU% (*}/boost/version.hpp" 2>/dev/null | head -1)
        [ -z "$v" ] || v="$((v/100000)).$((v/100%1000)).$((v%100))"
        echo "  found     Boost ${v:-of unknown version} at $SYS_GPU, for the boost-gpu rows; not fetched$([ "$v" = 1.87.0 ] || echo ' (the published rows used 1.87.0)')"
    fi
    [ -z "$GPU_NONE" ] || echo "  skipped   Boost 1.87.0: BOOST_ROOT=none selects no Boost for the boost-gpu rows"
    [ -z "$BOOST_GET" ] || bash references/fetch_boost.sh $BOOST_GET
}
gsl_library() {
    if [ -n "$SYS_GSL" ]; then
        echo "  system    GSL 2.6 at $SYS_GSL, for the gsl row; not fetched"
    else
        bash references/fetch_gsl.sh 2.6
    fi
}
[ $LEAN = yes ] || item "Bessels.jl upstream" bash references/fetch.sh
item "Boost headers" boost_headers
item "GSL" gsl_library
# consequence <name>: what a failed item changes, for its summary line.
consequence() {
    case "$1" in
        "Bessels.jl upstream")
            echo "the provenance sources only; the bessels.jl rows are built from the C ports in references/bessels_jl/port/ and are unaffected" ;;
        "Boost headers")
            local c g
            c=$(bash benchmark/tools/boost_detect.sh --select cpu 2>/dev/null)
            g=$(bash benchmark/tools/boost_detect.sh --select gpu 2>/dev/null)
            printf '%s' "$([ -n "$c" ] && echo "the boost-1.75.0 row uses the Boost 1.75.0 at $c" || echo "the boost-1.75.0 row is left out")"
            # the GPU half only where 1.87.0 is in BOOST_GET (not for a pipeline without a GPU part)
            case " $BOOST_GET " in *" 1.87.0 "*)
                printf '; %s' "$([ -n "$g" ] && echo "the GPU rows use the Boost at $g" || echo "the boost-gpu rows and the GPU stage that checks Boost's K_nu short cut near nu = 5/2 are left out")" ;;
            esac ;;
        GSL)
            local gv
            if gv=$(bash benchmark/tools/gsl_flags.sh --modversion 2>/dev/null) && [ -n "$gv" ]; then
                echo "2.6 not built here; the gsl row uses the system GSL $gv$([ "$gv" = 2.6 ] || echo ' (the published rows used 2.6)')"
            else
                echo "2.6 not built here and no system GSL; the gsl row is left out"
            fi ;;
        "NumPy, Matplotlib")
            echo "the figures and tables are not made; make requirements names another way to add them" ;;
    esac
}
for _i in "${!SUMMARY[@]}"; do
    case "${SUMMARY[$_i]}" in *" FAILED")
        _n=$(printf '%s' "${SUMMARY[$_i]}" | sed 's/  *FAILED$//')
        [ "$_n" = "NumPy, Matplotlib" ] && continue    # its item runs below
        SUMMARY[$_i]="$(printf '%-22s FAILED: %s' "$_n" "$(consequence "$_n")")" ;;
    esac
done

py="${PYTHON:-python3 -B}"; py=${py%% *}
# PY_PINS: the versions that drew the published figures, from wheels only.  Where pip cannot
# install them, PY_FLOORS: pip takes the newest versions at or above the oldest stack the
# generators are tested with.  Both from tools/py_stack.py, which also decides whether an
# interpreter's NumPy and Matplotlib count (at or above PY_FLOORS).
PY_PINS=$(sed -n 's/^PY_PINS = "\(.*\)"$/\1/p' tools/py_stack.py)
PY_FLOORS=$(sed -n 's/^PY_FLOORS = "\(.*\)"$/\1/p' tools/py_stack.py)
PY_PINS_PYTHON=$(sed -n 's/^PY_PINS_PYTHON = "\(.*\)"$/\1/p' tools/py_stack.py)
PY_PUBLISHED=$(sed -n 's/^PY_PUBLISHED = "\(.*\)"$/\1/p' tools/py_stack.py)
# The first interpreter that runs tools/py_stack.py itself (it needs no NumPy).
py_runner() {
    local p
    for p in "$py" python3 /usr/bin/python3 python; do
        command -v "$p" >/dev/null 2>&1 && "$p" tools/py_stack.py --pins >/dev/null 2>&1 && { echo "$p"; return 0; }
    done
    return 1
}
python_packages() {
    local st runner b kind bpath bver vver="" made=no
    # An interpreter with the published versions needs nothing; one at or above the tested
    # minimum is used as it is only when no .venv/ with the published versions can be made.
    if command -v "$py" >/dev/null 2>&1 && st=$("$py" tools/py_stack.py --exact 2>/dev/null); then
        echo "  present   $st for $(command -v "$py")"; return 0
    fi
    if [ -x .venv/bin/python3 ] && st=$(.venv/bin/python3 tools/py_stack.py --exact 2>/dev/null); then
        echo "  present   .venv: $st"
        python_user_set && echo "  note      PYTHON=${PYTHON:-} is set (${PYTHON_ORIGIN:-}), so the Makefile uses it, not .venv/bin/python3"
        return 0
    fi
    runner=$(py_runner) || {
        echo "  failed    no Python found: install Python $PY_PINS_PYTHON or newer (the published figures used $PY_PUBLISHED; make requirements names the package)"
        return 1; }
    [ -x .venv/bin/python3 ] && vver=$(.venv/bin/python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)
    if [ -n "$vver" ] && .venv/bin/python3 -c "import sys; sys.exit(sys.version_info[:2] < tuple(map(int, '$PY_PINS_PYTHON'.split('.'))))"; then
        # .venv/ runs a Python that takes the published versions: they are installed into it.
        st=$(.venv/bin/python3 tools/py_stack.py 2>/dev/null)
        echo "  updating  .venv (Python $vver, ${st:-no numpy and matplotlib}), installing $PY_PINS"
        kind=pins; made=yes
    else
        # The interpreter .venv/ is made from (tools/py_stack.py --base: PYTHON_BASE, else the
        # newest Python 3.11+ with the venv module, else the newest 3.9+).
        b=$("$runner" tools/py_stack.py --base 2>/dev/null)
        read -r kind bpath bver <<<"$b"
        case $kind in
            pins|floors) ;;
            *) echo "  failed    $("$runner" tools/py_stack.py --advice)"
               return 1 ;;
        esac
        if [ $kind = floors ] && [ -n "$vver" ] && .venv/bin/python3 tools/py_stack.py >/dev/null 2>&1; then
            echo "  present   .venv (Python $vver): $(.venv/bin/python3 tools/py_stack.py), at or above the tested minimum, not the published $PY_PINS"
            echo "  note      the published versions need Python $PY_PINS_PYTHON or newer (the published figures used $PY_PUBLISHED; make requirements names the package); with one installed, make externals replaces .venv"
            return 0
        fi
        if [ -e .venv ]; then
            echo "  replacing .venv ($([ -n "$vver" ] && echo "Python $vver" || echo "its interpreter does not run")) by one from Python $bver ($bpath)"
            "$bpath" -m venv --clear .venv || { echo "  failed    $bpath -m venv --clear .venv"; return 1; }
        else
            echo "  creating  .venv from Python $bver ($bpath)"
            "$bpath" -m venv .venv || {
                echo "  failed    $bpath -m venv (on Debian/Ubuntu: sudo apt install python3-venv)"
                rm -rf .venv; return 1; }
        fi
    fi
    if [ $kind = pins ]; then
        if ! .venv/bin/python3 -m pip install --quiet --disable-pip-version-check \
                --only-binary=numpy,matplotlib $PY_PINS; then
            echo "  fallback  pip install $PY_PINS (the versions of the published figures) failed;"
            echo "            installing $PY_FLOORS; figures drawn with other versions can differ in detail"
            .venv/bin/python3 -m pip install --quiet --disable-pip-version-check $PY_FLOORS || {
                echo "  failed    pip install $PY_FLOORS"; return 1; }
        fi
    else
        echo "  installing $PY_FLOORS: the published versions ($PY_PINS) need Python $PY_PINS_PYTHON or newer,"
        echo "            this .venv has Python $bver; figures drawn with other versions can differ in detail"
        .venv/bin/python3 -m pip install --quiet --disable-pip-version-check $PY_FLOORS || {
            echo "  failed    pip install $PY_FLOORS"; return 1; }
        echo "  note      for the published versions install Python $PY_PINS_PYTHON or newer (make requirements names the package), then make externals again"
    fi
    echo "  verified  .venv: $(.venv/bin/python3 tools/py_stack.py)"
    if python_user_set; then
        echo "  note      PYTHON=${PYTHON:-} is set (${PYTHON_ORIGIN:-}), so the Makefile uses it, not .venv/bin/python3;"
        echo "            without PYTHON it searches PYTHON_CANDIDATES (make python-info shows the result)"
    else
        # the search of the Makefile for PYTHON: the published versions first, then the minimum
        local p m sel=""
        for m in --exact ""; do
            for p in ${PYTHON_CANDIDATES:-python3 /usr/bin/python3 python $PWD/.venv/bin/python3}; do
                command -v "$p" >/dev/null 2>&1 && "$p" tools/py_stack.py $m >/dev/null 2>&1 && { sel=$p; break 2; }
            done
        done
        echo "  selected  ${sel:-none}: the Makefile now takes the first of PYTHON_CANDIDATES with the published"
        echo "            versions, else the first at or above the tested minimum (make python-info shows it)"
    fi
}
item "NumPy, Matplotlib" python_packages
case "${SUMMARY[${#SUMMARY[@]}-1]}" in *" FAILED")
    SUMMARY[${#SUMMARY[@]}-1]="$(printf '%-22s FAILED: %s' "NumPy, Matplotlib" "$(consequence "NumPy, Matplotlib")")" ;;
esac

echo; echo "== externals"
printf '  %s\n' "${SUMMARY[@]}"
if [ $LEAN = no ]; then
    echo
    echo "What cannot be downloaded (system packages, KAUST) is listed by make requirements:"
    # PYTHON and PYTHON_ORIGIN reach this script from make's search, done before .venv/
    # existed; without them the nested make searches again.  A PYTHON the user set is kept.
    if python_user_set; then
        "${MAKE:-make}" --no-print-directory requirements
    else
        env -u PYTHON -u PYTHON_ORIGIN "${MAKE:-make}" --no-print-directory requirements
    fi
fi
# make pipeline: every item is optional, so the failed ones go to the file, exit status 0.
if [ -n "${EXTERNALS_STATUS:-}" ]; then
    printf '%s\n' "${SUMMARY[@]}" | grep 'FAILED' > "$EXTERNALS_STATUS"
    exit 0
fi
exit $fail
