#!/bin/bash
# externals.sh -- downloads the freely available externals without root (make externals):
# Bessels.jl sources, Boost 1.75.0/1.87.0 headers, GSL 2.6, NumPy/Matplotlib into .venv/.
# KAUST (all rights reserved) is not among them; `make kaust` states the copyright situation.
set -u
cd "$(dirname "$0")/.." || exit 1

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

item "Bessels.jl upstream" bash references/fetch.sh
item "Boost headers" bash references/fetch_boost.sh 1.75.0 1.87.0
item "GSL" bash references/fetch_gsl.sh 2.6
# consequence <name>: what a failed item changes, for its summary line.
consequence() {
    case "$1" in
        "Bessels.jl upstream")
            echo "the provenance sources only; the bessels.jl rows are built from the C ports in references/bessels_jl/port/ and are unaffected" ;;
        "Boost headers")
            local c g
            c=$(bash benchmark/tools/boost_detect.sh --select cpu 2>/dev/null)
            g=$(bash benchmark/tools/boost_detect.sh --select gpu 2>/dev/null)
            printf '%s; %s' \
                "$([ -n "$c" ] && echo "the boost-1.75.0 row uses the Boost 1.75.0 at $c" || echo "the boost-1.75.0 row is left out")" \
                "$([ -n "$g" ] && echo "the GPU rows use the Boost at $g" || echo "the boost-gpu rows and the GPU stage that checks Boost's K_nu short cut near nu = 5/2 are left out")" ;;
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
# Minimum versions for pip: the oldest stack the generators are tested with.
PY_FLOORS="numpy>=2.0.2 matplotlib>=3.9.4"
python_packages() {
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
        echo "  present   $("$py" -c 'import numpy,matplotlib; print("numpy", numpy.__version__, "and matplotlib", matplotlib.__version__)') for $(command -v "$py")"
        return 0
    fi
    if [ -x .venv/bin/python3 ] && .venv/bin/python3 -c 'import numpy, matplotlib' >/dev/null 2>&1; then
        echo "  present   .venv (numpy and matplotlib)"
        python_user_set && echo "  note      PYTHON=${PYTHON:-} is set (${PYTHON_ORIGIN:-}), so the Makefile uses it, not .venv/bin/python3"
        return 0
    fi
    local base made=no
    # A .venv whose interpreter runs but does not import both (e.g. after a Python minor
    # upgrade) is adapted to the current Python in place.
    if [ -x .venv/bin/python3 ] && .venv/bin/python3 -c '' >/dev/null 2>&1; then
        echo "  updating  .venv ($(.venv/bin/python3 -c 'import sys; print("Python", sys.version.split()[0])') does not import numpy and matplotlib), installing $PY_FLOORS"
        .venv/bin/python3 -m venv --upgrade .venv && made=yes
    fi
    if [ $made = no ]; then
        base=$(command -v python3 2>/dev/null) || { echo "  failed    no python3 to create .venv from"; return 1; }
        echo "  creating  .venv from $base, installing $PY_FLOORS"
        if ! "$base" -m venv .venv; then
            echo "  failed    python3 -m venv (on Debian/Ubuntu: sudo apt install python3-venv)"
            rm -rf .venv; return 1
        fi
    fi
    .venv/bin/python3 -m pip install --quiet --disable-pip-version-check $PY_FLOORS || {
        echo "  failed    pip install $PY_FLOORS"; return 1; }
    echo "  verified  .venv: $(.venv/bin/python3 -c 'import numpy,matplotlib; print("numpy", numpy.__version__, "matplotlib", matplotlib.__version__)')"
    if python_user_set; then
        echo "  note      PYTHON=${PYTHON:-} is set (${PYTHON_ORIGIN:-}), so the Makefile uses it, not .venv/bin/python3;"
        echo "            without PYTHON it searches PYTHON_CANDIDATES (make python-info shows the result)"
    else
        # the search of the Makefile for PYTHON
        local p sel=""
        for p in ${PYTHON_CANDIDATES:-python3 /usr/bin/python3 python $PWD/.venv/bin/python3}; do
            command -v "$p" >/dev/null 2>&1 && "$p" -c 'import numpy, matplotlib' >/dev/null 2>&1 && { sel=$p; break; }
        done
        echo "  selected  ${sel:-none}: the Makefile now takes the first of PYTHON_CANDIDATES that imports"
        echo "            both (make python-info shows it)"
    fi
}
item "NumPy, Matplotlib" python_packages
case "${SUMMARY[${#SUMMARY[@]}-1]}" in *" FAILED")
    SUMMARY[${#SUMMARY[@]}-1]="$(printf '%-22s FAILED: %s' "NumPy, Matplotlib" "$(consequence "NumPy, Matplotlib")")" ;;
esac

echo; echo "== externals"
printf '  %s\n' "${SUMMARY[@]}"
echo
echo "What cannot be downloaded (system packages, KAUST) is listed by make requirements:"
# PYTHON and PYTHON_ORIGIN reach this script from make's search, done before .venv/ existed;
# without them the nested make searches again.  A PYTHON the user set is kept.
if python_user_set; then
    "${MAKE:-make}" --no-print-directory requirements
else
    env -u PYTHON -u PYTHON_ORIGIN "${MAKE:-make}" --no-print-directory requirements
fi
# make pipeline: every item is optional, so the failed ones go to the file, exit status 0.
if [ -n "${EXTERNALS_STATUS:-}" ]; then
    printf '%s\n' "${SUMMARY[@]}" | grep 'FAILED' > "$EXTERNALS_STATUS"
    exit 0
fi
exit $fail
