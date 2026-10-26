#!/bin/bash
# missing.sh probe | rows <title> <cpu|threads|gpu> <parameter> <value> <list> | lines <key>...
#            | fetchable [key...]
#   -- names the external software a run lacks, each with the command that adds it, from
#   make externals-missing (tools/requirements.sh).  Never fails the command that calls it.
set -u
cd "$(dirname "$0")/../.." || exit 0

# probe: writes the probe into a new temporary file and prints its path; a caller exports it
# as BENCH_MISSING_CACHE and removes it at exit.  Without it, rows and lines probe for themselves.
probe() {
    local f
    f=$(mktemp "${TMPDIR:-/tmp}/bench_missing.XXXXXX") || return 0
    "${MAKE:-make}" -s --no-print-directory externals-missing > "$f" 2>/dev/null
    printf '%s\n' "$f"
}

CACHE=${BENCH_MISSING_CACHE:-}
if [ "${1:-}" != probe ] && { [ -z "$CACHE" ] || [ ! -r "$CACHE" ]; }; then
    CACHE=$(probe); trap 'rm -f "$CACHE"' EXIT
fi

# field <key> <n>: field n of the key's probe line, empty when the key is present.  Probe
# lines are `key|name|add|detail` (tools/requirements.sh missing).
field() {
    grep -m1 "^$1|" "$CACHE" 2>/dev/null | cut -d'|' -f"$2"
}

# label <key>: the component as a stage names it
label() {
    local n
    case "$1" in
        cxx)       echo "std::cyl_bessel_k" ;;
        boost-cpu) echo "Boost 1.75.0" ;;
        gsl)       echo "GSL" ;;
        kaust)     echo "KAUST CuBesselK" ;;
        nvcc)      echo "nvcc" ;;
        gpu)       echo "NVIDIA GPU" ;;
        boost-gpu) echo "Boost >= 1.87 (GPU)" ;;
        kaust-dev) echo "KAUST CuBesselK (GPU)" ;;
        python)    # "NumPy, Matplotlib (<interpreter>)" or "Python 3", as the probe names it
                   n=$(field python 2); echo "${n:-Python 3}" ;;
        pdflatex)  echo "pdflatex" ;;
        bessels-upstream) echo "Bessels.jl sources" ;;
        download)  echo "curl or wget" ;;
        bzip2)     echo "bzip2" ;;
        sha256)    echo "sha256sum/shasum/openssl" ;;
        *)         n=$(field "$1" 2); echo "${n:-$1}" ;;
    esac
}
# text <key>: the command that adds it, else why it is absent; empty when present.
text() {
    local line add detail
    line=$(grep -m1 "^$1|" "$CACHE" 2>/dev/null) || return 0
    add=$(printf '%s' "$line" | cut -d'|' -f3)
    detail=$(printf '%s' "$line" | cut -d'|' -f4-)
    if [ -n "$add" ]; then printf '%s' "$add"; else printf '%s' "$detail"; fi
}

# size <key>: the approximate download of a fetchable key.
size() {
    case "$1" in
        boost-cpu)        echo "about 120-130 MB download (Boost 1.75.0 headers)" ;;
        boost-gpu)        echo "about 120-130 MB download (Boost 1.87.0 headers)" ;;
        gsl)              echo "about 8 MB download, then a build of a few minutes" ;;
        bessels-upstream) echo "under 1 MB download" ;;
        python)           echo "about 40 MB download into .venv/" ;;
        *)                echo "size not known" ;;
    esac
}

# fetchable [key...]: one `key|label|size` line per absent key that make externals or make
# references adds without root (its add field starts with either); every probed key when
# none is given.
fetchable() {
    local k add
    [ "$#" -gt 0 ] || set -- $(cut -d'|' -f1 "$CACHE" 2>/dev/null)
    for k in "$@"; do
        add=$(field "$k" 3)
        case "$add" in
            "make externals"*|"make references"*) printf '%s|%s|%s\n' "$k" "$(label "$k")" "$(size "$k")" ;;
        esac
    done
    return 0
}

# lines <key>...: one `<label><TAB><command or reason>` line per absent key.
lines() {
    local k t
    for k in "$@"; do
        t=$(text "$k"); [ -n "$t" ] && printf '%s\t%s\n' "$(label "$k")" "$t"
    done
    return 0
}

# rows: the block a supplier stage prints at its start: the rows its --list output lacks, each
# with its command, and whether built KAUST rows run under <parameter>=<value>.
rows() {
    local title=$1 kind=$2 param=$3 value=$4 list=$5 prog keys k pat t got="" absent=""
    local -a out=()
    case "$kind" in
        cpu)     prog=build/suppliers;     keys="cxx:std-cyl_bessel_k boost-cpu:boost- gsl:gsl- kaust:kaust" ;;
        threads) prog=build/threads_cpu;   keys="cxx:std-cyl_bessel_k boost-cpu:boost- gsl:gsl- kaust:kaust" ;;
        gpu)     prog=build/suppliers_gpu; keys="boost-gpu:boost-gpu kaust-dev:kaust" ;;
        *)       return 0 ;;
    esac
    # the rows the build has: --list lines of the form `  <row> ...`
    local built; built=$(printf '%s\n' "$list" | awk '/^  [a-z]/{print $1}')
    for k in $keys; do
        pat=${k#*:}; k=${k%%:*}
        if printf '%s\n' "$built" | grep -q "^$pat"; then
            [ "${k%-dev}" = kaust ] && got=yes
            continue
        fi
        t=$(text "$k")
        if [ -n "$t" ]; then absent=yes; else t="present, but not in this build: make $prog"; fi
        out+=("$(printf '    %-24s %s' "$(label "$k")" "$t")")
    done
    local kaust_off=""
    [ "$got" = yes ] && [ -n "$param" ] && [ "${value:-ship}" != all ] && \
        kaust_off="the KAUST rows are built but not run ($param=all adds them)"
    if [ ${#out[@]} -eq 0 ]; then
        if [ -n "$kaust_off" ]; then echo "$title: every supplier row but KAUST's ($param=all adds them)"
        else echo "$title: every supplier row"; fi
        return 0
    fi
    echo "$title: without the rows of"
    printf '%s\n' "${out[@]}"
    [ -n "$kaust_off" ] && echo "  $kaust_off"
    [ -n "$absent" ] && \
        echo "  once added: make pipeline rebuilds the supplier programs; run directly, make $prog first"
    return 0
}

case "${1:-}" in
    probe) probe ;;
    rows)  shift; rows "${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" ;;
    lines) shift; lines "$@" ;;
    fetchable) shift; fetchable "$@" ;;
    *)     echo "usage: missing.sh probe | rows <title> <cpu|threads|gpu> <parameter> <value> <list> | lines <key>... | fetchable [key...]" >&2 ;;
esac
exit 0
