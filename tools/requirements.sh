#!/bin/bash
# requirements.sh [all|suppliers|missing] -- reports the external software the build and
# benchmark can use and how to add what is absent (make requirements | references |
# externals-missing).  Measures and changes nothing; always exits 0.
set -u
cd "$(dirname "$0")/.." || exit 1

SECTION="${1:-all}"
case "$SECTION" in all|suppliers|missing) ;; *)
    echo "requirements.sh: section must be all, suppliers or missing (got '$SECTION')" >&2; exit 2 ;;
esac
# `missing`: the report goes nowhere, the key lines to the original standard output.
[ "$SECTION" = missing ] && exec 3>&1 1>/dev/null

CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-g++}"
NVCC_BIN="${NVCC:-nvcc}"
# Same default as PYTHON_CANDIDATES in the Makefile (used when run on its own).
PYTHON_CANDIDATES="${PYTHON_CANDIDATES:-python3 /usr/bin/python3 python $PWD/.venv/bin/python3}"

# The Makefile passes each variable's make origin.  A variable the user set is used
# as given, so a hint to install the default tool would not apply to it.
is_user_set() {
    case "${1:-}" in "command line"|environment|"environment override"|override) return 0 ;; esac
    return 1
}
# user_chose <origin> <value> <default names...>: set by the user to a non-default tool.
# CXX=g++ exported by an environment names a default tool, so installing it is the fix.
user_chose() {
    is_user_set "$1" || return 1
    local w=${2%% *} d; shift 2
    for d in "$@"; do [ "$w" = "$d" ] && return 1; done
    return 0
}
# have <command [args]>: the command word exists (tool variables may carry arguments).
have() { command -v "${1%% *}" >/dev/null 2>&1; }

# Install hints name packages for apt, dnf, pacman and brew; elsewhere they name the component.
if   command -v apt-get >/dev/null 2>&1; then PM=apt
elif command -v dnf     >/dev/null 2>&1; then PM=dnf
elif command -v pacman  >/dev/null 2>&1; then PM=pacman
elif command -v brew    >/dev/null 2>&1; then PM=brew
else PM=other; fi

# externally_managed <python>: the interpreter refuses pip installs (PEP 668: an
# EXTERNALLY-MANAGED file beside its standard library, and it is not a virtual environment).
externally_managed() {
    "$1" -c 'import os, sys, sysconfig
m = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
sys.exit(0 if os.path.isfile(m) and sys.prefix == sys.base_prefix else 1)' >/dev/null 2>&1
}

# hint <component> [python]: the command that adds it on this machine's package manager;
# for pymods, [python] is the interpreter the packages are for.
hint() {
    case "$PM:$1" in
        apt:cxx)        echo "sudo apt install g++" ;;
        apt:gsl)        echo "sudo apt install libgsl-dev pkg-config" ;;
        apt:pkgconfig)  echo "sudo apt install pkg-config" ;;
        apt:curl)       echo "sudo apt install curl" ;;
        apt:objcopy)    echo "sudo apt install binutils" ;;
        apt:python)     echo "sudo apt install python3" ;;
        apt:pymods)     echo "sudo apt install python3-numpy python3-matplotlib   (for the system python3)" ;;
        apt:pdflatex)   echo "sudo apt install texlive-latex-extra" ;;
        apt:nvcc)       echo "sudo apt install nvidia-cuda-toolkit   (or NVIDIA's CUDA toolkit; NVCC=<path>)" ;;
        apt:bzip2)      echo "sudo apt install bzip2" ;;
        apt:sha256)     echo "sudo apt install coreutils" ;;
        dnf:cxx)        echo "sudo dnf install gcc-c++" ;;
        dnf:gsl)        echo "sudo dnf install gsl-devel pkgconf-pkg-config" ;;
        dnf:pkgconfig)  echo "sudo dnf install pkgconf-pkg-config" ;;
        dnf:curl)       echo "sudo dnf install curl" ;;
        dnf:objcopy)    echo "sudo dnf install binutils" ;;
        dnf:python)     echo "sudo dnf install python3" ;;
        dnf:pymods)     echo "sudo dnf install python3-numpy python3-matplotlib   (for the system python3)" ;;
        dnf:pdflatex)   echo "install a TeX distribution with pdflatex (TeX Live)" ;;
        dnf:bzip2)      echo "sudo dnf install bzip2" ;;
        dnf:sha256)     echo "sudo dnf install coreutils" ;;
        pacman:cxx)       echo "sudo pacman -S gcc" ;;
        pacman:gsl)       echo "sudo pacman -S gsl pkgconf" ;;
        pacman:pkgconfig) echo "sudo pacman -S pkgconf" ;;
        pacman:curl)      echo "sudo pacman -S curl" ;;
        pacman:objcopy)   echo "sudo pacman -S binutils" ;;
        pacman:python)    echo "sudo pacman -S python" ;;
        pacman:pymods)    echo "sudo pacman -S python-numpy python-matplotlib   (for the system python)" ;;
        pacman:pdflatex)  echo "sudo pacman -S texlive-basic texlive-latex texlive-latexrecommended" ;;
        pacman:nvcc)      echo "sudo pacman -S cuda   (nvcc is /opt/cuda/bin/nvcc: NVCC=/opt/cuda/bin/nvcc, or /opt/cuda/bin in PATH)" ;;
        pacman:bzip2)     echo "sudo pacman -S bzip2" ;;
        pacman:sha256)    echo "sudo pacman -S coreutils" ;;
        brew:cxx)       echo "xcode-select --install" ;;
        brew:gsl)       echo "brew install gsl pkg-config" ;;
        brew:pkgconfig) echo "brew install pkg-config" ;;
        brew:curl)      echo "brew install curl" ;;
        brew:objcopy)   echo "brew install binutils" ;;
        brew:python)    echo "brew install python" ;;
        brew:pymods)    externally_managed "${2:-python3}" || echo "${2:-python3} -m pip install numpy matplotlib" ;;
        brew:pdflatex)  echo "brew install --cask mactex-no-gui" ;;
        brew:bzip2)     echo "brew install bzip2" ;;
        brew:sha256)    echo "brew install coreutils" ;;
        *:cxx)          echo "install a C++17 compiler (g++ or clang++)" ;;
        *:gsl)          echo "install GSL with its pkg-config file" ;;
        *:pkgconfig)    echo "install pkg-config" ;;
        *:curl)         echo "install curl or wget" ;;
        *:objcopy)      echo "install GNU binutils" ;;
        *:python)       echo "install Python 3" ;;
        *:pymods)       if [ -n "${2:-}" ] && externally_managed "$2"; then
                            echo "the distribution's NumPy and Matplotlib packages (for the system python)"
                        else echo "${2:-<python>} -m pip install numpy matplotlib"; fi ;;
        *:pdflatex)     echo "install a TeX distribution with pdflatex (TeX Live)" ;;
        *:nvcc)         echo "install NVIDIA's CUDA toolkit (NVCC=<path>)" ;;
        *:bzip2)        echo "install bzip2" ;;
        *:sha256)       echo "install sha256sum (GNU coreutils), shasum or openssl" ;;
    esac
}

MISSING=0
# row <state> <name> <detail> [enables] [how to add] [key] [probe name]
# `missing` prints instead one line `key|name|add|detail` per absent component (`add`
# empty when nothing can be added; `name` is [probe name] when given), read by
# benchmark/tools/missing.sh; keys: download bzip2 sha256 bessels-upstream cxx boost-cpu
# gsl kaust nvcc gpu boost-gpu kaust-dev python pdflatex.  Rows without a key are not printed.
row() {
    local state=$1 name=$2 detail=$3 enables=${4:-} add=${5:-} key=${6:-} pname=${7:-$2}
    if [ "$SECTION" = missing ]; then
        [ "$state" != found ] && [ -n "$key" ] && printf '%s|%s|%s|%s\n' "$key" "$pname" "$add" "$detail" >&3
        return 0
    fi
    printf "  %-8s %-26s %s\n" "$state" "$name" "$detail"
    [ -n "$enables" ] && printf "  %-8s %-26s %s\n" "" "" "-> $enables"
    if [ "$state" != found ] && [ -n "$add" ]; then
        printf "  %-8s %-26s %s\n" "" "" "add: $add"
        MISSING=$((MISSING + 1))
    fi
}

version_of() { "$@" 2>/dev/null | head -1; }

section() { echo; echo "== $1"; }

toolchain() {
    section "toolchain"
    if have "$CC_BIN"; then
        row found "C compiler" "$CC_BIN: $(version_of $CC_BIN --version)" "library, tests, benchmark drivers"
    elif user_chose "${CC_ORIGIN:-}" "$CC_BIN" cc gcc; then
        row MISSING "C compiler" "CC=$CC_BIN (set on the ${CC_ORIGIN}) not found" "everything" \
            "unset CC, or set it to an installed compiler"
    else
        row MISSING "C compiler" "$CC_BIN not found" "everything" "install gcc or clang (CC=<compiler>)"
    fi
    if command -v objcopy >/dev/null 2>&1; then
        row found "objcopy" "$(command -v objcopy)" "the library (every non-public symbol made local)"
    else
        row MISSING "objcopy" "not found" "the library build" "$(hint objcopy)"
    fi
    # nm comes with objcopy in binutils, so the same package adds it.
    if command -v nm >/dev/null 2>&1; then
        row found "nm" "$(command -v nm)" "make check (exports: the global symbols of build/libmatern.a)"
    else
        row MISSING "nm" "not found" "make check (exports)" "$(hint objcopy)"
    fi
    if command -v pkg-config >/dev/null 2>&1; then
        row found "pkg-config" "$(command -v pkg-config)" "GSL detection"
    else
        row absent "pkg-config" "not found" "a system GSL; optional (a GSL built by make externals needs none)" "$(hint pkgconfig)"
    fi
    if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then
        row found "curl/wget" "$(command -v curl || command -v wget)" \
            "every download: the corpus (make corpus DOWNLOAD=yes), make externals, make boost-headers, make references, the KAUST sources (make kaust)"
    else
        row MISSING "curl/wget" "neither found" \
            "every download: the corpus (make corpus), Boost and GSL (make externals, make boost-headers), the Bessels.jl sources (make references), the KAUST sources (make kaust)" \
            "$(hint curl)" download
    fi
    if command -v bzip2 >/dev/null 2>&1; then
        row found "bzip2" "$(command -v bzip2)" "unpacking the Boost archives (make externals, make boost-headers)"
    else
        row absent "bzip2" "not found" \
            "unpacking the Boost archives: make externals and make boost-headers fetch no Boost without it" \
            "$(hint bzip2)" bzip2
    fi
    # The corpus checks with the first of sha256sum, shasum and openssl; the fetchers of
    # make externals and make references with sha256sum.
    if command -v sha256sum >/dev/null 2>&1; then
        row found "sha256sum" "$(command -v sha256sum)" "the digests of every download and of make corpus-check"
    elif command -v shasum >/dev/null 2>&1 || command -v openssl >/dev/null 2>&1; then
        row found "shasum/openssl" "$(command -v shasum || command -v openssl)" "the corpus digests (make corpus, make corpus-check)"
        row absent "sha256sum" "not found" \
            "the digests of make externals, make boost-headers and make references; without it they fetch nothing" \
            "$(hint sha256)"
    else
        row MISSING "sha256sum" "none of sha256sum, shasum, openssl found" \
            "the digests of every download and of make corpus-check; without one the corpus cannot be downloaded or checked" \
            "$(hint sha256)" sha256 "sha256sum/shasum/openssl"
    fi
}

# boost_rows <have_cxx>: every Boost found and the one each row selects; sets
# BOOST_GPU_SEL and BOOST_GPU_NOTE.
boost_rows() {
    local bd cpu gpu line
    bd=$(CXX="$CXX_BIN" bash benchmark/tools/boost_detect.sh 2>/dev/null)
    cpu=$(printf '%s\n' "$bd" | sed -n 's/^cpu row *: *//p' | head -1)
    gpu=$(printf '%s\n' "$bd" | sed -n 's/^gpu row *: *//p' | head -1)
    while read -r line; do
        [ -n "$line" ] && printf "  %-8s %-26s %s\n" "" "Boost on this machine" "${line#found }"
    done <<< "$(printf '%s\n' "$bd" | grep '^  found ')"
    BOOST_GPU_NEED=$(printf '%s\n' "$bd" | sed -n 's/^gpu needs *: *//p' | head -1)
    local cneed; cneed=$(printf '%s\n' "$bd" | sed -n 's/^cpu needs *: *//p' | head -1)
    BOOST_GPU_SEL=""; BOOST_GPU_NOTE="${gpu#none  -- }"
    case "$gpu" in none*) ;; *) BOOST_GPU_SEL=$gpu ;; esac
    if [ "$1" != yes ]; then
        row absent "Boost (CPU row)" "not built: no C++ compiler" "boost row; optional" "" boost-cpu
    elif [ "${cpu#none}" != "$cpu" ]; then
        local cadd
        case "$cneed" in
            unset)    cadd="unset BOOST_CPU_ROOT, or BOOST_CPU_ROOT=<dir holding Boost 1.75.0>" ;;
            fix-root) cadd="BOOST_CPU_ROOT=<dir holding Boost 1.75.0>, or unset it (no other Boost is used while it is set)" ;;
            *)        cadd="make externals   (or make boost-headers BOOST_VERSIONS=1.75.0; or BOOST_CPU_ROOT=<dir holding Boost 1.75.0>)" ;;
        esac
        row absent "Boost (CPU row)" "${cpu#none  -- }" "boost-1.75.0 row; optional, the row is left out" "$cadd" boost-cpu
    else
        row found "Boost (CPU row)" "$cpu" "boost-1.75.0 row"
    fi
    printf "  %-8s %-26s %s\n" "" "" "   CPU row: Boost 1.75.0 only (BOOST_CPU_ROOT=<dir> names it, =none disables);"
    printf "  %-8s %-26s %s\n" "" "" "   GPU row: BOOST_ROOT or the highest Boost >= 1.87; make boost-headers fetches both"
}

suppliers() {
    section "external K_nu suppliers (make build/suppliers; build/suppliers --list shows the built rows)"
    row found "Bessels.jl (2 rows)" "C port in references/bessels_jl/port/, always built" \
        "bessels.jl, bessels.jl-old"
    local up=references/bessels_jl/upstream n
    n=$(ls "$up"/*.jl 2>/dev/null | wc -l)
    if [ "$n" -ge 12 ]; then
        row found "Bessels.jl upstream" "$up ($n files; make references verifies the digests)" \
            "provenance of the port only; not compiled"
    else
        row absent "Bessels.jl upstream" "$up has $n of 12 files" \
            "provenance of the port only; not compiled" "make references" bessels-upstream
    fi

    local have_cxx=no cxxerr
    if have "$CXX_BIN"; then
        cxxerr=$(printf '#include <cmath>\nint main(){return (int)std::cyl_bessel_k(1.5,2.0);}\n' \
                 | $CXX_BIN -std=c++17 -x c++ - -o /dev/null 2>&1)
        if [ $? -eq 0 ]; then
            have_cxx=yes
            row found "C++ compiler" "$CXX_BIN: $(version_of $CXX_BIN --version)" "std-cyl_bessel_k row (and the Boost row)"
        else
            row absent "C++ compiler" "$CXX_BIN cannot build std::cyl_bessel_k: $(printf '%s\n' "$cxxerr" | grep -m1 -i error)" \
                "std-cyl_bessel_k and boost rows; optional, both rows are left out" \
                "a C++17 compiler with libstdc++ (e.g. CXX=g++; libc++ has no C++17 special functions)" cxx
        fi
    elif user_chose "${CXX_ORIGIN:-}" "$CXX_BIN" c++ g++; then
        row absent "C++ compiler" "CXX=$CXX_BIN (set on the ${CXX_ORIGIN}) not found" \
            "std-cyl_bessel_k and boost rows; optional, both rows are left out" \
            "unset CXX, or set it to an installed compiler" cxx
    else
        row absent "C++ compiler" "$CXX_BIN not found" \
            "std-cyl_bessel_k and boost rows; optional, both rows are left out" "$(hint cxx)" cxx
    fi
    boost_rows "$have_cxx"

    if bash benchmark/tools/gsl_flags.sh --exists; then
        row found "GSL" "$(bash benchmark/tools/gsl_flags.sh --modversion) at $(bash benchmark/tools/gsl_flags.sh --where)" "gsl row (CPU only)"
    else
        row absent "GSL" "no GSL found (references/gsl/<version>/ or pkg-config)" "gsl row; optional, the row is left out" \
            "make externals   (builds GSL 2.6, the published version, without root)   or: $(hint gsl)" gsl
    fi

    # CuBesselK is not distributed.  The hint is `make kaust`, whose notice states the
    # copyright situation first.
    if [ -e references/build/kaust/kaust_derived.h ]; then
        row found "KAUST CuBesselK" "ported (references/build/kaust/; deleted at the end of make pipeline)" "kaust-128, kaust-40, kaust-adaptive rows"
    else
        row absent "KAUST CuBesselK" "not ported (not distributed: all rights reserved)" \
            "kaust-* rows; optional, the rows are left out" \
            "make kaust   (not part of this distribution: the copyright situation and how to obtain it)" kaust
    fi

    section "device suppliers (make build/suppliers_gpu; needs nvcc, and an NVIDIA GPU to run)"
    if have "$NVCC_BIN"; then
        local nv nverr ntmp nmax
        nv=$($NVCC_BIN --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p')
        ntmp=$(mktemp -d)
        printf '__global__ void k(double *x){ x[0] = 1.0; }\nint main(){ return 0; }\n' > "$ntmp/t.cu"
        nverr=$($NVCC_BIN -c "$ntmp/t.cu" -o "$ntmp/t.o" 2>&1)
        local nrc=$?
        rm -rf "$ntmp"
        if [ $nrc -eq 0 ]; then
            row found "nvcc" "$NVCC_BIN: $nv" "CUDA backend, GPU benchmark, device supplier rows"
        else
            nmax=$(printf '%s\n' "$nverr" | sed -n 's/.*versions later than \([0-9]*\) are not supported.*/\1/p' | head -1)
            if [ -n "$nmax" ]; then
                row absent "nvcc" "$NVCC_BIN: $nv; its host compiler is newer than GCC $nmax, which this CUDA supports" \
                    "CUDA backend, GPU benchmark and every device row; optional" \
                    "install g++-$nmax and pass NVCC=\"nvcc -ccbin g++-$nmax\" to every GPU target" nvcc
            else
                row absent "nvcc" "$NVCC_BIN: $nv; a test kernel does not compile: $(printf '%s\n' "$nverr" | grep -m1 -i error)" \
                    "CUDA backend, GPU benchmark and every device row; optional" \
                    "check the CUDA installation (make cuda shows the full error)" nvcc
            fi
        fi
    elif user_chose "${NVCC_ORIGIN:-}" "$NVCC_BIN" nvcc; then
        row absent "nvcc" "NVCC=$NVCC_BIN (set on the ${NVCC_ORIGIN}) not found" \
            "CUDA backend, GPU benchmark and every device row; optional" \
            "unset NVCC, or set it to an installed nvcc" nvcc
    else
        row absent "nvcc" "$NVCC_BIN not found" \
            "CUDA backend, GPU benchmark and every device row; optional" "$(hint nvcc)" nvcc
    fi
    if [ -n "$BOOST_GPU_SEL" ]; then
        row found "Boost (GPU row)" "$BOOST_GPU_SEL" "boost-gpu-<version> row"
    else
        local gadd
        case "$BOOST_GPU_NEED" in
            unset) gadd="unset BOOST_ROOT, or BOOST_ROOT=<dir holding boost/> with Boost >= 1.87" ;;
            *)     gadd="make externals   (or make boost-headers BOOST_VERSIONS=1.87.0: headers only, beside the Boost found; the CPU row keeps its own)" ;;
        esac
        row absent "Boost (GPU row)" "$BOOST_GPU_NOTE" "boost-gpu row; optional" "$gadd" boost-gpu
    fi
    if [ -e references/build/kaust/kaust_derived.h ]; then
        row found "KAUST device rows" "references/kaust_logbesselk/kaust_port.c compiled as CUDA" "kaust GPU rows"
    else
        row absent "KAUST device rows" "not ported" "kaust GPU rows; optional" \
            "make kaust   (not part of this distribution; as for the CPU rows)" kaust-dev
    fi
}

generators() {
    section "figure and table generators (make paper-figures, paper-tables, paper-tables-check)"
    local chosen=${PYTHON:-} p sel=""
    for p in $PYTHON_CANDIDATES; do
        local why
        if ! command -v "$p" >/dev/null 2>&1; then
            why="not found"
        elif ! "$p" -c 'import numpy' >/dev/null 2>&1; then
            why="$(command -v "$p") $("$p" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null): no numpy"
        elif ! "$p" -c 'import matplotlib' >/dev/null 2>&1; then
            why="$(command -v "$p") $("$p" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null): no matplotlib"
        else
            why="$(command -v "$p") $("$p" -c 'import sys,numpy,matplotlib; print(sys.version.split()[0], "numpy", numpy.__version__, "matplotlib", matplotlib.__version__)' 2>/dev/null)"
            [ -n "$sel" ] || sel=$p
        fi
        printf "  %-8s %-26s %s\n" "" "candidate $p" "$why"
    done
    # The interpreter the Makefile uses: PYTHON as the user set it, else the first candidate
    # above that imports both, else python3 (the Makefile's search and its fallback).
    local first user=no
    if is_user_set "${PYTHON_ORIGIN:-}"; then
        user=yes; first=${chosen%% *}; [ -n "$first" ] || first=python3
        printf "  %-8s %-26s %s\n" "" "PYTHON (Makefile)" "$chosen   (set on the ${PYTHON_ORIGIN}; the candidates are not searched)"
    else
        first=${sel:-python3}
        printf "  %-8s %-26s %s\n" "" "PYTHON (Makefile)" "$first -B   ($([ -n "$sel" ] && echo "the first candidate that imports both" || echo "no candidate imports both"); override: make PYTHON=<interpreter>)"
    fi
    if ! command -v "$first" >/dev/null 2>&1; then
        if user_chose "${PYTHON_ORIGIN:-}" "$chosen" python3 python /usr/bin/python3; then
            row MISSING "Python 3" "PYTHON=$chosen (set on the ${PYTHON_ORIGIN}) not found" "figures and tables, make check (audit, readme-example)" \
                "unset PYTHON, or set it to an installed interpreter" python
        else
            row MISSING "Python 3" "$first not found" "figures and tables, make check (audit, readme-example)" "$(hint python)" python
        fi
    elif "$first" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
        row found "NumPy, Matplotlib" "$("$first" -c 'import numpy,matplotlib; print("numpy", numpy.__version__, "matplotlib", matplotlib.__version__)') ($first)" \
            "figures and tables"
    else
        # pip only where the interpreter accepts it (PEP 668)
        local pip="" mods em=no
        if externally_managed "$first"; then em=yes; else pip="$first -m pip install numpy matplotlib"; fi
        if [ $user = yes ]; then
            row MISSING "NumPy, Matplotlib" "not importable by $first (PYTHON, set on the ${PYTHON_ORIGIN})" \
                "figures and tables; without them make paper-figures and make paper-tables stop" \
                "unset PYTHON, then make externals   (installs NumPy and Matplotlib into .venv/, which the Makefile then selects)${pip:+   or: $pip}" \
                python "NumPy, Matplotlib ($first)"
        else
            row MISSING "NumPy, Matplotlib" "not importable by $first$([ -z "$sel" ] && echo " (no candidate imports both)")" \
                "figures and tables; without them make paper-figures and make paper-tables stop" \
                "make externals   (installs NumPy and Matplotlib into .venv/, which the Makefile then selects)${pip:+   or: $pip}" \
                python "NumPy, Matplotlib ($first)"
            mods=$(hint pymods "$first")
            if [ -n "$mods" ] && [ "$mods" != "$pip" ]; then
                printf "  %-8s %-26s %s\n" "" "" "  or: $mods"
                printf "  %-8s %-26s %s\n" "" "" "      the Makefile then selects the first candidate that imports both"
            fi
        fi
        [ $em = yes ] && \
            printf "  %-8s %-26s %s\n" "" "" "  $first refuses pip installs (PEP 668); the .venv/ of make externals accepts them"
    fi

    if command -v pdflatex >/dev/null 2>&1; then
        local pkg miss=""
        if command -v kpsewhich >/dev/null 2>&1; then
            for pkg in geometry amsmath amssymb booktabs array float rotating; do
                kpsewhich "$pkg.sty" >/dev/null 2>&1 || miss="$miss $pkg"
            done
        fi
        if [ -z "$miss" ]; then
            row found "pdflatex" "$(version_of pdflatex --version)" \
                "the tables: widths fitted to the column, preview PDFs; make paper-tables-check"
        else
            row MISSING "LaTeX packages" "pdflatex found, missing:$miss" \
                "the tables (column fitting, previews) and make paper-tables-check; without them both fail on a LaTeX error" "$(hint pdflatex)" pdflatex
        fi
    else
        row MISSING "pdflatex" "not found" \
            "the tables (column fitting, previews) and make paper-tables-check; without it both stop (make paper-tables ARGS=--no-preview writes the tables unfitted)" \
            "$(hint pdflatex)" pdflatex
    fi
}

corpus() {
    local d=${DATA:-data} m dd="" rec
    d=${d%/}; [ "$d" = data ] || dd=" DATA=$d"
    # The record id is set once, in tools/corpus.sh.
    rec=$(sed -n 's/^RECORD=//p' tools/corpus.sh | head -1)
    section "evaluation corpus ($d/; published on Zenodo, doi:10.5281/zenodo.$rec)"
    if m=$(DATA="$d" sh tools/corpus.sh missing); then
        row found "corpus" "all files of data/MANIFEST.md present in $d/ (make corpus-check$dd verifies them)" \
            "every benchmark stage"
    elif DATA="$d" sh tools/corpus.sh need requirements tracked </dev/null >/dev/null 2>&1; then
        row MISSING "corpus" "$m: the 2^25-point files" \
            "the size laws, the ladders and the thread stage; STAGES=rest runs on the small files" \
            "make corpus$dd   (what it is, its size and download time; make corpus DOWNLOAD=yes$dd downloads it)"
    else
        row MISSING "corpus" "$m" "every benchmark stage (the benchmark commands ask before they start)" \
            "make corpus$dd   (what it is, its size and download time; make corpus DOWNLOAD=yes$dd downloads it)"
    fi
}

gpu() {
    section "GPU (optional: make cuda, benchmark-gpu)"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        row found "GPU" "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)" "GPU stages"
    else
        row absent "GPU" "no NVIDIA GPU visible (nvidia-smi)" "GPU stages; optional (nvcc and the device suppliers: above)" "" gpu
    fi
}

echo "requirements ($SECTION) -- package manager: $PM"
case "$SECTION" in
    suppliers) suppliers ;;
    all)       toolchain; suppliers; generators; corpus; gpu ;;
    missing)   toolchain; suppliers; generators; gpu; exit 0 ;;
esac
echo
if [ "$MISSING" -eq 0 ]; then
    echo "nothing to add."
else
    echo "$MISSING item(s) absent; each carries an 'add:' line. Lines marked MISSING stop a target;"
    echo "lines marked absent only remove rows or an optional check."
fi
exit 0
