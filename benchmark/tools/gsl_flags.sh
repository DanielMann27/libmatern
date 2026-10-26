#!/bin/bash
# gsl_flags.sh --exists|--cflags|--libs|--modversion|--where|--library -- the GSL the build
# uses, for references/Makefile, tools/requirements.sh and tools/externals.sh: the highest one
# built by make externals (references/gsl/<version>/, no pkg-config needed), else pkg-config's.
# --exists: 0 if found.  tools/externals.sh reads the "(pkg-config)" suffix of --where.
# --library: the build record build_cpu_supplier_gsl_library, `<file> (<origin>); <build>;
# <fused multiply-adds>`: the library file the link takes ($CC links a test program for a
# pkg-config GSL), the compiler and options of bessel_Knu.c (knu_producer, else the
# build-flags.txt of make externals) and the package of a pkg-config GSL, and the FMA count.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mode="${1:---exists}"

# fma_count FILE: "<bessel> <all>", the fused multiply-add instructions in FILE's Bessel
# functions (names starting gsl_sf_bessel_ or containing temme or steed, compiler clones such
# as .part.0 included; most of the K_nu code is among them) and in all of FILE; nothing when
# objdump finds no instructions in FILE (not an object, or LTO bytecode only).  An
# instruction counts for the symbol before it, so without a symbol table (a stripped shared
# library) the first count also takes in the local functions that follow a Bessel function.
fma_count() {
    objdump -d --no-show-raw-insn "$1" 2>/dev/null | awk '
        /^[0-9a-f]+ <[^>]*>:$/ { b = $0 !~ /@plt>:$/ && $0 ~ /<_?gsl_sf_bessel_|temme|steed/; next }
        /^ *[0-9a-f]+:\t/      { n++ }
        /\tvfn?m(add|sub)/     { all++; if (b) k++ }
        END                    { if (n) printf "%d %d\n", k, all }'
}

# knu_producer FILE: the compiler and options FILE's debugging information records for
# bessel_Knu.c (DW_AT_producer, from readelf), cut to the compiler and its -march, -mtune,
# -mfma, -mno-fma, -O and -f options; nothing without readelf or such a record.
knu_producer() {
    command -v readelf >/dev/null 2>&1 || return 0
    readelf --debug-dump=info --dwarf-depth=1 "$1" 2>/dev/null | awk '
        /DW_TAG_compile_unit/ { p = "" }
        /DW_AT_producer/ {
            p = $0; sub(/^.*DW_AT_producer[ \t]*:[ \t]*/, "", p); sub(/^\([^)]*\):[ \t]*/, "", p) }
        /DW_AT_name/ && /[\/ ]bessel_Knu\.c[ \t]*$/ {
            n = split(p, w, " "); out = ""; opt = 0
            for (i = 1; i <= n; i++) {
                if (w[i] ~ /^-/) opt = 1
                if (opt && w[i] !~ /^-(march=|mtune=|mfma$|mno-fma$|O|f)/) continue
                out = out (out == "" ? "" : " ") w[i]
            }
            if (out != "") print out
            exit
        }'
}

# library_record FILE ORIGIN BUILD: the --library line; quotes, backslashes, `$` and
# backticks are dropped, since the value becomes a C string literal on a make command line.
library_record() {
    if [ -z "$1" ] || [ ! -f "$1" ]; then
        fm="fused multiply-adds not counted (library file not found)"
    elif ! case "$(uname -m)" in x86_64|amd64) true ;; *) false ;; esac; then
        fm="fused multiply-adds not counted (x86-64 instructions only)"
    elif ! command -v objdump >/dev/null 2>&1; then
        fm="fused multiply-adds not counted (no objdump)"
    elif c=$(fma_count "$1") && [ -n "$c" ]; then
        fm="fused multiply-adds: ${c% *} in the Bessel functions, ${c#* } in the library"
    else
        fm="fused multiply-adds not counted (objdump finds no instructions in the file)"
    fi
    printf '%s (%s); %s; %s\n' "${1:-libgsl}" "$2" "$3" "$fm" | tr -d "\"'\\\\\$\`" \
        | awk '{ $1 = $1; printf "%s%s", s, $0; s = " " } END { print "" }'
}

local_dir=""
for d in $(ls -d "$HERE"/references/gsl/*/ 2>/dev/null | sort -V); do
    [ -f "$d/lib/libgsl.a" ] && [ -f "$d/include/gsl/gsl_version.h" ] && local_dir=${d%/}
done

if [ -n "$local_dir" ]; then
    case "$mode" in
        --exists)     exit 0 ;;
        --cflags)     echo "-I$local_dir/include" ;;
        --libs)       echo "$local_dir/lib/libgsl.a $local_dir/lib/libgslcblas.a -lm" ;;
        --modversion) sed -n 's/^#define GSL_VERSION *"\(.*\)"/\1/p' "$local_dir/include/gsl/gsl_version.h" | head -1 ;;
        --where)      echo "$local_dir (make externals)" ;;
        --library)
            # bessel_Knu.c as the debugging information records it, else as build-flags.txt
            # (references/fetch_gsl.sh) does: its compile command and the compiler version.
            b="build flags not recorded"
            k=$(knu_producer "$local_dir/lib/libgsl.a")
            if [ -n "$k" ]; then
                b="bessel_Knu.c compiled by $k"
            elif [ -f "$local_dir/build-flags.txt" ]; then
                cmd=$(sed -n 's/^bessel_Knu\.c  *//p' "$local_dir/build-flags.txt" | head -1)
                ver=$(sed -n 's/^compiler  *//p' "$local_dir/build-flags.txt" | head -1)
                [ -z "$cmd" ] || b="bessel_Knu.c compiled with $cmd${ver:+ ($ver)}"
            fi
            library_record "$local_dir/lib/libgsl.a" "make externals" "$b" ;;
        *) echo "gsl_flags.sh: unknown mode $mode" >&2; exit 2 ;;
    esac
    exit 0
fi

command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gsl 2>/dev/null || exit 1
case "$mode" in
    --exists)     exit 0 ;;
    --cflags)     pkg-config --cflags gsl ;;
    --libs)       pkg-config --libs gsl ;;
    --modversion) pkg-config --modversion gsl ;;
    --where)      echo "$(pkg-config --variable=includedir gsl 2>/dev/null) (pkg-config)" ;;
    --library)
        # The file the linker opens for -lgsl (its trace, -t), with symbolic links resolved,
        # the package that owns it (rpm or dpkg) and its debugging information.
        f=""
        if t=$(mktemp -d 2>/dev/null); then
            printf 'int main(void) { return 0; }\n' > "$t/m.c"
            f=$(${CC:-cc} "$t/m.c" -o "$t/m" $(pkg-config --libs gsl) -Wl,-t 2>&1 \
                | grep -oE '[^[:space:]()]*/libgsl\.(so[.0-9]*|a|[.0-9]*dylib|tbd)' | head -1)
            rm -rf "$t"
        fi
        [ -z "$f" ] || f=$(readlink -f "$f" 2>/dev/null || echo "$f")
        p=""
        if [ -n "$f" ] && command -v rpm >/dev/null 2>&1 &&
           p=$(rpm -qf --qf '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}\n' "$f" 2>/dev/null); then
            p=$(printf '%s\n' "$p" | head -1)
        elif [ -n "$f" ] && command -v dpkg >/dev/null 2>&1 && p=$(dpkg -S "$f" 2>/dev/null); then
            p=$(printf '%s\n' "$p" | head -1); p=${p%%: *}
            p="$p $(dpkg-query -W -f='${Version}' "$p" 2>/dev/null)"
        else
            p=""
        fi
        b="package ${p:-not determined}"
        k=""; [ -z "$f" ] || k=$(knu_producer "$f")
        [ -z "$k" ] || b="${p:+package $p, }bessel_Knu.c compiled by $k"
        library_record "$f" pkg-config "$b" ;;
    *) echo "gsl_flags.sh: unknown mode $mode" >&2; exit 2 ;;
esac
