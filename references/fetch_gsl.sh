#!/bin/bash
# fetch_gsl.sh [2.6] -- fetches GSL from the first host below that answers, checks its
# SHA-256 digest and builds a static library only (so no LD_LIBRARY_PATH is needed) into
# references/gsl/<version>/ (make externals), where benchmark/tools/gsl_flags.sh finds it.
set -u
cd "$(dirname "$0")" || exit 1

PINNED="
2.6|b782339fc7a38fe17689cb39966c4d821236c28018b6593ddb6fd59ee40786a8
"
HOSTS="https://ftp.gnu.org/gnu/gsl https://ftpmirror.gnu.org/gsl
       https://mirrors.kernel.org/gnu/gsl https://ftp.fau.de/gnu/gsl"

[ "$#" -gt 0 ] || set -- 2.6

if command -v curl >/dev/null 2>&1; then
    get() { curl -sSfL --connect-timeout 20 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    get() { wget -q --connect-timeout=20 --tries=1 -O "$2" "$1"; }
else
    echo "fetch_gsl.sh: neither curl nor wget is available" >&2; exit 1
fi
command -v "${CC:-cc}" >/dev/null 2>&1 || { echo "fetch_gsl.sh: no C compiler (${CC:-cc}) to build GSL" >&2; exit 1; }
command -v make >/dev/null 2>&1 || { echo "fetch_gsl.sh: make is needed to build GSL" >&2; exit 1; }

jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
fail=0
for v in "$@"; do
    want=$(printf '%s\n' "$PINNED" | sed -n "s/^$v|//p")
    if [ -z "$want" ]; then
        echo "  invalid   $v (2.6 is the GSL of the published rows)"; fail=1; continue
    fi
    dest=$PWD/gsl/$v
    if [ -f "$dest/lib/pkgconfig/gsl.pc" ] && [ -f "$dest/lib/libgsl.a" ]; then
        echo "  present   references/gsl/$v"; continue
    fi
    tmp=$(mktemp -d) || exit 1
    got=no
    for base in $HOSTS; do
        echo "  fetching  $base/gsl-$v.tar.gz"
        if get "$base/gsl-$v.tar.gz" "$tmp/gsl.tar.gz"; then got=yes; break; fi
    done
    if [ $got = no ]; then
        echo "  failed    download of $v (no host answered)"; rm -rf "$tmp"; fail=1; continue
    fi
    if [ "$(sha256sum "$tmp/gsl.tar.gz" | cut -d' ' -f1)" != "$want" ]; then
        echo "  mismatch  gsl-$v.tar.gz"; rm -rf "$tmp"; fail=1; continue
    fi
    echo "  building  GSL $v (static, $jobs jobs; a few minutes)"
    log="$tmp/build.log"
    if tar -xzf "$tmp/gsl.tar.gz" -C "$tmp" &&
       ( cd "$tmp/gsl-$v" &&
         ./configure --prefix="$dest" --disable-shared --enable-static CC="${CC:-cc}" &&
         make -j"$jobs" && make install ) > "$log" 2>&1; then
        echo "  verified  references/gsl/$v (sha256; static library)"
    else
        rm -rf "$dest"
        cp "$log" "$PWD/gsl/build-$v.log" 2>/dev/null
        echo "  failed    building GSL $v; log: references/gsl/build-$v.log"; fail=1
    fi
    rm -rf "$tmp"
done
exit $fail
