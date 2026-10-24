#!/bin/bash
# fetch_boost.sh [1.75.0] [1.87.0] -- fetches the Boost headers of the CPU row (1.75.0)
# and the GPU row (1.87.0), default both, into references/boost/<version>/boost/ (make
# boost-headers), checking the archives against the SHA-256 digests below; nothing is built.
set -u
cd "$(dirname "$0")" || exit 1

PINNED="
1.75.0|953db31e016db7bb207f11432bef7df100516eeb746843fa0486a222e3fd49cb
1.87.0|af57be25cb4c4f4b413ed692fe378affb4352ea50fbe294a11ef548f4d527d89
"
BASE=https://archives.boost.io/release

[ "$#" -gt 0 ] || set -- 1.75.0 1.87.0

if command -v curl >/dev/null 2>&1; then
    get() { curl -sSfL --connect-timeout 20 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    get() { wget -q --connect-timeout=20 --tries=1 -O "$2" "$1"; }
else
    echo "fetch_boost.sh: neither curl nor wget is available" >&2; exit 1
fi
command -v bzip2 >/dev/null 2>&1 || { echo "fetch_boost.sh: bzip2 is needed to unpack the archive" >&2; exit 1; }

fail=0
for v in "$@"; do
    want=$(printf '%s\n' "$PINNED" | sed -n "s/^$v|//p")
    if [ -z "$want" ]; then
        echo "  invalid   $v (1.75.0 is the CPU row's Boost, 1.87.0 the GPU row's)"; fail=1; continue
    fi
    dest=boost/$v
    # A tree in place is kept; its archive digest cannot be checked once unpacked, so its
    # BOOST_VERSION is checked instead, read as benchmark/tools/boost_detect.sh reads it.
    IFS=. read -r v_major v_minor v_patch <<< "$v"
    need=$((v_major * 100000 + v_minor * 100 + v_patch))
    if [ -f "$dest/boost/version.hpp" ]; then
        have=$(sed -n 's/^#define BOOST_VERSION  *\([0-9]*\).*/\1/p' "$dest/boost/version.hpp" | head -1)
        if [ "$have" = "$need" ]; then
            echo "  present   references/$dest (BOOST_VERSION $have)"
        else
            echo "  mismatch  references/$dest holds BOOST_VERSION ${have:-none}, not $need;" \
                 "remove it to fetch $v"
            fail=1
        fi
        continue
    fi
    name=boost_${v//./_}
    tmp=$(mktemp -d) || exit 1
    echo "  fetching  $BASE/$v/source/$name.tar.bz2"
    echo "            (120-130 MB, then the headers are unpacked; depending on the connection"
    echo "            and the load on the server this can take several minutes, without output)"
    if ! get "$BASE/$v/source/$name.tar.bz2" "$tmp/a.tar.bz2"; then
        echo "  failed    download of $v"; rm -rf "$tmp"; fail=1; continue
    fi
    got=$(sha256sum "$tmp/a.tar.bz2" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "  mismatch  $name.tar.bz2"; rm -rf "$tmp"; fail=1; continue
    fi
    mkdir -p "$dest"
    if tar -xjf "$tmp/a.tar.bz2" -C "$tmp" "$name/boost" "$name/LICENSE_1_0.txt" &&
       mv "$tmp/$name/boost" "$dest/boost" && mv "$tmp/$name/LICENSE_1_0.txt" "$dest/"; then
        echo "  verified  references/$dest (sha256)"
    else
        echo "  failed    unpacking $v"; rm -rf "$dest"; fail=1
    fi
    rm -rf "$tmp"
done
exit $fail
