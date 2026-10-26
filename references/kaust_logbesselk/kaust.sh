#!/bin/bash
# kaust.sh [notice|box|ask|given|download|clean] [--quiet] -- the KAUST CuBesselK rows: copyright
# notice (make kaust), its short form in a box, the question of make pipeline (ask; given: for KAUST=yes),
# download and port (make kaust DOWNLOAD=yes, the only mode that downloads CuBesselK), and clean.
# KAUST_PIPELINE=1 (make pipeline, after its question showed the box): download without the
# notice and with a one-line result.
set -u
cd "$(dirname "$0")/../.." || exit 1
. ./tools/ask.sh || exit 2

K=references/kaust_logbesselk
UP=$K/upstream
DOC=$K/PROVENANCE.md
HDR=references/build/kaust/kaust_derived.h
MODE=${1:-notice}
QUIET=no; [ "${2:-}" = --quiet ] && QUIET=yes
PY=${PYTHON:-python3}

# The pinned commit and the two files' paths and sha256 digests, from the PROVENANCE.md table.
COMMIT=$(grep -o -m1 '[0-9a-f]\{40\}' "$DOC")
# one line per file: <local name> <repository path> <sha256>
FILES=$(awk -F'|' '/^\| `upstream\// {
            gsub(/[` ]/, "", $2); gsub(/[` ]/, "", $3); gsub(/[` ]/, "", $5)
            sub(/^upstream\//, "", $2); print $2, $3, $5 }' "$DOC")
[ -n "$COMMIT" ] && [ "$(printf '%s\n' "$FILES" | grep -c .)" = 2 ] || {
    echo "kaust.sh: cannot read the commit and the two digests from $DOC" >&2; exit 2; }

notice_text() {
    cat <<EOF
KAUST CuBesselK -- copyright notice

  CuBesselK (Geng et al., King Abdullah University of Science and Technology) is
  published at https://github.com/stsds/CuBesselK.  Its files carry

      Copyright (c) 2025  King Abdullah University of Science and Technology
      All rights reserved.

  and the repository has no licence file.  Its README says that its CUDA functions
  may be copied into a program; that sentence is not a licence grant.  Its
  small-argument branch also carries the Chebyshev tables g1 and g2 of the GNU
  Scientific Library (GPL-3.0), so the files, and the port generated from them,
  contain GSL data as well.

  Apart from the short fragments that derive_port.py matches to find the code it
  rewrites (declarations, function signatures, single statements), no CuBesselK
  code is part of this distribution.  The KAUST rows of the benchmark need two
  files of commit ${COMMIT:0:7} (src/logbesselk.cu, include/logbesselk.h) and a port of
  them that is generated on your machine by this project's own code
  ($K/derive_port.py, kaust_port.c).
  The files are downloaded only on your explicit yes: make kaust DOWNLOAD=yes,
  KAUST=yes for make pipeline or make reproduce (after this notice in short), or
  the question of make pipeline and make reproduce, which on a terminal shows this
  notice in short and asks at their start.  Whether your use of the files is permitted is for you to
  judge.  Do not redistribute them or anything generated from them.

  make kaust DOWNLOAD=yes   downloads the two files from the public repository into
                            $UP/ (files already there with the
                            recorded sha256 are used as they are), checks their
                            sha256 and generates the port
  make kaust CLEAN=yes      deletes the generated port: the header, the KAUST objects
                            and the supplier programs built with them; the
                            downloaded files are kept
  The generated port is also deleted at the end of make pipeline and make reproduce.

EOF
}

# state of one upstream file: absent | verified | MISMATCH
file_state() {
    local name=$1 want=$2
    [ -e "$UP/$name" ] || { echo absent; return; }
    [ "$(sha256sum "$UP/$name" | cut -d' ' -f1)" = "$want" ] && echo verified || echo MISMATCH
}

state() {
    echo "state"
    while read -r name path sum; do
        case $(file_state "$name" "$sum") in
            absent)   printf '  %-28s absent\n' "upstream/$name" ;;
            verified) printf '  %-28s present, sha256 as recorded\n' "upstream/$name" ;;
            MISMATCH) printf '  %-28s present, sha256 mismatch (not the recorded file)\n' "upstream/$name" ;;
        esac
    done <<< "$FILES"
    if [ -f "$HDR" ]; then
        printf '  %-28s generated (%s)\n' "port" "$HDR"
        echo "next: build (make build/suppliers, build/suppliers_gpu); delete with make kaust CLEAN=yes"
    else
        printf '  %-28s not generated\n' "port"
        echo "next: make kaust DOWNLOAD=yes"
    fi
}

# The notice in short, framed for the question of make pipeline, with the state.
box() {
    local name path sum st=""
    while read -r name path sum; do
        case $(file_state "$name" "$sum") in
            absent)   st="$st, upstream/$name absent" ;;
            verified) st="$st, upstream/$name present (sha256 as recorded)" ;;
            MISMATCH) st="$st, upstream/$name present, sha256 mismatch" ;;
        esac
    done <<< "$FILES"
    if [ -f "$HDR" ]; then st="$st; the port is generated"; else st="$st; the port is not generated"; fi
    ask_box <<EOF
KAUST CuBesselK: copyright notice (read it before answering)

CuBesselK (Geng et al., https://github.com/stsds/CuBesselK) carries "Copyright (c) 2025 King Abdullah University of Science and Technology. All rights reserved." The repository has no licence file. Its README says that its CUDA functions may be copied into a program; that sentence is not a licence grant. Its small-argument branch also contains the Chebyshev tables g1 and g2 of the GNU Scientific Library (GPL-3.0).

CuBesselK is not part of libmatern and is not redistributed with it.

A yes downloads src/logbesselk.cu and include/logbesselk.h of commit ${COMMIT:0:7} from that repository, checks their sha256 and generates the port on this machine. The port is deleted at the end of make pipeline or make reproduce; the two files are kept. A no runs the benchmark without the KAUST rows.

Whether your use of the files is permitted is for you to judge. Do not redistribute them or anything generated from them.

State: ${st#, }.
EOF
}

# can_fetch: curl or wget is available
can_fetch() {
    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1
}

# cannot: why make kaust DOWNLOAD=yes cannot succeed here; empty when it can
cannot() {
    local name path sum need=no
    while read -r name path sum; do
        case $(file_state "$name" "$sum") in
            MISMATCH) echo "upstream/$name is not the recorded file (sha256 mismatch); remove it or restore the file of commit ${COMMIT:0:7}"
                      return ;;
            absent)   need=yes ;;
        esac
    done <<< "$FILES"
    [ $need = no ] || can_fetch || { echo "neither curl nor wget is available"; return; }
    command -v "${PY%% *}" >/dev/null 2>&1 || echo "no Python (${PY%% *}) to generate the port"
}

# ask: the box, then the question; 0 yes, 1 no (also without a terminal), 2 cannot.
# ask given: the box and the check alone, for an answer given as KAUST=yes; 0, or 2 cannot.
ask() {
    local why
    box
    why=$(cannot)
    if [ -n "$why" ]; then
        echo "KAUST: cannot download and port here: $why"
        return 2
    fi
    [ "${1:-}" = given ] && return 0
    ask_yn "Download the two CuBesselK files and generate the port at the start of the run?"
}

download() {
    if command -v curl >/dev/null 2>&1; then
        get() { curl -sSfL --connect-timeout 20 -o "$2" "$1"; }
    elif command -v wget >/dev/null 2>&1; then
        get() { wget -q --connect-timeout=20 --tries=1 -O "$2" "$1"; }
    else
        get() { echo "kaust.sh: neither curl nor wget is available" >&2; return 1; }
    fi
    mkdir -p "$UP" || return 1
    local base=https://raw.githubusercontent.com/stsds/CuBesselK/$COMMIT tmp got
    while read -r name path sum; do
        case $(file_state "$name" "$sum") in
            verified) echo "  present   upstream/$name (sha256 as recorded; not downloaded)"; continue ;;
            MISMATCH) echo "  mismatch  upstream/$name is not the recorded file; it is left as it is."
                      echo "            Remove it (or restore the file of commit ${COMMIT:0:7}) and run again."
                      return 1 ;;
        esac
        tmp=$(mktemp) || return 1
        echo "  fetching  $base/$path"
        if ! get "$base/$path" "$tmp"; then
            rm -f "$tmp"; echo "  failed    download of $path"; return 1
        fi
        got=$(sha256sum "$tmp" | cut -d' ' -f1)
        if [ "$got" != "$sum" ]; then
            rm -f "$tmp"; echo "  mismatch  downloaded $path has sha256 $got, recorded $sum"; return 1
        fi
        mv "$tmp" "$UP/$name" && chmod 644 "$UP/$name" || return 1
        echo "  verified  upstream/$name (sha256)"
    done <<< "$FILES"
    mkdir -p "$(dirname "$HDR")" || return 1
    echo "  porting   $K/derive_port.py -> $HDR"
    $PY "$K/derive_port.py" "$UP" "$HDR" || { rm -f "$HDR"; return 1; }
    [ -n "${KAUST_PIPELINE:-}" ] && { echo "  done      the KAUST rows are built with the supplier programs"; return 0; }
    echo "  done      the KAUST rows are built with the supplier programs"
    echo "            (make build/suppliers, build/threads_cpu, build/suppliers_gpu);"
    echo "            the port is deleted at the end of make pipeline or make reproduce and by"
    echo "            make kaust CLEAN=yes"
}

clean() {
    # What holds ported code: the CPU programs and the supplier build settings when
    # defines.txt names the port (they relink on every supplier build, so it is current),
    # the device program only when the device object exists.
    local f gone=()
    if grep -q -- '-DCPU_SUPPLIER_WITH_KAUST' references/build/defines.txt 2>/dev/null; then
        for f in build/suppliers build/threads_cpu references/build/defines.txt; do
            [ -e "$f" ] && gone+=("$f")
        done
    fi
    grep -q 'kaust_port.o' references/build/link.txt 2>/dev/null && gone+=(references/build/link.txt)
    [ -e build/kaust_dev.o ] && [ -e build/suppliers_gpu ] && gone+=(build/suppliers_gpu)
    for f in "$HDR" references/build/kaust_port.o build/kaust_dev.o; do
        [ -e "$f" ] && gone+=("$f")
    done
    if [ ${#gone[@]} -eq 0 ]; then
        [ $QUIET = yes ] || echo "  nothing to delete: no generated KAUST port"
        return 0
    fi
    rm -f "${gone[@]}" || return 1
    rmdir references/build/kaust 2>/dev/null || true
    for f in "${gone[@]}"; do echo "  deleted   $f"; done
    echo "  kept      $UP/ (the downloaded files)"
}

case "$MODE" in
    notice)   notice_text; state ;;
    box)      box ;;
    ask)      ask; exit $? ;;
    given)    ask given; exit $? ;;
    download) [ -n "${KAUST_PIPELINE:-}" ] || notice_text
              download || { echo "make kaust DOWNLOAD=yes: FAILED"; exit 1; } ;;
    clean)    clean ;;
    *)        echo "usage: kaust.sh [notice|box|ask|given|download|clean] [--quiet]" >&2; exit 2 ;;
esac
