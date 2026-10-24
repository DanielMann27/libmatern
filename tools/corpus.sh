#!/bin/sh
# corpus.sh -- the evaluation corpus from Zenodo (make corpus): notice, sha256-checked download
# into the folders of DATA, the check before a run and a comparison with the record's file
# list.  POSIX sh; needs curl or wget, and sha256sum, shasum or openssl.
set -u
ORIG=$(pwd)
cd "$(dirname "$0")/.." || exit 1
. ./tools/ask.sh

RECORD=23393956
DOI=10.5281/zenodo.$RECORD
# ZENODO_FILES_URL replaces the record's file list.
LISTING=${ZENODO_FILES_URL:-https://zenodo.org/api/records/$RECORD/files}
DATA=${DATA:-data}
DATA=${DATA%/}
# CORPUS_MANIFEST replaces data/MANIFEST.md (tests; relative to the repository root, as DATA).
MANI=${CORPUS_MANIFEST:-data/MANIFEST.md}
# the command that downloads into this root
CMD="make corpus DOWNLOAD=yes"; [ "$DATA" = data ] || CMD="$CMD DATA=$DATA"
MODE=${1:-notice}
NL='
'
[ -f "$MANI" ] || { echo "corpus.sh: no $MANI" >&2; exit 2; }

# The source, CORPUS_URL (default the Zenodo record): SRC_URL, a base URL (a file at
# <base>/<name>?download=1), or SRC_DIR, a directory of the files, flat or in their folders.
SRC_URL=""; SRC_DIR=""
case "${CORPUS_URL:-}" in
    "")                 SRC_URL="https://zenodo.org/records/$RECORD/files" ;;
    http://*|https://*) SRC_URL=${CORPUS_URL%/} ;;
    *)                  SRC_DIR=${CORPUS_URL#file://}; SRC_DIR=${SRC_DIR%/} ;;
esac

# One line per file: <sha256>  <name>
names() { grep -E '^[0-9a-f]{64}  ' "$MANI"; }
folder() {
    case "$1" in
        z_nu-*)           echo z_nu ;;
        h_nu_s-*)         echo h_nu_s ;;
        *boost_shortcut*) echo boost_shortcut ;;
        *)                echo "" ;;
    esac
}
# Size in bytes, as in benchmark/corpus_bin.h: a 32-byte header, then 8 bytes per
# point (18 in a reference file: hi, lo, bits).
size() {
    case "$1" in
        *label_seam_100000.txt) echo 1032000 ;;
        *.bin) _n=${1##*_}; _n=${_n%.bin}
               case "$1" in *-ref_*|refK_*) echo $((32 + 18 * _n)) ;;
                            *)              echo $((32 + 8 * _n)) ;; esac ;;
        *) echo 0 ;;
    esac
}
# The path of a present file: its folder first, then the corpus root.
where() {
    _f=$(folder "$1")
    if [ -n "$_f" ] && [ -f "$DATA/$_f/$1" ]; then echo "$DATA/$_f/$1"; return 0; fi
    if [ -f "$DATA/$1" ]; then echo "$DATA/$1"; return 0; fi
    return 1
}
# h_nu_s-s_* and h_nu_s-* carry the names h_nu_rho-rho_* and h_nu_rho-* in record 23219731;
# rho_where: the path of such a file in DATA/h_nu_rho/, DATA/h_nu_s/ or DATA/.
rho_name() {
    case "$1" in
        h_nu_s-s_*) echo "h_nu_rho-rho_${1#h_nu_s-s_}" ;;
        h_nu_s-*)   echo "h_nu_rho-${1#h_nu_s-}" ;;
        *)          return 1 ;;
    esac
}
rho_where() {
    _rn=$(rho_name "$1") || return 1
    for _rd in "$DATA/h_nu_rho" "$DATA/h_nu_s" "$DATA"; do
        [ -f "$_rd/$_rn" ] && { echo "$_rd/$_rn"; return 0; }
    done
    return 1
}
# src_file <name>: the file in the CORPUS_URL directory, flat or in its folder.
src_file() {
    [ -n "$SRC_DIR" ] || return 1
    _sf=$(folder "$1")
    for _sp in "$SRC_DIR/$1" "$SRC_DIR/${_sf:+$_sf/}$1"; do
        [ -f "$_sp" ] && { echo "$_sp"; return 0; }
    done
    return 1
}
files() { if [ "$1" -eq 1 ]; then echo "1 file"; else echo "$1 files"; fi; }
human() { awk -v b="$1" 'BEGIN{ if (b >= 1e12) printf "%.1f TB", b/1e12; else if (b >= 1e9) printf "%.2f GB", b/1e9;
                                else if (b >= 1e6) printf "%.0f MB", b/1e6;
                                else printf "%.0f kB", b/1e3 }'; }
bytes_of() { wc -c < "$1" | tr -d ' '; }

# spot <path> <name>: why the file is not the listed <name>, from its size and its 32-byte
# header (benchmark/corpus_bin.h, little-endian), or nothing.  The label .txt: size only.
spot() {
    [ -f "$1" ] && [ -r "$1" ] || { echo "not a readable file"; return 0; }
    case "$2" in
        *.bin) _sn=${2##*_}; _sn=${_sn%.bin}
               case "$_sn" in ""|*[!0-9]*) echo "not a corpus file name"; return 0 ;; esac
               case "$2" in *-ref_*|refK_*) _sk=2 ;; *) _sk=1 ;; esac ;;
        *)     _sn=0; _sk=0 ;;
    esac
    od -A n -t u1 -N 32 -v "$1" 2>/dev/null |
    awk -v sz="$(wc -c < "$1")" -v want="$(size "$2")" -v kind="$_sk" -v n="$_sn" '
        function u32(o) { return b[o+1] + 256 * (b[o+2] + 256 * (b[o+3] + 256 * b[o+4])) }
        { for (i = 1; i <= NF; i++) b[++nb] = $i + 0 }
        END {
            if (nb > 0 && b[1] == 60) { print "an HTML page, not the file"; exit }
            if (kind > 0 && sz + 0 < 32) { printf "%.0f bytes, shorter than the 32-byte header\n", sz; exit }
            if (want + 0 > 0 && sz + 0 != want + 0) { printf "%.0f bytes, the listed file has %.0f\n", sz, want; exit }
            if (kind == 0) exit
            split("87 77 67 79 82 80 48 49", m, " ")
            for (i = 1; i <= 8; i++) if (b[i] != m[i]) { print "not a corpus file (bad magic)"; exit }
            if (u32(8) != 1) { printf "format version %.0f, not 1\n", u32(8); exit }
            if (u32(12) != kind) { printf "kind %.0f in the header, %d for this name\n", u32(12), kind; exit }
            lo = u32(16); hi = u32(20)
            if (hi != int(n / 4294967296) || lo != n % 4294967296) {
                printf "count %.0f in the header, %.0f in the name\n", lo + 4294967296 * hi, n; exit }
            if (u32(24) > 1) { printf "text format %.0f, not 0 or 1\n", u32(24); exit }
            if (u32(28) != 0) { printf "reserved field %.0f, not 0\n", u32(28); exit }
        }'
}

# Sets N_ALL, B_ALL, B_LARGE (the 2^25 files), N_ABS, B_ABS, N_ABS_SMALL, ABSENT (a present
# file that fails spot counts as absent), DAMAGED ("<name> <reason>", N_BAD files) and RHO
# (absent files present under their h_nu_rho name; N_RHO files, B_RHO bytes).  Changes no file.
scan() {
    N_ALL=0; B_ALL=0; B_LARGE=0; N_ABS=0; B_ABS=0; N_ABS_SMALL=0; ABSENT=""; N_BAD=0; DAMAGED=""
    RHO=""; N_RHO=0; B_RHO=0
    _list=$(names | awk '{print $2}')
    for _nm in $_list; do
        _sz=$(size "$_nm")
        N_ALL=$((N_ALL + 1)); B_ALL=$((B_ALL + _sz))
        case "$_nm" in *_size_*_large_*) B_LARGE=$((B_LARGE + _sz)) ;; esac
        if _p=$(where "$_nm"); then
            _why=$(spot "$_p" "$_nm")
            [ -z "$_why" ] && continue
            N_BAD=$((N_BAD + 1)); DAMAGED="$DAMAGED$_nm $_why
"
        fi
        N_ABS=$((N_ABS + 1)); B_ABS=$((B_ABS + _sz))
        case "$_nm" in *_size_*_large_*) ;; *) N_ABS_SMALL=$((N_ABS_SMALL + 1)) ;; esac
        ABSENT="$ABSENT$_nm
"
        if _p=$(rho_where "$_nm") && [ -z "$(spot "$_p" "$_nm")" ]; then
            RHO="$RHO$_nm$NL"; N_RHO=$((N_RHO + 1)); B_RHO=$((B_RHO + _sz))
        fi
    done
}
# plan: after scan, GET lists the absent files other than RHO (N_GET files, B_GET bytes),
# the ones to download; NEED_BYTES is what they still take in DATA/, net of partial
# downloads.
plan() {
    GET=""; N_GET=0; B_GET=0
    for _pn in $ABSENT; do
        case "$NL$RHO" in *"$NL$_pn$NL"*) continue ;; esac
        GET="$GET$_pn$NL"; N_GET=$((N_GET + 1)); B_GET=$((B_GET + $(size "$_pn")))
    done
    NEED_BYTES=$B_GET
    for _pn in $GET; do
        [ -f "$DATA/$_pn.part" ] && NEED_BYTES=$((NEED_BYTES - $(bytes_of "$DATA/$_pn.part")))
    done
    [ "$NEED_BYTES" -ge 0 ] || NEED_BYTES=0
}
# damaged <prefix>: one "<prefix><name>: <reason>" line per damaged file.
damaged() { printf '%s' "$DAMAGED" | while read -r _dn _dr; do printf '%s%s: %s\n' "$1" "$_dn" "$_dr"; done; }
# headline <command>: what the command lacks.
headline() {
    _v=are; [ "$N_ABS" -eq 1 ] && _v=is
    if [ "$N_BAD" -eq 0 ]; then _w="$_v not in $DATA/"; else _w="$_v missing or damaged in $DATA/"; fi
    echo "$1 needs the corpus; $N_ABS of its $N_ALL files ($(human "$B_ABS")) $_w."
}

notice() {
    scan; plan
    cat <<EOF
Evaluation corpus -- download notice

  The benchmark scores and times against the evaluation corpus of $MANI:
  $N_ALL files, $(human "$B_ALL"), of which the 2^25-point files are $(human "$B_LARGE").
  All of them are published on Zenodo:

      doi:$DOI   https://zenodo.org/records/$RECORD
      licence CC-BY-4.0 -- cite the DOI when you use the data

  $CMD downloads the files that are missing in $DATA/,
  checks each one against the sha256 in $MANI and moves it into its folder
  ($DATA/z_nu/, $DATA/h_nu_s/, $DATA/boost_shortcut/); files already lying flat in
  $DATA/ (downloaded by hand from Zenodo), and files named h_nu_rho-* (record
  23219731), are checked and moved the same way, without a download.  A file with the
  wrong size or header counts as missing: it is renamed to <file>.bad and downloaded
  again.  An interrupted download continues where it stopped when the command is run
  again.  DATA=<dir> keeps the corpus elsewhere, e.g. on a scratch file system;
  CORPUS_URL=<URL or directory> takes the files from a mirror or a local copy.

  The download time depends on the connection and on the load of Zenodo:
  $(human "$B_LARGE") takes about $(dur "$B_LARGE" 10) at 10 MB/s and about $(dur "$B_LARGE" 1) at 1 MB/s.

EOF
    echo "state"
    printf '  %-9s %d of %d files\n' present $((N_ALL - N_ABS)) "$N_ALL"
    [ "$N_BAD" -gt 0 ] && damaged "  damaged   "
    if [ "$N_ABS" -eq 0 ]; then
        printf '  %-9s none -- the corpus is complete (make corpus-check verifies it)\n' missing
        return 0
    fi
    [ "$N_RHO" -gt 0 ] && printf '  %-9s %s (%s) named h_nu_rho-*: checked and moved into place, no download\n' \
        rename "$(files "$N_RHO")" "$(human "$B_RHO")"
    if [ "$N_GET" -gt 0 ]; then
        _dm=""; [ "$N_BAD" -gt 0 ] && _dm=" ($N_BAD of them damaged)"
        printf '  %-9s %s%s, %s: %s at 10 MB/s, %s at 1 MB/s\n' missing "$(files "$N_GET")" "$_dm" \
            "$(human "$B_GET")" "$(about "$(dur "$NEED_BYTES" 10)")" "$(about "$(dur "$NEED_BYTES" 1)")"
        printf '  %-9s %s needed in %s/, %s free\n' space "$(human "$NEED_BYTES")" "$DATA" "$(human "$(free_bytes)")"
    fi
    echo "next: $CMD"
}
# dur <bytes> <MB/s>; about <duration>: the duration with `about` unless it is `under 1 s`
about() { case $1 in under*) echo "$1" ;; *) echo "about $1" ;; esac; }
dur() { awk -v b="$1" -v r="$2" 'BEGIN{ s = b / (r * 1e6);
        if (s < 1) printf "under 1 s"; else if (s < 90) printf "%d s", s + 0.5; else if (s < 5400) printf "%d min", s/60 + 0.5;
        else printf "%.1f h", s/3600 }'; }
# Free space in the nearest existing directory of DATA.
free_bytes() {
    _d=$DATA; while [ ! -d "$_d" ] && [ "$_d" != . ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
    df -Pk "$_d" 2>/dev/null | awk 'NR==2{printf "%.0f", $4 * 1024}'
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 "$1" | awk '{print $NF}'
    else return 1; fi
}
md5_of() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then md5 -q "$1"
    elif command -v openssl >/dev/null 2>&1; then openssl dgst -md5 "$1" | awk '{print $NF}'
    else return 1; fi
}
want() { names | awk -v n="$1" '$2 == n {print $1}'; }
TTY=no; [ -t 1 ] && TTY=yes
# fetch <name> <destination>: one attempt, resuming <destination> if it exists.
fetch() {
    if [ -n "$SRC_DIR" ]; then
        _fs=$(src_file "$1") || { echo "corpus.sh: no $1 in $SRC_DIR" >&2; return 1; }
        cp "$_fs" "$2"
    elif command -v curl >/dev/null 2>&1; then
        if [ $TTY = yes ]; then _p=-#; else _p=-sS; fi
        curl -fL $_p --connect-timeout 30 --retry 2 --retry-delay 5 -C - -o "$2" "$SRC_URL/$1?download=1"
    elif command -v wget >/dev/null 2>&1; then
        if [ $TTY = yes ]; then _p=""; else _p=-q; fi
        wget $_p -c --tries=2 --timeout=30 -O "$2" "$SRC_URL/$1?download=1"
    else
        echo "corpus.sh: neither curl nor wget is available" >&2; return 1
    fi
}
# place <file> <name>: check the sha256, then move the file into its folder.
place() {
    _got=$(sha256 "$1") || { echo "corpus.sh: no sha256sum, shasum or openssl" >&2; return 1; }
    if [ "$_got" != "$(want "$2")" ]; then
        echo "  mismatch  $2: sha256 $_got, $MANI lists $(want "$2")"; return 1
    fi
    _f=$(folder "$2"); _to="$DATA/${_f:+$_f/}$2"
    [ "$1" = "$_to" ] && return 0
    # a symlink is moved as it is: chmod would change the file it points to
    mkdir -p "$DATA/$_f" && mv "$1" "$_to" && { [ -L "$_to" ] || chmod 644 "$_to"; }
}

# Why the source does not deliver (one byte is requested).  Zenodo answers 403/404/410,
# or redirects to its login, for a restricted, deleted or superseded record.
diagnose() {
    [ -n "$SRC_URL" ] || return 0
    command -v curl >/dev/null 2>&1 || return 0
    _code=$(curl -s -o /dev/null -r 0-0 --max-time 30 -w '%{http_code} %{redirect_url}' --connect-timeout 20 "$SRC_URL/$1?download=1")
    case "$_code" in
        2*) return 0 ;;
        000*) echo "  the source does not answer (network, proxy or firewall): $SRC_URL" ;;
        *login*|403*|404*|410*)
            echo "  the source answers ${_code%% *} for $1: the record may have been restricted,"
            echo "  removed or replaced by a newer version (https://zenodo.org/records/$RECORD lists its versions)."
            echo "  The files can also be downloaded by hand into $DATA/ (they are checked and moved"
            echo "  by the next $CMD), or taken from a mirror: CORPUS_URL=<URL or directory>" ;;
        *) echo "  the source answers ${_code%% *} for $1 ($SRC_URL)" ;;
    esac
}

# zenodo [name...]: the record's file list (key, size, md5) against the sizes of MANIFEST and
# the md5 of the files present in DATA; without names every listed file, then README.md,
# DATASETS.md and MANIFEST.md against those beside MANIFEST.  Reports only; returns 0.
zenodo() {
    _zt=${TMPDIR:-/tmp}/corpus_zenodo.$$
    case "$LISTING" in
        file://*) cat "${LISTING#file://}" >"$_zt" 2>/dev/null && _zc=200 || {
                      echo "zenodo: cannot read ${LISTING#file://}; nothing compared"; rm -f "$_zt"; return 0; } ;;
        *) if command -v curl >/dev/null 2>&1; then
               _zc=$(curl -fsSL --connect-timeout 20 -o "$_zt" -w '%{http_code}' "$LISTING" 2>/dev/null)
           elif command -v wget >/dev/null 2>&1; then
               wget -q -S --tries=1 --timeout=20 -O "$_zt" "$LISTING" 2>"$_zt.h"
               _zc=$(awk '$1 ~ /^HTTP\// {c = $2} END {printf "%03d", c}' "$_zt.h"); rm -f "$_zt.h"
           else
               echo "zenodo: neither curl nor wget is available; nothing compared"; return 0
           fi ;;
    esac
    case "${_zc:-000}" in
        2*) ;;
        000) echo "zenodo: no answer from $LISTING (network, proxy or firewall); nothing compared" ;;
        401|403) echo "zenodo: the record's file list is not public (HTTP $_zc from $LISTING); nothing compared" ;;
        404|410) echo "zenodo: no such record (HTTP $_zc from $LISTING); nothing compared" ;;
        *) echo "zenodo: HTTP $_zc from $LISTING; nothing compared" ;;
    esac
    case "${_zc:-000}" in 2*) ;; *) rm -f "$_zt"; return 0 ;; esac
    # "<key> <size> <md5>" per object that has a key, nested objects tracked by depth.
    _ze=$(awk '{ gsub(/[{},]/, "\n&\n"); print }' "$_zt" | awk '
        function val(t) { sub(/^[^:]*:[ \t]*/, "", t); gsub(/"/, "", t); sub(/[ \t]*$/, "", t); return t }
        $0 == "{" { d++; k[d] = ""; s[d] = "-"; m[d] = "-"; next }
        $0 == "}" { if (d > 0 && k[d] != "") print k[d], s[d], m[d]; if (d > 0) d--; next }
        /^[ \t]*"key"[ \t]*:/      { k[d] = val($0); next }
        /^[ \t]*"size"[ \t]*:/     { s[d] = val($0); next }
        /^[ \t]*"checksum"[ \t]*:/ { v = val($0); if (v ~ /^md5:/) m[d] = substr(v, 5); next }')
    rm -f "$_zt"
    [ -n "$_ze" ] || { echo "zenodo: the reply of $LISTING lists no files (not the expected JSON); nothing compared"; return 0; }
    _zm=yes; md5_of "$MANI" >/dev/null 2>&1 || _zm=no
    _zall=no; [ $# -gt 0 ] || { _zall=yes; set -- $(names | awk '{print $2}'); }
    echo "zenodo: $LISTING against $MANI and $DATA/"
    printf '  %-44s %-12s %s\n' file size md5
    _zn=0; _zs=0; _zp=0; _zmd=0
    for _nm in "$@"; do
        if [ -z "$(want "$_nm")" ]; then printf '  %-44s not listed in %s\n' "$_nm" "$MANI"; continue; fi
        _zn=$((_zn + 1))
        _e=$(printf '%s\n' "$_ze" | awk -v n="$_nm" '$1 == n {print $2, $3; exit}')
        if [ -z "$_e" ]; then printf '  %-44s not in the record\n' "$_nm"; continue; fi
        _rs=${_e% *}; _rm=${_e#* }; _ls=$(size "$_nm")
        if [ "$_rs" = "$_ls" ]; then _c1=same; _zs=$((_zs + 1)); else _c1="differs: record $_rs, listed $_ls"; fi
        if ! _p=$(where "$_nm"); then _c2="(absent here)"
        else
            _zp=$((_zp + 1))
            if [ "$_rm" = - ]; then _c2="(no md5 in the record)"
            elif [ $_zm = no ]; then _c2="(no md5sum, md5 or openssl)"
            elif [ "$(md5_of "$_p")" = "$_rm" ]; then _c2=same; _zmd=$((_zmd + 1))
            else _c2="differs from $_p"; fi
        fi
        printf '  %-44s %-12s %s\n' "$_nm" "$_c1" "$_c2"
    done
    if [ $_zall = yes ]; then
        _zd=$(dirname "$MANI")
        for _zk in README.md DATASETS.md MANIFEST.md; do
            _e=$(printf '%s\n' "$_ze" | awk -v n="$_zk" '$1 == n {print $2, $3; exit}')
            if [ -z "$_e" ]; then printf '  %-44s not in the record\n' "$_zk"; continue; fi
            _rm=${_e#* }
            if [ ! -f "$_zd/$_zk" ]; then _c2="(no $_zd/$_zk)"
            elif [ "$_rm" = - ] || [ $_zm = no ]; then _c2="(no md5 to compare)"
            elif [ "$(md5_of "$_zd/$_zk")" = "$_rm" ]; then _c2="same as $_zd/$_zk"
            else _c2="differs from $_zd/$_zk"; fi
            printf '  %-44s %-12s %s\n' "$_zk" "${_e% *}" "$_c2"
        done
        printf '%s\n' "$_ze" | while read -r _k _s _m; do
            case "$_k" in README.md|DATASETS.md|MANIFEST.md) continue ;; esac
            [ -n "$(want "$_k")" ] || printf '  %-44s %-12s %s\n' "$_k" "$_s" "(not listed in $MANI)"
        done
    fi
    echo "zenodo: $_zs of $_zn sizes as listed; $_zmd of $_zp present files with the record's md5"
    return 0
}

# source_name: where the download takes the files from, for the questions.
source_name() { if [ -n "${CORPUS_URL:-}" ]; then echo "$CORPUS_URL"; else echo Zenodo; fi; }

download() {
    scan
    sha256 "$MANI" >/dev/null || { echo "corpus.sh: no sha256sum, shasum or openssl" >&2; return 1; }
    mkdir -p "$DATA" || return 1
    # A damaged file is renamed to <file>.bad, so that it is downloaded again.
    printf '%s' "$DAMAGED" | while read -r _dn _dr; do
        _p=$(where "$_dn") || continue
        if mv -f "$_p" "$_p.bad"; then echo "  damaged   $_dn ($_dr): renamed to $_p.bad"
        else echo "  damaged   $_dn ($_dr): could not be renamed to $_p.bad"; fi
    done
    # Files lying flat in the root (a hand download) are checked and moved first.
    _moved=0; _bad=0
    for _nm in $(names | awk '{print $2}'); do
        _f=$(folder "$_nm")
        [ -n "$_f" ] && [ -f "$DATA/$_nm" ] && [ ! -f "$DATA/$_f/$_nm" ] || continue
        if place "$DATA/$_nm" "$_nm"; then echo "  moved     $_nm -> $DATA/$_f/"; _moved=$((_moved + 1))
        else _bad=1; fi
    done
    [ $_bad -eq 0 ] || { echo "$CMD: stopped -- a file in $DATA/ is not the listed one;"
                         echo "  it is left as it is: remove it (or restore it) and run again"; return 1; }
    # Files named h_nu_rho-* take their listed name when the sha256 matches.
    for _nm in $RHO; do
        _p=$(rho_where "$_nm") || continue
        _f=$(folder "$_nm")
        if [ "$(sha256 "$_p")" = "$(want "$_nm")" ] && mkdir -p "$DATA/$_f" && mv "$_p" "$DATA/$_f/$_nm" \
           && { [ -L "$DATA/$_f/$_nm" ] || chmod 644 "$DATA/$_f/$_nm"; }; then
            echo "  moved     $(basename "$_p") -> $DATA/$_f/$_nm"; _moved=$((_moved + 1))
        else
            echo "  kept      $_p: sha256 is not that of $_nm in $MANI"
        fi
    done
    rmdir "$DATA/h_nu_rho" 2>/dev/null
    scan
    RHO=""; N_RHO=0; B_RHO=0
    if [ "$N_ABS" -eq 0 ]; then
        _mv=""; [ $_moved -gt 0 ] && _mv=" ($_moved moved into place)"
        echo "corpus: all $N_ALL files present$_mv; make corpus-check verifies them"
        return 0
    fi
    if [ -n "$SRC_URL" ] && ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        echo "corpus.sh: neither curl nor wget is available" >&2; return 1
    fi
    plan
    _free=$(free_bytes)
    if [ -n "$_free" ] && [ "$_free" -lt "$NEED_BYTES" ]; then
        echo "$CMD: $(human "$NEED_BYTES") to download, $(human "$_free") free in $DATA/ -- stopped"
        return 1
    fi
    echo "downloading $(files "$N_GET"), $(human "$B_GET"), from ${CORPUS_URL:-doi:$DOI}"
    _i=0; _fail=""
    for _nm in $GET; do
        _i=$((_i + 1)); _part="$DATA/$_nm.part"; _f=$(folder "$_nm")
        echo "  [$_i/$N_GET] $_nm ($(human "$(size "$_nm")"))"
        _ok=no
        for _try in 1 2 3; do
            # A complete .part is not fetched again.
            if [ -f "$_part" ] && [ "$(bytes_of "$_part")" -ge "$(size "$_nm")" ]; then _ok=yes; break; fi
            fetch "$_nm" "$_part" && { _ok=yes; break; }
            # A local directory is tried once.
            [ -z "$SRC_DIR" ] || break
            echo "            attempt $_try failed; the next one resumes"; sleep 5
        done
        if [ $_ok = yes ]; then
            _why=$(spot "$_part" "$_nm")
            if [ -n "$_why" ]; then
                echo "  refused   $_nm: $_why"
                rm -f "$_part"; _fail="$_fail $_nm"; continue
            fi
        fi
        if [ $_ok = yes ] && place "$_part" "$_nm"; then
            _rb=""
            for _b in "$DATA/${_f:+$_f/}$_nm.bad" "$DATA/$_nm.bad"; do
                [ -f "$_b" ] && rm -f "$_b" && _rb="; $_nm.bad removed"
            done
            echo "  verified  $_nm -> $DATA/$_f/$_rb"
        else
            # A part with the wrong content cannot be resumed into the right one.
            [ $_ok = yes ] && rm -f "$_part"
            _fail="$_fail $_nm"
        fi
    done
    scan
    if [ -n "$_fail" ] || [ "$N_ABS" -gt 0 ]; then
        [ -n "$_fail" ] || _fail=" $(echo $ABSENT)"
        echo "$CMD: FAILED for:$_fail"
        echo "  still missing or damaged in $DATA/: $(files "$N_ABS") ($(human "$B_ABS"))"
        [ "$N_BAD" -gt 0 ] && damaged "  damaged   "
        diagnose $_fail
        # The record's own sizes and checksums, when the source is Zenodo or a list is given.
        { [ -n "$SRC_URL" ] || [ -n "${ZENODO_FILES_URL:-}" ]; } && zenodo $_fail | sed 's/^/  /'
        echo "  run it again: a partial download continues where it stopped"
        return 1
    fi
    if [ "$_i" -eq 1 ]; then _m="the downloaded file matches"; else _m="the $_i downloaded files match"; fi
    _mv=""; [ $_moved -gt 0 ] && _mv=", $_moved moved into place"
    echo "corpus: all $N_ALL files present; $_m $MANI$_mv"
}

missing() {
    scan
    [ "${1:-}" = --quiet ] || {
        if [ "$N_ABS" -eq 0 ]; then echo "complete"
        elif [ "$N_BAD" -eq 0 ]; then echo "$(files "$N_ABS") ($(human "$B_ABS")) missing"
        else echo "$(files "$N_ABS") ($(human "$B_ABS")) missing or damaged"; fi; }
    [ "$N_ABS" -eq 0 ]
}

# question: the download question for the state of scan and plan.
question() {
    if [ "$N_GET" -eq 0 ]; then echo "Move the $(files "$N_RHO") named h_nu_rho-* into place (no download)"
    else echo "Download the missing $(files "$N_GET") ($(human "$B_GET")) from $(source_name)"; fi
}

# need <command> [tracked]: with tracked, a run that reads no 2^25-point file passes
# when only those are absent.
need() {
    scan
    [ "$N_ABS" -eq 0 ] && return 0
    if [ "${2:-}" = tracked ] && [ "$N_ABS_SMALL" -eq 0 ]; then
        _dm=""; [ "$N_BAD" -gt 0 ] && _dm=" or are damaged"
        echo "step 0: the $(files "$N_ABS") of 2^25 points ($(human "$B_ABS")) are not in $DATA/$_dm; this run reads"
        echo "  none of them, and a law that needs them is reported missing in STATUS ($CMD gets them)."
        [ "$N_BAD" -gt 0 ] && damaged "  damaged   "
        return 0
    fi
    echo
    echo "== step 0: the evaluation corpus"
    headline "$1"
    echo
    notice
    echo
    if ask_yn "$(question) now?"; then
        download && { echo; return 0; }
        if ask_yn "Try the download again (it continues where it stopped)?" && download; then
            echo; return 0
        fi
        echo "$1: stopped -- the corpus is not complete"
        return 2
    fi
    echo "$1: stopped -- $CMD downloads the missing files (see above)"
    return 2
}

# ask <command>: the notice and the question alone; 0 for yes or when nothing is missing.
ask_corpus() {
    scan
    if [ "$N_ABS" -eq 0 ]; then echo "corpus: all $N_ALL files present in $DATA/"; return 0; fi
    headline "$1"
    echo
    notice
    echo
    ask_yn "$(question) at the start of the run?"
}

usage() {
    echo "usage: corpus.sh notice | download [--no-notice] | missing [--quiet] | need <command> [tracked]" >&2
    echo "                 | ask <command> | spot <path> [name] | zenodo [name...]" >&2
    exit 2
}

case "$MODE" in
    notice)   notice ;;
    download) [ "${2:-}" = --no-notice ] || notice; download || exit 1 ;;
    missing)  missing "${2:-}" ;;
    need)     need "${2:-this command}" "${3:-}" ;;
    ask)      ask_corpus "${2:-this command}" ;;
    spot)     [ $# -ge 2 ] || usage
              case "$2" in /*) _sp=$2 ;; *) _sp=$ORIG/$2 ;; esac
              _why=$(spot "$_sp" "${3:-$(basename "$2")}")
              [ -z "$_why" ] || { echo "$_why"; exit 1; } ;;
    zenodo)   shift; zenodo "$@" ;;
    *)        usage ;;
esac
