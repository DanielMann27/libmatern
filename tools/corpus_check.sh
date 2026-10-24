#!/bin/sh
# corpus_check.sh [DATA] [--list] -- checks the corpus files against the sha256 list
# of data/MANIFEST.md (make corpus-check); an absent file is reported, an unlisted one
# is a mismatch.  POSIX sh; needs sha256sum, shasum or openssl.
set -u
DATA=data; LIST=0
for a in "$@"; do
    case "$a" in --list) LIST=1 ;; *) DATA=$a ;; esac
done
DATA=${DATA%/}
# CORPUS_MANIFEST replaces data/MANIFEST.md (tests).
M=${CORPUS_MANIFEST:-"$(cd "$(dirname "$0")/.." && pwd)/data/MANIFEST.md"}
[ -f "$M" ] || { echo "corpus_check: no $M" >&2; exit 2; }
[ -d "$DATA" ] || { echo "corpus_check: no corpus root $DATA" >&2; exit 2; }

# The first of sha256sum, shasum and openssl, as in tools/corpus.sh.
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v openssl >/dev/null 2>&1; then
    sha256() { openssl dgst -sha256 "$1" | awk '{print $NF}'; }
else
    echo "corpus_check: no sha256sum, shasum or openssl" >&2; exit 2
fi

# One "<name> <sha256>" line per listed file, sorted by name.
list=$(grep -E '^[0-9a-f]{64}  ' "$M" | awk '{print $2, $1}' | LC_ALL=C sort)
names=$(printf '%s\n' "$list" | awk '{print $1}')

where() {
    for d in "$DATA"/z_nu "$DATA"/h_nu_s "$DATA"/boost_shortcut "$DATA"; do
        [ -e "$d/$1" ] && { printf '%s\n' "$d/$1"; return 0; }
    done
    return 1
}

# large of largeall: absent files of the 2^25-point sets; others: absent small files.
n=0; large=0; largeall=0; nother=0; others=""; bad=""
while read -r name want; do
    [ -n "$name" ] || continue
    case "$name" in *_size_random_large_*|*_size_half_large_*) big=1 ;; *) big=0 ;; esac
    [ $big = 1 ] && largeall=$((largeall+1))
    p=$(where "$name") || {
        if [ $big = 1 ]; then large=$((large+1))
        else nother=$((nother+1)); others="$others $name"; fi
        continue; }
    got=$(sha256 "$p" </dev/null)
    [ $LIST = 1 ] && printf '%s  %s\n' "$got" "$p"
    n=$((n+1))
    [ "$got" = "$want" ] || bad="$bad $name"
done <<EOF
$list
EOF
for f in "$DATA"/z_nu/* "$DATA"/h_nu_s/* "$DATA"/boost_shortcut/* "$DATA"/*.bin "$DATA"/*.txt; do
    case "$f" in *.bin|*label_seam*.txt) ;; *) continue ;; esac
    [ -e "$f" ] || continue
    b=$(basename "$f")
    printf '%s\n' "$names" | grep -Fqx -- "$b" || bad="$bad $b(not listed)"
done
# The download command, built as in tools/corpus.sh.
cmd="make corpus DOWNLOAD=yes"; [ "$DATA" = data ] || cmd="$cmd DATA=$DATA"
note=""
if [ $large -gt 0 ]; then
    if [ $large = $largeall ]; then note=" the $large files"; else note=" $large of the $largeall files"; fi
    note="$note of the 2^25-point sets size_random_large and size_half_large"
fi
if [ $nother -gt 8 ]; then others=" $nother small files"
elif [ $nother -gt 1 ]; then others=" the small files$others"
elif [ $nother = 1 ]; then others=" the small file$others"; fi
[ $nother -gt 0 ] && note="${note:+$note, and}$others"
[ -n "$note" ] && note="; not in $DATA/:$note ($cmd gets them)"
# Output contract: the --list lines ("<sha256>  <path>", C-locale name order; recorded
# by benchmark/run_benchmark.sh), then this summary as the last line ("all match",
# grepped by benchmark/tests/run_tests.sh, "MISMATCH:" or "no listed file present").
# Exit status: 0 every listed file present and matching; 1 a mismatch or an unlisted
# file; 3 every present file matching, some absent; 2 no manifest, root or sha256 tool.
ok="all match (each is byte-identical to the published file, by sha256)"
[ $n = 0 ] && ok="no listed file present"
echo "corpus: $n files in $DATA/ checked against data/MANIFEST.md, ${bad:+MISMATCH:}${bad:-$ok}$note"
[ -z "$bad" ] || exit 1
[ $((large + nother)) -eq 0 ] || exit 3
