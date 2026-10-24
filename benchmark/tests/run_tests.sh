#!/bin/bash
# run_tests.sh -- test suite of the benchmark pipeline; where possible a check is paired
# with a positive control that shows it can fail.
set -u
cd "$(dirname "$0")/../.." || exit 1
B=benchmark; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# Runs go to a private temporary root, not benchmark/runs/; the figure tests run a copy
# of the generator, so nothing is written into benchmark/plots/ either.
export WM_RUNS_DIR="$TMP/runs"
# run_gpu.sh defaults to LADDER_SET=large (the 2^25 ladder); the suite pins the small set
# so that no test or inherited environment starts it.
export LADDER_SET=small
# The full-run notice (benchmark/tools/full_run.sh) is for measuring commands only.
export WM_FULLRUN_NOTICE=shown
# The suite reads the whole corpus, the 2^25 files included.
DATA=data sh tools/corpus.sh need "make benchmark-test" || exit 2
# One probe of the missing external software (benchmark/tools/missing.sh) serves the suite.
_mx=$(bash benchmark/tools/missing.sh probe) && mv -f "$_mx" "$TMP/missing.txt"
export WM_MISSING_CACHE="$TMP/missing.txt"
RUNSDIR="$WM_RUNS_DIR"
export RUNSDIR
mkdir -p "$RUNSDIR"
CC=${CC:-gcc}; CFLAGS="-std=c11 -O2 -Wall -Wextra -ffp-contract=fast"
INC="-I$B -I$B/common -Iinclude"
pass=0; fail=0
# PYTHONDONTWRITEBYTECODE keeps __pycache__ out of the source tree.
export PYTHONDONTWRITEBYTECODE=1

# The CSV schema number is read from wm_metrics.h, not hardcoded.
SCHEMA=$(sed -n 's/^#define WM_SCHEMA  *\([0-9][0-9]*\).*/\1/p' \
         benchmark/common/wm_metrics.h | head -1)
[ -n "$SCHEMA" ] || { echo "cannot read WM_SCHEMA from wm_metrics.h"; exit 1; }
export SCHEMA
echo "  (schema under test: $SCHEMA)"

run(){ printf '  %-34s ' "$1"; shift; if "$@" >"$TMP/o" 2>&1; then echo "PASS"; pass=$((pass+1));
       else echo "FAIL"; sed 's/^/      /' "$TMP/o" | head -12; fail=$((fail+1)); fi; }
# skip <name> <why>: a test that cannot run here, said with what it needs.
skip(){ printf '  %-34s SKIPPED (%s)\n' "$1" "$2"; }
# report <name> <command...>: like run, but the output is printed on PASS as well.
report(){ printf '  %-34s ' "$1"; shift; if "$@" >"$TMP/o" 2>&1; then echo "PASS"; pass=$((pass+1));
          else echo "FAIL"; fail=$((fail+1)); fi; sed 's/^/      /' "$TMP/o" | head -12; }

# BK: the best backend this CPU runs (build/detect --live-backend); LIVE: all of them.  Both
# are exported for the `bash -c` bodies, which do not inherit the shell function `has`.
[ -x build/detect ] || { echo "build/detect is missing: make benchmark-build first"; exit 2; }
LIVE=""
for _b in avx512 avx2 matched_fma matched scalar; do
    [ "$(./build/detect --live-backend "$_b" 2>/dev/null)" = "$_b" ] && LIVE="${LIVE:+$LIVE }$_b"
done
BK=${LIVE%% *}
[ -n "$BK" ] || { echo "build/detect --live-backend reports no backend: make benchmark-build first"; exit 2; }
export BK LIVE
has(){ case " $LIVE " in *" $1 "*) return 0 ;; esac; return 1; }
echo "  (backend under test: $BK; this CPU runs: $LIVE)"

echo "== build =="
$CC $CFLAGS $INC -c $B/corpus_dataset.c -o "$TMP/cd.o" || { echo "corpus_dataset.c FAILED TO BUILD"; exit 1; }
echo "  corpus_dataset.c                   builds clean (-Wall -Wextra)"

echo "== unit: ULP modes =="
$CC $CFLAGS $INC -o "$TMP/t_ulp" $B/tests/test_ulpmodes.c -lm 2>/dev/null
run "19 assertions incl. binades"  "$TMP/t_ulp"

echo "== unit: corpus reader guards =="
$CC $CFLAGS $INC -o "$TMP/t_g" $B/tests/test_corpus_guards.c "$TMP/cd.o" 2>/dev/null
run "intact law loads"             "$TMP/t_g" data z_nu small_z
run "h_nu_s loads"                 "$TMP/t_g" data h_nu_s grid
run "missing law REFUSED"          bash -c '! '"$TMP/t_g"' data z_nu no_such_law 2>/dev/null'
# Positive controls: damaged copies of one law are refused, each with its reason.
CV="$TMP/cv"; $CC $CFLAGS -I$B -o "$CV" tools/corpus_convert.c 2>/dev/null
mk() { rm -rf "$TMP/$1"; mkdir -p "$TMP/$1/z_nu"; cp data/z_nu/z_nu-{nu,z,ref}_small_z_100000.bin "$TMP/$1/z_nu/"; }
mk trunc; head -c -2 data/z_nu/z_nu-ref_small_z_100000.bin > "$TMP/trunc/z_nu/z_nu-ref_small_z_100000.bin"
run "truncated ref REFUSED (ctrl)" bash -c '
    '"$TMP/t_g"' '"$TMP"'/trunc z_nu small_z 2>&1 | grep -q "the header.s 100000 records need"'
mk magic; printf X | dd of="$TMP/magic/z_nu/z_nu-ref_small_z_100000.bin" bs=1 seek=0 conv=notrunc 2>/dev/null
run "bad magic REFUSED (ctrl)"     bash -c '
    '"$TMP/t_g"' '"$TMP"'/magic z_nu small_z 2>&1 | grep -q "bad magic"'
mk kind; cp data/z_nu/z_nu-nu_small_z_100000.bin "$TMP/kind/z_nu/z_nu-ref_small_z_100000.bin"
run "axis file as ref REFUSED (ctrl)" bash -c '
    '"$TMP/t_g"' '"$TMP"'/kind z_nu small_z 2>&1 | grep -q "not a reference file"'
mk count; "$CV" to-text data/z_nu/z_nu-ref_small_z_100000.bin | head -n 99999 > "$TMP/count/r.txt"
"$CV" to-bin "$TMP/count/r.txt" "$TMP/count/z_nu/z_nu-ref_small_z_100000.bin"
run "short ref REFUSED (ctrl)"     bash -c '
    '"$TMP/t_g"' '"$TMP"'/count z_nu small_z 2>&1 | grep -q "POINT COUNTS DISAGREE"'
run "converter round trip is byte-exact" bash -c '
    "'"$CV"'" to-text data/z_nu/z_nu-ref_small_z_100000.bin > "'"$TMP"'/rt.txt" &&
    "'"$CV"'" to-bin "'"$TMP"'/rt.txt" "'"$TMP"'/rt.bin" && cmp -s "'"$TMP"'/rt.bin" data/z_nu/z_nu-ref_small_z_100000.bin'
# File names carry the family prefix, so the corpus may also sit flat in data/.
rm -rf "$TMP/flat"; mkdir -p "$TMP/flat"; cp data/z_nu/z_nu-{nu,z,ref}_small_z_100000.bin "$TMP/flat/"
run "flat layout loads"            "$TMP/t_g" "$TMP/flat" z_nu small_z
run "corpus matches MANIFEST.md"   bash tools/corpus_check.sh data
cp data/MANIFEST.md "$TMP/magic/"
run "damaged file is a MISMATCH (ctrl)" bash -c '
    ! bash tools/corpus_check.sh '"$TMP"'/magic | grep -q "all match"'
run "absent files: corpus_check exits 3" bash -c '
    d="'"$TMP"'/part"; rm -rf "$d"; mkdir -p "$d/z_nu"; cp data/MANIFEST.md "$d/"
    cp data/z_nu/z_nu-nu_small_z_100000.bin "$d/z_nu/"
    bash tools/corpus_check.sh "$d" >/dev/null 2>&1; [ $? -eq 3 ]'
run "a mismatch exits 1 (ctrl)"     bash -c '
    bash tools/corpus_check.sh '"$TMP"'/magic >/dev/null 2>&1; [ $? -eq 1 ]'
# The spot check (size and header, no sha256) that every measuring command runs first.
run "spot check passes the corpus"  bash -c 'DATA=data sh tools/corpus.sh missing >/dev/null'
run "spot: truncated file (ctrl)"   bash -c '
    sh tools/corpus.sh spot '"$TMP"'/trunc/z_nu/z_nu-ref_small_z_100000.bin | grep -q "the listed file has"'
run "spot: bad magic (ctrl)"        bash -c '
    sh tools/corpus.sh spot '"$TMP"'/magic/z_nu/z_nu-ref_small_z_100000.bin | grep -q "bad magic"'
run "spot: HTML page (ctrl)"        bash -c '
    printf "<!DOCTYPE html>\n" > '"$TMP"'/z_nu-z_seam_100000.bin
    sh tools/corpus.sh spot '"$TMP"'/z_nu-z_seam_100000.bin | grep -q "HTML page"'

echo "== unit: NUMA binding (degrades, never fails) =="
cat > "$TMP/tn.c" <<'X'
#include "numa_bind.h"
#include <stdio.h>
int main(int c,char**v){char w[256];int nn=0;int b=(c>1)?atoi(v[1]):wm_best_node(&nn);
 int st=wm_bind_node(b<0?0:b,w,sizeof w);
 printf("nodes=%d node=%d st=%d %s\n",nn,b,st,w); return 0;}
X
$CC $CFLAGS $INC -o "$TMP/tn" "$TMP/tn.c" 2>/dev/null
run "auto-selects a node"          "$TMP/tn"
run "nonexistent node -> unpinned" bash -c '"'"$TMP/tn"'" 99 | grep -q UNPINNED'
run "restricted mask still runs"   bash -c 'taskset -c 0-1 "'"$TMP/tn"'" >/dev/null'

echo "== stage 1: detect =="
# detect calls internal entries, so it links build/libmatern_internal.a.
$CC $CFLAGS $INC -o "$TMP/det" $B/detect.c "$TMP/cd.o" build/libmatern_internal.a -lm 2>/dev/null
run "runs and exits 0"             "$TMP/det"
run "reports both families"        bash -c '"'"$TMP/det"'" | grep -qE "h_nu_s +[0-9]+ datasets"'
run "refuses a bad corpus root"    bash -c '! "'"$TMP/det"'" /nonexistent >/dev/null 2>&1'
$CC -std=c11 -O2 -mfma -ffp-contract=fast $INC -o "$TMP/dF" $B/detect.c "$TMP/cd.o" build/libmatern_internal.a -lm 2>/dev/null
$CC -std=c11 -O2 -mfma -ffp-contract=off  $INC -o "$TMP/dN" $B/detect.c "$TMP/cd.o" build/libmatern_internal.a -lm 2>/dev/null
run "fp-contract probe: sees FUSED"     bash -c '"'"$TMP/dF"'" | grep -q "FUSED (a\*b+c"'
run "fp-contract probe: sees NOT FUSED" bash -c '"'"$TMP/dN"'" | grep -q "NOT FUSED, and the target HAS fma"'

echo "== unit: metrics =="
$CC $CFLAGS $INC -o "$TMP/t_m" $B/tests/test_metrics.c -lm 2>/dev/null
run "metrics unit assertions"      "$TMP/t_m"

echo "== stage 2: benchmark_cpu =="
$CC $CFLAGS $INC -o "$TMP/bcpu" $B/benchmark_cpu.c "$TMP/cd.o" build/libmatern.a -lm 2>/dev/null
run "accuracy runs"                bash -c '"'"$TMP/bcpu"'" --family z_nu --law grid --backend $BK --metric accuracy >/dev/null'
# Each backend runs in a re-executed process; at least three ULP maxima must differ.
DIFFER=""; for _b in scalar matched avx2 avx512; do has "$_b" && DIFFER="${DIFFER:+$DIFFER,}$_b"; done
export DIFFER
if has avx2 || has avx512; then
run "backends differ (re-exec)"    bash -c '
    n=$("'"$TMP/bcpu"'" --family z_nu --law grid --backend "$DIFFER"         --metric accuracy 2>/dev/null | grep " ULP " | grep -o "max=[0-9.]*" | sort -u | wc -l)
    [ "$n" -ge 3 ]'
else skip "backends differ (re-exec)" "needs avx2 or avx512; this CPU runs $LIVE"; fi
# The published value is not a bound: the check fails only on a missing or non-finite row.
max_ulp(){ "$TMP/bcpu" --family z_nu --law "$1" --backend avx512 --metric accuracy 2>/dev/null |
           grep " ULP " | grep -o "max=[^ ]*" | head -1 | sed 's/^max=//'; }
vs_recorded(){ printf 'recorded %s, measured %s\n' "$2" "${1:-none}"
            case $1 in ""|*[!0-9.]*) return 1 ;; esac; }
if has avx512; then
report "grid maximum (avx512)"     vs_recorded "$(max_ulp grid)" 12.4813
report "large_nu maximum (avx512)" vs_recorded "$(max_ulp large_nu)" 25.1282
else
skip "grid maximum (avx512)"     "the recorded value is the avx512 backend's; this CPU runs $BK"
skip "large_nu maximum (avx512)" "the recorded value is the avx512 backend's; this CPU runs $BK"
fi
run "all 4 ULP modes agree"        bash -c '
    n=$(for m in up down muller straddle; do
          "'"$TMP/bcpu"'" --family z_nu --law large_nu --backend $BK --ulp-mode $m             --metric accuracy 2>/dev/null | grep " ULP " | grep -o "max=[0-9.]*"
        done | sort -u | wc -l)
    [ "$n" -eq 1 ]'
run "csv carries the schema"       bash -c '"'"$TMP/bcpu"'" --family z_nu --law grid --backend $BK         --metric accuracy --format csv 2>/dev/null | grep -q "^'"$SCHEMA"',cpu,-,z_nu,grid,$BK,up,"'
run "h_nu_s scores too"            bash -c '"'"$TMP/bcpu"'" --family h_nu_s --law grid --backend $BK         --metric accuracy 2>/dev/null | grep -q "h_nu_s"'
run "speed path produces ns/elem"  bash -c '"'"$TMP/bcpu"'" --family z_nu --law small_z --backend $BK         --metric speed --reps 3 2>/dev/null | grep -q "ns/elem"'
run "rejects an unknown flag"      bash -c '! "'"$TMP/bcpu"'" --bogus >/dev/null 2>&1'
run "bogus backend REFUSED"        bash -c '
    ! "'"$TMP/bcpu"'" --family z_nu --law grid --backend nosuchbackend \
        --metric accuracy --max-points 500 >/dev/null 2>&1'
run "bogus backend names the live one" bash -c '
    "'"$TMP/bcpu"'" --family z_nu --law grid --backend nosuchbackend --metric accuracy \
        --max-points 500 2>&1 | grep -q "Refusing to label"'
run "bogus law REFUSED"            bash -c '
    ! "'"$TMP/bcpu"'" --family z_nu --law nosuchlaw --backend $BK \
        --metric accuracy >/dev/null 2>&1'
run "bogus family REFUSED"         bash -c '
    ! "'"$TMP/bcpu"'" --family nosuchfam --law grid --backend $BK \
        --metric accuracy >/dev/null 2>&1'
run "failure survives the re-exec" bash -c '
    ! "'"$TMP/bcpu"'" --family z_nu --law grid --backend $BK,nosuchbackend \
        --metric accuracy --max-points 500 >/dev/null 2>&1'

echo "== stage 2: the ladder =="
run "13 rungs, 2^8..2^20"          bash -c '
    n=$("'"$TMP/bcpu"'" --family z_nu --law size_half --backend $BK --metric speed \
        --ladder --reps 3 --format csv 2>/dev/null | grep -c "^'"$SCHEMA"',cpu,-,z_nu,size_half")
    [ "$n" -eq 13 ]'
run "constant-nu law: quantiles FLAT" bash -c '
    n=$("'"$TMP/bcpu"'" --family z_nu --law size_half --backend $BK --metric speed \
        --ladder --reps 3 --format csv 2>/dev/null | tail -n +2 | cut -d, -f13 | sort -u | wc -l)
    [ "$n" -eq 1 ]'
# The setup cost is per call, so ns_setup_per_elem * N is roughly constant along the ladder.
run "setup*N constant down the ladder" bash -c '
    "'"$TMP/bcpu"'" --family z_nu --law size_half --backend $BK --metric speed \
        --ladder --reps 5 --format csv 2>/dev/null | awk -F, "NR>1 && \$7<=8192 {print \$10*\$7}" |
    awk "{s+=\$1; a[NR]=\$1} END{if(NR<4) exit 2; m=s/NR;
          for(i=1;i<=NR;i++) if(a[i]<0.4*m||a[i]>2.5*m) exit 1}"'
run "csv header matches values"    bash -c '
    h=$("'"$TMP/bcpu"'" --family z_nu --law size_half --backend $BK --metric speed \
        --ladder --reps 3 --format csv 2>/dev/null | head -1 | tr , "\n" | wc -l)
    v=$("'"$TMP/bcpu"'" --family z_nu --law size_half --backend $BK --metric speed \
        --ladder --reps 3 --format csv 2>/dev/null | sed -n 2p | tr , "\n" | wc -l)
    [ "$h" -eq "$v" ] && [ "$h" -eq 36 ]'
# The 36-column count is pinned, so a column added to header and row is still noticed.

echo "== structural: (z, nu) validity, short AND vector =="
# On the (z, nu) entry scale aliases nu, so validity has no separate scale > 0 test.
$CC $CFLAGS $INC -o "$TMP/tv" $B/tests/test_entry_z_validity.c build/libmatern.a -lm 2>/dev/null
for b in avx512 avx2 matched scalar; do
  if has "$b"; then
  run "validity intact ($b)"       bash -c 'MATERN_BACKEND='"$b"' "'"$TMP/tv"'" >/dev/null'
  else skip "validity intact ($b)" "this CPU cannot run $b"; fi
done

echo "== structural: short calls (n < vector width) =="
$CC $CFLAGS $INC -o "$TMP/tail" $B/tests/test_tail_structural.c build/libmatern.a -lm 2>/dev/null
# One output row per entry, checked for each of avx512, avx2, matched, scalar this CPU runs.
SHORT=""; for _b in avx512 avx2 matched scalar; do
    if has "$_b"; then SHORT="${SHORT:+$SHORT }$_b"; else skip "short calls ($_b)" "this CPU cannot run $_b"; fi
done
export SHORT
run "covariance (h,nu,s): 0 nonfin" bash -c '
    for b in $SHORT; do
        MATERN_BACKEND=$b "'"$TMP/tail"'" | grep "covariance  " | grep -q "non-finite 0 " || exit 1
    done'
run "covariance_z: 0 nonfin"       bash -c '
    for b in $SHORT; do
        MATERN_BACKEND=$b "'"$TMP/tail"'" | grep "covariance_z" | grep -q "non-finite 0 " || exit 1
    done'

echo "== stage 3: the run record =="
# A reduced run (2 backends, 3 repetitions, one ladder law): run record and status.
run "a run completes"              bash -c '
    FAMILIES=z_nu,h_nu_s BACKENDS=$BK,scalar REPS=3 LADDER_LAWS=size_half \
        bash benchmark/run_benchmark.sh >/dev/null 2>&1'
LAST=$(ls -td $RUNSDIR/*/ 2>/dev/null | head -1)
run "writes STATUS complete"       bash -c '[ -n "'"$LAST"'" ] && grep -q "^STATUS   complete" "'"$LAST"'/STATUS"'
run "40 accuracy rows (2 fam x 10 laws x 2 bk)" bash -c '[ "$(grep -c "^'"$SCHEMA"'," "'"$LAST"'/accuracy.csv")" -eq 40 ]'
run "52 ladder rows (13 rungs x 2 fam x 2 bk)" bash -c '[ "$(grep -c "^'"$SCHEMA"'," "'"$LAST"'/ladder.csv")" -eq 52 ]'
run "exactly one csv header"       bash -c '[ "$(grep -c "^schema," "'"$LAST"'/accuracy.csv")" -eq 1 ]'
run "header is line 1 (fork flush)" bash -c 'head -1 "'"$LAST"'/accuracy.csv" | grep -q "^schema,"'
run "BOTH families present"        bash -c '
    [ "$(tail -n +2 "'"$LAST"'/accuracy.csv" | cut -d, -f4 | sort -u | tr "\n" " ")" = "h_nu_s z_nu " ]'
run "both backends present"        bash -c '
    [ "$(tail -n +2 "'"$LAST"'/accuracy.csv" | cut -d, -f6 | sort -u | wc -l)" -eq 2 ]'
run "provenance records per-object fma" bash -c 'grep -q "^fma_insns_vector_avx512" "'"$LAST"'/PROVENANCE.txt"'
run "provenance records load before+after" bash -c '
    grep -q "^load_before" "'"$LAST"'/PROVENANCE.txt" && grep -q "^load_after" "'"$LAST"'/PROVENANCE.txt"'
# Toolchain keys come from the artefacts and build/BUILD_KNOBS; cc_on_path names the PATH compiler.
run "provenance records the as-compiled toolchain" bash -c '
    p="'"$LAST"'/PROVENANCE.txt"
    grep -q "^build_cpu_library_compiler "   "$p" &&
    grep -q "^build_cpu_benchmark_compiler " "$p" &&
    grep -q "^make_cc "                      "$p" &&
    grep -q "^cc_on_path "                   "$p" &&
    grep -q "^fma_knob  *FMA=o[nf]* -- from the objects" "$p" &&
    ! grep -q "^compiler "      "$p" &&
    ! grep -q "^nvcc "          "$p" &&
    ! grep -q "^boost_default " "$p"'
run "provenance records the ULP metric and the timer clock" bash -c '
    p="'"$LAST"'/PROVENANCE.txt"
    grep -q "^ulp_mode  *up$" "$p" && grep -q "^ulp_formula " "$p" &&
    grep -q "^timer_clock  *CLOCK_MONOTONIC" "$p" && grep -q "^timer_warning " "$p" &&
    grep -q "^scalar_isa " "$p"'
run "manifest checksums the corpus" bash -c 'grep -q "data/MANIFEST.md" "'"$LAST"'/MANIFEST.txt"'
run "device+supplier columns exist" bash -c '
    head -1 "'"$LAST"'/accuracy.csv" | grep -q "^schema,device,supplier,family," &&
    head -1 "'"$LAST"'/ladder.csv"   | grep -q "^schema,device,supplier,family,"'
run "library rows are cpu / supplier -" bash -c '
    [ "$(tail -n +2 "'"$LAST"'/accuracy.csv" | cut -d, -f2,3 | sort -u)" = "cpu,-" ]'
# EXPECTED.txt holds avx512 values only; the check fails only on a missing or non-finite u_max.
if has avx512; then
run "EXPECTED parser produced rows"  bash -c '
    [ "$(grep -c avx512 "'"$LAST"'/EXPECTED.txt")" -eq 4 ] &&
    [ ! -s "'"$LAST"'/expected.err" ]'
report "expected: recorded, measured" bash -c '
    e="'"$LAST"'/EXPECTED.txt"; f="'"$LAST"'/accuracy.csv"; bad=0
    for law in grid large_nu large_z seam; do
        rec=$(awk -v l="$law" "\$1==\"z_nu\" && \$2==l && \$3==\"avx512\" {print \$4}" "$e")
        mea=$(awk -F, -v l="$law" "NR==1 {for (i=1;i<=NF;i++) c[\$i]=i; next}
              \$c[\"family\"]==\"z_nu\" && \$c[\"law\"]==l && \$c[\"backend\"]==\"avx512\" {print \$c[\"u_max\"]}" "$f")
        echo "z_nu/$law: recorded ${rec:-none}, measured ${mea:-none}"
        case $mea in ""|*[!0-9.eE+-]*) bad=1 ;; esac
    done
    exit $bad'
else
skip "EXPECTED parser produced rows" "its rows are the avx512 backend's; this CPU runs $BK"
skip "expected: recorded, measured"  "its rows are the avx512 backend's; this CPU runs $BK"
fi
# Positive control: a failing stage must be visible in STATUS and in the exit code.
run "a failed stage -> partial + rc" bash -c '
    cp build/detect "'"$TMP"'/d.ok"
    printf "#!/bin/sh\nexit 3\n" > build/detect; chmod +x build/detect
    FAMILIES=z_nu BACKENDS=$BK REPS=3 LADDER_LAWS=size_half \
        bash benchmark/run_benchmark.sh >/dev/null 2>&1; rc=$?
    cp "'"$TMP"'/d.ok" build/detect
    r=$(ls -td $RUNSDIR/*/ | head -1)
    [ "$rc" -ne 0 ] && grep -q "^STATUS   partial" "$r/STATUS" && grep -q "missing  detect" "$r/STATUS"'

run "suite's own column indices are right" bash -c '
    a=$(head -1 "'"$LAST"'/accuracy.csv"); l=$(head -1 "'"$LAST"'/ladder.csv")
    [ "$(echo "$a" | cut -d, -f4)"  = family            ] &&
    [ "$(echo "$a" | cut -d, -f6)"  = backend           ] &&
    [ "$(echo "$a" | cut -d, -f2)"  = device            ] &&
    [ "$(echo "$a" | cut -d, -f3)"  = supplier          ] &&
    [ "$(echo "$l" | cut -d, -f7)"  = n                 ] &&
    [ "$(echo "$l" | cut -d, -f10)" = ns_setup_per_elem ] &&
    [ "$(echo "$l" | cut -d, -f11)" = reps              ] &&
    [ "$(echo "$l" | cut -d, -f13)" = nu_med            ]'

run "intact corpus: no STATUS warning" bash -c '! grep -q "^warning" "'"$LAST"'/STATUS"'
# Positive control: a byte changed in a data region passes the spot check, and the run
# names the file in a STATUS warning while it completes.
run "sha256 mismatch -> STATUS warning (ctrl)" bash -c '
    d="'"$TMP"'/flip"; rm -rf "$d"; mkdir -p "$d"; cp data/*.md "$d/"
    for f in $(cd data && find . \( -type f -o -type l \) \( -name "*.bin" -o -name "*.txt" \)); do
        mkdir -p "$d/$(dirname "$f")"; ln -s "$(readlink -f "data/$f")" "$d/$f"; done
    f=z_nu/z_nu-nu_seam_100000.bin; rm "$d/$f"; cp "data/$f" "$d/$f"
    b=$(od -A n -t u1 -j 100 -N 1 "$d/$f" | tr -d " ")
    if [ "$b" = 255 ]; then printf "\000"; else printf "\377"; fi |
        dd of="$d/$f" bs=1 seek=100 conv=notrunc 2>/dev/null
    WM_RUNS_DIR="'"$TMP"'/runs_flip" DATA="$d" STAGES=record \
        bash benchmark/run_benchmark.sh >/dev/null 2>&1
    r=$(ls -td "'"$TMP"'"/runs_flip/*/ | head -1); rm -rf "$d"
    grep -q "^STATUS   complete" "$r/STATUS" &&
    grep -q "^warning  corpus: sha256 differs from data/MANIFEST.md for z_nu-nu_seam_100000.bin" "$r/STATUS"'

echo "== stage 2c: external suppliers =="
if [ -x build/suppliers ]; then
  # The supplier rows this build lacks are listed; a test needing one is skipped, not failed.
  bash benchmark/tools/missing.sh rows "  suppliers under test" cpu "" "" "$(./build/suppliers --list)"
  SUP_ROWS=$(./build/suppliers --list | awk '/^  [a-z]/{print $1}')
  NSUP=$(printf '%s\n' "$SUP_ROWS" | grep -c .)
  has_boost=no; printf '%s\n' "$SUP_ROWS" | grep -qx 'boost-1.75.0' && has_boost=yes
  NO_BOOST=$(bash benchmark/tools/missing.sh lines boost-cpu | cut -f2)
  NO_BOOST="Boost 1.75.0 not in this build; ${NO_BOOST:-make build/suppliers}"
  run "lists suppliers"            bash -c './build/suppliers --list | grep -q "supplier(s) in this build"'
  run "bogus supplier REFUSED"     bash -c '
      ! ./build/suppliers --family z_nu --law grid --supplier nope --max-points 500 \
          >/dev/null 2>&1'
  run "bogus supplier lists the real ones" bash -c '
      ./build/suppliers --family z_nu --law grid --supplier nope --max-points 500 2>&1 \
          | grep -q "Available:"'
  # bessels.jl is always built, so the schema test runs in every build
  run "csv is schema $SCHEMA"       bash -c '
      ./build/suppliers --family z_nu --law grid --supplier bessels.jl --max-points 2000 \
          --format csv 2>/dev/null | grep -q "^'"$SCHEMA"',cpu,bessels.jl"'
  # every supplier this build lists has rows ($NSUP of them)
  run "supplier column is POPULATED" bash -c '
      v=$(./build/suppliers --family z_nu --law grid --max-points 2000 --format csv 2>/dev/null \
          | tail -n +2 | cut -d, -f3 | sort -u | grep -c .)
      [ "$v" -ge '"$NSUP"' ] && [ "$v" -ge 2 ]'
  run "library rows keep supplier=-" bash -c '
      [ "$(./build/benchmark_cpu --family z_nu --law grid --backend $BK --metric accuracy \
           --format csv 2>/dev/null | tail -n +2 | cut -d, -f3 | sort -u)" = "-" ]'
  if [ $has_boost = yes ]; then
  run "no scaled row without a scaled K" bash -c '
      # boost provides no scaled K e^z, so it emits direct and logtrick rows only
      n=$(./build/suppliers --family z_nu --law grid --supplier boost-1.75.0 --max-points 2000 \
          --format csv 2>/dev/null | awk -F, "\$6==\"direct\"{d++} \$6==\"scaled\"{s++} END{print d+0, s+0}")
      [ "$n" = "1 0" ]'
  else skip "no scaled row without a scaled K" "$NO_BOOST"; fi
  run "bessels.jl DOES get a scaled row" bash -c '
      n=$(./build/suppliers --family z_nu --law grid --supplier bessels.jl --max-points 2000 \
          --format csv 2>/dev/null | awk -F, "\$6==\"scaled\"" | wc -l)
      [ "$n" -eq 1 ]'
  if [ $has_boost = yes ]; then
  run "n_scored exposes lost points"  bash -c '
      # boost cannot represent K over much of large_nu; n_scored < n records it
      ./build/suppliers --family z_nu --law large_nu --supplier boost-1.75.0 \
          --format csv 2>/dev/null | awk -F, "\$6==\"direct\"{f=1; r=(\$9 < \$8)} END{exit !(f && r)}"'
  else skip "n_scored exposes lost points" "$NO_BOOST"; fi
  run "both families score"        bash -c '
      [ "$(./build/suppliers --family z_nu,h_nu_s --law grid --supplier bessels.jl \
           --max-points 2000 --format csv 2>/dev/null | tail -n +2 | cut -d, -f4 \
           | sort -u | tr "\n" " ")" = "h_nu_s z_nu " ]'
else
  echo "  (build/suppliers absent -- skipped; make build/suppliers builds it)"
fi

echo "== stage 2d: CUDA =="
if [ -x build/benchmark_gpu ]; then
  if ./build/benchmark_gpu --law grid --max-points 500 2>&1 | grep -q "no CUDA device"; then
    run "no device -> says why, exits 0" bash -c '
        ./build/benchmark_gpu --law grid --max-points 500 2>&1 | grep -q "not a failure" &&
        ./build/benchmark_gpu --law grid --max-points 500 >/dev/null 2>&1'
    echo "  (no GPU here: accuracy rows are absent by design, not by failure)"
  else
    run "csv says device=cuda"      bash -c '
        ./build/benchmark_gpu --family z_nu --law grid --max-points 5000 --format csv \
            2>/dev/null | grep -q "^'"$SCHEMA"',cuda,-,z_nu,grid,"'
    run "all four transfer modes"   bash -c '
        n=$(./build/benchmark_gpu --family z_nu --law grid --max-points 5000 --format csv \
            2>/dev/null | tail -n +2 | cut -d, -f6 | sort -u | wc -l); [ "$n" -eq 4 ]'
    run "modes are BIT-IDENTICAL"   bash -c '
        ! ./build/benchmark_gpu --family z_nu --law grid,large_z --max-points 20000 2>&1 \
            | grep -q "NOT bit-identical"'
    # Scorer control: large_z reaches M < DBL_MIN, where a rounded reference scores 0 as exact.
    run "large_z ULP is NOT zero (ctrl)" bash -c '
        v=$(./build/benchmark_gpu --family z_nu --law large_z --max-points 20000 --format csv \
            2>/dev/null | awk -F, "NR==2{print (\$22>0)}"); [ "$v" = 1 ]'
  fi
else
  echo "  (build/benchmark_gpu absent -- skipped; it needs nvcc: $(bash benchmark/tools/missing.sh lines nvcc | cut -f2))"
fi

echo "== stage 4: figures (parses only, measures nothing) =="
PY=${PY:-python3}
# make_paper_figures.py finds runs/ and its output from its own path, so each test runs a
# copy of benchmark/plots/ beside fixture runs; FIGMIRROR prints that scratch root.
FIGMIRROR='m=$(mktemp -d); mkdir -p "$m/runs";
    cp -r benchmark/plots "$m/plots";
    rm -rf "$m/plots/paper" "$m/plots/paper-final" "$m/plots/__pycache__";
    echo "$m"'
export FIGMIRROR
run "figures: a complete run draws" bash -c '
    m=$(bash -c "$FIGMIRROR"); cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hosta_test"
    "'"$PY"'" "$m/plots/make_paper_figures.py" >/dev/null 2>&1; rc=$?
    n=$(find "$m/plots/paper" -name "*.png" 2>/dev/null | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && [ "$n" -gt 0 ]'
# The default is the paper set (PAPER_FIGURES): no attribution figure; --full draws it.
run "figures: paper set by default, --full all" bash -c '
    m=$(bash -c "$FIGMIRROR"); cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hosta_test"
    "'"$PY"'" "$m/plots/make_paper_figures.py" >/dev/null 2>&1; rc=$?
    n=$(find "$m/plots/paper" -name "*.png" | wc -l)
    a=$(find "$m/plots/paper" -name "fig-paper-attribution-*.png" | wc -l)
    "'"$PY"'" "$m/plots/make_paper_figures.py" --full >/dev/null 2>&1; rc2=$?
    f=$(find "$m/plots/paper" -name "fig-paper-attribution-*.png" | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && [ "$rc2" -eq 0 ] && [ "$n" -gt 0 ] && [ "$a" -eq 0 ] && [ "$f" -gt 0 ]'
run "figures: --paper writes captions.tex" bash -c '
    m=$(bash -c "$FIGMIRROR"); cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hosta_test"
    "'"$PY"'" "$m/plots/make_paper_figures.py" --paper >/dev/null 2>&1; rc=$?
    n=$(find "$m/plots/paper-final" -name captions.tex 2>/dev/null | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && [ "$n" -gt 0 ]'
# A CSV of another schema is refused: its figures are skipped, the exit status is nonzero.
run "figures: refuses a schema mismatch" bash -c '
    m=$(bash -c "$FIGMIRROR"); d="$m/runs/2026-01-01T00-00Z_hosta_test"
    cp -r "'"$LAST"'" "$d"
    sed -i "s/^'"$SCHEMA"',/999,/" "$d/accuracy.csv"
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" 2>&1); rc=$?; rm -rf "$m"
    [ "$rc" -ne 0 ] && echo "$out" | grep -q "accuracy.csv is schema 999"'
# Positive control: a partial run is reported and drawn, and --strict refuses it.
run "figures: partial run, --strict REFUSES" bash -c '
    m=$(bash -c "$FIGMIRROR"); d="$m/runs/2026-01-01T00-00Z_hosta_test"
    cp -r "'"$LAST"'" "$d"
    sed -i "s/^STATUS   complete/STATUS   partial/" "$d/STATUS"
    "'"$PY"'" "$m/plots/make_paper_figures.py" --strict >/dev/null 2>&1; rc=$?
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" 2>&1); rc2=$?; rm -rf "$m"
    [ "$rc" -eq 2 ] && [ "$rc2" -eq 0 ] && echo "$out" | grep -q "runs not marked complete"'

# Partial runs plot: every figure (--full) is drawn or listed as not drawn, with exit
# status 0.
# The fixtures are inline because `bash -c` does not inherit shell functions.
run "subset: one backend only"     bash -c '
    m=$(bash -c "$FIGMIRROR"); d="$m/runs/2026-01-01T00-00Z_hosta_test"; mkdir -p "$d"
    cp "'"$LAST"'"/STATUS "'"$LAST"'"/PROVENANCE.txt "$d/" 2>/dev/null
    for f in accuracy.csv ladder.csv; do
        head -1 "'"$LAST"'/$f" > "$d/$f"
        tail -n +2 "'"$LAST"'/$f" | grep ",scalar," >> "$d/$f"
    done
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" --full 2>&1); rc=$?
    n=$(find "$m/plots/paper" -name "*.png" 2>/dev/null | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && { [ "$n" -gt 0 ] || echo "$out" | grep -q "NOT drawn"; }'
run "subset: no AVX-512 (laptop)"  bash -c '
    m=$(bash -c "$FIGMIRROR"); d="$m/runs/2026-01-01T00-00Z_hosta_test"; mkdir -p "$d"
    cp "'"$LAST"'"/STATUS "'"$LAST"'"/PROVENANCE.txt "$d/" 2>/dev/null
    for f in accuracy.csv ladder.csv; do
        head -1 "'"$LAST"'/$f" > "$d/$f"
        tail -n +2 "'"$LAST"'/$f" | grep -v ",avx512," >> "$d/$f"
    done
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" --full 2>&1); rc=$?
    n=$(find "$m/plots/paper" -name "*.png" 2>/dev/null | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && { [ "$n" -gt 0 ] || echo "$out" | grep -q "NOT drawn"; }'
run "subset: accuracy, no timing"  bash -c '
    m=$(bash -c "$FIGMIRROR"); d="$m/runs/2026-01-01T00-00Z_hosta_test"; mkdir -p "$d"
    cp "'"$LAST"'"/STATUS "'"$LAST"'"/PROVENANCE.txt "'"$LAST"'"/accuracy.csv "$d/" 2>/dev/null
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" --full 2>&1); rc=$?
    n=$(find "$m/plots/paper" -name "*.png" 2>/dev/null | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && [ "$n" -gt 0 ]'
run "subset: short ladder (3 rungs)" bash -c '
    m=$(bash -c "$FIGMIRROR"); d="$m/runs/2026-01-01T00-00Z_hosta_test"; mkdir -p "$d"
    cp "'"$LAST"'"/STATUS "'"$LAST"'"/PROVENANCE.txt "'"$LAST"'"/accuracy.csv "$d/" 2>/dev/null
    head -1 "'"$LAST"'/ladder.csv" > "$d/ladder.csv"
    awk -F, "NR>1 && (\$7==256||\$7==512||\$7==1024)" "'"$LAST"'/ladder.csv" >> "$d/ladder.csv"
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" --full 2>&1); rc=$?
    n=$(find "$m/plots/paper" -name "*.png" 2>/dev/null | wc -l); rm -rf "$m"
    [ "$rc" -eq 0 ] && [ "$n" -gt 0 ]'
# Runs are grouped into machines by CPU model, so a same-model pair is one machine.
run "same-CPU pair: machine figure NOT drawn" bash -c '
    m=$(bash -c "$FIGMIRROR")
    cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hosta_test"
    cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hostb_test"
    out=$("'"$PY"'" "$m/plots/make_paper_figures.py" 2>&1)
    n=$(ls "$m"/plots/paper/cross-machine/fig-paper-machine-independence.png 2>/dev/null | wc -l)
    rm -rf "$m"
    [ "$n" -eq 0 ] && echo "$out" | grep -q "needs 2 CPU architectures"'
# Positive control: the fixture gives the two runs different CPU models.
run "different-CPU pair: machine figure IS drawn" bash -c '
    m=$(bash -c "$FIGMIRROR")
    cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hosta_test"
    cp -r "'"$LAST"'" "$m/runs/2026-01-01T00-00Z_hostb_test"
    sed -i "s/^cpu .*/cpu               AMD EPYC 9454/" "$m/runs/2026-01-01T00-00Z_hosta_test/PROVENANCE.txt"
    sed -i "s/^cpu .*/cpu               Intel(R) Xeon 8358/" "$m/runs/2026-01-01T00-00Z_hostb_test/PROVENANCE.txt"
    "'"$PY"'" "$m/plots/make_paper_figures.py" >/dev/null 2>&1
    n=$(ls "$m"/plots/paper/cross-machine/fig-paper-machine-independence.png 2>/dev/null | wc -l)
    rm -rf "$m"
    [ "$n" -eq 1 ]'

# A format string short of specifiers compiles and misaligns the columns.
run "accuracy: header == row width" bash -c '
    o=$("'"$TMP/bcpu"'" --family z_nu --law grid --backend $BK \
        --metric accuracy --format csv 2>/dev/null)
    h=$(echo "$o" | head -1 | tr , "\n" | wc -l)
    v=$(echo "$o" | sed -n 2p | tr , "\n" | wc -l)
    [ "$h" -gt 1 ] && [ "$h" -eq "$v" ]'
if [ -x build/suppliers ]; then
run "suppliers: header == row width" bash -c '
    o=$(./build/suppliers --family z_nu --law grid --supplier bessels.jl \
        --max-points 2000 --format csv 2>/dev/null)
    h=$(echo "$o" | head -1 | tr , "\n" | wc -l)
    v=$(echo "$o" | sed -n 2p | tr , "\n" | wc -l)
    [ "$h" -eq "$v" ]'
fi
# n_file is the population before decimation; the detail sidecar must agree with its row.
run "accuracy: n_file present, n == n_file uncapped" bash -c '
    "'"$TMP/bcpu"'" --family z_nu --law grid --backend $BK --metric accuracy \
        --reps 2 --format csv 2>/dev/null \
    | "'"$PY"'" -c "import csv,sys; r=list(csv.DictReader(sys.stdin)); \
sys.exit(0 if r and all(x[\"n_file\"]==x[\"n\"] for x in r) else 1)"'
if [ -x build/suppliers ]; then
run "suppliers: a capped row shows n < n_file" bash -c '
    ./build/suppliers --family z_nu --law grid --supplier bessels.jl \
        --max-points 2000 --reps 2 --format csv 2>/dev/null \
    | "'"$PY"'" -c "import csv,sys; r=list(csv.DictReader(sys.stdin)); \
sys.exit(0 if r and all(int(x[\"n\"])<int(x[\"n_file\"]) for x in r) else 1)"'
fi
run "accuracy detail agrees with its row" bash -c '
    WM_ACC_DETAIL="'"$TMP"'/det" "'"$TMP/bcpu"'" --family z_nu,h_nu_s --law large_nu \
        --backend $BK --metric accuracy --reps 2 --format csv \
        > "'"$TMP"'/det.csv" 2>/dev/null
    "'"$PY"'" - "'"$TMP"'" <<PYEOF
import csv, sys
t = sys.argv[1]
rows = list(csv.DictReader(open(t + "/det.csv")))
D = list(csv.DictReader(open(t + "/det.$BK")))
bad = 0
for r in rows:
    k = (r["family"], r["law"], r["backend"])
    m = [d for d in D if (d["family"], d["law"], d["backend"]) == k]
    hist = sum(int(d["count"]) for d in m if d["kind"] == "hist")
    w = [d for d in m if d["kind"] == "worst"]
    ex = [d for d in m if d["kind"] == "excluded"]
    ok = (hist == int(r["n_scored"]) and w and abs(float(w[0]["ulp"])) == float(r["u_max"])
          and len(ex) == int(r["nonfinite"])
          and (r["family"] != "h_nu_s" or w[0]["aux"] != ""))
    if not ok:
        bad += 1; print("detail disagrees:", k, file=sys.stderr)
sys.exit(1 if bad or not rows else 0)
PYEOF'
# Without a device the GPU emitters' format strings are counted from the source.
run "gpu emitters: format == header" bash -c '
    "'"$PY"'" - <<PYEOF
import re, sys
bad = []
for f in ("benchmark/benchmark_gpu.cu", "benchmark/suppliers_gpu.cu"):
    try: src = open(f).read()
    except OSError: continue
    for m in re.finditer(r"printf\(\s*((?:\"[^\"]*\"\s*)+)", src):
        lit = "".join(re.findall(r"\"([^\"]*)\"", m.group(1)))
        if "schema," not in lit and "%d,cuda" not in lit: continue
        if "schema," in lit:                 # a header line
            hdr = lit.rstrip("\\n").count(",") + 1
            last_hdr = hdr
        else:                                 # the matching row line
            spec = len(re.findall(r"%[-0-9.zl]*[dsgfuzx]", lit))
            lits = lit.rstrip("\\n").count(",") + 1
            if lits != last_hdr:
                bad.append((f, last_hdr, lits))
if bad:
    for f, h, v in bad: print(f"{f}: header {h} cols, row {v} cols", file=sys.stderr)
    sys.exit(1)
PYEOF'

# The public CUDA entry and the direct launch must score alike (z derived as the kernel does).
run "gpu: raw kernel == public entry" bash -c '
    # Only a run at the current schema is checked.
    r=$(ls -td $RUNSDIR/*/ 2>/dev/null | while read d; do
            [ -s "$d/gpu.csv" ] && [ -s "$d/gpu_kernels.csv" ] || continue
            [ "$(awk -F, "NR==2{print \$1}" "$d/gpu.csv")" = "$SCHEMA" ] || continue
            echo "$d" && break
        done)
    [ -n "$r" ] || { echo "no run with both gpu.csv and gpu_kernels.csv -- skipped"; exit 0; }
    "'"$PY"'" - "$r" <<PYEOF
import csv, sys
d = sys.argv[1]
LIB = ("libmatern-cuda", "libwhittle-cuda")  # both labels occur in the recorded runs
def rows(p):
    try: return list(csv.DictReader(open(d + "/" + p)))
    except OSError: return []
ent = {(r["family"], r["law"]): r["u_max"]
       for r in rows("gpu.csv") if r.get("backend") == "sync"}
raw = {(r["family"], r["law"]): r["u_max"]
       for r in rows("gpu_kernels.csv")
       if r.get("supplier") in LIB
       and (r.get("timing_scope") or "kernel") == "kernel"}
if not raw:
    print(f"{d}: no kernel-scope libmatern-cuda rows in gpu_kernels.csv", file=sys.stderr)
    sys.exit(1)
common = [k for k in ent if k in raw]
if not common:
    print(f"{d}: no (family, law) has both a gpu.csv sync row and a libmatern-cuda kernel row",
          file=sys.stderr)
    sys.exit(1)
bad = [(k, ent[k], raw[k]) for k in common if ent[k] != raw[k]]
for k, a, b in bad:
    print(f"{k}: entry {a} != raw kernel {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PYEOF'

# Only runs whose gpu_kernels.csv carries n_file are checked.
run "gpu: call rows == public entry, per mode" bash -c '
    r=$(ls -td $RUNSDIR/*/ 2>/dev/null | while read d; do
            [ -s "$d/gpu.csv" ] && [ -s "$d/gpu_kernels.csv" ] || continue
            head -1 "$d/gpu_kernels.csv" | grep -q ",n_file" || continue
            echo "$d" && break
        done)
    [ -n "$r" ] || { echo "no post-fix run with gpu.csv and gpu_kernels.csv -- skipped"; exit 0; }
    "'"$PY"'" - "$r" <<PYEOF
import csv, sys
d = sys.argv[1]
LIB = ("libmatern-cuda", "libwhittle-cuda")  # both labels occur in the recorded runs
ent = {(r["family"], r["law"], r["backend"]): r["u_max"]
       for r in csv.DictReader(open(d + "/gpu.csv"))}
bad = n = lib = 0
for r in csv.DictReader(open(d + "/gpu_kernels.csv")):
    if r.get("supplier") not in LIB or r.get("timing_scope") != "call":
        continue
    lib += 1
    mode = r["backend"].split("/", 1)[-1]
    k = (r["family"], r["law"], mode)
    if k in ent:
        n += 1
        u = r["u_max"]
        if ent[k] != u:
            bad += 1; print(f"{k}: entry {ent[k]} != call row {u}", file=sys.stderr)
if not lib:
    print(f"{d}: no call-scope libmatern-cuda rows in gpu_kernels.csv", file=sys.stderr)
elif not n:
    print(f"{d}: no libmatern-cuda call row matches a gpu.csv (family, law, mode) row",
          file=sys.stderr)
sys.exit(1 if bad or not n else 0)
PYEOF'

# Every supplier moves the same inputs at call scope, so the kernel ordering must hold there.
run "gpu: call scope cannot invert kernel ranking" bash -c '
    r=$(ls -td $RUNSDIR/*/ 2>/dev/null | while read d; do
            [ -s "$d/gpu_kernels.csv" ] || continue
            [ "$(awk -F, "NR==2{print \$1}" "$d/gpu_kernels.csv")" = "$SCHEMA" ] || continue
            echo "$d" && break
        done)
    [ -n "$r" ] || { echo "no current-schema kernel run -- skipped"; exit 0; }
    "'"$PY"'" - "$r" <<PYEOF
import csv, sys
d = sys.argv[1]
rows = list(csv.DictReader(open(d + "/gpu_kernels.csv")))
K, C = {}, {}
for r in rows:
    if r.get("backend") != "direct": continue
    key = (r["family"], r["law"], r["supplier"])
    (K if r["timing_scope"] == "kernel" else C)[key] = float(r["ns_min"])
bad = []
for (fam, law, s1) in list(K):
    for (f2, l2, s2) in list(K):
        if (fam, law) != (f2, l2) or s1 >= s2: continue
        a, b = (fam, law, s1), (fam, law, s2)
        if a not in C or b not in C: continue
        if (K[a] < K[b]) != (C[a] < C[b]):
            bad.append((fam, law, s1, s2, K[a], K[b], C[a], C[b]))
for t in bad[:6]:
    print("inversion %s/%s: %s vs %s  kernel %.3f/%.3f  call %.3f/%.3f" % t,
          file=sys.stderr)
sys.exit(1 if bad else 0)
PYEOF'

# A transfer mode names one strategy in both files: its two timings agree within a factor 2.
run "gpu: mode names agree across files" bash -c '
    r=$(ls -td $RUNSDIR/*/ 2>/dev/null | while read d; do
            [ -s "$d/gpu.csv" ] && [ -s "$d/gpu_kernels.csv" ] || continue
            [ "$(awk -F, "NR==2{print \$1}" "$d/gpu.csv")" = "$SCHEMA" ] || continue
            echo "$d" && break
        done)
    [ -n "$r" ] || { echo "no current-schema gpu run -- skipped"; exit 0; }
    "'"$PY"'" - "$r" <<PYEOF
import csv, sys
d = sys.argv[1]
LIB = ("libmatern-cuda", "libwhittle-cuda")  # both labels occur in the recorded runs
lib = {}
for r in csv.DictReader(open(d + "/gpu.csv")):
    lib[(r["family"], r["law"], r["backend"])] = float(r["ns_min"])
bad = []
nlib = n = 0
for r in csv.DictReader(open(d + "/gpu_kernels.csv")):
    if r.get("supplier") not in LIB or r.get("timing_scope") != "call":
        continue
    nlib += 1
    mode = r["backend"].split("/")[-1]
    k = (r["family"], r["law"], mode)
    if k not in lib: continue
    n += 1
    a, b = lib[k], float(r["ns_min"])
    if max(a, b) / max(min(a, b), 1e-12) > 2.0:
        bad.append((k, a, b))
if not nlib:
    print(f"{d}: no call-scope libmatern-cuda rows in gpu_kernels.csv", file=sys.stderr)
    sys.exit(1)
if not n:
    print(f"{d}: no libmatern-cuda call row matches a gpu.csv (family, law, mode) row",
          file=sys.stderr)
    sys.exit(1)
for k, a, b in bad[:6]:
    print(f"{k}: gpu.csv {a:.3f} vs gpu_kernels.csv {b:.3f} ns/elem", file=sys.stderr)
sys.exit(1 if bad else 0)
PYEOF'

# The device tables are statically initialised: nothing uploads __constant__ memory.
run "cuda: no table upload" bash -c '
    ! grep -n "cudaMemcpyToSymbol" src/backends/cuda.cu include/matern_device.cuh include/matern_device/*.cuh benchmark/*.cu 2>/dev/null'
run "cuda: no raw launch outside the launcher" bash -c '
    "'"$PY"'" - <<PYEOF
import re, sys
src = open("src/backends/cuda.cu").read()
m = re.search(r"static inline void matern_cuda_launch\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
inside = m.group(1) if m else ""
total  = len(re.findall(r"matern_cuda_kernel(?:_z)?\s*<<<", src))
within = len(re.findall(r"matern_cuda_kernel(?:_z)?\s*<<<", inside))
if total != within:
    print(f"{total-within} raw kernel launch(es) bypass the launcher", file=sys.stderr)
    sys.exit(1)
PYEOF'

# With nvcc present the GPU sources must compile, so an absent binary is not a failed build.
run "gpu sources build when nvcc exists" bash -c '
    command -v nvcc >/dev/null 2>&1 || { echo "no nvcc -- genuinely skipped ($(bash benchmark/tools/missing.sh lines nvcc | cut -f2))"; exit 0; }
    make build/benchmark_gpu build/suppliers_gpu >/dev/null 2>&1'

# A skipped supplier stage is noted by sup_note and kept out of MISSING, which marks a run partial.
run "split run: skipped != failed" python3 -c '
import re, sys
s = open("benchmark/run_benchmark.sh").read()
bad = []
m = re.search(r"^sup_note\(\) *\{(.*?)^\}$", s, re.M | re.S)
if not m: bad.append("sup_note not defined")
elif "record" in m.group(1) or "MISSING" in m.group(1) or "stage_rc" in m.group(1):
    bad.append("sup_note marks the run failed")
for stage, why in (("suppliers", "STAGES=ladder"), ("supplier-ladder", "STAGES=rest")):
    if not re.search(r"sup_note " + stage + r" +\"SKIPPED \(" + why + r"\)\"", s):
        bad.append(stage + ": no SKIPPED (" + why + ") note")
if re.search(r"record +sup", s): bad.append("a supplier stage is recorded with record")
if bad: print("; ".join(bad), file=sys.stderr); sys.exit(1)
'

# The manifest checksums the CUDA sources as well as the C sources.
run "manifest checksums the .cu sources" bash -c '
    grep -q "benchmark/suppliers_gpu.cu"  "'"$LAST"'/MANIFEST.txt" &&
    grep -q "benchmark/benchmark_gpu.cu"  "'"$LAST"'/MANIFEST.txt" &&
    grep -q "src/backends/cuda.cu"        "'"$LAST"'/MANIFEST.txt"'

echo "== stage 5: the two A/B stages =="

# The stage selector rejects an unknown STAGES value.
run "STAGES rejects an unknown value" bash -c '
    ! STAGES=levinn bash benchmark/run_benchmark.sh >/dev/null 2>&1'
run "STAGES accepts levin and threads" bash -c '
    grep -q "all|ladder|rest|levin|threads" benchmark/run_benchmark.sh'

# STAGES=record writes the run record only; the run is complete with both CPU stages skipped.
run "STAGES=record measures nothing on the CPU" bash -c '
    STAGES=record SCOPE=record-test WM_GPU_PIPELINE=1 NOTIFY=none \
        bash benchmark/run_benchmark.sh >/dev/null 2>&1 || exit 1
    d=$(ls -d "$WM_RUNS_DIR"/*_record-test 2>/dev/null | tail -1)
    [ -n "$d" ] && [ -s "$d/PROVENANCE.txt" ] && [ -s "$d/MANIFEST.txt" ] && [ -s "$d/detect.txt" ] || exit 1
    ! ls "$d" | grep -Eq "^(accuracy|ladder|suppliers|timing_raw)" || { ls "$d"; exit 1; }
    grep -q "^STATUS   complete" "$d/STATUS" &&
    grep -q "accuracy (STAGES=record" "$d/STATUS" && grep -q "ladder (STAGES=record" "$d/STATUS"'

# call-large is selected by name only; `all` is the stage set of the published run.
run "GPU_STAGES=all never selects call-large" bash -c '
    fn=$(sed -n "/^want_gpu(){/,/^}/p" benchmark/run_gpu.sh)
    [ -n "$fn" ] || exit 1
    eval "$fn"
    GPU_STAGES=all;              want_gpu core && want_gpu threads && ! want_gpu call-large &&
    GPU_STAGES=call-large;       want_gpu call-large && ! want_gpu core &&
    GPU_STAGES=core,call-large;  want_gpu call-large && want_gpu core && ! want_gpu levin'

# --call-only and --ladder are refused together, before any device is touched.
if [ -x build/suppliers_gpu ]; then
    run "suppliers_gpu: --call-only with --ladder refused" bash -c '
        ./build/suppliers_gpu --call-only --ladder >/dev/null 2>&1; [ $? -eq 2 ]'
else
    skip "suppliers_gpu: --call-only with --ladder refused" "no build/suppliers_gpu (needs nvcc)"
fi

# Main stages run when STAGES is all or names them, so other STAGES values do not run them.
run "main stages gate on inclusion" bash -c '
    grep -q "STAGES\" = all \] || \[ \"\$STAGES\" = rest" benchmark/run_benchmark.sh &&
    grep -q "STAGES\" = all \] || \[ \"\$STAGES\" = ladder" benchmark/run_benchmark.sh &&
    ! grep -q "STAGES\" != ladder" benchmark/run_benchmark.sh &&
    ! grep -q "STAGES\" != rest"   benchmark/run_benchmark.sh'

# The Levin A/B build and stage both refuse arms that report the same routing.
run "levin build refuses identical arms" bash -c '
    grep -q "two arms report the same routing" Makefile'
run "levin stage refuses identical arms" bash -c '
    grep -q "refusing to measure" benchmark/run_benchmark.sh'
# MATERN_WITHOUT_LEVIN (vector) and WMS_LEVIN (scalar) are independent, so both are reported.
run "benchmark_cpu reports BOTH routers" bash -c '
    o=$(./build/benchmark_cpu --print-build);
    printf "%s" "$o" | grep -qE "^levin_vector=(on|off)$" &&
    printf "%s" "$o" | grep -qE "^levin_scalar=(on|off)$"'
run "shipped build: vector on, scalar off" bash -c '
    ./build/benchmark_cpu --print-build \
      | tr "\n" " " | grep -q "levin_vector=on levin_scalar=off"'
run "the scalar Levin band is a real opt-in" bash -c '
    grep -q "#ifdef WMS_LEVIN" src/backends/scalar/region_router.c'
run "levin A/B builds three arms" bash -c '
    grep -q "benchmark_cpu_base"  Makefile &&
    grep -q "benchmark_cpu_novec" Makefile &&
    grep -q "benchmark_cpu_scal"  Makefile'
run "levin stage cross-checks macro independence" bash -c '
    grep -q "scalar_unmoved_by_vector_macro" benchmark/run_benchmark.sh &&
    grep -q "vector_unmoved_by_scalar_macro" benchmark/run_benchmark.sh'

# Both device routing macros are overridable, so -D changes both entries together.
run "both CUDA flag macros are overridable" bash -c '
    grep -q "#ifndef MATERN_CUDA_FLAGS$"   src/backends/cuda.cu &&
    grep -q "#ifndef MATERN_CUDA_FLAGS_Z$" src/backends/cuda.cu'
run "the Levin variants add ROUTE_L, not remove" bash -c '
    grep -q "MATERN_ROUTE_FLAGS_PRIMEAL  (MATERN_ROUTE_FLAGS_PRIMEA  | MATERN_ROUTE_L)" \
        src/router/region_router_device.cuh &&
    grep -q "MATERN_ROUTE_FLAGS_PRIMEAZL (MATERN_ROUTE_FLAGS_PRIMEAZ | MATERN_ROUTE_L)" \
        src/router/region_router_device.cuh'
# cuda.cu static-asserts the public Levin bit; this check needs no CUDA toolchain.
run "public Levin bit mirrors the router" bash -c '
    a=$(sed -n "s/^#define MATERN_ROUTE_LEVIN_BIT  *\([0-9][0-9]*\).*/\1/p" include/matern.h);
    b=$(sed -n "s/^#define MATERN_ROUTE_L  *\([0-9][0-9]*\).*/\1/p" src/router/region_router_device.cuh);
    [ -n "$a" ] && [ "$a" = "$b" ]'

if make build/threads_cpu >/dev/null 2>&1; then
  run "threads_cpu emits its header" bash -c '
      ./build/threads_cpu --threads 1 --reps 2 --n 4096 --backend scalar \
          --data-dir data --format csv 2>/dev/null \
        | head -1 | grep -q "^schema,device,supplier,family,law,backend,threads,"'
  run "threads_cpu: answers agree across threads" bash -c '
      ./build/threads_cpu --threads 1,2 --reps 2 --n 4096 --backend scalar \
          --data-dir data --format csv 2>/dev/null \
        | awk -F, "NR>1 && \$7 > 1 && \$19 != \"yes\" { bad=1 } END { exit bad+0 }"'
  # T=1 reports `n/a` and is skipped by the check above.
  run "threads_cpu refuses a wrong backend label" bash -c '
      ! ./build/threads_cpu --backend nosuchbackend --threads 1 --reps 1 \
          --n 1024 --data-dir data >/dev/null 2>&1'
else
  echo "  threads_cpu                        SKIPPED (did not build)"
fi

echo "== stage 5f: the prefactor-fold stage =="
# STAGES=fold is opt-in, runs on the small corpus files and has no supplier side.
run "STAGES accepts fold" grep -qF \
    'case "$STAGES" in all|ladder|rest|levin|threads|record|fold) ;; *)' benchmark/run_benchmark.sh
run "fold needs no 2^25-point file" grep -qF \
    'rest|record|fold) _corpus_scope=tracked ;;' benchmark/run_benchmark.sh
run "fold stage has no supplier side" grep -qF \
    '[ "$STAGES" = threads ] || [ "$STAGES" = fold ] || [ "$STAGES" = record ]; then' benchmark/run_benchmark.sh
# The Makefile refuses the twin's switches in XCF, so they never reach the shipped build.
run "the twin is refused in XCF" bash -c '
    grep -q "WMS_UNFOLDED and WM_FOLD belong to the build/fold_cpu rules" Makefile &&
    ! make -s -n XCF=-DWMS_UNFOLDED=1 all >/dev/null 2>&1'
if make build/fold_cpu >/dev/null 2>&1; then
  run "fold_cpu reports fold=on and both routers" bash -c '
      o=$(./build/fold_cpu --print-build | tr "\n" " ");
      echo "$o" | grep -q "levin_vector=on levin_scalar=off fold=on"'
  run "fold objects export two entries each" bash -c '
      for f in build/backends/scalar_control_*.o build/backends/scalar_unfolded_*.o; do
          [ "$(nm -g --defined-only "$f" | wc -l)" -eq 2 ] || exit 1; done'
  run "libmatern.a holds no twin symbol" bash -c '
      ! nm build/libmatern.a 2>/dev/null | grep -q "unfolded\|compose_unfolded"'
  # Twin and control against the backend, per scalar instruction set this CPU runs.
  $CC $CFLAGS $INC -o "$TMP/t_fold" $B/tests/test_fold_control.c \
      build/backends/scalar_control_*.o build/backends/scalar_unfolded_direct_*.o \
      build/libmatern_internal.a -lm 2>/dev/null
  run "fold controls hold" "$TMP/t_fold"
  run "fold_cpu refuses the ladder" bash -c '
      ! ./build/fold_cpu --ladder --metric speed >/dev/null 2>&1'
  run "fold_cpu refuses an unknown backend" bash -c '
      ! ./build/fold_cpu --backend nosuch --law grid --max-points 100 --reps 1 \
          --data-dir data --format csv >/dev/null 2>&1'
  # fold_cpu is benchmark_cpu with the twin backends: same header and scalar accuracy fields.
  ./build/fold_cpu --family z_nu --law grid,large_nu --max-points 4000 --reps 2 \
      --data-dir data --format csv > "$TMP/fold.csv" 2>/dev/null
  ./build/benchmark_cpu --family z_nu --law grid,large_nu --backend scalar \
      --max-points 4000 --reps 2 --data-dir data --format csv > "$TMP/bscalar.csv" 2>/dev/null
  run "fold_cpu writes benchmark_cpu's header" bash -c '
      [ "$(head -1 "'"$TMP"'/fold.csv")" = "$(head -1 "'"$TMP"'/bscalar.csv")" ]'
  run "fold_cpu scalar rows = benchmark_cpu's" bash -c '
      a=$(awk -F, "NR>1 && \$6==\"scalar\"" "'"$TMP"'/fold.csv" | cut -d, -f1-31)
      b=$(awk -F, "NR>1 && \$6==\"scalar\"" "'"$TMP"'/bscalar.csv" | cut -d, -f1-31)
      [ -n "$a" ] && [ "$a" = "$b" ]'
  run "the control equals the scalar backend" bash -c '
      a=$(awk -F, "NR>1 && \$6==\"scalar\"" "'"$TMP"'/fold.csv" | cut -d, -f1-5,7-31)
      c=$(awk -F, "NR>1 && \$6==\"scalar-control\"" "'"$TMP"'/fold.csv" | cut -d, -f1-5,7-31)
      [ -n "$a" ] && [ "$a" = "$c" ]'
  # Positive control: Gamma(nu) overflows past nu ~ 171.6, so the direct twin is non-finite.
  run "direct twin is non-finite on large_nu" bash -c '
      h=$(head -1 "'"$TMP"'/fold.csv"); c=$(echo "$h" | tr "," "\n" | grep -n "^nonfinite$" | cut -d: -f1)
      awk -F, -v c="$c" "\$5==\"large_nu\" && \$6==\"scalar-unfolded-direct\" && \$c+0>0 {ok=1} END {exit !ok}" \
          "'"$TMP"'/fold.csv"'
  # The stage itself, reduced.
  STAGES=fold REPS=2 TIME_MAX_POINTS=1024 FAMILIES=z_nu bash benchmark/run_benchmark.sh \
      >"$TMP/fold_run.log" 2>&1
  FOLDRUN=$(readlink -f "$RUNSDIR/latest")
  run "fold stage: rows, cross-checks, no suppliers" bash -c '
      d="'"$FOLDRUN"'"
      case "$d" in *-fold) ;; *) exit 1 ;; esac
      grep -q "fold rows" "$d/STATUS" &&
      grep -q "^fold_xcheck_control_equals_scalar ok" "$d/STAGES.txt" &&
      grep -q "^fold_xcheck_twin_differs ok" "$d/STAGES.txt" &&
      grep -q "^suppliers .*SKIPPED" "$d/STAGES.txt"'
  run "figures: the fold run draws" bash -c '
      m=$(bash -c "$FIGMIRROR"); cp -r "'"$FOLDRUN"'" "$m/runs/2026-01-01T00-00Z_hosta_cpu-fold"
      "'"$PY"'" "$m/plots/make_paper_figures.py" --full >/dev/null 2>&1; rc=$?
      f=$(find "$m/plots/paper" -name "fig-paper-fold.png" | wc -l)
      e=$(find "$m/plots/paper" -name "fig-paper-fold-ecdf-z_nu.png" | wc -l); rm -rf "$m"
      [ "$rc" -eq 0 ] && [ "$f" -eq 1 ] && [ "$e" -eq 1 ]'
  # Marker control: a scalar row given u_max 1e15 and non-finite points reads `Cap (dag)`.
  run "tables: the fold tables, markers fire" bash -c '
      m=$(mktemp -d); mkdir -p "$m/runs"
      cp -r "'"$FOLDRUN"'" "$m/runs/2026-01-01T00-00Z_hosta_cpu-fold"
      f="$m/runs/2026-01-01T00-00Z_hosta_cpu-fold/fold.csv"
      awk -F, -v OFS=, "NR==1 { for (i=1;i<=NF;i++) c[\$i]=i; print; next }
          \$c[\"law\"]==\"grid\" && \$c[\"backend\"]==\"scalar\" { \$c[\"u_max\"]=1e15; \$c[\"nonfinite\"]=5 }
          { print }" "$f" > "$f.new" && mv "$f.new" "$f"
      "'"$PY"'" benchmark/tables/make_paper_tables.py --runs "$m/runs" --out "$m/t" \
          --no-preview --full --only tab-fold >/dev/null 2>&1; rc=$?
      q=$(ls "$m/t"/tab-fold-quantiles-z_nu.tex 2>/dev/null | wc -l)
      c=$(ls "$m/t"/*/tab-fold-cpu.md 2>/dev/null | head -1)
      ok=0; [ -n "$c" ] && grep "z_nu / grid" "$c" | grep -q "Cap (dag)" && ok=1
      rm -rf "$m"; [ "$rc" -eq 0 ] && [ "$q" -eq 1 ] && [ "$ok" -eq 1 ]'
else
  echo "  fold_cpu                           SKIPPED (did not build)"
fi

echo "== stage 5w: the workspace is INSIDE every library timing but the ladder's decomposition =="
# Only the ladder's `reused` series times a warm workspace.
run "accuracy-row timing builds a workspace per rep" bash -c '
    awk "/^static wm_time time_accuracy_law/,/^}/" benchmark/benchmark_cpu.c |
      awk "/for \\(int r = 0; r < reps; r\\+\\+\\)/,/t\\[r\\] =/" |
      grep -q "matern_workspace_create"'
# Positive control for the check above: a body with a warm workspace must fail it.
run "  ...and the check sees a warm body" bash -c '
    ! printf "static wm_time time_accuracy_law(void)\n{\n    matern_workspace_create(&ws, 0);\n    for (int r = 0; r < reps; r++) {\n        evaluate();\n        t[r] = 1;\n    }\n}\n" |
      awk "/^static wm_time time_accuracy_law/,/^}/" |
      awk "/for \\(int r = 0; r < reps; r\\+\\+\\)/,/t\\[r\\] =/" |
      grep -q "matern_workspace_create"'
run "ladder carries the build series distribution" bash -c '
    ./build/benchmark_cpu --family z_nu --law size_half --backend $BK \
        --metric speed --ladder --reps 3 --max-reps 3 --format csv --data-dir data \
      2>/dev/null | head -1 | grep -q ",ns_ws_per_call_med,.*,ns_ws_per_call_spread,"'
run "build median >= build min, > reused min at 2^8" bash -c '
    ./build/benchmark_cpu --family z_nu --law size_half --backend $BK \
        --metric speed --ladder --reps 5 --max-reps 5 --format csv --data-dir data \
      2>/dev/null | awk -F, "NR==1{for(i=1;i<=NF;i++)c[\$i]=i; next}
         { if (\$c[\"ns_ws_per_call_med\"] < \$c[\"ns_ws_per_call_min\"]) bad=1 }
         \$c[\"n\"]==256 { if (!(\$c[\"ns_ws_per_call_med\"] > \$c[\"ns_ws_reused_min\"])) bad=1; seen=1 }
         END { exit (bad || !seen) }"'
# Only the build series is dumped: 13 rungs x 3 reps, not 2 x that.
run "timing_raw holds ONE series per rung" bash -c '
    f=$(mktemp); WM_RAW_REPS=$f ./build/benchmark_cpu --family z_nu --law size_half \
        --backend $BK --metric speed --ladder --reps 3 --max-reps 3 --format csv \
        --data-dir data >/dev/null 2>&1
    n=$(awk -F, "NR>1 && \$6==256" "$f.$BK" | wc -l); rm -f "$f" "$f.$BK"
    [ "$n" -eq 3 ]'
if [ -x build/threads_cpu ]; then
  run "threads: no reused-pass columns" bash -c '
      h=$(./build/threads_cpu --threads 1 --reps 2 --n 4096 --backend $BK \
          --data-dir data --format csv 2>/dev/null | head -1)
      echo "$h" | grep -q ",ns_per_elem," &&
      ! echo "$h" | grep -q "ws_per_call"'
  run "threads: workspace built inside the timed rep" bash -c '
      awk "/^static void \*run_worker/,/^}/" benchmark/threads_cpu.c |
        grep -q "matern_workspace_create(&ws_, 0)" &&
      ! awk "/^static void \*run_worker/,/^}/" benchmark/threads_cpu.c |
        grep -q "matern_workspace_create(&workspace"'
fi

echo "== stage 5c: the workspace cost is MEASURED, not differenced =="

run "ladder emits ns_workspace"  bash -c '
    ./build/benchmark_cpu --family z_nu --law size_random --backend $BK \
        --metric speed --ladder --reps 3 --max-reps 3 --format csv --data-dir data \
      2>/dev/null | head -1 | grep -q ",ns_workspace,"'
# The workspace is tile-sized, so ns_workspace must be independent of N.
run "ns_workspace is flat in N"  bash -c '
    ./build/benchmark_cpu --family z_nu --law size_random --backend $BK \
        --metric speed --ladder --reps 5 --max-reps 5 --format csv --data-dir data \
      2>/dev/null | awk -F, "NR>1 && \$15>0 { if (!lo || \$15<lo) lo=\$15;
                                              if (\$15>hi) hi=\$15 }
         END { exit !(lo>0 && hi/lo < 4.0) }"'
run "ns_workspace is 0 where none is built" bash -c '
    ./build/benchmark_cpu --family z_nu --law size_random --backend scalar \
        --metric speed --ladder --reps 3 --max-reps 3 --format csv --data-dir data \
      2>/dev/null | awk -F, "NR>1 { if (\$15+0 != 0) bad=1 } END { exit bad+0 }"'
run "it uses the row's own rep count" bash -c '
    grep -q "time_workspace(pol.max_reps)" benchmark/benchmark_cpu.c'
# The setup panel plots the workspace cost relative to the call, not per element.
run "the figure plots the measurement" bash -c '
    grep -q "100.0 \* ws / call" benchmark/plots/paper_figures.py'
run "the setup panel is not per-element" bash -c '
    ! grep -q "ns_workspace\") / _f(r, \"n\")" benchmark/plots/paper_figures.py'
run "all three timings are warmed" bash -c '
    awk "/^static wm_time time_workspace/,/^}/" benchmark/benchmark_cpu.c |
        grep -q "for (int w = 0; w < WM_WARMUP_REPS; w++)" &&
    awk "/^static wm_time time_call\\(/,/^}/" benchmark/benchmark_cpu.c |
        grep -q "for (int w = 0; w < WM_WARMUP_REPS; w++)" &&
    awk "/^static wm_time time_call_ws_reused/,/^}/" benchmark/benchmark_cpu.c |
        grep -q "for (int w = 0; w < WM_WARMUP_REPS; w++)"'

echo "== stage 5b: the workspace exemption =="

$CC $CFLAGS $INC -o "$TMP/t_wsneed" $B/tests/test_workspace_need.c \
    build/libmatern_internal.a -lm 2>/dev/null
run "scalar needs none, matched does"  "$TMP/t_wsneed"
# The harness exempts the scalar builds only; `matched` needs a workspace.
run "harness exempts scalar only" bash -c '
    grep -q "return strncmp(backend, \"scalar\", 6) != 0;" benchmark/benchmark_cpu.c &&
    ! grep -q "matched.*!= 0;" benchmark/benchmark_cpu.c'
run "timed region is gated on that" bash -c '
    grep -q "if (ws_needed) matern_workspace_create" benchmark/benchmark_cpu.c &&
    grep -q "if (ws_needed) matern_workspace_destroy" benchmark/benchmark_cpu.c'
run "the figure drops it from the setup panel" bash -c '
    grep -q "^NO_WORKSPACE = {\"scalar\"}" benchmark/plots/paper_figures.py'

echo "== stage 5d: paper mode =="

# Paper mode moves the on-canvas notes into captions.tex.
run "paper mode is opt-in"          bash -c '
    cd benchmark/plots && python3 -c "import style; raise SystemExit(0 if not style.PAPER else 1)"'
run "paper mode moves the title to the caption" bash -c '
    cd benchmark/plots && python3 -c "
import style
style.PAPER = True
t = style.fig_title(\"Worst-case error and cost on two CPU architectures\")
raise SystemExit(0 if t == \"\" and style.CAPTION[\"title\"] ==
                 \"Worst-case error and cost on two CPU architectures.\" else 1)"'
run "captions escape text, not math" bash -c '
    cd benchmark/plots && python3 -c "
import style
r = style._tex_escape_outside_math(r\"\$(z,\nu)\$ on run a_b_c\")
raise SystemExit(0 if r.count(chr(92)+\"_\") == 2 and r\"\$(z,\nu)\$\" in r else 1)"'
run "paper mode writes a separate tree" bash -c '
    grep -q "plots\", \"paper-final\"" benchmark/plots/make_paper_figures.py'
run "stale captions are cleared"    bash -c '
    grep -q "if \"captions.tex\" in _f" benchmark/plots/make_paper_figures.py'
run "absent GPU is not a label"     bash -c '
    cd benchmark/plots && python3 -c "
import runs as mp, os, tempfile
d = tempfile.mkdtemp()
open(os.path.join(d,\"PROVENANCE.txt\"),\"w\").write(
    \"cpu AMD EPYC 9454\ngpu none -- no nvidia-smi on this machine (CPU run, no device rows)\n\")
raise SystemExit(0 if \"nvidia\" not in mp.arch_label(d,\"x_y\")[1] else 1)"'
run "no legend bypasses the stamp band" bash -c '
    ! grep -q "bbox_to_anchor=(0.5, -0" benchmark/plots/paper_figures.py'

echo "== stage 5e: caption prose =="

run "a newline becomes a full stop" bash -c '
    cd benchmark/plots && python3 -c "
import style
style.PAPER = True
style.fig_title(\"A -- x\nb follows\nc follows\")
raise SystemExit(0 if style.CAPTION[\"title\"] == \"A -- x. B follows. C follows.\" else 1)"'
run "canvas fragments become sentences" bash -c '
    cd benchmark/plots && python3 -c "
import style
raise SystemExit(0 if style._sentence(\"solid = a thing\") == \"Solid = a thing.\" else 1)"'
run "identifiers keep their case" bash -c '
    cd benchmark/plots && python3 -c "
import style
raise SystemExit(0 if style._sentence(\"avx512 wins\").startswith(\"avx512\") else 1)"'
run "column names are translated" bash -c '
    cd benchmark/plots && python3 -c "
import style
style.PAPER = True
t = style.panel_title(\"only in u_fence, u_iqr, u_mean, u_q1, u_q3\")
raise SystemExit(0 if \"u_q1\" not in t and \"quartile\" in t else 1)"'
# Allowed ALL-CAPS words are acronyms or file names; identifiers with an underscore do not match.
run "no ALL-CAPS emphasis in figure text" python3 -c '
import re, sys, tokenize
allow = {"CUDA", "NUMA", "NVIDIA", "PROVENANCE", "CATASTROPHIC", "STATUS"}
bad = []
for f in ("benchmark/plots/paper_figures.py", "benchmark/plots/style.py"):
    for t in tokenize.generate_tokens(open(f).readline):
        if t.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
            bad += [(f, t.start[0], w) for w in re.findall(r"\b[A-Z]{4,}\b", t.string)
                    if w not in allow]
for b in bad: print("%s:%d: %s" % b, file=sys.stderr)
sys.exit(1 if bad else 0)
'
run "the workspace title names the family" bash -c '
    grep -q "_law_title(law)}, {FAMILY_LABEL.get(family, family)}" \
        benchmark/plots/paper_figures.py'
run "run ids are a note, not a caption" bash -c '
    grep -q "def caption_finalise" benchmark/plots/style.py &&
    ! grep -q "Measured on run" benchmark/plots/paper_figures.py'
run "cross-machine axes name the arch" bash -c '
    grep -q "arch_short" benchmark/plots/make_paper_figures.py'

echo "== stage 6: keys and captions stay out of the data =="

# Legends sit outside the axes: on other machines' runs the curves may fill any corner.
run "no legend is drawn inside an axes" bash -c '
    ! grep -qE "^[[:space:]]*(ax|axa|axb|axl|axr|axs|top|bot)\.legend\(" \
        benchmark/plots/paper_figures.py'
# loc="best", the default of .legend(), follows the data; comments are stripped before the check.
run "no legend relies on loc=best" bash -c '
    src=$(sed "s/#.*$//" benchmark/plots/paper_figures.py);
    ! printf "%s" "$src" | grep -qE "legend\(\)" &&
    ! printf "%s" "$src" | grep -q "loc=\"best\""'
# Every note but the stamp goes through note_below/footer, which handle paper mode.
run "only the stamp draws raw text"  bash -c '
    [ "$(grep -c "fig\.text(" benchmark/plots/paper_figures.py)" -eq 1 ]'

run "the shared placement helpers exist" bash -c '
    grep -q "^def legend_below"  benchmark/plots/style.py &&
    grep -q "^def legend_figure" benchmark/plots/style.py &&
    grep -q "^def footer"        benchmark/plots/style.py &&
    grep -q "^def wrap_title"    benchmark/plots/style.py'
run "one font floor, not per call site" bash -c '
    grep -q "^LEGEND_FONTSIZE = " benchmark/plots/style.py &&
    grep -q "^NOTE_FONTSIZE   = " benchmark/plots/style.py'

# The top rung is the dataset size; the small set is the first 2^20 points of the 2^25 files.
run "ladder cap: small set stops at 2^20" bash -c '
    top=$(./build/benchmark_cpu --family z_nu --law size_random --backend scalar \
          --metric speed --ladder --reps 1 --max-reps 1 --format csv 2>/dev/null \
          | tail -n +2 | cut -d, -f7 | sort -n | tail -1)
    [ "$top" = 1048576 ]'
run "ladder cap: large set goes past 2^20" bash -c '
    { compgen -G "data/z_nu/z_nu-nu_size_random_large_*.bin" || compgen -G "data/z_nu-nu_size_random_large_*.bin"; } >/dev/null || { echo "no 2^25 corpus -- skipped"; exit 0; }
    top=$(./build/benchmark_cpu --family z_nu --law size_random_large --backend $BK \
          --metric speed --ladder --ladder-max 2097152 --reps 1 --max-reps 1 --format csv 2>/dev/null \
          | tail -n +2 | cut -d, -f7 | sort -n | tail -1)
    [ "$top" = 2097152 ]'
run "ladder cap: --ladder-max sets the top rung" bash -c '
    top=$(./build/benchmark_cpu --family z_nu --law size_half --backend $BK \
          --metric speed --ladder --ladder-max 4096 --reps 1 --max-reps 1 --format csv 2>/dev/null \
          | tail -n +2 | cut -d, -f7 | sort -n | tail -1)
    [ "$top" = 4096 ]'
run "size laws: the 2^20 prefix of the 2^25 sets" bash -c '
    n=$(./build/benchmark_cpu --family h_nu_s --law size_random --backend $BK \
        --metric accuracy --max-points 1024 --format csv 2>/dev/null \
        | python3 -c "import csv,sys; print({r[\"n_file\"] for r in csv.DictReader(sys.stdin)})")
    [ "$n" = "{'"'"'1048576'"'"'}" ]'
run "ladder estimate: small set tops at 2^20" bash -c '
    [ "$(LADDER_SET=small bash benchmark/tools/ladder_estimate.sh --vendor amd | awk "\$1==\"ladder_top_n\"{print \$2}")" = 1048576 ]'
run "LADDER_MAX is validated" bash -c '
    out=$(LADDER_MAX=abc bash benchmark/run_benchmark.sh 2>&1); rc=$?
    [ "$rc" -ne 0 ] && echo "$out" | grep -q "LADDER_MAX must be a positive number"'
run "FAMILIES is validated, before any run" bash -c '
    n0=$(ls "$RUNSDIR" | wc -l)
    out=$(FAMILIES=z_nu,nosuchfam bash benchmark/run_benchmark.sh 2>&1); rc=$?
    [ "$rc" -eq 2 ] && echo "$out" | grep -q "unknown family .nosuchfam." &&
    [ "$(ls "$RUNSDIR" | wc -l)" -eq "$n0" ]'
# The drivers refuse an unknown family; the valid pair is the positive control.
run "drivers refuse an unknown family" bash -c '
    ./build/benchmark_cpu --family z_nu,nosuchfam --law grid --backend scalar \
        --metric accuracy --max-points 64 --format csv >/dev/null 2>&1; [ $? -eq 2 ] &&
    ./build/benchmark_cpu --family z_nu,h_nu_s --law grid --backend scalar \
        --metric accuracy --max-points 64 --format csv >/dev/null 2>&1'

run "law all: excludes the 2^25 ladders" bash -c '
    l=$(./build/benchmark_cpu --family z_nu --law all --backend scalar --metric accuracy \
        --max-points 256 --format csv 2>/dev/null | tail -n +2 | cut -d, -f5 | sort -u)
    ! echo "$l" | grep -q "_large"'

# Both ladder sets read the 2^25 files; tools/corpus.sh need refuses before the size-law check.
run "ladders refuse without the 2^25 sets" bash -c '
    t=$(mktemp -d); mkdir -p "$t/z_nu" "$t/h_nu_s"
    big=$(DATA="$t" LADDER_SET=large bash benchmark/run_benchmark.sh 2>&1); rb=$?
    small=$(DATA="$t" LADDER_SET=small bash benchmark/run_benchmark.sh 2>&1); rs=$?
    rm -rf "$t"
    [ "$rb" -ne 0 ] && [ "$rs" -ne 0 ] &&
    echo "$big"   | grep -q "needs the corpus" &&
    echo "$small" | grep -q "needs the corpus"'

run "threads: scores ULP against the reference" bash -c '
    out=$(./build/threads_cpu --family z_nu --law size_random --backend $BK \
          --threads 2 --reps 2 --n 4096 --format csv 2>/dev/null)
    # Columns are located by name in the header.
    h=$(echo "$out" | head -1)
    cu=$(echo "$h" | tr "," "\n" | grep -n "^u_max$" | cut -d: -f1)
    ck=$(echo "$h" | tr "," "\n" | grep -n "^u_scored$" | cut -d: -f1)
    u=$(echo "$out" | tail -1 | awk -F, -v c="$cu" "{print \$c}")
    k=$(echo "$out" | tail -1 | awk -F, -v c="$ck" "{print \$c}")
    [ "$k" = 4096 ] && awk -v v="$u" "BEGIN{exit !(v>0 && v<1e6)}"'

run "threads: T=1 identical is n/a, not yes" bash -c '
    out=$(./build/threads_cpu --family z_nu --law size_random --backend $BK \
          --threads 1,2 --reps 2 --n 2048 --format csv 2>/dev/null)
    h=$(echo "$out" | head -1); ci=$(echo "$h" | tr "," "\n" | grep -n "^identical$" | cut -d: -f1)
    one=$(echo "$out" | awk -F, -v c="$ci" "\$7==1{print \$c}" | head -1)
    two=$(echo "$out" | awk -F, -v c="$ci" "\$7==2{print \$c}" | head -1)
    [ "$one" = "n/a" ] && [ "$two" = "yes" ]'

echo
echo "  $pass passed, $fail failed"
exit $((fail>0))
