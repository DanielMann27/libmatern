#!/bin/bash
# ladder_estimate.sh [--vendor amd|intel] -- estimates the wall time of a run's ladder stage (a
# planning model) from the run parameters in the environment, with run_benchmark.sh's defaults;
# BACKENDS unset counts all five backends.
# Output keys: ladder_estimate_s ladder_library_s ladder_suppliers_s ladder_top_n ladder_vendor.
set -u
vendor=""
case "${1:-}" in --vendor) vendor="${2:-}" ;; --vendor=*) vendor="${1#--vendor=}" ;; esac
if [ -z "$vendor" ]; then
  case "$(sed -n 's/^vendor_id[ \t]*: //p' /proc/cpuinfo 2>/dev/null | head -1)" in
    *AMD*) vendor=amd ;; *) vendor=intel ;;
  esac
fi
# Calibrated per vendor on the published ladder runs: c_lib is the library's wall time per
# element and call (corpus loading, warm-up and workspace included), sumns the time per element
# in nanoseconds, summed over the redistributable supplier rows of both ladder laws.
case "$vendor" in
  amd)   c_lib=147e-9; sumns=7143 ;;
  intel) c_lib=180e-9; sumns=8809 ;;
  *) echo "ladder_estimate.sh: --vendor must be amd or intel" >&2; exit 2 ;;
esac

laws="${LADDER_LAWS:-}"
if [ -z "$laws" ]; then
  case "${LADDER_SET:-small}" in
    large) laws="size_random_large,size_half_large" ;;
    *)     laws="size_random,size_half" ;;
  esac
fi
first=$(printf '%s' "$laws" | cut -d, -f1)
f=$(ls "${DATA:-data}"/z_nu/z_nu-nu_"${first}"_[0-9]*.bin "${DATA:-data}"/z_nu-nu_"${first}"_[0-9]*.bin 2>/dev/null | head -1)
if [ -n "$f" ]; then
  ntop=$(printf '%s' "$f" | sed 's/.*_\([0-9]*\)\.bin$/\1/')
else
  # the 2^20 size laws are prefixes of the 2^25 sets
  case "$first" in *_large) ntop=33554432 ;; *) ntop=1048576 ;; esac
fi
if [ -n "${LADDER_MAX:-}" ] && [ "$LADDER_MAX" -lt "$ntop" ] 2>/dev/null; then
  ntop=$(awk -v m="$LADDER_MAX" 'BEGIN{n=256; while (n*2 <= m) n*=2; print n}')
fi

nfam=$(printf '%s' "${FAMILIES:-z_nu,h_nu_s}" | tr ',' '\n' | grep -c .)
nbk=$(printf '%s'  "${BACKENDS:-avx512,avx2,matched_fma,matched,scalar}" | tr ',' '\n' | grep -c .)
nlaw=$(printf '%s' "$laws" | tr ',' '\n' | grep -c .)
rows=$((nfam * nbk * nlaw))
reps="${REPS:-65}"
case "$reps" in ''|*[!0-9]*) reps=65 ;; esac
reps=$((10#$reps))        # base 10: a leading zero is not octal
[ "$reps" -gt 0 ] || reps=65
calls=$((reps + 2))
# w, in units of N_top: rungs double, so a ladder costs about 2 N_top; with LADDER_PASS_FROM=P
# the rungs from P up come from one pass, P + N_top.
_pf="${LADDER_PASS_FROM:-4194304}"; [ "$_pf" = off ] && _pf=0
w=$(awk -v n="$ntop" -v s="$_pf" 'BEGIN{ if (s > 0 && s <= n) printf "%.6f", s/n + 1; else print 2 }')
lib=$(awk -v r="$rows" -v n="$ntop" -v c="$c_lib" -v w="$w" -v k="$calls" 'BEGIN{printf "%d", w*r*k*n*c}')

supsum=$(awk -v l="$nlaw" -v f="$nfam" -v S="$sumns" 'BEGIN{printf "%d", S*l/2*f}')
nrows=$((13 * nlaw * nfam))
# SUP_LADDER=all adds the 6 KAUST rows per law and family, modelled as factor 11 on sumns.
if [ "${SUP_LADDER:-ship}" = all ]; then supsum=$((supsum * 11)); nrows=$((19 * nlaw * nfam)); fi
# The calibration used REPS=65, i.e. 67 calls with the 2 warm-up calls (BENCH_WARMUP_REPS).
sup=$(awk -v n="$ntop" -v S="$supsum" -v w="$w" -v k="$calls" 'BEGIN{printf "%d", w*n*65*S*1e-9*k/67}')
# A row stops after its first rung over the budget: at most 4 x the budget per row.
cap=$((nrows * 4 * ${SUP_LADDER_BUDGET:-20}))
[ "$sup" -gt "$cap" ] && sup=$cap

echo "ladder_estimate_s  $((lib + sup))"
echo "ladder_library_s   $lib"
echo "ladder_suppliers_s $sup"
echo "ladder_top_n       $ntop"
echo "ladder_vendor      $vendor"
