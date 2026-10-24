#!/bin/bash
# boost_detect.sh [--select cpu|gpu] -- finds Boost installations and reports them, or prints the
# include directory selected for the CPU row (1.75.0 only) or the GPU row (>= 1.87, with
# BOOST_MATH_GPU_ENABLED); BOOST_CPU_ROOT / BOOST_ROOT override them, and none disables a row.
set -u
MODE=report
ROLE=""
case "${1:-}" in
    --select) MODE=select; ROLE="${2:-}"
              case "$ROLE" in cpu|gpu) ;; *) echo "boost_detect.sh: --select cpu|gpu" >&2; exit 2 ;; esac ;;
    "") ;;
    *) echo "usage: boost_detect.sh [--select cpu|gpu]" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")/../.." && pwd)
CPU_VERSION=107500         # 1.75.0: the Boost of every published CPU row

# A relative override is taken relative to the current directory, else to the
# repository root (references/Makefile runs this from references/).
absdir() {
  local d=$1
  case "$d" in ""|none) echo "$d"; return ;; esac
  case "$d" in /*) ;; *)
    if [ -d "$d" ]; then d=$(cd "$d" && pwd); elif [ -d "$HERE/$d" ]; then d=$(cd "$HERE/$d" && pwd); fi ;;
  esac
  incroot "$d"
}
# incroot <dir>: the include root for <dir>, which may name the root (holding
# boost/) or boost/ itself, as BOOST_INC_DIR does under some environment modules.
incroot() {
  local d=${1%/}
  if [ ! -f "$d/boost/version.hpp" ] && [ -f "$d/version.hpp" ] && [ "${d##*/}" = boost ]; then
    echo "${d%/boost}"
  else
    echo "$d"
  fi
}
BOOST_CPU_ROOT=$(absdir "${BOOST_CPU_ROOT:-}")
BOOST_ROOT=$(absdir "${BOOST_ROOT:-}")
[ -n "$BOOST_CPU_ROOT" ] || unset BOOST_CPU_ROOT
[ -n "$BOOST_ROOT" ] || unset BOOST_ROOT

# Candidates in search order.  First the Boost a bare compile resolves, which includes paths
# that environment modules add without setting BOOST_ROOT.
CANDIDATES=()
_resolved=$(echo '#include <boost/version.hpp>' \
            | ${CXX:-g++} -E -x c++ - 2>/dev/null \
            | sed -n 's/^# *[0-9][0-9]* *"\(.*\)\/boost\/version\.hpp".*/\1/p' \
            | head -1)
[ -n "$_resolved" ] && CANDIDATES+=("$_resolved")
for _v in "${BOOST_DIR:-}" "${BOOST_ROOT:-}" "${BOOST_CPU_ROOT:-}" "${BOOST_INC_DIR:-}" "${BOOST_INCLUDE_DIR:-}"; do
    [ -n "$_v" ] && [ "$_v" != none ] && CANDIDATES+=("$(incroot "$_v")")
done
for _d in "$HERE"/references/boost/*/; do
    [ -d "$_d" ] && CANDIDATES+=("${_d%/}")
done
[ -n "${CONDA_PREFIX:-}" ] && CANDIDATES+=("$CONDA_PREFIX/include")
command -v brew >/dev/null 2>&1 && CANDIDATES+=("$(brew --prefix 2>/dev/null)/include")
# the compiler's own C++ search path, in its own order
while read -r _d; do
  case "$_d" in /*) CANDIDATES+=("$_d");; esac
done < <(${CXX:-g++} -E -x c++ -v /dev/null 2>&1 \
         | sed -n '/^#include <...> search starts here:/,/^End of search list/p' \
         | sed -n 's/^ \(\/.*\)$/\1/p')
CANDIDATES+=(/usr/include /usr/local/include /opt/homebrew/include /opt/local/include)

version_at() { sed -n 's/^#define BOOST_VERSION  *\([0-9]*\).*/\1/p' "$1/boost/version.hpp" 2>/dev/null | head -1; }
gpu_capable() { grep -q BOOST_MATH_GPU_ENABLED "$1/boost/math/special_functions/bessel.hpp" 2>/dev/null; }
dotted() { echo "$(($1/100000)).$(($1/100%1000)).$(($1%100))"; }

DIRS=(); VERS=()
declare -A _seen=()
for d in "${CANDIDATES[@]}"; do
  d=$(cd "$d" 2>/dev/null && pwd) || continue
  [ -n "${_seen[$d]:-}" ] && continue        # one entry per directory, not per way of naming it
  _seen[$d]=1
  v=$(version_at "$d")
  [ -n "$v" ] || continue
  DIRS+=("$d"); VERS+=("$v")
done

# select_cpu / select_gpu set SEL (directory, empty for none), NOTE (why) and NEED, read by
# tools/requirements.sh.  CPU NEED: ok; unset (BOOST_CPU_ROOT=none); fix-root (BOOST_CPU_ROOT
# names no 1.75.0; no other Boost is used while it is set); fetch (no 1.75.0 found).
select_cpu() {
  SEL=""; NOTE=""; NEED=ok
  if [ -n "${BOOST_CPU_ROOT:-}" ]; then
    if [ "$BOOST_CPU_ROOT" = none ]; then NOTE="disabled (BOOST_CPU_ROOT=none)"; NEED=unset; return; fi
    local cv; cv=$(version_at "$BOOST_CPU_ROOT")
    if [ "$cv" = "$CPU_VERSION" ]; then
      SEL=$(cd "$BOOST_CPU_ROOT" && pwd); NOTE="BOOST_CPU_ROOT"; return
    fi
    if [ -n "$cv" ]; then
      NOTE="BOOST_CPU_ROOT=$BOOST_CPU_ROOT is Boost $(dotted "$cv"); the CPU row is Boost 1.75.0"
    else
      NOTE="BOOST_CPU_ROOT=$BOOST_CPU_ROOT holds no boost/version.hpp"
    fi
    NEED=fix-root; return
  fi
  local i
  for i in "${!DIRS[@]}"; do
    if [ "${VERS[$i]}" -eq "$CPU_VERSION" ]; then
      SEL=${DIRS[$i]}; NOTE="Boost 1.75.0, the version of the published CPU rows"; return
    fi
  done
  if [ "${#DIRS[@]}" -gt 0 ]; then
    local found="" v
    for v in "${VERS[@]}"; do found="${found:+$found, }$(dotted "$v")"; done
    NOTE="no Boost 1.75.0 found (found: $found); the CPU row uses 1.75.0 only"
  else
    NOTE="no Boost 1.75.0 found"
  fi
  NEED=fetch
}
# GPU NEED: ok; unset (BOOST_ROOT=none); fetch (no GPU-capable Boost).  A BOOST_ROOT without
# GPU support (often set for other builds) is reported and passed over for the highest one found.
select_gpu() {
  SEL=""; NOTE=""; NEED=ok
  local pre=""
  if [ -n "${BOOST_ROOT:-}" ]; then
    if [ "$BOOST_ROOT" = none ]; then NOTE="disabled (BOOST_ROOT=none)"; NEED=unset; return; fi
    if gpu_capable "$BOOST_ROOT"; then SEL=$(cd "$BOOST_ROOT" && pwd); NOTE="BOOST_ROOT"; return; fi
    local rv; rv=$(version_at "$BOOST_ROOT")
    if [ -n "$rv" ]; then
      pre="BOOST_ROOT=$BOOST_ROOT is Boost $(dotted "$rv"), which has no GPU support (needs >= 1.87); "
    else
      pre="BOOST_ROOT=$BOOST_ROOT holds no boost/version.hpp; "
    fi
  fi
  local i b=-1 bv=0 hv=0
  for i in "${!DIRS[@]}"; do
    [ "${VERS[$i]}" -gt "$hv" ] && hv=${VERS[$i]}
    if gpu_capable "${DIRS[$i]}" && [ "${VERS[$i]}" -gt "$bv" ]; then bv=${VERS[$i]}; b=$i; fi
  done
  if [ "$b" -ge 0 ]; then
    SEL=${DIRS[$b]}; NOTE="${pre:+${pre}using the }highest Boost with BOOST_MATH_GPU_ENABLED"
  elif [ -n "$pre" ]; then
    NOTE="${pre}no other Boost found has GPU support"; NEED=fetch
  elif [ "$hv" -gt 0 ]; then
    NOTE="${pre}highest Boost found is $(dotted "$hv"), which has no GPU support (needs >= 1.87)"; NEED=fetch
  else
    NOTE="${pre}no Boost found"; NEED=fetch
  fi
}

if [ "$MODE" = select ]; then
  if [ "$ROLE" = cpu ]; then select_cpu; else select_gpu; fi
  echo "$SEL"
  exit 0
fi

for i in "${!DIRS[@]}"; do
  g=""; gpu_capable "${DIRS[$i]}" && g="  (GPU-capable)"
  [ "${VERS[$i]}" -eq "$CPU_VERSION" ] && g="$g  (CPU row version)"
  printf "  found %-45s BOOST_VERSION=%s (%s)%s\n" "${DIRS[$i]}" "${VERS[$i]}" "$(dotted "${VERS[$i]}")" "$g"
done
echo
select_cpu
if [ -n "$SEL" ]; then
  echo "cpu row      : $SEL  ($(dotted "$(version_at "$SEL")"))  -- $NOTE"
else
  echo "cpu row      : none  -- $NOTE"
fi
echo "cpu needs    : $NEED"
select_gpu
if [ -n "$SEL" ]; then
  echo "gpu row      : $SEL  ($(dotted "$(version_at "$SEL")"))  -- $NOTE"
else
  echo "gpu row      : none  -- $NOTE"
fi
echo "gpu needs    : $NEED"
