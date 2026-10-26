#!/bin/bash
# boost_detect.sh [--select|--where cpu|gpu] -- finds Boost installations and reports them, or
# prints the include directory selected for the CPU row (1.75.0 only) or the GPU row (>= 1.87,
# with BOOST_MATH_GPU_ENABLED); BOOST_CPU_ROOT / BOOST_ROOT override them, and none disables a
# row.  --where prints "<dir> (<origin>)" for the build records (references/Makefile, the
# Makefile) and tools/externals.sh: the origin is the override (BOOST_CPU_ROOT, BOOST_ROOT)
# where it selects the directory, else the first way the directory is found: make externals
# (references/boost/, which make boost-headers also fills), the variable that names it, or
# system (the compiler's search path, Homebrew or a standard directory).
set -u
MODE=report
ROLE=""
case "${1:-}" in
    --select|--where) MODE=${1#--}; ROLE="${2:-}"
              case "$ROLE" in cpu|gpu) ;; *) echo "boost_detect.sh: $1 cpu|gpu" >&2; exit 2 ;; esac ;;
    "") ;;
    *) echo "usage: boost_detect.sh [--select|--where cpu|gpu]" >&2; exit 2 ;;
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

# Candidates in search order, each with its origin (FROMS, for --where).  First the Boost a
# bare compile resolves, which includes paths that environment modules add without setting
# BOOST_ROOT.
CANDIDATES=(); FROMS=()
cand() { CANDIDATES+=("$1"); FROMS+=("$2"); }
_resolved=$(echo '#include <boost/version.hpp>' \
            | ${CXX:-g++} -E -x c++ - 2>/dev/null \
            | sed -n 's/^# *[0-9][0-9]* *"\(.*\)\/boost\/version\.hpp".*/\1/p' \
            | head -1)
[ -n "$_resolved" ] && cand "$_resolved" system
for _n in BOOST_DIR BOOST_ROOT BOOST_CPU_ROOT BOOST_INC_DIR BOOST_INCLUDE_DIR; do
    _v=${!_n:-}
    [ -n "$_v" ] && [ "$_v" != none ] && cand "$(incroot "$_v")" "$_n"
done
for _d in "$HERE"/references/boost/*/; do
    [ -d "$_d" ] && cand "${_d%/}" "make externals"
done
[ -n "${CONDA_PREFIX:-}" ] && cand "$CONDA_PREFIX/include" CONDA_PREFIX
command -v brew >/dev/null 2>&1 && cand "$(brew --prefix 2>/dev/null)/include" system
# the compiler's own C++ search path, in its own order
while read -r _d; do
  case "$_d" in /*) cand "$_d" system;; esac
done < <(${CXX:-g++} -E -x c++ -v /dev/null 2>&1 \
         | sed -n '/^#include <...> search starts here:/,/^End of search list/p' \
         | sed -n 's/^ \(\/.*\)$/\1/p')
for _d in /usr/include /usr/local/include /opt/homebrew/include /opt/local/include; do
  cand "$_d" system
done

version_at() { sed -n 's/^#define BOOST_VERSION  *\([0-9]*\).*/\1/p' "$1/boost/version.hpp" 2>/dev/null | head -1; }
gpu_capable() { grep -q BOOST_MATH_GPU_ENABLED "$1/boost/math/special_functions/bessel.hpp" 2>/dev/null; }
dotted() { echo "$(($1/100000)).$(($1/100%1000)).$(($1%100))"; }

# A directory found in several ways keeps the origin of the first.
DIRS=(); VERS=(); ORIGS=()
declare -A _seen=()
for i in "${!CANDIDATES[@]}"; do
  d=$(cd "${CANDIDATES[$i]}" 2>/dev/null && pwd) || continue
  [ -n "${_seen[$d]:-}" ] && continue        # one entry per directory, not per way of naming it
  _seen[$d]=1
  v=$(version_at "$d")
  [ -n "$v" ] || continue
  DIRS+=("$d"); VERS+=("$v"); ORIGS+=("${FROMS[$i]}")
done

# select_cpu / select_gpu set SEL (directory, empty for none), ORIG (its origin), NOTE (why)
# and NEED, read by tools/requirements.sh.  CPU NEED: ok; unset (BOOST_CPU_ROOT=none);
# fix-root (BOOST_CPU_ROOT names no 1.75.0; no other Boost is used while it is set); fetch
# (no 1.75.0 found).
select_cpu() {
  SEL=""; ORIG=""; NOTE=""; NEED=ok
  if [ -n "${BOOST_CPU_ROOT:-}" ]; then
    if [ "$BOOST_CPU_ROOT" = none ]; then NOTE="disabled (BOOST_CPU_ROOT=none)"; NEED=unset; return; fi
    local cv; cv=$(version_at "$BOOST_CPU_ROOT")
    if [ "$cv" = "$CPU_VERSION" ]; then
      SEL=$(cd "$BOOST_CPU_ROOT" && pwd); ORIG=BOOST_CPU_ROOT; NOTE="BOOST_CPU_ROOT"; return
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
      SEL=${DIRS[$i]}; ORIG=${ORIGS[$i]}; NOTE="Boost 1.75.0, the version of the published CPU rows"; return
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
  SEL=""; ORIG=""; NOTE=""; NEED=ok
  local pre=""
  if [ -n "${BOOST_ROOT:-}" ]; then
    if [ "$BOOST_ROOT" = none ]; then NOTE="disabled (BOOST_ROOT=none)"; NEED=unset; return; fi
    if gpu_capable "$BOOST_ROOT"; then SEL=$(cd "$BOOST_ROOT" && pwd); ORIG=BOOST_ROOT; NOTE="BOOST_ROOT"; return; fi
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
    SEL=${DIRS[$b]}; ORIG=${ORIGS[$b]}; NOTE="${pre:+${pre}using the }highest Boost with BOOST_MATH_GPU_ENABLED"
  elif [ -n "$pre" ]; then
    NOTE="${pre}no other Boost found has GPU support"; NEED=fetch
  elif [ "$hv" -gt 0 ]; then
    NOTE="${pre}highest Boost found is $(dotted "$hv"), which has no GPU support (needs >= 1.87)"; NEED=fetch
  else
    NOTE="${pre}no Boost found"; NEED=fetch
  fi
}

if [ "$MODE" != report ]; then
  if [ "$ROLE" = cpu ]; then select_cpu; else select_gpu; fi
  if [ "$MODE" = where ] && [ -n "$SEL" ]; then echo "$SEL ($ORIG)"; else echo "$SEL"; fi
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
