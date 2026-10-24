#!/bin/bash
# gsl_flags.sh --exists|--cflags|--libs|--modversion|--where -- the GSL the build uses, for
# references/Makefile and tools/requirements.sh: the highest one built by make externals
# (references/gsl/<version>/, no pkg-config needed), else pkg-config's.  --exists: 0 if found.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mode="${1:---exists}"

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
    *) echo "gsl_flags.sh: unknown mode $mode" >&2; exit 2 ;;
esac
