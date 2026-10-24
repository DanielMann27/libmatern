#!/bin/bash
# fetch.sh -- fetches the Bessels.jl (MIT) upstream sources into bessels_jl/upstream/
# (make references) and verifies every file against its SHA-256 digest; a download
# replaces a file only when its digest matches.
set -u
cd "$(dirname "$0")" || exit 1

# Bessels.jl: the pinned commit, and the commit of release 0.2.8 (tag v0.2.8),
# whose src/besselk.jl cpu_bessels_old.c ports (kept as besselk_old.jl).
BESSELS_COMMIT=009b615347d91d1dd69962641647521901362af3
BESSELS_V028=1475e9e57464dbb7b8ba2d812766e34d0f822c4d
BESSELS_URL=https://raw.githubusercontent.com/JuliaMath/Bessels.jl

# local name | commit | upstream path | sha256
BESSELS_FILES="
besselk.jl|$BESSELS_COMMIT|src/BesselFunctions/besselk.jl|3560c95f769af60cfba49d932d4ce6e6e041b01128cb00bfd4b6a480066695b1
besseli.jl|$BESSELS_COMMIT|src/BesselFunctions/besseli.jl|99df3dcd86ccba44186279da13659e66db6e4d205e6f99290f3ce2de86e18fb6
constants.jl|$BESSELS_COMMIT|src/BesselFunctions/constants.jl|92936551ce9563e112ff538d0b5de40014a695d9f7cec23ce1924dd43e58fa07
U_polynomials.jl|$BESSELS_COMMIT|src/BesselFunctions/U_polynomials.jl|87e3cee08cf2cda5d763b4c10f11d5a0dce112179736913f1388485ce6eafb4c
recurrence.jl|$BESSELS_COMMIT|src/BesselFunctions/recurrence.jl|71536a72dd8f752203d7cf4a66fba4cbdee795fd44e1a75c719d483d2e385322
sphericalbessel.jl|$BESSELS_COMMIT|src/BesselFunctions/sphericalbessel.jl|a8e55b8c7729c08bf7c8d7e700e5a0ba86b0e94951cfab9361d55cb1eeb18a30
modifiedsphericalbessel.jl|$BESSELS_COMMIT|src/BesselFunctions/modifiedsphericalbessel.jl|7dcd41ef5c5350cdf712dda6c1eeae6168fc4854f9bc20158e640af29eaa51b5
Math.jl|$BESSELS_COMMIT|src/Math/Math.jl|0541b470fecdff0de96462c653568c6055e9738ac77091f349ec0423906dc982
math_constants.jl|$BESSELS_COMMIT|src/Math/math_constants.jl|39a9baae6be2bbc0b99a5d28202d3c1775a667e21df82e561975094d7e5c7e82
Bessels.jl|$BESSELS_COMMIT|src/Bessels.jl|e1ea3129c05e27586ba99d20a695607a1fcbf6bb3c7ad26e31306d8b2e0f9426
BF.jl|$BESSELS_COMMIT|src/BesselFunctions/BesselFunctions.jl|6a634c1d66e8f297acfec67b43bbeaf6ebe625482c4bef683288282ffd6ed372
LICENSE|$BESSELS_COMMIT|LICENSE|1ab0da1b8fd834885da485f90a5093fd1ce7430a44459472721589d10eb2a384
besselk_old.jl|$BESSELS_V028|src/besselk.jl|e684931f3283258d74b4b8661836ee4cf0ab99a14f13c2acbdf735ce55292b5f
"

# KAUST CuBesselK (all rights reserved): verified if present, never downloaded here
# (`make kaust` prints its copyright notice).
KAUST_FILES="
logbesselk.cu|5a51e1a5c7c08ecc31696db58d3a69e9fc9e3fe6d8071e57af26dcaa666ee5ed
logbesselk.h|650bafdad9c4444992fbf1b5435bc2d16db0e70fff3135066d79f08eeea30550
"

fail=0
digest() { sha256sum "$1" | cut -d' ' -f1; }

if command -v curl >/dev/null 2>&1; then
    get() { curl -sSfL --connect-timeout 20 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    get() { wget -q --connect-timeout=20 --tries=1 -O "$2" "$1"; }
else
    echo "fetch.sh: neither curl nor wget is available" >&2; exit 1
fi

mkdir -p bessels_jl/upstream || exit 1
for entry in $BESSELS_FILES; do
    IFS='|' read -r name ref path sum <<< "$entry"
    out=bessels_jl/upstream/$name
    if [ -s "$out" ] && [ "$(digest "$out")" = "$sum" ]; then
        echo "  verified  $out"; continue
    fi
    if ! get "$BESSELS_URL/$ref/$path" "$out.tmp"; then
        echo "  failed    $out"; rm -f "$out.tmp"; fail=1; continue
    fi
    got=$(digest "$out.tmp")
    if [ "$got" = "$sum" ]; then
        mv "$out.tmp" "$out" || { echo "  failed    $out"; rm -f "$out.tmp"; fail=1; continue; }
        echo "  verified  $out"
    else
        rm -f "$out.tmp"
        echo "  mismatch  $out: downloaded sha256 $got, recorded $sum"; fail=1
    fi
done

# bessels_jl/LICENSE, shipped with the ports, must equal the verified upstream LICENSE.
lic=bessels_jl/upstream/LICENSE
lic_sum=$(printf '%s\n' $BESSELS_FILES | awk -F'|' '$1 == "LICENSE" { print $NF }')
if [ -s "$lic" ] && [ "$(digest "$lic")" = "$lic_sum" ]; then
    if cmp -s "$lic" bessels_jl/LICENSE; then
        echo "  verified  bessels_jl/LICENSE (identical to $lic)"
    else
        echo "  mismatch  bessels_jl/LICENSE differs from the upstream LICENSE"; fail=1
    fi
fi

# Where `make kaust DOWNLOAD=yes` puts the KAUST files; not distributed.
mkdir -p kaust_logbesselk/upstream || fail=1
for entry in $KAUST_FILES; do
    IFS='|' read -r name sum <<< "$entry"
    out=kaust_logbesselk/upstream/$name
    if [ ! -e "$out" ]; then
        echo "  absent    $out (not distributed; make kaust)"
    elif [ "$(digest "$out")" = "$sum" ]; then
        echo "  verified  $out"
    else
        echo "  mismatch  $out"; fail=1
    fi
done

exit $fail
