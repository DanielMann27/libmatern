"""py_stack.py [--pins|--floors|--exact|--base [python...]|--advice] -- the NumPy and
Matplotlib of the figure and table generators, and the Python they need.

Without an option it prints `numpy <v> matplotlib <v>` for this interpreter and exits 0 when
both are at or above PY_FLOORS, 1 when one is below (the line then names it), 2 when one does
not import.  --exact exits 0 only when both are the PY_PINS versions.  --pins and --floors print
the two requirement strings, which make externals installs into .venv/ (tools/externals.sh).

--base finds the interpreter make externals builds .venv/ from: PYTHON_BASE when set, else the
given interpreters, else BASE_CANDIDATES; the newest Python at or above PY_PINS_PYTHON with the
venv module, else the newest at or above PY_FLOORS_PYTHON.  It prints `pins <path> <version>`
(exit 0), `floors <path> <version>` (exit 1) or `none <reason>` (exit 2).  --advice prints, from
--base, the one next step that gets the generators a NumPy and Matplotlib: make externals where
it can install them, else the Python to install first.

The Makefile's search for PYTHON, make requirements and make externals all decide with this
script."""
import os, shutil, subprocess, sys

# PY_PINS: the versions that drew the published figures.  PY_FLOORS: the oldest versions the
# generators are tested with; an interpreter below them is not used.
PY_PINS = "numpy==2.2.5 matplotlib==3.11.0"
PY_FLOORS = "numpy>=2.0.2 matplotlib>=3.9.4"
# The Python the pins install on (Matplotlib 3.11.0 and contourpy 1.3.3 require 3.11, NumPy
# 2.2.5 3.10), the one the floors install on (NumPy 2.0.2 and Matplotlib 3.9.4: 3.9), and the
# Python of the published figures.
PY_PINS_PYTHON = "3.11"
PY_FLOORS_PYTHON = "3.9"
PY_PUBLISHED = "3.12.9"
# The interpreters make externals may build .venv/ from when PYTHON_BASE is not set.
BASE_CANDIDATES = "python3.14 python3.13 python3.12 python3.11 python3 /usr/bin/python3"
# One line per candidate: its version first, then whether .venv/ can be made from it.
PROBE = "import sys; print('%d.%d.%d' % sys.version_info[:3]); import venv, ensurepip"


def release(version):
    """The leading numbers of a version string: `2.0.0rc1` -> (2, 0, 0)."""
    out = []
    for part in version.split(".")[:3]:
        digits = ""
        for c in part:
            if not c.isdigit():
                break
            digits += c
        out.append(int(digits or 0))
        if digits != part:
            break
    return tuple(out)


def base(candidates):
    """(kind, path, version, reason) of the interpreter .venv/ is built from: kind `pins`,
    `floors` or `none` (path and version empty, reason why)."""
    seen, found, notes = set(), [], []
    for c in candidates:
        path = shutil.which(c)
        if not path:
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        try:
            r = subprocess.run([path, "-c", PROBE], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        version = (r.stdout.split() or [""])[0]
        if not version[:1].isdigit():
            continue
        if r.returncode != 0:
            notes.append("%s (Python %s) has no venv module" % (path, version))
            continue
        found.append((release(version), path, version))
    for kind, need in (("pins", PY_PINS_PYTHON), ("floors", PY_FLOORS_PYTHON)):
        ok = [f for f in found if f[0] >= release(need)]
        if ok:
            _, path, version = max(ok, key=lambda f: f[0])
            return kind, path, version, ""
    notes += ["%s is Python %s" % (p, v) for _, p, v in found]
    reason = ("no Python %s or newer with the venv module: %s"
              % (PY_FLOORS_PYTHON, "; ".join(notes) if notes else "no candidate found"))
    return "none", "", "", reason


def advice(kind, path, version, reason):
    """The next step for the generators' NumPy and Matplotlib, after base()."""
    if kind == "pins":
        return ("make externals (installs the published versions, %s, into .venv/ from Python %s, %s)"
                % (PY_PINS, version, path))
    install = ("install Python %s or newer (the published figures used %s; make requirements "
               "names the package)" % (PY_PINS_PYTHON, PY_PUBLISHED))
    if kind == "floors":
        return ("make externals (installs NumPy and Matplotlib at or above the tested minimum, "
                "%s, into .venv/ from Python %s, %s); for the published versions, %s, then "
                "make externals" % (PY_FLOORS, version, path, install))
    return "%s, then make externals (%s)" % (install, reason)


def base_candidates(argv):
    if os.environ.get("PYTHON_BASE"):
        return [os.environ["PYTHON_BASE"]]
    return argv or BASE_CANDIDATES.split()


def main(argv):
    if argv[1:] == ["--pins"]:
        print(PY_PINS)
        return 0
    if argv[1:] == ["--floors"]:
        print(PY_FLOORS)
        return 0
    if argv[1:2] == ["--base"]:
        kind, path, version, reason = base(base_candidates(argv[2:]))
        print("%s %s %s" % (kind, path, version) if kind != "none" else "none " + reason)
        return {"pins": 0, "floors": 1}.get(kind, 2)
    if argv[1:] == ["--advice"]:
        print(advice(*base(base_candidates([]))))
        return 0
    exact = argv[1:] == ["--exact"]
    have = {}
    for name in ("numpy", "matplotlib"):
        try:
            have[name] = __import__(name).__version__
        except Exception:
            print("no %s" % name)
            return 2
    line = "numpy %s matplotlib %s" % (have["numpy"], have["matplotlib"])
    if exact:
        pins = dict(p.split("==") for p in PY_PINS.split())
        same = all(have[name] == pins[name] for name in pins)
        print(line + ("" if same else ", not the published %s" % PY_PINS))
        return 0 if same else 1
    low = []
    for req in PY_FLOORS.split():
        name, floor = req.split(">=")
        if release(have[name]) < release(floor):
            low.append("%s %s" % (name, floor))
    if low:
        print("%s, below the tested minimum (%s)" % (line, ", ".join(low)))
        return 1
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
