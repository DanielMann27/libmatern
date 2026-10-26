"""py_stack.py [--pins|--floors] -- the NumPy and Matplotlib of the figure and table
generators.  Without an option it prints `numpy <v> matplotlib <v>` for this interpreter and
exits 0 when both are at or above PY_FLOORS, 1 when one is below (the line then names it),
2 when one does not import.  --pins and --floors print the two requirement strings, which
make externals installs into .venv/ (tools/externals.sh).  The Makefile's search for
PYTHON, make requirements and make externals all decide with this script."""
import sys

# PY_PINS: the versions that drew the published figures.  PY_FLOORS: the oldest versions the
# generators are tested with; an interpreter below them is not used.
PY_PINS = "numpy==2.2.5 matplotlib==3.11.0"
PY_FLOORS = "numpy>=2.0.2 matplotlib>=3.9.4"


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


def main(argv):
    if argv[1:] == ["--pins"]:
        print(PY_PINS)
        return 0
    if argv[1:] == ["--floors"]:
        print(PY_FLOORS)
        return 0
    have = {}
    for name in ("numpy", "matplotlib"):
        try:
            have[name] = __import__(name).__version__
        except Exception:
            print("no %s" % name)
            return 2
    line = "numpy %s matplotlib %s" % (have["numpy"], have["matplotlib"])
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
