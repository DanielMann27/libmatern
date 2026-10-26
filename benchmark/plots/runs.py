"""runs.py -- run-directory readers shared by the figure and table generators: STATUS,
schema-checked CSV rows, machine names from PROVENANCE.txt and the accuracy-detail
histograms."""
import csv, os, platform, re

# style imports this module only inside a function, so there is no import cycle.
import style as _style

SCHEMA_ACCEPTED = (10, 11)


def generator_versions():
    """[(program, version)] of the Python, NumPy and Matplotlib of this process; version
    strings only, never a path."""
    out = [("Python", platform.python_version())]
    for name, module in (("NumPy", "numpy"), ("Matplotlib", "matplotlib")):
        try:
            out.append((name, str(__import__(module).__version__)))
        except Exception:
            out.append((name, "not importable"))
    return out


def read_status(run_dir):
    """Return (status, missing).  A run without a STATUS file is `unknown`."""
    p = os.path.join(run_dir, "STATUS")
    if not os.path.exists(p):
        return "unknown", "no STATUS file"
    status, missing = "unknown", ""
    for line in open(p):
        f = line.split(None, 1)
        if not f:
            continue
        if f[0] == "STATUS":
            status = f[1].strip() if len(f) > 1 else "unknown"
        elif f[0] == "missing":
            missing = f[1].strip() if len(f) > 1 else ""
    return status, missing


def _cpu_gpu(run_dir):
    """The `cpu` and `gpu` values recorded in PROVENANCE.txt ("" if absent)."""
    p = os.path.join(run_dir, "PROVENANCE.txt")
    cpu = gpu = ""
    if os.path.exists(p):
        for line in open(p):
            f = line.split(None, 1)
            if len(f) == 2 and f[0] == "cpu":
                cpu = f[1].strip()
            elif len(f) == 2 and f[0] == "gpu":
                gpu = f[1].strip()
    return cpu, gpu


def run_label(run_id):
    """A run as named in generated text: `<stamp>_<host>_<label>` reads `<stamp> <label>`;
    an id of any other form is returned unchanged."""
    parts = str(run_id).split("_")
    if len(parts) < 3:
        return str(run_id)
    return parts[0] + " " + "_".join(parts[2:])


# PROVENANCE `cpu` values that name no model: the probe's failure sentinel and `uname -p` output.
_NO_MODEL = ("unknown", "x86_64", "amd64", "i386", "i686", "aarch64", "arm64", "arm")


def cpu_recorded(cpu):
    """The CPU model string, or "" when the recorded value names no model."""
    c = str(cpu or "").strip()
    if c.lower().startswith("unavailable") or c.lower() in _NO_MODEL:
        return ""
    return c


def machine_slug(cpu):
    """The machine label of a CPU model, e.g. `AMD EPYC 9454 48-Core Processor` ->
    `epyc-9454`; "" when nothing is left."""
    s = re.sub(r"\((R|TM|r|tm)\)", " ", cpu_recorded(cpu))
    model, _, clock = s.partition("@")
    model = re.sub(r"\b\d+-Core\b", " ", model, flags=re.I)
    words = [w for w in model.split()
             if w.lower() not in ("intel", "amd", "cpu", "processor")]
    if not re.search(r"\d", " ".join(words)):
        words += clock.split()
    return re.sub(r"[^a-z0-9]+", "-", " ".join(words).lower()).strip("-")


def machine_key(run_dir, run_id):
    """The machine key of a run: `machine_slug` of its CPU model, shared by hosts of one
    model; a run without a recorded model gets a key of its own from its label."""
    slug = machine_slug(_cpu_gpu(run_dir)[0])
    if slug:
        return slug
    return "cpu-not-recorded-" + re.sub(r"[^a-z0-9]+", "-",
                                        run_label(run_id).lower()).strip("-")


def dedicated(pv):
    """`yes`, `no` or `unknown` from PROVENANCE `exclusive` (the run's CPUs held no other
    process); "" when not recorded."""
    v = (pv or {}).get("exclusive") or ""
    return v.split(" --")[0].strip()


def arch_label(run_dir, run_id):
    """(short, pretty) machine labels from the recorded CPU model; nothing is inferred."""
    cpu, gpu = _cpu_gpu(run_dir)
    cpu = cpu_recorded(cpu)
    if not cpu:
        return "unknown", "CPU model not recorded"
    for _junk in ("(R)", "(TM)", "(r)", "(tm)"):
        cpu = cpu.replace(_junk, "")
    cpu = cpu.replace(" CPU @ ", " @ ")
    cpu = " ".join(cpu.split())
    low = cpu.lower()
    if "amd" in low or "epyc" in low:
        short = "AMD"
    elif "intel" in low or "xeon" in low:
        short = "Intel"
    else:
        short = cpu.split()[0]
    pretty = cpu
    # run_benchmark.sh records an absent GPU as a sentence starting with "none".
    if gpu and not gpu.lower().startswith("none"):
        pretty += f" + {gpu}"
        short += "+GPU"
    return short, pretty


def cpu_arch_key(label):
    """The CPU architecture of an `arch_label` short tag (`Intel+GPU` -> `Intel`), the key
    of cross-machine comparisons."""
    return (label or "").split("+")[0].strip()


def machine_name(label, gpu=False):
    """(name, kind) for a caption, e.g. ("AMD", "CPU") or, with `gpu` and a device in the
    label, ("A100", "GPU"); ("", "") when the label names no model."""
    s = " ".join(str(label or "").split())
    if gpu:
        dev = (s.split(" on ", 1)[0] if " on " in s else
               s.split(" + ", 1)[1] if " + " in s else "")
        words = [w for w in dev.split()
                 if w.lower() not in ("nvidia", "tesla", "geforce", "quadro")]
        if words:
            name = words[0].split("-")[0]
            if not re.search(r"\d", name) and len(words) > 1:
                name += " " + words[1].split("-")[0]
            return name, "GPU"
    cpu = s.split(" on ", 1)[-1].split(" + ", 1)[0].split("(")[0].strip()
    if not cpu or cpu == "CPU model not recorded":
        return "", ""
    low = cpu.lower()
    if "amd" in low or "epyc" in low:
        return "AMD", "CPU"
    if "intel" in low or "xeon" in low:
        return "Intel", "CPU"
    return cpu.split()[0], "CPU"


class SchemaMismatch(Exception):
    """A run CSV whose rows disagree with its header or the accepted schemas."""


def read_csv(path):
    """The rows of one run CSV ([] if missing); SchemaMismatch on a wrong field count or
    schema.  csv.DictReader pads short rows and keys surplus fields under None."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        header = rdr.fieldnames or []
        rows = []
        for r in rdr:
            if not r.get("schema"):
                continue
            if None in r or any(v is None for v in r.values()):
                got = (len(header) + len(r.get(None) or [])) if None in r else \
                      sum(1 for v in r.values() if v is not None)
                raise SchemaMismatch(
                    f"{os.path.basename(path)} line {rdr.line_num}: "
                    f"{got} fields against a {len(header)}-field header. "
                    f"The file mixes two schemas and cannot be read. "
                    f"Re-run the stage that wrote it.")
            if "supplier" in r:
                r["supplier"] = _style.LEGACY_SUPPLIER.get(r["supplier"],
                                                           r["supplier"])
            if "family" in r:
                r["family"] = _style.LEGACY_FAMILY.get(r["family"], r["family"])
            rows.append(r)
    for r in rows:
        try:
            s = int(r.get("schema", -1))
        except ValueError:
            continue
        if s not in SCHEMA_ACCEPTED:
            raise SchemaMismatch(
                f"{os.path.basename(path)} is schema {s}, this script reads "
                f"{', '.join(map(str, SCHEMA_ACCEPTED))}. Re-run the benchmark or use "
                f"the matching script.")
    if "ladder" in os.path.basename(path):
        for r in rows:
            r.setdefault("ladder_mode", "rung")
    return rows


# Accuracy detail sidecars (bench_metrics.h bench_detail_emit), plain or .gz, one per scoring process:
#     accuracy_detail.<backend> (accuracy.csv), .suppliers.<supplier> (suppliers.csv),
#     .gpu_kernels (gpu_kernels.csv), .gpu (gpu.csv); the Levin A/B sidecars are not read.
# A block is keyed like its accuracy row, (device, supplier, family, law, backend), per run.
# DETAIL_HEADER must match the header bench_detail_open writes.
DETAIL_HEADER = ["device", "supplier", "family", "law", "backend", "kind", "key",
                 "sign", "lo_log2", "hi_log2", "count", "ulp", "nu", "arg",
                 "aux", "value", "ref", "bits", "index", "reason"]
DETAIL_SKIP = ("accuracy_detail.levin_", "accuracy_detail.gpu_levin_")

# Histogram layout; must match bench_metrics.h BENCH_HIST_* (bin 0 zero, bin 49 overflow).
HIST_BINS = 48
HIST_LO_LOG = -10.0
HIST_HI_LOG = 10.0

_DETAIL_CACHE = {}


def hist_edges(b):
    """(lo_log2, hi_log2) of histogram bin `b`, as bench_hist_bin assigns it."""
    inf = float("inf")
    step = (HIST_HI_LOG - HIST_LO_LOG) / HIST_BINS
    if b == 0:
        return (-inf, -inf)
    if b == 1:
        return (-inf, HIST_LO_LOG + step)
    if b == HIST_BINS + 1:
        return (HIST_HI_LOG, inf)
    lo = HIST_LO_LOG + (b - 1) * step
    return (lo, lo + step)


def _same_edge(a, b):
    if a == b:                       # also both -inf or both inf
        return True
    return abs(a - b) <= 1e-9


def _open_detail(path):
    import gzip
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def read_accuracy_detail(run_dir):
    """(index, problems) for the accuracy detail sidecars in `run_dir`, cached per directory.

    index maps (device, supplier, family, law, backend) to entries {file, counts (|ulp| per
    bin), pos, neg, total, repeated, incomplete, worst_ref (reference at the rank-0 worst
    point)}; problems lists the unreadable files, which contribute nothing.
    """
    run_dir = os.path.abspath(run_dir)
    if run_dir in _DETAIL_CACHE:
        return _DETAIL_CACHE[run_dir]
    index, problems = {}, []
    try:
        names = sorted(os.listdir(run_dir))
    except OSError as e:
        names = []
        problems.append(f"{run_dir}: {e}")
    nb = HIST_BINS + 2
    for fn in names:
        if not fn.startswith("accuracy_detail.") or fn.startswith(DETAIL_SKIP):
            continue
        path = os.path.join(run_dir, fn)
        blocks, bad = {}, ""
        try:
            with _open_detail(path) as fh:
                head = fh.readline().rstrip("\r\n").split(",")
                if head != DETAIL_HEADER:
                    bad = f"header {head} is not the detail schema"
                else:
                    for ln, line in enumerate(fh, 2):
                        if ",worst,0," in line:
                            # the rank-0 worst point follows its block's hist rows
                            f = line.rstrip("\r\n").split(",")
                            if len(f) == len(DETAIL_HEADER) and f[5] == "worst":
                                f[1] = _style.LEGACY_SUPPLIER.get(f[1], f[1])
                                f[2] = _style.LEGACY_FAMILY.get(f[2], f[2])
                                e = blocks.get(tuple(f[0:5]))
                                if e is not None:
                                    try:
                                        e["worst_ref"] = float(f[16])
                                    except ValueError:
                                        pass
                            continue
                        if ",hist," not in line:
                            continue
                        f = line.rstrip("\r\n").split(",")
                        if len(f) != len(DETAIL_HEADER) or f[5] != "hist":
                            bad = f"line {ln}: not a histogram row of the schema"
                            break
                        try:
                            b, c = int(f[6]), int(f[10])
                            lo, hi = float(f[8]), float(f[9])
                        except ValueError:
                            bad = f"line {ln}: unreadable bin, edge or count"
                            break
                        if not 0 <= b < nb or f[7] not in "+-" or c < 0:
                            bad = f"line {ln}: bin {f[6]} sign {f[7]!r} count {f[10]}"
                            break
                        elo, ehi = hist_edges(b)
                        if not (_same_edge(lo, elo) and _same_edge(hi, ehi)):
                            bad = (f"line {ln}: bin {b} has edges [{lo}, {hi}), "
                                   f"the schema's are [{elo}, {ehi})")
                            break
                        # mapped as read_csv maps it, so the keys match the rows
                        f[1] = _style.LEGACY_SUPPLIER.get(f[1], f[1])
                        f[2] = _style.LEGACY_FAMILY.get(f[2], f[2])
                        key = tuple(f[0:5])
                        e = blocks.get(key)
                        if e is None:
                            e = blocks[key] = {"file": fn, "pos": [0] * nb,
                                               "neg": [0] * nb, "seen": set(),
                                               "repeated": False,
                                               "worst_ref": None}
                        if (f[7], b) in e["seen"]:
                            e["repeated"] = True
                        e["seen"].add((f[7], b))
                        (e["pos"] if f[7] == "+" else e["neg"])[b] += c
        except (OSError, EOFError, UnicodeDecodeError) as e:
            bad = f"cannot be read ({e})"
        if bad:
            problems.append(f"{fn}: {bad}")
            continue
        for key, e in blocks.items():
            # bench_detail_emit writes every bin of both signs, empty or not
            e["incomplete"] = len(e["seen"]) != 2 * nb
            del e["seen"]
            e["counts"] = [p + q for p, q in zip(e["pos"], e["neg"])]
            e["total"] = sum(e["counts"])
            index.setdefault(key, []).append(e)
    _DETAIL_CACHE[run_dir] = (index, problems)
    return index, problems


def detail_key(row):
    """The sidecar key of an accuracy row."""
    return tuple((row.get(k) or "").strip()
                 for k in ("device", "supplier", "family", "law", "backend"))


def detail_for(row):
    """(entry, "") for one accuracy row's histogram, or (None, reason).  The row carries its
    run directory in `_dir`; a histogram whose key is absent or repeated, or whose total
    differs from `n_scored`, is refused."""
    d = row.get("_dir")
    key = detail_key(row)
    what = "/".join(key)
    if not d:
        return None, f"{what}: the row carries no run directory"
    index, problems = read_accuracy_detail(d)
    cands = index.get(key, [])
    run = os.path.basename(os.path.abspath(d))
    if not cands:
        why = f"{what}: no histogram in {run}/accuracy_detail.*"
        if problems:
            why += " (unreadable: " + "; ".join(problems) + ")"
        return None, why
    if len(cands) > 1:
        return None, (f"{what}: {len(cands)} histograms in {run} ("
                      + ", ".join(e["file"] for e in cands) + ")")
    e = cands[0]
    if e["repeated"] or e["incomplete"]:
        return None, (f"{what}: the block in {run}/{e['file']} repeats or "
                      f"lacks bins")
    try:
        ns = int(float(row.get("n_scored")))
    except (TypeError, ValueError):
        return None, f"{what}: the row has no n_scored"
    if e["total"] != ns:
        return None, (f"{what}: histogram total {e['total']} differs from "
                      f"n_scored {ns} ({run}/{e['file']})")
    return e, ""

