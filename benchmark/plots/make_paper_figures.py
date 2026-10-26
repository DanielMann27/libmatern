#!/usr/bin/env python3
"""make_paper_figures.py -- builds the paper figure set from the recorded runs in
benchmark/runs/ and measures nothing.  Runs from hosts of one CPU model are pooled into one
machine, and each kind of result comes from that machine's newest run that has it;
--paper writes into plots/paper-final/ with captions.tex.  By default only the paper set
is drawn (PAPER_FIGURES); --full draws every figure."""
import argparse, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
RUNS = os.path.join(BENCH, "runs")
OUT = os.path.join(BENCH, "plots", "paper")
sys.path.insert(0, HERE)

import runs as mp
import style
import paper_figures as pf


FILES = {
    "accuracy": "accuracy.csv", "suppliers": "suppliers.csv",
    "ladder": "ladder.csv", "sup_ladder": "suppliers_ladder.csv",
    "gpu": "gpu.csv", "gpu_sup": "gpu_kernels.csv",
    "gpu_ladder": "gpu_ladder.csv", "gpu_klad": "gpu_kernels_ladder.csv",
    # Optional stages; an absent file reads as no rows.
    "levin": "levin.csv", "gpu_levin": "gpu_levin.csv", "fold": "fold.csv",
    "threads": "threads.csv", "gpu_threads": "gpu_threads.csv",
    "gpu_boost": "gpu_boost_shortcut.csv",
}

# Kinds of a CPU run; a GPU-pipeline run holds them only as a one-backend reference.
CPU_KINDS = ("accuracy", "suppliers", "ladder", "sup_ladder", "levin", "fold",
             "threads")

UNREADABLE = []


def read_raw_reps(d, rid):
    """Every per-repetition timing sample (`timing_raw.*`, written under the BENCH_TIMING_RAW
    prefix by common/bench_timing.h) in one run directory, concatenated: for example
    `timing_raw.<backend>`, `.suppliers.<supplier>`, `.suppliers_ladder.<supplier>`,
    `.ladder.<backend>`, `.gpu*`; the `.err` and `.sched.csv` files are skipped.
    mp.read_csv would skip these rows (no `schema` column), so the header is checked
    against RAW_HEADER; a mismatch goes to UNREADABLE."""
    RAW_HEADER = ["device", "supplier", "family", "law", "backend",
                  "n", "rep", "ns_per_elem"]
    out = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for fn in names:
        if not fn.startswith("timing_raw.") or fn.endswith((".err", ".sched.csv")):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path, newline="") as fh:
                rdr = csv.DictReader(fh)
                if (rdr.fieldnames or []) != RAW_HEADER:
                    UNREADABLE.append(
                        f"{rid}/{fn}: header {rdr.fieldnames} is not the "
                        f"per-repetition schema {RAW_HEADER}")
                    continue
                for r in rdr:
                    if None in r or any(v is None for v in r.values()):
                        UNREADABLE.append(
                            f"{rid}/{fn} line {rdr.line_num}: field count "
                            f"disagrees with the header")
                        break
                    # mapped as mp.read_csv maps it, so samples join their summary rows
                    r["supplier"] = style.LEGACY_SUPPLIER.get(r["supplier"],
                                                              r["supplier"])
                    r["family"] = style.LEGACY_FAMILY.get(r["family"], r["family"])
                    r["_run"] = rid
                    out.append(r)
        except OSError:
            pass
    return out


def load_all():
    """Every run directory under RUNS, whatever its STATUS; an unreadable CSV goes to
    UNREADABLE and reads as no rows."""
    runs = []
    if not os.path.isdir(RUNS):
        return runs
    for rid in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, rid)
        # `runs/latest` is a symlink to the newest run; skipped so no run counts twice.
        if not os.path.isdir(d) or os.path.islink(d):
            continue
        st, missing = mp.read_status(d)
        short, pretty = mp.arch_label(d, rid)
        pv = read_provenance(d)
        # A GPU-pipeline run (run_gpu.sh): PROVENANCE `pipeline gpu`, or an id ending in `_gpu`.
        _gp = pv.get("pipeline", "").strip() == "gpu" or rid.endswith("_gpu")
        rec = {"id": rid, "dir": d, "status": st, "missing": missing, "gpu_pipeline": _gp,
               "arch_short": short, "arch_pretty": pretty, "pv": pv,
               "machine": mp.machine_key(d, rid), "label": mp.run_label(rid)}
        rec["raw"] = read_raw_reps(d, rid)
        for key, fn in FILES.items():
            try:
                rec[key] = mp.read_csv(os.path.join(d, fn))
                # `_run` keeps pooled stages apart when samples join summary rows (the
                # Levin A/B re-times the same keys); `_dir` locates the detail sidecars.
                for _r in rec[key]:
                    _r["_run"] = rid
                    _r["_dir"] = d
            except mp.SchemaMismatch as e:
                rec[key] = []
                UNREADABLE.append(f"{rid}/{fn}: {e}")
        # Outside FILES, so newest_per_stage leaves it with this run: the GPU figures compare
        # against the CPU ladder of the same run.
        rec["gpu_cpu_ladder"] = rec["ladder"] if _gp else []
        runs.append(rec)
    # Colliding labels (same stamp and stage) add the machine, then a run number.
    seen = {}
    for rec in runs:
        seen.setdefault(rec["label"], []).append(rec)
    for lab, recs in seen.items():
        if len(recs) < 2:
            continue
        for i, rec in enumerate(recs, 1):
            twins = [x for x in recs if x["machine"] == rec["machine"]]
            rec["label"] = (f"{lab} ({rec['machine']})" if len(twins) < 2
                            else f"{lab} ({rec['machine']}, {i})")
    return runs


def device_label(rs):
    """The GPU model from the `device` line the GPU stages write to `gpu.err` and its
    siblings; "" if absent."""
    for r in rs:
        for fn in ("gpu.err", "gpu_ladder.err", "gpu_kernels.err"):
            path = os.path.join(r["dir"], fn)
            if not os.path.exists(path):
                continue
            for line in open(path, errors="replace"):
                if "device" in line.lower():
                    return line.split("device", 1)[1].strip(" :\n")
    return ""


def machine_of(r):
    """The machine of a run: runs.machine_key of its CPU model, not its host; runs of one
    model are one machine, joined by newest_per_stage."""
    return r.get("machine") or mp.machine_key(r["dir"], r["id"])


def machine_order(bymach):
    """Machine labels in listing order: one with a GPU-pipeline run first, so the shared
    tables (from the first CPU machine) describe the device tables' CPU; then by label."""
    return sorted(bymach, key=lambda m: (
        not any(r.get("gpu_pipeline") for r in bymach[m]), m))


def kind_owner(rs):
    """{kind: run} for one machine after newest_per_stage (one run per kind)."""
    return {k: r for r in rs for k in FILES if r.get(k)}


def source_runs(rs, *kinds):
    """The runs of one machine that supply `kinds`, in run order; all of them when none does."""
    own = kind_owner(rs)
    ids = {own[k]["id"] for k in kinds if k in own}
    return [r for r in rs if r["id"] in ids] or list(rs)


def run_stamp(runs_):
    """The provenance stamp of a figure: the labels of its runs."""
    st = "" if all(x["status"] == "complete" for x in runs_) else "  [PARTIAL RUN]"
    return " + ".join(x["label"] for x in runs_) + st


def machine_pretty(rs):
    """The CPU model of a machine, followed by the GPU of any run that has one."""
    cpu = rs[0]["arch_pretty"].split("[")[0].split(" + ")[0].strip()
    gpus = []
    for r in rs:
        parts = r["arch_pretty"].split("[")[0].split(" + ", 1)
        if len(parts) == 2 and parts[1].strip() not in gpus:
            gpus.append(parts[1].strip())
    return " + ".join([cpu] + gpus)


def read_provenance(run_dir):
    """PROVENANCE.txt as a dict.  `key<spaces>value` per line, value may be empty."""
    out = {}
    path = os.path.join(run_dir, "PROVENANCE.txt")
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split(None, 1)
        out[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return out


# The Boost and GSL copy behind the CPU rows and the Boost behind the GPU rows, recorded as
# `<dir> (<origin>)` by references/Makefile (boost_detect.sh --where cpu, gsl_flags.sh
# --where) and the Makefile (boost_detect.sh --where gpu).
SOURCE_KEYS = ("build_cpu_supplier_cxx_boost_source", "build_cpu_supplier_gsl_source",
               "build_gpu_suppliers_boost_source")
# The GSL library file behind the gsl row, `<file> (<origin>); <build>; <fused multiply-adds>`
# (gsl_flags.sh --library); its file is shown as a `_source` directory is.
LIBRARY_KEY = "build_cpu_supplier_gsl_library"


def _source_short(d, origin):
    """A `_source` directory as HARDWARE.md shows it, with its origin: a copy under
    `references/` named from there, a directory under /usr/ or /opt/ in full, any other cut
    to its last component."""
    import re
    d = d.rstrip("/")
    ref = re.search(r"(?:^|/)(references/(?:boost|gsl)/[^/]+)$", d)
    if ref:
        d = ref.group(1)
    elif not re.match(r"/(?:usr|opt)(?:/|$)", d):
        d = d.rsplit("/", 1)[-1]
    return f"{d} ({origin})" if d else f"({origin})"


def _hardware_value(key, v):
    """A PROVENANCE value as HARDWARE.md shows it: absolute paths cut to the last component
    (a `_source` value with its origin by _source_short, and the file of LIBRARY_KEY in the
    same way, one under `references/gsl/` named from there); `exclusive`, an absent `gpu` and
    `placement` reduced to their recorded result."""
    import re
    v = " ".join(str(v).split())
    if key == "exclusive":
        return mp.dedicated({"exclusive": v}) or v
    if key == "gpu" and v.lower().startswith("none"):
        return "none"
    if key == "placement":
        v = re.sub(r"\s*\([^()]*\)", "", v)
    if key in SOURCE_KEYS:
        m = re.fullmatch(r"(.*?) ?\(([^()]*)\)", v)
        if m:
            return _source_short(m.group(1), m.group(2))
    if key == LIBRARY_KEY:
        head, sep, tail = v.partition("; ")
        m = re.fullmatch(r"(.*?) ?\(([^()]*)\)", head)
        if m:
            ref = re.search(r"(?:^|/)(references/gsl/[^/]+/.+)$", m.group(1))
            head = (f"{ref.group(1)} ({m.group(2)})" if ref
                    else _source_short(m.group(1), m.group(2)))
            return head + sep + re.sub(r"(?<![\w.])/(?:[^\s/:]+/)+", "", tail)
    return re.sub(r"(?<![\w.])/(?:[^\s/:]+/)+", "", v)


def write_hardware(runs, outdir, prov=None):
    """Write HARDWARE.md: generator versions, then per machine and run its PROVENANCE keys;
    with `prov` (style.caption_finalise, paper mode) a section per directory and figure."""
    order = []
    for r in runs:
        if machine_of(r) not in order:
            order.append(machine_of(r))
    rows = []
    for r in sorted(runs, key=lambda x: (order.index(machine_of(x)), x["id"])):
        pv = read_provenance(r["dir"])
        if not pv:
            continue
        rows.append((r, pv))

    KEYS = [
        ("cpu", "CPU"), ("cpus_online", "hardware threads"),
        ("sockets", "sockets"), ("numa_nodes", "NUMA domains"),
        ("os", "OS"), ("kernel", "kernel"),
        # `build_*` keys name what built the artefacts (detect --build-info, build/BUILD_KNOBS);
        # `compiler` and `nvcc` name the toolchain on PATH at run time.
        ("build_cpu_library_compiler", "library compiler"),
        ("build_cpu_benchmark_compiler", "CPU driver compiler"),
        ("build_cpu_supplier_c_compiler", "C supplier compiler"),
        ("build_cpu_supplier_cxx_compiler", "C++ supplier compiler"),
        ("build_cpu_supplier_cxx_boost", "Boost, CPU row (compiled against)"),
        ("build_cpu_supplier_cxx_boost_source", "Boost, CPU row (copy used)"),
        ("build_cpu_supplier_cxx_boost_eval_type", "Boost evaluation type, CPU row"),
        ("build_cpu_supplier_gsl_headers", "GSL headers"),
        ("build_cpu_supplier_gsl_runtime", "GSL library (loaded)"),
        ("build_cpu_supplier_gsl_source", "GSL (copy used)"),
        (LIBRARY_KEY, "GSL library (linked)"),
        ("build_gpu_benchmark_compiler", "nvcc (compiled with)"),
        ("build_gpu_benchmark_host_compiler", "nvcc host compiler"),
        ("build_gpu_library_host_compiler", "library device object, host compiler"),
        ("build_gpu_benchmark_cuda_headers", "CUDA headers"),
        ("build_gpu_benchmark_cuda_runtime", "CUDA runtime (loaded)"),
        ("build_gpu_suppliers_boost", "Boost, GPU row (compiled against)"),
        ("build_gpu_suppliers_boost_source", "Boost, GPU row (copy used)"),
        ("build_gpu_suppliers_boost_eval_type", "Boost evaluation type, GPU row"),
        ("make_cc", "make CC"), ("make_cxx", "make CXX"), ("make_nvcc", "make NVCC"),
        ("make_fma", "make FMA"),
        ("cc_on_path", "$CC at run time"), ("nvcc_on_path", "nvcc at run time"),
        ("scalar_isa", "scalar build selected"),
        ("ulp_mode", "ULP mode"),
        ("cpu_mhz_timed", "CPU clock in the timed rows"),
        ("timer_clock", "timer clock"), ("timer_clocksource", "clocksource"),
        ("timer_tsc_invariant", "invariant TSC"), ("timer_warning", "timer warning"),
        ("compiler", "compiler (older runs: $CC at run time)"), ("fma_knob", "FMA knob"),
        ("gpu", "GPU"), ("gpu_compute_cap", "compute capability"),
        ("gpu_memory_total", "device memory"), ("gpu_driver", "driver"),
        ("gpu_cuda_driver", "CUDA driver API"),
        ("nvcc", "nvcc (older runs: on PATH at run time)"),
        ("gpu_arch_built", "built for"),
        ("gpu_ecc", "ECC"), ("gpu_persistence", "persistence mode"),
        ("gpu_clock_sm_max", "max SM clock"),
        ("gpu_clock_mem_max", "max memory clock"),
        ("gpu_power_limit", "power limit"),
        ("gpu_power_default", "power limit (card default)"),
        ("gpu_throttle", "throttle reasons sampled"),
        ("gpu_throttle_cap", "capping reason active"),
        ("boost_default", "Boost (older runs: default include path)"),
        ("boost_root", "Boost (older runs: BOOST_ROOT)"),
        ("exclusive", "dedicated to the run"),
        ("load_before", "load at open"), ("load_after", "load at close"),
        ("placement", "placement achieved"),
    ]
    # The clause on the `_source` keys and on LIBRARY_KEY names those a run records.
    has_src = any(k in pv for _, pv in rows for k in SOURCE_KEYS)
    has_lib = any(LIBRARY_KEY in pv for _, pv in rows)
    if has_src and has_lib:
        paths = ["paths are shortened to the file name (in the `_source` keys and the first",
                 "field of `build_cpu_supplier_gsl_library` a path under `references/` is",
                 "named from there and one under `/usr/` or `/opt/` in full, followed by its",
                 "origin), and"]
    elif has_src:
        paths = ["paths are shortened to the file name (in the `_source` keys a directory",
                 "under `references/` is named from there and one under `/usr/` or `/opt/`",
                 "in full, followed by its origin), and"]
    elif has_lib:
        paths = ["paths are shortened to the file name (in the first field of",
                 "`build_cpu_supplier_gsl_library` a path under `references/` is named from",
                 "there and one under `/usr/` or `/opt/` in full, followed by its origin), and"]
    else:
        paths = ["paths are shortened to the file name, and"]
    lines = ["# Environment, generated from the runs behind the figures", "",
             "Written by `make paper-figures%s`.  The first table names the Python,"
             % ("-final" if style.PAPER else ""),
             "NumPy and Matplotlib that drew the figures.  Every table value after it is a",
             "key in that run's `PROVENANCE.txt`, in the run's directory (under",
             "`benchmark/runs/` unless `--runs` named another root); the key is",
             "named in the first column.  A field the run does not record is",
             "omitted; `--` marks a field recorded without a value.  Absolute",
             *paths,
             "`exclusive`, an absent `gpu` and `placement` are shown without the",
             "explanation recorded beside them.  Runs are named by stamp and stage",
             "and grouped by machine, which is named after its CPU model.", "",
             "## Generators", "", "| program | version |", "|---|---|"]
    lines += [f"| {name} | {ver} |" for name, ver in mp.generator_versions()]
    lines.append("")
    if not rows:
        lines += ["No run behind these figures records a `PROVENANCE.txt`.", ""]
    mach = None
    for r, pv in rows:
        if machine_of(r) != mach:
            mach = machine_of(r)
            lines += [f"## Machine `{mach}`", ""]
        lines += [f"### Run `{r['label']}`", "",
                  "| PROVENANCE key | field | value |", "|---|---|---|"]
        for k, label in KEYS:
            if k not in pv:
                continue
            v = _hardware_value(k, pv.get(k)) if pv.get(k) else "--"
            lines.append(f"| `{k}` | {label} | {v or '--'} |")
        lines.append("")
    if prov and prov.get("figures"):
        lines += ["## Figures", "",
                  "Per directory, the runs its figures were drawn from, with the",
                  "conditions each run recorded; per figure, its runs and the",
                  "details its caption in `captions.tex` does not carry (title,",
                  "notes, $N$ per dataset, repetitions, counts and checks).", ""]
        rel = lambda d: os.path.relpath(d, outdir).replace(os.sep, "/")
        dirs = dict(prov.get("dirs") or [])
        for d in sorted({f[0] for f in prov["figures"]}, key=rel):
            lines += [f"### Directory `{rel(d)}`", ""]
            for r, env in dirs.get(d, []):
                lines.append(f"- `{r}`" + (f": {env}" if env else ""))
            lines.append("")
            for _d, stem, runs_, facts in prov["figures"]:
                if _d != d:
                    continue
                lines += [f"#### `{stem}`", "",
                          "- Runs: " + ", ".join(f"`{r}`" for r in runs_)]
                lines += [f"- {t}" for t in facts]
                lines.append("")
    path = os.path.join(outdir, "HARDWARE.md")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def newest_per_stage(rs):
    """The runs of one machine, each kind of result (FILES) taken from the newest run that
    has it and from no other; a run supplying no kind is left out.  Run ids begin with a
    UTC stamp, so the newest is the lexicographic maximum."""
    owner = {}
    for k in FILES:
        have = [r for r in rs if r.get(k)]
        # A GPU-pipeline run's one-backend CPU rows are used only when no other run has the kind.
        if k in CPU_KINDS:
            full = [r for r in have if not r.get("gpu_pipeline")]
            if full:
                have = full
        if have:
            owner[k] = max(have, key=lambda r: r["id"])
    kept = []
    for r in sorted(rs, key=lambda r: r["id"]):
        mine = [k for k, o in owner.items() if o is r]
        if not mine:
            continue
        rr = dict(r)
        for k in FILES:
            if owner.get(k) is not r:
                rr[k] = []
        kept.append(rr)
    # A machine without rows keeps its newest run, so the figures report what is missing.
    if not kept and rs:
        kept = [max(rs, key=lambda r: r["id"])]
    return kept


def merge(runs, key):
    out = []
    for r in runs:
        out.extend(r[key])
    return out


def preferred_assembly(sup_rows, family="z_nu", scope="call"):
    """{supplier: assembly} chosen from the accuracy rows by the `best_assembly` rule: no
    non-finite results first, then the smallest worst-case max ULP over the laws.
    `scope="kernel"` applies it to the device rows."""
    worst = {}
    for r in sup_rows:
        if r["family"] != family or scope_of_row(r) != scope:
            continue
        k = (r["supplier"], r["backend"])
        try:
            u = float(r["u_max"])
        except (TypeError, ValueError):
            continue
        nf = 1 if int(float(r.get("nonfinite") or 0)) > 0 else 0
        prev = worst.get(k, (0, 0.0))
        worst[k] = (max(prev[0], nf), max(prev[1], u))
    out = {}
    for (sname, asm), rank in worst.items():
        if sname not in out or rank < out[sname][0]:
            out[sname] = (rank, asm)
    return {k: v[1] for k, v in out.items()}


def scope_of_row(r):
    return (r.get("timing_scope") or "call").strip()


def group_by_machine(runs):
    """({machine label: [runs]} in machine_order, each reduced by newest_per_stage, the
    labels of the superseded runs)."""
    bymach = {}
    for r in runs:
        bymach.setdefault(machine_of(r), []).append(r)
    dropped = []
    for m in bymach:
        kept = newest_per_stage(bymach[m])
        _ids = {r["id"] for r in kept}
        dropped += [r["label"] for r in bymach[m] if r["id"] not in _ids]
        bymach[m] = kept
    return {m: bymach[m] for m in machine_order(bymach)}, dropped


def families(rows):
    """The argument conventions present in `rows`, (z,nu) first."""
    return [f for f in ("z_nu", "h_nu_s") if any(r.get("family") == f for r in rows)]


def is_cpu_machine(rs):
    """True if the machine's accuracy rows come from a CPU run, not only from a
    GPU-pipeline run's one-backend reference."""
    return any(r["accuracy"] and not r.get("gpu_pipeline") for r in rs)


# The paper set: the figures and tables the paper uses, drawn by default; --full draws
# every one.  Names are stems without the fig-paper- / tab- prefix,
# as fnmatch patterns, by the machine they come from: "cpu" the machine that cpu_source
# picks for the CPU data the result is drawn from, "gpu" a machine with GPU data, "every"
# each machine, "shared" the tree root (tables) or cross-machine/ (figures).
# levin-cpu-vector-<family> is the vector router's Levin figure, whichever backend
# carries it.  Both argument conventions, (z,nu) beside the paper's (h,nu,s); the ladder
# is drawn for (z,nu) only.  fold-* would also take the fold ECDFs, so the fold figures
# are named.
PAPER_FIGURES = {
    "cpu": ("cost-accuracy-*", "accuracy-distribution-*", "assembly-*", "fold-z_nu",
            "fold-h_nu_s", "ladder-*", "workspace-*", "levin-cpu-vector-*"),
    "gpu": ("gpu-plane-*", "gpu-accuracy-distribution-*", "gpu-assembly-*",
            "gpu-modes-size_random*", "gpu-boost-shortcut", "gpu-knobs",
            "levin-cuda-*"),
    "every": ("threads-*",),
    "shared": ("machine-independence",),
}
# The tables the paper's own tables are taken from, with their (z,nu) twins.
PAPER_TABLES = {
    "cpu": ("throughput-*",),
    "gpu": ("gpu-medmax-*", "gpu-cost-*", "gpu-call-*", "gpu-exclusions-*"),
    "shared": ("medmax-*", "exclusions-*", "references", "environment", "datasets"),
}


def in_paper_set(spec, name, roles=None):
    """True if `name` (a stem without its prefix) is in the paper set `spec` for a machine
    with `roles`; roles=None asks whether any role has it.  A name may itself be a pattern
    ("throughput-*"), which matches a listed stem it covers."""
    from fnmatch import fnmatchcase
    pats = [p for role, ps in spec.items() if roles is None or role in roles for p in ps]
    return any(fnmatchcase(name, p) or fnmatchcase(p, name) for p in pats)


# The device kinds of a run: a machine with any of them is a GPU host.
GPU_KINDS = ("gpu", "gpu_sup", "gpu_ladder", "gpu_klad", "gpu_levin", "gpu_threads",
             "gpu_boost")


def has_gpu(rs):
    return any(r.get("gpu_pipeline") or any(r.get(k) for k in GPU_KINDS) for r in rs)


def cpu_order(bymach, name=""):
    """The machines in the order the paper set takes CPU results from: `name` first if
    given, then those without GPU data (the paper's CPU host is a CPU node; the GPU host
    runs the GPU stages), then the rest, each group in machine order."""
    if name and name not in bymach:
        raise SystemExit(f"--cpu-machine {name}: no such machine (have: "
                         f"{', '.join(bymach) or 'none'})")
    return sorted(bymach, key=lambda m: (m != name, has_gpu(bymach[m])))


def cpu_source(bymach, order, kinds):
    """The machine a CPU result drawn from the run kinds `kinds` comes from: the first in
    `order` whose CPU runs carry one of them, else the first whose GPU-pipeline run does
    (its one-backend CPU reference); None if no run does.  So any one machine, or a stage
    run on one machine only, still yields its results."""
    for reference in (False, True):
        for m in order:
            if any(r.get(k) and (reference or not r.get("gpu_pipeline"))
                   for r in bymach[m] for k in kinds):
                return m
    return None


def machine_roles(mach, bymach, order, kinds=()):
    """The paper-set roles of machine `mach` (None: the tree root, "shared") for a result
    drawn from the CPU run kinds `kinds`."""
    if mach is None:
        return {"shared"}
    roles = {"every"}
    if has_gpu(bymach[mach]):
        roles.add("gpu")
    if kinds and cpu_source(bymach, order, kinds) == mach:
        roles.add("cpu")
    return roles


def cpu_sources_line(bymach, order):
    """Where the paper set's CPU results come from, for the console: `machine (kinds)`."""
    by = {}
    for k in ("accuracy", "ladder", "levin", "fold"):
        m = cpu_source(bymach, order, (k,))
        if m:
            by.setdefault(m, []).append(k)
    return ", ".join(f"{m} ({', '.join(ks)})" for m, ks in by.items()) or "no CPU run"


def register_conditions(bymach):
    """Hand each run's PROVENANCE and machine label to style, for the stamps."""
    for m, rs in bymach.items():
        for r in rs:
            style.register_run(r["label"], r.get("pv") or read_provenance(r["dir"]), m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the runs and row counts, draw nothing")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to draw from a run whose STATUS is not complete")
    ap.add_argument("--paper", action="store_true",
                    help="publication output: no on-canvas titles, notes or run "
                         "stamp; the captions are written to captions.tex and "
                         "the other details to HARDWARE.md")
    ap.add_argument("--runs", default="",
                    help="read runs from this directory instead of "
                         "benchmark/runs")
    ap.add_argument("--full", action="store_true",
                    help="draw every figure, not only the paper set (PAPER_FIGURES), "
                         "and PNG beside PDF in paper mode")
    ap.add_argument("--cpu-machine", default="",
                    help="the machine the paper set takes its CPU figures from first "
                         "(default: machines without GPU data first; a stage this "
                         "machine lacks comes from the next machine that has it)")
    args = ap.parse_args()

    global OUT, RUNS
    if args.runs:
        RUNS = os.path.abspath(args.runs)
    # style.PAPER is read at draw time, so it is set before any figure is drawn.
    if args.paper:
        style.PAPER = True
        OUT = os.path.join(BENCH, "plots", "paper-final")
        # the manuscript includes the PDF; the full set keeps a PNG beside it
        if not args.full:
            style.FORMATS = ("pdf",)
    # captions.tex is appended to, so every existing one in the tree is removed first.
    if style.PAPER and os.path.isdir(OUT):
        for _root, _d, _f in os.walk(OUT):
            if "captions.tex" in _f:
                os.remove(os.path.join(_root, "captions.tex"))

    runs = load_all()
    if not runs:
        print("no runs in", RUNS); return 1
    if args.list:
        for r in runs:
            print(f"{r['id']:44s} {r['status']:9s} {r['machine']:20s} "
                  f"{r['arch_pretty']}")
            for k in ("accuracy", "suppliers", "ladder", "sup_ladder",
                      "gpu", "gpu_sup", "gpu_ladder", "gpu_klad", "fold"):
                if r[k]:
                    print(f"    {k:11s} {len(r[k]):5d} rows")
        return 0

    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style.apply(plt)

    bymach, dropped = group_by_machine(runs)
    if dropped:
        print(f"   superseded, not merged: {', '.join(dropped)}")
    register_conditions(bymach)

    # Only the runs drawn from are checked; a run without a STATUS file is incomplete.
    bad = [r["label"] for rs_ in bymach.values() for r in rs_ if r["status"] != "complete"]
    if bad:
        msg = "runs not marked complete: " + ", ".join(bad)
        if args.strict:
            print("refusing under --strict:", msg); return 2
        print("!!", msg, "-- figures will be stamped")

    n = 0
    written = []

    def note(count, stem):
        nonlocal n
        n += count
        if count:
            written.append(stem)

    order = cpu_order(bymach, args.cpu_machine)
    if args.full:
        print("figure set: every figure (--full)")
    else:
        print(f"figure set: the paper set (--full for every figure); CPU figures from "
              f"{cpu_sources_line(bymach, order)}")
    cur = None                       # the machine being drawn; None for cross-machine/

    def want(name, *kinds):
        """Whether figure `name` (its stem without fig-paper-) is drawn for the current
        machine; `kinds`, the CPU run kinds it is drawn from, place a CPU figure."""
        return args.full or in_paper_set(
            PAPER_FIGURES, name, machine_roles(cur, bymach, order, kinds))

    ACC, LAD = ("accuracy", "suppliers"), ("ladder", "sup_ladder")
    for mach, rs in bymach.items():
        cur = mach
        arch = rs[0]["arch_pretty"].split("[")[0].split(" + ")[0].strip()

        def stamp_of(*kinds):
            return run_stamp(source_runs(rs, *kinds))

        own = kind_owner(rs)
        acc, sup = merge(rs, "accuracy"), merge(rs, "suppliers")
        lad, slad = merge(rs, "ladder"), merge(rs, "sup_ladder")
        gpu, gsup = merge(rs, "gpu"), merge(rs, "gpu_sup")
        glad = merge(rs, "gpu_ladder")
        lev, glev = merge(rs, "levin"), merge(rs, "gpu_levin")
        thr, gthr = merge(rs, "threads"), merge(rs, "gpu_threads")
        fold = merge(rs, "fold")
        gbsc = merge(rs, "gpu_boost")
        sub = os.path.join(OUT, mach)
        os.makedirs(sub, exist_ok=True)
        pf.SKIP_PREFIX = f"{mach}/"
        print(f"== {mach}  {machine_pretty(rs)}")

        if acc or sup:
            stamp = stamp_of("accuracy", "suppliers")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"cost-accuracy-{fam}", *ACC):
                    note(pf.fig_cost_accuracy_plane(plt, acc, sup, sub, stamp,
                                                    family=fam, arch=arch),
                         f"{mach}/cost-accuracy-{fam}")
        if acc or sup:
            stamp = stamp_of("accuracy", "suppliers")
            raw = merge(rs, "raw")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"accuracy-distribution-{fam}", *ACC):
                    note(pf.fig_accuracy_distribution(plt, acc, sup, sub, stamp,
                                                      family=fam, arch=arch),
                         f"{mach}/accuracy-distribution-{fam}")
                if want(f"accuracy-ecdf-{fam}", *ACC):
                    note(pf.fig_accuracy_ecdf(plt, acc, sup, sub, stamp,
                                              family=fam, arch=arch),
                         f"{mach}/accuracy-ecdf-{fam}")
                if want(f"timing-distribution-{fam}", *ACC):
                    note(pf.fig_timing_distribution(plt, acc, sup, raw, sub, stamp,
                                                    family=fam, arch=arch),
                         f"{mach}/timing-distribution-{fam}")
        if acc:
            stamp = stamp_of("accuracy")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"attribution-{fam}", *ACC):
                    note(pf.fig_attribution(plt, acc, sub, stamp, family=fam,
                                            arch=arch),
                         f"{mach}/attribution-{fam}")
        if acc and sup:
            stamp = stamp_of("accuracy", "suppliers")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"worst-case-map-{fam}", *ACC):
                    note(pf.fig_worst_case_map(plt, acc, sup, sub, stamp,
                                               family=fam, arch=arch),
                         f"{mach}/worst-case-map-{fam}")
                if want(f"assembly-{fam}", *ACC):
                    note(pf.fig_assembly(plt, sup, acc, sub, stamp, family=fam,
                                         arch=arch),
                         f"{mach}/assembly-{fam}")
        if acc and want("convention", *ACC):
            # Only a GPU-pipeline run's single-backend rows require several backends.
            note(pf.fig_family_delta(plt, acc, sub, stamp_of("accuracy"), arch=arch,
                                     require_multi_backend=any(
                                         x.get("gpu_pipeline")
                                         for x in source_runs(rs, "accuracy"))),
                 f"{mach}/convention")
        if lad or slad:
            # Ladder files carry no accuracy; assemblies come from the supplier accuracy rows.
            asm = preferred_assembly(sup)
            stamp = stamp_of("ladder", "sup_ladder",
                             *(("suppliers",) if slad and sup else ()))
            for law in pf.ladder_laws_present(lad) or pf.ladder_laws_present(slad):
                if want(f"ladder-{law}", *LAD):
                    note(pf.fig_ladder(plt, lad, slad, sub, stamp, law=law,
                                       arch=arch, asm_of=asm),
                         f"{mach}/ladder-{law}")
        if lad:
            stamp = stamp_of("ladder")
            for law in pf.ladder_laws_present(lad):
                for fam in ("z_nu", "h_nu_s"):
                    if want(f"workspace-{law}-{fam}", "ladder"):
                        note(pf.fig_workspace(plt, lad, sub, stamp, law=law,
                                              family=fam, arch=arch),
                             f"{mach}/workspace-{law}-{fam}")
        # Device figures name the CPU and GPU of their own run; the scraped device name is
        # added only when PROVENANCE lacks it.
        _gruns = [own[k] for k in ("gpu_ladder", "gpu_sup", "gpu", "gpu_klad",
                                   "gpu_levin", "gpu_threads", "gpu_boost")
                  if k in own]
        garch0 = (_gruns[0]["arch_pretty"].split("[")[0].strip() if _gruns
                  else arch)
        dev = device_label(_gruns or rs)
        garch = garch0 if (not dev or dev in garch0) else f"{dev} on {garch0}"
        raw = merge(rs, "raw") if (glad or gsup) else []
        if glad:
            stamp = stamp_of("gpu_ladder")
            # The CPU curve is the GPU-pipeline run's own CPU ladder: one run, one machine.
            cpu_ref = own["gpu_ladder"].get("gpu_cpu_ladder") or []
            for law in pf.ladder_laws_present(glad):
                for fam in ("z_nu", "h_nu_s"):
                    if want(f"gpu-modes-{law}-{fam}"):
                        note(pf.fig_gpu_modes(plt, glad, cpu_ref, sub, stamp, law=law,
                                              raw=raw, family=fam, arch=garch),
                             f"{mach}/gpu-modes-{law}-{fam}")
        if gsup:
            stamp = stamp_of("gpu_sup")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"gpu-modes-suppliers-{fam}"):
                    note(pf.fig_gpu_modes_suppliers(plt, gsup, sub, stamp,
                                                    family=fam, arch=garch),
                         f"{mach}/gpu-modes-suppliers-{fam}")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"gpu-plane-{fam}"):
                    note(pf.fig_gpu_plane(plt, gsup, sub, stamp, family=fam,
                                          arch=garch), f"{mach}/gpu-plane-{fam}")
                if want(f"gpu-accuracy-distribution-{fam}"):
                    note(pf.fig_gpu_accuracy_distribution(plt, gsup, sub, stamp,
                                                          family=fam, arch=garch),
                         f"{mach}/gpu-accuracy-distribution-{fam}")
                if want(f"gpu-accuracy-ecdf-{fam}"):
                    note(pf.fig_gpu_accuracy_ecdf(plt, gsup, sub, stamp,
                                                  family=fam, arch=garch),
                         f"{mach}/gpu-accuracy-ecdf-{fam}")
                if want(f"gpu-timing-distribution-{fam}"):
                    note(pf.fig_gpu_timing_distribution(plt, gsup, raw, sub, stamp,
                                                        family=fam, arch=garch),
                         f"{mach}/gpu-timing-distribution-{fam}")
                if want(f"gpu-assembly-{fam}"):
                    note(pf.fig_gpu_assembly(plt, gsup, sub, stamp, family=fam,
                                             arch=garch),
                         f"{mach}/gpu-assembly-{fam}")

        if gbsc and want("gpu-boost-shortcut"):
            note(pf.fig_gpu_boost_shortcut(plt, gbsc, sub, stamp_of("gpu_boost"),
                                           arch=garch),
                 f"{mach}/gpu-boost-shortcut")

        # Levin: per device and per backend; the scalar and vector routers gate the band
        # behind different macros.
        if lev:
            # The vector router by the best vector backend present (pf.BACKEND_PREFERENCE).
            _vec = pf.lib_backend_for([r for r in lev if (r.get("device") or "cpu").strip() == "cpu"
                                       and r.get("backend") != "scalar"]) or "avx512"
            for bk in (_vec, "scalar"):
                for fam in families(lev):
                    if want(f"levin-cpu-vector-{fam}" if bk == _vec
                            else f"levin-cpu-{bk}-{fam}", "levin"):
                        note(pf.fig_levin(plt, lev, sub, stamp_of("levin"),
                                          device="cpu", backend=bk, arch=arch,
                                          family=fam),
                             f"{mach}/levin-cpu-{bk}-{fam}")
        if glev:
            # On the device `backend` is the transfer mode; the band does not depend on it, so
            # one mode is shown, the default async-direct when present.
            gbk = "async-direct"
            have = {r.get("backend") for r in glev}
            if gbk not in have and have:
                gbk = sorted(have)[0]
            for fam in families(glev):
                if want(f"levin-cuda-{fam}"):
                    note(pf.fig_levin(plt, glev, sub, stamp_of("gpu_levin"),
                                      device="cuda", backend=gbk, arch=garch,
                                      family=fam),
                         f"{mach}/levin-cuda-{fam}")
        if fold:
            for fam in families(fold):
                if want(f"fold-{fam}", "fold"):
                    note(pf.fig_fold(plt, fold, sub, stamp_of("fold"), arch=arch,
                                     family=fam),
                         f"{mach}/fold-{fam}")
            for fam in ("z_nu", "h_nu_s"):
                if want(f"fold-ecdf-{fam}", "fold"):
                    note(pf.fig_fold_ecdf(plt, fold, sub, stamp_of("fold"),
                                          family=fam, arch=arch),
                         f"{mach}/fold-ecdf-{fam}")
        for fam in families(thr):
            if want(f"threads-{fam}", "threads"):
                note(pf.fig_threads_combined(plt, thr, sub, stamp_of("threads"),
                                             arch=arch, family=fam),
                     f"{mach}/threads-{fam}")
        if gthr and want("gpu-knobs"):
            note(pf.fig_gpu_knobs(plt, gthr, sub, stamp_of("gpu_threads"),
                                  arch=garch),
                 f"{mach}/gpu-knobs")

    pf.SKIP_PREFIX = ""
    cur = None

    def _disp(mach, rs):
        short = mp.cpu_arch_key(rs[0].get("arch_short")) or mach
        return short if style.PAPER else f"{short} ({mach})"

    _cpu_m = [(m, rs) for m, rs in bymach.items() if is_cpu_machine(rs)]
    cpu_acc = [(_disp(m, rs), merge(rs, "accuracy")) for m, rs in _cpu_m]
    # From the short label: the display label carries the machine in development mode.
    _keys = [mp.cpu_arch_key(rs[0].get("arch_short") or m) for m, rs in _cpu_m]
    _arch = sorted(set(_keys))
    if len(_arch) < 2:
        pf.SKIPPED.append((
            "cross-machine/machine-independence",
            f"needs 2 CPU architectures, found {len(_arch)}"
            f"{' (' + ', '.join(_arch) + ')' if _arch else ''}"
            f" across {len(cpu_acc)} machine(s); runs with the same CPU "
            "model are one machine and are not compared"))
    if len(_arch) >= 2 and want("machine-independence"):
        # Keep one machine per architecture.
        _seen, _pair, _pair_rs = set(), [], []
        for (m, rs), k, (_m, mrs) in zip(cpu_acc, _keys, _cpu_m):
            if k not in _seen:
                _seen.add(k)
                _pair.append((m, rs))
                _pair_rs.append(mrs)
        cpu_acc = _pair
        xdir = os.path.join(OUT, "cross-machine")
        os.makedirs(xdir, exist_ok=True)
        ids = " vs ".join(run_stamp(source_runs(mrs, "accuracy"))
                          for mrs in _pair_rs[:2])
        note(pf.fig_machine_independence(plt, cpu_acc[:2], xdir, ids),
             f"cross-machine/machine-independence")
    # Image files not produced by this run are listed, not deleted.
    produced = set()
    for root, _dirs, files in os.walk(OUT):
        for fn in files:
            if fn.endswith((".png", ".pdf")):
                produced.add(os.path.relpath(os.path.join(root, fn), OUT))
    fresh = {os.path.relpath(p_, OUT).replace(os.sep, "/")
             for p_ in style.SAVED}
    stale = sorted(p_ for p_ in produced
                   if p_.replace(os.sep, "/") not in fresh)
    if stale:
        print(f"\n!! {len(stale)} file(s) in {OUT} were NOT produced by this run.")
        print("   They are from an earlier naming or an earlier run and are stale:")
        for p_ in stale[:20]:
            print(f"     {p_}")
        if len(stale) > 20:
            print(f"     ... and {len(stale) - 20} more")
        print("   Remove them, or regenerate into a clean directory.")

    prov = None
    if style.PAPER:
        prov = style.caption_finalise()
        _n = _files = 0
        for _root, _d, _f in os.walk(OUT):
            if "captions.tex" in _f:
                _files += 1
                # Each figure environment starts with a "% ----" marker line.
                _n += sum(1 for l in open(os.path.join(_root, "captions.tex"))
                          if l.startswith("% ----"))
        if _files:
            print(f"\ncaptions -> {_files} captions.tex beside the figures, "
                  f"{_n} figure environments")

    hw = write_hardware([r for rs in bymach.values() for r in rs], OUT, prov)
    if hw:
        print(f"\nenvironment -> {hw}")
    print(f"\n{n} figures -> {OUT}")
    for w in written:
        print("   ", w)

    if style.SIZES:
        print(f"\nprinted size at \\linewidth = {style.LATEX_LINEWIDTH:g} in "
              f"(limits: height {style.TEXT_H_MAX:g} in, text {style.FS_MIN:g} pt):")
        for line in style.size_report(OUT):
            print(line)

    if pf.SKIPPED:
        print(f"\n{len(pf.SKIPPED)} figure(s) NOT drawn, with the input that was "
              f"missing:")
        _seen = set()
        for stem, why in pf.SKIPPED:
            if (stem, why) in _seen:
                continue
            _seen.add((stem, why))
            print(f"    {stem}")
            print(f"        {why}")
    if pf.SERIES_SKIPPED:
        print(f"\n{len(pf.SERIES_SKIPPED)} series NOT drawn in a figure that was "
              f"drawn:")
        for stem, name, law, why in pf.SERIES_SKIPPED:
            print(f"    {stem}: {name} on {law}")
            print(f"        {why}")
    if UNREADABLE:
        print("\nINPUTS REFUSED (figures needing them were skipped):")
        for u in UNREADABLE:
            print("   ", u)
        return 2 if args.strict else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
