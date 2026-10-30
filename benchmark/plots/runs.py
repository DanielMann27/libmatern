"""runs.py -- run-directory readers shared by the figure and table generators: STATUS,
schema-checked CSV rows, machine names from PROVENANCE.txt and the accuracy-detail
histograms."""
import csv, hashlib, os, platform, re, subprocess

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


# The tree the generators run from, two levels above benchmark/plots.
_TREE = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _git(*args):
    """stdout of `git -C <tree> <args>`, or None when git fails or is absent."""
    try:
        r = subprocess.run(["git", "-C", _TREE] + list(args), capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def generator_commit():
    """(release, commit, dirty) of the tree the generators run from, found as
    run_benchmark.sh finds a run's: the commit from git when the tree is the top level of a
    git checkout (dirty `yes` or `no`, untracked files not counted), else from the RELEASE
    file that `git archive` fills in (dirty `unknown`)."""
    release = "not recorded"
    try:
        for line in open(os.path.join(_TREE, "include", "matern.h")):
            m = re.match(r'#define MATERN_VERSION_STRING\s+"([^"]*)"', line)
            if m:
                release = m.group(1)
                break
    except OSError:
        pass
    top = _git("rev-parse", "--show-toplevel")
    if top and os.path.realpath(top) == _TREE:
        head = _git("rev-parse", "HEAD") or "not recorded"
        st = _git("--no-optional-locks", "status", "--porcelain", "--untracked-files=no")
        return release, head, ("unknown" if st is None else "yes" if st else "no")
    try:
        for line in open(os.path.join(_TREE, "RELEASE")):
            f = line.split()
            if len(f) == 2 and f[0] == "commit" and re.fullmatch(r"[0-9a-f]{40}", f[1]):
                return release, f[1], "unknown"
    except OSError:
        pass
    return release, "not recorded", "unknown"


def run_commit(pv):
    """(release, commit, dirty) as a run's PROVENANCE.txt records them (`release`, the first
    word of `source_commit` and of `git_dirty`)."""
    pv = pv or {}
    sc = (pv.get("source_commit") or "").split()
    gd = (pv.get("git_dirty") or "").split()
    return (pv.get("release") or "not recorded", sc[0] if sc else "not recorded",
            gd[0] if gd else "not recorded")


# The figure and table generators: the files that draw from the run records.
GENERATOR_DIRS = ("benchmark/plots/", "benchmark/tables/")
# The test suites: the benchmark's (run_tests.sh) and the library's (make check).
TEST_DIRS = ("benchmark/tests/", "test/")
# MANIFEST.txt paths a run made or downloaded rather than read from the tree: the build
# outputs and the KAUST files (run_benchmark.sh, the MANIFEST.txt block).
MANIFEST_MADE = ("build/", "references/build/", "references/kaust_logbesselk/upstream/")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_sources(run_dir):
    """{path: sha256} of the tree files a run's MANIFEST.txt records.  Absolute paths and the
    corpus files under data/<family>/ are left out: the run checked the corpus against
    data/MANIFEST.md, whose own digest is kept."""
    out = {}
    try:
        lines = open(os.path.join(run_dir, "MANIFEST.txt"), errors="replace").read().splitlines()
    except OSError:
        return out
    for line in lines:
        f = line.split(None, 1)
        if len(f) != 2 or not re.fullmatch(r"[0-9a-f]{64}", f[0]):
            continue
        path = f[1].strip()
        if path.startswith("/") or (path.startswith("data/") and path.count("/") > 1):
            continue
        out[path] = f[0]
    return out


def _kinds(paths):
    """`paths` in words, grouped as the generators, the tests (benchmark/tests/, test/), the
    documentation (*.md, docs/) and the rest by name."""
    gen = [p for p in paths if p.startswith(GENERATOR_DIRS)]
    tst = [p for p in paths if p not in gen and p.startswith(TEST_DIRS)]
    doc = [p for p in paths if p not in gen and p not in tst
           and (p.endswith(".md") or p.startswith("docs/"))]
    rest = [p for p in paths if p not in gen and p not in tst and p not in doc]
    parts = []
    for group, name in ((gen, "the figure and table generators"), (tst, "the tests"),
                        (doc, "the documentation")):
        if group:
            parts.append(f"{name} ({len(group)} file{'s' * (len(group) > 1)})")
    if len(rest) > 8:
        tops = {}
        for p in rest:
            top = p.split("/")[0] + "/" if "/" in p else p
            tops[top] = tops.get(top, 0) + 1
        parts.append(", ".join(f"`{t}` ({n})" for t, n in sorted(tops.items())))
    else:
        parts += [f"`{p}`" for p in rest]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def source_difference(commit, run_dirs):
    """(detail, summary) for the runs at `commit` against the tree the generators run from.

    In a git checkout that holds `commit`, `git diff --name-only <commit> HEAD` and the
    uncommitted changes name every differing file.  Otherwise (an archive, or a commit the
    checkout lacks) the sources the runs' MANIFEST.txt records are hashed in this tree; that
    list holds the library, the benchmark drivers, the supplier sources and the Makefile, not
    the generators, the shell runners or the documentation."""
    top = _git("rev-parse", "--show-toplevel")
    if (top and os.path.realpath(top) == _TREE and re.fullmatch(r"[0-9a-f]{7,40}", commit)
            and _git("cat-file", "-e", commit + "^{commit}") is not None):
        changed = set((_git("diff", "--name-only", commit, "HEAD") or "").splitlines())
        # tracked files changed against HEAD, staged or not (untracked files do not count, as in
        # generator_commit)
        changed |= set((_git("diff", "--name-only", "HEAD") or "").splitlines())
        paths = sorted(changed)
        how = f"`git diff --name-only {commit[:12]}` against this checkout, uncommitted changes included"
        if not paths:
            return f"no file differs ({how})", "no file differs"
        words = _kinds(paths)
        detail = (f"{len(paths)} file{'s' * (len(paths) > 1)} differ{'s' * (len(paths) == 1)} "
                  f"({how}): {words}")
        if len(paths) <= 12:
            detail += "; " + ", ".join(f"`{p}`" for p in paths)
        return detail, "the files that differ are " + words
    recorded = {}
    for d in run_dirs:
        for path, digest in manifest_sources(d).items():
            recorded.setdefault(path, set()).add(digest)
    if not recorded:
        return ("no run at this commit records a MANIFEST.txt with sources, so nothing is "
                "compared", "nothing is compared (no MANIFEST.txt)")
    src, made = {"same": [], "differ": [], "absent": []}, {"same": [], "differ": [], "absent": []}
    for path, digests in sorted(recorded.items()):
        group = made if path.startswith(MANIFEST_MADE) else src
        full = os.path.join(_TREE, path)
        if not os.path.isfile(full):
            group["absent"].append(path)
        elif _sha256(full) in digests and len(digests) == 1:
            group["same"].append(path)
        else:
            group["differ"].append(path)
    n = sum(len(v) for v in src.values())
    detail = (f"of the {n} sources the runs' MANIFEST.txt records, {len(src['same'])} are "
              f"identical in this tree")
    if src["differ"]:
        detail += f", {len(src['differ'])} differ: " + _kinds(src["differ"])
    if src["absent"]:
        detail += f", {len(src['absent'])} are not in it: " + _kinds(src["absent"])
    m = sum(len(v) for v in made.values())
    if m:
        x = len(made["absent"])
        detail += (f"; of the {m} file{'s' * (m > 1)} the runs built or downloaded, "
                   f"{len(made['same'])} identical here, {len(made['differ'])} differing and "
                   f"{x} {'is' if x == 1 else 'are'} not here")
    detail += ("; the generators, the shell runners and the documentation are not recorded "
               "in MANIFEST.txt")
    if src["differ"] or src["absent"]:
        k, a = len(src["differ"]), len(src["absent"])
        summary = (f"{k} of the {n} recorded sources differ{'s' * (k == 1)} and "
                   f"{a} {'is' if a == 1 else 'are'} missing (MANIFEST.txt)")
    else:
        summary = f"all {n} sources the runs record are identical (MANIFEST.txt)"
    return detail, summary


def version_of_record(runs):
    """(markdown lines, verdict): the release and commit of the generators and of each run
    ([(label, PROVENANCE dict)] or [(label, PROVENANCE dict, run directory)]), and whether all
    are one version of record -- one commit, with no uncommitted changes recorded.  When they
    are not, each run commit other than the generators' is compared with the generators'
    tree (source_difference): the runs' directories, when given, supply their MANIFEST.txt."""
    gr, gc, gd = generator_commit()
    lines = ["## Version of record", "",
             "The release and commit of the generators that wrote this file and of each run",
             "behind it (`release`, `source_commit` and `git_dirty` in the run's",
             "`PROVENANCE.txt`).", "",
             "| | release | commit | uncommitted changes |", "|---|---|---|---|",
             f"| generators | {gr} | `{gc}` | {gd} |"]
    commits, dirty, at = {gc}, [gd], {}
    for run in runs:
        label, pv = run[0], run[1]
        r, c, d = run_commit(pv)
        lines.append(f"| run `{label}` | {r} | `{c}` | {d} |")
        commits.add(c)
        dirty.append(d)
        at.setdefault(c, []).append((label, run[2] if len(run) > 2 else None))
    if len(commits) == 1 and gc != "not recorded":
        verdict = (f"the generators and every run are at commit {gc[:12]}"
                   + ("" if all(d == "no" for d in dirty) else
                      "; uncommitted changes are not ruled out for every one of them"))
        return lines + ["", verdict[0].upper() + verdict[1:] + ".", ""], verdict
    verdict = ("the generators and the runs are NOT at one commit: "
               + ", ".join(sorted(c[:12] for c in commits)))
    diff_lines, summaries = [], []
    for c in sorted(at):
        if c == gc or c == "not recorded":
            continue
        detail, summary = source_difference(c, [d for _, d in at[c] if d])
        names = ", ".join(f"`{label}`" for label, _ in at[c])
        diff_lines.append(f"- runs at `{c[:12]}` ({names}): {detail}.")
        summaries.append(f"against the runs at {c[:12]}, {summary}")
    if diff_lines:
        lines += ["", "What the generators' tree changes against each run commit:", ""] + diff_lines
        verdict += "; " + "; ".join(summaries)
    return lines + ["", verdict[0].upper() + verdict[1:] + ".", ""], verdict


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

