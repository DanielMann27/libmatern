#!/usr/bin/env python3
"""durations.py [--runs DIR] [--check] -- the durations of the published run.

benchmark/durations.tsv holds them, one line per part of the full run and machine; this
script is its only writer and the only renderer of the places that state them: the
`text_durations` function of benchmark/tools/full_run.sh (the full-run notice), the
durations table of docs/benchmark.md and the hour count in README.md, each between markers.

  --runs DIR   read the run records under DIR (`make durations`, default benchmark/runs),
               write benchmark/durations.tsv from them, then render the three places
  (none)       render the three places from benchmark/durations.tsv
  --check      exit 1 when a place differs from what benchmark/durations.tsv renders
               (make audit; needs no run record)

A run records the wall time of each stage in TIMES.txt (benchmark/run_benchmark.sh,
benchmark/run_gpu.sh): `<stage> <seconds since the previous entry>`, and `run_total
<seconds>` at the end of each runner (a GPU run has two; the last covers it all)."""
import argparse, os, re, sys, textwrap

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmark", "plots"))

TSV = os.path.join(ROOT, "benchmark", "durations.tsv")
FULL_RUN = os.path.join(ROOT, "benchmark", "tools", "full_run.sh")
DOCS = os.path.join(ROOT, "docs", "benchmark.md")
README = os.path.join(ROOT, "README.md")
COLUMNS = ["part", "cpu", "device", "seconds", "run", "source"]

# The parts, in the order they are stated, with their labels.
PARTS = [("ladders", "ladders (up to 2^25 points)"),
         ("rest", "accuracy and suppliers"),
         ("kaust", "  of it the KAUST rows"),
         ("levin", "Levin A/B"),
         ("threads", "thread scaling"),
         ("fold", "prefactor fold"),
         ("gpu", "GPU (core, Levin, threads)")]
SOURCES = {
    "record": "as each run records them (TIMES.txt: the wall time of its runner, build "
              "excluded)",
    "slurm-wall": "as the wall time of each batch job, build included (these runs record no "
                  "times)",
    "log": "are from the time stamps of the run's log",
}


# ---------------------------------------------------------------- the run records

def read_times(run_dir):
    """[(stage, seconds)] from TIMES.txt; [] when the run records none."""
    p = os.path.join(run_dir, "TIMES.txt")
    out = []
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            if len(f) == 2 and not line.startswith("#") and f[1].isdigit():
                out.append((f[0], int(f[1])))
    return out


def cpu_name(cpu):
    """`AMD EPYC 9454 48-Core Processor` -> `AMD EPYC 9454`; `Intel(R) Xeon(R) Platinum
    8358 CPU @ 2.60GHz` -> `Intel Xeon Platinum 8358`."""
    s = re.sub(r"\((R|TM|r|tm)\)", " ", cpu or "").split("@")[0]
    s = re.sub(r"\b\d+-Core\b|\bProcessor\b|\bCPU\b", " ", s, flags=re.I)
    return " ".join(s.split())


def provenance(run_dir):
    pv = {}
    p = os.path.join(run_dir, "PROVENANCE.txt")
    if os.path.exists(p):
        for line in open(p):
            f = line.split(None, 1)
            if len(f) == 2:
                pv.setdefault(f[0], f[1].strip())
    return pv


def parts_of(times):
    """{part: seconds} of one run, from the stage names it recorded (the run id's label
    depends on SCOPE and the settings, so it is not read).  A GPU run (run_gpu.sh's
    stages, named gpu*) counts whole: its CPU reference stages belong to it.  Otherwise
    each part is the sum of its stages, and a run that holds a single part counts whole, so
    that its record, manifest and detect steps are included."""
    total = [s for n, s in times if n == "run_total"]
    total = total[-1] if total else None
    names = [n for n, _ in times]
    if any(n.startswith("gpu") for n in names):
        return {"gpu": total} if total is not None else {}
    def stages(*prefixes):
        return sum(s for n, s in times if n != "run_total"
                   and any(n == p or n.startswith(p) for p in prefixes))
    out = {}
    for part, prefixes in (("ladders", ("ladder", "supplier-ladder:")),
                           ("rest", ("accuracy", "suppliers:")),
                           ("levin", ("levin",)),
                           ("threads", ("threads",)),
                           ("fold", ("fold",))):
        if any(n == p or n.startswith(p) for n in names for p in prefixes):
            out[part] = stages(*prefixes)
    if len(out) == 1 and total is not None:
        out = {k: total for k in out}
    kaust = stages("suppliers:kaust")
    if "rest" in out and kaust:
        out["kaust"] = kaust
    return out


def from_runs(runs_dir):
    """The rows of durations.tsv from the newest run of each part and machine."""
    best = {}
    for run_id in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, run_id)
        if run_id == "latest" or os.path.islink(d) or not os.path.isdir(d):
            continue
        times = read_times(d)
        if not times:
            continue
        pv = provenance(d)
        cpu = cpu_name(pv.get("cpu", "")) or "CPU not recorded"
        gpu = pv.get("gpu", "")
        dev = "" if not gpu or gpu.lower().startswith("none") else gpu.split(",")[0].strip()
        for part, sec in parts_of(times).items():
            # run ids begin with a UTC stamp: the newest is the last in sorted order
            best[(part, cpu)] = {"part": part, "cpu": cpu,
                                 "device": dev if part == "gpu" else "",
                                 "seconds": str(sec), "run": run_id, "source": "record"}
    order = {p: i for i, (p, _) in enumerate(PARTS)}
    return sorted(best.values(), key=lambda r: (order[r["part"]], r["cpu"]))


# ---------------------------------------------------------------- the TSV

HEADER = ("# benchmark/durations.tsv -- the durations of the published run, one line per part\n"
          "# and machine.  Written by `make durations` (benchmark/tools/durations.py) from the\n"
          "# run records; do not edit by hand.  `source`: record = TIMES.txt of the run (wall\n"
          "# time of the runner, build excluded); slurm-wall = wall time of the batch job, build\n"
          "# included; log = time stamps of the run's log.\n")


def write_tsv(rows):
    with open(TSV, "w") as f:
        f.write(HEADER + "\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in COLUMNS) + "\n")


def read_tsv():
    rows = []
    for line in open(TSV):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if f == COLUMNS:
            continue
        rows.append(dict(zip(COLUMNS, f)))
    return rows


# ---------------------------------------------------------------- rendering

def fmt(seconds):
    s = int(seconds)
    if s < 60:
        return "%d s" % s
    return "%.1f h" % (s / 3600.0) if s >= 3600 else "%d min" % round(s / 60.0)


def per_part(rows):
    """[(label, "12.1 h (AMD EPYC 9454), 14.3 h (Intel ...)")] in the order of PARTS."""
    out = []
    for part, label in PARTS:
        rs = [r for r in rows if r["part"] == part]
        if rs:
            m = main_source(rows)
            out.append((label, ", ".join("%s (%s%s)" % (fmt(r["seconds"]), short(r),
                                                        ", " + r["source"] if r["source"] != m
                                                        else "") for r in rs)))
    return out


def hours_one_machine(rows):
    """The whole run on the machine with the GPU, in hours: every part on its CPU but the
    KAUST rows, which the accuracy part contains; None without a GPU part."""
    g = [r for r in rows if r["part"] == "gpu"]
    if not g:
        return None
    host = g[0]["cpu"]
    return sum(int(r["seconds"]) for r in rows
               if r["cpu"] == host and r["part"] != "kaust") / 3600.0


def main_source(rows):
    """The source of most rows; the others are starred (per_part) and named (source_sentence)."""
    srcs = [r["source"] for r in rows]
    return max(sorted(set(srcs)), key=srcs.count) if srcs else "record"


def source_sentence(rows):
    m = main_source(rows)
    other = sorted({r["source"] for r in rows} - {m})
    return SOURCES.get(m, m) + "".join("; those marked %s %s" % (o, SOURCES.get(o, o))
                                       for o in other)


def short(r):
    """The name a value carries: the CPU vendor (AMD, Intel), or the GPU (A100)."""
    if r["device"]:
        return gpu_short([r])
    w = r["cpu"].split()
    return w[0] if w and w[0] in ("AMD", "Intel") else r["cpu"]


def machines(rows):
    """`AMD: AMD EPYC 9454; Intel: Intel Xeon Platinum 8358; A100: NVIDIA A100 80GB PCIe`."""
    seen = {}
    for r in rows:
        seen.setdefault(short(r), r["device"] or r["cpu"])
    return "; ".join("%s: %s" % kv for kv in seen.items())


def render_full_run(rows):
    h = hours_one_machine(rows)
    lines = []
    if h is not None:
        lines.append("The full run takes about %d hours on one machine with an %s GPU, most of "
                     "it the CPU ladders." % (round(h), gpu_short(rows)))
    lines.append("The ladders time every rung up to 2^21 points on its own and the rungs from "
                 "2^22 up in one cumulative pass per repetition (LADDER_PASS_FROM, default "
                 "2^22); LADDER_PASS_FROM=off times every rung on its own and about doubles "
                 "the ladder time.  Durations of the published run (benchmark/"
                 "durations.tsv), %s.  %s." % (source_sentence(rows), machines(rows)))
    text = textwrap.fill(" ".join(lines), width=90, initial_indent="  ", subsequent_indent="  ")
    body = [text]
    for label, val in per_part(rows):
        body.append(textwrap.fill(val, width=90, initial_indent="      %-30s" % label,
                                  subsequent_indent=" " * 36))
    return ("# BEGIN durations -- written by benchmark/tools/durations.py from\n"
            "# benchmark/durations.tsv (make durations); do not edit by hand.\n"
            "text_durations() {\n    cat <<'EOF'\n" + "\n".join(body) + "\nEOF\n}\n"
            "# END durations\n")


def gpu_short(rows):
    dev = next((r["device"] for r in rows if r["part"] == "gpu" and r["device"]), "")
    words = [w for w in dev.split() if w.lower() != "nvidia"]
    return words[0] if words else "NVIDIA"


def render_docs(rows):
    h = hours_one_machine(rows)
    head = []
    if h is not None:
        head.append("The full run takes about %d hours on one machine with an %s GPU, most of "
                    "it the CPU ladders." % (round(h), gpu_short(rows)))
    head.append("The ladders time every rung up to 2^21 points on its own and the rungs from "
                "2^22 up in one cumulative pass per repetition (`LADDER_PASS_FROM`, default "
                "2^22); `LADDER_PASS_FROM=off` times every rung on its own and about doubles "
                "the ladder time. Durations of the published run, %s; `make durations` writes "
                "them from the run records into `benchmark/durations.tsv`. %s."
                % (source_sentence(rows).replace("TIMES.txt", "`TIMES.txt`"), machines(rows)))
    text = textwrap.fill(" ".join(head), width=92)
    table = ["| part | time |", "|---|---|"]
    table += ["| %s | %s |" % (label.strip(), val) for label, val in per_part(rows)]
    return ("<!-- BEGIN durations: written by `make durations` from benchmark/durations.tsv -->\n"
            + text + "\n\n" + "\n".join(table) + "\n<!-- END durations -->\n")


def render_readme_hours(rows):
    h = hours_one_machine(rows)
    return "about %d hours" % round(h) if h is not None else "many hours"


# ---------------------------------------------------------------- the three places

def places(rows):
    """[(path, pattern, replacement)]: each place between its markers."""
    return [
        (FULL_RUN, re.compile(r"# BEGIN durations.*?# END durations\n", re.S), render_full_run(rows)),
        (DOCS, re.compile(r"<!-- BEGIN durations.*?<!-- END durations -->\n", re.S), render_docs(rows)),
        (README, re.compile(r"(<!-- durations:hours -->).*?(<!-- /durations:hours -->)", re.S),
         r"\g<1>" + render_readme_hours(rows) + r"\g<2>"),
    ]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", help="read the run records under this directory")
    ap.add_argument("--check", action="store_true", help="compare, change nothing")
    a = ap.parse_args(argv)
    if a.runs:
        rows = from_runs(a.runs)
        if not rows:
            print("durations: no run under %s records its stage times (TIMES.txt); "
                  "benchmark/durations.tsv is kept" % a.runs)
            return 1
        write_tsv(rows)
        print("durations: %d parts from %s -> benchmark/durations.tsv" % (len(rows), a.runs))
    rows = read_tsv()
    bad = []
    for path, pat, new in places(rows):
        old = open(path).read()
        if not pat.search(old):
            bad.append("%s: no durations markers" % os.path.relpath(path, ROOT))
            continue
        upd = pat.sub(lambda m: m.expand(new) if "\\g<" in new else new, old, count=1)
        if upd != old:
            if a.check:
                bad.append("%s differs from benchmark/durations.tsv (make durations)"
                           % os.path.relpath(path, ROOT))
            else:
                open(path, "w").write(upd)
                print("durations: rendered %s" % os.path.relpath(path, ROOT))
    for b in bad:
        print("durations: " + b, file=sys.stderr if not a.check else sys.stdout)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
