#!/usr/bin/env python3
"""make_paper_tables.py -- builds the paper's tables (make_paper_figures.PAPER_TABLES) into
benchmark/tables/paper/; needs pdflatex unless --no-preview.  Runs are loaded and grouped
by make_paper_figures; the tables come from paper_tables.  Exit status: 1 for no runs, a
refused input or a LaTeX error in a preview; 2 for no pdflatex, or under --strict an
incomplete run or a refused input."""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
PLOTS = os.path.join(BENCH, "plots")
OUT = os.path.join(HERE, "paper")
sys.path.insert(0, PLOTS)
sys.path.insert(0, HERE)

import make_paper_figures as mpf
import paper_tables as pt
import runs as mp
import style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the runs and row counts, write nothing")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to build from a run whose STATUS is not complete")
    ap.add_argument("--only", default="",
                    help="substring filter on the table stem")
    ap.add_argument("--out", default=OUT, help="output directory")
    ap.add_argument("--runs", default="",
                    help="read runs from this directory instead of "
                         "benchmark/runs")
    ap.add_argument("--no-preview", action="store_true",
                    help="write preview.tex but do not run pdflatex, and do "
                         "not fit tables to the column")
    ap.add_argument("--cpu-machine", default="",
                    help="the machine the paper set takes its CPU tables from first, "
                         "the machine-independent ones included (default: machines "
                         "without GPU data first; a stage this machine lacks comes "
                         "from the next machine that has it)")
    args = ap.parse_args()

    if not args.list and not args.no_preview and not shutil.which("pdflatex"):
        print("tables: needs pdflatex (column fitting, previews); not built --\n"
              "        add a TeX distribution with pdflatex (TeX Live; "
              "make requirements names the package for this system),\n"
              "        or `make tables ARGS=--no-preview` to write them "
              "unfitted to the column")
        return 2

    if args.runs:
        mpf.RUNS = os.path.abspath(args.runs)
    runs = mpf.load_all()
    if not runs:
        print(f"no runs in {mpf.RUNS}")
        return 1
    if args.list:
        for r in runs:
            print(f"{r['id']:44s} {r['status']:9s} {r['machine']:20s} "
                  f"{r['arch_pretty']}")
            for k in ("accuracy", "suppliers", "ladder", "sup_ladder",
                      "gpu", "gpu_sup", "gpu_ladder", "levin", "fold", "threads"):
                if r[k]:
                    print(f"    {k:11s} {len(r[k]):5d} rows")
        return 0

    out = args.out
    os.makedirs(out, exist_ok=True)

    bymach, dropped = mpf.group_by_machine(runs)
    if dropped:
        print(f"   superseded, not merged: {', '.join(dropped)}")

    # Only the runs built from are checked; a run without a STATUS file is incomplete.
    bad = [r["label"] for rs_ in bymach.values() for r in rs_ if r["status"] != "complete"]
    if bad:
        msg = "runs not marked complete: " + ", ".join(bad)
        if args.strict:
            print("refusing under --strict:", msg)
            return 2
        print("!!", msg, "-- the README.md of the tables will say so")

    order = mpf.cpu_order(bymach, args.cpu_machine)
    print(f"table set: the sources of the paper's tables; "
          f"CPU tables from {mpf.cpu_sources_line(bymach, order)}")
    cur = None              # the machine being built; None for the tree root

    def want(stem, *kinds, any_role=False):
        """Whether table `stem` is built here: --only, then the paper set for the current
        machine; `kinds`, the CPU run kinds it is built from, place a CPU table (any role
        for the skipped-stage notes, whose stems are joined lists)."""
        if args.only and args.only not in stem:
            return False
        roles = None if any_role else mpf.machine_roles(cur, bymach, order, kinds)
        names = [x.strip().replace("tab-", "", 1) for x in stem.split(" / ")]
        return any(mpf.in_paper_set(mpf.PAPER_TABLES, x, roles) for x in names)

    LAD = ("ladder", "sup_ladder")

    n = 0

    def emit(count, stem):
        nonlocal n
        n += count
        if count:
            print(f"    {stem}")

    # The accuracy tables are written once at the root, from the first CPU machine
    # (mpf.machine_order); machine_agreement checks every field across machines.
    cpu = [(m, rs) for m, rs in bymach.items() if mpf.is_cpu_machine(rs)]
    # The paper set's root tables come from the first CPU machine in its order.
    cpu.sort(key=lambda mr: order.index(mr[0]))
    cpu_acc = [(m, mpf.merge(rs, "accuracy")) for m, rs in cpu]
    cpu_sup = [(m, mpf.merge(rs, "suppliers")) for m, rs in cpu]
    agreement = (pt.machine_agreement([(lab, a + b) for (lab, a), (_, b)
                                       in zip(cpu_acc, cpu_sup)])
                 if len(cpu_acc) > 1 else None)

    all_runs = sorted((r for rs in bymach.values() for r in rs),
                      key=lambda r: r["id"])
    env_note = _runs_note(all_runs)
    # The root tables name only the CPU runs they read.
    cpu_runs = sorted((r for _, rs in cpu for r in rs
                       if r["accuracy"] or r["suppliers"]), key=lambda r: r["id"])
    root_note = _runs_note(cpu_runs)

    if cpu:
        acc = mpf.merge(cpu[0][1], "accuracy")
        sup = mpf.merge(cpu[0][1], "suppliers")
        arch0 = _arch(mpf.source_runs(cpu[0][1], "accuracy", "suppliers"))
        if agreement and agreement[0]:
            nf, bad, _ = agreement
            check = (f"{nf - bad} of {nf} accuracy fields agree across "
                     f"{len(cpu_acc)} CPU machines")
        else:
            check = "one CPU machine; agreement across machines not checked"
        print(f"== shared ({check})")
        for fam in ("z_nu", "h_nu_s"):
            if want(f"tab-medmax-{fam}"):
                emit(pt.tab_medmax(acc, sup, out, root_note, family=fam,
                                   arch=arch0),
                     f"tab-medmax-{fam}")
            if want(f"tab-exclusions-{fam}"):
                emit(pt.tab_exclusions(acc, sup, out, root_note, family=fam,
                                       arch=arch0),
                     f"tab-exclusions-{fam}")
        if want("tab-datasets"):
            emit(pt.tab_datasets(acc, out, root_note), "tab-datasets")
        if want("tab-references"):
            emit(pt.tab_references(acc, sup, out, root_note), "tab-references")
    else:
        pt.SKIPPED.append((
            "tab-medmax / tab-exclusions / tab-datasets / tab-references",
            "no CPU run carries accuracy rows -- these tables are the accuracy "
            "stage and there is nothing to read"))

    if want("tab-environment"):
        # One column per machine class (runs.arch_label short tag, `+GPU` with a device).
        byclass = {}
        for m, rs in bymach.items():
            for r in rs:
                lab = r.get("arch_short") or m
                pv = mpf.read_provenance(r["dir"])
                if pv:
                    # Derived fields (machine label, foreign load from loadavg.tsv) join
                    # the PROVENANCE dict.
                    pv = dict(pv, machine=m)
                    fl = pt.env_foreign_load(r["dir"])
                    if fl is not None:
                        pv = dict(pv, load_foreign=fl)
                byclass.setdefault(lab, []).append((r["id"], pv))
        machines = list(byclass.items())    # in machine order
        emit(pt.tab_environment(machines, out, env_note), "tab-environment")

    # Per machine: every table whose values depend on the hardware.
    for mach, rs in bymach.items():
        arch = _arch(rs)
        sub = os.path.join(out, mach)

        def note_of(*kinds):
            return _runs_note(mpf.source_runs(rs, *kinds))

        # The caption's machine, from the same runs as note_of.
        def arch_of(*kinds):
            return _arch(mpf.source_runs(rs, *kinds))

        own = mpf.kind_owner(rs)
        sup = mpf.merge(rs, "suppliers")
        lad, slad = mpf.merge(rs, "ladder"), mpf.merge(rs, "sup_ladder")
        gsup = mpf.merge(rs, "gpu_sup")
        print(f"== {mach}  {mpf.machine_pretty(rs)}")
        cur = mach
        # Labels carry the machine until cleared, so two machines' tables do not collide.
        pt.LABEL_SUFFIX = mach

        # Device tables name the CPU and GPU of their own run; the scraped device name is
        # added only when PROVENANCE lacks it.
        _gruns = [own[k] for k in ("gpu_sup", "gpu_ladder", "gpu", "gpu_klad",
                                   "gpu_levin") if k in own]
        garch0 = (_gruns[0]["arch_pretty"].split("[")[0].strip() if _gruns
                  else arch)
        dev = mpf.device_label(_gruns or rs)
        garch = garch0 if (not dev or dev in garch0) else f"{dev} on {garch0}"

        for fam in ("z_nu", "h_nu_s"):
            if lad and want(f"tab-throughput-{fam}", *LAD):
                # Assemblies from the family's supplier accuracy rows (for z_nu,
                # those of the ladder figure).
                emit(pt.tab_throughput(lad, slad, sub,
                                       note_of("ladder", "sup_ladder",
                                               *(("suppliers",) if slad and sup
                                                 else ())),
                                       arch_of("ladder", "sup_ladder"),
                                       family=fam,
                                       asm_of=mpf.preferred_assembly(sup,
                                                                     family=fam)),
                     f"{mach}/tab-throughput-{fam}")
        for fam in ("z_nu", "h_nu_s"):
            if gsup and want(f"tab-gpu-cost-{fam}"):
                emit(pt.tab_gpu_cost(gsup, sub, note_of("gpu_sup"), garch,
                                     family=fam),
                     f"{mach}/tab-gpu-cost-{fam}")
            if gsup and want(f"tab-gpu-exclusions-{fam}"):
                emit(pt.tab_gpu_exclusions(gsup, sub, note_of("gpu_sup"), garch,
                                           family=fam),
                     f"{mach}/tab-gpu-exclusions-{fam}")
            if gsup and want(f"tab-gpu-medmax-{fam}"):
                emit(pt.tab_gpu_medmax(gsup, sub, note_of("gpu_sup"), garch,
                                       family=fam),
                     f"{mach}/tab-gpu-medmax-{fam}")
        for fam in ("z_nu", "h_nu_s"):
            if gsup and want(f"tab-gpu-call-{fam}"):
                emit(pt.tab_gpu_call(
                         gsup, sub, note_of("gpu_sup"), garch, family=fam),
                     f"{mach}/tab-gpu-call-{fam}")
        pt.LABEL_SUFFIX = ""

    # A table whose stage no run carries is never attempted; it is listed as skipped here.
    for key, stems, why in (
        ("gpu_sup", "tab-gpu-cost / tab-gpu-call / tab-gpu-medmax / "
                    "tab-gpu-exclusions",
         "no run carries device kernel rows -- the GPU stage needs a machine "
         "with a CUDA device and the externals built for it"),
        ("ladder", "tab-throughput-*",
         "no run carries a ladder -- throughput is quoted at a ladder rung"),
    ):
        if (not any(r[key] for rs in bymach.values() for r in rs)
                and want(stems, any_role=True)):
            pt.SKIPPED.append((stems, why))

    # Output files this run did not write are listed, not deleted.  The preview.tex of a
    # directory that gets a preview below is not one of them: it is rewritten after this check.
    preview_dirs = {os.path.normpath(d) for d in [out] + [os.path.join(out, m) for m in bymach]}
    produced = set()
    for root, _d, files in os.walk(out):
        for fn in files:
            if fn == "preview.tex" and os.path.normpath(root) in preview_dirs:
                continue
            if fn.endswith((".tex", ".md", ".csv")):
                produced.add(os.path.relpath(os.path.join(root, fn), out)
                             .replace(os.sep, "/"))
    fresh = {os.path.relpath(p, out).replace(os.sep, "/") for p in pt.WRITTEN}
    fresh |= {"README.md"}
    stale = sorted(produced - fresh)
    if stale:
        print(f"\n!! {len(stale)} file(s) in {out} were NOT produced by this run.")
        print("   A stale .tex is silent in the manuscript -- remove them, or")
        print("   regenerate into a clean directory:")
        for p in stale[:20]:
            print(f"     {p}")
        if len(stale) > 20:
            print(f"     ... and {len(stale) - 20} more")

    # One preview document per directory, at the manuscript's page geometry, so width
    # overflows are reported.
    print()
    previews = 0
    latex_errors = 0
    for d in [out] + [os.path.join(out, m) for m in bymach]:
        if not os.path.isdir(d):
            continue
        if d == out:
            title = "Machine-independent tables"
            sub = ("Tables without a cost column, written once for all "
                   "machines; the cross-machine comparison of the accuracy "
                   "fields is stated in README.md when two CPU "
                   "architectures are present.")
        else:
            m = os.path.basename(d)
            rs = bymach.get(m, [])
            title = f"{mpf.machine_pretty(rs) if rs else m}"
            has_dev = any(r.get(k) for r in rs
                          for k in ("gpu", "gpu_sup", "gpu_ladder", "gpu_klad",
                                    "gpu_levin"))
            sub = (r"Machine \texttt{%s}. The tables with a cost column "
                   r"measured on this machine%s."
                   % (_tex(m), ", and its device tables" if has_dev else ""))
        tex, pdf, over = pt.write_preview(d, title, sub,
                                          run=not args.no_preview)
        if not tex:
            continue
        previews += 1
        rel = os.path.relpath(d, out)
        where = "." if rel == "." else rel
        # pdflatex in nonstopmode writes a partial PDF on an error: reported as incomplete.
        errs = [o for o in over if len(o) == 2 and o[0] == "error"]
        if args.no_preview:
            print(f"    preview -> {where}/preview.tex  (--no-preview: not compiled, "
                  f"tables not fitted)")
        elif pdf and errs:
            print(f"    preview -> {where}/preview.pdf  "
                  f"(INCOMPLETE -- LaTeX stopped on an error, see below)")
        elif pdf:
            print(f"    preview -> {where}/preview.pdf")
        else:
            print(f"    preview -> {where}/preview.tex  (no PDF -- LaTeX stopped"
                  f"{' on an error, see below' if errs else ''})")
        if not args.no_preview and (errs or not pdf):
            latex_errors += 1
        wide = [o for o in over if len(o) == 3]
        for _k, msg in errs:
            print(f"        !! LaTeX error in this directory: {msg}")
        # After fit_to_column, `wide` holds only tables too wide at the smallest size.
        forms = pt.forms(d)
        rot = [f for f, _s, r in forms if r]
        small = sorted({s_ for _f, s_, r in forms if not r})
        if small:
            print(f"        upright at {', '.join(small)}"
                  + (f"; {len(rot)} rotated: {', '.join(rot)}" if rot else ""))
        elif rot:
            print(f"        {len(rot)} rotated: {', '.join(rot)}")
        for name, excess, size in sorted(wide, key=lambda x: -x[1]):
            print(f"        !! {name} is {excess:.0f}pt too wide even at {size} -- "
                  f"drop a column, or set it in a landscape float")

    write_index(out, bymach, all_runs, _pdflatex_used(args.no_preview))
    print(f"\n{n} tables, {previews} preview(s) -> {out}")

    if pt.SKIPPED:
        print(f"\n{len(pt.SKIPPED)} table(s) NOT built, with the input that was "
              f"missing:")
        seen = set()
        for stem, why in pt.SKIPPED:
            if (stem, why) in seen:
                continue
            seen.add((stem, why))
            print(f"    {stem}")
            print(f"        {why}")
    if mpf.UNREADABLE:
        print("\nINPUTS REFUSED (tables needing them were skipped):")
        for u in mpf.UNREADABLE:
            print("   ", u)
        return 2 if args.strict else 1
    # A preview stopped by a LaTeX error leaves its directory's tables unchecked.
    if latex_errors:
        print(f"\ntables: the preview of {latex_errors} director"
              f"{'y' if latex_errors == 1 else 'ies'} stopped on a LaTeX error (above) "
              f"-- the tables are written, but not all of them were fitted to the column")
        return 1
    return 0


def _runs_note(runs_):
    """The runs a table is built from, for its README.md section; incomplete runs flagged."""
    return ", ".join(f"`{r['label']}`" + ("" if r["status"] == "complete"
                                          else " (not marked complete)")
                     for r in runs_)


def _arch(rs):
    """The CPU model of the runs `rs`, from which the captions name the machine."""
    return rs[0]["arch_pretty"].split("[")[0].split(" + ")[0].strip()


def _tex(t):
    return style.tex_escape(t)


def _pdflatex_used(no_preview):
    """The first line of `pdflatex --version` (no path), or the reason none is used."""
    if no_preview:
        return "not run (--no-preview): every table at its first size, not fitted to the column"
    try:
        r = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True)
        first = (r.stdout.splitlines() or ["unknown"])[0].strip()
    except OSError:
        first = "unknown"
    return first or "unknown"


def write_index(out, bymach, runs_, pdflatex):
    """Write README.md into the output tree: file formats, machine directories, each table's
    fitted form, the runs, per table its runs and facts (paper_tables.MOVED), generators."""
    lines = [
        "# `benchmark/tables/paper/` -- the generated table set", "",
        "Written by `make tables`.  No value here is measured by the",
        "generator: every value is parsed from the CSV files of the runs",
        "(`<id>/*.csv` under `benchmark/runs/`, unless `--runs` named another",
        "root), so this directory can be deleted and rebuilt from the runs.", "",
        "Three files are written per table:", "",
        "| extension | contents |",
        "|---|---|",
        "| `.tex` | `booktabs` tabular inside a `table` environment, with the "
        "caption, the label and the note; the required packages are listed in "
        "its `% requires:` line |",
        "| `.md` | the same table in Markdown |",
        "| `.csv` | the same table at full precision, without presentation "
        "markers, including values the `.tex` does not print (`Cap` cells) |",
        "",
        "## Machine-dependent tables", "",
        "Tables without a cost column are at the root, except the device",
        "tables.  Every table with a cost column, and every device table, is in",
        "a per-machine subdirectory named after the machine's CPU model; a cost",
        "table's caption names the processor it was measured on (the CPU by",
        "vendor, the GPU by model), and the table below names the machine in",
        "full.  Runs made with the same CPU model are one machine.", "",
        "| directory | machine |", "|---|---|",
    ]
    for m, rs in bymach.items():
        lines.append(f"| `{m}/` | {mpf.machine_pretty(rs)} |")
    lines += ["", "## How each table is set", "",
              "The generator measures each table against the manuscript's",
              "column width and steps it down a fixed sequence until it fits:",
              "`\\small` -> `\\footnotesize` -> `\\scriptsize` -> a rotated",
              "(`sidewaystable`) float.  Rotation is the last step, so a table",
              "that fits upright is not rotated.  A rotated fragment declares",
              "`rotating` in its `% requires:` line.", "",
              "| table | set at | rotated |", "|---|---|---|"]
    for d in [out] + [os.path.join(out, m) for m in bymach]:
        if not os.path.isdir(d):
            continue
        rel = "" if d == out else os.path.basename(d) + "/"
        for fn, size, rot in pt.forms(d):
            lines.append(f"| `{rel}{fn}` | `\\{size}` | {'yes' if rot else ''} |")
    lines += ["", "## Runs behind this build", "",
              "Each run is named by its stamp and stage, with its machine.", ""]
    for r in runs_:
        lines.append(f"- `{r['label']}` ({mpf.machine_of(r)})")
    lines += ["", "## Tables", "",
              "Per table, the runs it is built from and the details its",
              "caption and note do not carry: method, counts, checks and the",
              "machine in full.", ""]
    for d in [out] + [os.path.join(out, m) for m in bymach]:
        rel = "" if d == out else os.path.basename(d) + "/"
        for path, runs, facts in sorted(
                (e for e in pt.MOVED
                 if os.path.dirname(e[0]) == os.path.abspath(d)),
                key=lambda e: e[0]):
            lines += [f"### `{rel}{os.path.basename(path)[:-4]}`", ""]
            if runs:
                lines.append(f"- Runs: {runs}")
            lines += [f"- {pt._demath(f)}" for f in facts]
            lines.append("")
    lines += ["", "## Generators", "",
              "The programs that generated this table set (version strings "
              "only).  pdflatex sets the type size and rotation of each table.",
              "", "| program | version |", "|---|---|"]
    lines += [f"| {name} | {ver} |" for name, ver in mp.generator_versions()]
    lines.append(f"| pdflatex | {pdflatex} |")
    lines += ["", "Regenerate with `make tables`; `make tables "
              "ARGS=--list` prints what it would use and writes nothing."]
    open(os.path.join(out, "README.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
