#!/usr/bin/env python3
"""check_tex.py -- compiles every generated table fragment in one document (two
pdflatex passes, in a temporary directory) and fails on a LaTeX error, a duplicate
label, an undefined reference, a missing PDF or a missing pdflatex.  Captions, header
cells and notes are written into the fragments unescaped, so only LaTeX checks them."""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
% The packages the fragments declare in their `% requires:` lines, and the
% page geometry of the manuscript, at which the tables are compiled.
\usepackage{geometry}
\geometry{left=27mm,right=27mm,top=25mm,bottom=26mm}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{rotating}
\begin{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "paper"),
                    help="the table tree to check")
    ap.add_argument("--keep", action="store_true",
                    help="leave the scratch build for inspection")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"check_tex: no table tree at {args.dir} -- run "
              f"`make tables` first")
        return 1
    engine = shutil.which("pdflatex")
    if not engine:
        # No pdflatex is a failure, not a pass.
        print("check_tex: no pdflatex on this machine -- the tables are NOT checked;\n"
              "           add a TeX distribution with pdflatex (TeX Live; make "
              "requirements names the package for this system)")
        return 1

    tex = []
    for d, _s, fs in os.walk(args.dir):
        for f in sorted(fs):
            # `preview.tex` is a complete document, not a fragment.
            if f.endswith(".tex") and f != "preview.tex":
                tex.append(os.path.abspath(os.path.join(d, f)))
    if not tex:
        print(f"check_tex: no .tex files under {args.dir}")
        return 1

    work = tempfile.mkdtemp(prefix="bench-tabcheck-")
    doc = PREAMBLE + "\n".join(
        r"\clearpage\input{%s}" % t.replace("\\", "/") for t in sorted(tex)
    ) + "\n\\end{document}\n"
    open(os.path.join(work, "check.tex"), "w").write(doc)

    log = ""
    for _pass in (1, 2):
        r = subprocess.run([engine, "-interaction=nonstopmode", "check.tex"],
                           cwd=work, capture_output=True, text=True)
        log = r.stdout + r.stderr
    logfile = os.path.join(work, "check.log")
    if os.path.exists(logfile):
        log = open(logfile, errors="replace").read()

    errors = [l for l in log.splitlines() if l.startswith("!")]
    dup = [l for l in log.splitlines() if "multiply defined" in l]
    undef = [l for l in log.splitlines()
             if "Reference" in l and "undefined" in l]
    pdf = os.path.exists(os.path.join(work, "check.pdf"))

    print(f"check_tex: {len(tex)} table(s) from {args.dir}")
    ok = True
    if errors:
        ok = False
        print(f"  {len(errors)} LaTeX error(s):")
        for e in errors[:12]:
            print("   ", e)
    # A duplicate label fails: `\ref` resolves to the fragment read last.
    if dup:
        ok = False
        print(f"  {len(dup)} duplicate label(s):")
        for d in dup[:12]:
            print("   ", d.strip())
    # An undefined reference (printed `??`) fails.  The only `\ref` emitted is
    # the continuation note of a split table, pointing to its own first float.
    if undef:
        ok = False
        print(f"  {len(undef)} undefined reference(s) -- a table is pointing at "
              f"one that was not built; make it prose:")
        for u in sorted(set(undef))[:8]:
            print("   ", u.strip())
    if not pdf:
        ok = False
        print("  no PDF produced")
    print("  " + ("PASS" if ok else "FAIL"))
    if args.keep:
        print("  scratch:", work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
