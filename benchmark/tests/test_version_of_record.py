"""test_version_of_record.py -- runs.version_of_record against fixture runs.

A copy of benchmark/plots in a scratch tree outside any git checkout, with RELEASE naming the
generators' commit, reads its version from RELEASE and compares a run's MANIFEST.txt with its
own files:
  same commit                       -> one version of record
  other commit, sources identical   -> "all 2 sources ... identical (MANIFEST.txt)"
  one source edited (control)       -> "1 of the 2 recorded sources differs"
  one source removed (control)      -> "1 is missing"
and, when the tree is the top of a git checkout with a parent commit, the git route: a run at
HEAD~1 is told every file of `git diff --name-only HEAD~1` (uncommitted changes included).
Exit 0 when every case holds; each failure is printed."""
import hashlib, os, shutil, subprocess, sys, tempfile

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
G = "a" * 40           # the generators' commit in the scratch tree
C = "b" * 40           # a run commit that is not the generators'
fails = []


def check(name, cond, got):
    if not cond:
        fails.append(f"{name}: {got}")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_dir(base, name, commit, manifest):
    d = os.path.join(base, name)
    os.makedirs(d)
    with open(os.path.join(d, "PROVENANCE.txt"), "w") as f:
        f.write(f"release            0.9.5\nsource_commit      {commit} (fixture)\n"
                "git_dirty          no\n")
    with open(os.path.join(d, "MANIFEST.txt"), "w") as f:
        f.write("# fixture\n" + "".join(f"{h}  {p}\n" for p, h in manifest))
    return d


tmp = tempfile.mkdtemp()
try:
    tree = os.path.join(tmp, "tree")
    shutil.copytree(os.path.join(ROOT, "benchmark", "plots"), os.path.join(tree, "benchmark", "plots"),
                    ignore=shutil.ignore_patterns("paper-final", "__pycache__"))
    for rel in ("src/router/pipeline_vector.h", "Makefile"):
        os.makedirs(os.path.dirname(os.path.join(tree, rel)) or tree, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), os.path.join(tree, rel))
    with open(os.path.join(tree, "RELEASE"), "w") as f:
        f.write(f"commit   {G}\ndescribe v0.0.0\n")
    manifest = [(p, sha(os.path.join(tree, p))) for p in ("src/router/pipeline_vector.h", "Makefile")]
    manifest += [("build/libmatern.a", "0" * 64),                 # a build output, not in the tree
                 ("/data/z_nu/z_nu-grid_262144.bin", "1" * 64)]  # the corpus, left out
    sys.path.insert(0, os.path.join(tree, "benchmark", "plots"))
    import runs as mp
    check("scratch tree reads RELEASE", mp.generator_commit()[1] == G, mp.generator_commit())

    runs = os.path.join(tmp, "runs")
    same = run_dir(runs, "same", G, manifest)
    _, v = mp.version_of_record([("same", {"source_commit": G, "git_dirty": "no"}, same)])
    check("same commit", v.startswith(f"the generators and every run are at commit {G[:12]}"), v)

    other = run_dir(runs, "other", C, manifest)
    pv = {"source_commit": f"{C} (fixture)", "git_dirty": "no"}
    lines, v = mp.version_of_record([("other", pv, other)])
    check("other commit: NOT at one commit", "NOT at one commit" in v, v)
    check("other commit: sources identical",
          "all 2 sources the runs record are identical (MANIFEST.txt)" in v, v)
    check("other commit: the build output is not in the tree",
          any("1 is not here" in ln for ln in lines), lines[-4:])

    src = os.path.join(tree, "src/router/pipeline_vector.h")
    keep = open(src, "rb").read()
    with open(src, "ab") as f:
        f.write(b"/* edited */\n")
    _, v = mp.version_of_record([("other", pv, other)])
    check("control: an edited source is caught", "1 of the 2 recorded sources differs" in v, v)
    with open(src, "wb") as f:
        f.write(keep)

    os.remove(os.path.join(tree, "Makefile"))
    _, v = mp.version_of_record([("other", pv, other)])
    check("control: a missing source is caught", "0 of the 2 recorded sources differ and 1 is missing" in v, v)

    # the git route, on the real tree when it is the top of a checkout with a parent
    def git(*a):
        r = subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    top, parent = git("rev-parse", "--show-toplevel"), git("rev-parse", "HEAD~1")
    if top and os.path.realpath(top) == ROOT and parent:
        want = set((git("diff", "--name-only", parent, "HEAD") or "").splitlines())
        want |= set((git("diff", "--name-only", "HEAD") or "").splitlines())
        code = (f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'benchmark', 'plots')!r});"
                "import runs as mp; print(mp.source_difference(sys.argv[1], [])[0])")
        out = subprocess.run([sys.executable, "-c", code, parent], capture_output=True, text=True).stdout
        n = len(want)
        lead = "no file differs" if not n else f"{n} file{'s' * (n > 1)} differ"
        check("git route: counts git diff", out.startswith(lead), out.strip()[:200])
        check("git route: says how", "git diff --name-only" in out, out.strip()[:200])
    else:
        print("git route: SKIPPED (the tree is not the top of a git checkout with a parent commit)")
finally:
    shutil.rmtree(tmp)

for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
