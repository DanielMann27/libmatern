#!/usr/bin/env python3
"""audit.py -- static checks on the source tree (make audit, from the repository root)."""
import os, re, sys, glob, fnmatch, subprocess

FAIL = []


def check(name, bad, note=""):
    if bad:
        FAIL.append(name)
        print("  FAIL  %s (%d)" % (name, len(bad)))
        for b in bad[:8]:
            print("          " + str(b))
        if note:
            print("        " + note)
    else:
        print("  ok    %s" % name)


def skip(name, why):
    print("  skip  %s (%s)" % (name, why))


SOURCE_HEADERS = sorted(glob.glob('src/*/*.h') + glob.glob('src/*/*/*.h')
                        + glob.glob('src/*/*.cuh') + glob.glob('include/matern_device/*.cuh'))
SOURCE_FILES = sorted(glob.glob('src/*/*.c') + glob.glob('src/*/*/*.c')
                      + glob.glob('src/*/*.cu'))
LIBRARY_SOURCES = SOURCE_HEADERS + SOURCE_FILES + ['include/matern.h', 'include/matern_device.cuh']


def code_only(path):
    """Source text with C comments removed, split into lines."""
    text = open(path, encoding='utf-8').read()
    text = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return text.split('\n')


# 1. The library includes nothing from outside its own folder.
ROOT = os.path.realpath('.')
bad = []
for f in LIBRARY_SOURCES:
    for n, l in enumerate(open(f, encoding='utf-8'), 1):
        m = re.search(r'#\s*include\s*"(\.\.[^"]*)"', l)
        if m:
            target = os.path.realpath(os.path.join(os.path.dirname(f), m.group(1)))
            if not (target.startswith(ROOT + os.sep) and os.path.exists(target)):
                bad.append((f, n))
check("no include escapes the library", bad)

# 2. Build products live under a build directory only.
SKIP_DIRS = {'.git', 'build', 'build-ab'}
# Externals fetched by `make externals` (git-ignored, outside the sources).
SKIP_PATHS = re.compile(r'^(benchmark/runs|references/boost|references/gsl/[^/]+|\.venv)$')
stray = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS
               and not SKIP_PATHS.match(os.path.relpath(os.path.join(root, d), '.'))]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), '.')
        if re.search(r'\.(o|a|so|d|pyc)$', f) or '__pycache__' in rel:
            stray.append(rel)
check("no build products outside build/", sorted(stray))

# 3. Every header under src/ and include/matern_device/ is included by something.
allh = {os.path.basename(f) for f in SOURCE_HEADERS}
refd = set()
for f in SOURCE_HEADERS + SOURCE_FILES + ['include/matern_device.cuh']:
    for m in re.finditer(r'#\s*include\s*"([^"]+)"', open(f, encoding='utf-8').read()):
        refd.add(os.path.basename(m.group(1)))
check("no orphaned headers", sorted(allh - refd))

# 4. A table row of README.md or docs/*.md whose first cell holds a backticked repository
# path (starting with a top-level folder) names a path that exists.
TOP = {d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')}
missing = []
for doc in ['README.md'] + sorted(glob.glob('docs/*.md')):
    for n, line in enumerate(open(doc, encoding='utf-8'), 1):
        line = line.strip()
        if not line.startswith('|') or line.count('|') < 2:
            continue
        for m in re.finditer(r'`([A-Za-z0-9_.*/-]+)`', line.split('|')[1]):
            path = m.group(1)
            if path.split('/')[0] in TOP and '/' in path and not glob.glob(path.rstrip('/')):
                missing.append("%s:%d: `%s`" % (doc, n, path))
check("README and docs tables name paths that exist", missing)

# 5. ISA-neutral kernels (bessel/ without ISA suffix, *_portable.h, scalar/) use no raw intrinsics.
neutral = ([f for f in SOURCE_HEADERS if '/bessel/' in f and f.endswith('.h')
            and not re.search(r'_avx(512|2)\.h$', f)]
           + [f for f in SOURCE_HEADERS if f.endswith('_portable.h')]
           + glob.glob('src/backends/scalar/*.c'))
bad = [(f, n) for f in neutral for n, l in enumerate(code_only(f), 1)
       if re.search(r'_mm512_|_mm256_|__m512d|__m256d', l)]
check("ISA-neutral kernels use no raw intrinsics", bad,
      "the per-ISA form belongs in src/numeric")

# 6. Floating-point contraction is pinned in the Makefile (comments ignored).
def uncommented(path):
    return "\n".join(line.split('#', 1)[0] for line in open(path, encoding='utf-8'))


bad = [] if '-ffp-contract=' in uncommented('Makefile') else ['Makefile']
check("floating-point contraction pinned", bad)

# Checks 7 and 10 use git's file list only at the top level of a git checkout:
# inside another repository git would list that repository's files.
def tracked_files():
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True)
        if top.returncode != 0 or os.path.realpath(top.stdout.strip()) != ROOT:
            return None
        ls = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return ls.stdout.split() if ls.returncode == 0 else None


def ignored(path, patterns):
    """`path` matches a .gitignore pattern (the forms .gitignore uses here)."""
    parts = path.split('/')
    for p in patterns:
        if p.endswith('/'):
            p = p[:-1]
            dirs = ['/'.join(parts[:i]) for i in range(1, len(parts))]
            if ('/' in p and any(fnmatch.fnmatch(d, p) for d in dirs)) or \
               ('/' not in p and any(fnmatch.fnmatch(d, p) for d in parts[:-1])):
                return True
        elif fnmatch.fnmatch(path if '/' in p else parts[-1], p):
            return True
    return False


def files_on_disk():
    """The files of this tree that .gitignore does not exclude."""
    patterns = [l.strip() for l in open('.gitignore', encoding='utf-8')
                if l.strip() and not l.startswith('#')]
    out = []
    for d, dirs, files in os.walk('.'):
        dirs[:] = [x for x in dirs if x != '.git']
        for f in files:
            p = os.path.relpath(os.path.join(d, f), '.')
            if not ignored(p, patterns):
                out.append(p)
    return sorted(out)


TRACKED = tracked_files()

# 7. Third-party sources without a redistribution licence are not tracked.
NOT_REDISTRIBUTABLE = re.compile(
    r'^references/(kaust_logbesselk/upstream/|bessels_jl/upstream/|build/)')
if TRACKED is None:
    skip("no fetched upstream source or build product is tracked",
         "not a git checkout; the check applies to the committed tree")
else:
    bad = [t for t in TRACKED if NOT_REDISTRIBUTABLE.search(t)]
    bad += [t for t in TRACKED if re.search(r'\.(o|a|so|d|pyc)$|\.bak', t)]
    check("no fetched upstream source or build product is tracked", bad,
          "upstream sources are fetched by `make references` (references/fetch.sh), "
          "the KAUST ones only by `make kaust DOWNLOAD=yes`")

# 8. Every `make <target>` named in the documentation exists in the Makefile.
targets = set(re.findall(r'^([A-Za-z0-9_./-]+)\s*:(?!=)', open('Makefile').read(), re.M))
DOCS = (glob.glob('*.md') + glob.glob('docs/*.md') + glob.glob('*/README.md')
        + glob.glob('*/*/README.md') + glob.glob('data/*.md'))
bad = []
for doc in DOCS:
    for n, l in enumerate(open(doc, encoding='utf-8'), 1):
        for t in re.findall(r'`make ([A-Za-z0-9_./-]+)', l):
            if t not in targets and not t.startswith('-') and '=' not in t:
                bad.append('%s:%d  make %s' % (doc, n, t))
check("every `make` target named in the docs exists", bad)

# 9. Comments and documentation describe the code, not its development history.
HISTORY = re.compile(r'\b20\d\d-\d\d-\d\d\b(?!T)|\[(CORRECTED|ADDED|CHANGED|FIXED)\b|predecessor'
                     r'|DESIGN\.md|DESIGN \d|RESULTS\.md|ACCURACY\.md|PERFORMANCE\.md|BENCHMARK_PLAN'
                     r'|\bbench/|reference-generation/|Paper_WM')
TEXT = [f for f in LIBRARY_SOURCES + glob.glob('test/*') + glob.glob('example/*')
        + glob.glob('tools/*.py') + glob.glob('benchmark/**/*', recursive=True)
        + glob.glob('references/*/*.[ch]*') + glob.glob('references/*/*/*.[ch]') + DOCS
        if os.path.isfile(f) and not f.endswith(('PROVENANCE.md', 'audit.py'))
        and '/upstream/' not in f
        and re.search(r'\.(c|h|cu|cuh|cpp|py|sh|sbatch|md)$', f)]
bad = ['%s:%d' % (f, n) for f in sorted(set(TEXT))
       for n, l in enumerate(open(f, encoding='utf-8', errors='replace'), 1) if HISTORY.search(l)]
check("no development-history markers in sources or docs", bad)

# 10. No tracked file carries KAUST CuBesselK's text ("All rights reserved").  Exempt:
# derive_port.py (the fragments it matches), PROVENANCE.md (quotes the notice), data/ and
# this file.
# The first two markers lead CuBesselK's g1 and g2 Chebyshev tables (GSL's tables,
# PROVENANCE.md, Licence).
KAUST_MARKERS = re.compile(r'1\.14516408366268311786898152867|1\.882645524949671835019616975350'
                           r'|cheb_eval_cuda|besselk_scaled_temme|modified_besselkKernel'
                           r"|Takekawa's implementation")
KAUST_EXEMPT = ('references/kaust_logbesselk/derive_port.py',
                'references/kaust_logbesselk/PROVENANCE.md', 'tools/audit.py')
# Without a git index: the files on disk that .gitignore does not exclude.
bad = []
for t in (TRACKED if TRACKED is not None else files_on_disk()):
    if t in KAUST_EXEMPT or t.startswith('data/') or not os.path.isfile(t):
        continue
    try:
        text = open(t, encoding='utf-8', errors='strict').read()
    except (UnicodeDecodeError, OSError):
        continue
    bad += ['%s:%d' % (t, n) for n, l in enumerate(text.splitlines(), 1)
            if KAUST_MARKERS.search(l)]
check("no KAUST CuBesselK text is tracked" if TRACKED is not None else
      "no KAUST CuBesselK text in the files of this tree (not a git checkout)", bad,
      "the KAUST kernels are ported on the user's machine by make kaust DOWNLOAD=yes")

# 11. One release in include/matern.h (STRING and MAJOR.MINOR.PATCH), CITATION.cff
# (the top-level `version:` only) and README.md ("libmatern X.Y.Z").
header = open('include/matern.h', encoding='utf-8').read()
parts = [re.search(r'^#define MATERN_VERSION_%s\s+(\d+)\s*$' % k, header, re.M)
         for k in ('MAJOR', 'MINOR', 'PATCH')]
string = re.search(r'^#define MATERN_VERSION_STRING\s+"([^"]*)"\s*$', header, re.M)
cff = re.findall(r'^version:\s*"?([^"\s]+)"?\s*$',
                 open('CITATION.cff', encoding='utf-8').read(), re.M)
readme = re.findall(r'\blibmatern (\d+\.\d+\.\d+)\b', open('README.md', encoding='utf-8').read())
bad = []
if not string or not all(parts):
    bad.append('include/matern.h: MATERN_VERSION_MAJOR, _MINOR, _PATCH and _STRING '
               'are not all defined')
else:
    release = string.group(1)
    spelled = '.'.join(m.group(1) for m in parts)
    if spelled != release:
        bad.append('include/matern.h: MATERN_VERSION_STRING "%s", MAJOR.MINOR.PATCH %s'
                   % (release, spelled))
    if len(cff) != 1:
        bad.append('CITATION.cff: %d top-level `version:` keys, not 1' % len(cff))
    elif cff[0] != release:
        bad.append('CITATION.cff: version %s, include/matern.h %s' % (cff[0], release))
    if not readme:
        bad.append('README.md: no "libmatern X.Y.Z" sentence')
    bad += ['README.md: libmatern %s, include/matern.h %s' % (v, release)
            for v in readme if v != release]
check("the release agrees in include/matern.h, CITATION.cff and README.md", bad)

# 12. The device half-integer table equals tools/half_integer_table_device.c's output, which
# depends on long double: compared only where long double has a 64-bit mantissa.
TABLE_HEADER = glob.glob('include/matern_device/half_integer_closed_form_device.cuh')
gen = 'build/half_integer_table_device'
os.makedirs('build', exist_ok=True)
built = subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-o', gen,
                        'tools/half_integer_table_device.c'], capture_output=True)
if built.returncode != 0 or len(TABLE_HEADER) != 1:
    check("the device half-integer table is the generator's", ['the generator does not build'
          if built.returncode else 'not exactly one half_integer_closed_form_device.cuh'])
else:
    out = subprocess.run([gen], capture_output=True, text=True).stdout
    mant = re.match(r'/\* LDBL_MANT_DIG (\d+) \*/', out)
    if not mant or mant.group(1) != '64':
        skip("the device half-integer table is the generator's",
             "long double is not x87 extended here")
    else:
        hexnum = r'-?0x[0-9a-f]+(?:\.[0-9a-f]*)?p[+-]\d+'
        want = [float.fromhex(x) for x in re.findall(hexnum, out)]
        text = open(TABLE_HEADER[0], encoding='utf-8').read()
        body = re.search(r'matern_half_integer_coefficients_device\[[^]]*\]\s*=\s*\{(.*?)\};', text, re.S)
        have = [float.fromhex(x) for x in re.findall(hexnum, body.group(1))] if body else []
        bad = [] if have == want else ['%s: %d values, %d differ from the generator'
                                       % (TABLE_HEADER[0], len(have),
                                          sum(a != b for a, b in zip(have, want)) + abs(len(have) - len(want)))]
        check("the device half-integer table is the generator's", bad)

# 13. Every place that states the durations of the published run (the full-run notice,
# docs/benchmark.md, README.md) is the rendering of benchmark/durations.tsv.
r = subprocess.run([sys.executable, '-B', 'benchmark/tools/durations.py', '--check'],
                   capture_output=True, text=True)
check("the durations agree with benchmark/durations.tsv",
      [l for l in r.stdout.splitlines() + r.stderr.splitlines() if l.strip()] or
      (['durations.py --check exited %d' % r.returncode] if r.returncode else []),
      "make durations renders them again")

sys.exit(1 if FAIL else 0)
