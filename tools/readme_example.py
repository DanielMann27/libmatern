#!/usr/bin/env python3
"""readme_example.py -- compiles the first ```c block under SECTION of README.md against
build/libmatern.a (no warnings allowed) and checks its M(...) lines against the first plain
``` block starting with "backend "; a missing section or block is an error."""
import os, re, subprocess, sys, tempfile

SECTION = '## Quick start'

def die(msg):
    print("readme-example: " + msg)
    sys.exit(1)

readme = open('README.md').read()
if SECTION not in readme:
    die("README.md has no %r section -- it was renamed or removed, and this check\n"
        "                cannot silently pass for having found nothing." % SECTION)
tail = readme[readme.index(SECTION):]

c_blocks = re.findall(r'```c\n(.*?)```', tail, re.S)
if not c_blocks:
    die("no ```c block under %r." % SECTION)
program = c_blocks[0]
if 'int main' not in program:
    die("the example is not a program (no `int main`).  It is a fragment, so the\n"
        "                README's claim that it can be copied and compiled is false.")

expected = re.findall(r'```\n(backend .*?)```', tail, re.S)
if not expected:
    die("the example has no claimed-output block, so there is nothing to verify it\n"
        "                against.")

lib = 'build/libmatern.a'
if not os.path.exists(lib):
    die("%s is missing; run `make` first." % lib)

with tempfile.TemporaryDirectory() as d:
    src, exe = os.path.join(d, 'readme.c'), os.path.join(d, 'readme')
    open(src, 'w').write(program)
    cc = os.environ.get('CC', 'cc')
    r = subprocess.run([cc, '-O2', '-Wall', '-Wextra', '-Iinclude', '-o', exe, src, lib, '-lm'],
                       capture_output=True, text=True)
    if r.returncode:
        die("the README's example DOES NOT COMPILE as printed:\n" + r.stderr.rstrip())
    if r.stderr.strip():
        die("the README's example compiles with warnings, which a first example must\n"
            "                not:\n" + r.stderr.rstrip())
    run = subprocess.run([exe], capture_output=True, text=True)
    if run.returncode:
        die("the README's example compiled but exited %d." % run.returncode)

def numbers(text):
    """The M(...) value lines of an output; the backend line is ignored."""
    return [l.strip() for l in text.strip().splitlines() if l.strip().startswith('M(')]

want, got = numbers(expected[0]), numbers(run.stdout)
if not want:
    die("the claimed-output block carries no M(...) lines to compare.")
if want != got:
    die("the README claims output the example does not produce.\n"
        "                claimed: %s\n                actual : %s" % (want, got))

print("readme-example: compiles clean, runs, and its %d printed values match the README" % len(want))
