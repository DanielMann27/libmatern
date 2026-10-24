#!/usr/bin/env python3
# derive_port.py <upstream dir> <output file> -- derives the KAUST CuBesselK kernels for
# kaust_port.c (host C and CUDA) from the user's sha256-checked copy; every rewriting rule
# must match a fixed number of times.  The output contains KAUST code ("All rights reserved",
# no licence) and is written under references/build/, which is not distributed.  This script
# holds only the rules and the short fragments they match.
import hashlib
import re
import sys

COMMIT = "94ec9ca4bb9d7524c196f044503519ab30d8eb39"
DIGESTS = {
    "logbesselk.cu": "5a51e1a5c7c08ecc31696db58d3a69e9fc9e3fe6d8071e57af26dcaa666ee5ed",
    "logbesselk.h": "650bafdad9c4444992fbf1b5435bc2d16db0e70fff3135066d79f08eeea30550",
}


def die(msg):
    sys.stderr.write("derive_port.py: %s\n" % msg)
    sys.exit(1)


def read_checked(updir, name):
    path = "%s/%s" % (updir, name)
    try:
        data = open(path, "rb").read()
    except OSError as e:
        die("%s: %s" % (path, e.strerror))
    got = hashlib.sha256(data).hexdigest()
    if got != DIGESTS[name]:
        die("%s: sha256 %s, expected %s (stsds/CuBesselK @%s)"
            % (path, got, DIGESTS[name], COMMIT[:7]))
    return data.decode("utf-8")


def replace(text, old, new, count, what):
    n = text.count(old)
    if n != count:
        die("rule '%s': %d matches of %r, expected %d" % (what, n, old, count))
    return text.replace(old, new)


def subn(text, pattern, new, count, what, flags=0):
    out, n = re.subn(pattern, new, text, flags=flags)
    if n != count:
        die("rule '%s': %d matches of /%s/, expected %d" % (what, n, pattern, count))
    return out


def block_after(text, start, what):
    """The text between the '{' that ends `start` and its matching '}'."""
    i = text.find(start)
    if i < 0 or text.find(start, i + 1) >= 0:
        die("rule '%s': anchor %r must occur exactly once" % (what, start))
    j = text.index("{", i + len(start) - 1)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k], text[i:k + 1]
    die("rule '%s': unbalanced braces" % what)


def definition(text, signature, what):
    """A whole definition starting at `signature` up to its closing brace (and a
    following ';' for an initialiser)."""
    body, whole = block_after(text, signature, what)
    end = text.index(whole) + len(whole)
    if text[end:end + 1] == ";":
        whole += ";"
    return whole


def main():
    if len(sys.argv) != 3:
        die("usage: derive_port.py <upstream dir> <output file>")
    updir, out = sys.argv[1], sys.argv[2]
    cu = read_checked(updir, "logbesselk.cu")
    hdr = read_checked(updir, "logbesselk.h")

    # logbesselk.h is read for two constants, not included: it redefines DBL_EPSILON,
    # which <float.h> owns.
    eps = re.findall(r"^#define\s+DBL_EPSILON\s+(\S+)\s*$", hdr, re.M)
    smax = re.findall(r"^#define\s+SQRT_DBL_MAX\s+(\S+)\s*$", hdr, re.M)
    if len(eps) != 1 or len(smax) != 1:
        die("logbesselk.h: expected one DBL_EPSILON and one SQRT_DBL_MAX")

    parts = [
        definition(cu, "__constant__ double g1_dat[14] =", "g1 table"),
        definition(cu, "__constant__ cheb_series g1_cs =", "g1 series"),
        definition(cu, "__constant__ double g2_dat[15] =", "g2 table"),
        definition(cu, "__constant__ cheb_series g2_cs =", "g2 series"),
        definition(cu, "__device__ void cheb_eval_cuda(", "Chebyshev evaluation"),
        definition(cu, "__device__ void temme_gamma(", "Temme gamma"),
        definition(cu, "__device__ void besselk_scaled_temme(", "Temme series"),
        definition(cu, "__device__ double f(", "integrand"),
    ]
    src = "\n\n".join(parts)

    # names and qualifiers
    for old, new, n in [("__constant__", "KA_DATA", 4), ("__device__", "KA_FN", 4)]:
        src = replace(src, old, new, n, "qualifier " + old)
    for old, new, n in [(r"\bg1_dat\b", "ka_g1_dat", 2), (r"\bg2_dat\b", "ka_g2_dat", 2),
                        (r"\bg1_cs\b", "ka_g1_cs", 2), (r"\bg2_cs\b", "ka_g2_cs", 2),
                        (r"\bcheb_series\b", "ka_cheb", 3),
                        (r"\bcheb_eval_cuda\b", "ka_cheb_eval", 3),
                        (r"\btemme_gamma\b", "ka_temme_gamma", 2),
                        (r"\bbesselk_scaled_temme\b", "ka_besselk_scaled_temme", 1),
                        (r"\bDBL_EPSILON\b", "KA_DBL_EPSILON", 3),
                        (r"\bdouble f\(", "double ka_f(", 1)]:
        src = subn(src, old, new, n, "rename " + new)

    # modified_besselk: the two branches; kaust_port.c applies e^{-x}, and e10 only when fix = 1
    body, _ = block_after(cu, "__device__ double modified_besselk(double nu, double x) {",
                          "modified_besselk")
    small, _ = block_after(body, "if (x <= 0.1) {", "small-argument branch")
    quad, _ = block_after(body, "else {", "quadrature branch")

    small = replace(small, "return K_nu * exp(-x);",
                    "*K_out   = K_nu;                       /* still e^{+x}-scaled */\n"
                    "        *e10_out = e10;", 1, "small branch return")
    small = subn(small, r"\bbesselk_scaled_temme\b", "ka_besselk_scaled_temme", 1,
                 "rename in small branch")
    small = subn(small, r"\bSQRT_DBL_MAX\b", "KA_SQRT_DBL_MAX", 1, "SQRT_DBL_MAX")

    quad = subn(quad, r"\bd_intervals\b", "iv", 5, "panel count")
    quad = replace(quad, "return exp(res);", "return res;", 1, "quadrature return")
    quad = subn(quad, r"\bf\(", "ka_f(", 2, "integrand calls")
    stable = replace(quad, "ka_f(", "ka_f_stable(", 2, "stable integrand")
    rng = subn(stable, r"const double h = 9\.0 / iv;[^\n]*\n",
               "const double h = (t1 - t0) / iv;\n", 1, "range step")
    rng = replace(rng, "double t_m = m * h;", "double t_m = t0 + m * h;", 2, "range nodes")

    fn = lambda sig, b: "KA_FN %s\n{%s}\n" % (sig, b)
    branches = "\n".join([
        fn("void ka_small_core(double nu, double x, double *K_out, int *e10_out)", small),
        fn("double ka_trapezoid_log(double nu, double x, int iv)", quad),
        fn("double ka_trapezoid_log_stable(double nu, double x, int iv)", stable),
        fn("double ka_trapezoid_log_range(double nu, double x, int iv, double t0, double t1)",
           rng),
    ])

    text = (
        "/* GENERATED by references/kaust_logbesselk/derive_port.py from the user's\n"
        " * copy of stsds/CuBesselK @%s (src/logbesselk.cu, include/logbesselk.h).\n"
        " * Contains KAUST code (\"All rights reserved\"): not to be distributed.  Do not\n"
        " * edit; rerun the derivation. */\n"
        "#ifndef KAUST_DERIVED_H\n#define KAUST_DERIVED_H\n\n"
        "#define KA_DBL_EPSILON  %s\n#define KA_SQRT_DBL_MAX %s\n\n"
        "%s\n\n%s\n#endif\n" % (COMMIT[:7], eps[0], smax[0], src, branches))
    with open(out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
