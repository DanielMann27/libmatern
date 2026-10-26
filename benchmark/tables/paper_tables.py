"""paper_tables.py -- the paper tables (.tex, .md, .csv) from the recorded runs,
with the ranking rules and labels of paper_figures; driven by make_paper_tables.py.
Accuracy tables go to the tree root, timing tables to one directory per machine."""
import os, csv, math, re, sys

# style, runs and paper_figures live in benchmark/plots/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "plots"))

import style
import runs
import paper_figures as pf
from paper_figures import (ACC_LAWS, LADDER_LAWS, LAW_LABEL, FAMILY_LABEL,
                           CATASTROPHIC,
                           best_assembly, lib_backend_for, scope_of, failed,
                           _f, _i, _ok)

LIB = style.LIB
# The device library's supplier name in the GPU CSVs.
LIB_CUDA = style.LIB_CUDA

# Library name in the comparison-row cells.
LIB_SHORT = "lib"

# Backend display order, from scalar to widest; shared with the figures.
BACKEND_ORDER = style.BACKEND_ORDER

# Every file written, in order.
WRITTEN = []

# (stem, reason) for every table not built; separate from paper_figures.SKIPPED.
SKIPPED = []

# (path of the .tex, runs, facts) per table written, for the README.md of the set.
MOVED = []

# Appended to every label while the driver builds the per-machine tables, so the
# copies carry distinct LaTeX labels; the driver sets and clears it.
LABEL_SUFFIX = ""

# (font size, \tabcolsep, environment), tried in order; the upright rungs must
# precede the rotated ones (fit_to_column steps a tall table down upright only).
SIZE_LADDER = [(r"\small", "4pt", "table"),
               (r"\footnotesize", "3.5pt", "table"),
               (r"\scriptsize", "3pt", "table"),
               (r"\footnotesize", "4pt", "sidewaystable"),
               (r"\scriptsize", "3pt", "sidewaystable")]

# Wrapping row-label column; `>{...}` requires the `array` package.
def wrapcol(width="2.4cm"):
    return r">{\raggedright\arraybackslash}p{%s}" % width


def fraccol(frac):
    """A wrapping column of `frac` of the line width, less the `\\tabcolsep`
    padding on both sides, so the fractions sum to the text width."""
    return (r">{\raggedright\arraybackslash}"
            r"p{\dimexpr %.4f\linewidth-2\tabcolsep\relax}" % frac)


# {abs path of the .tex: Table}, so fit_to_column can re-emit a table smaller.
BUILT = {}


def _skip(stem, why):
    SKIPPED.append((stem, why))
    return 0


# A cell: LaTeX for the `.tex`, plain text for the `.md`, and the full-precision
# value without markers for the `.csv`.
class Cell:
    __slots__ = ("tex", "md", "csv")

    def __init__(self, tex, md=None, csv=None):
        self.tex = tex
        self.md = str(tex) if md is None else md
        self.csv = self.md if csv is None else csv

    def __repr__(self):
        return f"Cell({self.md!r})"


def text(s):
    """A literal string, escaped for LaTeX."""
    return Cell(str(s))


def raw(tex, md=None, csv=None):
    """Already-LaTeX text: passes `tex_escape` untouched."""
    return Cell(style.Raw(tex), md, csv)


DASH = raw(r"--", "--", "")


def sig(v, n=3):
    """(LaTeX, plain) of `v` to `n` significant digits; (None, None) if missing."""
    if not _ok(v):
        return None, None
    a = abs(v)
    if a == 0:
        return "0", "0"
    if 1e-3 <= a < 1e4:
        dec = max(0, n - 1 - int(math.floor(math.log10(a))))
        s = f"{v:.{dec}f}"
        return s, s
    e = int(math.floor(math.log10(a)))
    m = v / (10.0 ** e)
    return (f"${m:.{n-1}f}\\times10^{{{e}}}$", f"{m:.{n-1}f}e{e}")


def num(v, n=3):
    """A number to `n` significant digits; `--` when missing."""
    t = sig(v, n)
    if t[0] is None:
        return DASH
    return Cell(style.Raw(t[0]), t[1], repr(v))


def count(v):
    """An integer, thin-space (LaTeX) or comma (Markdown) grouped, plain in CSV."""
    try:
        i = int(float(v))
    except (TypeError, ValueError):
        return DASH
    s = f"{i:,}"
    return Cell(style.Raw(s.replace(",", r"\,")), s, str(i))


def _grouped(i):
    """`1\\,048\\,576`: a thousands-grouped integer for LaTeX prose."""
    return f"{int(i):,}".replace(",", r"\,")


def ratio(v, n=2):
    """A dimensionless ratio, printed with a multiplication sign."""
    t = sig(v, n + 1)
    if t[0] is None:
        return DASH
    return Cell(style.Raw(t[0] + r"$\times$"), t[1] + "x", repr(v))


def ulp(row, key="u_max"):
    """A ULP statistic: `Cap` at or above CATASTROPHIC, plus a dagger when taken
    over a reduced population (`nonfinite > 0`); the CSV twin is unmarked."""
    v = _f(row, key)
    if not _ok(v):
        return DASH
    if v >= CATASTROPHIC:
        if _i(row, "nonfinite") > 0:
            return Cell(style.Raw(r"\textit{Cap}\textsuperscript{$\dagger$}"),
                        "Cap (dag)", repr(v))
        return Cell(style.Raw(r"\textit{Cap}"), "Cap", repr(v))
    t = sig(v, 3)
    if _i(row, "nonfinite") > 0:
        # `\textsuperscript`, not `$^{\dagger}$`: beside a math-mode value a
        # second math group is spaced as a separate formula.
        return Cell(style.Raw(t[0] + r"\textsuperscript{$\dagger$}"),
                    t[1] + " (dag)", repr(v))
    return Cell(style.Raw(t[0]), t[1], repr(v))


def dagger_note(rows):
    """The dagger legend, or "" when no row of `rows` has a reduced population."""
    if any(_i(r, "nonfinite") > 0 for r in rows):
        return (r"$\dagger$: maximum over a reduced population, non-finite "
                r"results excluded. ")
    return ""


def marker_note(rows, keys=("u_max",)):
    """The legends of the `Cap` and dagger markers `rows` carry in `keys`."""
    rows = [r for r in rows if r is not None]
    out = ""
    # $10^{6}$ here and in FINITE_NOTE is paper_figures.CATASTROPHIC, written out.
    if any(_ok(_f(r, k)) and _f(r, k) >= CATASTROPHIC
           for r in rows for k in keys):
        out += (r"\textit{Cap}: a "
                + ("maximum" if list(keys) == ["u_max"] else "statistic")
                + r" at or above the $10^{6}$ ULP cap; the value is in the "
                r"\texttt{.csv}. ")
    return out + dagger_note(rows)


def comparison_note(quantity, dashed):
    """The legend of the two comparison rows; `quantity` is "maximum" or "cost",
    `dashed` whether a cell of them shows `--`."""
    return (r"\emph{Lowest external}: the external implementation with the "
            r"lowest " + quantity + r" among those with a finite result at "
            r"every point of the dataset and a maximum below $10^{6}$ ULP; "
            r"\emph{Advantage factor}: the ratio of the "
            r"larger to the smaller of its value and this library's, labelled "
            r"with the implementation it favours (\texttt{" + LIB_SHORT
            + r"}: this library)"
            + (r"; --: no such implementation or, in the Advantage factor row, "
               r"no ratio because a value is missing or zero. "
               if dashed else ". "))


def dashed_cells(*rows):
    """Whether a cell of the comparison rows (label cell excluded) shows `--`."""
    return any(c is DASH or "--" in c.md for r in rows for c in r[1:])


# $10^{6}$: paper_figures.CATASTROPHIC (`failed` tests at or above it).
FINITE_NOTE = (r"\emph{finite}: number of datasets with a finite result at "
               r"every point and a maximum below $10^{6}$ ULP. ")


# The rule of paper_figures.best_assembly, as the captions state it.
BEST_RULE = ("the one with the lowest maximum ULP error among those with the "
             "fewest non-finite results")
BEST_ASSEMBLY = ("Each external implementation is shown in its best assembly for "
                 "each dataset, " + BEST_RULE + ".")
# The statistic of every cost the tables show (`ns_med`, `ns_ws_per_call_med`).
MEDIAN_COST = "Costs are medians over the timed repetitions."


def _sets(family):
    """`the $(z,\\nu)$ datasets`: a family as the captions name it."""
    return "the %s datasets" % pf.CONV.get(family, esc(family))


def _obtained(arch, gpu=False):
    """The caption sentence naming the machine, or "" when the label names none."""
    return esc(pf._obtained(arch, gpu))


def _caption(*sentences):
    return " ".join(s for s in sentences if s)


def _machine(arch):
    """The README fact naming the machine in full."""
    return (r"Machine: " + esc(arch) + ".") if arch else ""


class Table:
    """One table as a `.tex` fragment, a `.md` twin and a full-precision `.csv`
    without markers; a table with no data rows is not written."""

    def __init__(self, stem, caption, label, header, align=None,
                 header_groups=None, note="", md_note=""):
        self.stem = stem
        self.caption = caption
        self.label = label + (f"-{LABEL_SUFFIX}" if LABEL_SUFFIX else "")
        self.header = header            # LaTeX, written by this module
        # No spanning header in Markdown and CSV: the group title prefixes the name.
        self.md_header = [_demath(h) for h in header]
        for first, last, title in (header_groups or ()):
            for i in range(first - 1, min(last, len(self.md_header))):
                self.md_header[i] = f"{_demath(title)}: {self.md_header[i]}"
        self.align = align
        self.header_groups = header_groups
        self.note = note
        self.md_note = md_note or note
        self.runs = ""
        self.facts = []
        self.rows = []
        self.rules = set()
        self.level = 0                 # index into SIZE_LADDER

    def provenance(self, runs, *facts):
        """The runs and the facts (LaTeX sentences) for the README.md (MOVED)."""
        self.runs = runs
        self.facts = [f for f in facts if f and str(f).strip()]

    def add(self, cells, rule=False):
        assert len(cells) == len(self.header), \
            f"{self.stem}: {len(cells)} cells against {len(self.header)} columns"
        self.rows.append(cells)
        if rule:
            self.rules.add(len(self.rows) - 1)

    def _write_tex(self, path):
        size, colsep, env = SIZE_LADDER[min(self.level, len(SIZE_LADDER) - 1)]
        need = "booktabs, array" + (", rotating" if env != "table" else "")
        style.booktabs(
            path, self.caption, self.label, self.header,
            [[c.tex for c in r] for r in self.rows],
            align=self.align, note=self.note, note_raw=True,
            header_groups=self.header_groups,
            rules_after=self.rules,
            preamble_note=need,
            size=size, colsep=colsep, env=env,
            # A `sidewaystable` takes no optional placement argument.
            placement=None if env != "table" else "t")

    def shrink(self, path):
        """One step down the size ladder.  False when there is no step left."""
        if self.level >= len(SIZE_LADDER) - 1:
            return False
        self.level += 1
        self._write_tex(path)
        return True

    def rule_here(self):
        """Draw a rule after the row most recently added."""
        if self.rows:
            self.rules.add(len(self.rows) - 1)

    def write(self, outdir):
        if not self.rows:
            return 0
        # ROW_NOTES, attached here so every table showing such a row carries it.
        _md = self.md_note or getattr(self.note, "md", None) or ""
        _extra = _row_notes(self.rows)
        if _extra and _extra not in (self.note or ""):
            self.note = ((self.note or "") + " " + _extra).strip()
            if _md:
                _md = (_md + " " + _extra).strip()
        os.makedirs(outdir, exist_ok=True)
        tex = os.path.join(outdir, self.stem + ".tex")
        self._write_tex(tex)
        BUILT[os.path.abspath(tex)] = self
        md = os.path.join(outdir, self.stem + ".md")
        lines = [f"# {_demath(self.caption)}", ""]
        lines.append("| " + " | ".join(self.md_header) + " |")
        lines.append("|" + "|".join("---" for _ in self.md_header) + "|")
        for i, r in enumerate(self.rows):
            lines.append("| " + " | ".join(c.md for c in r) + " |")
        # `note` is often set after construction.
        mdn = _md or self.note
        if mdn:
            lines += ["", _demath(mdn)]
        open(md, "w").write("\n".join(lines) + "\n")
        cs = os.path.join(outdir, self.stem + ".csv")
        with open(cs, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.md_header)
            for r in self.rows:
                w.writerow([c.csv for c in r])
        WRITTEN.extend([tex, md, cs])
        MOVED.append((os.path.abspath(tex), self.runs, list(self.facts)))
        return 1


_TEX_CMD = re.compile(r"\\(?:texttt|textit|textbf|emph|textsuperscript|textsc"
                      r"|mathrm|mathbf)\s*\{([^{}]*)\}")
_TEX_REF = re.compile(r"\\(?:ref|label|cite)\s*\{([^{}]*)\}")
_TEX_SUP = re.compile(r"([_^])\{([^{}]*)\}")
_TEX_ESC = re.compile(r"\\([_&#%$])")

# Applied in order, before the command regexes, which anchor on the braces.
_TEX_WORDS = [
    # Before `\,`, which becomes the thousands comma.
    (r"\,\%", " %"), (r"\,/\,", " / "),
    (r"$\times$", "x"), (r"\times", "x"), (r"\,", ","), (r"\;", " "),
    (r"\ ", " "), (r"\nu", "nu"), (r"\Gamma", "Gamma"),
    (r"\varepsilon", "eps"), (r"\sqrt", "sqrt"), (r"\dagger", "dag"),
    (r"\ddagger", "ddag"), (r"\sim", "~"), (r"\cdot", "."),
    (r"\mathrm", ""), (r"~", " "),
    (r"\le", "<="), (r"\ge", ">="), (r"\ast", "*"),
    (r"\max", "max"), (r"\min", "min"), (r"\approx", "~"),
    # `\infty` precedes `\in`, which is a prefix of it.
    (r"\infty", "inf"), (r"\in", " in "), (r"\log", "log"),
    (r"\cosh", "cosh"), (r"\'e", "\u00e9"),
]


def _demath(t):
    """LaTeX -> plain text for the Markdown twin and the README.md facts (write_index);
    the constructs emitted here."""
    t = str(t)
    for a, b in _TEX_WORDS:
        t = t.replace(a, b)
    for _ in range(3):                    # nested one level, e.g. \textit{$x$}
        t = _TEX_CMD.sub(r"\1", t)
    t = _TEX_REF.sub(r"\1", t)
    t = _TEX_SUP.sub(r"\1\2", t)
    t = _TEX_ESC.sub(r"\1", t)
    t = t.replace("$", "").replace("{", "").replace("}", "")
    return " ".join(t.split())


# Shared selection
def laws_present(rows, laws=ACC_LAWS):
    """The laws in `laws` that `rows` carries, in the order of `laws`."""
    have = {r.get("law") for r in rows}
    return [l for l in laws if l in have]


def lib_rows(acc, family, backend):
    """{law: row} for one library backend."""
    return {r["law"]: r for r in acc
            if r.get("family") == family and r.get("backend") == backend
            and scope_of(r) == "call"}


def external_rows(sup, family, scope="call"):
    """{(supplier, law): row}, each supplier at its best assembly."""
    best = best_assembly([r for r in sup if scope_of(r) == scope])
    return {(s, l): r for (s, f, l), r in best.items() if f == family}


def suppliers_present(sup, family, scope="call"):
    """External supplier names (the library excluded), sorted."""
    return sorted({r["supplier"] for r in sup
                   if r.get("family") == family and scope_of(r) == scope
                   and (r.get("supplier") or "-") != "-"})


def machine_agreement(per_machine, fields=("u_max", "u_med", "u_p99", "u_p999",
                                           "n_scored", "nonfinite", "rel_max")):
    """(n_compared, n_differing, [description]) of `fields` over the rows present
    on every machine of `per_machine` ([(label, rows)]).  Fields are compared as the
    recorded strings, for exact equality; that is why the root tables can be built
    from one machine."""
    if len(per_machine) < 2:
        return 0, 0, []
    keyed = []
    for label, rows in per_machine:
        keyed.append((label, {(r.get("supplier", "-"), r.get("family"),
                               r.get("law"), r.get("backend")): r
                              for r in rows}))
    common = set(keyed[0][1])
    for _, k in keyed[1:]:
        common &= set(k)
    n = bad = 0
    detail = {}
    for key in sorted(common):
        for f in fields:
            vals = {lab: k[key].get(f) for lab, k in keyed}
            n += 1
            if len(set(vals.values())) > 1:
                bad += 1
                who = key[0] if key[0] not in ("", "-") else LIB
                detail.setdefault(who, set()).add(f)
    return n, bad, [f"{k} ({', '.join(sorted(v))})"
                    for k, v in sorted(detail.items())]


# Per-law tables: implementations are rows, laws are columns; the heads
# abbreviate `paper_figures.LAW_LABEL`.
LAW_HEAD = {
    "grid": "Grid", "random": "Random", "small_z": r"Small $z$",
    "near_int": r"Near-int $\nu$", "half_int": r"Half-int $\nu$",
    "large_z": r"Large $z$", "large_nu": r"Large $\nu$", "seam": "Seams",
    "size_random": "Size / rand.", "size_half": "Size / half",
    "size_random_large": "Size / rand. $2^{25}$",
    "size_half_large": "Size / half $2^{25}$",
}


def _impl_series(acc, sup, family, backend):
    """[(display name, {law: row})]: the library, then the externals sorted."""
    out = []
    if acc:
        out.append((f"{LIB} ({backend})", lib_rows(acc, family, backend)))
    ext = external_rows(sup, family)
    for s in suppliers_present(sup, family):
        out.append((s, {l: r for (sn, l), r in ext.items() if sn == s}))
    return out


# Comparison rows of the accuracy and cost tables
_VERSION = re.compile(r"-\d+\.[\w.]*(?=-|$)")
_SHORT = {"kaust-adaptive": "kaust-ad.", "std-cyl_bessel_k": "std"}
_SHORT_SUFFIX = (("-adapted", "-ad."),)


def short_impl(name):
    """A supplier name that shares a cell with a number: dotted versions dropped,
    `-adapted` -> `-ad.`, configuration suffixes (`kaust-128`) kept."""
    if name in _SHORT:
        return _SHORT[name]
    s = _VERSION.sub("", name)
    for long_, short in _SHORT_SUFFIX:
        if s.endswith(long_):
            s = s[:-len(long_)] + short
    return s


def comparable(row, key="u_max"):
    """True when `key` is rankable: the row is not `failed` (CATASTROPHIC or any
    non-finite result), so a daggered cell is excluded."""
    return row is not None and not failed(row) and _ok(_f(row, key))


def score_cells(by_law, laws):
    # `finite`: the laws on which the row is comparable.
    b = sum(1 for l in laws if l in by_law and comparable(by_law[l]))
    return [text(f"{b}/{len(laws)}")]


def advantage(better_name, factor):
    """A factor >= 1, labelled with the implementation it favours."""
    t = sig(factor, 3)
    if t[0] is None:
        return DASH
    return Cell(style.Raw(esc(better_name) + "~" + t[0] + r"$\times$"),
                f"{better_name} {t[1]}x", repr(factor))


def comparison_rows(series, lib_by, laws, key):
    """(near, adv, incomplete): the Lowest external and Advantage factor rows for
    `key` (lower is better), without the caller's trailing columns."""
    near = [raw("Lowest external", "Lowest external")]
    adv = [raw("Advantage factor", "Advantage factor")]
    incomplete = 0
    for l in laws:
        others = [(n, b[l]) for n, b in series
                  if not n.startswith(LIB) and l in b and comparable(b[l], key)]
        base = lib_by.get(l)
        bv = _f(base, key) if base is not None else float("nan")
        if not others:
            incomplete += 1
            near.append(DASH)
            adv.append(DASH)
            continue
        name, row = min(others, key=lambda o: _f(o[1], key))
        v = _f(row, key)
        c = ulp(row) if key == "u_max" else num(v)
        sn = short_impl(name)
        near.append(Cell(style.Raw(esc(sn) + "~" + c.tex), sn + " " + c.md,
                         c.csv))
        if not (_ok(bv) and bv > 0 and _ok(v) and v > 0):
            adv.append(DASH)
            continue
        r = v / bv
        adv.append(advantage(sn, 1.0 / r) if r < 1
                   else advantage(LIB_SHORT, r))
    return near, adv, incomplete


def medmax_comparison_rows(series, lib_by, laws):
    """(lowest, adv, incomplete): the closing rows of the median/maximum tables, one
    median/maximum pair per cell.  The median half ranks every external with a
    median, daggered and `Cap` rows included; the maximum half ranks the
    `comparable` rows only, as `comparison_rows` does.  `incomplete` counts the
    laws without a maximum half."""
    lowest = [raw("Lowest external", "Lowest external")]
    adv = [raw("Advantage factor", "Advantage factor")]
    incomplete = 0
    for l in laws:
        halves = []                    # per half: (name, favoured, factor) or None
        for key, rankable in (("u_med", lambda r: _ok(_f(r, "u_med"))),
                              ("u_max", lambda r: comparable(r, "u_max"))):
            others = [(n, b[l]) for n, b in series
                      if not n.startswith(LIB) and l in b and rankable(b[l])]
            if not others:
                halves.append(None)
                continue
            name, row = min(others, key=lambda o: _f(o[1], key))
            v = _f(row, key)
            base = lib_by.get(l)
            bv = _f(base, key) if base is not None else float("nan")
            if _ok(bv) and bv > 0 and _ok(v) and v > 0:
                fav, fac = ((short_impl(name), bv / v) if v < bv
                            else (LIB_SHORT, v / bv))
            else:
                fav, fac = None, float("nan")
            halves.append((name, fav, fac))
        if halves[1] is None:
            incomplete += 1
        if halves == [None, None]:
            lowest.append(DASH)
            adv.append(DASH)
            continue
        # Lowest external: one name when both halves name the same implementation.
        names = [short_impl(h[0]) if h else "--" for h in halves]
        full = "/".join(h[0] if h else "" for h in halves)
        if names[0] == names[1]:
            lowest.append(Cell(style.Raw(esc(names[0])), names[0], full))
        else:
            # two names stacked, so that this row does not set the column width
            lowest.append(Cell(style.Raw(_stack(esc(names[0]) + "/", esc(names[1]))),
                               names[0] + "/" + names[1], full))
        # Advantage factor: the favoured implementation once when both halves agree.
        parts = [(h[1], sig(h[2], 3)) if (h and h[1]) else (None, (None, None))
                 for h in halves]
        csv_ = "/".join(repr(h[2]) if (h and h[1]) else "" for h in halves)
        if parts[0][0] is None and parts[1][0] is None:
            adv.append(DASH)
            continue
        if parts[0][0] == parts[1][0]:
            fav = parts[0][0]
            adv.append(Cell(style.Raw(esc(fav) + "~" + parts[0][1][0] + "/"
                                      + parts[1][1][0]),
                            f"{fav} {parts[0][1][1]}/{parts[1][1][1]}", csv_))
            continue
        tex = [esc(f) + "~" + t[0] if f else "--" for f, t in parts]
        md = [f"{f} {t[1]}" if f else "--" for f, t in parts]
        adv.append(Cell(style.Raw(_stack(tex[0] + "/", tex[1])), "/".join(md), csv_))
    return lowest, adv, incomplete


def _stack(top, bottom):
    r"""Two lines in one right-aligned cell, the first on the row's baseline."""
    return r"\begin{tabular}[t]{@{}r@{}}" + top + r"\\" + bottom + r"\end{tabular}"


def medmax_comparison_note(dashed, rows):
    """The legend of the closing rows of `medmax_comparison_rows`; `rows` are the
    cells shown, so only the markers `marker_note` defines are named; `dashed`
    whether a cell of the closing rows shows `--`."""
    marks = ([r"$\dagger$"] if any(_i(r, "nonfinite") > 0 for r in rows) else []) \
        + ([r"\textit{Cap}"] if any(_ok(_f(r, "u_max")) and _f(r, "u_max")
                                    >= CATASTROPHIC for r in rows) else [])
    return (r"\emph{Lowest external}: the external implementation with the "
            r"lowest median and the one with the lowest maximum "
            r"(median/maximum; one name where both coincide). The median is "
            r"ranked over all external implementations"
            + (", rows marked " + " or ".join(marks) + " included" if marks
               else "")
            + r"; the maximum only over those with a "
            r"finite result at every point and a maximum below the $10^{6}$ ULP "
            r"cap. \emph{Advantage factor}: for the median and the maximum, the "
            r"ratio of the larger to the smaller of the lowest external value and "
            r"this library's, labelled with the implementation it favours "
            r"(\texttt{" + LIB_SHORT + r"}: this library)"
            + (r"; --: no such implementation or, in the Advantage factor row, "
               r"no ratio because a value is missing or zero. "
               if dashed else ". "))


# Accuracy: maximum ULP by law and implementation
def tab_accuracy(acc, sup, outdir, runs_note, family="z_nu", backend=None,
                 agreement=None, arch=""):
    """Maximum ULP per law and implementation (externals at their best assembly),
    a `finite` column and the two comparison rows."""
    stem = f"tab-accuracy-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows -- the accuracy stage did not run")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    if not laws:
        return _skip(stem, "no rows on any of the paper's accuracy laws")
    series = _impl_series(acc, sup, family, backend)
    if len(series) < 2:
        return _skip(stem, "only the library is present -- a comparison table "
                           "with one implementation compares the library with "
                           "itself; build the externals to get this table")

    head = (["implementation"] + [LAW_HEAD.get(l, l) for l in laws]
            + ["finite"])
    t = Table(stem,
              _caption("Maximum ULP error over all scored points for "
                       + _sets(family) + ", by implementation.",
                       BEST_ASSEMBLY,
                       _obtained(arch)),
              f"tab:accuracy-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws) + "rr")
    for name, by_law in series:
        cells = [text(name)]
        for l in laws:
            cells.append(ulp(by_law[l]) if l in by_law else DASH)
        cells += score_cells(by_law, laws)
        t.add(cells)
        if name.startswith(LIB):
            t.rule_here()

    lib_by = dict(series[0][1]) if series[0][0].startswith(LIB) else {}
    incomplete, dashed = 0, False
    if lib_by and len(series) > 1:
        near, adv, incomplete = comparison_rows(series, lib_by, laws, "u_max")
        dashed = dashed_cells(near, adv)
        # `finite` is empty on the comparison rows.
        t.rules.add(len(t.rows) - 1)
        t.add(near + [DASH])
        t.add(adv + [DASH])

    nf = sum(1 for _, b in series for r in b.values() if _i(r, "nonfinite") > 0)
    shown = [b[l] for _, b in series for l in laws if l in b]

    t.note = _note(marker_note(shown) + FINITE_NOTE
                   + (comparison_note("maximum", dashed)
                      if lib_by and len(series) > 1 else ""))
    t.provenance(
        runs_note,
        r"Each cell is the maximum ULP error over every scored point of the "
        r"dataset. `\textit{Cap}' marks a maximum at or above the $10^{6}$~ULP cap; the "
        r"measured value is in the accompanying \texttt{.csv}.",
        (r"A dagger marks a maximum taken over a \emph{reduced} population "
         r"because the implementation returned a non-finite result somewhere "
         r"on that dataset; those counts are in \texttt{tab-exclusions-%s}."
         % esc(family)) if nf else "",
        r"\emph{finite} is computed identically for every row, and a daggered "
        r"cell does not count.",
        r"The last two rows compare with the most precise external that "
        r"returned a finite number at every point of that dataset and a maximum "
        r"below $10^{6}$~ULP; the advantage factor is at least one.",
        (f"`--' marks the {incomplete} dataset(s) on which no external returned a "
         r"finite number at every point and a maximum below "
         r"$10^{6}$~ULP.") if incomplete else "",
        _relerr_phrase(series, laws),
        _agreement_phrase(agreement),
        _machine(arch))
    return t.write(outdir)


def _relerr_phrase(series, laws):
    """README.md sentence on why relative error is not tabulated, with the
    library's largest relative maximum and the others' range on that law."""
    if not series:
        return ""
    lib_by = series[0][1] if series[0][0].startswith(LIB) else {}
    best = None
    for l in laws:
        v = _f(lib_by[l], "rel_max") if l in lib_by else float("nan")
        if _ok(v) and (best is None or v > best[1]):
            best = (l, v)
    if best is None or best[1] <= 0:
        return (r"Relative error is not tabulated, because it is undefined at a "
                r"zero reference and the corpus contains references at and near "
                r"zero; it is in the \texttt{.csv}. ")
    law, lv = best
    others = [_f(b[law], "rel_max") for n, b in series[1:] if law in b]
    others = [v for v in others if _ok(v) and v > 0]
    out = (r"Relative error is not tabulated, because it is undefined at a zero "
           r"reference and the corpus contains references at and near zero. On "
           r"\texttt{" + esc(law) + r"} the maximum relative error of this "
           r"library is " + (sig(lv, 2)[0] or "--")
           + r", at a reference near the bottom of the double range")
    if others:
        out += (r", and the other implementations on the same dataset spread from "
                + (sig(min(others), 2)[0] or "--") + r" to "
                + (sig(max(others), 2)[0] or "--"))
    out += (r". The relative error is in the \texttt{.csv}. ")
    return out


def _agreement_phrase(agreement):
    """The result of `machine_agreement` as a README.md sentence, or ""."""
    if not agreement:
        return ""
    n, bad, who = agreement
    if not n:
        return ""
    if not bad:
        return (f"All {n:,} accuracy fields agree between the two "
                f"machines. ")
    return (f"{n - bad:,} of {n:,} accuracy fields agree between the two "
            f"machines; the differing rows are "
            f"{', '.join(esc(w) for w in who)}. ")


def esc(t):
    """Escape a data-derived string: notes and captions are emitted raw
    (`note_raw=True`), so every CSV, provenance or run-id string passes here."""
    return style.tex_escape(t)


# (key, note): `Table.write` appends the note to every table with a cell that
# contains the key.  references/kaust_logbesselk/PROVENANCE.md requires the
# kaust-adaptive disclosure wherever that row is quoted.
ROW_NOTES = [(
    "kaust-adaptive",
    r"\texttt{kaust-adaptive}: an adjusted configuration of the KAUST code, "
    r"more accurate than the code as released (the adaptive integration range "
    r"of its paper in place of $t \in [0,9]$, 128 panels, the decade factor of "
    r"the small-$x$ recurrence applied, $\log \cosh$ without the intermediate "
    r"$\cosh$)."
)]


def _row_notes(rows):
    """Concatenate the notes whose key appears in any cell of `rows`."""
    out = []
    for key, text in ROW_NOTES:
        if any(key in str(c.md) for r in rows for c in r):
            out.append(text)
    return " ".join(out)


class _Note(str):
    """A LaTeX table note with its Markdown form in `.md`."""
    md = None


def _note(tex, md=None):
    """A raw LaTeX table note; `md` replaces `_demath(tex)` in the Markdown twin."""
    n = _Note(tex)
    n.md = md
    return n


# Median ULP by law and implementation
def tab_median(acc, sup, outdir, runs_note, family="z_nu", backend=None,
               arch=""):
    """Median ULP per law, rows and order of `tab_accuracy`; `tab_medmax` combines
    the two, and a document uses one form or the other."""
    stem = f"tab-median-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows -- the accuracy stage did not run")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    if not laws:
        return _skip(stem, "no rows on any of the paper's accuracy laws")
    series = _impl_series(acc, sup, family, backend)
    if len(series) < 2:
        return _skip(stem, "only the library is present")

    head = ["implementation"] + [LAW_HEAD.get(l, l) for l in laws]
    t = Table(stem,
              _caption("Median ULP error over all scored points for "
                       + _sets(family) + ", by implementation.",
                       BEST_ASSEMBLY,
                       _obtained(arch)),
              f"tab:median-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws))
    for name, by_law in series:
        cells = [text(name)]
        for l in laws:
            cells.append(num(_f(by_law[l], "u_med")) if l in by_law else DASH)
        t.add(cells)
        if name.startswith(LIB):
            t.rule_here()

    # The library's largest max/median ratio, for the README.md.
    lib_by = dict(series[0][1]) if series[0][0].startswith(LIB) else {}
    worst = None
    for l in laws:
        if l not in lib_by:
            continue
        m, x = _f(lib_by[l], "u_med"), _f(lib_by[l], "u_max")
        if _ok(m) and _ok(x) and m > 0 and (worst is None or x / m > worst[1]):
            worst = (l, x / m)

    t.provenance(
        runs_note,
        r"Each cell is the median ULP error over every scored point, the "
        r"same population as the maximum in \texttt{tab-accuracy-%s}, with the "
        r"same rows in the same order. The `\textit{Cap}' and dagger markers "
        r"apply to maxima and are not used here." % esc(family),
        (r"The largest ratio of maximum to median for this library is "
         r"$%.0f\times$, on \texttt{%s}." % (worst[1], esc(worst[0]))
         if worst else ""),
        r"Quantiles are nearest-rank on the exact sorted sample, so every "
        r"value here occurred.",
        _machine(arch))
    return t.write(outdir)


# Median and maximum in one cell
def tab_medmax(acc, sup, outdir, runs_note, family="z_nu", backend=None,
               arch=""):
    """`median / maximum` per cell: `tab_accuracy` and `tab_median` in one table;
    the markers apply to the maximum only."""
    stem = f"tab-medmax-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows -- the accuracy stage did not run")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    if not laws:
        return _skip(stem, "no rows on any of the paper's accuracy laws")
    series = _impl_series(acc, sup, family, backend)
    if len(series) < 2:
        return _skip(stem, "only the library is present")

    head = (["implementation"] + [LAW_HEAD.get(l, l) for l in laws]
            + ["finite"])
    t = Table(stem,
              _caption(r"Median and maximum ULP error (median\,/\,maximum) "
                       "over all scored points for " + _sets(family)
                       + ", by implementation.",
                       BEST_ASSEMBLY,
                       _obtained(arch)),
              f"tab:medmax-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws) + "rr")
    for name, by_law in series:
        cells = [text(name)]
        for l in laws:
            if l not in by_law:
                cells.append(DASH)
                continue
            m, x = num(_f(by_law[l], "u_med")), ulp(by_law[l])
            cells.append(Cell(style.Raw(m.tex + r"\,/\,"  + x.tex),
                              m.md + " / " + x.md, m.csv + "/" + x.csv))
        cells += score_cells(by_law, laws)
        t.add(cells)
        if name.startswith(LIB):
            t.rule_here()

    lib_by = dict(series[0][1]) if series[0][0].startswith(LIB) else {}
    incomplete, dashed = 0, False
    if lib_by and len(series) > 1:
        near, adv, incomplete = medmax_comparison_rows(series, lib_by, laws)
        dashed = dashed_cells(near, adv)
        t.rules.add(len(t.rows) - 1)
        t.add(near + [DASH])
        t.add(adv + [DASH])

    shown = [b[l] for _, b in series for l in laws if l in b]
    t.note = _note(marker_note(shown) + FINITE_NOTE
                   + (medmax_comparison_note(dashed, shown)
                      if lib_by and len(series) > 1 else ""))
    t.provenance(
        runs_note,
        r"Each cell is the median\,/\,maximum ULP error over the scored "
        r"points of the dataset. The maximum and its markers are those of "
        r"\texttt{tab-accuracy-%s}, which this table combines with "
        r"\texttt{tab-median-%s}; the `\textit{Cap}' and dagger markers apply "
        r"to the maximum only. The two closing rows compare the median and the "
        r"maximum separately; their maximum half is the comparison of "
        r"\texttt{tab-accuracy-%s}." % (esc(family), esc(family), esc(family)),
        r"Quantiles are nearest-rank on the sorted sample. The \texttt{.csv} "
        r"writes the cell as \texttt{median/maximum} at full precision; the "
        r"two quantities are also in the \texttt{.csv} of each combined "
        r"table, one per field.",
        (f"`--' marks the {incomplete} dataset(s) on which no external returned a "
         r"finite number at every point and a maximum below "
         r"$10^{6}$~ULP.") if incomplete else "",
        _machine(arch))
    return t.write(outdir)


# Library ULP distribution per law and location of the maximum
def tab_distribution(acc, outdir, runs_note, family="z_nu", backend=None,
                     arch=""):
    """The library's ULP quantiles per law and the point (`worst_arg`, `worst_nu`)
    of the ULP maximum."""
    stem = f"tab-distribution-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows -- the accuracy stage did not run")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    by_law = lib_rows(acc, family, backend)
    laws = [l for l in laws if l in by_law]
    if not laws:
        return _skip(stem, f"no library rows on backend {backend}")

    argname = "z" if family == "z_nu" else "h"
    head = ["dataset", r"$q_1$", "median", r"$q_3$", r"$p_{99}$",
            r"$p_{99.9}$", "max", r"max $/$ med",
            rf"${argname}^{{\ast}}$", r"$\nu^{\ast}$"]
    t = Table(stem,
              _caption(r"ULP error quantiles of the \texttt{" + esc(backend)
                       + r"} backend for " + _sets(family) + r", and the point "
                       r"$(" + argname + r"^{\ast}, \nu^{\ast})$ at which the "
                       r"maximum is attained.", _obtained(arch)),
              f"tab:distribution-{family.replace('_', '-')}",
              head, align=wrapcol("2.2cm") + "r" * 9)
    for l in laws:
        r = by_law[l]
        med, mx = _f(r, "u_med"), _f(r, "u_max")
        # Maximum through `ulp`, other quantiles through `qcell`, as elsewhere.
        t.add([raw(LAW_HEAD.get(l, l), l),
               qcell(r, "u_q1"), qcell(r, "u_med"), qcell(r, "u_q3"),
               qcell(r, "u_p99"), qcell(r, "u_p999"), ulp(r),
               ratio(mx / med) if (_ok(med) and _ok(mx) and med > 0) else DASH,
               num(_f(r, "worst_arg"), 4), num(_f(r, "worst_nu"), 4)])

    t.note = _note(
        r"$q_1$, $q_3$: first and third quartiles; $p_{99}$, $p_{99.9}$: 99th "
        r"and 99.9th percentiles; max $/$ med: ratio of the maximum to the "
        r"median. "
        + marker_note([by_law[l] for l in laws],
                      keys=("u_q1", "u_med", "u_q3", "u_p99", "u_p999", "u_max")))
    t.provenance(
        runs_note,
        r"Quantiles of the ULP error over the scored points of each dataset "
        r"(nearest-rank), and the $(" + argname + r", \nu)$ at which the "
        r"maximum is attained; the maximum can be reproduced by evaluating "
        r"that one point. A large \emph{max / med} means that the maximum is "
        r"attained in a small part of the dataset.",
        _machine(arch))
    return t.write(outdir)


# ULP distribution of every implementation, by law
# The quantiles are the harness's own (benchmark/common/bench_metrics.h, nearest
# rank), read from the CSV row; the share below 1 ulp is summed from the detail
# histogram (runs.detail_for), refused when its total differs from `n_scored`.
QUANTILE_COLS = [("u_min", "min"), ("u_q1", r"$q_1$"), ("u_med", "median"),
                 ("u_q3", r"$q_3$"), ("u_p99", r"$p_{99}$"),
                 ("u_p999", r"$p_{99.9}$"), ("u_max", "max"),
                 ("u_mean", "mean")]

# Rows per float, law headings included; a taller table is split between laws
# (fit_to_column does not split).  This many rows, with caption and note, fit the
# text height at the smallest upright size.
QUANTILE_ROWS_PER_FLOAT = 44


class GroupedTable(Table):
    """A `Table` whose first column (group key in its CSV form) names groups of
    rows: a spanning heading row per group in the `.tex`; in the `.md` filled on a
    group's first row, in the `.csv` on every row.  `breaks` starts new floats
    (labels `-2`, `-3`, ...), all set at one size; `cont_captions` holds one caption
    per break and grows with `breaks` (`_write_tex` reads `cont_captions[k - 1]`)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.breaks = []
        self.cont_captions = []

    def _write_tex(self, path):
        size, colsep, env = SIZE_LADDER[min(self.level, len(SIZE_LADDER) - 1)]
        need = "booktabs, array" + (", rotating" if env != "table" else "")
        ncol = len(self.header) - 1
        cuts = [0] + list(self.breaks) + [len(self.rows)]
        parts = []
        for k in range(len(cuts) - 1):
            a, b = cuts[k], cuts[k + 1]
            body, rules, prev = [], set(), None
            for i in range(a, b):
                r = self.rows[i]
                if r[0].csv != prev:
                    # A one-element row is emitted as that cell alone.
                    body.append([style.Raw(r"\multicolumn{%d}{l}{\textit{%s}}"
                                           % (ncol, r[0].tex))])
                    prev = r[0].csv
                body.append([c.tex for c in r[1:]])
                if i in self.rules:
                    rules.add(len(body) - 1)
            if k == 0:
                cap, lab, note = self.caption, self.label, self.note
            else:
                cap = self.cont_captions[k - 1]
                lab = f"{self.label}-{k + 1}"
                note = r"Continued from Table~\ref{%s}." % self.label
            tmp = path + ".part"
            style.booktabs(
                tmp, cap, lab, self.header[1:], body,
                align=self.align, note=note, note_raw=True,
                rules_after=rules, preamble_note=need, size=size,
                colsep=colsep, env=env,
                placement=None if env != "table" else "t")
            parts.append(open(tmp).read())
            os.remove(tmp)
        open(path, "w").write("\n".join(parts))


def qcell(row, key):
    """A ULP statistic other than the maximum: the value, or `Cap` at or above
    CATASTROPHIC; the row's dagger is carried by its maximum."""
    v = _f(row, key)
    if not _ok(v):
        return DASH
    if v >= CATASTROPHIC:
        return Cell(style.Raw(r"\textit{Cap}"), "Cap", repr(v))
    return num(v)


def pct_below(share):
    """A share in [0, 1] as a percentage, one decimal; 0 and 100 only when exact."""
    if not _ok(share):
        return DASH
    p = 100.0 * share
    if share == 1.0:
        s = "100"
    elif share == 0.0:
        s = "0"
    else:
        s = f"{p:.1f}"
        if s == "100.0":
            s = ">99.9"
        elif s == "0.0":
            s = "<0.1"
    tex = s.replace(">", "$>$").replace("<", "$<$")
    return Cell(style.Raw(tex), s, repr(p))


QUANTILE_FIELDS = tuple(k for k, _h in QUANTILE_COLS) + ("n_scored", "nonfinite")


def quantile_agreement(per_machine, family):
    """`machine_agreement` over the rows and fields `tab_quantiles` shows;
    `per_machine` is [(label, acc, sup)]."""
    shown = []
    for lab, acc, sup in per_machine:
        backend = lib_backend_for(acc, family=family)
        rows = [r for _n, b in _impl_series(acc, sup, family, backend)
                for l, r in b.items() if l in ACC_LAWS]
        shown.append((lab, rows))
    return machine_agreement(shown, fields=QUANTILE_FIELDS)


def _quantile_table(stem, caption, label, series, laws, rowhead, share=True):
    """(table, missing): the quantile table, one block per law; `missing` lists
    (name, law, reason) with no share below 1 ulp (printed `--`)."""
    head = (["dataset", rowhead] + [h for _k, h in QUANTILE_COLS]
            + ([r"$<1$ ULP (\%)"] if share else []))
    # `align` omits the law column, a heading row in the .tex.
    t = GroupedTable(stem, caption, label, head,
                     align=wrapcol("2.6cm") + "r" * (len(QUANTILE_COLS)
                                                    + (1 if share else 0)))
    missing = []
    starts = []                      # (first row index, law) of each block
    for l in laws:
        block = [(n, b[l]) for n, b in series if l in b]
        if not block:
            continue
        starts.append((len(t.rows), l))
        lab = LAW_LABEL.get(l, l)
        for i, (name, r) in enumerate(block):
            cells = [Cell(style.Raw(lab), _demath(lab) if i == 0 else "", l),
                     text(name)]
            for k, _h in QUANTILE_COLS:
                cells.append(ulp(r, k) if k == "u_max" else qcell(r, k))
            if not share:
                t.add(cells)
                continue
            e, why = runs.detail_for(r)
            if e is None:
                missing.append((name, l, why))
                print(f"    !! {stem}: {name} on {l}: no share below 1 ulp -- "
                      f"{why}")
                cells.append(DASH)
            else:
                cells.append(pct_below(runs.hist_share_below(e, 0.0)))
            t.add(cells)
        t.rule_here()
    # Split between laws within the row budget (one heading row per law).
    n = len(t.rows) + len(starts)
    if n > QUANTILE_ROWS_PER_FLOAT and len(starts) > 1:
        k = -(-n // QUANTILE_ROWS_PER_FLOAT)            # floats needed
        per = -(-len(starts) // k)                       # laws per float
        for j in range(per, len(starts), per):
            t.breaks.append(starts[j][0])
            t.cont_captions.append(
                caption.split(". ")[0].rstrip(".")
                + " (continued): " + ", ".join(
                    LAW_LABEL.get(s[1], s[1])
                    for s in starts[j:j + per]) + ".")
    return t, missing


def _quantile_note(series, laws, missing, share=True):
    """The note of the quantile tables; `share=False` without the share column."""
    return (
        r"$q_1$, $q_3$: first and third quartiles; $p_{99}$, $p_{99.9}$: 99th "
        r"and 99.9th percentiles. "
        + ((r"$<1$ ULP: share of the scored points with a ULP error below 1"
            + (r"; --: not available. " if missing else ". ")) if share else "")
        + marker_note([b[l] for _, b in series for l in laws if l in b],
                      keys=[k for k, _h in QUANTILE_COLS]))


def _quantile_facts(missing, nf, share=True):
    """The README facts of the quantile tables (see `_quantile_note`)."""
    return [
        r"Each row is the distribution of the ULP error over the scored "
        r"points of one dataset: the minimum, the quartiles ($q_1$, median, $q_3$), "
        r"the 99th and 99.9th percentiles, the maximum and the mean. The "
        r"quantiles are the benchmark's own, nearest-rank on the exact sorted "
        r"sample, so every value except the mean occurred.",
        r"Every statistic of a row with a dagger is taken over the same finite "
        r"points." if nf else "",
        (r"The last column is the share of the scored points whose ULP "
         r"error is below 1, computed from the histogram of the ULP "
         r"error the benchmark records beside each row (log-spaced bins "
         r"with an edge at 1~ULP), not from the sorted sample; an error "
         r"within about $10^{-15}$ below 1~ULP falls in the bin above the "
         r"edge and is counted as not below.") if share else "",
        (r"`--' in that column marks %d row(s) whose histogram was absent or "
         r"did not cover exactly the scored points: %s."
         % (len(missing), "; ".join(r"%s on \texttt{%s} (%s)"
                                    % (esc(n), esc(l), esc(w))
                                    for n, l, w in missing)))
        if missing and share else ""]


def tab_quantiles(acc, sup, outdir, runs_note, family="z_nu", backend=None,
                  agreement=None, arch=""):
    """The ULP distribution of every `tab_accuracy` row, law by law, with the
    share below 1 ulp."""
    stem = f"tab-quantiles-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows -- the accuracy stage did not run")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    if not laws:
        return _skip(stem, "no rows on any of the paper's accuracy laws")
    series = _impl_series(acc, sup, family, backend)
    if len(series) < 2:
        return _skip(stem, "only the library is present -- the table compares "
                           "implementations; build the externals to get it")
    t, missing = _quantile_table(
        stem,
        _caption("Distribution of the ULP error over all scored points for "
                 + _sets(family) + ", by implementation.",
                 BEST_ASSEMBLY,
                 _obtained(arch)),
        f"tab:quantiles-{family.replace('_', '-')}", series, laws,
        "implementation")
    nf = sum(1 for _, b in series for l, r in b.items()
             if l in laws and _i(r, "nonfinite") > 0)
    t.note = _note(_quantile_note(series, laws, missing))
    t.provenance(runs_note, *_quantile_facts(missing, nf),
                 _agreement_phrase(agreement), _machine(arch))
    return t.write(outdir)


def tab_gpu_quantiles(gsup, outdir, runs_note, arch, family="z_nu",
                      scope="kernel"):
    """`tab_quantiles` for the device kernels of `tab_gpu` (kernel scope)."""
    stem = f"tab-gpu-quantiles-{family}"
    if not gsup:
        return _skip(stem, "no device supplier rows -- the GPU stage did not "
                           "run, or ran on a host with no CUDA device")
    rows = [r for r in gsup if scope_of(r) == scope and r.get("family") == family]
    if not rows:
        return _skip(stem, f"the device rows carry no {scope}-scope "
                           f"measurement for {family}")
    laws = laws_present(rows)
    best = best_assembly(rows)
    names = sorted({r["supplier"] for r in rows})
    names.sort(key=lambda s: (0 if s.startswith(LIB) else 1, s))
    if not laws or len(names) < 2:
        return _skip(stem, "needs the device library and at least one external "
                           "kernel on a shared law")
    series = [(s, {l: r for (sn, f, l), r in best.items()
                   if sn == s and f == family}) for s in names]
    t, missing = _quantile_table(
        stem,
        _caption("Distribution of the ULP error over all scored points of the "
                 "GPU kernels for " + _sets(family) + ".",
                 BEST_ASSEMBLY,
                 _obtained(arch, gpu=True)),
        f"tab:gpu-quantiles-{family.replace('_', '-')}", series, laws, "kernel")
    nf = sum(1 for _, b in series for r in b.values() if _i(r, "nonfinite") > 0)
    t.note = _note(_quantile_note(series, laws, missing))
    t.provenance(runs_note, *_quantile_facts(missing, nf), _machine(arch))
    return t.write(outdir)


# Cost by law and implementation
def tab_cost(acc, sup, outdir, runs_note, arch, family="z_nu", backend=None):
    """`ns_med` (as in fig_cost_accuracy_plane) per law and implementation, in the
    layout of `tab_accuracy`."""
    stem = f"tab-cost-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows -- costs are recorded on them")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    series = _impl_series(acc, sup, family, backend)
    if not laws or len(series) < 2:
        return _skip(stem, "needs the library and at least one external on a "
                           "shared law")
    head = (["implementation"] + [LAW_HEAD.get(l, l) for l in laws]
            + [r"worst spread"])
    t = Table(stem,
              _caption("Single-threaded runtime cost in nanoseconds per element "
                       "for " + _sets(family) + ", by implementation.",
                       MEDIAN_COST, BEST_ASSEMBLY,
                       _obtained(arch)),
              f"tab:cost-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws) + "r")
    lib_by = None
    worst_spread = {}
    for name, by_law in series:
        cells = [text(name)]
        for l in laws:
            cells.append(num(_f(by_law[l], "ns_med")) if l in by_law else DASH)
        # Worst spread: max/min over the repetitions, worst over the laws shown.
        sp = [_f(by_law[l], "ns_spread") for l in laws if l in by_law]
        sp = [v for v in sp if _ok(v) and v > 0]
        worst_spread[name] = max(sp) if sp else float("nan")
        cells.append(ratio(worst_spread[name]) if sp else DASH)
        t.add(cells)
        if name.startswith(LIB):
            lib_by = by_law
            t.rule_here()
    incomplete, dashed = 0, False
    if lib_by and len(series) > 1:
        near, adv, incomplete = comparison_rows(series, lib_by, laws, "ns_med")
        dashed = dashed_cells(near, adv)
        t.rules.add(len(t.rows) - 1)
        t.add(near + [DASH])
        t.add(adv + [DASH])
    t.note = _note(
        r"\emph{worst spread}: largest ratio of the slowest to the fastest "
        r"repetition over the datasets of the row. "
        + (comparison_note("cost", dashed)
           if lib_by and len(series) > 1 else ""))
    t.provenance(
        runs_note,
        r"Each cell is the median over repetitions of a single-threaded call "
        r"on the whole array, including the bucketing this library performs "
        r"inside the call. Same rows and same order as the accuracy table.",
        r"The last two rows compare with the fastest external that returned a "
        r"finite number at every point of that dataset and a maximum below "
        r"$10^{6}$~ULP; the advantage factor is at least one.",
        (f"`--' marks the {incomplete} dataset(s) on which no external returned a "
         r"finite number at every point and a maximum below "
         r"$10^{6}$~ULP.") if incomplete else "",
        r"The \emph{worst spread} is largest on the cheapest cells, where one "
        r"interruption is a large multiple of the measured time.",
        _machine(arch))
    return t.write(outdir)


# Exclusions: non-finite results per implementation
def tab_exclusions(acc, sup, outdir, runs_note, family="z_nu", backend=None,
                   arch=""):
    """Non-finite counts per law; a non-zero cell means a daggered maximum in
    `tab_accuracy`."""
    stem = f"tab-exclusions-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows")
    backend = backend or lib_backend_for(acc, family=family)
    laws = laws_present(acc)
    series = _impl_series(acc, sup, family, backend)
    if not laws or len(series) < 2:
        return _skip(stem, "needs the library and at least one external")
    # Not written when every count is zero.
    total = sum(_i(r, "nonfinite") for _, b in series for r in b.values())
    if total == 0:
        return _skip(stem, "every implementation returned a finite result at "
                           "every point; no table of zeros is written")
    head = ["implementation"] + [LAW_HEAD.get(l, l) for l in laws] + ["points"]
    t = Table(stem,
              _caption("Points at which an implementation returned no finite "
                       "number, for " + _sets(family) + ".", _obtained(arch)),
              f"tab:exclusions-{family.replace('_', '-')}",
              head, align=wrapcol("2.3cm") + "r" * (len(laws) + 1))
    for name, by_law in series:
        cells = [text(name)]
        for l in laws:
            cells.append(count(_i(by_law[l], "nonfinite")) if l in by_law
                         else DASH)
        ns = {_i(r, "n") for r in by_law.values() if _i(r, "n")}
        cells.append(count(max(ns)) if len(ns) == 1 else
                     (raw("varies", "varies") if ns else DASH))
        t.add(cells)
        if name.startswith(LIB):
            t.rule_here()
    clean = sum(1 for _, b in series for r in b.values()
                if _i(r, "nonfinite") == 0)
    tot = sum(len(b) for _, b in series)
    varies = any(r[-1].md == "varies" for r in t.rows)
    t.note = _note(
        r"\emph{points}: size of the datasets"
        + (r"; \emph{varies}: the datasets differ in size. " if varies
           else ". "))
    t.provenance(
        runs_note,
        r"Each cell is a count of points, not a rate. A zero row is an "
        r"implementation that returned a finite double at every point of every "
        r"dataset; for a non-zero cell the maximum in \texttt{tab-accuracy-%s} is "
        r"taken over fewer points and carries a dagger there, also where it "
        r"reads `\textit{Cap}'." % esc(family),
        r"%d of %d measured cells are zero." % (clean, tot),
        _machine(arch))
    return t.write(outdir)


# Assembly selected for each external
def tab_assembly(sup, outdir, runs_note, family="z_nu", scope="call",
                 stem_prefix="tab-assembly", lib_supplier=None, arch=""):
    """The assembly `best_assembly` selects per external and law (`direct`,
    `logtrick`, and `scaled` where a scaled $K$ is published)."""
    stem = f"{stem_prefix}-{family}"
    if not sup:
        return _skip(stem, "no external supplier rows -- nothing to choose "
                           "between")
    # The device variant drops the library row, which has a single assembly.
    rows = [r for r in sup if scope_of(r) == scope and r.get("family") == family]
    if lib_supplier:
        rows = [r for r in rows if r.get("supplier") != lib_supplier]
    laws = laws_present(rows)
    names = suppliers_present(rows, family, scope=scope)
    if not laws or not names:
        return _skip(stem, f"no external carries a {family} accuracy row")
    best = external_rows(rows, family, scope=scope)
    # Only suppliers with more than one assembly are listed.
    multi = [s for s in names
             if len({r.get("backend") for r in rows if r["supplier"] == s}) > 1]
    if not multi:
        return _skip(stem, "every external was measured in a single assembly, "
                           "so there is no choice to record")
    head = ["external"] + [LAW_HEAD.get(l, l) for l in laws]
    t = Table(stem,
              _caption("Assembly selected for each external implementation "
                       "across the %s %s datasets."
                       % (pf.CONV.get(family, esc(family)),
                          "GPU" if scope == "kernel" else "CPU"),
                       "The assembly is selected by the caller rather than by "
                       "the external implementation; each implementation is "
                       "evaluated in its best assembly, " + BEST_RULE + ".",
                       _obtained(arch, gpu=scope == "kernel")),
              f"tab:{stem_prefix.replace('tab-', '')}-{family.replace('_', '-')}",
              head, align=wrapcol("2.3cm") + "l" * len(laws))
    for sname in multi:
        cells = [text(sname)]
        for l in laws:
            r = best.get((sname, l))
            cells.append(text(pf.asm_label(r.get("backend"))) if r else DASH)
        t.add(cells)
    scaled = any(r.get("backend") == "scaled" for r in rows)
    t.note = _note(
        r"\texttt{direct}: $2^{1-\nu}\Gamma(\nu)^{-1}z^{\nu}K_{\nu}(z)$ as "
        r"written; \texttt{exp-log}: the whole prefactor through a logarithm"
        + (r"; \texttt{scaled}: from a published $K_{\nu}(z)e^{z}$. " if scaled
           else ". "))
    t.provenance(
        runs_note,
        r"No supplier here publishes a scaled $K_{\nu}(z)e^{z}$, so the "
        r"choice is two-way." if not scaled else "",
        r"The selection rule is the one the figures use: a row that returns "
        r"a finite number at every point is preferred to one that does not, "
        r"then the larger scored population, then the smaller worst case.")
    return t.write(outdir)


# Attribution across backends
def tab_attribution(acc, outdir, runs_note, arch, family="z_nu"):
    """Maximum ULP and cost per law for each library backend, scalar to widest."""
    stem = f"tab-attribution-{family}"
    if not acc:
        return _skip(stem, "no accuracy rows")
    backends = [b for b in BACKEND_ORDER
                if any(r.get("backend") == b and r.get("family") == family
                       for r in acc)]
    if len(backends) < 2:
        return _skip(stem, f"{len(backends)} library backend(s) built "
                           f"({', '.join(backends) or 'none'}) --- the "
                           f"attribution chain needs at least two to be a "
                           f"difference")
    laws = laws_present(acc)
    if not laws:
        return _skip(stem, "no rows on the paper's accuracy laws")
    by = {(r["backend"], r["law"]): r for r in acc
          if r.get("family") == family and scope_of(r) == "call"}
    head = ["dataset"] + [esc(b) for b in backends] * 2
    n = len(backends)
    _count = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))
    t = Table(stem,
              _caption("Maximum ULP error and runtime cost per element of the "
                       "library backends for " + _sets(family) + ".",
                       MEDIAN_COST, _obtained(arch)),
              f"tab:attribution-{family.replace('_', '-')}",
              head, align="l" + "r" * (2 * n),
              header_groups=[(2, 1 + n, "max ULP error"),
                             (2 + n, 1 + 2 * n, "ns / element")])
    for l in laws:
        cells = [raw(LAW_HEAD.get(l, l), _demath(LAW_HEAD.get(l, l)))]
        for b in backends:
            cells.append(ulp(by[(b, l)]) if (b, l) in by else DASH)
        for b in backends:
            cells.append(num(_f(by[(b, l)], "ns_med")) if (b, l) in by else DASH)
        t.add(cells)
    t.note = _note(marker_note(
        [by[(b, l)] for l in laws for b in backends if (b, l) in by]))
    t.provenance(
        runs_note,
        r"The %s objects of the library backends are built from one "
        r"source." % _count,
        r"\texttt{scalar} is the direct formula without the double-double "
        r"arithmetic and argument corrections, so the difference between it "
        r"and \texttt{matched} is the effect of those; \texttt{matched} is the "
        r"vector algorithm compiled at width one, so the difference between it "
        r"and the vector backends is the effect of the vector units. On a "
        r"machine with neither AVX2 nor AVX-512, \texttt{matched} is the "
        r"backend the library selects.",
        _machine(arch))
    return t.write(outdir)


# Throughput at one ladder size
def _spread_pct(ns_at, rows, minkey, medkey, N, laws, **sel):
    """Worst over the laws of (median - minimum)/minimum, in per cent."""
    # `ns_at` is the caller's row lookup (a closure in the table function).
    worst = None
    for l in laws:
        a = ns_at(rows, minkey, N, law=l, **sel)
        b = ns_at(rows, medkey, N, law=l, **sel)
        if _ok(a) and _ok(b) and a > 0 and b >= a:
            v = 100.0 * (b / a - 1.0)
            worst = v if worst is None else max(worst, v)
    return DASH if worst is None else text("%.1f %%" % worst)


def tab_throughput(lad, slad, outdir, runs_note, arch, family="z_nu",
                   asm_of=None):
    """Cost per element at the largest ladder rung common to every row, one row
    per implementation, one column pair (cost, ratio) per ladder law."""
    stem = f"tab-throughput-{family}"
    if not lad:
        return _skip(stem, "no ladder rows -- the ladder stage did not run")
    laws = [l for l in LADDER_LAWS if any(r.get("law") == l for r in lad)]
    if not laws:
        return _skip(stem, "no rows on either ladder law")
    backends = [b for b in BACKEND_ORDER
                if any(r.get("backend") == b and r.get("family") == family
                       for r in lad)]
    if not backends:
        return _skip(stem, f"the ladder carries no {family} rows")

    def ns_at(rows, key, want_n, **sel):
        for r in rows:
            if all(r.get(k) == v for k, v in sel.items()) and _i(r, "n") == want_n:
                return _f(r, key)
        return float("nan")

    slad = [r for r in (slad or []) if r.get("family") == family]
    sup_names = sorted({r["supplier"] for r in slad
                        if (r.get("supplier") or "-") != "-"}) if slad else []
    # Rungs the library has on every law, intersected with every external's.
    rungs = None
    for l in laws:
        got = {_i(r, "n") for r in lad if r.get("law") == l
               and r.get("family") == family}
        rungs = got if rungs is None else (rungs & got)
    for s in sup_names:
        for l in laws:
            got = {_i(r, "n") for r in slad if r.get("supplier") == s
                   and r.get("law") == l}
            if got:
                rungs &= got
    if not rungs:
        return _skip(stem, "the library and the externals share no ladder rung "
                           "-- every row would be quoted at a different N")
    N = max(rungs)

    head = ["implementation"] + [LAW_HEAD.get(l, l) for l in laws] \
           + [LAW_HEAD.get(l, l) for l in laws] + [r"med$-$min"]
    nl = len(laws)
    t = Table(stem,
              _caption(f"Single-threaded runtime cost per element at "
                       f"$N={_grouped(N)}$ for the size-scaling datasets "
                       + pf.CONV_SUPPLIED.get(family, esc(family)) + ".",
                       "The cost columns give the cost in nanoseconds per "
                       "element for the " + pf._and(pf._size_set(l) for l in laws)
                       + r" datasets, for the library rows of the \texttt{avx512} "
                       r"and \texttt{avx2} backends, which use a workspace, including "
                       r"its creation and destruction in each call; "
                       r"the ratio columns give the speedup of the "
                       r"\texttt{" + esc(backends[-1]) + r"} backend over each "
                       "external implementation, computed from the corresponding "
                       "costs.", MEDIAN_COST, _obtained(arch)),
              f"tab:throughput-{family.replace('_', '-')}", head,
              align=wrapcol("3.4cm") + "r" * (2 * nl + 1),
              header_groups=[(2, 1 + nl, "ns / element"),
                             (2 + nl, 1 + 2 * nl, r"$\times$ this library")])
    base = {}
    for b in backends:
        cells = [text(f"{LIB} ({b})")]
        for l in laws:
            # On avx512 and avx2 the workspace creation and destruction are timed
            # (`ns_ws_per_call_*`).
            v = ns_at(lad, "ns_ws_per_call_med", N, backend=b, family=family, law=l)
            cells.append(num(v))
            if b == backends[-1]:
                base[l] = v
        cells += [DASH] * nl
        cells.append(_spread_pct(ns_at, lad, "ns_ws_per_call_min",
                                 "ns_ws_per_call_med", N, laws,
                                 backend=b, family=family))
        t.add(cells)
    t.rule_here()
    asm_of = asm_of or {}
    for s in sup_names:
        want = asm_of.get(s)
        # Assembly display name as in the figures (exp-log, not logtrick).
        cells = [text(s + (f" ({pf.asm_label(want)})" if want else ""))]
        vals = []
        for l in laws:
            sel = {"supplier": s, "law": l}
            if want:
                sel["backend"] = want
            # Median on both sides of every ratio.
            v = ns_at(slad, "ns_med", N, **sel)
            vals.append(v)
            cells.append(num(v))
        for l, v in zip(laws, vals):
            b = base.get(l)
            cells.append(ratio(v / b) if _ok(v) and _ok(b) and b > 0 else DASH)
        cells.append(_spread_pct(ns_at, slad, "ns_min", "ns_med", N, laws,
                                 supplier=s,
                                 **({"backend": want} if want else {})))
        t.add(cells)
    # `--` outside the library rows' ratio columns marks a missing value.
    _gap = any(c is DASH for i, r in enumerate(t.rows) for j, c in enumerate(r)
               if not (i < len(backends) and 1 + nl <= j <= 2 * nl))
    t.note = _note(
        r"\emph{med$-$min}: largest excess of the median over the fastest "
        r"repetition over the datasets of the row, in per cent. "
        + (r"--: not applicable or not measured. " if _gap
           else r"--: not applicable. "))
    t.provenance(
        runs_note,
        r"Each cell is the median over repetitions of one call. On the "
        r"\texttt{avx512} and \texttt{avx2} backends of this library every "
        r"repetition creates and destroys the workspace inside the timed region "
        r"(\texttt{ns\_ws\_per\_call\_med}); the other backends and the externals "
        r"use no workspace.",
        r"$N$ is the largest rung every row reached: an external's ladder "
        r"stops when a rung exceeds its wall-clock budget, so the comparison "
        r"covers $N$ up to this value.",
        r"Each external is shown in the assembly selected from its accuracy "
        r"rows, the same in every table.",
        _machine(arch))
    return t.write(outdir)


def tab_gpu_throughput(gklad, outdir, runs_note, arch, family="z_nu",
                       asm_of=None, lib_supplier=LIB_CUDA):
    """`tab_throughput` for the device kernels at kernel scope (no transfers, no
    workspace), the device library first."""
    stem = f"tab-gpu-throughput-{family}"
    if not gklad:
        return _skip(stem, "no device kernel-ladder rows -- the GPU "
                           "kernels-ladder stage did not run")
    rows = [r for r in gklad if r.get("family") == family
            and scope_of(r) == "kernel"]
    laws = [l for l in LADDER_LAWS if any(r.get("law") == l for r in rows)]
    if not laws:
        return _skip(stem, f"the device kernel ladder carries no {family} "
                           "rows on a ladder law")
    names = sorted({r["supplier"] for r in rows})
    if lib_supplier not in names or len(names) < 2:
        return _skip(stem, "needs the device library and at least one "
                           "external kernel on the kernel ladder")
    names = [lib_supplier] + [n for n in names if n != lib_supplier]
    asm_of = asm_of or {}

    def sel_of(s):
        want = asm_of.get(s) if s != lib_supplier else None
        return {"supplier": s, **({"backend": want} if want else {})}

    def ns_at(rs, key, want_n, **sel):
        for r in rs:
            if all(r.get(k) == v for k, v in sel.items()) and _i(r, "n") == want_n:
                return _f(r, key)
        return float("nan")

    rungs = None
    for s in names:
        for l in laws:
            got = {_i(r, "n") for r in rows if r.get("law") == l
                   and all(r.get(k) == v for k, v in sel_of(s).items())}
            rungs = got if rungs is None else (rungs & got)
    if not rungs:
        return _skip(stem, "the device kernels share no ladder rung -- every "
                           "row would be quoted at a different N")
    N = max(rungs)
    top = max(_i(r, "n") for r in rows)

    # Column heads without the corpus size; N is given in the caption.
    lhead = {"size_random_large": "Size / rand.", "size_half_large": "Size / half"}
    head = ["kernel"] + [lhead.get(l, LAW_HEAD.get(l, l)) for l in laws] \
           + [lhead.get(l, LAW_HEAD.get(l, l)) for l in laws] + [r"med$-$min"]
    nl = len(laws)
    reach = ("the top of the device kernel ladder, which every row reached"
             if N == top else
             "the largest rung every row reached; the ladder itself "
             f"went to ${_grouped(top)}$")
    t = Table(stem,
              _caption(f"Per-element runtime of the GPU kernels at "
                       f"$N={_grouped(N)}$ for the size-scaling datasets "
                       + pf.CONV_SUPPLIED.get(family, esc(family)) + ".",
                       "The cost columns give the kernel cost in nanoseconds "
                       "per element for the "
                       + pf._and(pf._size_set(l) for l in laws)
                       + " datasets; the ratio columns give the speedup of "
                       + esc(lib_supplier) + " over each external "
                       "implementation, computed from the corresponding kernel "
                       "costs.", MEDIAN_COST, _obtained(arch, gpu=True)),
              f"tab:gpu-throughput-{family.replace('_', '-')}", head,
              align=wrapcol("3.4cm") + "r" * (2 * nl + 1),
              header_groups=[(2, 1 + nl, "ns / element"),
                             (2 + nl, 1 + 2 * nl, r"$\times$ this library")])
    base = {}
    for s in names:
        want = sel_of(s).get("backend")
        cells = [text(s + (f" ({pf.asm_label(want)})" if want else ""))]
        vals = [ns_at(rows, "ns_med", N, law=l, **sel_of(s)) for l in laws]
        cells += [num(v) for v in vals]
        if s == lib_supplier:
            base = dict(zip(laws, vals))
            cells += [DASH] * nl
        else:
            for l, v in zip(laws, vals):
                b = base.get(l)
                cells.append(ratio(v / b) if _ok(v) and _ok(b) and b > 0
                             else DASH)
        cells.append(_spread_pct(ns_at, rows, "ns_min", "ns_med", N, laws,
                                 **sel_of(s)))
        t.add(cells)
        if s == lib_supplier:
            t.rule_here()
    # `--` outside the library row's ratio columns marks a missing value.
    _gap = any(c is DASH for i, r in enumerate(t.rows) for j, c in enumerate(r)
               if not (i == 0 and 1 + nl <= j <= 2 * nl))
    t.note = _note(
        r"\emph{med$-$min}: largest excess of the median over the fastest "
        r"repetition over the datasets of the row, in per cent. "
        + (r"--: not applicable or not measured. " if _gap
           else r"--: not applicable. "))
    t.provenance(
        runs_note,
        r"$N$ is " + reach + ".",
        r"Each cell is the median over the timed repetitions of the elapsed "
        r"device time of the kernel alone; the ratio is the external's cost "
        r"divided by \texttt{" + esc(lib_supplier) + r"}'s. Kernel scope "
        r"excludes the host-device transfers, which are the same for every "
        r"row and are compared in the transfer-strategy table.",
        r"Each external is shown in the assembly selected from its device "
        r"accuracy rows, by the rule of the host throughput table.",
        _machine(arch))
    return t.write(outdir)


# The two argument conventions
def tab_convention(acc, outdir, runs_note, arch, backend=None):
    """Maximum ULP and cost per law with $(z,\\nu)$ supplied and with $z$ derived
    from $(h,\\nu,s)$, and their ratios."""
    stem = "tab-convention"
    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in acc)]
    if len(fams) < 2:
        return _skip(stem, f"{len(fams)} argument convention(s) measured "
                           f"({', '.join(fams) or 'none'}) --- the comparison "
                           f"needs both")
    backend = backend or lib_backend_for(acc)
    laws = laws_present(acc)
    if not laws:
        return _skip(stem, "no rows on the paper's accuracy laws")
    by = {(r["family"], r["law"]): r for r in acc
          if r.get("backend") == backend and scope_of(r) == "call"}
    head = ["dataset", r"$(z,\nu)$", r"$(h,\nu,s)$", "ratio",
            r"$(z,\nu)$", r"$(h,\nu,s)$", "ratio"]
    t = Table(stem,
              _caption(r"Maximum ULP error and runtime cost per element of the "
                       r"\texttt{" + esc(backend) + r"} backend with $z$ "
                       r"supplied, $(z,\nu)$, and with $z$ computed from "
                       r"$(h,\nu,s)$, by dataset.", MEDIAN_COST,
                       _obtained(arch)),
              "tab:convention", head, align="lrrrrrr",
              header_groups=[(2, 4, "max ULP error"), (5, 7, "ns / element")])
    for l in laws:
        a, b = by.get(("z_nu", l)), by.get(("h_nu_s", l))
        if not (a and b):
            continue
        ua, ub = _f(a, "u_max"), _f(b, "u_max")
        ca, cb = _f(a, "ns_med"), _f(b, "ns_med")
        t.add([raw(LAW_HEAD.get(l, l), _demath(LAW_HEAD.get(l, l))),
               ulp(a), ulp(b),
               ratio(ub / ua) if _ok(ua) and ua > 0 and _ok(ub) else DASH,
               num(ca), num(cb),
               ratio(cb / ca) if _ok(ca) and ca > 0 and _ok(cb) else DASH])
    if not t.rows:
        return _skip(stem, "no law carries both conventions")
    t.note = _note(
        r"ratio: the $(h,\nu,s)$ value divided by the $(z,\nu)$ value. "
        + marker_note([r for l in laws
                       if by.get(("z_nu", l)) and by.get(("h_nu_s", l))
                       for r in (by[("z_nu", l)], by[("h_nu_s", l)])]))
    t.provenance(
        runs_note,
        r"Computing $z$ from $(h,\nu,s)$ in double precision has an error of "
        r"about $z$ times the machine epsilon, and since $M(h)$ decays like "
        r"$e^{-z}$, that absolute error in $z$ reaches the result as a "
        r"relative error of the same size.",
        r"$z$ is derived in the router, before any backend-specific kernel, "
        r"in the same way for every backend; one backend therefore suffices.",
        _machine(arch))
    return t.write(outdir)


# Measurement environment
# (provenance key, row label) in display order; a key absent from every run is
# omitted, a key absent from one run prints `--`.
ENV_KEYS = [
    # Derived keys, added by the driver (runs.machine_key, env_foreign_load).
    ("machine",           "Machine"),
    ("load_foreign",      "Other load, mean"),
    ("cpu",               "CPU"),
    ("cpus_physical",     "Physical cores"),
    ("cpus_online",       "Hardware threads"),
    ("sockets",           "Sockets"),
    ("numa_nodes",        "NUMA domains"),
    ("os",                "OS"),
    ("kernel",            "Kernel"),
    # Read from the artefacts (detect --build-info); see ENV_FALLBACK.
    ("build_cpu_library_compiler",       "Compiler, library"),
    ("build_cpu_supplier_cxx_compiler",  "Compiler, C++ suppliers"),
    # The Makefile's `FMA` setting for the scalar_portable and matched objects.
    ("fma_knob",          "FMA in scalar_portable/matched"),
    ("gpu",               "GPU"),
    ("gpu_compute_cap",   "Compute capability"),
    ("gpu_memory_total",  "Device memory"),
    ("gpu_driver",        "Driver"),
    ("gpu_cuda_driver",   "CUDA driver API"),
    ("build_gpu_benchmark_compiler",      "nvcc"),
    ("build_gpu_benchmark_host_compiler", "nvcc host compiler"),
    ("gpu_arch_built",    "Built for"),
    ("gpu_ecc",           "ECC"),
    ("gpu_persistence",   "Persistence mode"),
    ("gpu_clock_sm_max",  "Max SM clock"),
    ("gpu_clock_mem_max", "Max memory clock"),
    ("gpu_power_limit",   "Power limit"),
    # Recorded before the timed stages start.
    ("gpu_throttle",      "Throttle before run"),
    ("build_cpu_supplier_cxx_boost",     "Boost, CPU row"),
    ("build_gpu_suppliers_boost",        "Boost, GPU row"),
    ("build_cpu_supplier_gsl_runtime",   "GSL"),
    ("l3_total",          "L3 cache (total)"),
    ("cpufreq_driver",    "cpufreq driver"),
    ("cpufreq_governor",  "cpufreq governor"),
    ("cpufreq_range_kHz", "cpufreq range"),
    # The clock right after each timed row (runs since the sidecars record it).
    ("cpu_mhz_timed",     "Clock in the timed rows"),
    ("turbo",             "Turbo boost"),
    ("cpu_flags_simd",    "SIMD ISA"),
]

# Cache rows from the recorded `cache_L<level><type>` keys (e.g. `cache_L3U`).
def _cache_keys(machines):
    seen = []
    for _, rs in machines:
        for _, pv in rs:
            for k in pv:
                if k.startswith("cache_L") and k not in seen:
                    seen.append(k)
    return [(k, k.replace("cache_L", "L").replace("D", "d").replace("I", "i")
              .replace("U", "") + " cache") for k in sorted(seen)]


# A key absent or empty in a run is filled from the first non-empty key listed for
# it: a run without a build_* key carries the run-time toolchain key, and the driver
# API that nvidia-smi did not report is the one the device programs recorded
# (cudaDriverGetVersion, the same number).
ENV_FALLBACK = {
    "build_cpu_library_compiler":   ("compiler",),
    "build_gpu_benchmark_compiler": ("nvcc",),
    "build_gpu_suppliers_boost":    ("boost_root",),
    "build_cpu_supplier_gsl_runtime": ("gsl",),
    "gpu_cuda_driver": ("build_gpu_benchmark_cuda_driver_api",
                        "build_gpu_suppliers_cuda_driver_api"),
    # The published runs record it as l3_per_socket; the value is the total there as well.
    "l3_total": ("l3_per_socket",),
}


def _env_with_fallback(pv):
    """`pv` with each absent or empty ENV_FALLBACK key filled from the first non-empty
    key listed for it."""
    out = dict(pv)
    for new, olds in ENV_FALLBACK.items():
        if out.get(new) in (None, ""):
            for o in olds:
                if out.get(o) not in (None, ""):
                    out[new] = out[o]
                    break
    return out


# Stated once in the caption when identical on every machine, otherwise rows.
ENV_CAPTION_KEYS = ("sockets", "kernel", "fma_knob", "cpu_flags_simd")


def env_foreign_load(run_dir):
    """Mean over the run's `loadavg.tsv` samples of `foreign_busy * n_foreign /
    (n_foreign + n_own)` (`foreign_busy`: busy fraction of the `n_foreign` CPUs not
    given to the run; `n_own`: the run's CPUs): the share of the machine's CPUs
    kept busy by other processes; None without usable samples."""
    path = os.path.join(run_dir, "loadavg.tsv")
    if not os.path.exists(path):
        return None
    vals, cols = [], None
    for line in open(path, errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if cols is None:
            cols = f
            continue
        r = dict(zip(cols, f))
        try:
            nf, no = int(r["n_foreign"]), int(r["n_own"])
        except (KeyError, ValueError):
            continue
        if nf == 0:
            vals.append(0.0)
            continue
        try:
            vals.append(float(r["foreign_busy"]) * nf / (nf + no))
        except (KeyError, ValueError):
            continue            # `NA`: the sampler could not read that interval
    return sum(vals) / len(vals) if vals else None


def _env_stage(rid):
    """The stage of a run id: `<stamp>_<host>_cpu-rest` -> `rest`;
    `..._gpu` -> `gpu`."""
    tail = str(rid).rsplit("_", 1)[-1]
    return tail[4:] if tail.startswith("cpu-") else tail


def _env_recorded(key, v):
    """A provenance value as the `.csv` carries it: whitespace collapsed, an
    absent GPU (`none -- <reason>`) as `none`."""
    v = " ".join(str(v).split())
    if key == "gpu" and v.lower().startswith("none"):
        return "none"
    return v


def _env_tex(val):
    """`breakable` without the break after an opening parenthesis."""
    return style.Raw(str(breakable(val)).replace(r"(\allowbreak{}", "("))


def _cpu_count(cpulist):
    """`0-31,64-95` -> 64.  None when the list is not in that form."""
    n = 0
    for part in str(cpulist).split(","):
        a, _, b = part.partition("-")
        if not a.isdigit() or (b and not b.isdigit()):
            return None
        n += (int(b) - int(a) + 1) if b else 1
    return n


def _env_smt(pv):
    """Hardware threads per physical core for one run; 0 when unrecorded."""
    try:
        on, ph = int(pv.get("cpus_online")), int(pv.get("cpus_physical"))
    except (TypeError, ValueError):
        return 0
    return on // ph if ph and on % ph == 0 else 0


def _simd_names(flags):
    """`avx avx2 avx512dq avx512f avx512vl fma sse2` ->
    `SSE2, AVX, AVX2, FMA, AVX-512 (F, DQ, VL)` -- the vendors' spelling."""
    fl = str(flags).split()
    base = [f.upper() for f in ("sse2", "avx", "avx2", "fma") if f in fl]
    sub = [f[6:].upper() for f in ("avx512f", "avx512dq", "avx512vl",
                                   "avx512bw", "avx512cd") if f in fl]
    other = [f.upper() for f in fl
             if f not in ("sse2", "avx", "avx2", "fma") and not f.startswith("avx512")]
    return ", ".join(base + other + ([f"AVX-512 ({', '.join(sub)})"] if sub else []))


def _env_short(key, v, smt=0):
    """The `.tex`/`.md` form of one provenance value, None for `--`: the part
    before ` -- `, except for `gpu_throttle` (`<mask> -- GpuIdle`)."""
    v = " ".join(str(v).split())
    if key == "load_foreign":
        return f"{100 * float(v):.2f} %"
    if key == "gpu_throttle":
        return v.split(" -- ", 1)[-1]
    head = v.split(" -- ", 1)[0]
    if key == "machine":
        return head
    if key == "cpu":
        head = head.replace("(R)", "").replace(" CPU ", " ")
        # Drop "N-Core", which duplicates the physical-cores row.
        head = re.sub(r"\s\d+-Core\b", "", head.replace(" Processor", ""))
        return " ".join(head.split())
    if key == "os":
        head = head.replace("Red Hat Enterprise Linux", "RHEL")
        return re.sub(r"\s*\([^)]*\)$", "", head)
    if key == "kernel":
        return re.sub(r"\.(x86_64|aarch64)$", "", head)
    if key in ("compiler", "build_cpu_library_compiler",
               "build_cpu_supplier_cxx_compiler", "build_gpu_benchmark_host_compiler"):
        # `gcc (GCC) 14.2.0`, `gcc 11.5.0 20240719 ...` (__VERSION__) or
        # `GCC: (GNU) 11.5.0 ...  [n of m object(s)]` (.comment stamp).
        head = re.sub(r"\s*\[\d+ of \d+ object\(s\)\]$", "", head)
        m = re.match(r"(gcc|clang)\b.*?(\d+\.\d+(?:\.\d+)?)", head, re.I)
        return f"{'GCC' if m.group(1).lower() == 'gcc' else 'Clang'} {m.group(2)}" \
            if m else _shorten(head)
    if key in ("nvcc", "build_gpu_benchmark_compiler"):
        m = re.search(r"V(\d+\.\d+\.\d+)", head) or \
            re.match(r"nvcc (\d+\.\d+\.\d+)", head)
        return m.group(1) if m else head
    if key == "gpu" and head.startswith("none"):
        return "None"
    # On a machine without a device the recorded "unknown ..." means no GPU.
    if key == "gpu_arch_built" and head.startswith("unknown"):
        return None
    if key in ("boost_root", "build_gpu_suppliers_boost",
               "build_cpu_supplier_cxx_boost"):
        m = re.search(r"\(([\d.]+)\)", head)
        return m.group(1) if m else head
    if key in ("gsl", "build_cpu_supplier_gsl_runtime"):
        return head.split(" at ", 1)[0]
    if key == "cpu_flags_simd":
        return _simd_names(head)
    if key.startswith("cache_L"):
        m = re.match(r"size=(\S+) shared_by=(\S+)$", head)
        if m:
            n = _cpu_count(m.group(2))
            # sysfs reports `48K` / `49152K`; convert to KiB / MiB.
            size = m.group(1)
            kb = re.match(r"(\d+)K$", size)
            if kb:
                k = int(kb.group(1))
                size = f"{k / 1024:g} MiB" if k >= 1024 else f"{k} KiB"
            if n and smt > 1 and n % smt == 0:
                c = n // smt
                return size + (" per core" if c == 1 else f" per {c} cores")
            return size + (f", shared by {n} threads" if n else "")
    if key == "cpufreq_range_kHz":
        nums = re.findall(r"\d+", head)
        if len(nums) == 2:
            return f"{int(nums[0]) / 1e6:g}-{int(nums[1]) / 1e6:g} GHz"
    if key == "turbo":
        m = re.search(r"(no_turbo|boost)=([01])", head)
        if m:
            on = (m.group(2) == "0") == (m.group(1) == "no_turbo")
            return "On" if on else "Off"
    if key == "fma_knob":
        m = re.match(r"FMA=(\w+)$", head)
        if m:
            return m.group(1)
    return _shorten(head)


def tab_environment(machines, outdir, runs_note):
    """One column per machine class, one row per field; `machines` is
    [(label, [(run_id, provenance)])].  A field that differs between the runs of
    a class shows the newest value with a double dagger."""
    stem = "tab-environment"
    machines = [(lab, [(rid, _env_with_fallback(pv)) for rid, pv in rs if pv])
                for lab, rs in machines]
    machines = [(lab, rs) for lab, rs in machines if rs]
    if not machines:
        return _skip(stem, "no run records a PROVENANCE.txt")
    keys = [(k, lab) for k, lab in ENV_KEYS + _cache_keys(machines)
            if any(pv.get(k) not in (None, "") for _, rs in machines
                   for _, pv in rs)]
    if not keys:
        return _skip(stem, "PROVENANCE.txt records none of the fields this "
                           "table reports")
    head = ["Field"] + [esc(m) for m, _ in machines]
    rows, clashes, caption_bits = [], [], []
    for k, label in keys:
        cells, shown = [], []
        for mlab, rs in machines:
            got = [(_env_stage(rid), pv[k], _env_smt(pv))
                   for rid, pv in rs if pv.get(k) not in (None, "")]
            if not got:
                cells.append(DASH)
                shown.append(None)
                continue
            full = " | ".join(f"{s}: {_env_recorded(k, v)}" for s, v, _ in got)
            if k == "load_foreign":
                val = _env_short(k, sum(float(v) for _, v, _ in got) / len(got))
                cells.append(Cell(style.Raw(val.replace(" %", r"\,\%")), val, full))
                shown.append(None)
                continue
            short = {}
            for s, v, smt in got:
                short.setdefault(_env_short(k, v, smt), []).append(s)
            short = {v: sorted(ss) for v, ss in
                     sorted(short.items(), key=lambda kv: str(kv[0]))}
            if k == "machine":
                val = "; ".join(f"{v} ({', '.join(ss)})" for v, ss in short.items())
                cells.append(Cell(_env_tex(val), val, full))
                shown.append(None)
            elif len(short) == 1:
                val = next(iter(short))
                if val is None:
                    cells.append(Cell(DASH.tex, DASH.md, full))
                else:
                    cells.append(Cell(_env_tex(val), val, full))
                shown.append(val)
            else:
                clashes.append(f"{mlab}/{k}")
                val = str(_env_short(k, got[-1][1], got[-1][2]))
                cells.append(Cell(style.Raw(breakable(val)
                                            + r"\textsuperscript{$\ddagger$}"),
                                  val + " (ddag)", full))
                shown.append(None)
        if k in ENV_CAPTION_KEYS and len(machines) > 1 \
                and shown[0] is not None and all(s == shown[0] for s in shown):
            caption_bits.append((k, label, shown[0]))
            continue
        rows.append([Cell(label, f"{label} [{k}]", k)] + cells)
    cap = ("Benchmark environments used for the measurements, one column per "
           "machine class.")
    if caption_bits:
        say = {"sockets": lambda v: f"{v} sockets",
               "kernel": lambda v: f"{v} kernel",
               "fma_knob": lambda v: ("scalar_portable and matched backends built "
                                      + ("without" if v == "off" else "with")
                                      + " FMA"),
               "cpu_flags_simd": lambda v: f"SIMD extensions {v}"}
        cap += " Common to all machines: " + ", ".join(
            say.get(k, lambda v, _l=lab: f"{_l} {v}")(v)
            for k, lab, v in caption_bits) + "."
    t = Table(stem, esc(cap), "tab:environment", head,
              align=_env_align(len(machines)))
    for r in rows:
        t.add(r)
    t.note = _note(
        r"\emph{Machine}: the machine and the stages run on it. "
        r"\emph{Other load}: mean share of the machine's CPUs kept busy by "
        r"other processes during the runs. "
        + (r"$\ddagger$: not the same in every run on that machine class; the "
           r"newest value is shown. " if clashes else "")
        + (r"--: not applicable or not recorded. "
           if any(c.md == "--" for r in rows for c in r) else ""))
    t.provenance(
        runs_note,
        r"\emph{Other load} is sampled every 30~s and averaged over the runs "
        r"on that machine class; it is 0 on a machine dedicated to the run.",
        (r"Fields marked with a double dagger: "
         + ", ".join(r"\texttt{%s}" % esc(c) for c in clashes) + ".")
        if clashes else "",
        r"Values are abbreviated; the accompanying \texttt{.csv} names every "
        r"field by its provenance key and carries each run's value as "
        r"recorded, an absent GPU as \texttt{none}.")
    return t.write(outdir)


def _env_align(n_machines):
    """The field column (ENV_FIELD_FRAC), then the machine classes in the rest."""
    # 0.98 rather than 1.00 leaves room for the inter-column rules.
    rest = (0.98 - ENV_FIELD_FRAC) / max(1, n_machines)
    return fraccol(ENV_FIELD_FRAC) + fraccol(rest) * n_machines


# Wide enough for the longest label ("Throttle before run") on one line at \small.
ENV_FIELD_FRAC = 0.235


def breakable(v):
    """An environment value with `\\allowbreak` after separators and after every
    ten characters without one."""
    # Escape per character, so that no break falls inside an escape sequence.
    out, run = [], 0
    for ch in " ".join(str(v).split()):
        out.append(style.tex_escape(ch))
        run += 1
        if ch.isspace():
            run = 0
        elif ch in "-/,;:=.+()@_":
            out.append(r"\allowbreak{}")
            run = 0
        elif run >= 10:
            out.append(r"\allowbreak{}")
            run = 0
    return style.Raw("".join(out))


def _shorten(v, n=110):
    """`v` truncated to `n` characters with an ellipsis; the CSV keeps it whole."""
    v = " ".join(str(v).split())
    return v if len(v) <= n else v[:n - 1] + "…"


# Test corpus
def tab_datasets(acc, outdir, runs_note):
    """Each law and its size per argument convention, from the accuracy rows."""
    stem = "tab-datasets"
    if not acc:
        return _skip(stem, "no accuracy rows -- the set sizes are recorded on "
                           "them")
    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in acc)]
    laws = laws_present(acc, ACC_LAWS + LADDER_LAWS)
    if not laws or not fams:
        return _skip(stem, "no law/convention pair present")
    head = ["dataset"] + [FAMILY_LABEL.get(f, f) for f in fams]
    t = Table(stem,
              "Benchmark datasets and their sizes, in points, for each "
              "argument convention.",
              "tab:datasets", head, align="l" + "r" * len(fams))
    for l in laws:
        cells = [raw(LAW_HEAD.get(l, l), _demath(LAW_HEAD.get(l, l)))]
        for f in fams:
            ns = {_i(r, "n") for r in acc
                  if r.get("law") == l and r.get("family") == f and _i(r, "n")}
            cells.append(count(max(ns)) if ns else DASH)
        t.add(cells)
        if l == (ACC_LAWS[-1] if ACC_LAWS[-1] in laws else None):
            t.rule_here()
    size = [l for l in laws if l in LADDER_LAWS]
    t.note = _note(
        r"Size: the datasets of the size-scaling benchmarks. " if size else "")
    t.provenance(
        runs_note,
        r"Each cell is the number of $(z,\nu)$ or $(h,\nu,s)$ tuples the "
        r"harness evaluated, read from the accuracy rows of the runs; the "
        r"accuracy and cost results are scored over these points.",
        r"The size datasets are swept over a ladder of set sizes; the ladder "
        r"figure shows the whole ladder and the throughput table quotes its "
        r"largest common rung." if size else "")
    return t.write(outdir)


# References below the normal range
_DBL_MIN = 2.2250738585072014e-308


def _share(v, n):
    """`v (p %)`: a count with its share of `n`; a bare 0 for zero."""
    if v == 0:
        return count(0)
    p = 100.0 * v / n
    s = f"{v:,}"
    tex = s.replace(",", r"\,") + f" ({p:.1f}" + r"\,\%)"
    return Cell(style.Raw(tex), f"{s} ({p:.1f} %)", str(v))


def tab_references(acc, sup, outdir, runs_note):
    """Zero and subnormal references per law, from the rows that scored every
    point; the README.md facts use the rank-0 `worst` record of the sidecars."""
    stem = "tab-references"
    if not acc:
        return _skip(stem, "no accuracy rows -- the reference counts are "
                           "recorded on them")
    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in acc)]
    laws = laws_present(acc)
    if not laws or not fams:
        return _skip(stem, "no law/convention pair present")
    rows = [r for r in list(acc) + list(sup or []) if scope_of(r) == "call"]

    # Corpus counts, from the rows that scored every point.
    full = {}
    for f in fams:
        for l in laws:
            got = {(_i(r, "n"), _i(r, "ref_zero"), _i(r, "ref_subnormal"))
                   for r in rows if r.get("family") == f and r.get("law") == l
                   and _i(r, "n") and _i(r, "nonfinite") == 0
                   and _i(r, "n_scored") == _i(r, "n")}
            if len(got) != 1:
                return _skip(stem, f"{f}/{l}: {len(got)} distinct reference "
                                   f"counts among the rows that scored every "
                                   f"point (expected 1)")
            full[(f, l)] = got.pop()
    same = all(full[(fams[0], l)] == full[(f, l)] for f in fams for l in laws)
    shown = fams[:1] if same else fams

    head = ["dataset", "points"]
    for f in shown:
        lab = "" if same else f" {FAMILY_LABEL.get(f, f)}"
        head += [r"$M = 0$" + lab, r"subnormal $M$" + lab]
    t = Table(stem,
              "Count of zero and subnormal reference values for each dataset"
              + (", identical for both argument conventions"
                 if same and len(fams) > 1 else "") + ".",
              "tab:references", head, align="lr" + "r" * (2 * len(shown)))
    for l in laws:
        n = full[(shown[0], l)][0]
        cells = [raw(LAW_HEAD.get(l, l), _demath(LAW_HEAD.get(l, l))), count(n)]
        for f in shown:
            _, z, s = full[(f, l)]
            cells += [_share(z, n), _share(s, n)]
        t.add(cells)

    # Rows of the accuracy tables that score fewer of these points.
    tiny = [l for l in laws if any(full[(f, l)][1] + full[(f, l)][2]
                                   for f in fams)]
    fewer = {}                          # (law, zero, subnormal) -> [names]
    checked = at_tiny = unknown = 0
    for f in fams:
        backend = lib_backend_for(acc, family=f)
        for name, by_law in _impl_series(acc, sup, f, backend):
            for l in tiny:
                r = by_law.get(l)
                if r is None:
                    continue
                z, s = _i(r, "ref_zero"), _i(r, "ref_subnormal")
                if (z, s) != full[(f, l)][1:]:
                    fewer.setdefault((l, z, s), set()).add(name)
                e, _ = runs.detail_for(r)
                if e is None or e.get("worst_ref") is None:
                    unknown += 1
                    continue
                checked += 1
                if abs(e["worst_ref"]) < _DBL_MIN:
                    at_tiny += 1
    if fewer:
        parts = []
        for (l, z, s), names in sorted(fewer.items()):
            what = (r"none of them" if z == 0 and s == 0 else
                    f"{z} zero and {s} subnormal ones")
            parts.append(f"on {LAW_HEAD.get(l, l)}, "
                         + ", ".join(r"\texttt{%s}" % style.tex_escape(n)
                                     for n in sorted(names))
                         + f" score {what}")
        fewer_tex = (r"Points at which an implementation returned a non-finite "
                     r"result are excluded from its statistics ("
                     + r" and ".join(r"\texttt{tab-exclusions-%s}" % esc(f)
                                     for f in fams)
                     + r"); among the rows of the "
                     r"accuracy tables, " + "; ".join(parts) + ". ")
    else:
        fewer_tex = (r"Every row of the accuracy tables scores all of these "
                     r"points. ")
    if checked and not at_tiny:
        worst_tex = (r"In all %d rows of the accuracy tables on these datasets the "
                     r"maximum ULP error lies at a normal reference, so the "
                     r"maxima are unaffected" % checked)
    elif checked:
        worst_tex = (r"In %d of %d rows of the accuracy tables on these datasets "
                     r"the maximum ULP error lies at a zero or subnormal "
                     r"reference" % (at_tiny, checked))
    else:
        worst_tex = r"Where the maxima lie was not recorded"
    if unknown:
        worst_tex += r" (%d rows carry no worst-point record)" % unknown
    worst_tex += ". "
    t.note = _note(
        r"\emph{points}: size of the dataset. $M = 0$: the stored reference "
        r"is 0, the true value lies below $2^{-1075}$. Subnormal: "
        r"$0 < |M| < 2^{-1022}$. "
        + (r"Percentages are shares of the points. "
           if any("%" in c.md for r in t.rows for c in r) else ""))
    t.provenance(
        runs_note,
        r"Below $2^{-972}$ the low word of the double-double reference is a "
        r"multiple of $2^{-1074}$; where the high word is subnormal the low "
        r"word is 0 and the stored reference is exact only to half a ULP, so "
        r"the ULP error at these points can be understated by up to 0.5, and a "
        r"returned 0 at a zero reference scores 0. These points are scored "
        r"like every other point in the accuracy tables. Every implementation "
        r"is scored against the same references.",
        fewer_tex.strip(), worst_tex.strip())
    return t.write(outdir)


# Machine independence of accuracy
def tab_machine_independence(per_machine_acc, per_machine_sup, outdir,
                             runs_note, labels):
    """Per implementation, the accuracy fields equal across CPU architectures."""
    stem = "tab-machine-independence"
    # Two CPU architectures, not two labels: one model agrees by construction.
    import runs as _mp
    archs = sorted({_mp.cpu_arch_key(l) for l in labels})
    if len(archs) < 2:
        return _skip(stem, f"needs 2 CPU architectures, found {len(archs)} "
                           f"({', '.join(archs) or 'none'}) across "
                           f"{len(labels)} machine(s); runs with the same CPU "
                           f"model are one machine and are not compared")
    head = ["implementation", "fields compared", "differing", "verdict"]
    t = Table(stem,
              "Accuracy statistics of each implementation compared between "
              "the " + pf._and(esc(a) for a in archs) + " CPUs.",
              "tab:machine-independence", head, align="lrrl")
    groups = []
    if all(per_machine_acc):
        groups.append((LIB, per_machine_acc))
    sups = sorted({r["supplier"] for rows in per_machine_sup for r in rows
                   if (r.get("supplier") or "-") != "-"})
    for s in sups:
        groups.append((s, [[r for r in rows if r.get("supplier") == s]
                           for rows in per_machine_sup]))
    for name, rowsets in groups:
        if not all(rowsets):
            continue
        n, bad, _ = machine_agreement(list(zip(labels, rowsets)))
        if not n:
            continue
        t.add([text(name), count(n), count(bad),
               raw(r"bit-identical" if not bad else
                   r"\emph{differs}", "bit-identical" if not bad else "differs")])
    if not t.rows:
        return _skip(stem, "the machines share no comparable accuracy cell")
    t.note = _note(
        r"Compared: the maximum, median, 99th and 99.9th percentile ULP error, "
        r"the scored and non-finite counts and the maximum relative error of "
        r"every argument convention, dataset and backend both machines "
        r"measured. ")
    t.provenance(
        runs_note,
        r"Each compared statistic is a deterministic function of the returned "
        r"doubles, so a difference is a difference in the returned values.")
    return t.write(outdir)


# Device accuracy and cost at kernel scope
def tab_gpu(gsup, outdir, runs_note, arch, family="z_nu", scope="kernel",
            metric="accuracy"):
    r"""The accuracy or cost table (`metric`) of the device kernels."""
    stem = f"tab-gpu-{metric}-{family}"
    if not gsup:
        return _skip(stem, "no device supplier rows -- the GPU stage did not "
                           "run, or ran on a host with no CUDA device")
    rows = [r for r in gsup if scope_of(r) == scope and r.get("family") == family]
    if not rows:
        return _skip(stem, f"the device rows carry no {scope}-scope "
                           f"measurement for {family}")
    laws = laws_present(rows)
    best = best_assembly(rows)
    names = sorted({r["supplier"] for r in rows})
    names.sort(key=lambda s: (0 if s.startswith(LIB) else 1, s))
    if not laws or len(names) < 2:
        return _skip(stem, "needs the device library and at least one external "
                           "kernel on a shared law")
    acc = metric == "accuracy"
    head = ["kernel"] + [LAW_HEAD.get(l, l) for l in laws]
    t = Table(stem,
              _caption("Maximum ULP error over all scored points of the GPU "
                       "kernels for " + _sets(family) + ".",
                       BEST_ASSEMBLY,
                       _obtained(arch, gpu=True))
              if acc else
              _caption("GPU kernel execution cost for " + _sets(family) + ".",
                       "Entries give the kernel cost in nanoseconds per element "
                       "for each implementation and dataset; the final row "
                       "gives the speedup of " + esc(next(
                           (n for n in names if n.startswith(LIB)), LIB))
                       + " relative to the fastest external implementation for "
                       "each dataset.", MEDIAN_COST,
                       _obtained(arch, gpu=True)),
              f"tab:gpu-{metric}-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws))
    # Cost table, first row: the points timed per law (`ns_points`).
    if not acc:
        npts = {}
        for (sn, f, l), r in best.items():
            if f == family and _i(r, "ns_points") > 0:
                npts.setdefault(l, set()).add(_i(r, "ns_points"))
        t.add([raw(r"$N$ (points timed)", "N (points timed)", "N (points timed)")]
              + [count(max(npts[l])) if l in npts else DASH for l in laws])
        t.rule_here()
    med = {}                                   # {supplier: {law: ns_med}}
    for sname in names:
        by = {l: r for (sn, f, l), r in best.items()
              if sn == sname and f == family}
        cells = [text(sname)]
        for l in laws:
            if l not in by:
                cells.append(DASH)
            elif acc:
                cells.append(ulp(by[l]))
            else:
                v = _f(by[l], "ns_med")
                med.setdefault(sname, {})[l] = v
                cells.append(num(v))
        t.add(cells)
        if sname.startswith(LIB):
            t.rule_here()
    # Cost table, last row: fastest external cost / library cost, per law.
    if not acc:
        lib = next((n for n in names if n.startswith(LIB)), None)
        cells = [text("%s speed-up vs fastest external" % (lib or LIB))]
        for l in laws:
            a = med.get(lib, {}).get(l)
            ext = [med[n][l] for n in names if n != lib and l in med.get(n, {})
                   and _ok(med[n][l]) and med[n][l] > 0]
            cells.append(ratio(min(ext) / a) if (ext and a and _ok(a) and a > 0)
                         else DASH)
        t.rule_here()
        t.add(cells)
    shown = [r for (sn, f, l), r in best.items() if f == family and l in laws]
    t.note = _note(marker_note(shown) if acc else "")
    t.provenance(
        runs_note,
        r"Each cell is the maximum ULP error over every scored point."
        if acc else
        r"Each cell is the median over the timed repetitions of the elapsed "
        r"device time of the kernel alone, on the $N$ points stated per "
        r"column; the last row is the fastest external's cost divided by the "
        r"library's, per dataset. Kernel scope excludes the host-device "
        r"transfers, which are the same for every row and are compared in "
        r"the transfer-strategy table.",
        _machine(arch))
    return t.write(outdir)


def tab_gpu_exclusions(gsup, outdir, runs_note, arch, family="z_nu"):
    """`tab_exclusions` for the rows of the device accuracy table (kernel scope)."""
    stem = f"tab-gpu-exclusions-{family}"
    if not gsup:
        return _skip(stem, "no device supplier rows -- the GPU stage did not "
                           "run, or ran on a host with no CUDA device")
    rows = [r for r in gsup if scope_of(r) == "kernel"
            and r.get("family") == family]
    if not rows:
        return _skip(stem, f"the device rows carry no kernel-scope "
                           f"measurement for {family}")
    laws = laws_present(rows)
    best = best_assembly(rows)
    names = sorted({r["supplier"] for r in rows})
    names.sort(key=lambda s: (0 if s.startswith(LIB) else 1, s))
    if not laws or len(names) < 2:
        return _skip(stem, "needs the device library and at least one external "
                           "kernel on a shared law")
    by = {s: {l: r for (sn, f, l), r in best.items() if sn == s and f == family}
          for s in names}
    total = sum(_i(r, "nonfinite") for b in by.values() for r in b.values())
    if total == 0:
        return _skip(stem, "every device kernel returned a finite result at "
                           "every point")
    head = ["kernel"] + [LAW_HEAD.get(l, l) for l in laws]
    t = Table(stem,
              _caption("Points at which a GPU kernel returned no finite number, "
                       "for " + _sets(family) + ".", _obtained(arch, gpu=True)),
              f"tab:gpu-exclusions-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws))
    npts = {}
    for b in by.values():
        for l, r in b.items():
            if _i(r, "n"):
                npts.setdefault(l, set()).add(_i(r, "n"))
    t.add([raw(r"$N$ (points)", "N (points)", "N (points)")]
          + [count(max(npts[l])) if l in npts else DASH for l in laws])
    t.rule_here()
    for s in names:
        t.add([text(s)] + [count(_i(by[s][l], "nonfinite")) if l in by[s]
                           else DASH for l in laws])
        if s.startswith(LIB):
            t.rule_here()
    clean = sum(1 for b in by.values() for r in b.values()
                if _i(r, "nonfinite") == 0)
    cells = sum(len(b) for b in by.values())
    t.provenance(
        runs_note,
        r"Each cell is a count of points, not a rate; the first row is the size "
        r"of each dataset. For a non-zero cell the maximum in "
        r"\texttt{tab-gpu-accuracy-%s} is taken over fewer points and carries "
        r"a dagger there, also where it reads `\textit{Cap}'." % esc(family),
        r"As in that table, each external is shown on each dataset in its "
        r"best assembly: one that returned a finite result everywhere if there "
        r"is one, otherwise the one with the most scored points.",
        r"%d of %d measured cells are zero." % (clean, cells),
        _machine(arch))
    return t.write(outdir)


def tab_gpu_medmax(gsup, outdir, runs_note, arch, family="z_nu",
                   scope="kernel"):
    """`tab_medmax` for the device kernels of `tab_gpu`, with the `finite` column
    and the comparison rows."""
    stem = f"tab-gpu-medmax-{family}"
    if not gsup:
        return _skip(stem, "no device supplier rows -- the GPU stage did not "
                           "run, or ran on a host with no CUDA device")
    rows = [r for r in gsup if scope_of(r) == scope and r.get("family") == family]
    if not rows:
        return _skip(stem, f"the device rows carry no {scope}-scope "
                           f"measurement for {family}")
    laws = laws_present(rows)
    best = best_assembly(rows)
    names = sorted({r["supplier"] for r in rows})
    names.sort(key=lambda s: (0 if s.startswith(LIB) else 1, s))
    if not laws or len(names) < 2:
        return _skip(stem, "needs the device library and at least one external "
                           "kernel on a shared law")
    series = [(s, {l: r for (sn, f, l), r in best.items()
                   if sn == s and f == family}) for s in names]

    head = ["kernel"] + [LAW_HEAD.get(l, l) for l in laws] + ["finite"]
    t = Table(stem,
              _caption(r"Median and maximum ULP error (median\,/\,maximum) over "
                       "all scored points of the GPU kernels for "
                       + _sets(family) + ".",
                       BEST_ASSEMBLY,
                       _obtained(arch, gpu=True)),
              f"tab:gpu-medmax-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws) + "rr")
    for name, by_law in series:
        cells = [text(name)]
        for l in laws:
            if l not in by_law:
                cells.append(DASH)
                continue
            m, x = num(_f(by_law[l], "u_med")), ulp(by_law[l])
            cells.append(Cell(style.Raw(m.tex + r"\,/\,"  + x.tex),
                              m.md + " / " + x.md, m.csv + "/" + x.csv))
        cells += score_cells(by_law, laws)
        t.add(cells)
        if name.startswith(LIB):
            t.rule_here()

    lib_by = dict(series[0][1]) if series[0][0].startswith(LIB) else {}
    incomplete, dashed = 0, False
    if lib_by:
        near, adv, incomplete = medmax_comparison_rows(series, lib_by, laws)
        dashed = dashed_cells(near, adv)
        t.rules.add(len(t.rows) - 1)
        t.add(near + [DASH])
        t.add(adv + [DASH])

    shown = [b[l] for _, b in series for l in laws if l in b]
    t.note = _note(marker_note(shown) + FINITE_NOTE
                   + (medmax_comparison_note(dashed, shown) if lib_by
                      else ""))
    t.provenance(
        runs_note,
        r"Each cell is the median\,/\,maximum ULP error over the scored "
        r"points of the dataset, kernel scope. The maximum and its markers are "
        r"those of the device accuracy table, which this table extends with "
        r"the median; `\textit{Cap}' and the dagger apply to the maximum "
        r"only. The closing rows compare the median and the maximum "
        r"separately, as in the host table.",
        r"Quantiles are nearest-rank on the sorted sample. The \texttt{.csv} "
        r"writes the cell as \texttt{median/maximum} at full precision.",
        (f"`--' marks the {incomplete} dataset(s) on which no external returned a "
         r"finite number at every point and a maximum below "
         r"$10^{6}$~ULP.") if incomplete else "",
        _machine(arch))
    return t.write(outdir)


def tab_gpu_call(gsup, outdir, runs_note, arch, family="z_nu", asm_of=None,
                 lib_supplier=LIB_CUDA):
    """Call-scope device cost (transfers included): each row at its fastest
    transfer strategy, each external in its `asm_of` assembly."""
    stem = f"tab-gpu-call-{family}"
    if not gsup:
        return _skip(stem, "no device supplier rows -- the GPU stage did not "
                           "run, or ran on a host with no CUDA device")
    rows = [r for r in gsup if scope_of(r) == "call" and r.get("family") == family
            and "/" in (r.get("backend") or "")]
    if not rows:
        return _skip(stem, f"the device rows carry no call-scope measurement "
                           f"for {family}")
    asm_of = asm_of or {}
    best = {}                     # {supplier: {law: (ns_med, strategy, row)}}
    for r in rows:
        s = r["supplier"]
        asm, mode = r["backend"].split("/", 1)
        if s != lib_supplier and asm_of.get(s) and asm != asm_of[s]:
            continue
        v = _f(r, "ns_med")
        if not (_ok(v) and v > 0):
            continue
        cur = best.setdefault(s, {}).get(r["law"])
        if cur is None or v < cur[0]:
            best[s][r["law"]] = (v, mode, r)
    names = sorted(best)
    if lib_supplier not in names or len(names) < 2:
        return _skip(stem, "needs the device library and at least one external "
                           "kernel at call scope")
    names = [lib_supplier] + [n for n in names if n != lib_supplier]
    laws = [l for l in laws_present(rows) if l in best[lib_supplier]]
    if not laws:
        return _skip(stem, "the library carries no call-scope law")
    exts = names[1:]
    head = ["kernel"] + [LAW_HEAD.get(l, l) for l in laws]
    t = Table(stem,
              _caption("GPU call execution cost for " + _sets(family) + ".",
                       "Entries give the call cost in nanoseconds per element, "
                       "host-device transfers included, for each implementation "
                       "and dataset at its fastest data-transfer strategy; the "
                       "final row gives the speedup of " + esc(lib_supplier)
                       + " relative to the fastest external implementation for "
                       "each dataset.", MEDIAN_COST,
                       _obtained(arch, gpu=True)),
              f"tab:gpu-call-{family.replace('_', '-')}",
              head, align=wrapcol("2.6cm") + "r" * len(laws))
    t.add([raw(r"$N$ (points timed)", "N (points timed)", "N (points timed)")]
          + [count(_i(best[lib_supplier][l][2], "ns_points")
                   or _i(best[lib_supplier][l][2], "n")) for l in laws])
    t.add([text("fastest transfer strategy")]
          + [text(best[lib_supplier][l][1]) for l in laws])
    t.rule_here()

    # Externals are keyed [1], [2], ... in the LaTeX only.  The key is braced: a
    # row beginning with `[` after `\\` would be read as its optional length.
    key = {s: i + 1 for i, s in enumerate(exts)}

    def label(s):
        want = asm_of.get(s) if s != lib_supplier else None
        name = s + (f" ({pf.asm_label(want)})" if want else "")
        if s == lib_supplier:
            return text(name)
        return raw(f"{{[{key[s]}]}}~" + style.tex_escape(name), name, name)

    for s in names:
        t.add([label(s)] + [num(best[s][l][0]) if l in best[s] else DASH
                            for l in laws])
        if s == lib_supplier:
            t.rule_here()
    t.rule_here()
    # The fastest external may differ between laws, so it is named in its own row.
    fastest = {}
    for l in laws:
        ext = [(best[s][l][0], s) for s in exts if l in best[s]]
        if ext:
            fastest[l] = min(ext)
    t.add([text("fastest external")]
          + [raw(f"{{[{key[fastest[l][1]]}]}}", fastest[l][1], fastest[l][1])
             if l in fastest else DASH for l in laws])
    t.add([text("%s speed-up vs fastest external" % lib_supplier)]
          + [ratio(fastest[l][0] / best[lib_supplier][l][0]) if l in fastest
             else DASH for l in laws])
    # Laws whose rows differ in fastest strategy, named in the README.md.
    split = [l for l in laws
             if len({best[s][l][1] for s in names if l in best[s]}) > 1]

    def strategies(tex):
        if not split:
            return "The same strategy was fastest for every row of every dataset. "
        out = []
        for l in split:
            lib_mode = best[lib_supplier][l][1]
            other = {}                         # strategy -> [external]
            for s in exts:
                if l in best[s] and best[s][l][1] != lib_mode:
                    other.setdefault(best[s][l][1], []).append(s)
            parts = []
            for mode, who in sorted(other.items()):
                if len(who) == len([s for s in exts if l in best[s]]):
                    whom = "every external"
                elif tex:
                    whom = ", ".join(f"[{key[s]}]" for s in who)
                else:
                    whom = ", ".join(who)
                parts.append(f"{whom} with "
                             + (r"\texttt{%s}" % esc(mode) if tex else mode))
            out.append(f"on {LAW_LABEL.get(l, l)} the library is fastest with "
                       + (r"\texttt{%s}" % esc(lib_mode) if tex else lib_mode)
                       + " and " + "; ".join(parts))
        s = "; ".join(out)
        return ("Rows whose fastest strategy differs from the library's: "
                + s + ". Each row is shown at its own fastest. ")

    def note(tex):
        return ("fastest transfer strategy: the strategy of "
                + esc(lib_supplier) + ". "
                + (r"{[$n$]}: key of an external implementation; the "
                   r"fastest-external row names the fastest external "
                   r"implementation of the dataset by its key. " if tex else
                   "fastest external: the fastest external implementation of "
                   "the dataset. "))

    t.note = _note(note(True), note(False))
    t.provenance(
        runs_note,
        r"Each cell is the median over the timed repetitions of one "
        r"public-entry call on the $N$ points stated per column, with host "
        r"arrays as input and output, so the host-device transfers are "
        r"included. Each row is shown at its fastest transfer strategy on that "
        r"dataset (the strategies return bit-identical results).",
        strategies(False).strip(),
        r"The external kernels run inside this library's transfer path, so the "
        r"transfer cost is the same on both sides of every ratio, and the "
        r"ratios are smaller than the kernel-scope ones of the device cost "
        r"table. The last row is the fastest external's cost on that dataset "
        r"divided by the library's. Each external is shown in the assembly "
        r"selected by its device accuracy rows.",
        _machine(arch))
    return t.write(outdir)


# Device transfer strategies
def tab_gpu_modes(glad, outdir, runs_note, arch):
    """Call cost of each transfer strategy at the largest common rung (cost only:
    the strategies return bit-identical results)."""
    stem = "tab-gpu-modes"
    if not glad:
        return _skip(stem, "no device ladder rows")
    modes = sorted({r["backend"] for r in glad if r.get("backend")})
    pairs = sorted({(r["family"], r["law"]) for r in glad})
    if len(modes) < 2 or not pairs:
        return _skip(stem, f"{len(modes)} transfer mode(s) measured -- the "
                           f"comparison needs at least two")
    rungs = None
    for m in modes:
        got = {_i(r, "n") for r in glad if r.get("backend") == m}
        rungs = got if rungs is None else (rungs & got)
    if not rungs:
        return _skip(stem, "the transfer modes share no ladder rung")
    N = max(rungs)
    # One column per (convention, law), the convention as a spanning header.
    fams = sorted({f for f, _ in pairs})
    pairs = [(f, l) for f in fams for _, l in
             sorted({p for p in pairs if p[0] == f})]
    head = ["transfer strategy"] + [LAW_HEAD.get(l, l) for _, l in pairs]
    groups, col = [], 2
    for f in fams:
        k = sum(1 for pf_, _ in pairs if pf_ == f)
        if k:
            groups.append((col, col + k - 1, FAMILY_LABEL.get(f, f)))
            col += k
    t = Table(stem,
              _caption(f"GPU call cost in nanoseconds per element of "
                       f"{esc(LIB_CUDA)} under each transfer strategy at "
                       f"$N={_grouped(N)}$, host-device transfers included.",
                       "The rows give the transfer strategies and the columns "
                       "the size-scaling datasets of each argument convention.",
                       MEDIAN_COST, _obtained(arch, gpu=True)),
              "tab:gpu-modes", head,
              align=wrapcol("2.8cm") + "r" * len(pairs),
              header_groups=groups)
    for m in modes:
        cells = [text(m)]
        for f, l in pairs:
            v = [_f(r, "ns_med") for r in glad if r.get("backend") == m
                 and r.get("family") == f and r.get("law") == l
                 and _i(r, "n") == N]
            cells.append(num(v[0]) if v else DASH)
        t.add(cells)
    t.provenance(
        runs_note,
        r"Cost only: the %d strategies return bit-identical results, which the "
        r"benchmark verifies. Call scope, so the numbers include the "
        r"host-device transfers." % len(modes),
        _machine(arch))
    return t.write(outdir)


# Levin band routing
def tab_levin(lev, outdir, runs_note, arch, device="cpu", backend="avx512"):
    """Maximum ULP and `ns_med` with the Levin band included and excluded, on one
    backend (the scalar and vector routers gate the band differently)."""
    stem = f"tab-levin-{device}-{backend}" if device == "cpu" else \
           f"tab-levin-{device}"
    if not lev:
        return _skip(stem, "no Levin A/B rows -- that stage is opt-in at run "
                           "time and was not run")
    # `lev` is levin.csv or gpu_levin.csv; `backend` is a library backend or a
    # transfer mode accordingly.
    rows = [r for r in lev if r.get("backend") == backend]
    on = {(r["family"], r["law"]): r for r in rows if r.get("variant") == "levin_on"}
    off = {(r["family"], r["law"]): r for r in rows if r.get("variant") == "levin_off"}
    both = sorted(set(on) & set(off))
    if not both:
        return _skip(stem, f"no (law, convention) cell carries both arms on "
                           f"backend {backend}")
    # Only the datasets on which an accuracy column differs (pf.levin_reaches).
    keys = [k for k in both if pf.levin_reaches(on[k], off[k])]
    omitted = [k for k in both if k not in keys]
    if not keys:
        return _skip(stem, f"no dataset differs between the arms in an accuracy "
                           f"column on backend {backend}")
    shipped = {r.get("variant") for r in rows if r.get("shipped") == "yes"}
    ship = "included" if "levin_on" in shipped else \
           ("excluded" if "levin_off" in shipped else "unrecorded")
    head = ["convention / dataset", "included", "excluded", "ratio",
            "included", "excluded", "ratio"]
    on_what = (r"the \texttt{" + esc(backend) + r"} backend" if device == "cpu"
               else r"the CUDA backend with the \texttt{" + esc(backend)
               + r"} transfer strategy")
    t = Table(stem,
              _caption("Maximum ULP error and runtime cost per element with the "
                       "Levin band included and excluded, on " + on_what
                       + ", by argument convention, for the datasets on which "
                       "the two builds differ in an accuracy statistic.",
                       ("In the default build of this backend the band is %s."
                        % ship) if ship != "unrecorded" else "",
                       "Costs are medians over the timed repetitions and "
                       "include the host-device transfers."
                       if device != "cpu" and all(scope_of(r) == "call"
                                                  for r in rows) else MEDIAN_COST,
                       _obtained(arch, gpu=device != "cpu")),
              f"tab:levin-{device}-{backend}".replace("_", "-"),
              head, align=wrapcol("3.0cm") + "rrrrrr",
              header_groups=[(2, 4, "max ULP error"), (5, 7, "ns / element")])
    moved = 0
    for f, l in keys:
        a, b = on[(f, l)], off[(f, l)]
        ua, ub = _f(a, "u_max"), _f(b, "u_max")
        ca, cb = _f(a, "ns_med"), _f(b, "ns_med")
        if _ok(ua) and _ok(ub) and ua != ub:
            moved += 1
        t.add([raw(f"{FAMILY_LABEL.get(f, f)} / {LAW_HEAD.get(l, l)}",
                   f"{f} / {l}"),
               ulp(a), ulp(b),
               ratio(ub / ua) if _ok(ua) and ua > 0 and _ok(ub) else DASH,
               num(ca), num(cb),
               ratio(cb / ca) if _ok(ca) and ca > 0 and _ok(cb) else DASH])
    t.note = _note(
        r"ratio: the excluded value divided by the included value. "
        + marker_note([on[k] for k in keys] + [off[k] for k in keys]))
    t.provenance(
        runs_note,
        r"A ratio above one favours including the band.",
        (r"Omitted, with every accuracy statistic identical in both builds: "
         + ", ".join(f"{esc(f)} / {esc(l)}" for f, l in omitted) + ".")
        if omitted else "",
        (f"The worst case differs on {moved} of {len(keys)} datasets.")
        if moved < len(keys) else "",
        r"Costs are medians over repetitions.",
        _machine(arch))
    return t.write(outdir)


# The prefactor folded into the kernels or applied outside
FOLD_SHOWN = ("scalar", "scalar-unfolded-direct", "scalar-unfolded-explog")


def fold_agreement(per_machine, family):
    """`machine_agreement` over the fold rows `tab_fold_quantiles` shows, named
    by backend; `per_machine` is [(label, fold rows)]."""
    shown = []
    for lab, rows in per_machine:
        sel = [dict(r, supplier=r.get("backend")) for r in rows
               if r.get("family") == family and r.get("backend") in FOLD_SHOWN
               and r.get("law") in ACC_LAWS]
        shown.append((lab, sel))
    return machine_agreement(shown, fields=QUANTILE_FIELDS)


def tab_fold_quantiles(fold, outdir, runs_note, family="z_nu", agreement=None,
                       arch=""):
    """`tab_quantiles` for the folded scalar backend and its unfolded twin,
    without the share below 1 ulp."""
    stem = f"tab-fold-quantiles-{family}"
    rows = [r for r in fold if r.get("family") == family]
    if not rows:
        return _skip(stem, "no prefactor-fold rows -- that stage is opt-in at "
                           "run time and was not run")
    laws = laws_present(rows)
    series = [(style.FOLD_LABEL[a], {r["law"]: r for r in rows
                                     if r.get("backend") == a})
              for a in FOLD_SHOWN if any(r.get("backend") == a for r in rows)]
    if not laws or len(series) < 2 or series[0][0] != style.FOLD_LABEL["scalar"]:
        return _skip(stem, "needs the folded scalar backend and a twin on a "
                           "shared law")
    t, missing = _quantile_table(
        stem,
        _caption("Distribution of the ULP error over all scored points of the "
                 "scalar backend with the Mat\\'ern prefactor folded into the "
                 "kernels and of its unfolded twin, for " + _sets(family) + ".",
                 _obtained(arch)),
        f"tab:fold-quantiles-{family.replace('_', '-')}", series, laws,
        "backend", share=False)
    nf = sum(1 for _, b in series for l, r in b.items()
             if l in laws and _i(r, "nonfinite") > 0)
    t.note = _note(_quantile_note(series, laws, missing, share=False)
                   + pf.FOLD_TWIN_SENTENCE)
    t.provenance(
        runs_note, *_quantile_facts(missing, nf, share=False),
        r"The control backend of the stage is not shown: its accuracy equals "
        r"the scalar backend's, which the stage checks. Measured; no bound is "
        r"implied.",
        pf._FOLD_TWIN_DETAIL,
        _agreement_phrase(agreement), _machine(arch))
    return t.write(outdir)


def tab_fold_cpu(fold, outdir, runs_note, arch):
    """Maximum ULP and `ns_med` per law of the folded scalar backend, its unfolded
    twin and the control; cost ratios divide by the folded backend's."""
    stem = "tab-fold-cpu"
    if not fold:
        return _skip(stem, "no prefactor-fold rows -- that stage is opt-in at "
                           "run time and was not run")
    by = {(r.get("family"), r.get("law"), r.get("backend")): r for r in fold}
    keys = []
    for f in ("z_nu", "h_nu_s"):
        laws = [l for l in ACC_LAWS if (f, l, "scalar") in by]
        laws += sorted({l for (ff, l, b) in by if ff == f and b == "scalar"}
                       - set(laws))
        keys += [(f, l) for l in laws]
    if not keys:
        return _skip(stem, "no prefactor-fold rows of the scalar backend")
    head = ["convention / dataset", "folded", "direct", "exp-log",
            "folded", "direct", "exp-log", "control"]
    t = Table(stem,
              # The statistic is in the first sentence: the float is near the page height.
              _caption("Maximum ULP error and median cost per element of the "
                       "scalar backend with the Mat\\'ern prefactor folded into "
                       "the kernels, of its unfolded twin and of the control "
                       "build, by argument convention and dataset.",
                       _obtained(arch)),
              "tab:fold-cpu", head, align=wrapcol("3.0cm") + "rrrrrrr",
              header_groups=[(2, 4, "max ULP error"),
                             (5, 8, "ns / element; twins and control as ratio")])
    shown = []
    for f, l in keys:
        a = by[(f, l, "scalar")]
        d = by.get((f, l, "scalar-unfolded-direct"))
        e = by.get((f, l, "scalar-unfolded-explog"))
        c = by.get((f, l, "scalar-control"))
        shown += [x for x in (a, d, e) if x]
        ca = _f(a, "ns_med")

        def rat(x):
            v = _f(x, "ns_med") if x else float("nan")
            return ratio(v / ca) if _ok(v) and _ok(ca) and ca > 0 else DASH

        t.add([raw(f"{FAMILY_LABEL.get(f, f)} / {LAW_HEAD.get(l, l)}",
                   f"{f} / {l}"),
               ulp(a), ulp(d) if d else DASH, ulp(e) if e else DASH,
               num(ca), rat(d), rat(e), rat(c)])
    t.note = _note(
        r"Cost ratios: cost divided by that of the folded backend. Control: "
        r"the folded source built like the twin. " + pf.FOLD_TWIN_SENTENCE + " "
        + marker_note(shown))
    t.provenance(
        runs_note,
        r"The cost ratios divide each backend's median cost by the folded "
        r"backend's on the same dataset, so a ratio above one means slower than "
        r"the folded backend. The control is the shipped scalar source built "
        r"the way the twin is; its ratio shows how far the build and the code "
        r"placement alone move a cost.",
        pf._FOLD_TWIN_DETAIL,
        r"Costs are medians over repetitions.",
        _machine(arch))
    return t.write(outdir)


# Thread scaling
def _why_no_suppliers(thr, family):
    """README sentence on why the externals are not in the efficiency table,
    with the run's efficiency ranges."""
    rows = [r for r in (thr or []) if r.get("family") == family
            and _i(r, "threads") > 1 and _ok(_f(r, "efficiency"))]
    if not rows:
        return ""
    sup = [r for r in rows if (r.get("supplier") or "-") != "-"]
    lib = [r for r in rows if (r.get("supplier") or "-") == "-"]
    if not sup or not lib:
        return ""
    se = [100.0 * _f(r, "efficiency") for r in sup]
    le = [100.0 * _f(r, "efficiency") for r in lib]
    worst = min(lib, key=lambda r: _f(r, "efficiency"))
    if min(le) > min(se):          # the premise does not hold on this run
        return (r"The external suppliers are timed in this stage but are not "
                r"listed here, because efficiency is a ratio against each row's "
                r"own one-thread cost; their throughput is in "
                r"\texttt{tab-threads-throughput}.")
    return (r"The external suppliers are timed in this stage but are not listed "
            r"here, because efficiency is a ratio against each row's own "
            r"one-thread cost, which is higher for a slower row. On this run the "
            r"externals span %.1f--%.1f\,\%% and this library spans "
            r"%.1f--%.1f\,\%%; the lowest efficiency is this library's "
            r"\texttt{%s} at %.1f\,\%%%s. The throughput of all rows is in "
            r"\texttt{tab-threads-throughput}."
            % (min(se), max(se), min(le), max(le),
               esc(worst.get("backend", "?")), min(le),
               _clock_clause(thr, worst, family)))


def _clock_clause(thr, worst, family):
    """The clock drop (`mhz_end`) from the one-thread step to `worst`, or '' unless
    it is at least half the efficiency loss."""
    def same(r):
        return (r.get("family") == family
                and (r.get("supplier") or "-") == (worst.get("supplier") or "-")
                and r.get("backend") == worst.get("backend"))
    one = [r for r in (thr or []) if same(r) and _i(r, "threads") == 1]
    if not one:
        return ""
    a, b = _f(one[0], "mhz_end"), _f(worst, "mhz_end")
    eff = _f(worst, "efficiency")
    if not (_ok(a) and _ok(b) and _ok(eff)) or a <= 0 or b <= 0 or b >= a:
        return ""
    drop, loss = 1.0 - b / a, 1.0 - eff
    if loss <= 0 or drop < 0.5 * loss:
        return ""
    return (r"; over the same steps the recorded clock of its cores fell "
            r"from %.0f to %.0f\,MHz (one reading per step), which alone "
            r"accounts for a %.1f\,\%% cost increase of the %.1f\,\%% measured"
            % (a, b, 100.0 * (a / b - 1.0), 100.0 * (1.0 / eff - 1.0)))


def tab_threads(thr, outdir, runs_note, arch, family="z_nu"):
    """Cost and parallel efficiency of the library backends against pinned
    thread count; the externals are in `tab_threads_throughput`."""
    stem = "tab-threads"
    if not thr:
        return _skip(stem, "no thread rows -- the scaling stage did not run")
    # Median over repetitions; a run without medians is not tabulated.
    thr, _thr_med = pf.threads_median(thr)
    if not _thr_med:
        return _skip(stem, pf.THREADS_NO_MEDIAN)
    # Library rows only; supplier rows carry the assembly in `backend`.
    rows = [r for r in thr if r.get("family") == family
            and (r.get("supplier") or "-") == "-"]
    if not rows:
        return _skip(stem, f"the scaling stage carries no {family} library rows")
    backends = [b for b in BACKEND_ORDER
                if any(r.get("backend") == b for r in rows)]
    counts = sorted({_i(r, "threads") for r in rows if _i(r, "threads")})
    if not backends or len(counts) < 2:
        return _skip(stem, f"{len(counts)} thread count(s) measured -- scaling "
                           f"needs at least two")
    by = {(r["backend"], _i(r, "threads")): r for r in rows}
    nb = len(backends)
    # One cost group: `ns_per_elem` includes workspace creation on avx512 and avx2
    # (threads_cpu.c).
    ng = 2
    head = ["threads"] + [esc(b) for b in backends] * ng
    groups = [(2, 1 + nb, "ns / element"),
              (2 + nb, 1 + 2 * nb, "efficiency")]
    t = Table(stem,
              _caption("Runtime cost per element and parallel efficiency of the "
                       "library backends against the number of pinned threads "
                       "(weak scaling), with " + pf.CONV.get(family, esc(family))
                       + " supplied.", _obtained(arch)),
              "tab:threads", head, align="r" + "r" * (ng * nb),
              header_groups=groups)
    for c in counts:
        cells = [count(c)]
        for b in backends:
            r = by.get((b, c))
            cells.append(num(_f(r, "ns_per_elem")) if r else DASH)
        for b in backends:
            r = by.get((b, c))
            e = _f(r, "efficiency") if r else float("nan")
            cells.append(Cell(style.Raw(f"{e * 100:.0f}\\%"), f"{e * 100:.0f}%",
                              repr(e)) if _ok(e) else DASH)
        t.add(cells)
    t.provenance(
        runs_note,
        r"Each thread evaluates its own array of the same size, so a perfectly "
        r"scaling implementation holds the cost per element flat and the "
        r"efficiency at 100\%. Cost is the median over repetitions of the "
        r"slowest thread's time in each (repetitions start together on a "
        r"barrier).",
        # Result identity and ULP invariance across thread counts.
        pf.thread_integrity(thr, family)[0].strip(),
        r"On the \texttt{avx512} and \texttt{avx2} backends the cost includes the "
        r"workspace: every thread creates and destroys its own inside the timed "
        r"region on every repetition, and all threads allocate at the same time, "
        r"so the cost includes allocator contention. The other backends and the "
        r"external suppliers allocate no workspace.",
        _why_no_suppliers(thr, family),
        _machine(arch))
    return t.write(outdir)


def tab_threads_throughput(thr, outdir, runs_note, arch, family="z_nu"):
    """Values per second summed over the threads against thread count, library
    and suppliers (the reciprocal of the aggregate `ns_per_elem`)."""
    stem = "tab-threads-throughput"
    if not thr:
        return _skip(stem, "no thread rows -- the scaling stage did not run")
    # Median over repetitions; a run without medians is not tabulated.
    thr, _thr_med = pf.threads_median(thr)
    if not _thr_med:
        return _skip(stem, pf.THREADS_NO_MEDIAN)
    rows = [r for r in thr if r.get("family") == family
            and _ok(_f(r, "values_per_s"))]
    if not rows:
        return _skip(stem, "the scaling stage predates the values_per_s column; "
                           "re-run it to get this table")
    counts = sorted({_i(r, "threads") for r in rows if _i(r, "threads")})
    if len(counts) < 2:
        return _skip(stem, "scaling needs at least two thread counts")

    # One column per library backend or supplier.
    def key(r):
        sup = r.get("supplier") or "-"
        return f"{LIB} ({r.get('backend')})" if sup == "-" else sup
    series, seen = [], set()
    for b in BACKEND_ORDER:                       # the library first, in order
        k = f"{LIB} ({b})"
        if any(key(r) == k for r in rows) and k not in seen:
            series.append(k); seen.add(k)
    for k in sorted({key(r) for r in rows} - seen):
        series.append(k)
    if len(series) < 2:
        return _skip(stem, "only one row present -- nothing to compare")

    by = {(key(r), _i(r, "threads")): r for r in rows}

    # Spanning headers `libmatern` and `external`; heads name backend or supplier.
    lib_s = [k for k in series if k.startswith(LIB + " (")]
    ext_s = [k for k in series if k not in lib_s]
    short = {k: k[len(LIB) + 2:-1] for k in lib_s}      # "libmatern (avx512)" -> "avx512"
    # External headings drop the version suffix; the note gives the full names.
    for k in ext_s:
        short[k] = {"std-cyl_bessel_k": "std"}.get(k, k.rsplit("-", 1)[0]
                                                   if k.rsplit("-", 1)[-1][:1].isdigit() else k)
    head = ["$T$"] + [esc(short.get(k, k)) for k in lib_s + ext_s]
    series = lib_s + ext_s
    groups = []
    if lib_s:
        groups.append((2, 1 + len(lib_s), LIB))
    if ext_s:
        groups.append((2 + len(lib_s), 1 + len(series), "external"))
    t = Table(stem,
              _caption("Covariance values per second summed over the threads, "
                       "against the number of pinned threads, with "
                       + pf.CONV.get(family, esc(family)) + " supplied.",
                       _obtained(arch)),
              "tab:threads-throughput", head,
              align="r" + "r" * len(series),
              header_groups=groups)
    for c in counts:
        cells = [count(c)]
        for k in series:
            r = by.get((k, c))
            cells.append(num(_f(r, "values_per_s"), 3) if r else DASH)
        t.add(cells)

    # Last row: ratios at the widest thread count to the fastest library backend.
    top = counts[-1]
    ref, base = None, float("nan")
    for k in [k for k in series if k.startswith(LIB)]:
        v = _f(by.get((k, top)), "values_per_s") if (k, top) in by else float("nan")
        if _ok(v) and (not _ok(base) or v > base): ref, base = k, v
    if ref and _ok(base) and base > 0:
        cells = [raw(r"$\times$ " + esc(ref), "x " + ref)]
        for k in series:
            v = _f(by.get((k, top)), "values_per_s") if (k, top) in by else float("nan")
            cells.append(ratio(base / v) if (_ok(v) and v > 0) else DASH)
        t.add(cells)
        t.rules.add(len(t.rows) - 2)

    # Map the shortened headings back to the full supplier names.
    _named = ", ".join(r"\texttt{%s}: \texttt{%s}" % (esc(short[k]), esc(k))
                       for k in ext_s if short[k] != k)
    t.note = _note(
        r"$T$: number of pinned threads. "
        + ((r"Last row: the values per second of " + esc(ref)
            + r" at the largest $T$ divided by those of each column. ")
           if ref and _ok(base) and base > 0 else "")
        + ((r"Shortened headings: " + _named + ". ") if _named else ""))
    t.provenance(
        runs_note,
        r"Each cell is the number of values per second summed over the "
        r"threads, so a row that scales perfectly doubles when the thread "
        r"count doubles. The time is the median over repetitions of the "
        r"slowest thread's time in each (repetitions start together on a "
        r"barrier).",
        r"Parallel efficiency (\texttt{tab-threads}) is measured against each "
        r"row's own one-thread cost, so a row that is slower per element asks "
        r"less of the shared hardware and can reach a higher efficiency at a "
        r"lower throughput.",
        r"Every external is timed in the \texttt{direct} assembly here, unlike "
        r"the throughput table, where each supplier is shown in the assembly "
        r"selected from its accuracy rows: the size datasets carry timing only, "
        r"and \texttt{direct} is the assembly every supplier supports.",
        r"The reference of the last row is the fastest library backend at the "
        r"widest thread count (\texttt{scalar} is never selected "
        r"automatically).",
        _machine(arch))
    return t.write(outdir)


# Preview: one compiled document per directory, whose overfull boxes drive
# fit_to_column.  Its class and geometry must equal the manuscript's, so that a
# table that fits the preview fits the paper.
PREVIEW_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
% The manuscript's class and geometry.
\usepackage{geometry}
\geometry{left=27mm,right=27mm,top=25mm,bottom=26mm}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{rotating}
% Every table is placed where it is \input, overriding the `[t]` the fragments
% carry, so that each table stays under its filename heading.
% `\floatplacement{table}{H}` only sets the default, which an explicit `[t]`
% overrides; the environment is therefore redefined to discard the optional
% argument and use `H`.
\makeatletter
\let\savedtable\table
\let\savedendtable\endtable
\renewenvironment{table}[1][]{\savedtable[H]}{\savedendtable}
\makeatother
\setlength{\parindent}{0pt}
\begin{document}
"""


def _build_preview(outdir):
    r"""Run pdflatex twice in `outdir`, so that references resolve before widths
    are read; return (log text, pdf exists)."""
    import subprocess, shutil
    engine = shutil.which("pdflatex")
    if not engine:
        return "", False
    for _pass in (1, 2):
        subprocess.run([engine, "-interaction=nonstopmode", "preview.tex"],
                       cwd=outdir, capture_output=True, text=True)
    logf = os.path.join(outdir, "preview.log")
    log = open(logf, errors="replace").read() if os.path.exists(logf) else ""
    return log, os.path.exists(os.path.join(outdir, "preview.pdf"))


# TeX breaks log lines at max_print_line, 79 characters, so a fragment's name can be split
# over two lines ("(./tab-" / "environment.tex"); the box after it would then go to the
# fragment before.  The breaks are undone before the log is read.
_LOG_WIDTH = 79


def _unwrap(log):
    """`log` with TeX's line breaks at _LOG_WIDTH characters undone."""
    return re.sub(r"(?m)^(.{%d})\n" % _LOG_WIDTH, r"\1", log)


_OPENS = re.compile(r"\((\./)?([-\w.]+\.tex)")
_OVER = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\)")
# A float taller than the page; write_preview ships each page before the next
# fragment opens, so the open-file attribution applies.
_TALL = re.compile(r"Overfull \\vbox \(([\d.]+)pt too high\)")


def overtall_by_file(log, outdir):
    """{abs path of fragment: worst excess height in pt}: `overfull_by_file`
    for tables that are too tall."""
    cur, worst = None, {}
    for line in _unwrap(log).splitlines():
        for m in _OPENS.finditer(line):
            if m.group(2) != "preview.tex":
                cur = os.path.abspath(os.path.join(outdir, m.group(2)))
        m = _TALL.search(line)
        if m and cur:
            worst[cur] = max(worst.get(cur, 0.0), float(m.group(1)))
    return worst


def overfull_by_file(log, outdir):
    """{abs path of fragment: worst overfull in pt}; TeX does not name the file,
    so each box goes to the most recently opened `.tex`."""
    cur = None
    worst = {}
    for line in _unwrap(log).splitlines():
        for m in _OPENS.finditer(line):
            fn = m.group(2)
            if fn != "preview.tex":
                cur = os.path.abspath(os.path.join(outdir, fn))
        m = _OVER.search(line)
        if m and cur:
            worst[cur] = max(worst.get(cur, 0.0), float(m.group(1)))
    return worst


def forms(outdir):
    """[(file, size, rotated)] per table in `outdir`, as fit_to_column set it."""
    out = []
    for path, t in sorted(BUILT.items()):
        if os.path.dirname(os.path.abspath(path)) != os.path.abspath(outdir):
            continue
        size, _cs, env = SIZE_LADDER[min(t.level, len(SIZE_LADDER) - 1)]
        out.append((os.path.basename(path), size.lstrip("\\"),
                    env != "table"))
    return out


def fit_to_column(outdir, rounds=None):
    r"""Step each overflowing table in `outdir` one rung down SIZE_LADDER per
    preview build; return [(name, pt over, size)] still too wide at the bottom."""
    rounds = len(SIZE_LADDER) if rounds is None else rounds
    left = []
    for _r in range(rounds):
        log, _pdf = _build_preview(outdir)
        if not log:
            return []
        worst = overfull_by_file(log, outdir)
        # Too tall: upright rungs only, since rotation does not reduce height.
        tall = 0
        for path in overtall_by_file(log, outdir):
            t = BUILT.get(path)
            if path in worst or t is None:
                continue            # the width step below shrinks it anyway
            if t.level + 1 < len(SIZE_LADDER) \
                    and SIZE_LADDER[t.level + 1][2] == "table":
                t.shrink(path)
                tall += 1
        if not worst:
            if tall:
                continue
            return []
        left, shrank = [], tall
        for path, pt in sorted(worst.items(), key=lambda kv: -kv[1]):
            t = BUILT.get(path)
            if t is not None and t.shrink(path):
                shrank += 1
            else:
                left.append((os.path.basename(path), pt,
                             SIZE_LADDER[t.level][0] if t else "?"))
        if not shrank:
            break
    return left


def write_preview(outdir, title, subtitle="", run=True):
    r"""Write `preview.tex` in `outdir` and, with pdflatex, `preview.pdf`; return
    (tex, pdf or None, [tables still overfull, and the first LaTeX error])."""
    import shutil
    names = sorted(f for f in os.listdir(outdir)
                   if f.endswith(".tex") and f != "preview.tex")
    if not names:
        return None, None, []
    body = [r"\section*{%s}" % style.tex_escape(title)]
    if subtitle:
        body.append(r"{\small %s}\par\medskip" % subtitle)
    body.append(r"{\small %d table(s) in this directory. Each is shown under "
                r"the file that produces it, at the manuscript's column width, "
                r"so a table that does not fit is visible here.}\par"
                % len(names))
    for fn in names:
        # `\clearpage` flushes every float (a `sidewaystable` ignores `[H]`) before
        # the next file opens, so overfull_by_file attributes each box correctly.
        body += [r"\bigskip\hrule\medskip",
                 r"\texttt{\small %s}\par\smallskip" % style.tex_escape(fn),
                 r"\input{%s}" % fn[:-4],
                 r"\clearpage"]
    open(os.path.join(outdir, "preview.tex"), "w").write(
        PREVIEW_PREAMBLE + "\n".join(body) + "\n\\end{document}\n")
    tex = os.path.join(outdir, "preview.tex")
    WRITTEN.append(tex)

    engine = shutil.which("pdflatex")
    if not (run and engine):
        return tex, None, []
    # Remove any previous PDF first, so that a failed build leaves no PDF.
    _pdf = os.path.join(outdir, "preview.pdf")
    if os.path.exists(_pdf):
        os.remove(_pdf)
    # Fit first, then build the kept copy: `fit_to_column` re-emits fragments.
    over = fit_to_column(outdir)
    log, _has = _build_preview(outdir)
    errs = [l for l in log.splitlines() if l.startswith("!")]
    if errs:
        over = list(over) + [("error", errs[0])]
    # Remove the auxiliary files so the stale-file check does not report them.
    for ext in (".aux", ".log", ".out"):
        p = os.path.join(outdir, "preview" + ext)
        if os.path.exists(p):
            os.remove(p)
    pdf = os.path.join(outdir, "preview.pdf")
    if os.path.exists(pdf):
        WRITTEN.append(pdf)
        return tex, pdf, over
    return tex, None, over
