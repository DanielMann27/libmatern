"""style.py -- shared typography, colours, booktabs emitter and paper-mode caption
assembly for the paper figures and tables."""

# Backends on a sequential ramp in attribution order; `matched_fma` (`matched` built with
# -mfma) sits between `matched` and `avx2`.
BACKEND_ORDER = ["scalar", "matched", "matched_fma", "avx2", "avx512"]
BACKEND_COLOR = {
    "scalar":      "#08306b",
    "matched":     "#2171b5",
    "matched_fma": "#4292c6",
    "avx2":        "#6baed6",
    "avx512":      "#c1121f",   # the default backend, in the library red
}
BACKEND_MARK = {"scalar": "D", "matched": "s", "matched_fma": "P",
                "avx2": "^", "avx512": "o"}

# CUDA transfer modes, bit-identical by contract, so one hue family; keys as benchmark_gpu
# writes them.
MODE_COLOR = {"sync": "#08306b", "async-direct": "#2171b5",
              "async-ring": "#6baed6", "async-mapped": "#9ecae1"}


def mode_color(name):
    """Colour of a transfer mode; case and '_'/'-' insensitive."""
    k = str(name).strip().lower().replace("_", "-")
    return MODE_COLOR.get(k, "#b5179e")

# The library's display names; LIB_CUDA is also its supplier label in gpu_kernels.csv
# (suppliers_gpu.cu).  runs.py maps recorded labels through LEGACY_SUPPLIER and
# LEGACY_FAMILY.
LIB = "libmatern"
LIB_CUDA = "libmatern-cuda"
# legacy name: supplier labels written by runs made before the rename
LEGACY_SUPPLIER = {"libwhittle": LIB, "libwhittle-cuda": LIB_CUDA}  # legacy name
# legacy name: family value of recorded runs, read as `h_nu_s`
LEGACY_FAMILY = {"h_nu_rho": "h_nu_s"}  # legacy name

SUPPLIER_COLOR = {
    LIB:                 "#c1121f",
    "bessels.jl":        "#0353a4",
    "bessels.jl-old":    "#5fa8d3",
    "kaust-128":         "#2a9d8f",
    "kaust-40":          "#8ab17d",
    "kaust-adaptive":    "#e9c46a",
    "std-cyl_bessel_k":  "#7b2cbf",
    # Explicit, so boost and gsl do not share the magenta fallback colour.
    "boost":             "#386641",
    "gsl":               "#e76f51",
    # supplier_color() maps every `boost-...-adapted` label (nu ~ 5/2 workaround) here.
    "boost-adapted":     "#90be6d",
}
# The prefactor-fold stage (fold.csv), keyed by its `backend` column.
FOLD_ARMS = ["scalar", "scalar-control", "scalar-unfolded-direct",
             "scalar-unfolded-explog"]
FOLD_LABEL = {
    "scalar":                 "Folded",
    "scalar-control":         "Control",
    "scalar-unfolded-direct": "Unfolded, direct",
    "scalar-unfolded-explog": "Unfolded, exp-log",
}
FOLD_COLOR = {"scalar": "#c1121f", "scalar-control": "#c1121f",
              "scalar-unfolded-direct": "#2b2b2b",
              "scalar-unfolded-explog": "#8a8a8a"}
FOLD_MARK = {"scalar": "o", "scalar-control": "o",
             "scalar-unfolded-direct": "^", "scalar-unfolded-explog": "s"}
# Marker sizes (pt): a triangle looks smaller than a circle or square of the same size,
# so it is drawn largest.
FOLD_MS = {"scalar": 4.0, "scalar-control": 4.0,
           "scalar-unfolded-direct": 6.0, "scalar-unfolded-explog": 5.0}
FAMILY_MARK = {"z_nu": "o", "h_nu_s": "s"}

ULP_ONE  = 1.0

# Figures are drawn at the width they are printed at, so a type size here is the size on
# the page.  LATEX_LINEWIDTH is the paper's \linewidth in inches: 156 mm, the width of A4
# less the two 27 mm side margins.  TEXT_W is 0.08 in under it, leaving room for the tight
# bounding box.  TEXT_H_MAX leaves room for a caption under a full-page figure.
LATEX_LINEWIDTH = 6.14
TEXT_W          = 6.06
TEXT_H_MAX      = 8.3

# The type scale, in points on the page; size_report flags text under FS_MIN.
FS_TICK   = 7.5
FS_LABEL  = 8.5
FS_TITLE  = 8.5
FS_LEGEND = 8.0
FS_ANNOT  = 7.0
FS_MIN    = 7.0


def apply(plt):
    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE, "figure.titlesize": FS_TITLE,
        "legend.fontsize": FS_LEGEND,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.5,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def supplier_color(name):
    if name in SUPPLIER_COLOR:
        return SUPPLIER_COLOR[name]
    for k, lab in FOLD_LABEL.items():
        if name == lab:
            return FOLD_COLOR[k]
    if name.startswith("boost") and name.endswith("-adapted"):
        return SUPPLIER_COLOR["boost-adapted"]
    # boost-1.87 -> boost, and LIB_CUDA -> LIB (the library's colour)
    for k, v in SUPPLIER_COLOR.items():
        if name.startswith(k.split("-")[0]):
            return v
    return "#b5179e"


# Every path written by save(), in order: the driver deletes nothing, so this is the record
# of what a run produced.
SAVED = []


# Backend names display as AVX2/AVX-512 (applied at save time); whole words only, so
# `matched_fma` and paths are untouched.
import re as _re
_DISPLAY_CAPS = [(_re.compile(r"(?<![A-Za-z0-9_./-])avx512(?![A-Za-z0-9_])"), "AVX-512"),
                 (_re.compile(r"(?<![A-Za-z0-9_./-])avx2(?![A-Za-z0-9_])"), "AVX2")]


def display_caps(text):
    for rx, rep in _DISPLAY_CAPS:
        text = rx.sub(rep, text)
    return text


# Supplier names as displayed in figures, captions and tables; the records and every lookup
# (colours, footnotes, rankings) keep the recorded names.  The adjusted configurations
# (`kaust-adaptive`, `boost-...-adapted`) display as `-modified`, and no name carries a
# version: the environment table states the versions.
# One pass, so `kaust-adaptive/boost-1.75.0` (two names in one cell) maps both.  Versioned
# names never occur inside paths in the outputs (module paths read `boost/1.87.0`).
_VER = r"-\d+(?:\.\d+)+"
_DISPLAY_NAME = _re.compile(
    r"(?<![A-Za-z0-9_.])(?:"
    r"(?P<kaust>kaust-adaptive)(?![A-Za-z0-9_])"
    r"|(?P<kaust_short>kaust-ad\.)"
    r"|(?P<boost_mod>boost(?:-gpu)?)(?:" + _VER + r")?-adapted(?![A-Za-z0-9_])"
    r"|(?P<boost_short>boost(?:-gpu)?)(?:" + _VER + r")?-ad\."
    r"|(?P<versioned>boost(?:-gpu)?|gsl)" + _VER + r"(?!\.?[0-9]))")


def _display_name(m):
    if m.group("kaust"):
        return "kaust-modified"
    if m.group("kaust_short"):
        return "kaust-mod."
    if m.group("boost_mod"):
        return m.group("boost_mod") + "-modified"
    if m.group("boost_short"):
        return m.group("boost_short") + "-mod."
    return m.group("versioned")


def display_names(text):
    """Supplier names as displayed: the adjusted configurations as `-modified` (`-mod.`),
    no version in any name (the environment table states the versions)."""
    return _DISPLAY_NAME.sub(_display_name, text)


def display_text(text):
    """Every display form: AVX2 / AVX-512 and the supplier names."""
    return display_names(display_caps(text))


def tick_labels(labels):
    """Tick labels in their display form (display_text); set_[xy]ticklabels takes these,
    since a tick label is drawn from its formatter, not from a Text object kept until save."""
    return [display_text(str(s)) for s in labels]


def apply_display_caps(fig):
    from matplotlib.text import Text
    # Tick labels are drawn anew from their formatter at save time, so labels given to
    # set_[xy]ticklabels pass through tick_labels() where they are set.
    for t in fig.findobj(Text):
        s = t.get_text()
        if s:
            d = display_text(s)
            if d != s:
                t.set_text(d)


def save(fig, outdir, stem):
    """Write `fig` in `outdir` as `<stem>.pdf`, the file the manuscript includes, and record
    its printed size (check_size)."""
    import os
    apply_display_caps(fig)
    path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(path)
    SAVED.append(path)
    check_size(fig, path)
    return 1


# [(pdf path, printed width in, printed height in, smallest printed pt, its text, scale)] of
# every figure saved, filled by check_size, reported by make_paper_figures.py.
SIZES = []


def check_size(fig, pdf):
    """Record the size `pdf` prints at when included at \\linewidth, and its smallest text.

    The size is read from the saved page, which a tight bounding box crops or grows
    relative to `figsize`.
    LaTeX scales the page to LATEX_LINEWIDTH, so every type size is scaled by the same
    factor."""
    import re
    from matplotlib.text import Text
    with open(pdf, "rb") as fh:
        m = re.search(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)",
                      fh.read())
    if not m:
        return
    w = (float(m.group(3)) - float(m.group(1))) / 72.0
    h = (float(m.group(4)) - float(m.group(2))) / 72.0
    scale = LATEX_LINEWIDTH / w
    r = fig.canvas.get_renderer()
    box = fig.bbox
    small, what = float("inf"), ""
    for t in fig.findobj(Text):
        s = t.get_text()
        if not (s and s.strip() and t.get_visible()):
            continue
        try:
            e = t.get_window_extent(r)
        except Exception:
            continue
        # Text outside the canvas (unused tick objects) is not printed.
        if e.x1 < box.x0 or e.x0 > box.x1 or e.y1 < box.y0 or e.y0 > box.y1:
            continue
        if t.get_fontsize() < small:
            small, what = t.get_fontsize(), s
    SIZES.append((pdf, w * scale, h * scale, small * scale, what, scale))


def size_report(root):
    """Lines describing the recorded sizes (check_size), paths relative to `root`; a line
    starts with `!!` when the figure is taller than TEXT_H_MAX or prints text under
    FS_MIN."""
    import os
    out = []
    for pdf, w, h, pt, what, scale in SIZES:
        bad = []
        if h > TEXT_H_MAX:
            bad.append(f"taller than {TEXT_H_MAX:g} in")
        if pt < FS_MIN:
            bad.append(f"text under {FS_MIN:g} pt: {what[:40]!r}")
        out.append("%s %-62s %4.2f x %4.2f in  x%.2f  smallest %4.1f pt%s"
                   % ("!!" if bad else "  ", os.path.relpath(pdf, root), w, h, scale,
                      pt, ("  (" + "; ".join(bad) + ")") if bad else ""))
    return out


def refline(ax, y, text, side="left", backdrop=False):
    """A horizontal reference line at `y`, labelled in the axes rather than the legend;
    `backdrop` lifts the label over the data on a white ground (for a line inside the
    data)."""
    ax.axhline(y, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=1)
    x, ha = (0.015, "left") if side == "left" else (0.985, "right")
    kw = (dict(zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
          if backdrop else {})
    ax.text(x, y, f" {text} ", transform=ax.get_yaxis_transform(),
            ha=ha, va="bottom", fontsize=FS_ANNOT, color="0.35", **kw)


class Raw(str):
    """A table cell already in LaTeX: `tex_escape` passes it through, elsewhere a plain `str`."""


def tex_escape(t):
    if isinstance(t, Raw):
        return str(t)
    t = str(t)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        t = t.replace(a, b)
    return t


def booktabs(path, caption, label, header, rows, align=None, note=None,
             header_groups=None, rules_after=None, env="table",
             placement="t", note_raw=False, preamble_note=None,
             size=None, colsep=None):
    """Write a booktabs table float to `path`; return the number of body rows.

    Body cells are escaped with `tex_escape`, header cells are not.  `header_groups` is
    [(first_col, last_col, title)], 1-based and inclusive; `rules_after` holds 0-based row
    indices followed by a \\midrule; `env` may name a rotated float (sidewaystable).
    """
    align = align or ("l" * 1 + "r" * (len(header) - 1))
    out = [r"% generated by benchmark/plots --- do not edit; regenerate instead"]
    if preamble_note:
        out.append(r"% requires: " + preamble_note)
    # `sidewaystable` takes no optional placement argument.
    out += [r"\begin{" + env + "}" + (f"[{placement}]" if placement else ""),
            r"  \centering"]
    _cap = [f"  \\caption{{{caption}}}", f"  \\label{{{label}}}"]
    # Type size and \tabcolsep are set inside the float, so they are scoped to this table.
    # In a rotated float \linewidth stays at \textwidth; a \textheight minipage makes it
    # the rotated width.
    rot = env != "table"
    if rot:
        out.append(r"  \begin{minipage}{\textheight}")
        out.append(r"  \centering")
    out += _cap
    if colsep:
        out.append(r"  \setlength{\tabcolsep}{%s}" % colsep)
    if size:
        out.append("  " + size)
    out += [f"  \\begin{{tabular}}{{{align}}}", r"    \toprule"]
    if header_groups:
        cells, rules, col = [], [], 1
        for first, last, title in header_groups:
            if first > col:
                cells.append(r"\multicolumn{%d}{c}{}" % (first - col))
            cells.append(r"\multicolumn{%d}{c}{%s}" % (last - first + 1, title))
            rules.append(r"\cmidrule(lr){%d-%d}" % (first, last))
            col = last + 1
        if col <= len(header):
            cells.append(r"\multicolumn{%d}{c}{}" % (len(header) - col + 1))
        out.append("    " + " & ".join(cells) + r" \\")
        out.append("    " + "".join(rules))
    # Header cells are not escaped: they may carry math ($N$, $\\times$).
    out += ["    " + " & ".join(str(h) for h in header) + r" \\",
            r"    \midrule"]
    # No \midrule after the last row, directly above \bottomrule.
    cut = set(rules_after or ())
    for i, r in enumerate(rows):
        out.append("    " + " & ".join(tex_escape(c) for c in r) + r" \\")
        if i in cut and i != len(rows) - 1:
            out.append(r"    \midrule")
    out += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        out.append(r"  \begin{minipage}{\linewidth}\footnotesize\vspace{2pt}"
                   + (str(note) if note_raw else tex_escape(note))
                   + r"\end{minipage}")
    if rot:
        out.append(r"  \end{minipage}")
    out += [r"\end{" + env + "}", ""]
    open(path, "w").write("\n".join(out))
    return len(rows)


# Keys and notes sit outside the axes, since the curves move between machines; the type
# sizes are fixed and a wide key gets more columns.
LEGEND_FONTSIZE = FS_LEGEND

# Must equal the fontsize of the run stamp in paper_figures._stamp.
STAMP_FONTSIZE = 5.2


def stamp_clearance(fig):
    """Figure-fraction height to leave free at the bottom for the run stamp."""
    return (STAMP_FONTSIZE * 1.6) / (fig.get_figheight() * 72.0) + 0.004


def _ncol_for(n, cap=4):
    """Legend columns for `n` entries, preferring wide and short keys."""
    if n <= cap:
        return max(1, n)
    return min(cap, (n + 1) // 2)


def legend_below(ax, y=-0.17, ncol=None, fontsize=LEGEND_FONTSIZE,
                 handles=None, labels=None, **kw):
    """One axes' key, under that axes (single-panel figures)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(handles, labels, frameon=False, fontsize=fontsize,
                     ncol=ncol or _ncol_for(len(handles)),
                     loc="upper center", bbox_to_anchor=(0.5, y), **kw)


def note_below(fig, text):
    """A figure note: recorded for write_hardware (HARDWARE.md), not drawn."""
    record_note(text)


def footer(fig, handles=None, labels=None, note=None, ncol=None,
           legend_fontsize=LEGEND_FONTSIZE, pad=0.012):
    """Lay the key out under the panels and return the figure-fraction height it occupies
    (for `tight_layout`); the note is recorded for write_hardware (HARDWARE.md)."""
    h_in = fig.get_figheight()
    y = stamp_clearance(fig)
    if note:
        record_note(note)
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=legend_fontsize,
                   ncol=ncol or _ncol_for(len(handles)),
                   loc="lower center", bbox_to_anchor=(0.5, y))
        rows = -(-len(handles) // (ncol or _ncol_for(len(handles))))
        y += rows * (legend_fontsize * 1.9) / (h_in * 72.0) + pad
    return y


# No titles or notes on the canvas: each figure's caption goes to captions.tex, its title,
# notes and facts to HARDWARE.md.

# Rewrites of CSV column names to their meaning in titles and captions.  They run before
# LaTeX escaping (plain underscores), and a longer pattern precedes any pattern it contains.
_PAPER_REWRITE = [
    ("u_fence, u_iqr, u_mean, u_q1, u_q3",
     "the outlier fence, the interquartile range, the mean and the two inner "
     "quartiles"),
    ("u_fence, u_iqr, u_mean, u_q1",
     "the outlier fence, the interquartile range, the mean and the lower quartile"),
    ("u_med_nontrivial", "the median over points with a nonzero reference"),
    ("u_iqr", "the interquartile range"), ("u_fence", "the outlier fence"),
    ("u_mean", "the mean"), ("u_q1", "the lower quartile"),
    ("u_q3", "the upper quartile"), ("u_med", "the median"),
]

# The current figure: title, notes and facts (HARDWARE.md), caption sentences (captions.tex)
# and run labels.
CAPTION = {"title": "", "notes": [], "text": [], "facts": [], "runs": []}

# {output directory: run labels of its figures}, returned by caption_finalise().
RUNS_SEEN = {}

# [(outdir, stem, runs, facts)] of every figure written.
FIGURES = []


def _rewrite(text):
    for a, b in _PAPER_REWRITE:
        text = text.replace(a, b)
    return text


def panel_title(text):
    """A panel label with each line's first letter capitalised and the phrase rewrites
    applied."""
    t = _rewrite(text)
    out = []
    for line in t.split("\n"):
        line = line.strip()
        out.append(line[0].upper() + line[1:] if line and line[0].islower() else line)
    return "\n".join(out)


def fig_title(text, width=96):
    """No title on the canvas: "" is returned and the text is recorded as sentences for
    write_hardware.  `width` is accepted for the callers' title layout and unused."""
    # Each line of a multi-line title is its own sentence.
    parts = [" ".join(l.split()) for l in text.split("\n")]
    CAPTION["title"] = _rewrite(_as_sentences([p for p in parts if p]))
    return ""


def _sentence(t):
    """One caption sentence with a full stop; the first letter is raised only when the first
    word is an ordinary lowercase word, so identifiers keep their case."""
    t = t.strip()
    if not t:
        return ""
    if t[0].islower() and not t.startswith(("$", "\\")):
        head = t.split(" ", 1)[0]
        # A digit, underscore or '=' marks code, which keeps its case.
        if (head.islower() and not any(c.isdigit() for c in head)
                and "_" not in head and "=" not in head):
            t = t[0].upper() + t[1:]
    return t if t.endswith((".", "!", "?")) else t + "."


def _as_sentences(parts):
    return " ".join(_sentence(p) for p in parts if p and p.strip())


def record_note(text):
    """Hold a note for write_hardware instead of drawing it."""
    CAPTION["notes"].append(_rewrite(" ".join(text.split())))


def paper_caption(*sentences):
    """Set the current figure's caption for captions.tex."""
    CAPTION["text"] = [s for s in sentences if s and s.strip()]


def record_fact(text):
    """Hold a sentence for the figure's section written by write_hardware."""
    CAPTION["facts"].append(_rewrite(" ".join(text.split())).replace("\\'e", "\u00e9"))


# {run label: (PROVENANCE dict, machine label)}, filled by register_run, read by env_line_for.
RUN_ENV = {}


def register_run(label, pv, machine):
    """Record the PROVENANCE dict and the machine label of one run."""
    RUN_ENV[label] = (dict(pv or {}), machine)


def stamp_runs(stamp):
    """The run labels a figure stamp names (`a + b`, or `a vs b`)."""
    out = []
    for r in str(stamp).replace(" vs ", " + ").split(" + "):
        r = r.strip()
        if r and r not in out:
            out.append(r)
    return out


def env_line_for(labels):
    """One line of conditions of the runs `labels` names: machine, dedicated, load at start,
    governor, turbo; differing values are joined by ` / `, the load lists one per run."""
    import runs as _mp
    # The `  [PARTIAL RUN]` suffix of a stamp is not part of the registered label.
    ents = [RUN_ENV[k] for k in (l.split("  [")[0] for l in labels)
            if k in RUN_ENV]
    if not ents:
        return ""

    def _all(vals):
        out = []
        for v in vals:
            if v and v not in out:
                out.append(v)
        return " / ".join(out)

    bits = []
    m = _all(mach for _pv, mach in ents)
    if m:
        bits.append("machine " + m)
    d = _all(_mp.dedicated(pv) for pv, _m in ents)
    if d:
        bits.append("dedicated=" + d)
    loads = [((pv.get("load_before") or "").split() or [""])[0]
             for pv, _m in ents]
    if any(loads):
        bits.append("load " + " / ".join(v or "?" for v in loads))
    g = _all(pv.get("cpufreq_governor") for pv, _m in ents)
    if g:
        bits.append(g + " governor")
    t = _all(pv.get("turbo") for pv, _m in ents)
    if t:
        bits.append("turbo: " + t)
    return ";  ".join(bits)


def note_provenance(outdir, stamp):
    """Record which runs a figure came from, per directory and per figure."""
    RUNS_SEEN.setdefault(outdir, [])
    CAPTION["runs"] = stamp_runs(stamp)
    for r in CAPTION["runs"]:
        if r not in RUNS_SEEN[outdir]:
            RUNS_SEEN[outdir].append(r)


def caption_finalise():
    """The provenance of the figure set for write_hardware, then cleared: {"dirs": [(outdir,
    [(run, conditions)])], "figures": [(outdir, stem, runs, facts)]}."""
    out = {"dirs": [(d, [(r, env_line_for([r])) for r in runs])
                    for d, runs in RUNS_SEEN.items()],
           "figures": list(FIGURES)}
    RUNS_SEEN.clear()
    FIGURES.clear()
    return out


def caption_reset():
    CAPTION["title"] = ""
    CAPTION["notes"] = []
    CAPTION["text"] = []
    CAPTION["facts"] = []
    CAPTION["runs"] = []


def _tex_escape_outside_math(t):
    """Escape LaTeX specials outside `$...$`, math verbatim; with an odd number of `$` the
    whole string goes through `tex_escape`."""
    parts = str(t).split("$")
    if len(parts) % 2 == 0:                      # unbalanced -> escape it all
        return tex_escape(t)
    out = []
    for i, seg in enumerate(parts):
        if i % 2:                                # inside math: verbatim
            out.append("$" + seg + "$")
        else:
            for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
                seg = seg.replace(a, b)
            out.append(seg)
    return "".join(out)


def _plain(text):
    # " | " becomes "; ": a bare `|` in LaTeX text mode does not set as a bar.
    text = text.replace("   |   ", "; ").replace(" | ", "; ")
    return display_text(text)            # AVX2 / AVX-512 and supplier names, as on the canvas


def caption_flush(outdir, stem):
    """Append the figure's caption to `captions.tex` as a figure environment; keep its title,
    notes and facts for write_hardware."""
    if not (CAPTION["title"] or CAPTION["notes"] or CAPTION["text"]):
        caption_reset()
        return
    import os as _os
    kept = [CAPTION["title"]] + CAPTION["notes"]
    facts = [_plain(_sentence(t)) for t in kept + CAPTION["facts"] if t.strip()]
    FIGURES.append((outdir, stem, list(CAPTION["runs"]), facts))
    # A figure without a caption of its own falls back to its title and notes.
    body = _tex_escape_outside_math(_plain(_as_sentences(CAPTION["text"] or kept)))
    label = stem.replace("fig-paper-", "").replace("_", "-")
    # make_paper_figures.py counts the `% ---- <stem> ----` marker lines.
    with open(_os.path.join(outdir, "captions.tex"), "a") as fh:
        fh.write("%% ---- %s ----\n"
                 "\\begin{figure}[t]\n"
                 "  \\centering\n"
                 "  \\includegraphics[width=\\linewidth]{%s.pdf}\n"
                 "  \\caption{%s}\n"
                 "  \\label{fig:%s}\n"
                 "\\end{figure}\n\n" % (stem, stem, body, label))
    caption_reset()


def glyph_key(plt, entries):
    """(handles, labels) in neutral grey for glyphs that are not a series; `entries` is
    [(marker, hollow, label), ...]."""
    hs, ls = [], []
    for marker, hollow, label in entries:
        kw = dict(ls="none", marker=marker, ms=6.0, color="0.35")
        if hollow:
            kw.update(mfc="none", mec="0.35", mew=1.3)
        hs.append(plt.Line2D([], [], **kw))
        ls.append(label)
    return hs, ls
