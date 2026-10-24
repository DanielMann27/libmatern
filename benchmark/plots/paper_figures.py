"""paper_figures.py -- the paper figure set, drawn from recorded run CSVs; measures
nothing (shared typography in style.py)."""
import math, re
from collections import defaultdict

import style

# The accuracy laws, in the order of the paper.
ACC_LAWS = ["grid", "random", "small_z", "near_int",
            "half_int", "large_z", "large_nu", "seam"]
# Timing-only ladder laws (the `*_large` sets are selected by LADDER_SET=large).
# The same set is listed in benchmark_cpu.c, suppliers.c, benchmark_gpu.cu and
# suppliers_gpu.cu and must be kept consistent.
LADDER_LAWS = ["size_random", "size_half",
               "size_random_large", "size_half_large"]


def ladder_laws_present(rows):
    """The ladder laws this run carries, in the canonical order."""
    have = {r.get("law") for r in (rows or [])}
    return [l for l in LADDER_LAWS if l in have]

# Display names of the laws (also imported by the table generator).
LAW_LABEL = {
    "grid": "Grid", "random": "Random", "small_z": "Small $z$",
    "near_int": r"Near-integer $\nu$", "half_int": r"Half-integer $\nu$",
    "large_z": "Large $z$", "large_nu": r"Large $\nu$", "seam": "Router seams",
    "size_random": "Size / random", "size_half": "Size / half-integer",
    "size_random_large": "Size / random ($2^{25}$)",
    "size_half_large": "Size / half-integer ($2^{25}$)",
}
# The CSV key `logtrick` (M = exp(log M)) is shown as "exp-log"; labels only.
ASM_LABEL = {"logtrick": "exp-log"}


def asm_label(a):
    return ASM_LABEL.get(a, a)


def _nonfinite_phrase(rows, points):
    """The panel sub-title 'non-finite: k rows, n points'."""
    return "non-finite: %d row%s, %s point%s" % (
        rows, "" if rows == 1 else "s", f"{points:,}", "" if points == 1 else "s")


FAMILY_LABEL = {"z_nu": r"$(z,\nu)$ supplied",
                "h_nu_s": r"$(h,\nu,s)$ --- $z$ computed"}
# The same labels on a canvas: matplotlib draws LaTeX's `---` as three hyphens.
FAMILY_PLOT_LABEL = {k: v.replace(" --- ", " \u2014 ") for k, v in FAMILY_LABEL.items()}

# ULP distances at or above this are failures, drawn in the hatched band: a fixed
# value above any approximation error and below failed evaluations.
CATASTROPHIC = 1e6

# (stem, reason) of every figure not drawn for a missing input; the driver reports them.
SKIPPED = []
# "<machine>/" prefix of skipped stems, set inside the driver's per-machine loop.
SKIP_PREFIX = ""


def _skip(stem, why):
    """Record `stem` as not drawn, with the reason; returns 0 for the caller."""
    SKIPPED.append((stem if "/" in stem else SKIP_PREFIX + stem, why))
    return 0


# Order in which a backend is chosen to represent the library in a figure.
BACKEND_PREFERENCE = ("avx512", "avx2", "matched_fma", "matched", "scalar")


def lib_backend_for(rows, prefer=None, family=None):
    """`prefer` if present in `rows`, else the first BACKEND_PREFERENCE match, or None."""
    have = {(r.get("backend") or "").strip() for r in rows
            if family is None or r.get("family") == family}
    if prefer and prefer in have:
        return prefer
    for b in BACKEND_PREFERENCE:
        if b in have:
            return b
    # a backend outside BACKEND_PREFERENCE is still a library row
    rest = sorted(h for h in have if h)
    return rest[0] if rest else None


# Backends that keep no workspace (src/backends/dispatch.c); `matched` runs the
# vector pipeline and needs one.
NO_WORKSPACE = {"scalar"}

BACKEND_LABEL = {
    "scalar":      "scalar (direct formulas, libm)",
    "matched":     "matched (vector algorithm, width 1)",
    # Not LaTeX-escaped (matplotlib draws a backslash); the tables use esc().
    "matched_fma": "matched_fma (width 1, hardware FMA)",
    "avx2":        "avx2 (vectorised elementary functions, 4 lanes)",
    "avx512":      "avx512 (8 lanes)",
}


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _i(row, key):
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return 0


def _ok(v):
    return v == v and math.isfinite(v)


def scope_of(r):
    return (r.get("timing_scope") or "call").strip()


def failed(r):
    """Non-finite on some point, or a maximum at or above CATASTROPHIC."""
    return _i(r, "nonfinite") > 0 or _f(r, "u_max") >= CATASTROPHIC


def best_assembly(rows, prefer="u_max"):
    """One row per (supplier, family, law): its best assembly (named in `backend`)."""
    best = {}
    for r in rows:
        k = (r["supplier"], r["family"], r["law"])
        v = _f(r, prefer)
        if not _ok(v):
            continue
        # Finite everywhere first (a maximum over fewer points is not better),
        # then more scored points, then u_max.
        rank = (1 if _i(r, "nonfinite") > 0 else 0, -_i(r, "n_scored"), v)
        if k not in best or rank < best[k][0]:
            best[k] = (rank, r)
    return {k: v[1] for k, v in best.items()}


def _band(ax, ymin, ytop=CATASTROPHIC):
    """Draw the hatched failure band under the data; return the y of off-scale points."""
    lo = ytop / 6.0
    ax.axhspan(lo, ytop, facecolor="0.92", edgecolor="0.55",
               hatch="////", lw=0.6, zorder=0)
    ax.set_ylim(ymin, ytop)
    return math.sqrt(lo * ytop)


def _band_x(ax, xmin, xtop=CATASTROPHIC):
    """`_band` on a horizontal value axis; `xmin` is required (else only the band shows)."""
    lo = xtop / 6.0
    ax.axvspan(lo, xtop, facecolor="0.92", edgecolor="0.55",
               hatch="////", lw=0.6, zorder=0)
    ax.set_xlim(xmin, xtop)
    return math.sqrt(lo * xtop)


def _stamp(fig, text):
    """Write the run provenance in the bottom-right corner of the figure."""
    fig.text(0.995, 0.003, text, ha="right", va="bottom",
             fontsize=5.2, color="0.45")


def _finish(fig, outdir, stem, stamp):
    """Stamp (development mode) or record the provenance (paper mode), save, close."""
    if not style.PAPER:
        env = style.env_line_for(style.stamp_runs(stamp))
        _stamp(fig, (stamp + "   |   " + env) if env else stamp)
    else:
        style.note_provenance(outdir, stamp)
    n = style.save(fig, outdir, stem)
    style.caption_flush(outdir, stem)
    import matplotlib.pyplot as _plt
    # not fig.clf(): clearing resets shared log axes to (0, 1)
    _plt.close(fig)
    return n


# Every plotted cost is a median over the timed repetitions; minima are never
# substituted.
THREADS_NO_MEDIAN = ("the thread rows carry no ns_med column; every cost in the "
                     "figure set is a median over the repetitions, and minima "
                     "are not substituted")


def threads_median(thr):
    """(rows, ok): derived thread columns rebuilt from `ns_med`; ok False if one lacks it.
    threads_cpu.c's own derived columns (each thread's fastest repetition) are not read."""
    rows = [dict(r) for r in (thr or [])]
    if not rows or not all(_ok(_f(r, "ns_med")) and _f(r, "ns_med") > 0
                           for r in rows):
        return rows, False
    base = {}
    key = lambda r: (r.get("family"), r.get("law"), r.get("supplier") or "-",
                     r.get("backend"))
    for r in sorted(rows, key=lambda r: _i(r, "threads")):
        base.setdefault(key(r), _f(r, "ns_med"))
    for r in rows:
        ns, t = _f(r, "ns_med"), max(_i(r, "threads"), 1)
        sp = base[key(r)] / ns
        r["ns_per_elem"] = repr(ns)
        r["speedup"], r["efficiency"] = repr(sp), repr(sp / t)
        r["values_per_s"] = repr(1e9 / ns)
    return rows, True


# Fact sentences (N per law, cost statistic, repetitions) for HARDWARE.md in paper mode.
def _num(v):
    return f"{int(v):,}".replace(",", " ")


def _law_title(law):
    """LAW_LABEL without the corpus size (N is recorded by write_hardware)."""
    return re.sub(r"\s*\(\$2\^\{\d+\}\$\)", "", LAW_LABEL.get(law, law))


def _vals(rows, key, laws=None):
    d = defaultdict(set)
    for r in rows or []:
        if laws is not None and r.get("law") not in laws:
            continue
        v = _i(r, key)
        if v > 0:
            d[r.get("law")].add(v)
    return d


def _law_order(d):
    known = [l for l in ACC_LAWS + LADDER_LAWS if l in d]
    return known + sorted(l for l in d if l not in known)


def _n_phrase(rows, key="n", laws=None):
    d = _vals(rows, key, laws)
    allv = sorted({v for vs in d.values() for v in vs})
    if not allv:
        return ""
    if len(allv) == 1:
        return _num(allv[0])
    return ", ".join("%s %s" % (_law_title(l), " / ".join(_num(v) for v in sorted(d[l])))
                     for l in _law_order(d))


def _reps_phrase(rows, laws=None):
    allv = sorted({v for vs in _vals(rows, "reps", laws).values() for v in vs})
    if not allv:
        return "the timed repetitions"
    if len(allv) == 1:
        return "%d timed repetitions" % allv[0]
    return "%d to %d timed repetitions" % (allv[0], allv[-1])


def _acc_sentence(rows, laws=None, what="the ULP error"):
    n = _n_phrase(rows, "n", laws)
    if not n:
        return ""
    many = "," in n
    # n < n_file (points on disk): the row is scored on a --max-points stride.
    sub = sorted({(r.get("supplier") or r.get("backend") or "?")
                  for r in rows or []
                  if (laws is None or r.get("law") in laws)
                  and _i(r, "n_file") and _i(r, "n") < _i(r, "n_file")})
    return ("Error: %s over the points of each dataset, $N$ = %s%s (a row that "
            "returned non-finite values is scored over fewer)."
            % (what, n, "" if many else " points per dataset")
            + ((" Subsampled: %s scored a stride of the dataset, not every point."
                % ", ".join(sub)) if sub else ""))


def _cost_sentence(rows, stat="median", laws=None, show_n=False):
    npd, nd = _vals(rows, "ns_points", laws), _vals(rows, "n", laws)
    reps = _reps_phrase(rows, laws)
    allp = sorted({v for vs in npd.values() for v in vs})
    if not allp:
        where = "one call"
    elif all(max(npd[l]) >= max(nd.get(l, {0})) for l in npd):
        where = ("one call on all $N$ points of the dataset" + (
                 " ($N$ = %s)" % _n_phrase(rows, "n", laws) if show_n else ""))
    elif len(allp) == 1:
        where = "one call on the first %s points of the dataset" % _num(allp[0])
    else:
        where = "one call on %s points" % _n_phrase(rows, "ns_points", laws)
    return "Cost: the %s over %s, each %s." % (stat, reps, where)


def _ladder_phrase(rows):
    ns = sorted({_i(r, "n") for r in rows or [] if _i(r, "n") > 0})
    if not ns:
        return ""
    lo, hi = ns[0], ns[-1]
    f = lambda v: ("$2^{%d}$ = %s" % (round(math.log2(v)), _num(v))
                   if v & (v - 1) == 0 else _num(v))
    return "$N$ from %s to %s elements per call (%d rungs)" % (f(lo), f(hi), len(ns))


def _pass_from(rows):
    """First rung read from the cumulative pass (LADDER_PASS_FROM), or 0 if none."""
    ss = [_i(r, "ladder_pass_from") for r in rows or []
          if r.get("ladder_mode") == "pass"]
    return min(ss) if ss else 0


def _pass_sentence(rows):
    """Fact sentence for ladder rows with a cumulative pass; empty otherwise."""
    S = _pass_from(rows)
    if not S:
        return ""
    return ("Rungs from $N$ = $2^{%d}$ up are read from one cumulative pass per "
            "repetition (one call per chunk $[0,S)$, $[S,2S)$, $[2S,4S)$, ... with "
            "$S = 2^{%d}$): the cost of rung $N$ is the time for the first $N$ "
            "points within the pass." % (round(math.log2(S)), round(math.log2(S))))


def _facts(*parts):
    """Sentences for the figure's section written by write_hardware (paper mode)."""
    text = " ".join(p for p in parts if p)
    if style.PAPER and text:
        style.record_fact(text)


# Paper-mode captions (captions.tex): the argument conventions as they name them.
# CONV, CONV_SUPPLIED, _obtained, _and and _size_set are also used by paper_tables.py.
CONV ={"z_nu": r"$(z,\nu)$", "h_nu_s": r"$(h,\nu,s)$"}
CONV_SUPPLIED = {"z_nu": r"with $(z,\nu)$ supplied",
                 "h_nu_s": r"with $(h,\nu,s)$ supplied and $z$ computed"}
_SIZE_SET = {"size_random": "random", "size_half": r"half-integer $\nu$"}


def _caption(*sentences):
    """The figure's caption (paper mode); nothing is drawn on the canvas."""
    if style.PAPER:
        style.paper_caption(*sentences)


def _obtained(arch, gpu=False):
    """The caption's closing sentence naming the machine, from the run's label."""
    import runs as mp
    name, kind = mp.machine_name(arch, gpu)
    return "Results were obtained on the %s %s." % (name, kind) if name else ""


def _and(items):
    items = [str(x) for x in items if x]
    return (", ".join(items[:-1]) + " and " + items[-1]) if len(items) > 1 else \
        "".join(items)


def _size_set(law):
    return _SIZE_SET.get(law.replace("_large", ""), _law_title(law))


def _columns(fams):
    """The sentence naming the convention of each column of the figure."""
    sup = {"z_nu": r"$(z,\nu)$ are supplied",
           "h_nu_s": r"$(h,\nu,s)$ are supplied and $z$ is computed"}
    if len(fams) == 2:
        return ("The left column shows results when %s, while the right column "
                "shows results when %s." % (sup.get(fams[0], fams[0]),
                                            sup.get(fams[1], fams[1])))
    return ("The panels show results when %s." % sup.get(fams[0], fams[0])
            if fams else "")


def fig_cost_accuracy_plane(plt, acc, sup, outdir, stamp, family="z_nu",
                            lib_backend="avx512", scope="call", arch=""):
    """Median and maximum ULP against cost per element, one panel per accuracy law."""
    _pool = [r for r in acc if scope_of(r) == scope and r["family"] == family]
    lib_backend = lib_backend_for(_pool, prefer=lib_backend) or lib_backend
    acc = [r for r in _pool if r["backend"] == lib_backend]
    sup = [r for r in sup if scope_of(r) == scope and r["family"] == family]
    if not sup or not acc:
        return _skip(f"cost-accuracy-{family}",
                     "no external supplier rows" if acc else "no library rows"
                     if sup else "neither library nor supplier rows")
    best = best_assembly(sup)
    laws = [l for l in ACC_LAWS
            if any(r["law"] == l for r in acc)
            or any(k[2] == l for k in best)]
    if not laws:
        return _skip(f"cost-accuracy-{family}",
                     f"no rows for family {family} in any known law")

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.1), sharex=True, sharey=True)
    axes = axes.ravel()
    handles = {}
    for ax, law in zip(axes, laws + [None] * (8 - len(laws))):
        if law is None:
            # removed, not hidden: a hidden shared axis pushes (0, 1) onto the log axes
            fig.delaxes(ax)
            continue
        top = _band(ax, 0.12)
        series = []
        for r in acc:
            if r["law"] == law:
                series.append((style.LIB, r))
        for (s, fam, l), r in best.items():
            if l == law:
                series.append((s, r))
        nfail = 0
        nfin = []
        _worst_off = None          # largest clipped u_max in this panel
        for name, r in series:
            x = _f(r, "ns_med")
            med, mx = _f(r, "u_med"), _f(r, "u_max")
            if not (_ok(x) and _ok(med)):
                continue
            c = style.supplier_color(name)
            is_lib = name == style.LIB
            z = 5 if is_lib else 3
            # the whisker first, so a marker is never hidden under it
            if _ok(mx) and mx > med:
                ax.plot([x, x], [min(med, top), min(mx, top)], color=c,
                        lw=1.0, alpha=0.55, solid_capstyle="butt", zorder=z)
            if _ok(mx):
                if mx >= CATASTROPHIC:
                    ax.plot([x], [top], marker="^", ms=5.0, mfc="none",
                            mec=c, mew=1.2, ls="none", zorder=z + 1)
                    nfail += 1
                    if _worst_off is None or mx > _worst_off:
                        _worst_off = mx
                else:
                    ax.plot([x], [mx], marker="_", ms=6.0, color=c,
                            mew=1.1, ls="none", zorder=z)
            # ring rows scored over fewer points; same predicate as the title count
            incomplete = _i(r, "nonfinite") > 0
            if incomplete:
                # square ring, matching the glyph key
                ax.plot([x], [med], marker="s",
                        ms=(6.2 if is_lib else 4.4) + 4.2, mfc="none",
                        mec="0.15", mew=0.9, ls="none", zorder=z + 3)
            h, = ax.plot([x], [med], marker="o" if is_lib else "s",
                         ms=6.2 if is_lib else 4.4,
                         mfc=c, mec="white" if is_lib else c,
                         mew=0.9 if is_lib else 0.6, ls="none", zorder=z + 2)
            handles.setdefault(name, h)
            if incomplete:
                nfin.append((x, _i(r, "nonfinite"), c))
        # non-finite counts go into the panel title, clear of the data
        _sub = ""
        if nfin:
            tot = sum(k for _, k, _ in nfin)
            _sub = "\n" + _nonfinite_phrase(len(nfin), tot)
        # clipped rows share one height; the title gives their count and maximum
        if _worst_off is not None:
            _sub += ("\n%d clipped; max $10^{%d}$ ULP"
                     % (nfail, round(math.log10(_worst_off))))
        ax.set_xscale("log")
        ax.set_yscale("log")
        style.refline(ax, style.ULP_ONE, "1 ULP")
        ax.set_title(LAW_LABEL.get(law, law) + _sub, pad=3, fontsize=7.4)
        ax.grid(True, which="major", alpha=0.30)
    for ax in axes[4:]:
        ax.set_xlabel("Cost  [ns / element]")
    fig.supylabel("ULP error  (median $\\rightarrow$ maximum)", fontsize=9,
                  x=0.015)

    order = [style.LIB] + sorted(k for k in handles if k != style.LIB)
    _gh, _gl = style.glyph_key(plt, [("^", True,  "clipped at $10^6$ ULP"),
                                     ("_", False, "maximum"),
                                     ("s", True,  "non-finite on some points")])
    style.legend_figure(fig,
                        handles=[handles[k] for k in order if k in handles] + _gh,
                        labels=[k for k in order if k in handles] + _gl,
                        ncol=4, y=0.012)
    fig.suptitle(style.fig_title(
                 f"Cost against accuracy, {FAMILY_LABEL.get(family, family)}"
                 f"   [{arch}, {scope} scope, library = {lib_backend}]"),
                 y=0.998, fontsize=9.5)
    style.note_below(fig,
             "one panel = one dataset = one $N$:  each implementation sits at a fixed "
             "cost and the bar is its error distribution, not a sweep over $N$.  "
             "A row marked non-finite is scored over fewer points, so its maximum "
             "is not comparable with the others", y=0.105)
    # room below the panels for the three-row legend and the note above it
    fig.tight_layout(rect=(0, 0.20, 1, 0.995))
    _rows = [r for r in (acc or []) + (sup or [])
             if r.get("family") == family and scope_of(r) == scope]
    _facts(_acc_sentence(_rows, laws, "median (marker) and maximum (tick) of "
                         "the ULP error"),
           _cost_sentence(_rows, "median", laws))
    _caption("Overview of runtime and accuracy for the %s datasets."
             % CONV.get(family, family),
             "Each panel compares the implementations by runtime, measured in "
             "nanoseconds per element, and accuracy, measured by the median "
             "(marker) and maximum (tick) ULP error; both axes are "
             "logarithmic.",
             "For implementations producing non-finite results, the panel title "
             "reports the number of affected rows and points; these points are "
             "excluded from the ULP statistics.",
             "The panel title also reports the number of maxima at or above the "
             "plotting limit of $10^6$ ULP and the largest ULP error, and "
             "off-scale maxima are drawn in the hatched band at the top of each "
             "panel.",
             "The library is shown with its %s backend." % lib_backend,
             _obtained(arch))
    return _finish(fig, outdir, f"fig-paper-cost-accuracy-{family}-{scope}", stamp)


def fig_attribution(plt, acc, outdir, stamp, family="z_nu", arch=""):
    """Maximum ULP and median cost per law of each library backend, side by side."""
    acc = [r for r in acc if r["family"] == family and scope_of(r) == "call"]
    if not acc:
        return _skip(f"attribution-{family}", "no library accuracy rows")
    laws = [l for l in ACC_LAWS if any(r["law"] == l for r in acc)]
    backs = [b for b in style.BACKEND_ORDER
             if any(r["backend"] == b for r in acc)]
    if len(backs) < 2 or not laws:
        return _skip(f"attribution-{family}",
                     f"needs >=2 library backends to attribute a difference to, "
                     f"found {len(backs)} ({', '.join(backs) or 'none'})"
                     if len(backs) < 2 else
                     f"no rows for family {family}")
    by = {(r["backend"], r["law"]): r for r in acc}

    fig, (axa, axs) = plt.subplots(1, 2, figsize=(9.4, 3.95))
    x = list(range(len(laws)))
    for b in backs:
        c = style.BACKEND_COLOR.get(b, "0.4")
        m = style.BACKEND_MARK.get(b, "o")
        ya = [_f(by[(b, l)], "u_max") if (b, l) in by else float("nan")
              for l in laws]
        ys = [_f(by[(b, l)], "ns_med") if (b, l) in by else float("nan")
              for l in laws]
        # categorical x axis: the faint connector only groups one backend
        axa.plot(x, ya, marker=m, ms=5.0, color=c, lw=0.5, alpha=0.45,
                 label=BACKEND_LABEL.get(b, b))
        axs.plot(x, ys, marker=m, ms=5.0, color=c, lw=0.5, alpha=0.45)
    for ax, lab in ((axa, "Maximum ULP error"), (axs, "Cost  [ns / element]")):
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LAW_LABEL.get(l, l) for l in laws],
                           rotation=35, ha="right")
        ax.set_ylabel(lab)
    style.refline(axa, style.ULP_ONE, "1 ULP")
    axa.set_title(style.panel_title("Accuracy"), pad=4)
    axs.set_title(style.panel_title("Cost"), pad=4)
    style.legend_figure(fig, axa)
    fig.suptitle(style.fig_title(
                 "Worst-case error and cost by backend"
                 f"   [{arch}, {FAMILY_LABEL.get(family, family)}]"),
                 y=0.995, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    _rows = [r for r in (acc or []) if r.get("family") == family]
    _facts(_acc_sentence(_rows, laws, "the maximum ULP error"),
           _cost_sentence(_rows, "median", laws))
    _caption("Worst-case accuracy and runtime cost of the library backends for "
             "the %s datasets." % CONV.get(family, family),
             "The left panel shows the maximum ULP error and the right panel "
             "the median cost per element, in nanoseconds, for each dataset; both "
             "vertical axes are logarithmic.",
             _obtained(arch))
    return _finish(fig, outdir, f"fig-paper-attribution-{family}", stamp)


def fig_worst_case_map(plt, acc, sup, outdir, stamp, family="z_nu", arch=""):
    """Each row's worst point (`worst_arg`, `worst_nu`), one panel per supplier."""
    _libbk = lib_backend_for([r for r in acc if r["family"] == family
                              and scope_of(r) == "call"])
    rows = [r for r in acc if r["family"] == family and scope_of(r) == "call"
            and r["backend"] == _libbk]
    lib = [dict(r, supplier=style.LIB) for r in rows]
    ext = list(best_assembly([r for r in sup if r["family"] == family
                              and scope_of(r) == "call"]).values())
    allr = lib + ext
    sups = [style.LIB] + sorted({r["supplier"] for r in ext})
    pts = [(r, _f(r, "worst_arg"), _f(r, "worst_nu"), _f(r, "u_max"))
           for r in allr]
    pts = [p for p in pts if _ok(p[1]) and _ok(p[2]) and p[1] > 0 and p[2] > 0]
    if not pts:
        return _skip(f"worst-case-map-{family}",
                     "no row carries a worst-point coordinate "
                     "(worst_nu / worst_arg) for this family")

    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=1.0, vmax=CATASTROPHIC)
    # three columns when they fill the last row, otherwise four
    ncol = 3 if len(sups) % 3 == 0 else 4
    nrow = (len(sups) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    sc = None
    for ax, s in zip(axes, sups + [None] * (len(axes) - len(sups))):
        if s is None:
            fig.delaxes(ax)          # see the note in fig_cost_accuracy_plane
            continue
        mine = [p for p in pts if p[0]["supplier"] == s]
        if mine:
            sc = ax.scatter([p[1] for p in mine], [p[2] for p in mine],
                            c=[min(p[3], CATASTROPHIC) for p in mine],
                            norm=norm, cmap="YlOrRd", s=38,
                            edgecolors="0.25", linewidths=0.5, zorder=3)
            # only the panel's worst point is labelled, so labels cover no data
            w = max(mine, key=lambda p: p[3])
            # near the right edge the label goes left of the point, inside the canvas
            _xa = ax.transLimits.transform(
                      ax.transScale.transform((w[1], w[2])))[0]
            _right = _xa > 0.62
            ax.annotate(LAW_LABEL.get(w[0]["law"], w[0]["law"])
                        .replace("$", "").replace("\\nu", r"$\nu$"),
                        (w[1], w[2]), textcoords="offset points",
                        xytext=(-5 if _right else 5, 4), fontsize=5.4,
                        ha="right" if _right else "left",
                        color="0.20", fontweight="bold",
                        annotation_clip=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(s + (" (this library)" if s == style.LIB else ""), pad=3,
                     fontweight="bold" if s == style.LIB else "normal",
                     color=style.supplier_color(s) if s == style.LIB else "black")
    # every panel with nothing under it keeps its own x tick labels
    for _pi, _pa in enumerate(axes[:len(sups)]):
        if _pi + ncol >= len(sups):
            _pa.tick_params(labelbottom=True)
    # `worst_arg` is z for z_nu and h for h_nu_s (s is not stored in the row)
    _argname = "h" if family == "h_nu_s" else "z"
    for ax in axes[-ncol:]:
        ax.set_xlabel("$%s$ at the worst point" % _argname)
    for i in range(0, len(axes), ncol):
        axes[i].set_ylabel(r"$\nu$ at the worst point")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.tolist() if hasattr(axes, "tolist")
                          else list(axes), fraction=0.020, pad=0.012,
                          extend="max")
        cb.set_label(r"ULP error at that point   (clamped at $10^6$)",
                     fontsize=7)
        cb.ax.tick_params(labelsize=6)
    _cl = [p[3] for p in pts if _ok(p[3]) and p[3] >= CATASTROPHIC]
    if _cl:
        style.note_below(fig,
            "%d of the %d worst points are past the top of the colour scale and "
            "are drawn in the cap colour, the largest at $10^{%d}$ ULP; the "
            "remaining %d are coloured by their true value"
            % (len(_cl), len(pts), round(math.log10(max(_cl))),
               len(pts) - len(_cl)), y=0.012)
    fig.suptitle(style.fig_title(
                 "Location of each implementation's worst case in the " +
                 (r"$(h,\nu)$ plane" if family == "h_nu_s" else r"$(z,\nu)$ plane")
                 + f"   [{arch}, "
                 f"{FAMILY_LABEL.get(family, family)}]"), y=0.998, fontsize=9.5)
    _rows = [r for r in (acc or []) + (sup or []) if r.get("family") == family]
    _facts(_acc_sentence(_rows, ACC_LAWS, "each marker is the point of maximum "
                         "ULP error"))
    _caption("Location of the worst case of each implementation for the %s "
             "datasets." % CONV.get(family, family),
             "Each marker is the point of maximum ULP error on one dataset, "
             "placed in the %s plane and coloured by its ULP error, with the "
             "colour scale capped at $10^6$ ULP; the largest value of each panel "
             "is labelled with its dataset."
             % (r"$(h,\nu)$" if family == "h_nu_s" else r"$(z,\nu)$"),
             ("The library is shown with its %s backend." % _libbk) if _libbk
             else "",
             _obtained(arch))
    return _finish(fig, outdir, f"fig-paper-worst-case-map-{family}", stamp)


def fig_assembly(plt, sup, acc, outdir, stamp, family="z_nu", arch="",
                 scope="call", lib_supplier=None, stem="fig-paper-assembly",
                 where=""):
    """Maximum ULP per assembly of the externals, on the laws where they differ most."""
    # on the device the library row comes from `sup` by name (`lib_supplier`)
    sup =[r for r in sup if r["family"] == family and scope_of(r) == scope]
    if lib_supplier:
        acc = [dict(r, backend=lib_supplier) for r in sup
               if r["supplier"] == lib_supplier]
        sup = [r for r in sup if r["supplier"] != lib_supplier]
    if not sup:
        return _skip(f"{stem}-{family}".replace("fig-paper-", ""),
                     "no external supplier rows; the assembly applies to the "
                     "external suppliers only")
    # Spread within one supplier; up to two laws where `direct` loses most and
    # two where it wins most (rows finite everywhere first).
    def _spread(law):
        """(ratio, supplier, best assembly, clean) of the widest spread, or None."""
        best = None
        for sname in {r["supplier"] for r in sup}:
            vals = {}
            clean = True
            for r in sup:
                if r["law"] != law or r["supplier"] != sname:
                    continue
                v = _f(r, "u_max")
                if _ok(v):
                    vals[r["backend"]] = v
                    if _i(r, "nonfinite") > 0:
                        clean = False
            if len(vals) < 2:
                continue
            lo = min(vals, key=vals.get)
            ratio = max(vals.values()) / max(min(vals.values()), 1e-30)
            cand = (ratio, sname, lo, clean)
            if best is None or ratio > best[0]:
                best = cand
        return best

    loses, wins = [], []          # `direct` loses / `direct` wins
    for l in ACC_LAWS:
        sp = _spread(l)
        if sp is None or sp[0] <= 10.0:
            continue
        (loses if sp[2] != "direct" else wins).append((l, sp))
    wins.sort(key=lambda t: (not t[1][3], -t[1][0]))     # clean rows first
    loses.sort(key=lambda t: -t[1][0])
    chosen = {l for l, _ in loses[:2]} | {l for l, _ in wins[:2]}
    direction = {l: sp[2] for l, sp in loses + wins}
    laws = [l for l in ACC_LAWS if l in chosen]         # in ACC_LAWS order
    if not laws:
        return _skip(f"{stem}-{family}".replace("fig-paper-", ""),
                     "no law where a supplier's assemblies differ by more than "
                     "10x; a supplier measured in a single assembly has no "
                     "spread, so a run without the exp-log or scaled variants "
                     "has no input for this figure")
    asms = [a for a in ("direct", "logtrick", "scaled")
            if any(r["backend"] == a for r in sup)]
    sups = sorted({r["supplier"] for r in sup})
    by = {(r["supplier"], r["law"], r["backend"]): r for r in sup}
    # library row for the dashed reference line
    _libbk = (lib_supplier or
              lib_backend_for([r for r in acc if r["family"] == family]))
    libby = {r["law"]: r for r in acc
             if r["family"] == family and r["backend"] == _libbk}

    n = len(laws)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig, grid = plt.subplots(nrow, ncol, figsize=(2.7 * ncol + 0.6,
                                                  2.45 * nrow + 1.25),
                             sharey=True, sharex=True, squeeze=False)
    cells = list(grid.flat)
    for spare in cells[n:]:
        spare.set_visible(False)
    axes = cells[:n]
    w = 0.8 / len(asms)
    hatches = {"direct": "", "logtrick": "//", "scaled": ".."}
    for ax, law in zip(axes, laws):
        _clamped, _worst_off = [], None
        for j, a in enumerate(asms):
            xs, ys = [], []
            for i, s in enumerate(sups):
                r = by.get((s, law, a))
                if r is None:
                    continue
                v = _f(r, "u_max")
                if not _ok(v):
                    continue
                xs.append(i - 0.4 + w * (j + 0.5))
                ys.append(min(v, CATASTROPHIC))
                # clipped bars: marked by a glyph and counted in the panel title
                if v >= CATASTROPHIC:
                    _clamped.append(xs[-1])
                    if _worst_off is None or v > _worst_off:
                        _worst_off = v
            ax.bar(xs, ys, width=w * 0.92, color="0.75" if j else "0.45",
                   edgecolor="0.25", lw=0.5, hatch=hatches.get(a, ""),
                   label=asm_label(a) if ax is axes[0] else None,
                   zorder=3)
        lr = libby.get(law)
        if lr is not None and _ok(_f(lr, "u_max")):
            ax.axhline(_f(lr, "u_max"), color=style.SUPPLIER_COLOR[style.LIB],
                       lw=1.3, ls="--", zorder=4,
                       label=(lib_supplier or style.LIB) if ax is axes[0] else None)
        _band(ax, 0.5)
        ax.set_yscale("log")
        if _clamped:
            ax.plot(_clamped, [CATASTROPHIC] * len(_clamped), marker="^",
                    ms=4.0, mfc="none", mec="0.15", mew=0.9, ls="none",
                    zorder=6, clip_on=False)
        ax.set_xticks(range(len(sups)))
        ax.set_xticklabels(sups, rotation=40, ha="right", fontsize=6.2)
        # suppliers are in the same order everywhere: only the bottom row is named
        _k = axes.index(ax)
        ax.tick_params(axis="x", labelbottom=_k + ncol >= n)
        d = direction.get(law, "")
        _sub = ("\n%d clipped; max: $10^{%d}$ ULP"
                % (len(_clamped), round(math.log10(_worst_off)))) if _clamped else ""
        ax.set_title(LAW_LABEL.get(law, law) + _sub, pad=10, fontsize=7.6)
        # direction of the spread
        ax.text(0.5, 1.012, "%s has lower error" % asm_label(d),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=style.NOTE_FONTSIZE,
                color="#b3261e" if d == "direct" else "#1b7f3b")
    for _k, ax in enumerate(axes):
        if _k % ncol == 0:
            ax.set_ylabel("Maximum ULP error")
    h, lb = axes[0].get_legend_handles_labels()
    _gh, _gl = style.glyph_key(plt, [("^", True, "bar clipped at $10^6$ ULP")])
    style.legend_figure(fig, handles=h + _gh, labels=lb + _gl, ncol=3,
                        y=0.008)
    # the selection rule; recorded for HARDWARE.md in paper mode
    style.note_below(fig,
             "%d of the %d accuracy datasets, at most two in each direction: those "
             "with the largest error ratio in favour of another assembly over "
             "direct, and those with the largest ratio in favour of direct, "
             "among datasets whose assemblies differ by more than $10\\times$;  the hatched band is "
             "above $10^6$ ULP, and a bar entering it is clipped at $10^6$ ULP, not "
             "measured there" % (len(laws), len(ACC_LAWS)))
    fig.suptitle(style.fig_title(
                 "Worst-case error by assembly   "
                 f"[{arch}{where}, {FAMILY_LABEL.get(family, family)}]"),
                 y=0.995, fontsize=9.5)
    # room for the note in development mode only
    fig.tight_layout(rect=(0, 0.075 if style.PAPER else 0.135, 1, 0.95))
    _rows = [r for r in (sup or []) + (acc or [])
             if r.get("family") == family and scope_of(r) == scope]
    _facts(_acc_sentence(_rows, laws, "the maximum ULP error"))
    _caption("Effect of the assembly strategy on the maximum ULP error of the "
             "external %simplementations for the %s datasets."
             % ("GPU " if lib_supplier else "", CONV.get(family, family)),
             "For each dataset, the largest ratio between the maximum ULP "
             "errors of the assemblies of any one implementation is taken, and "
             "among the datasets on which it exceeds 10 the panels show at most "
             "two on which another assembly is more accurate than direct and at "
             "most two on which direct is the most accurate, as stated above each "
             "panel.",
             "For each implementation, the bars correspond to the %s %s; the "
             "dashed line shows the error of %s, which uses a single assembly."
             % (_and(asm_label(a) for a in asms),
                "assemblies" if len(asms) > 1 else "assembly",
                lib_supplier or style.LIB),
             "Bars at or above the plotting limit of $10^6$ ULP are clipped and "
             "marked by triangles; each panel reports the number of clipped bars "
             "and the largest ULP error.",
             _obtained(arch, gpu=bool(lib_supplier)))
    return _finish(fig, outdir, f"{stem}-{family}", stamp)


def fig_gpu_assembly(plt, gsup, outdir, stamp, family="z_nu", arch=""):
    """fig_assembly on the device at kernel scope (no device supplier has a scaled K)."""
    return fig_assembly(plt, gsup, [], outdir, stamp, family=family, arch=arch,
                        scope="kernel", lib_supplier=style.LIB_CUDA,
                        stem="fig-paper-gpu-assembly", where=", kernel scope")


def fig_ladder(plt, lad, slad, outdir, stamp, law="size_random",
               family="z_nu", arch="", asm_of=None):
    """Cost per element against N on a ladder law, library backends and externals."""
    # one family: two conventions at the same N are two series, not two points
    lad =[r for r in lad if r["law"] == law and r["family"] == family]
    slad = [r for r in slad if r["law"] == law and scope_of(r) == "call"
            and r["family"] == family]
    if len({r["backend"] for r in lad}) < 2 and not slad:
        return _skip(f"ladder-{law}",
                     "fewer than 2 library backends on the ladder and no "
                     "external ladder rows")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    def _pts(rows, key):
        """Sorted (N, value) rungs; [] (reported) when an N repeats."""
        d = defaultdict(list)
        for r in rows:
            n, v = _f(r, "n"), _f(r, key)
            if _ok(n) and _ok(v):
                d[n].append(v)
        bad = {n: len(v) for n, v in d.items() if len(v) > 1}
        if bad:
            print(f"    !! ladder {law}/{family}: {len(bad)} rung(s) carry more "
                  f"than one row --- a dimension is unfiltered; series dropped")
            return []
        return sorted((n, v[0]) for n, v in d.items())

    for b in style.BACKEND_ORDER:
        # library rows: the workspace is built inside each call
        p = _pts([r for r in lad if r["backend"] == b], "ns_ws_per_call_med")
        if not p:
            continue
        ax.plot([x for x, _ in p], [y for _, y in p],
                marker=style.BACKEND_MARK.get(b, "o"), ms=4.0, lw=1.3,
                color=style.BACKEND_COLOR.get(b, "0.4"),
                label=f"{style.LIB} {b}")
    # One assembly per supplier: `asm_of` is best_assembly's choice from the
    # accuracy rows, so this curve and the cost-accuracy marker are one row.
    asm_of = asm_of or {}
    ends = {}
    for sname in sorted({r["supplier"] for r in slad}):
        mine = [r for r in slad if r["supplier"] == sname]
        want = asm_of.get(sname)
        if want and any(r["backend"] == want for r in mine):
            mine = [r for r in mine if r["backend"] == want]
        elif len({r["backend"] for r in mine}) > 1:
            # no accuracy row to choose an assembly: reported, not picked
            print(f"    !! ladder {law}: {sname} has "
                  f"{len({r['backend'] for r in mine})} assemblies and no "
                  f"accuracy row to choose one; series dropped")
            continue
        p = _pts(mine, "ns_med")
        if not p:
            continue
        lab = sname
        ax.plot([x for x, _ in p], [y for _, y in p], marker="s", ms=3.2,
                lw=1.0, ls="--", color=style.supplier_color(sname),
                label=lab, alpha=0.85)
        ends[sname] = p[-1]

    # A row stopped by the per-rung wall-clock budget of suppliers.c
    # (--ladder-budget) is ringed at its last rung (its largest `n`).
    if ends:
        # the top includes the library rows, so the furthest supplier is ringed too
        _lib_top = max((_f(r, "n") for r in lad if _ok(_f(r, "n"))), default=0.0)
        top = max([_lib_top] + [x for x, _ in ends.values()])
        short = {k: v for k, v in ends.items() if v[0] < top}
        for k, (x, y) in short.items():
            ax.plot([x], [y], marker="o", ms=8.5, mfc="none",
                    mec=style.supplier_color(k), mew=1.3, ls="none", zorder=7)
        if short:
            names = ", ".join(
                f"{k} at $N{{=}}2^{{{int(round(math.log(v[0], 2)))}}}$"
                for k, v in sorted(short.items()))
            style.note_below(fig,
                     "ring = last rung of a row stopped by the per-rung "
                     "wall-clock budget (suppliers --ladder-budget):  " + names)
    if not slad:
        # No supplier ladder for this family; run_benchmark.sh runs it on each of FAMILIES.
        style.note_below(fig,
                "library backends only for this convention:  the runs carry no "
                "supplier ladder rows for it")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$N$  [elements per call]")
    ax.set_ylabel("Cost  [ns / element]")
    _lh, _ll = ax.get_legend_handles_labels()
    if locals().get("short"):
        _gh, _gl = style.glyph_key(plt,
                                   [("o", True, "stopped by the wall-clock budget")])
        _lh, _ll = _lh + _gh, _ll + _gl
    style.legend_below(ax, y=-0.17, ncol=3, handles=_lh, labels=_ll)
    ax.set_title(style.fig_title(
                 f"Cost against problem size, {_law_title(law)}"
                 f"   [{arch}, {FAMILY_LABEL.get(family, family)}]", width=78),
                 fontsize=8.8)
    fig.tight_layout(rect=(0, 0.075 if style.PAPER else 0.115, 1, 1))
    _facts(_ladder_phrase([r for r in lad if r.get("law") == law
                           and r.get("family") == family]) + ".",
           "Library: the median over %s of one call with the workspace built "
           "inside it" % _reps_phrase([r for r in lad if r.get("law") == law])
           + (("; externals: the median over %s of one call."
               % _reps_phrase([r for r in (slad or []) if r.get("law") == law]))
              if slad else "."),
           _pass_sentence(lad + slad))
    _caption("Runtime scaling with the set size $N$ for the %s on the %s "
             "dataset, %s." % ("tested implementations" if slad else
                               "library backends", _size_set(law),
                               CONV_SUPPLIED.get(family, family)),
             "The plot shows the runtime cost per element, in nanoseconds, as $N$ "
             "increases; both axes are logarithmic, and the library rows include "
             "the creation of the workspace.",
             "Open circles mark the largest set size reached by implementations "
             "that were stopped by the wall-clock budget."
             if locals().get("short") else "",
             _obtained(arch))
    return _finish(fig, outdir, f"fig-paper-ladder-{law}-{family}", stamp)


def fig_workspace(plt, lad, outdir, stamp, law="size_random", family="z_nu",
                  arch=""):
    """Cost with a new and a reused workspace against N, and the allocation share."""
    # by family too: the ladder carries both conventions (two rows per rung)
    lad =[r for r in lad if r["law"] == law and r.get("family") == family]
    if not lad:
        return _skip(f"workspace-{law}-{family}", "no ladder rows")
    backs = [b for b in style.BACKEND_ORDER
             if any(r["backend"] == b for r in lad)]
    if len(backs) < 2:
        return _skip(f"workspace-{law}-{family}",
                     f"needs >=2 library backends, found {len(backs)}")
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    exhausted = {}
    below_one = {}
    no_ws = []
    # right panel: ns_workspace_med when present, else build minus reused
    _ws_med = any(_ok(_f(r, "ns_workspace_med")) and _f(r, "ns_workspace_med") > 0
                  for r in lad)
    for b in backs:
        rows = sorted((r for r in lad if r["backend"] == b),
                      key=lambda r: _f(r, "n"))
        ns = [_f(r, "n") for r in rows]
        c = style.BACKEND_COLOR.get(b, "0.4")
        m = style.BACKEND_MARK.get(b, "o")
        # no `label=`: the key is built explicitly below
        axl.plot(ns, [_f(r, "ns_ws_per_call_med") for r in rows], marker=m, ms=3.8,
                 lw=1.3, color=c)
        axl.plot(ns, [_f(r, "ns_ws_reused_med") for r in rows], marker=m, ms=3.8,
                 lw=1.3, ls=":", color=c, alpha=0.75)
        if b in NO_WORKSPACE:
            no_ws.append(b)
            continue
        # Share = workspace create+destroy / whole build-per-call.  The fallback
        # difference matches it only while N fits one tile (MATERN_TILE_DEFAULT)
        # and is truncated at its first non-positive value.
        has_ws = _ws_med and any(_ok(_f(r, "ns_workspace_med"))
                                 and _f(r, "ns_workspace_med") > 0 for r in rows)
        if has_ws:
            sv = []
            for r in rows:
                n = _f(r, "n")
                ws = _f(r, "ns_workspace_med")
                call = _f(r, "ns_ws_per_call_med") * n      # the whole build-per-call
                if _ok(ws) and _ok(call) and ws > 0 and call > 0:
                    sv.append((n, 100.0 * ws / call))
        else:
            # fallback: median build minus median reused, per element (not a share)
            sv =[(_f(r, "n"), _f(r, "ns_ws_per_call_med") - _f(r, "ns_ws_reused_med"))
                  for r in rows]
        keep = []
        for x, y in sv:
            if not (_ok(y) and y > 0.0):
                break
            keep.append((x, y))
        if keep:
            axr.plot([x for x, _ in keep], [y for _, y in keep], marker=m,
                     ms=3.8, lw=1.3, color=c)
            # ring the first rung with a share below 1.0 per cent
            for x, y in keep:
                if y < 1.0:
                    axr.plot([x], [y], marker="o", ms=8.5, mfc="none", mec=c,
                             mew=1.3, ls="none", zorder=7)
                    below_one[b] = x
                    break
            # record where a truncated series stopped
            if len(keep) < len(sv):
                exhausted[b] = keep[-1][0]
    if _ws_med:
        style.refline(axr, 1.0, "1 %", side="left")
    for ax in (axl, axr):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("$N$  [elements per call]")
    axl.set_ylabel("Cost  [ns / element]")
    axr.set_ylabel("Allocation overhead  [%]" if _ws_med
                   else "Allocation overhead  [ns / element]")
    _parts = []
    if _ws_med:
        _parts.append("the workspace is a fixed-size tile allocation, timed on its "
                      "own at the same repetition count as the two call "
                      "measurements, so its cost does not depend on $N$; the right "
                      "panel therefore reports its cost relative to the "
                      "call, and rings the rung at which it first falls below "
                      "1 %: " + (", ".join("%s at $N{=}%d$" % (b, int(x))
                                             for b, x in sorted(below_one.items()))
                                   if below_one else ""))
    if no_ws:
        _parts.append(", ".join(no_ws) + " builds no workspace "
                      "(the scalar path allocates none), so it has no setup "
                      "series and no ring, and its two call measurements "
                      "coincide")

    if exhausted:
        where = ", ".join(
            f"{b} beyond $N{{=}}2^{{{int(round(math.log(n, 2)))}}}$"
            for b, n in sorted(exhausted.items(),
                               key=lambda kv: style.BACKEND_ORDER.index(kv[0])
                               if kv[0] in style.BACKEND_ORDER else 99))
        _parts.append("ring = last rung where the setup is still resolvable; "
                      "beyond it the new-workspace minus reused-workspace cost is "
                      "smaller than the jitter of the two medians it differences, "
                      "and goes negative --- " + where)
    _note = "   |   ".join(_parts) if _parts else None
    axl.set_title(style.panel_title(
        "Runtime cost, new workspace (solid), reused workspace (dotted)"),
        pad=4, fontsize=8.4)
    # one key entry per backend; solid/dotted is stated in the left panel's title
    _h =[plt.Line2D([], [], color=style.BACKEND_COLOR.get(b, "0.4"),
                     marker=style.BACKEND_MARK.get(b, "o"), ms=3.8, lw=1.3)
          for b in backs]
    _l = list(backs)
    if below_one:
        _gh, _gl = style.glyph_key(plt,
                                   [("o", True, "allocation first below 1 %")])
        _h, _l = _h + _gh, _l + _gl
    _foot = style.footer(fig, _h, _l, note=_note, ncol=6, width=104)
    # the title names the family, so the per-family figures differ
    fig.suptitle(style.fig_title(
                 f"Runtime cost with a new and a reused workspace, and the allocation overhead   [{arch}, "
                 f"{_law_title(law)}, {FAMILY_LABEL.get(family, family)}]"),
                 y=0.995, fontsize=9.5)
    fig.tight_layout(rect=(0, _foot, 1, 0.93))
    _lr = [r for r in lad if r.get("law") == law and r.get("family") == family]
    _facts(_ladder_phrase(_lr) + ".",
           "Cost: the median over %s of one call, with the workspace built in "
           "the call and with it reused; %s."
           % (_reps_phrase(_lr),
              "the right panel divides the workspace's own create-and-destroy "
              "time (median over the same repetitions) by the median "
              "new-workspace cost of the call" if _ws_med else
              "the right panel is the median new-workspace cost minus the "
              "median reused-workspace cost, per element"),
           _pass_sentence(_lr))
    _caption("Effect of workspace allocation on the runtime of the library "
             "backends as a function of the number of elements per call $N$ for "
             "the %s dataset, %s." % (_size_set(law),
                                      CONV_SUPPLIED.get(family, family)),
             "The left panel shows the cost per element; solid lines include the "
             "allocation of a new workspace, whereas dotted lines reuse an "
             "existing one.",
             "The right panel shows the workspace allocation time as a percentage "
             "of the runtime of a first call; the dashed line marks 1%, and "
             "circles indicate the first $N$ for which the allocation share falls "
             "below it." if _ws_med else
             "The right panel shows the cost per element with a new workspace "
             "minus the cost with a reused one.",
             ("The %s backend allocates no workspace." % _and(no_ws))
             if no_ws else "",
             _obtained(arch))
    return _finish(fig, outdir, f"fig-paper-workspace-{law}-{family}", stamp)


def fig_family_delta(plt, acc, outdir, stamp, backend=None, arch="",
                     require_multi_backend=False):
    """Maximum ULP and cost per law of one backend, z supplied against z computed."""
    _pool = [r for r in acc if scope_of(r) == "call"]
    backend = lib_backend_for(_pool, prefer=backend)
    if backend is None:
        return _skip("convention", "no library accuracy rows at call scope")
    rows = [r for r in _pool if r["backend"] == backend]
    # set by the driver when the CPU rows are the GPU pipeline's one-backend stage
    if require_multi_backend and len({r["backend"] for r in acc}) < 2:
        return _skip("convention",
                     "the CPU rows of this machine are the GPU pipeline's "
                     "one-backend CPU stage; the figure is drawn from a "
                     "machine with the full CPU stage")
    by = {(r["family"], r["law"]): r for r in rows}
    laws = [l for l in ACC_LAWS
            if ("z_nu", l) in by and ("h_nu_s", l) in by]
    if not laws:
        return _skip("convention",
                     "no law carries both argument conventions (z_nu and "
                     "h_nu_s), which the figure compares")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 3.95))
    x = list(range(len(laws)))
    for fam, col in (("z_nu", "#0353a4"), ("h_nu_s", "#c1121f")):
        axa.plot(x, [_f(by[(fam, l)], "u_max") for l in laws],
                 marker=style.FAMILY_MARK[fam], ls=style.FAMILY_LS[fam],
                 color=col, ms=4.6, lw=1.3, label=FAMILY_PLOT_LABEL[fam])
        axb.plot(x, [_f(by[(fam, l)], "ns_med") for l in laws],
                 marker=style.FAMILY_MARK[fam], ls=style.FAMILY_LS[fam],
                 color=col, ms=4.6, lw=1.3)
    ratio = [_f(by[("h_nu_s", l)], "u_max") / _f(by[("z_nu", l)], "u_max")
             if _f(by[("z_nu", l)], "u_max") else float("nan") for l in laws]
    for i, rr in enumerate(ratio):
        if _ok(rr):
            axa.annotate(f"{rr:.1f}x", (i, _f(by[("h_nu_s", laws[i])], "u_max")),
                         textcoords="offset points", xytext=(0, 6),
                         fontsize=5.6, ha="center", color="0.35")
    for ax, lab in ((axa, "Maximum ULP error"), (axb, "Cost  [ns / element]")):
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LAW_LABEL.get(l, l) for l in laws],
                           rotation=35, ha="right")
        ax.set_ylabel(lab)
    style.refline(axa, style.ULP_ONE, "1 ULP")
    axa.set_title(style.panel_title("Accuracy"), pad=4)
    axb.set_title(style.panel_title("Cost"), pad=4)
    style.legend_figure(fig, axa)
    fig.suptitle(style.fig_title("The two argument conventions, same backend   "
                 f"[{arch}, {backend}]"), y=0.995, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    _rows = [r for r in (acc or []) if r.get("law") in laws]
    _facts(_acc_sentence(_rows, laws, "the maximum ULP error"),
           _cost_sentence(_rows, "median", laws))
    _caption("Effect of the argument convention on the %s backend." % backend,
             "The left panel shows the maximum ULP error and the right panel "
             "the median cost per element, in nanoseconds, for each dataset with "
             r"$(z,\nu)$ supplied and with $(h,\nu,s)$ supplied and $z$ computed; "
             "the annotations give the ratio of the maximum with $z$ computed "
             "to that with $z$ supplied.",
             _obtained(arch))
    return _finish(fig, outdir, f"fig-paper-convention-{backend}", stamp)


def _tukey_glyph(ax, i, c, q1, med, q3, wl, wh):
    """One box at x = `i`: interquartile rectangle, median bar, capped whiskers."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((i - 0.28, q1), 0.56, max(q3 - q1, 0.0),
                           facecolor=c, alpha=0.55, edgecolor=c, lw=0.9,
                           zorder=3))
    ax.plot([i - 0.34, i + 0.34], [med, med], "-", color="0.10", lw=1.4,
            zorder=5)
    ax.plot([i, i], [wl, q1], "-", color=c, lw=0.9, zorder=2)
    ax.plot([i, i], [q3, wh], "-", color=c, lw=0.9, zorder=2)
    for cap in (wl, wh):
        ax.plot([i - 0.12, i + 0.12], [cap, cap], "-", color=c, lw=0.9,
                zorder=2)


def _nearest_rank(v, p):
    """Nearest-rank quantile `p` of the sorted list `v`: v[ceil(p * len(v)) - 1]."""
    return v[max(0, int(math.ceil(p * len(v))) - 1)]


def _box_modes_at(ax, rows, modes, n, raw):
    """One Tukey box per transfer strategy over the library's repetitions at rung `n`."""
    law, fam = rows[0]["law"], rows[0]["family"]
    runs = {r.get("_run") for r in rows}
    # Keyed on `n` (gpu_ladder.csv has no `ns_points`), supplier "-" and the
    # rows' own law/family/run: the externals' repetitions share the file.
    samp = defaultdict(list)
    for r in (raw or []):
        if (r.get("device") == "cuda" and (r.get("supplier") or "-") == "-"
                and r.get("law") == law and r.get("family") == fam
                and r.get("_run") in runs and _f(r, "n") == n):
            v = _f(r, "ns_per_elem")
            if _ok(v):
                samp[r.get("backend")].append(v)
    boxes = []
    for i, m in enumerate(modes):
        r = next((q for q in rows if q["backend"] == m and _f(q, "n") == n), None)
        if r is None:
            continue
        q1, med, q3 = _f(r, "ns_q1"), _f(r, "ns_med"), _f(r, "ns_q3")
        mn, mx = _f(r, "ns_min"), _f(r, "ns_max")
        if not (_ok(q1) and _ok(med) and _ok(q3)):
            continue
        iqr = q3 - q1
        f_lo, f_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        v = samp.get(m) or [x for x in (mn, mx) if _ok(x)]
        inside = [x for x in v if f_lo <= x <= f_hi] or [q1, q3]
        out = [x for x in v if x < f_lo or x > f_hi]
        boxes.append((i, m, q1, med, q3, min(inside), max(inside), out))
    if not boxes:
        return
    hi = max(b[6] for b in boxes)
    lo = min(b[5] for b in boxes)
    pad = 0.08 * (hi - lo) if hi > lo else 0.05 * hi
    top = hi + 3.0 * pad
    for i, m, q1, med, q3, wl, wh, out in boxes:
        c = style.mode_color(m)
        _tukey_glyph(ax, i, c, q1, med, q3, wl, wh)
        far = [x for x in out if x > top]
        near = [x for x in out if x <= top]
        if near:
            ax.plot([i] * len(near), near, ls="none", marker="o", ms=3.0,
                    mfc="none", mec=c, mew=0.7, zorder=4)
        if far:
            ax.plot([i], [top - 0.4 * pad], ls="none", marker="^", ms=5.0,
                    mfc="none", mec=c, mew=1.1, zorder=4)
            ax.annotate(("1 rep at %.0f ns" % far[0]) if len(far) == 1 else
                        ("%d reps up to %.0f ns" % (len(far), max(far))),
                        (i, top - 0.4 * pad), textcoords="offset points",
                        xytext=(0, -9), ha="center", va="top", fontsize=5.6,
                        color="0.30")
    ax.set_xlim(-0.6, len(modes) - 0.4)
    ax.set_ylim(max(0.0, lo - pad), top)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=20, ha="right", fontsize=6.6)
    ax.set_ylabel("Cost  [ns / element]")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    # box-plot key on the canvas (paper mode moves notes to HARDWARE.md)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    _h = [Patch(facecolor="0.45", alpha=0.55, edgecolor="0.45", lw=0.9),
          Line2D([], [], color="0.10", lw=1.4),
          Line2D([], [], color="0.45", lw=0.9),
          Line2D([], [], ls="none", marker="o", ms=4, mfc="none", mec="0.45"),
          Line2D([], [], ls="none", marker="^", ms=5, mfc="none", mec="0.45")]
    _l = ["interquartile range", "median", "whisker (1.5 IQR)", "outlier",
          "off-axis repetitions"]
    style.legend_below(ax, y=-0.22, handles=_h, labels=_l, ncol=3)


def fig_gpu_modes_suppliers(plt, gsup, outdir, stamp, family="z_nu", arch=""):
    """Call cost per device supplier, one box per transfer strategy over the laws."""
    sup_modes = defaultdict(dict)        # {supplier: {mode: {law: ns}}}
    law_n = defaultdict(set)
    for r in (gsup or []):
        if r.get("family") != family or scope_of(r) != "call":
            continue
        bk = r.get("backend") or ""
        if "/" not in bk:
            continue
        asm, mode = bk.split("/", 1)
        v = _f(r, "ns_med")
        if not _ok(v):
            continue
        cur = sup_modes[r["supplier"]].setdefault(mode, {})
        # the cheapest assembly per supplier and law
        k = r["law"]
        if k not in cur or v < cur[k]:
            cur[k] = v
        np_ = _i(r, "ns_points") or _i(r, "n")
        if np_:
            law_n[k].add(np_)
    sup_modes = {k: v for k, v in sup_modes.items() if len(v) > 1}
    if not sup_modes:
        return _skip(f"gpu-modes-suppliers-{family}",
                     "no device supplier rows under more than one transfer "
                     "strategy at call scope")
    MODE_ORDER = ["sync", "async-direct", "async-ring", "async-mapped"]
    snames = sorted(sup_modes, key=lambda k: (style.LIB not in k, k))
    allm = [m for m in MODE_ORDER if any(m in v for v in sup_modes.values())] \
        + sorted({m for v in sup_modes.values() for m in v} - set(MODE_ORDER))
    ncol = min(3, len(snames))
    nrow = -(-len(snames) // ncol)
    fig, grid = plt.subplots(nrow, ncol, figsize=(2.45 * ncol + 0.35,
                                                  2.35 * nrow + 1.25),
                             sharey=True, sharex=True, squeeze=False)
    flat = [a for row in grid for a in row]
    for a in flat[len(snames):]:
        a.set_visible(False)
    axes = flat[:len(snames)]
    nl = set()
    lib_best = None
    for si, sname in enumerate(snames):
        ax = axes[si]
        per = sup_modes[sname]
        laws_here = set.intersection(*[set(d) for d in per.values()]) \
            if per else set()
        nl |= laws_here
        for mi, m in enumerate(allm):
            if m not in per:
                continue
            pts = [(per[m][l], l) for l in laws_here if l in per[m]]
            if not pts:
                continue
            v = sorted(p[0] for p in pts)
            q1, med, q3 = (_nearest_rank(v, 0.25), _nearest_rank(v, 0.5),
                           _nearest_rank(v, 0.75))
            # fences in log space, matching the logarithmic axis
            lq1, lq3 = math.log10(q1), math.log10(q3)
            f_lo = 10 ** (lq1 - 1.5 * (lq3 - lq1))
            f_hi = 10 ** (lq3 + 1.5 * (lq3 - lq1))
            inside = [x for x in v if f_lo <= x <= f_hi] or [q1, q3]
            c = style.mode_color(m)
            _tukey_glyph(ax, mi, c, q1, med, q3, min(inside), max(inside))
            srt = sorted(pts)
            for k, (x, l) in enumerate(srt):
                dx = (k - (len(srt) - 1) / 2.0) * (0.36 / max(1, len(srt) - 1))
                out = x < f_lo or x > f_hi
                ax.plot([mi + dx], [x], ls="none", marker="o",
                        ms=3.4 if out else 2.2,
                        mfc="none" if out else "0.15",
                        mec=c if out else "0.15", mew=0.8,
                        alpha=1.0 if out else 0.55, zorder=6)
                if out:
                    hi_ = x > f_hi
                    ax.annotate(LAW_LABEL.get(l, l), (mi + dx, x),
                                textcoords="offset points",
                                xytext=(0, 4 if hi_ else -4), ha="center",
                                va="bottom" if hi_ else "top", fontsize=5.4,
                                color="0.25")
            if style.LIB in sname:
                lib_best = med if lib_best is None else min(lib_best, med)
        ax.set_xticks(range(len(allm)))
        ax.set_xticklabels(allm, rotation=35, ha="right", fontsize=6.6)
        ax.set_xlim(-0.6, len(allm) - 0.4)
        ax.set_title(sname, fontsize=7.2, pad=3)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5, which="both")
    axes[0].set_yscale("log")
    for r_ in range(nrow):
        grid[r_][0].set_ylabel("Cost of one call\n[ns / element]")
    # sharex hides the labels of a panel above an unused slot; restore them
    for k, a in enumerate(axes):
        if k + ncol >= len(snames):
            a.xaxis.set_tick_params(labelbottom=True)
    if lib_best is not None:
        for ax in axes:
            ax.axhline(lib_best, color=style.supplier_color(style.LIB),
                       ls="--", lw=0.8, zorder=1)
    ns = sorted({n for l in nl for n in law_n.get(l, ())})
    _fmt = lambda n: f"{int(n):,}".replace(",", " ")
    ntxt = (("$N$ = " + _fmt(ns[0])) if len(ns) == 1 else
            ("$N$ = %s to %s, one $N$ per dataset" % (_fmt(ns[0]), _fmt(ns[-1]))))
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    _h = [Patch(facecolor="0.45", alpha=0.55, edgecolor="0.45", lw=0.9),
          Line2D([], [], color="0.10", lw=1.4),
          Line2D([], [], color="0.45", lw=0.9),
          Line2D([], [], ls="none", marker="o", ms=2.6, mfc="0.15", mec="0.15",
                 alpha=0.55),
          Line2D([], [], ls="none", marker="o", ms=4, mfc="none", mec="0.45"),
          Line2D([], [], color=style.supplier_color(style.LIB), ls="--",
                 lw=0.8)]
    _l = ["interquartile range over the %d datasets" % len(nl), "median",
          "whisker (1.5 IQR, log scale)", "one dataset", "outlier dataset (named)",
          f"{style.LIB_CUDA}, best strategy median"]
    fig.suptitle(style.fig_title(
        "Transfer strategy per device implementation, whole call   "
        f"[{arch}, {FAMILY_LABEL.get(family, family)}]", width=96),
        y=0.998, fontsize=8.8)
    _foot = style.footer(fig, _h, _l, ncol=3, note=(
        "per supplier and strategy: one box over the per-dataset medians of the "
        "accuracy datasets at one $N$ each (" + ntxt.replace("$N$ = ", "") + "), "
        "transfers included; not the ladder, whose top rung not every external "
        "reaches"))
    fig.tight_layout(rect=(0, _foot, 1, 0.955))
    _rows = [r for r in (gsup or []) if r.get("family") == family
             and scope_of(r) == "call"]
    _facts("Per dataset, cost is the median over %s of one call on all $N$ "
           "points of the dataset.  Each box summarises those per-dataset medians "
           "for one supplier and transfer strategy: box = interquartile "
           "range (nearest rank), bar = median, whiskers = 1.5 interquartile "
           "ranges in log space, and every dataset is drawn as a dot, outliers "
           "named.  The dashed line is the library's lowest median over the "
           "strategies." % _reps_phrase(_rows))
    _caption("Effect of the transfer strategy on the GPU call cost of each "
             "implementation for the %s datasets." % CONV.get(family, family),
             "Each panel shows one implementation, and each box summarises its "
             "median cost per element over the datasets, transfers included, "
             "under one transfer strategy%s." % (
                 "; the dashed line marks the lowest median of %s over the "
                 "strategies" % style.LIB_CUDA if lib_best is not None else ""),
             "Boxes span the interquartile range, bars mark the median, whiskers "
             "extend to 1.5 interquartile ranges on the logarithmic scale, and "
             "outlying datasets are named.",
             _obtained(arch, gpu=True))
    return _finish(fig, outdir, f"fig-paper-gpu-modes-suppliers-{family}", stamp)


def fig_gpu_modes(plt, glad, cpu_lad, outdir, stamp, law="size_random",
                  family="z_nu", arch="", raw=None):
    """Device call cost against N per transfer strategy, with the host's CPU curve."""
    # gpu_ladder.csv carries both argument conventions, so filter by family
    rows =[r for r in glad if r["law"] == law and r["family"] == family]
    if not rows:
        return _skip(f"gpu-modes-{law}-{family}",
                     "no GPU ladder rows for this law and family "
                     "(a run without a CUDA device has none)")
    dup = len(rows) - len({(r["backend"], r["n"]) for r in rows})
    if dup:
        # duplicate rungs after the family filter are reported, not aggregated
        print(f"    !! gpu ladder {law}/{family}: {dup} duplicate (backend, n) "
              f"rows --- figure would join them vertically; skipped")
        return _skip(f"gpu-modes-{law}-{family}",
                     f"{dup} duplicate (backend, n) GPU ladder rows")
    modes = sorted({r["backend"] for r in rows})
    # right panel: the library's strategies at the largest rung all of them reach
    _cnt = defaultdict(set)
    for r in rows:
        if _ok(_f(r, "ns_med")):
            _cnt[_f(r, "n")].add(r["backend"])
    _full = [n for n, ms in _cnt.items() if len(ms) == len(modes)]
    top_n = max(_full) if _full else None

    if top_n is not None:
        fig, (ax, axs) = plt.subplots(
            1, 2, figsize=(11.4, 4.9), gridspec_kw={"width_ratios": [1.45, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(6.8, 4.9))
        axs = None
    curves = {}
    for m in modes:
        p = sorted((_f(r, "n"), _f(r, "ns_med"))
                   for r in rows if r["backend"] == m)
        p = [(a, b) for a, b in p if _ok(a) and _ok(b)]
        if not p:
            continue
        curves[m] = p
        ax.plot([a for a, _ in p], [b for _, b in p], marker="o", ms=3.8,
                lw=1.4, color=style.mode_color(m), label=m)

    # the same machine's CPU curve, and the crossing
    cl =[r for r in (cpu_lad or [])
          if r["law"] == law and r["family"] == family]
    backs = [b for b in ("avx512", "avx2", "matched_fma", "matched", "scalar")
             if any(r["backend"] == b for r in cl)]
    cpu_pts = []
    if backs:
        b = backs[0]                      # the best backend this run resolved
        # build-per-call cost, since the GPU side is also the whole call
        cpu_pts = sorted((_f(r, "n"), _f(r, "ns_ws_per_call_med"))
                         for r in cl if r["backend"] == b)
        cpu_pts = [(x, y) for x, y in cpu_pts if _ok(x) and _ok(y)]
    if not cpu_pts:
        style.note_below(fig,
                "no same-machine CPU ladder for this convention in this run, "
                "so no CPU curve is drawn",
                color="#b3261e")
    if cpu_pts:
        ax.plot([x for x, _ in cpu_pts], [y for _, y in cpu_pts], marker="D",
                ms=4.0, lw=1.7, color=style.BACKEND_COLOR.get(backs[0], "#c1121f"),
                label=f"CPU {backs[0]} (same machine)", zorder=6)
        # Crossing: the first rung where the best GPU mode is below the CPU,
        # interpolated in log-log from the rung before it.
        best = {}
        for p in curves.values():
            for x, y in p:
                if x not in best or y < best[x]:
                    best[x] = y
        cpu = dict(cpu_pts)
        shared = sorted(set(best) & set(cpu))
        cross = None
        for i in range(1, len(shared)):
            x0, x1 = shared[i - 1], shared[i]
            d0, d1 = best[x0] - cpu[x0], best[x1] - cpu[x1]
            if d0 > 0 >= d1:
                t = d0 / (d0 - d1) if (d0 - d1) else 0.0
                lx = math.log(x0) + t * (math.log(x1) - math.log(x0))
                cross = math.exp(lx)
                break
        # A CPU cost at the crossing above 1.5 times its ladder minimum labels
        # the crossing as the CPU curve rising, not the device overtaking it.
        cpu_min = min(y for _, y in cpu_pts)
        cpu_at_cross = None
        if cross:
            near = min(cpu_pts, key=lambda p: abs(math.log(p[0]) - math.log(cross)))
            cpu_at_cross = near[1]
        cpu_gave_way = (cpu_at_cross is not None
                        and cpu_at_cross > 1.5 * cpu_min)
        if cross:
            ax.axvline(cross, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=1)
            # Label left of the line past the ladder's log midpoint, taken from
            # the data: the x axis is not yet logarithmic and its limits can be <= 0.
            _xs = [x for x, _ in cpu_pts]
            _mid = math.exp(0.5 * (math.log(min(_xs)) + math.log(max(_xs))))
            _right = cross < _mid
            _ha = "left" if _right else "right"
            _dx = 5 if _right else -5
            if cpu_gave_way:
                ax.annotate(f"curves meet at N $\\approx$ "
                            f"{int(round(cross)):,}".replace(",", " ")
                            + f"\nwhere the CPU cost is {cpu_at_cross/cpu_min:.1f}x"
                              "\nits own minimum over the ladder",
                            (cross, ax.get_ylim()[1]), textcoords="offset points",
                            xytext=(_dx, -14), fontsize=6.4, color="#b3261e",
                            va="top", ha=_ha)
            else:
                ax.annotate(f"GPU cost below CPU cost\nfrom N $\\approx$ "
                            f"{int(round(cross)):,}".replace(",", " "),
                            (cross, ax.get_ylim()[1]), textcoords="offset points",
                            xytext=(_dx, -14), fontsize=6.4, color="0.30",
                            va="top", ha=_ha)
        if not cross:
            # every nu of a `size_half*` law is half-integer (data/DATASETS.md)
            _why = (" (every $\\nu$ of this dataset is half-integer)"
                    if law.startswith("size_half") else "")
            style.note_below(fig,
                    "the CPU cost is below the device cost at every rung of "
                    "the ladder; the curves do not cross" + _why,
                    color=style.BACKEND_COLOR.get("avx512", "#c1121f"))
    # right panel: the library's strategies at the top rung, boxed
    if axs is not None:
        _box_modes_at(axs, rows, modes, top_n, raw)
        axs.set_title(style.panel_title(
            f"Largest $N$:  {style.LIB_CUDA} by transfer strategy"),
            pad=4, fontsize=7.6)
        style.note_below(fig,
            "right: one box per transfer strategy over the repetitions at the "
            "ladder's largest $N$, library only (the externals do not all reach "
            "that rung; their strategies are compared in a separate figure). "
            "Box = interquartile range, black tick = median, whiskers to the "
            "furthest repetition inside 1.5 IQR, hollow circle = outlier; a "
            "repetition above the axis is a triangle at its top edge, with "
            "its value", y=0.012)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$N$  [elements per call]")
    ax.set_ylabel("Cost  [ns / element]")
    _lh, _ll = ax.get_legend_handles_labels()
    style.legend_below(ax, y=-0.16, handles=_lh, labels=_ll)
    ax.set_title(style.fig_title(
                 "Cost against problem size, device transfer strategies and "
                 "the host CPU   "
                 f"[{arch}, {_law_title(law)}, "
                 f"{FAMILY_LABEL.get(family, family)}]", width=78), fontsize=8.8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _facts("Left: " + _ladder_phrase(rows) + ", dataset " + _law_title(law)
           + "; each point is the median over %s of one call, transfers "
           "included" % _reps_phrase(rows)
           + ("; the CPU row is the median over %s of one call with the "
              "workspace built inside it" % _reps_phrase(cl) if cl else "")
           + "." + ((" The boxed rung is $N = 2^{%d}$ = %s, the largest every "
                    "strategy reached; each box holds %s."
                    % (round(math.log2(top_n)), _num(top_n), _reps_phrase(
                        [r for r in rows if _f(r, "n") == top_n])))
                   if top_n else ""))
    _caption("Effect of the GPU transfer strategy on runtime for the %s dataset, "
             "%s, as a function of the set size $N$."
             % (_size_set(law), CONV_SUPPLIED.get(family, family)),
             "The %s shows the cost per element for the transfer strategies as "
             "$N$ increases, transfers included%s." % (
                 "left panel" if axs is not None else "plot",
                 (", together with the %s CPU backend on the same node for "
                  "reference" % backs[0]) if cpu_pts else ""),
             "The dashed vertical line marks the approximate set size at which "
             "the GPU cost falls below the CPU cost."
             if cpu_pts and cross else "",
             "The right panel summarises the cost of %s at the largest $N$ under "
             "each transfer strategy using the median, interquartile range and "
             "whiskers; outliers and off-axis repetitions are indicated "
             "separately." % style.LIB_CUDA if axs is not None else "",
             _obtained(arch, gpu=True))
    return _finish(fig, outdir, f"fig-paper-gpu-modes-{law}-{family}", stamp)


def fig_gpu_plane(plt, gsup, outdir, stamp, family="z_nu", arch=""):
    """fig_cost_accuracy_plane on the device at kernel scope (transfers excluded)."""
    rows = [r for r in gsup if r["family"] == family and scope_of(r) == "kernel"]
    if not rows:
        return _skip(f"gpu-plane-{family}",
                     "no GPU supplier rows at kernel scope "
                     "(needs a CUDA device and the device-side externals)")
    best = best_assembly(rows)
    laws = [l for l in ACC_LAWS if any(k[2] == l for k in best)]
    if not laws:
        return _skip(f"gpu-plane-{family}", "no GPU supplier row in a known law")
    ncol = 4
    nrow = (len(laws) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 2.6 * nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    handles = {}
    for ax, law in zip(axes, laws + [None] * (len(axes) - len(laws))):
        if law is None:
            # removed, not hidden (see fig_cost_accuracy_plane)
            fig.delaxes(ax)
            continue
        top = _band(ax, 0.12)
        # clipped and incomplete rows marked as in fig_cost_accuracy_plane
        _nfin, _worst_off, _nfail = [], None, 0
        for (s, fam, l), r in best.items():
            if l != law:
                continue
            x, med, mx = _f(r, "ns_med"), _f(r, "u_med"), _f(r, "u_max")
            if not (_ok(x) and _ok(med)):
                continue
            c = style.supplier_color(s)
            is_lib = s.startswith(style.LIB)
            if _ok(mx) and mx > med:
                ax.plot([x, x], [min(med, top), min(mx, top)], color=c,
                        lw=1.0, alpha=0.55, zorder=3)
            if _ok(mx):
                ax.plot([x], [top if mx >= CATASTROPHIC else mx],
                        marker="^" if mx >= CATASTROPHIC else "_",
                        ms=5.0 if mx >= CATASTROPHIC else 6.0,
                        mfc="none", mec=c, color=c, mew=1.2, ls="none", zorder=4)
                if mx >= CATASTROPHIC:
                    _nfail += 1
                    if _worst_off is None or mx > _worst_off:
                        _worst_off = mx
            incomplete = _i(r, "nonfinite") > 0
            if incomplete:
                # square ring, matching the glyph key
                ax.plot([x], [med], marker="s",
                        ms=(6.2 if is_lib else 4.4) + 4.2, mfc="none",
                        mec="0.15", mew=0.9, ls="none", zorder=7)
                _nfin.append(_i(r, "nonfinite"))
            h, = ax.plot([x], [med], marker="o" if is_lib else "s",
                         ms=6.2 if is_lib else 4.4, mfc=c,
                         mec="white" if is_lib else c, ls="none", zorder=6)
            handles.setdefault(s, h)
        _sub = ""
        if _nfin:
            _sub += "\n" + _nonfinite_phrase(len(_nfin), sum(_nfin))
        if _worst_off is not None:
            _sub += ("\n%d clipped; max $10^{%d}$ ULP"
                     % (_nfail, round(math.log10(_worst_off))))
        ax.set_xscale("log")
        ax.set_yscale("log")
        style.refline(ax, style.ULP_ONE, "1 ULP")
        ax.set_title(LAW_LABEL.get(law, law) + _sub, pad=3, fontsize=7.4)
    # every panel with nothing under it keeps its own x tick labels
    for _pi, _pa in enumerate(axes[:len(laws)]):
        if _pi + ncol >= len(laws):
            _pa.tick_params(labelbottom=True)
    for ax in axes[-ncol:]:
        ax.set_xlabel("Kernel cost  [ns / element]")
    fig.supylabel("ULP error  (median $\\rightarrow$ maximum)", fontsize=9,
                  x=0.015)
    order = sorted(handles, key=lambda s: (not s.startswith(style.LIB), s))
    _gh, _gl = style.glyph_key(plt, [("^", True,  "clipped at $10^6$ ULP"),
                                     ("_", False, "maximum"),
                                     ("s", True,  "non-finite on some points")])
    style.legend_figure(fig, handles=[handles[k] for k in order] + _gh,
                        labels=list(order) + _gl,
                        ncol=min(6, len(order) + 2))
    fig.suptitle(style.fig_title(
                 f"Device cost against accuracy, kernel scope   [{arch}, "
                 f"{FAMILY_LABEL.get(family, family)}]"), y=0.998, fontsize=9.5)
    style.note_below(fig,
             "one panel = one dataset = one $N$:  each kernel sits at a fixed cost, "
             "and the bar is its error distribution --- "
             "filled marker = median ULP, tick = max, "
             "hollow triangle = clipped at $10^6$ ULP (the hatched band), "
             "square frame = did not return a finite number on every "
             "point "
             "(such a row is scored over fewer points, so its max is not "
             "comparable with the others)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.995))
    _facts(_acc_sentence(rows, laws, "median (marker) and maximum (tick) of "
                         "the ULP error"),
           _cost_sentence(rows, "median", laws))
    _caption("Overview of GPU kernel cost and accuracy for the %s datasets."
             % CONV.get(family, family),
             "Each panel compares the implementations by kernel cost, measured in "
             "nanoseconds per element, and accuracy, summarised by the median "
             "(marker) and maximum (tick) ULP error; both axes are "
             "logarithmic.",
             "For implementations producing non-finite results, the panel title "
             "reports the number of affected rows and points; these points are "
             "excluded from the ULP statistics.",
             "The panel title also reports the number of maxima at or above the "
             "plotting limit of $10^6$ ULP and the largest ULP error, and "
             "off-scale maxima are drawn in the hatched band at the top of each "
             "panel.",
             _obtained(arch, gpu=True))
    return _finish(fig, outdir, f"fig-paper-gpu-plane-{family}", stamp)


# Five of the data set's nine offsets delta (the ramp separates at most five
# colours); delta = 0, where the short-cut is exact, is drawn in grey.
BOOST_SC_DELTAS = [2.0 ** -51, 1e-9, 1e-6, 1e-4, 1e-2]
BOOST_SC_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BOOST_SC_MARK = ["o", "s", "^", "D", "v"]          # second encoding beside the ramp
BOOST_SC_FLOOR = 1e-17                            # plotted floor: rel == 0 sits here
# error cap of the short-cut at the edge of its gate, 0.33 eps^(1/4)
BOOST_SC_CAP = {"double": 0.33 * 2.0 ** -13, "long-double": 0.33 * 2.0 ** (-63 / 4)}


def _same_delta(stored, wanted):
    """Stored delta is fl(2.5 + wanted) - 2.5: match by closeness, not equality."""
    return abs(stored - wanted) <= 1e-3 * wanted + 2.3e-16


def fig_gpu_boost_shortcut(plt, rows, outdir, stamp, arch=""):
    """Relative error of Boost's K_nu near nu = 2.5 against z: host, device, adapted."""
    if not rows:
        return _skip("gpu-boost-shortcut",
                     "no gpu_boost_shortcut.csv rows (needs Boost >= 1.87, a CUDA "
                     "device, and data/boost_shortcut/)")
    order = [("cpu", False), ("cuda", False), ("cuda", True)]
    panels = []
    for dev, adapted in order:
        sel = [r for r in rows if r.get("device") == dev
               and r.get("supplier", "").endswith("-adapted") == adapted]
        if sel:
            panels.append((dev, adapted, sel))
    if not panels:
        return _skip("gpu-boost-shortcut", "no row matches a known device/variant")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    handles = {}
    for ax, (dev, adapted, sel) in zip(axes, panels):
        prec = sel[0].get("precision", "double")
        pts = []
        for r in sel:
            try:
                pts.append((float.fromhex(r["delta"]), float.fromhex(r["z"]),
                            float(r["rel"])))
            except (KeyError, ValueError):
                continue
        curves = [(0.0, "0.55", None, r"$\delta=0$  ($\nu=2.5$ exactly)", 0.9)]
        curves += [(d, c, m, (r"$\delta=$1 ULP" if d == 2.0 ** -51 else
                              r"$\delta=10^{%d}$" % round(math.log10(d))), 1.4)
                   for d, c, m in zip(BOOST_SC_DELTAS, BOOST_SC_RAMP, BOOST_SC_MARK)]
        for d, col, mk, lab, lw in curves:
            xy = sorted((z, max(e, BOOST_SC_FLOOR) if _ok(e) else float("nan"))
                        for dd, z, e in pts
                        if (dd == 0.0 if d == 0.0 else _same_delta(dd, d)))
            if not xy:
                continue
            h, = ax.plot([a for a, _ in xy], [b for _, b in xy], color=col, lw=lw,
                         marker=mk, ms=3.0 if mk else 0, markeredgewidth=0,
                         markevery=6,
                         ls="-" if mk else (0, (3, 2)), zorder=4 if mk else 3)
            handles.setdefault(lab, h)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(BOOST_SC_FLOOR * 0.6, 3e-4)
        style.refline(ax, 2.0 ** -53, "1/2 ULP", side="right")
        if not adapted and prec in BOOST_SC_CAP:
            style.refline(ax, BOOST_SC_CAP[prec],
                          "short-cut cap %.0e" % BOOST_SC_CAP[prec], side="right")
        where = "host CPU" if dev == "cpu" else "device"
        how = prec.replace("-", " ") + (", recurrence rule" if adapted else "")
        # appended after panel_title, which would capitalise the supplier name
        ax.set_title(style.panel_title(f"{where}, {how}") + "\n" + sel[0]["supplier"],
                     pad=4, fontsize=7.8)
        ax.set_xlabel(r"$z$")
    axes[0].set_ylabel(r"Relative error of $K_\nu(z)$")
    labs = list(handles)
    vers = sorted({m.group(0) for _d, _a, sel in panels for r in sel
                   for m in [re.search(r"\d+(?:\.\d+)+", r.get("supplier", ""))]
                   if m})
    ver = _and(vers)
    fig.suptitle(style.fig_title(
        r"Boost's $K_\nu$ beside $\nu=5/2$: the Hankel short-cut of Boost 1.86+   "
        f"[{arch}, Boost {ver}]"), y=0.995, fontsize=9.5)
    _foot = style.footer(fig, [handles[k] for k in labs], labs, ncol=3, width=110, note=
        r"one curve per offset $\delta=\nu-2.5$, scored on $K_\nu$ itself against arb "
        r"($\geq$110 bits; no covariance assembly).  The short-cut switches on once "
        r"$z\gtrsim\delta/(5\cdot10^{-5})$ in double, and the error then falls as "
        r"$\approx0.4\,\delta/z$ for $z\ll1$ and $\approx\delta/z^4$ for $z\gg1$.  "
        r"In long double the gate is narrower and the error remains.  The CPU "
        r"supplier rows use the Boost found by the compiler (the version is part of "
        r"the supplier name); Boost 1.75.0 predates the short-cut.  The first panel "
        r"shows what that row computes with the Boost version named in the title.  "
        r"The recurrence $K_\nu=K_{\nu-2}+2(\nu-1)K_{\nu-1}/z$, taken wherever the "
        r"gate could fire (except $\delta=0$, where the short-cut is exact), removes "
        r"the error.")
    # explicit placement for the tall footer, plus room for the tick and z labels
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=_foot + 0.11,
                        wspace=0.10)
    _nd = len({r.get("delta") for r in rows}); _nz = len({r.get("z") for r in rows})
    _np = max((sum(1 for r in rows if (r.get("device"), r.get("supplier")) == k)
               for k in {(r.get("device"), r.get("supplier")) for r in rows}), default=0)
    _facts("$N$ = %s points per panel (%d offsets $\\delta$ x %d values of $z$, "
           "data/boost_shortcut); every point is drawn and is one evaluation "
           "of $K_\\nu$; no statistic over repetitions is taken, and the stage "
           "is not timed." % (_num(_np), _nd, _nz))
    _what = []
    for dev, adapted, sel in panels:
        prec = sel[0].get("precision", "double").replace("-", " ")
        _what.append("the adapted GPU implementation using the recurrence rule"
                     if adapted else
                     "the %s implementation in %s precision"
                     % ("host CPU" if dev == "cpu" else "GPU", prec))
    _cap = any(not a and s[0].get("precision", "double") in BOOST_SC_CAP
               for _d, a, s in panels)
    _caption(r"Effect of the Boost implementation and evaluation strategy on the "
             r"relative error of $K_\nu(z)$ near $\nu=2.5$.",
             "The panels compare %s (Boost %s)." % (_and(_what), ver),
             r"The curves correspond to different offsets $\delta$ from "
             r"$\nu=2.5$, including the half-integer case $\delta=0$.",
             "The dashed reference marks 1/2 ULP%s; both axes are logarithmic."
             % (", while the labelled horizontal lines in the panels without the "
                "recurrence rule mark the error cap of the short-cut"
                if _cap else ""),
             _obtained(arch, gpu=True))
    return _finish(fig, outdir, "fig-paper-gpu-boost-shortcut", stamp)


def fig_machine_independence(plt, runs_acc, outdir, stamp):
    """Maximum ULP and cost of each library row on one machine against another."""
    if len(runs_acc) < 2:
        return _skip("machine-independence",
                     f"needs accuracy rows from 2 machines, found "
                     f"{len(runs_acc)}")
    (na, A), (nb, B) = runs_acc[0], runs_acc[1]
    ka = {(r["family"], r["law"], r["backend"]): r for r in A
          if scope_of(r) == "call"}
    kb = {(r["family"], r["law"], r["backend"]): r for r in B
          if scope_of(r) == "call"}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return _skip("machine-independence",
                     "the two machines share no (family, law, backend) row")
    # identical means every accuracy column agrees, not only the plotted ones
    ACC_COLS = ["min_bits", "u_min", "u_q1", "u_med", "u_q3", "u_p99",
                "u_p999", "u_max", "u_mean", "u_iqr", "u_fence",
                "beyond_fence", "u_med_nontrivial", "rel_max", "abs_max",
                "worst_nu", "worst_arg", "n_scored", "nonfinite", "exact"]
    def _identical(k):
        return all(ka[k].get(c) == kb[k].get(c) for c in ACC_COLS)
    ident = {k: _identical(k) for k in keys}
    same = sum(1 for k in keys if ident[k])
    differing = sorted({(k[2], k[1]) for k in keys if not ident[k]})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.0, 4.3))
    for k in keys:
        c = style.BACKEND_COLOR.get(k[2], "0.4")
        m = style.FAMILY_MARK.get(k[0], "o")
        # a row that differs in any column is drawn hollow
        kw = dict(marker=m, ms=4.6, ls="none")
        if ident[k]:
            kw.update(mfc=c, mec="0.3", mew=0.4)
        else:
            kw.update(mfc="none", mec=c, mew=1.4, ms=7.0)
        axl.plot([_f(ka[k], "u_max")], [_f(kb[k], "u_max")], **kw)
        axr.plot([_f(ka[k], "ns_med")], [_f(kb[k], "ns_med")], **kw)
    for ax, lab in ((axl, "Maximum ULP error"), (axr, "Cost  [ns / element]")):
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls=(0, (4, 3)),
                zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{na}   {lab}")
        ax.set_ylabel(f"{nb}   {lab}")
    # paper mode: the count goes to HARDWARE.md and the panel keeps a plain title
    _same = (f"accuracy: {same} of {len(keys)} rows identical in every "
             f"accuracy column")
    _facts(_same)
    axl.set_title(style.panel_title("accuracy" if style.PAPER else _same),
                  pad=4, fontsize=8.6)
    if differing:
        movers = sorted({c for k in keys if not ident[k]
                         for c in ACC_COLS if ka[k].get(c) != kb[k].get(c)})
        which = ", ".join(f"{b} on {LAW_LABEL.get(l, l)}"
                          for b, l in differing[:4])
        style.note_below(fig,
                 f"hollow markers are the {len(differing)} rows that differ "
                 f"({which}); they differ only in {', '.join(movers)}",
                 color="0.25")
    axr.set_title(style.panel_title("cost on each machine"), pad=4)
    hs = [plt.Line2D([], [], ls="none", marker="o", ms=4.6,
                     mfc=style.BACKEND_COLOR.get(b, "0.4"), mec="0.3",
                     label=b) for b in style.BACKEND_ORDER]
    _gh, _gl = style.glyph_key(
        plt, [(style.FAMILY_MARK[f], False, FAMILY_PLOT_LABEL[f])
              for f in ("z_nu", "h_nu_s") if any(k[0] == f for k in keys)]
        + [("o", True, "differs between the machines")])
    style.legend_figure(fig, handles=hs + _gh,
                        labels=[h.get_label() for h in hs] + _gl, y=0.062)
    fig.suptitle(style.fig_title("Worst-case error and cost on two CPU architectures"),
                 y=0.995, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.135, 1, 0.93))
    _facts(r"Circles are the $(z,\nu)$ supplied convention, squares the "
           r"$(h,\nu,s)$ convention with $z$ computed; the colour is the "
           r"backend.",
           _acc_sentence(A, None, "the maximum ULP error"),
           _cost_sentence(A, "median"))
    import runs as mp
    _cpus = [mp.machine_name(m)[0] or m for m in (na, nb)]
    _caption("Performance of the library on the %s and %s CPUs." % tuple(_cpus),
             "The left plot compares the maximum ULP error and the right plot "
             "the median cost in nanoseconds per element of each backend and "
             r"dataset on the two machines; circles denote the $(z,\nu)$ "
             r"datasets, squares the $(h,\nu,s)$ datasets%s." % (
                 ", and hollow markers the rows that differ between the machines"
                 if differing else ""),
             "The dashed diagonal marks equal values on both machines.")
    return _finish(fig, outdir, "fig-paper-machine-independence", stamp)


# The accuracy columns of a Levin A/B row, the fields the run's Levin cross-check
# compares.  Accuracy is deterministic, so a dataset whose columns agree in both
# arms has no point in the band: both builds evaluate the same code there, and a
# cost difference comes from the build (code placement), not from the band.
LEVIN_ACC_KEYS = ("n_scored", "n_excluded", "nonfinite", "ref_zero",
                  "ref_subnormal", "exact", "min_bits", "u_min", "u_q1", "u_med",
                  "u_q3", "u_p99", "u_p999", "u_max", "u_mean", "u_iqr", "u_fence",
                  "beyond_fence", "u_med_nontrivial", "rel_max", "abs_max",
                  "worst_nu", "worst_arg")


def levin_reaches(on, off):
    """Whether the Levin band changes a dataset: both arms present and an
    accuracy column differs between them."""
    return bool(on and off) and any(on.get(k) != off.get(k)
                                    for k in LEVIN_ACC_KEYS)


def fig_levin(plt, lev, outdir, stamp, device="cpu", backend="avx512", arch=""):
    """Median ULP, maximum ULP and cost with and without the Levin band, per law."""
    rows = [r for r in lev if (r.get("device") or "cpu").strip() == device]
    if not rows:
        return _skip(f"levin-{device}-{backend}",
                     "no Levin A/B rows (the stage is opt-in at run time)")
    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in rows)]
    if not fams:
        return _skip(f"levin-{device}-{backend}", "no family in the Levin rows")

    def arms(sub, law):
        on = [r for r in sub if r.get("law") == law
              and r.get("variant") == "levin_on"]
        off = [r for r in sub if r.get("law") == law
               and r.get("variant") == "levin_off"]
        return (on[0] if on else None, off[0] if off else None)

    if not any(r.get("backend") == backend for r in rows):
        return _skip(f"levin-{device}-{backend}",
                     f"the Levin stage carries no `{backend}` rows (present: "
                     f"{', '.join(sorted({(r.get('backend') or '?') for r in rows})) or 'none'})")

    METRICS = (("u_med", "Median error (ULP)"),
               ("u_max", "Worst-case error (ULP)"),
               ("ns_med", "Cost  [ns / element]"))

    fig, axes = plt.subplots(len(METRICS), len(fams),
                             figsize=(3.7 * len(fams), 7.6), squeeze=False)
    dropped = kept = 0
    for ci, fam in enumerate(fams):
        col = [axes[mi][ci] for mi in range(len(METRICS))]
        sub = [r for r in rows if r.get("family") == fam
               and r.get("backend") == backend]
        order = [l for l in ACC_LAWS if any(r.get("law") == l for r in sub)]
        order += sorted({r.get("law") for r in sub} - set(order))

        laws = []
        for law in order:
            on, off = arms(sub, law)
            if levin_reaches(on, off):
                laws.append((law, on, off))
            elif on and off:
                dropped += 1
        kept += len(laws)

        ypos, ylab = [], []
        span = {ax: [] for ax in col}
        for y, (law, on, off) in enumerate(laws):
            ypos.append(y)
            ylab.append(LAW_LABEL.get(law, law))
            for ax, (key, _lbl) in zip(col, METRICS):
                a, b = _f(on, key), _f(off, key)
                if not (_ok(a) and _ok(b) and a > 0 and b > 0):
                    continue
                # red: larger without the band; green: smaller
                cc ="#c0392b" if b > a else "#27ae60"
                ax.plot([a, b], [y, y], "-", color=cc, lw=1.6, zorder=1,
                        solid_capstyle="round")
                ax.plot([a], [y], "o", color="#2c3e50", ms=5.0, zorder=3,
                        label="routes Levin" if (y == 0 and ci == 0) else None)
                ax.plot([b], [y], "s", color="#e67e22", ms=4.6, zorder=3,
                        label="does not route Levin" if (y == 0 and ci == 0) else None)
                span[ax].extend([a, b])
                # label the ratio only where it differs from 1 by at least 0.005
                if abs(b / a - 1.0) >= 0.005:
                    ax.annotate(r"$\times$%.2f" % (b / a), xy=(max(a, b), y),
                                xytext=(4, 0), textcoords="offset points",
                                fontsize=5.6, va="center", color=cc)

        for ax, (_k, lab) in zip(col, METRICS):
            ax.set_yticks(ypos)
            ax.set_yticklabels(ylab, fontsize=6.4)
            ax.invert_yaxis()
            v = [x for x in span[ax] if x > 0]
            if v and max(v) / min(v) >= 10.0:
                ax.set_xscale("log")
                # suppress minor tick labels, which overlap at this width
                import matplotlib.ticker as _tk
                ax.xaxis.set_minor_formatter(_tk.NullFormatter())
            elif v:
                lo, hi = min(v), max(v)
                pad = 0.18 * (hi - lo) if hi > lo else max(0.1 * hi, 1e-9)
                ax.set_xlim(lo - pad, hi + 2.2 * pad)
            ax.set_xlabel(lab, fontsize=7)
            ax.grid(True, axis="x", alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=6.2)
            if not ypos:
                ax.text(0.5, 0.5, "no law reaches the band", ha="center",
                        va="center", transform=ax.transAxes, fontsize=7,
                        color="0.45")
        col[0].set_title(FAMILY_PLOT_LABEL.get(fam, fam), fontsize=7.5)

    if kept == 0:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
        return _skip(f"levin-{device}-{backend}",
                     "no law in the Levin rows has both arms of the A/B "
                     "(band included and excluded)")

    # what the default build routes, from the `variant` and `shipped` columns
    # (it differs by backend and by device)
    ship ={r.get("variant") for r in rows
            if r.get("shipped") == "yes" and r.get("backend") == backend}
    ship = (list(ship)[0] if len(ship) == 1 else "?")
    dev = "CPU" if device == "cpu" else "CUDA"
    fig.suptitle(style.fig_title(
        "The Levin band, included and excluded --- %s, %s%s\n"
        "the default build %s the band; %d datasets reach it and %d do not "
        "(omitted)\n"
        "red = larger without the band,  green = smaller without the band, "
        "on all three panels"
        % (dev, backend, ("   [%s]" % arch) if arch else "",
           "routes" if ship == "levin_on" else "does not route", kept, dropped)),
        fontsize=8.0)
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        style.legend_figure(fig, handles=h, labels=l, ncol=4, y=0.012)
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    _facts(_acc_sentence(rows, None, "median and maximum of the ULP error"),
           _cost_sentence(rows, "median"))
    _caption("Effect of Levin routing on the accuracy and runtime of the %s "
             "implementation%s." % (
                 "CUDA" if device == "cuda" else backend,
                 (" with the %s transfer strategy" % backend)
                 if device == "cuda" else ""),
             _columns(fams),
             "From top to bottom, the rows show the median ULP error, the "
             "worst-case ULP error and the median runtime cost per element%s "
             "for the datasets with points in the Levin band, those on which the "
             "two builds differ in an accuracy statistic; on the other datasets "
             "both builds evaluate the same code." % (
                 ", host-device transfers included," if device == "cuda" and all(
                     scope_of(r) == "call" for r in rows
                     if r.get("backend") == backend) else ""),
             "Circles denote results with Levin routing and squares results "
             "without it; the annotated factors give the ratio of the value "
             "without Levin routing to that with it, in red where it is larger "
             "without and in green where it is smaller.",
             ("The default build %s the Levin band."
              % ("routes" if ship == "levin_on" else "does not route"))
             if ship != "?" else "",
             _obtained(arch, gpu=device == "cuda"))
    return _finish(fig, outdir, f"fig-paper-levin-{device}-{backend}", stamp)


# The twin's definition, shared with the fold tables (paper_tables.py).
FOLD_TWIN_SENTENCE = (
    "Unfolded twin: the scalar backend with the Mat\\'ern prefactor applied to "
    "$K_\\nu(z)$ outside the kernels, directly or through exp-log.")

# The twin in detail, for HARDWARE.md and the fold tables' README sections; the math
# uses only constructs that paper_tables._demath turns into plain text.
_FOLD_TWIN_DETAIL = (
    "The unfolded twin has the scalar backend's domain decomposition, series "
    "and stopping rules; each region returns $K_\\nu(z)$ by its own method and "
    "the Mat\\'ern prefactor is applied outside, directly as "
    "$2(z/2)^\\nu K_\\nu(z)/\\Gamma(\\nu)$ or as the exponential of "
    "$(1-\\nu) \\log 2 - \\log \\Gamma(\\nu) + \\nu \\log z + \\log K_\\nu(z)$; "
    "a finite value is clamped to $[0,1]$ where the backend clamps, an "
    "overflow stays non-finite.")


def fig_fold(plt, fold, outdir, stamp, arch=""):
    """Median ULP, maximum ULP and cost per law: scalar backend and unfolded twins."""
    rows = [r for r in fold if (r.get("device") or "cpu").strip() == "cpu"]
    if not rows or not any(r.get("backend") == "scalar" for r in rows):
        return _skip("fold", "no prefactor-fold rows with the scalar backend "
                             "(the stage is opt-in at run time)")
    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in rows)]
    # the control (folded source built as the twin) is not drawn; see tab-fold-cpu
    arms =[a for a in style.FOLD_ARMS if a != "scalar-control"
            and any(r.get("backend") == a for r in rows)]
    METRICS = (("u_med", "Median error (ULP)"),
               ("u_max", "Worst-case error (ULP)"),
               ("ns_med", "Cost  [ns / element]"))

    fig, axes = plt.subplots(len(METRICS), len(fams),
                             figsize=(4.1 * len(fams), 9.4), squeeze=False)
    seen = set()
    for ci, fam in enumerate(fams):
        col = [axes[mi][ci] for mi in range(len(METRICS))]
        sub = [r for r in rows if r.get("family") == fam]
        laws = [l for l in ACC_LAWS if any(r.get("law") == l for r in sub)]
        laws += sorted({r.get("law") for r in sub} - set(laws))
        for ax, (key, lab) in zip(col, METRICS):
            pts = []                                 # (y, arm, value, row)
            for y, law in enumerate(laws):
                for a in arms:
                    r = next((x for x in sub if x.get("law") == law
                              and x.get("backend") == a), None)
                    v = _f(r, key) if r else float("nan")
                    if _ok(v) and v > 0:
                        pts.append((y, a, v, r))
            fin = [v for _y, _a, v, _r in pts if v < CATASTROPHIC]
            if not fin:
                continue
            band = any(v >= CATASTROPHIC for _y, _a, v, _r in pts)
            lo, hi = min(fin), max(fin)
            if band:
                ax.set_xscale("log")
                xfail = _band_x(ax, lo / 2.0)
            elif hi / lo >= 10.0:
                ax.set_xscale("log")
            else:
                pad = 0.18 * (hi - lo) if hi > lo else max(0.1 * hi, 1e-9)
                ax.set_xlim(lo - pad, hi + 2.6 * pad)
            if ax.get_xscale() == "log":
                import matplotlib.ticker as _tk
                ax.xaxis.set_minor_formatter(_tk.NullFormatter())
            for y, _law in enumerate(laws):
                xs = [(xfail if (band and v >= CATASTROPHIC) else v)
                      for yy, _a, v, _r in pts if yy == y]
                if len(xs) > 1:
                    ax.plot([min(xs), max(xs)], [y, y], "-", color="0.75",
                            lw=1.1, zorder=1, solid_capstyle="round")
            for y, a, v, _r in pts:
                x = xfail if (band and v >= CATASTROPHIC) else v
                ax.plot([x], [y], style.FOLD_MARK[a], ms=5.0,
                        mfc=style.FOLD_COLOR[a], mec=style.FOLD_COLOR[a], mew=1.0,
                        zorder=4 if a == "scalar" else 3,
                        ls="none", label=style.FOLD_LABEL[a] if a not in seen else None)
                seen.add(a)
            ax.set_yticks(range(len(laws)))
            ax.set_yticklabels([LAW_LABEL.get(l, l) for l in laws], fontsize=6.4)
            ax.set_ylim(len(laws) - 0.4, -0.6)
            ax.set_xlabel(lab, fontsize=7)
            ax.grid(True, axis="x", alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=6.2)
        col[0].set_title(FAMILY_PLOT_LABEL.get(fam, fam), fontsize=7.5)

    fig.suptitle(style.fig_title(
        "The Matérn prefactor folded into the kernels or applied "
        "outside, scalar backend%s\n"
        "same domain decomposition; the twin applies the factors to "
        "$K_\\nu$ directly or through exp-log"
        % (("   [%s]" % arch) if arch else "")), fontsize=8.0)
    hh, ll = [], []
    for ax in axes.ravel():
        for hx, lx in zip(*ax.get_legend_handles_labels()):
            if lx not in ll:
                hh.append(hx); ll.append(lx)
    if hh:
        style.legend_figure(fig, handles=hh, labels=ll, ncol=4, y=0.012)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))
    _facts(_FOLD_TWIN_DETAIL,
           _acc_sentence(rows, None, "median and maximum of the ULP error"),
           "A maximum at or above $10^6$ ULP sits in the hatched band; a row with "
           "non-finite results is scored over its finite points (counts in the "
           "`nonfinite` column of the run's `fold.csv`).",
           _cost_sentence(rows, "median"))
    _caption("Effect of folding the Mat\\'ern prefactor into the kernels of the "
             "scalar backend.",
             _columns(fams),
             "From top to bottom, the rows show the median ULP error, the "
             "worst-case ULP error and the runtime cost per element.",
             "Circles denote the folded backend, triangles the unfolded twin "
             "applying the prefactor directly and squares the unfolded twin "
             "applying it through exp-log; maxima at or above $10^6$ ULP are "
             "drawn in the hatched band.",
             _obtained(arch))
    return _finish(fig, outdir, "fig-paper-fold", stamp)


def fig_fold_ecdf(plt, fold, outdir, stamp, family="z_nu", arch=""):
    """`_accuracy_ecdf` for the folded scalar backend and its twins (not the control)."""
    # The control is not drawn: its ULP histogram equals the scalar backend's.
    rows = [r for r in fold if r.get("family") == family
            and (r.get("device") or "cpu").strip() == "cpu"]
    names, bylaw = [], {}
    for a in ("scalar", "scalar-unfolded-direct", "scalar-unfolded-explog"):
        got = {r["law"]: r for r in rows if r.get("backend") == a
               and r.get("law") in ACC_LAWS}
        if got:
            names.append(style.FOLD_LABEL[a])
            for law, r in got.items():
                bylaw.setdefault(law, {})[style.FOLD_LABEL[a]] = r
    stem = f"fold-ecdf-{family}"
    if style.FOLD_LABEL["scalar"] not in names or len(names) < 2:
        return _skip(stem, f"no prefactor-fold rows for family {family} with the "
                           "scalar backend and a twin")
    return _accuracy_ecdf(
        plt, names, bylaw, outdir, f"fig-paper-fold-ecdf-{family}",
        f"Prefactor folded or applied outside: share of points at or above each "
        f"ULP error, by dataset, {FAMILY_LABEL.get(family, family)}"
        f"   [{arch}, scalar backend]", stamp, lead=style.FOLD_LABEL["scalar"],
        cap_of=("scalar backend with the Mat\\'ern prefactor folded into the "
                "kernels and for its unfolded twins, which apply the prefactor "
                "directly and through exp-log,", family, arch, False))


def thread_integrity(thr, family=None):
    """(LaTeX sentence, ok) for the thread checks `identical` and `u_max` invariant in T."""
    rows = [r for r in (thr or []) if family is None or r.get("family") == family]
    if not rows:
        return "", True

    multi = [r for r in rows if _i(r, "threads") > 1]
    bad_re = [r for r in multi if (r.get("identical") or "") != "yes"]
    tmax = max([_i(r, "threads") for r in rows] or [1])

    # every thread gets the same inputs, so a u_max that moves with T is a defect
    series = {}
    for r in rows:
        u = _f(r, "u_max")
        if not _ok(u):
            continue
        sup = r.get("supplier") or "-"
        key = ((style.LIB + ":" + str(r.get("backend"))) if sup == "-" else sup,
               r.get("family"), r.get("law"))
        series.setdefault(key, {})[_i(r, "threads")] = u
    drift = []
    for key, byT in series.items():
        v = [x for x in byT.values() if x > 0]
        if len(v) > 1 and max(v) / min(v) > 1.000001:
            drift.append((max(v) / min(v), key))
    lib_u = [u for k, byT in series.items() if k[0].startswith(style.LIB + ":")
             for u in byT.values() if _ok(u)]

    if bad_re:
        return (r"\textbf{Re-entrancy check failed:} %d row(s) returned results "
                r"that differ from the single-threaded run; the timings of those "
                r"rows are not scaling results. " % len(bad_re), False)
    if drift:
        worst, key = max(drift)
        return (r"\textbf{The ULP score changes with the thread count} for %d "
                r"series: worst \texttt{%s} on %s, a factor %.3g between thread "
                r"counts, with identical inputs to every thread. " %
                (len(drift), style.tex_escape(str(key[0])),
                 style.tex_escape(str(key[2])), worst), False)

    s = (r"Every thread's output is bit-identical to the single-threaded run "
         r"up to $T=%d$ (\texttt{memcmp}). " % tmax)
    if series:
        s += (r"Each row's ULP score against the reference is identical at "
              r"every $T$%s. " %
              (r", this library's worst being $%.3g$ ULP" % max(lib_u)
               if lib_u else ""))
    return s, True


def fig_threads_combined(plt, thr, outdir, stamp, arch=""):
    """Weak scaling of the library and the externals against the pinned thread count."""
    if not thr:
        return _skip("threads", "no threads rows (the stage is opt-in at run time)")
    thr, _thr_med = threads_median(thr)
    if not _thr_med:
        return _skip("threads", THREADS_NO_MEDIAN)
    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in thr)]
    if not fams:
        return _skip("threads", "no family in the threads rows")
    has_vps = any(_ok(_f(r, "values_per_s")) for r in thr)

    nrow = 3 if has_vps else 2
    fig, axes = plt.subplots(nrow, len(fams),
                             figsize=(3.9 * len(fams), 2.6 * nrow + 0.6),
                             squeeze=False, sharex=True)
    loads = [_f(r, "load1") for r in thr if _ok(_f(r, "load1"))]
    cores = max([_i(r, "physical_cores") for r in thr] or [0])
    band_gap = []
    # Only T > 1 rows are checked: `identical` is `n/a` at T = 1.
    reent = [r for r in thr
             if _i(r, "threads") > 1 and (r.get("identical") or "") != "yes"]
    n_sup = 0

    def _series(fam_rows, name, supplier):
        sub = [r for r in fam_rows
               if (r.get("supplier") or "-") == supplier
               and (supplier != "-" or r.get("backend") == name)]
        return sorted(sub, key=lambda r: _i(r, "threads"))

    for ci, fam in enumerate(fams):
        fam_rows = [r for r in thr if r.get("family") == fam]
        top, mid = axes[0][ci], axes[1][ci]
        bot = axes[2][ci] if has_vps else None
        best1 = 0.0

        # library backends solid, externals dashed
        lines =[(bk, "-", style.BACKEND_COLOR.get(bk, "magenta"),
                  style.BACKEND_MARK.get(bk, "o"), "-")
                 for bk in style.BACKEND_ORDER]
        sups = sorted({r["supplier"] for r in fam_rows
                       if (r.get("supplier") or "-") not in ("-", "")})
        n_sup = max(n_sup, len(sups))
        lines += [(s, "--", style.supplier_color(s), "s", s) for s in sups]

        for name, ls, col, mk, supplier in lines:
            sub = _series(fam_rows, name, supplier)
            if not sub:
                continue
            x = [_i(r, "threads") for r in sub]
            lw = 1.3 if supplier == "-" else 1.0
            ms = 3.6 if supplier == "-" else 3.0
            al = 1.0 if supplier == "-" else 0.85
            top.plot(x, [_f(r, "speedup") for r in sub], mk + ls, color=col,
                     ms=ms, lw=lw, alpha=al, label=name)
            mid.plot(x, [100.0 * _f(r, "efficiency") for r in sub], mk + ls,
                     color=col, ms=ms, lw=lw, alpha=al)
            if bot is not None:
                y = [_f(r, "values_per_s") for r in sub]
                if any(_ok(v) for v in y):
                    bot.plot(x, y, mk + ls, color=col, ms=ms, lw=lw, alpha=al)
                    if x and x[0] == 1 and _ok(y[0]):
                        best1 = max(best1, y[0])

        lim = max([_i(r, "threads") for r in fam_rows] or [1])
        # one style and label for the ideal on all three panels, so the legend
        # merge below keeps a single entry
        _IDEAL = dict(ls="--", color="0.40", lw=0.9, label="Ideal scaling")
        top.plot([1, lim], [1, lim], zorder=0, **_IDEAL)
        top.set_xscale("log", base=2)
        top.set_yscale("log", base=2)
        top.set_ylabel("Speedup", fontsize=7)
        top.set_title(FAMILY_PLOT_LABEL.get(fam, fam), fontsize=7.5)
        top.grid(True, alpha=0.25, lw=0.5)
        top.tick_params(labelsize=6.2)

        mid.axhline(100.0, zorder=0, **_IDEAL)
        mid.set_xscale("log", base=2)
        mid.set_ylim(0, 115)
        mid.set_ylabel("Parallel efficiency (%)", fontsize=7)
        mid.grid(True, alpha=0.25, lw=0.5)
        mid.tick_params(labelsize=6.2)
        if bot is None:
            mid.set_xlabel("Pinned threads", fontsize=7)

        if bot is not None:
            # alternate cost bands (_cost_clusters on the throughput at the
            # largest thread count) are shaded
            ends = {}
            for name, ls, col, mk, supplier in lines:
                sub = _series(fam_rows, name, supplier)
                if sub and _ok(_f(sub[-1], "values_per_s")):
                    ends[name] = _f(sub[-1], "values_per_s")
            if len(ends) > 2:
                order = sorted(ends, key=lambda k: ends[k])
                cl = _cost_clusters(order, ends, gap=0.5)
                for bi, (i0, i1, lo, hi) in enumerate(cl):
                    if bi % 2 == len(cl) % 2:
                        bot.axhspan(lo / 1.35, hi * 1.35, facecolor="0.85",
                                    alpha=0.35, lw=0, zorder=0)
                # widest gap between adjacent bands, for the title note
                ratio = _widest_band_gap(ends)
                if ratio is not None:
                    band_gap.append((fam, ratio))
            # ideal from the fastest one-thread row, drawn above the data
            if best1 > 0:
                bot.plot([1, lim], [best1, best1 * lim], zorder=8, **_IDEAL)
            bot.set_xscale("log", base=2)
            bot.set_yscale("log")
            bot.set_xlabel("Pinned threads", fontsize=7)
            bot.set_ylabel("Covariance values / s", fontsize=7)
            bot.grid(True, alpha=0.25, lw=0.5, which="both")
            bot.tick_params(labelsize=6.2)

    # legend handles from the top panel plus any unique ones from the bottom
    _h, _l = axes[0][0].get_legend_handles_labels()
    if has_vps:
        _hb, _lb = axes[2][0].get_legend_handles_labels()
        for h, l in zip(_hb, _lb):
            if l not in _l:
                _h.append(h); _l.append(l)
    style.legend_figure(fig, handles=_h, labels=_l, y=0.012, ncol=4)

    note = "%d physical cores" % cores if cores else ""
    # worst ULP per library backend over thread 0's scored points; it must not vary with T
    _ub = defaultdict(float)
    for r in thr:
        if (r.get("supplier") or "-") == "-" and _ok(_f(r, "u_max")):
            _ub[r.get("backend")] = max(_ub[r.get("backend")], _f(r, "u_max"))
    _drift = not thread_integrity(thr)[1]
    if _ub:
        note += "   |   worst ULP %s%s" % (
            ", ".join("%s %.3g" % (b, _ub[b]) for b in style.BACKEND_ORDER if b in _ub),
            ", changes with the thread count" if _drift else ", the same at every T")
    if n_sup:
        note += "   |   %d external supplier(s), dashed" % n_sup
    if band_gap:
        note += "   |   widest gap between cost bands %s" % ", ".join(
            "%.1fx" % r for _, r in band_gap)
    if loads:
        # load1 includes the stage's own threads, so it is reported, not interpreted
        note += "   |   load1 = %.0f, incl. this stage's own threads" % max(loads)
    if reent:
        note += "   |   %d rows failed the re-entrancy check" % len(reent)
    fig.suptitle(style.fig_title(
        "Weak scaling against pinned thread count%s\n%s"
        % ((" --- " + arch) if arch else "", note)), fontsize=8.2)
    # caveat for the two ratio panels
    style.note_below(fig,
        "speedup and efficiency are ratios against each row's own one-thread "
        "cost; an implementation with a higher cost per element moves less "
        "data per second and can reach a higher efficiency" + (
            ".  The bottom panel shows the absolute throughput"
            if has_vps else
            ".  The throughput panel needs the values_per_s column, which "
            "these rows do not carry"), y=0.088)
    fig.tight_layout(rect=(0, 0.155, 1, 0.945))
    _npt = sorted({_i(r, "n_per_thread") for r in thr if _i(r, "n_per_thread")})
    _tl = sorted({r.get("law") for r in thr if r.get("law")})
    _facts("Each thread evaluates its own %s points (dataset %s), so $N$ = %s "
           "times the thread count; %s.  Worst ULP (per backend, in the "
           "title) is the maximum over those first points of the dataset, scored "
           "once: every thread computes the same points and `identical` "
           "checks that they agree bit for bit.  The whole dataset is scored in "
           "the accuracy figures, so this is a smaller population."
           % (" / ".join(_num(v) for v in _npt) or "the same number of",
              ", ".join(_law_title(l) for l in _tl) or "?",
              " / ".join(_num(v) for v in _npt) or "$n$",
              ("a row's time is the median over %s of the slowest thread in "
               "each repetition (repetitions start together on a barrier)"
               % _reps_phrase(thr))),
           thread_integrity(thr)[0])
    _caption("Thread-scaling behaviour of the tested implementations.",
             _columns(fams),
             "From top to bottom, the rows show the speedup relative to one "
             "thread, the parallel efficiency%s as the number of pinned threads "
             "increases." % (" and the throughput in covariance values per second"
                             if has_vps else ""),
             "The grey dashed line indicates ideal scaling%s." % (
                 ", and dashed lines with square markers are external "
                 "implementations" if n_sup else ""),
             "Some rows failed the thread integrity check, so their timings are "
             "not scaling results." if (reent or _drift) else "",
             _obtained(arch))
    return _finish(fig, outdir, "fig-paper-threads", stamp)


def fig_gpu_knobs(plt, gthr, outdir, stamp, arch=""):
    """Cost against threads per block for every transfer strategy, and against host
    staging threads for the ring strategy, the only one that stages."""
    if not gthr:
        return _skip("gpu-knobs", "no GPU knob-sweep rows")
    blk = [r for r in gthr if r.get("knob") == "block"]
    mt = [r for r in gthr if r.get("knob") == "host_threads"]
    if not blk and not mt:
        return _skip("gpu-knobs",
                     "the GPU knob sweep carries neither a block-size nor a "
                     "host-thread series")

    fams = [f for f in ("z_nu", "h_nu_s")
            if any(r.get("family") == f for r in gthr)]
    fig, axes = plt.subplots(2, max(1, len(fams)),
                             figsize=(3.5 * max(1, len(fams)), 5.9),
                             squeeze=False)

    modes = ["sync", "direct", "ring", "mapped"]
    mcol = {"sync": "#7f8c8d", "direct": "#2980b9",
            "ring": "#27ae60", "mapped": "#8e44ad"}

    for ci, fam in enumerate(fams):
        top, bot = axes[0][ci], axes[1][ci]
        for m in modes:
            sub = sorted([r for r in blk if r.get("family") == fam
                          and r.get("transfer_mode") == m],
                         key=lambda r: _i(r, "knob_value"))
            if not sub:
                continue
            top.plot([_i(r, "knob_value") for r in sub],
                     [_f(r, "ns_med") for r in sub],
                     "o-", color=mcol[m], ms=3.4, lw=1.2, label=m)
        top.set_xscale("log", base=2)
        top.set_yscale("log")
        top.set_xlabel("Threads per block", fontsize=7)
        top.set_ylabel("Cost  [ns / element]", fontsize=7)
        top.set_title(FAMILY_PLOT_LABEL.get(fam, fam), fontsize=7.5)
        top.grid(True, alpha=0.25, lw=0.5)
        top.tick_params(labelsize=6.2)

        sub = sorted([r for r in mt if r.get("family") == fam],
                     key=lambda r: _i(r, "knob_value"))
        if sub:
            bot.plot([_i(r, "knob_value") for r in sub],
                     [_f(r, "ns_med") for r in sub],
                     "s-", color=mcol["ring"], ms=3.8, lw=1.3)
        bot.set_xscale("log", base=2)
        bot.set_xlabel("Host staging threads", fontsize=7)
        bot.set_ylabel("Cost  [ns / element]", fontsize=7)
        bot.set_title(style.panel_title("Ring transfer strategy"), fontsize=6.8)
        bot.grid(True, alpha=0.25, lw=0.5)
        bot.tick_params(labelsize=6.2)

    # one key under the whole grid
    style.legend_figure(fig, axes[0][0], y=0.012)

    bad = [r for r in gthr if (r.get("identical") or "yes") != "yes"]
    note = ("all knobs bit-identical to the default call"
            if not bad else "%d rows differ from the default call" % len(bad))
    fig.suptitle(style.fig_title("Device parallelism knobs%s\n%s"
                 % ((" --- " + arch) if arch else "", note)), fontsize=8.2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    _facts("Cost: the median over %s of one call on $N$ = %s points."
           % (_reps_phrase(gthr), _n_phrase(gthr, "n") or "?"))
    _caption("Effect of GPU tuning parameters on runtime.",
             _columns(fams),
             "The top row shows the cost per element as the number of threads "
             "per block is varied for each transfer strategy, while the bottom "
             "row shows the effect of the number of host staging threads for the "
             "ring transfer strategy.",
             _obtained(arch, gpu=True))
    return _finish(fig, outdir, "fig-paper-gpu-knobs", stamp)


# Distribution figures: one drawing core each for error and cost, shared by the
# host and the device callers.
def _dist_inputs_cpu(acc, sup, family, scope, lib_backend):
    """(names, {law: {name: row}}, lib_backend) for the host figures: one library
    backend and every external at its best assembly."""
    pool = [r for r in acc if scope_of(r) == scope and r["family"] == family]
    lib_backend = lib_backend_for(pool, prefer=lib_backend) or lib_backend
    accl = [r for r in pool if r["backend"] == lib_backend]
    supl = [r for r in sup if scope_of(r) == scope and r["family"] == family]
    best = best_assembly(supl)
    bylaw = defaultdict(dict)
    for r in accl:
        bylaw[r["law"]][style.LIB] = r
    for (s, _fam, l), r in best.items():
        bylaw[l][s] = r
    names = [style.LIB] + sorted({s for (s, _f_, _l) in best})
    return names, bylaw, lib_backend


def _lib_shown(bylaw, lib_backend):
    """lib_backend when the figure has a library row, else None (no caption sentence)."""
    return lib_backend if any(style.LIB in v for v in bylaw.values()) else None


def _dist_inputs_gpu(gsup, family):
    """(names, {law: {name: row}}) for the device figures, from the kernel-scope rows
    (`gpu_kernels.csv`), where libmatern-cuda is an ordinary row."""
    rows = [r for r in gsup if r["family"] == family and scope_of(r) == "kernel"]
    best = best_assembly(rows)
    bylaw = defaultdict(dict)
    for (s, _fam, l), r in best.items():
        bylaw[l][s] = r
    have = {s for (s, _f_, _l) in best}
    lead = [s for s in have if style.LIB in s]
    names = sorted(lead) + sorted(have - set(lead))
    return names, bylaw


def _dist_laws(bylaw):
    return [l for l in ACC_LAWS if bylaw.get(l)]


def _dist_axes(plt, names, laws, height=5.4, sharex=True):
    fig, axes = plt.subplots(2, 4, figsize=(10.6, height), sharex=sharex,
                             sharey=True)
    axes = axes.ravel()
    for ax, law in zip(axes, laws + [None] * (8 - len(laws))):
        if law is None:
            fig.delaxes(ax)          # see the note in fig_cost_accuracy_plane
    return fig, axes


def _dist_yaxis(ax, names):
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[len(names) - 1 - k] for k in range(len(names))],
                       fontsize=6.4)


# left edge of the offset axis, below any expected interquartile range
_OFFSET_FLOOR = 1e-4


def _widest_band_gap(ends):
    """Largest ratio between adjacent cost bands of `ends` ({series: values per second at
    the largest thread count}); None with fewer than three series or two bands."""
    ends = {k: v for k, v in ends.items() if _ok(v) and v > 0}
    if len(ends) <= 2:
        return None
    order = sorted(ends, key=lambda k: ends[k])
    cl = _cost_clusters(order, ends, gap=0.5)
    if len(cl) < 2:
        return None
    return max(cl[k + 1][2] / cl[k][3] for k in range(len(cl) - 1))


def _cost_clusters(order, cost, gap=0.5, span=1.0):
    """[(first_index, last_index, lo, hi)] of consecutive rows of `order` (ascending
    `cost`), cut at a gap of more than `gap` decades or a width of more than `span`."""
    out, start = [], 0
    for i in range(1, len(order) + 1):
        cut = (i == len(order))
        if not cut:
            a, b = cost.get(order[i - 1]), cost.get(order[i])
            first = cost.get(order[start])
            ok = (a and b and a > 0 and b > 0
                  and math.log10(b) - math.log10(a) <= gap)
            # the span cap stops a chain of small gaps forming one wide cluster
            if ok and first and first > 0:
                ok = math.log10(b) - math.log10(first) <= span
            cut = not ok
        if cut:
            lo = cost.get(order[start]) or 0.0
            hi = cost.get(order[i - 1]) or 0.0
            out.append((start, i - 1, lo, hi))
            start = i
    return out


def _fmt_ns(v):
    if v >= 1000.0:
        return "%.1f us" % (v / 1000.0)
    return "%.0f ns" % v if v >= 10 else "%.1f ns" % v


def _accuracy_distribution(plt, names, bylaw, outdir, stem, title, stamp,
                           lead=style.LIB, cap_of=None):
    """ULP distribution per row, host and device.  Every statistic, `u_fence` and
    `beyond_fence` included, is read from the columns of wm_metrics.h, none recomputed."""
    laws = _dist_laws(bylaw)
    fig, axes = _dist_axes(plt, names, laws)
    for ax, law in zip(axes, laws):
        # the position of off-scale markers depends only on the band; the axis
        # limit is set after the panel's smallest value is known
        right = math.sqrt((CATASTROPHIC / 6.0) * CATASTROPHIC)
        seen, nfail = [], 0
        for i, name in enumerate(names):
            r = bylaw[law].get(name)
            if r is None:
                continue
            y = len(names) - 1 - i          # first implementation at the top
            c = style.supplier_color(name)
            is_lead = name == lead
            z = 5 if is_lead else 3
            lo, q1 = _f(r, "u_min"), _f(r, "u_q1")
            med, q3 = _f(r, "u_med"), _f(r, "u_q3")
            fence, mx = _f(r, "u_fence"), _f(r, "u_max")
            nout = _i(r, "beyond_fence")
            if not _ok(med):
                continue
            # a ULP distance of 0 is clamped to a tenth of an ulp for the log axis
            floor_ = style.ULP_ONE / 10.0
            lo = max(lo, floor_) if _ok(lo) else floor_
            seen.append(lo)
            ax.plot([lo, min(fence if _ok(fence) else med, right)], [y, y], "-",
                    color=c, lw=0.9, alpha=0.8, solid_capstyle="butt", zorder=z)
            if _ok(q1) and _ok(q3):
                ax.plot([max(q1, floor_), min(q3, right)], [y, y], "-", color=c,
                        lw=5.2, alpha=0.30, solid_capstyle="butt", zorder=z)
            ax.plot([med, med], [y - 0.30, y + 0.30], "-", color=c, lw=1.6,
                    zorder=z + 2)
            if nout > 0 and _ok(mx) and _ok(fence) and mx > fence:
                ax.plot([min(fence, right), min(mx, right)], [y, y],
                        ls=(0, (2.2, 1.6)), color=c, lw=0.9, alpha=0.8, zorder=z)
            # the upper percentiles, read from the harness columns
            for key, filled in (("u_p99", True), ("u_p999", False)):
                v = _f(r, key)
                if _ok(v) and v < CATASTROPHIC:
                    ax.plot([max(v, floor_)], [y], marker="D", ms=3.4,
                            mfc=c if filled else "white", mec=c, mew=0.9,
                            ls="none", zorder=z + 1)
            if _ok(mx):
                if mx >= CATASTROPHIC:
                    ax.plot([right], [y], marker=">", ms=5.0, mfc="none", mec=c,
                            mew=1.2, ls="none", zorder=z + 1)
                    nfail += 1
                else:
                    ax.plot([mx], [y], marker="x", ms=4.6, color=c, mew=1.1,
                            ls="none", zorder=z + 1)
            # same predicate as fig_cost_accuracy_plane: a row scored over fewer
            # points is ringed
            if _i(r, "nonfinite") > 0:
                ax.plot([med], [y], marker="s", ms=8.0, mfc="none", mec="0.15",
                        mew=0.9, ls="none", zorder=z + 3)
        ax.set_xscale("log")
        _band_x(ax, (min(seen) / 2.0) if seen else 1e-2)
        _dist_yaxis(ax, names)
        ax.axvline(style.ULP_ONE, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_title(LAW_LABEL.get(law, law)
                     + (("\n%d clipped" % nfail) if nfail else ""),
                     pad=3, fontsize=7.4)
        ax.grid(True, which="major", axis="x", alpha=0.30)
    for ax in axes[4:]:
        ax.set_xlabel("ULP error")
    # glyph key shared with the timing distribution figure
    from matplotlib.lines import Line2D as _L2D
    _gh = [_L2D([], [], color="0.45", lw=5.2, alpha=0.30),
           _L2D([], [], color="0.45", lw=1.6),
           _L2D([], [], color="0.45", lw=0.9),
           _L2D([], [], color="0.45", lw=0.9, ls=(0, (2.2, 1.6))),
           _L2D([], [], ls="none", marker="D", ms=3.4, mfc="0.45", mec="0.45",
                mew=0.9),
           _L2D([], [], ls="none", marker="D", ms=3.4, mfc="white", mec="0.45",
                mew=0.9),
           _L2D([], [], ls="none", marker="x", color="0.45", ms=5, mew=1.1),
           _L2D([], [], ls="none", marker=">", mfc="none", mec="0.45", ms=5),
           _L2D([], [], ls="none", marker="s", mfc="none", mec="0.15", ms=6),
           _L2D([], [], color="0.35", lw=0.8, ls=(0, (4, 3)))]
    _gl = ["Interquartile range", "Median", "Whisker", "Outlier reach",
           "99th percentile", "99.9th percentile",
           "Maximum", "Clipped", "Non-finite on some points", "1 ULP"]
    style.legend_figure(fig, handles=_gh, labels=_gl, ncol=5, y=0.012)
    fig.suptitle(style.fig_title(title), y=0.998, fontsize=9.5)
    style.note_below(fig,
             "the outlier reach is drawn only where points fell past the fence; "
             "a percentile at or above $10^{6}$ ULP is not drawn",
             y=0.075)
    fig.tight_layout(rect=(0, 0.075 if style.PAPER else 0.155, 1, 0.995))
    _rows = [r for l in bylaw for r in bylaw[l].values()]
    _facts(_acc_sentence(_rows, laws, "the box is the distribution of the "
                         "ULP error"))
    # An optional fifth element names the library backend shown.
    who, fam, arch, gpu, *lib_bk = cap_of or ("tested implementations", "", "",
                                              False)
    _caption("Distribution of ULP errors for the %s across the %s datasets."
             % (who, CONV.get(fam, fam)),
             "Each panel shows the ULP-error distribution for one dataset on a "
             "logarithmic scale; the median, interquartile range, whiskers, "
             "outlier reach, percentiles and maximum are identified in the "
             "legend.",
             "The dashed vertical line marks 1 ULP.",
             "An outlined square around the median indicates that the "
             "implementation produced non-finite results for some inputs; these "
             "are excluded from the statistics, which describe the remaining "
             "finite results.",
             "Values at or above the plotting limit of $10^6$ ULP are indicated in the "
             "hatched region, and each panel reports the number of "
             "implementations with off-scale values.",
             ("The library is shown with its %s backend." % lib_bk[0])
             if lib_bk and lib_bk[0] else "",
             _obtained(arch, gpu))
    return _finish(fig, outdir, stem, stamp)


def _timing_distribution(plt, names, bylaw, reps, outdir, stem, title, stamp,
                         device="cpu", lead=style.LIB, mode="absolute",
                         cap_of=None):
    """Cost over the timed repetitions per row, host and device: box from `ns_q1`,
    `ns_med`, `ns_q3`; whiskers and outliers from `timing_raw.*` (WM_RAW_REPS) if present."""
    laws = _dist_laws(bylaw)
    # one row order for all panels (shared y axis): median cost over the laws
    med_cost = {}
    for nm in names:
        v = [_f(bylaw[l][nm], "ns_med") for l in laws
             if nm in bylaw[l] and _ok(_f(bylaw[l][nm], "ns_med"))]
        if v:
            med_cost[nm] = sorted(v)[len(v) // 2]
    names = sorted(names, key=lambda n: med_cost.get(n, float("inf")))
    clusters = _cost_clusters(names, med_cost)
    # one x scale per panel, so an outlier on one law does not compress the others
    fig, axes = _dist_axes(plt, names, laws, sharex=False)
    n_raw = 0
    # one magnification factor for the whole figure, so spreads compare across panels
    _figK = 10.0 if mode == "magnified" else 1.0
    for ax, law in zip(axes, laws):
        hi = 0.0
        lo_seen = []
        magK = _figK
        for i, name in enumerate(names):
            r = bylaw[law].get(name)
            if r is None:
                continue
            y = len(names) - 1 - i
            c = style.supplier_color(name)
            is_lead = name == lead
            z = 5 if is_lead else 3
            mn, q1 = _f(r, "ns_min"), _f(r, "ns_q1")
            med, q3 = _f(r, "ns_med"), _f(r, "ns_q3")
            mx = _f(r, "ns_max")
            if not (_ok(med) and _ok(mn) and mn > 0):
                continue
            # keyed on `ns_points` (not `n`) and on `_run`, see _reps_index
            key =(r.get("_run"), (r.get("supplier") or "-"), law,
                   r.get("backend"), _i(r, "ns_points"))
            got = sorted(reps.get(key, []))

            # x coordinate of a cost: "absolute" the cost, "percent" per cent and
            # "offset" (log axis) the excess over the row's minimum, "magnified"
            # the log distance from the median scaled by magK for every row
            def _rel(v):
                if mode == "absolute":
                    return v
                if mode == "percent":
                    return 100.0 * (v / mn - 1.0)
                if mode == "magnified":
                    return med * math.exp(magK * math.log(v / med)) if (
                        v > 0 and med > 0) else v
                return max(v - mn, _OFFSET_FLOOR)

            # Tukey: whiskers to the extreme samples inside q1 -/+ 1.5*IQR, the
            # rest are outliers
            iqr = (q3 - q1) if (_ok(q1) and _ok(q3)) else 0.0
            f_lo, f_hi = (q1 - 1.5 * iqr), (q3 + 1.5 * iqr)
            samples = [v for _, v in got] if got else []
            if samples:
                inside = [v for v in samples if f_lo <= v <= f_hi] or samples
                w_lo, w_hi = min(inside), max(inside)
                outliers = [v for v in samples if v < f_lo or v > f_hi]
            else:
                # no per-repetition file: the summary extremes clipped to the
                # fences, a bound on the true whisker
                w_lo = max(mn, f_lo) if _ok(mn) else q1
                w_hi = min(mx, f_hi) if _ok(mx) else q3
                outliers = [v for v in (mn, mx)
                            if _ok(v) and (v < f_lo or v > f_hi)]
            if _ok(w_lo) and _ok(w_hi) and w_hi > w_lo:
                hi = max(hi, _rel(w_hi))
                ax.plot([_rel(w_lo), _rel(w_hi)], [y, y], "-", color=c, lw=0.9,
                        alpha=0.8, solid_capstyle="butt", zorder=z)
                for cap in (w_lo, w_hi):
                    ax.plot([_rel(cap)] * 2, [y - 0.16, y + 0.16], "-",
                            color=c, lw=0.9, alpha=0.8, zorder=z)
            # outliers are not magnified, which would move them off the axis
            for v in outliers:
                hi = max(hi, v)
                ax.plot([v], [y], marker="o", ms=2.6, mfc="none", mec=c,
                        mew=0.7, ls="none", zorder=z + 1)
            # the interquartile range as a filled box in the row's colour
            if _ok(q1) and _ok(q3):
                from matplotlib.patches import Rectangle as _Rect
                x0, x1 = _rel(q1), _rel(q3)
                ax.add_patch(_Rect((x0, y - 0.22), max(x1 - x0, 0.0), 0.44,
                                   facecolor=c, alpha=0.55, edgecolor=c,
                                   lw=0.8, zorder=z))
                lo_seen += [v for v in (x0, x1) if v > _OFFSET_FLOOR]
            ax.plot([_rel(med)] * 2, [y - 0.30, y + 0.30], "-", color=c,
                    lw=1.6, zorder=z + 2)
            if got:
                n_raw += 1
        # log axis except in "percent" mode
        if mode in ("absolute", "offset", "magnified"):
            ax.set_xscale("log")
            if mode == "offset":
                # a zero offset sits at _OFFSET_FLOOR; the axis starts below
                # the smallest box in the panel
                ax.set_xlim(left=(min(lo_seen) / 3.0) if lo_seen
                            else _OFFSET_FLOOR)
            # minor tick labels overlap at this width and are suppressed; the major ticks
            # are the first set that puts two ticks in view (decades, then 1 and 3, then
            # 1, 2, 3, 4 and 6 times a decade: the coefficients the formatter labels)
            import matplotlib.ticker as _tk
            _lo, _hi = ax.get_xlim()
            _dec = range(math.floor(math.log10(_lo)), math.ceil(math.log10(_hi)) + 1)
            for _subs in ((1.0,), (1.0, 3.0), (1.0, 2.0, 3.0, 4.0, 6.0)):
                if sum(_lo <= m * 10.0 ** e <= _hi for e in _dec for m in _subs) >= 2:
                    if len(_subs) > 1:
                        ax.xaxis.set_major_locator(_tk.LogLocator(subs=_subs))
                    ax.xaxis.set_minor_formatter(_tk.NullFormatter())
                    break
            else:
                # no set puts two ticks in view (a range of less than about half a
                # decade): a few evenly spaced ticks, labelled as plain numbers
                _vals = [v for v in _tk.MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
                         .tick_values(_lo, _hi) if _lo <= v <= _hi]
                ax.xaxis.set_major_locator(_tk.FixedLocator(_vals))
                ax.xaxis.set_major_formatter(
                    _tk.FuncFormatter(lambda v, _pos: "%g" % v))
                ax.xaxis.set_minor_formatter(_tk.NullFormatter())
        else:
            ax.set_xlim(left=0.0)
            ax.axvline(0.0, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=1)
        # cost clusters are named in the note, not shaded like a box
        _dist_yaxis(ax, names)
        # the panel title names the law only; the axis label names the mode
        ax.set_title(LAW_LABEL.get(law, law), pad=3, fontsize=7.4)
        ax.grid(True, which="major", axis="x", alpha=0.30)
    for ax in axes:
        ax.set_xlabel({"absolute": "Cost  [ns / element]",
                       "magnified": "Cost  [ns / element]",
                       "percent": "Per cent above that row's own minimum"}.get(
                          mode, "ns / element above that row's own minimum"),
                      fontsize=7.2)
    _OFFSET = mode == "offset"
    _band_caption = " | ".join(
        (_fmt_ns(lo) if i0 == i1 else "%s-%s" % (_fmt_ns(lo), _fmt_ns(hi)))
        for i0, i1, lo, hi in clusters)
    # glyph key: box, median, whisker and outlier
    from matplotlib.lines import Line2D as _L2D
    _sfx = ("  ($\\times$%g)" % _figK) if (mode == "magnified" and _figK > 1) else ""
    from matplotlib.patches import Patch as _Patch
    _gh = [_Patch(facecolor="0.45", alpha=0.55, edgecolor="0.45", lw=0.8),
           _L2D([], [], color="0.45", lw=1.6),
           _L2D([], [], color="0.45", lw=0.9),
           _L2D([], [], ls="none", marker="o", mfc="none", mec="0.45", ms=5)]
    # the factor labels the glyphs it scales; positions stay absolute
    _gl = ["Interquartile range" + _sfx, "Median", "Whisker" + _sfx,
           "Outlier, drawn where it occurred"]
    style.legend_figure(fig, handles=_gh, labels=_gl, ncol=4, y=0.012)
    fig.suptitle(style.fig_title(title), y=0.998, fontsize=9.5)
    # the note states the axis mode and the cost clusters
    style.note_below(fig,
             ({"magnified": "spread scaled about each row's median by the factor "
                            "in the legend; one factor for the whole figure, so "
                            "spreads compare across rows and panels alike",
               "offset":    "ns/element above each row's own minimum, its left "
                            "edge",
               "absolute": "absolute cost on one axis for every row; the "
                           "spread is not scaled",
               }.get(mode, "each row scaled to its own minimum"))
             + ".  Rows ordered by cost; cost groups (rows within a decade "
               "of each other): " + _band_caption
             + ("" if n_raw else
                ".  No per-repetition file in this run, so the whiskers are the "
                "reported extremes clipped to the fences, a bound on the true "
                "whisker"), y=0.062)
    # in paper mode the note goes to HARDWARE.md and no strip is reserved
    fig.tight_layout(rect=(0, 0.075 if style.PAPER else 0.135, 1, 0.995))
    _rows = [r for l in bylaw for r in bylaw[l].values()]
    _facts(_cost_sentence(_rows, "distribution (box, whiskers, outliers)",
                          laws, show_n=True))
    # An optional fifth element names the library backend shown.
    who, fam, arch, gpu, *lib_bk = cap_of or ("tested implementations", "", "",
                                              False)
    _caption("Distribution of the runtime cost per element over the timed "
             "repetitions for the %s across the %s datasets."
             % (who, CONV.get(fam, fam)),
             "Each panel shows one dataset on its own cost axis, with the "
             "implementations ordered by cost; boxes span the interquartile "
             "range, bars mark the median, %s, and circles mark outliers."
             % ("whiskers extend to the furthest repetition within 1.5 "
                "interquartile ranges" if n_raw else
                "whiskers extend to the recorded minimum and maximum, clipped "
                "to 1.5 interquartile ranges"),
             ("The library is shown with its %s backend." % lib_bk[0])
             if lib_bk and lib_bk[0] else "",
             _obtained(arch, gpu))
    return _finish(fig, outdir, stem, stamp)


def _reps_index(raw, family, device):
    """{(run, supplier, law, backend, points): [(rep, ns)]} for one device."""
    reps = defaultdict(list)
    for r in (raw or []):
        if r.get("device") != device or r.get("family") != family:
            continue
        key = (r.get("_run"), r.get("supplier") or "-", r.get("law"),
               r.get("backend"), _i(r, "n"))
        reps[key].append((_i(r, "rep"), _f(r, "ns_per_elem")))
    return reps


def fig_accuracy_distribution(plt, acc, sup, outdir, stamp, family="z_nu",
                              lib_backend="avx512", scope="call", arch=""):
    """ULP distribution per implementation on the host, one panel per accuracy law."""
    names, bylaw, lib_backend = _dist_inputs_cpu(acc, sup, family, scope,
                                                 lib_backend)
    if not _dist_laws(bylaw):
        return _skip(f"accuracy-distribution-{family}",
                     f"no library or supplier accuracy rows for family {family}")
    # a single implementation (the device host's one-backend CPU stage) is skipped
    if len(names) < 2:
        return _skip(f"accuracy-distribution-{family}",
                     "only one implementation on this machine (%s)"
                     % (names[0] if names else "none"))

    return _accuracy_distribution(
        plt, names, bylaw, outdir, f"fig-paper-accuracy-distribution-{family}",
        f"Distribution of the ULP error by dataset, "
        f"{FAMILY_LABEL.get(family, family)}"
        f"   [{arch}, {scope} scope, library = {lib_backend}]", stamp,
        cap_of=("tested implementations", family, arch, False,
                _lib_shown(bylaw, lib_backend)))


def fig_timing_distribution(plt, acc, sup, raw, outdir, stamp, family="z_nu",
                            lib_backend="avx512", scope="call", arch=""):
    """Cost distribution over the repetitions per implementation on the host."""
    names, bylaw, lib_backend = _dist_inputs_cpu(acc, sup, family, scope,
                                                 lib_backend)
    if not _dist_laws(bylaw):
        return _skip(f"timing-distribution-{family}",
                     f"no library or supplier timing rows for family {family}")
    # a single implementation (the device host's one-backend CPU stage) is skipped
    if len(names) < 2:
        return _skip(f"timing-distribution-{family}",
                     "only one implementation on this machine (%s)"
                     % (names[0] if names else "none"))

    return _timing_distribution(
        plt, names, bylaw, _reps_index(raw, family, "cpu"), outdir,
        f"fig-paper-timing-distribution-{family}",
        f"Distribution of the cost over the timed repetitions, "
        f"{FAMILY_LABEL.get(family, family)}"
        f"   [{arch}, {scope} scope, library = {lib_backend}]", stamp,
        cap_of=("tested implementations", family, arch, False,
                _lib_shown(bylaw, lib_backend)))


def fig_gpu_accuracy_distribution(plt, gsup, outdir, stamp, family="z_nu",
                                  arch=""):
    """`fig_accuracy_distribution` on the device, kernel scope."""
    names, bylaw = _dist_inputs_gpu(gsup, family)
    if not _dist_laws(bylaw):
        return _skip(f"gpu-accuracy-distribution-{family}",
                     "no GPU supplier rows at kernel scope "
                     "(needs a CUDA device and the device-side externals)")
    return _accuracy_distribution(
        plt, names, bylaw, outdir,
        f"fig-paper-gpu-accuracy-distribution-{family}",
        f"Distribution of the ULP error by dataset, device kernels, "
        f"{FAMILY_LABEL.get(family, family)}   [{arch}, kernel scope]", stamp,
        lead=next((n for n in names if style.LIB in n), None),
        cap_of=("tested GPU implementations", family, arch, True))


def fig_gpu_timing_distribution(plt, gsup, raw, outdir, stamp, family="z_nu",
                                arch=""):
    """`fig_timing_distribution` on the device, kernel scope."""
    names, bylaw = _dist_inputs_gpu(gsup, family)
    if not _dist_laws(bylaw):
        return _skip(f"gpu-timing-distribution-{family}",
                     "no GPU supplier rows at kernel scope")
    return _timing_distribution(
        plt, names, bylaw, _reps_index(raw, family, "cuda"), outdir,
        f"fig-paper-gpu-timing-distribution-{family}",
        f"Distribution of the cost over the timed repetitions, device kernels, "
        f"{FAMILY_LABEL.get(family, family)}   [{arch}, kernel scope]", stamp,
        device="cuda", lead=next((n for n in names if style.LIB in n), None),
        cap_of=("tested GPU implementations", family, arch, True))


# (stem, name, law, why) of every curve not drawn: runs.detail_for refuses a
# histogram whose total differs from the row's `n_scored`; the driver prints them.
SERIES_SKIPPED = []

# dash patterns beside the colour, for similar hues; the library is always solid
_ECDF_DASHES = ["-", (0, (5.0, 1.6)), (0, (1.4, 1.3)), (0, (6.0, 1.6, 1.4, 1.6)),
                (0, (3.0, 1.4))]

# right edge: one octave past the histogram's last finite edge (2^10 ulp), for the
# overflow strip
_ECDF_XMAX = 2.0 ** 11


def _accuracy_ecdf(plt, names, bylaw, outdir, stem, title, stamp,
                   lead=style.LIB, cap_of=None):
    """Share of the scored points with |ulp| >= x per law, from the recorded histogram:
    exact at the bin edges, held between them; the overflow share is marked at the right."""
    import runs as mp
    from matplotlib.lines import Line2D as _L2D
    laws = _dist_laws(bylaw)
    fig, axes = _dist_axes(plt, names, laws, height=5.8)
    style_of = {}
    k = 0
    for name in names:
        if name == lead:
            style_of[name] = ("-", 1.7, 6)
        else:
            style_of[name] = (_ECDF_DASHES[k % len(_ECDF_DASHES)], 1.0, 4)
            k += 1
    xlo = 2.0 ** mp.HIST_LO_LOG
    xover = 2.0 ** mp.HIST_HI_LOG
    nmax, drawn, skipped = 1, set(), []
    curves = []                                  # (ax, name, [(x, share)])
    for ax, law in zip(axes, laws):
        for name in names:
            r = bylaw[law].get(name)
            if r is None:
                continue
            e, why = mp.detail_for(r)
            if e is None:
                skipped.append((name, law, why))
                SERIES_SKIPPED.append((stem, name, law, why))
                print(f"   !! {stem}: {name} on {law} not drawn -- {why}")
                continue
            nmax = max(nmax, e["total"])
            curves.append((ax, name, mp.hist_tail(e)))
            drawn.add(name)
    ybot = 0.5 / nmax
    for ax, name, tail in curves:
        c = style.supplier_color(name)
        ls, lw, z = style_of[name]
        # `tail` ends with the overflow share; the share is non-increasing, so
        # the edges with a non-zero share below 2^10 form a prefix
        over = tail[-1][1]
        head = [(x, g) for x, g in tail[:-1] if g > 0]
        m = len(head)
        if m:
            # held to the next edge, then down to the overflow share or the axis floor
            xs = [x for x, _g in head] + [tail[m][0]]
            ys = [g for _x, g in head] + [over if over > 0 else ybot]
            ax.plot(xs, ys, drawstyle="steps-post", color=c, ls=ls, lw=lw,
                    zorder=z, solid_capstyle="butt")
        if over > 0:
            ax.plot([math.sqrt(xover * _ECDF_XMAX)], [over], marker=">",
                    ms=4.2, mfc=c, mec=c, ls="none", zorder=z + 1)
    for ax, law in zip(axes, laws):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, _ECDF_XMAX)
        ax.set_ylim(ybot, 1.6)
        ax.axvspan(xover, _ECDF_XMAX, facecolor="0.93", edgecolor="none",
                   zorder=0)
        ax.axvline(style.ULP_ONE, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=1)
        for yv, lab in ((1e-2, "$p_{99}$"), (1e-3, "$p_{99.9}$")):
            ax.axhline(yv, color="0.55", lw=0.6, ls=(0, (1, 2)), zorder=1)
            ax.text(xlo * 1.25, yv * 1.12, lab, fontsize=5.6, color="0.40",
                    ha="left", va="bottom", zorder=2)
        ax.set_title(LAW_LABEL.get(law, law), pad=3, fontsize=7.4)
        ax.grid(True, which="major", alpha=0.30)
        ax.tick_params(labelsize=6.6)
    for ax in axes[4:]:
        ax.set_xlabel("ULP error $x$", fontsize=7.2)
    for ax in (axes[0], axes[4]) if len(laws) > 4 else (axes[0],):
        ax.set_ylabel("Share of points with error $\\geq x$", fontsize=7.2)
    _sh = [(_L2D([], [], color=style.supplier_color(n), ls=style_of[n][0],
                 lw=style_of[n][1]), n) for n in names if n in drawn]
    _gh = [_L2D([], [], color="0.35", lw=0.8, ls=(0, (4, 3))),
           _L2D([], [], ls="none", marker=">", ms=4.2, color="0.45")]
    _gl = ["1 ULP", "Share at or above $2^{10}$ ULP (not resolved)"]
    style.legend_figure(fig, handles=[h for h, _n in _sh] + _gh,
                        labels=[n for _h, n in _sh] + _gl, ncol=6, y=0.012)
    fig.suptitle(style.fig_title(title), y=0.998, fontsize=9.5)
    note = ("each curve is the share of the scored points whose ULP error is "
            "at least $x$, from the histogram recorded with each accuracy row "
            "(48 log-spaced bins over [$2^{-10}$, $2^{10}$) ULP, one overflow bin "
            "above); exact at the bin edges and held constant between them, so "
            "it crosses the dotted lines at $10^{-2}$ and $10^{-3}$ within one "
            "bin of the 99th and 99.9th percentiles; a curve drops to the axis "
            "floor at the upper edge of the bin holding its maximum, and the "
            "share at or above $2^{10}$ ULP is marked in the grey strip")
    if skipped:
        note += ("; not drawn, because the histogram was absent or did not "
                 "cover exactly the scored points: " + ", ".join(
                     "%s on %s" % (n, LAW_LABEL.get(l, l)) for n, l, _w in skipped))
    style.note_below(fig, note, y=0.095, width=170)
    fig.tight_layout(rect=(0, 0.085 if style.PAPER else 0.205, 1, 0.995))
    _rows = [r for l in bylaw for r in bylaw[l].values()]
    _facts(_acc_sentence(_rows, laws))
    # An optional fifth element names the library backend shown.
    who, fam, arch, gpu, *lib_bk = cap_of or ("tested implementations", "", "",
                                              False)
    _caption("Share of points with a ULP error of at least $x$ for the %s "
             "across the %s datasets." % (who, CONV.get(fam, fam)),
             "Each panel shows one dataset; both axes are logarithmic.",
             "The dashed vertical line marks 1 ULP.",
             "The dotted lines at $10^{-2}$ and $10^{-3}$ cross each curve within "
             "one histogram bin of its 99th and 99.9th percentiles, and the share "
             "at or above $2^{%d}$ ULP is marked in the grey strip."
             % round(mp.HIST_HI_LOG),
             ("The library is shown with its %s backend." % lib_bk[0])
             if lib_bk and lib_bk[0] else "",
             _obtained(arch, gpu))
    return _finish(fig, outdir, stem, stamp)


def fig_accuracy_ecdf(plt, acc, sup, outdir, stamp, family="z_nu",
                      lib_backend="avx512", scope="call", arch=""):
    """`_accuracy_ecdf` for the host rows of `fig_accuracy_distribution`."""
    names, bylaw, lib_backend = _dist_inputs_cpu(acc, sup, family, scope,
                                                 lib_backend)
    if not _dist_laws(bylaw):
        return _skip(f"accuracy-ecdf-{family}",
                     f"no library or supplier accuracy rows for family {family}")
    if len(names) < 2:
        return _skip(f"accuracy-ecdf-{family}",
                     "only one implementation on this machine (%s)"
                     % (names[0] if names else "none"))
    return _accuracy_ecdf(
        plt, names, bylaw, outdir, f"fig-paper-accuracy-ecdf-{family}",
        f"Share of points at or above each ULP error, by dataset, "
        f"{FAMILY_LABEL.get(family, family)}"
        f"   [{arch}, {scope} scope, library = {lib_backend}]", stamp,
        cap_of=("tested implementations", family, arch, False,
                _lib_shown(bylaw, lib_backend)))


def fig_gpu_accuracy_ecdf(plt, gsup, outdir, stamp, family="z_nu", arch=""):
    """`_accuracy_ecdf` on the device, kernel scope."""
    names, bylaw = _dist_inputs_gpu(gsup, family)
    if not _dist_laws(bylaw):
        return _skip(f"gpu-accuracy-ecdf-{family}",
                     "no GPU supplier rows at kernel scope "
                     "(needs a CUDA device and the device-side externals)")
    return _accuracy_ecdf(
        plt, names, bylaw, outdir, f"fig-paper-gpu-accuracy-ecdf-{family}",
        f"Share of points at or above each ULP error, by dataset, device "
        f"kernels, {FAMILY_LABEL.get(family, family)}   [{arch}, kernel scope]",
        stamp, lead=next((n for n in names if style.LIB in n), None),
        cap_of=("tested GPU implementations", family, arch, True))
