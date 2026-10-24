/* ulpmodes.h -- four ULP spacing conventions, selectable per run; WM_ULP_UP is
 * errstats.h's wmq_ulp_at.  They coincide except where the reference is a power of two. */
#ifndef WM_ULPMODES_H
#define WM_ULPMODES_H

#include <math.h>
#include <stdio.h>
#include "errstats.h"          /* wmq_ulp_at, the canonical definition */

/* Every mode takes the ulp at the reference, never at the computed value; floors it at
 * 2^-1074, so the metric never divides by zero; and keeps the error signed until it is
 * reduced (wm_ulp_signed). */
#define WM_ULP_DENORM_MIN 4.9406564584124654e-324   /* 2^-1074 */

typedef enum {
    WM_ULP_UP       = 0,  /* nextafter(r, +inf) - r : spacing above r.
                           * The default; computed by wmq_ulp_at. */
    WM_ULP_DOWN     = 1,  /* r - nextafter(r, -inf) : spacing below r.  Half of UP
                           * when r is a power of two. */
    WM_ULP_MULLER   = 2,  /* 2^(e-p+1) for 2^e <= |r| < 2^(e+1) (Handbook of
                           * Floating-Point Arithmetic).  Equals UP for normals;
                           * defined explicitly at powers of two and subnormals. */
    WM_ULP_STRADDLE = 3,  /* the gap between the two doubles straddling the exact
                           * value (Harrison / Kahan).  At a power of two it agrees
                           * with DOWN when the exact value lies below r, and with
                           * UP otherwise. */
    WM_ULP_MODE_COUNT = 4
} wm_ulp_mode;

/* The mode every driver scores with (benchmark_cpu --ulp-mode overrides it); the
 * `ulp_mode` column of every CSV row names the mode used. */
#define WM_ULP_MODE_DEFAULT WM_ULP_UP

static inline const char *wm_ulp_mode_name(wm_ulp_mode m)
{
    switch (m) {
        case WM_ULP_UP:       return "up";
        case WM_ULP_DOWN:     return "down";
        case WM_ULP_MULLER:   return "muller";
        case WM_ULP_STRADDLE: return "straddle";
        default:              return "?";
    }
}

/* The spacing at the reference under one convention.  `r` is the high word of the
 * double-double reference and `lo` its low word, which STRADDLE uses to decide on
 * which side of r the exact value lies. */
static inline double wm_ulp_at_mode(double r, double lo, wm_ulp_mode mode)
{
    double a = fabs(r), u;

    if (!(a > 0.0) || !isfinite(a))
        return WM_ULP_DENORM_MIN;

    switch (mode) {
        case WM_ULP_DOWN: {
            double d = a - nextafter(a, 0.0);
            u = (d > 0.0) ? d : WM_ULP_DENORM_MIN;
            break;
        }
        case WM_ULP_MULLER: {
            /* 2^(e-52) with 2^e <= a < 2^(e+1).  frexp gives a = f*2^E with
             * f in [0.5,1), so e = E-1 and the spacing is 2^(E-53).  Subnormals
             * land below the floor and are clamped. */
            int E; (void)frexp(a, &E);
            u = ldexp(1.0, E - 53);
            if (!(u > 0.0) || u < WM_ULP_DENORM_MIN) u = WM_ULP_DENORM_MIN;
            break;
        }
        case WM_ULP_STRADDLE: {
            /* The true value is r + lo.  Take the gap of the interval it lies in:
             * below r when lo < 0, above r otherwise.  At a power of two with
             * lo < 0 this is the smaller side, which is the defining case. */
            double d = (lo < 0.0) ? (a - nextafter(a, 0.0))
                                  : (nextafter(a, INFINITY) - a);
            u = (d > 0.0 && isfinite(d)) ? d : WM_ULP_DENORM_MIN;
            break;
        }
        case WM_ULP_UP:
        default:
            return wmq_ulp_at(r);       /* delegate: one canonical definition */
    }
    return u;
}

/* Signed ulp error of `v` against the double-double reference (hi + lo).
 * Reduce with fabs(); the histogram keeps the sign. */
static inline double wm_ulp_signed(double v, double hi, double lo, wm_ulp_mode mode)
{
    return (v - hi - lo) / wm_ulp_at_mode(hi, lo, mode);
}

/* The metric under `mode` as `<indent><key> <value>` lines (build/detect --ulp, into
 * PROVENANCE.txt).  It restates wm_ulp_signed and wm_ulp_at_mode: change them together. */
static inline void wm_ulp_mode_describe(FILE *f, const char *indent, wm_ulp_mode mode)
{
    const char *s;
    switch (mode) {
        case WM_ULP_DOWN:     s = "s = |r_hi| - nextafter(|r_hi|, 0), the spacing below the reference"; break;
        case WM_ULP_MULLER:   s = "s = 2^(e-52) for 2^e <= |r_hi| < 2^(e+1)"; break;
        case WM_ULP_STRADDLE: s = "s = |r_hi| - nextafter(|r_hi|, 0) if r_lo < 0, else nextafter(|r_hi|, +inf) - |r_hi|"; break;
        case WM_ULP_UP:
        default:              s = "s = nextafter(|r_hi|, +inf) - |r_hi|, the spacing above the reference"; break;
    }
    fprintf(f, "%s%-18s %s\n", indent, "ulp_mode", wm_ulp_mode_name(mode));
    fprintf(f, "%s%-18s u = ((v - r_hi) - r_lo) / s, evaluated in double; |u| is reported, "
               "the sign is kept in the histogram; v is the computed value, r_hi + r_lo the "
               "double-double reference\n", indent, "ulp_formula");
    fprintf(f, "%s%-18s %s; s = 2^-1074 when r_hi is zero or not finite, or the spacing "
               "evaluates to zero\n", indent, "ulp_spacing", s);
    fprintf(f, "%s%-18s the spacing is taken at the reference, not at the computed value; u is "
               "real-valued, not an integer count of doubles between v and the reference\n",
            indent, "ulp_scope");
    fprintf(f, "%s%-18s up, down, muller and straddle coincide except where |r_hi| is a power "
               "of two; there down, and straddle when r_lo < 0, take the spacing below (half "
               "the up spacing for |r_hi| > 2^-1022), and muller equals up\n",
            indent, "ulp_modes_differ");
    fprintf(f, "%s%-18s the `ulp_mode` column of each CSV row names the mode that row was "
               "scored with\n", indent, "ulp_per_row");
}

#endif /* WM_ULPMODES_H */
