/* vector.c -- entry point of the vector backends (see the accompanying paper, CPU
 * implementation and SIMD strategy): half-integer runs go to the closed form, the rest to
 * the routing pipeline; width 1 works per point, continued-fraction points four at a time. */
#include <stddef.h>

/* avx512 and avx2 route half-integer orders as a region of their own (region_partition.h),
 * so that every such point takes the closed form wherever it sits in the batch; the width-1
 * entries select the closed form per point before classifying. */
#if !defined(MATERN_TARGET_SCALAR)
#define MATERN_ROUTE_HALF_INTEGER
#endif

#include "internal.h"
#include "vector_current.h"
#include "region_map.h"
#include "pipeline_vector.h"

/* Compiled once per target (AVX-512, AVX2, width 1) and entry; every symbol except the
 * entry point is static or localised by objcopy, so the objects link into one binary.
 * Under -DMATERN_ENTRY_Z `bessel_derive_argument` is an identity, the classifier reads
 * z, and the entry is named `_z`. */
#if defined(MATERN_TARGET_AVX2)
  #ifdef MATERN_ENTRY_Z
    #define MATERN_VECTOR_ENTRY matern_evaluate_avx2_z
  #else
    #define MATERN_VECTOR_ENTRY matern_evaluate_avx2
  #endif
  #define MATERN_ORDER_CEILING HALF_INTEGER_VECTOR_CAP
#elif defined(MATERN_TARGET_SCALAR)
  /* Width 1: the vector source over the width-1 vocabulary, with the vector order cap. */
  #if defined(MATERN_TARGET_FMA)
    /* `matched_fma`, selected only behind an FMA cpuid check (dispatch.c); -mfma
     * implies -mavx, legal on every FMA3 machine.  Without -mfma (`matched`) fused
     * operations are a*b + c except at the VEC_*_CANCEL sites. */
    #ifdef MATERN_ENTRY_Z
      #define MATERN_VECTOR_ENTRY matern_evaluate_matched_fma_z
    #else
      #define MATERN_VECTOR_ENTRY matern_evaluate_matched_fma
    #endif
  #else
  #ifdef MATERN_ENTRY_Z
    #define MATERN_VECTOR_ENTRY matern_evaluate_matched_z
  #else
    #define MATERN_VECTOR_ENTRY matern_evaluate_matched
  #endif
  #endif
  #define MATERN_ORDER_CEILING HALF_INTEGER_VECTOR_CAP
#else
  #ifdef MATERN_ENTRY_Z
    #define MATERN_VECTOR_ENTRY matern_evaluate_avx512_z
  #else
    #define MATERN_VECTOR_ENTRY matern_evaluate_avx512
  #endif
  #define MATERN_ORDER_CEILING HALF_INTEGER_VECTOR_CAP
#endif

#ifdef MATERN_TARGET_SCALAR
/* Width 1: a half-integer order within the ceiling takes the closed form; any other point
 * is classified by matern_classify, evaluated by its region's one-vector cell and clamped
 * as in phase D, except continued-fraction points.  These wait in four-slot queues, one per
 * value of continued_fraction_bucket_of (the driver's sort key, so that the points sharing
 * a block have similar z), and run four at a time through continued_fraction_cell_x4; a
 * queue left with fewer than four points at the end runs continued_fraction_cell_x1 per
 * point.  Which points share a block depends on the batch, but each lane of the cell stops
 * at its own test, so every point gets the result it gets alone. */
#define CONTINUED_FRACTION_QUEUES 32                 /* the range of continued_fraction_bucket_of */

/* The closed form's per-run setup (matern_half_integer_run), kept for the last order seen. */
typedef struct {
    int             degree;                          /* -1: nothing kept */
    double          c[HALF_INTEGER_MAX_DEGREE + 1];
    SIMD_TYPE_FLOAT S, SL;                           /* sqrt(2nu) as a double-double */
} half_integer_setup;

static inline __attribute__((always_inline))
double one_point_half_integer(double h, double nu, double s, int p, half_integer_setup *m)
{
    if (m->degree != p) {
        half_integer_coefficients(p, m->c);
        const double two_nu = nu + nu;                   /* exact */
        const double shd = sqrt(two_nu);
        const double sqh = shd * shd;                    /* rounded square; the fma recovers its error */
        const double sld = ((two_nu - sqh) - fma_cancel(shd, shd, -sqh)) / (shd + shd);
        m->S = VEC_VSET(shd);
        m->SL = VEC_VSET(sld);
        m->degree = p;
    }
    return vec_first_lane(half_integer_cell(VEC_VSET(h), VEC_VSET(s), m->S, m->SL, m->c, p));
}

/* Phase A for one point. */
static inline __attribute__((always_inline))
unsigned char one_point_region(double h, double nu, double s)
{
    unsigned char r;
    size_t cnt[REGION_TOTAL] = {0};
    matern_classify(&h, &nu, &s, &r, 1, cnt);
    return r;
}

/* The phase D clamp. */
static inline __attribute__((always_inline))
double one_point_clamp(double v)
{
    if (!(v > 0.0)) v = 0.0; else if (v > 1.0) v = 1.0;
    return v;
}

/* Region r's cell for one point, clamped (the trivial bucket is exempt, as in phase D). */
static inline __attribute__((always_inline))
double one_point_cell(unsigned char r, double h, double nu, double s)
{
    if (r == REGION_TRIVIAL) {
        double o;
        matern_trivial_run(&o, &h, &nu, &s, 1);
        return o;
    }
    const SIMD_TYPE_FLOAT H = VEC_VSET(h), NU = VEC_VSET(nu), R = VEC_VSET(s);
    SIMD_TYPE_FLOAT zh, zl;
    double v;
    switch (r) {
    case REGION_LARGE_ARGUMENT_LOW_ORDER:
    case REGION_LARGE_ARGUMENT: {
        SIMD_TYPE_FLOAT o;
        large_argument_cell(&H, &NU, &R, &o);
        v = vec_first_lane(o);
        break; }
    case REGION_LEVIN:
        bessel_derive_argument(H, NU, R, &zh, &zl);
        v = vec_first_lane(levin_cell(zh, zl, NU));
        break;
    case REGION_CONTINUED_FRACTION: {
        double z;
        continued_fraction_derive_arguments(&z, &h, &nu, &s, 1);
        continued_fraction_cell_x1(&z, &nu, &v);
        break; }
    case REGION_DEBYE:
        bessel_derive_argument(H, NU, R, &zh, &zl);
        v = vec_first_lane(debye_cell(zh, zl, NU));
        break;
    default:                                         /* REGION_SMALL_ARGUMENT */
        v = vec_first_lane(small_argument_cell(small_argument_derive(H, NU, R), NU, H, R));
        break;
    }
    return one_point_clamp(v);
}

typedef struct {
    double z[4], nu[4];
    size_t at[4];                                    /* caller positions */
    int    count;
} continued_fraction_queue;

static inline __attribute__((always_inline))
void continued_fraction_queue_run(continued_fraction_queue *q, double *out)
{
    double o[4];
    continued_fraction_cell_x4(q->z, q->nu, o);
    for (int j = 0; j < 4; j++)
        out[q->at[j]] = one_point_clamp(o[j]);
    q->count = 0;
}

static inline __attribute__((always_inline))
void continued_fraction_queue_flush(continued_fraction_queue *q, double *out)
{
    for (int j = 0; j < q->count; j++) {
        double o;
        continued_fraction_cell_x1(&q->z[j], &q->nu[j], &o);
        out[q->at[j]] = one_point_clamp(o);
    }
    q->count = 0;
}

/* The workspace is accepted and ignored. */
int MATERN_VECTOR_ENTRY(double *out, const double *h, const double *nu,
                        const double *scale, size_t n,
                        matern_workspace *workspace)
{
    (void)workspace;
    (void)matern_half_integer_run;                   /* width 1 uses one_point_half_integer */
    if (n == 0)
        return MATERN_OK;
    if (!out || !h || !nu || !scale)
        return MATERN_INVALID_ARGUMENT;

    half_integer_setup setup;
    setup.degree = -1;
    continued_fraction_queue queue[CONTINUED_FRACTION_QUEUES];
    for (int k = 0; k < CONTINUED_FRACTION_QUEUES; k++)
        queue[k].count = 0;

    for (size_t i = 0; i < n; i++) {
        const int p = half_integer_degree(nu[i], MATERN_ORDER_CEILING);
        if (p >= 0) {
            out[i] = one_point_half_integer(h[i], nu[i], scale[i], p, &setup);
            continue;
        }
        const unsigned char r = one_point_region(h[i], nu[i], scale[i]);
        if (r != REGION_CONTINUED_FRACTION) {
            out[i] = one_point_cell(r, h[i], nu[i], scale[i]);
            continue;
        }
        double z;
        continued_fraction_derive_arguments(&z, h + i, nu + i, scale + i, 1);
        continued_fraction_queue *q = &queue[continued_fraction_bucket_of(z)];
        q->z[q->count] = z;
        q->nu[q->count] = nu[i];
        q->at[q->count] = i;
        if (++q->count == 4)
            continued_fraction_queue_run(q, out);
    }
    for (int k = 0; k < CONTINUED_FRACTION_QUEUES; k++)
        continued_fraction_queue_flush(&queue[k], out);
    return MATERN_OK;
}

#else
/* Length of the leading segment of half-integer orders within the ceiling. */
static size_t half_integer_segment(const double *nu, size_t n)
{
    size_t i = 0;
    while (i < n && half_integer_degree(nu[i], MATERN_ORDER_CEILING) >= 0)
        i++;
    return i;
}

/* m <= workspace->tile points of mixed half-integer order: counting-sorted by degree
 * (stored in region_tag), one closed-form run per degree, scattered back.  nu is
 * exactly p + 1/2 for degree p (half_integer_degree is exact), hence (double)p + 0.5. */
static void half_integer_mixed_tile(double *out, const double *h, const double *nu,
                                    const double *scale, size_t m,
                                    matern_workspace *workspace)
{
    unsigned char *degree = workspace->region_tag;
    size_t        *source = workspace->source_index;
    double *hs = workspace->sorted_h, *rs = workspace->sorted_scale, *os = workspace->sorted_out;
    size_t start[MATERN_ORDER_CEILING + 2] = {0};
    size_t next [MATERN_ORDER_CEILING + 1];

    for (size_t i = 0; i < m; i++) {
        degree[i] = (unsigned char)half_integer_degree(nu[i], MATERN_ORDER_CEILING);
        start[degree[i] + 1]++;
    }
    for (int p = 0; p <= MATERN_ORDER_CEILING; p++) {
        start[p + 1] += start[p];
        next[p] = start[p];
    }
    for (size_t i = 0; i < m; i++) {
        const size_t d = next[degree[i]]++;
        source[d] = i;
        hs[d] = h[i];
        rs[d] = scale[i];
    }
    for (int p = 0; p <= MATERN_ORDER_CEILING; p++) {
        const size_t s = start[p], k = start[p + 1] - s;
        if (k)
            matern_half_integer_run(os + s, hs + s, (double)p + 0.5, rs + s, k, p);
    }
    for (size_t d = 0; d < m; d++)
        out[source[d]] = os[d];
}

int MATERN_VECTOR_ENTRY(double *out, const double *h, const double *nu,
                        const double *scale, size_t n,
                        matern_workspace *workspace)
{
    if (n == 0)
        return MATERN_OK;
    if (!out || !h || !nu || !scale || !workspace || !workspace->tile)
        return MATERN_INVALID_ARGUMENT;

    size_t offset = 0;

    while (offset < n) {
        const size_t remaining = n - offset;
        const size_t run       = constant_run_length(nu + offset, remaining);
        const int    degree    = half_integer_degree(nu[offset], MATERN_ORDER_CEILING);

        if (degree >= 0 && (run >= HALF_INTEGER_MIN_RUN || run == remaining)) {
            matern_half_integer_run(out + offset, h + offset, nu[offset],
                                    scale + offset, run, degree);
            offset += run;
            continue;
        }

        if (degree >= 0) {
            /* a short half-integer run: the direct path still applies if it opens a
             * long enough segment of half-integer orders (otherwise the router gives
             * its points the same closed form) */
            const size_t segment = half_integer_segment(nu + offset, remaining);
            if (segment > run && (segment >= HALF_INTEGER_MIN_RUN || segment == remaining)) {
                for (size_t done = 0; done < segment; ) {
                    const size_t left = segment - done;
                    const size_t m = left < workspace->tile ? left : workspace->tile;
                    half_integer_mixed_tile(out + offset + done, h + offset + done,
                                            nu + offset + done, scale + offset + done,
                                            m, workspace);
                    done += m;
                }
                offset += segment;
                continue;
            }
        }

        if (run >= HALF_INTEGER_MIN_RUN) {
            /* a long run, but this order has no closed form */
            matern_route(out + offset, h + offset, nu + offset, scale + offset,
                         run, workspace);
            offset += run;
            continue;
        }

        /* otherwise one pipeline call for the whole remainder, which keeps the scan
         * linear; later runs in the batch are not split out */
        matern_route(out + offset, h + offset, nu + offset, scale + offset,
                     remaining, workspace);
        return MATERN_OK;
    }

    return MATERN_OK;
}

#endif /* MATERN_TARGET_SCALAR */
