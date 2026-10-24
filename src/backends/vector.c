/* vector.c -- entry point of the vector backends: runs of half-integer order go to the
 * closed form, everything else to the routing pipeline (see the accompanying paper, CPU
 * implementation and SIMD strategy). */
#include <stddef.h>

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
            /* a short half-integer run: the closed form still applies if it opens
             * a long enough segment of half-integer orders */
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
