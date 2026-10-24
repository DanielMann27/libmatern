/* internal.h -- not a public header and not part of the ABI: struct matern_workspace, the
 * per-thread workspace accessor and the (h, nu, s) and (z, nu) entry points of every
 * backend build, which stay local in both libraries (matern.map). */
#ifndef MATERN_INTERNAL_H
#define MATERN_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "matern.h"

/* Routing scratch for one tile, owned by the caller: a region tag, the source position
 * and sorted copies of the three inputs and the output, 1 + 8 + 4*8 = 41 bytes per
 * element, plus 28 for the continued-fraction sort, 69 in total.  The sizes must match
 * the carving in workspace.c and the size stated in matern.h. */
struct matern_workspace {
    unsigned char *region_tag;      /* which region each point belongs to     */
    size_t        *source_index;    /* where in the tile the point came from  */
    double        *sorted_h;
    double        *sorted_nu;
    double        *sorted_scale;
    double        *sorted_out;

    /* The continued-fraction run sorted again by the exponent of its derived argument:
     * argument, order, result and permutation, 8 + 8 + 8 + 4 = 28 bytes per element.
     * Sized on the tile, since the router never hands a region more than one tile. */
    double        *fraction_argument;
    double        *fraction_order;
    double        *fraction_result;
    uint32_t      *fraction_index;

    size_t         tile;            /* elements per tile                      */
    void          *backing;         /* the single allocation the rest carves  */
};

/* The per-thread workspace behind `workspace = NULL`. */
matern_workspace *matern_thread_workspace(int *status);

int matern_evaluate_avx512(double *out, const double *h, const double *nu,
                           const double *scale, size_t n,
                           matern_workspace *workspace);

int matern_evaluate_avx2  (double *out, const double *h, const double *nu,
                           const double *scale, size_t n,
                           matern_workspace *workspace);

/* `matched`: the vector source at width 1, with libm's elementary functions.  Without
 * -mfma it evaluates fused operations as a*b + c except at the cancelling sites
 * (numeric/fused_multiply_add_scalar.h). */
int matern_evaluate_matched(double *out, const double *h, const double *nu,
                            const double *scale, size_t n,
                            matern_workspace *workspace);

/* `matched_fma`: the same source with -mfma, the width-1 twin of AVX2; selected behind
 * an FMA cpuid check. */
int matern_evaluate_matched_fma(double *out, const double *h, const double *nu,
                                const double *scale, size_t n,
                                matern_workspace *workspace);

/* `scalar`: the direct implementation (src/backends/scalar), built for the x86-64
 * baseline (`scalar_portable`), AVX-512, AVX2 with FMA, and FMA; MATERN_BACKEND=scalar
 * selects the best build the CPU runs. */
int matern_evaluate_scalar(double *out, const double *h, const double *nu,
                           const double *scale, size_t n,
                           matern_workspace *workspace);
int matern_evaluate_scalar_avx512(double *out, const double *h, const double *nu,
                                  const double *scale, size_t n,
                                  matern_workspace *workspace);
int matern_evaluate_scalar_avx2(double *out, const double *h, const double *nu,
                                const double *scale, size_t n,
                                matern_workspace *workspace);
int matern_evaluate_scalar_fma(double *out, const double *h, const double *nu,
                               const double *scale, size_t n,
                               matern_workspace *workspace);

/* The (z, nu) entries: `scale` is accepted and ignored, and matern_covariance_z passes
 * `nu` for it so every load is in bounds.  The vector and width-1 entries are vector.c
 * compiled again under -DMATERN_ENTRY_Z. */
int matern_evaluate_avx512_z (double *out, const double *z, const double *nu,
                              const double *scale, size_t n,
                              matern_workspace *workspace);

int matern_evaluate_avx2_z   (double *out, const double *z, const double *nu,
                              const double *scale, size_t n,
                              matern_workspace *workspace);

int matern_evaluate_matched_z(double *out, const double *z, const double *nu,
                              const double *scale, size_t n,
                              matern_workspace *workspace);

int matern_evaluate_matched_fma_z(double *out, const double *z, const double *nu,
                                  const double *scale, size_t n,
                                  matern_workspace *workspace);

int matern_evaluate_scalar_z (double *out, const double *z, const double *nu,
                              const double *scale, size_t n,
                              matern_workspace *workspace);
int matern_evaluate_scalar_avx512_z(double *out, const double *z, const double *nu,
                                    const double *scale, size_t n,
                                    matern_workspace *workspace);
int matern_evaluate_scalar_avx2_z(double *out, const double *z, const double *nu,
                                  const double *scale, size_t n,
                                  matern_workspace *workspace);
int matern_evaluate_scalar_fma_z(double *out, const double *z, const double *nu,
                                 const double *scale, size_t n,
                                 matern_workspace *workspace);

/* The instruction set of the scalar build that MATERN_BACKEND=scalar resolved to:
 * "avx512", "avx2", "fma" or "portable" ("none" when another backend is selected). */
const char *matern_scalar_isa(void);

#endif /* MATERN_INTERNAL_H */
