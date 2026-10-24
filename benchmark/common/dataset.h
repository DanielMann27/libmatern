/* dataset.h -- in-memory form of one evaluation set, filled by corpus_dataset.c.  The
 * reference stays a hi + lo pair: summed into one double it could not judge a double
 * result. */
#ifndef MATERN_DATASET_H
#define MATERN_DATASET_H

#include <stddef.h>

/* C linkage, so that CUDA and C++ harnesses share the type. */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int     has_length_scale;   /* 1 = (h, nu, s); 0 = (z, nu) direct          */
    size_t  count;

    double *h;                  /* (h, nu, s) sets only                        */
    double *length_scale;       /* s; NULL on a (z, nu) set                    */
    double *order;              /* nu, both families                           */
    double *argument;           /* z; (z, nu) sets only                        */

    double *reference_high;     /* the reference is reference_high + reference_low */
    double *reference_low;
} wm_dataset;

#ifdef __cplusplus
}
#endif

#endif /* MATERN_DATASET_H */
