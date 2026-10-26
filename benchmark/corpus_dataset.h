/* corpus_dataset.h -- reads one law of the evaluation corpus (data/z_nu/, data/h_nu_s/
 * or the files directly in data/; one binary file per column, benchmark/corpus_bin.h)
 * into a bench_dataset (common/dataset.h). */
#ifndef MATERN_CORPUS_DATASET_H
#define MATERN_CORPUS_DATASET_H

#include <stddef.h>
#include "dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bench_dataset set;          /* inputs and reference                            */
    int    *accuracy_bits;     /* the corpus `bits` column, one per point         */
    size_t  file_points;       /* points on disk, before any subsampling          */
    size_t  stride;            /* 1: the whole law is read                        */
    int     min_bits;          /* smallest loaded accuracy_bits                   */
    char    family[16];        /* "z_nu" | "h_nu_s"                               */
    char    law[40];           /* "grid", "large_nu", ...                         */
} corpus_dataset;

/* Loads `<family>-{nu,z|h,s,ref}_<law>_<N>.bin` from `<root>/<family>/`, else from
 * `<root>/`; on `h_nu_s` the scale parameter s is in the `s` files.  `max_points == 0`
 * reads the law whole, otherwise a stride keeps at most about `max_points` evenly
 * spaced points (not a prefix: several laws are structured).  Returns 0, or non-zero
 * after printing the reason. */
int  corpus_dataset_load(corpus_dataset *set, const char *root,
                         const char *family, const char *law, size_t max_points);

/* As corpus_dataset_load with max_points = 0, reading only the first `prefix` points
 * (0: the whole law), for a capped ladder; `file_points` still gives the law's size.
 * Load the `size_*` ladder laws whole or by prefix (max_points = 0), never at a stride. */
int  corpus_dataset_load_prefix(corpus_dataset *set, const char *root,
                                const char *family, const char *law, size_t prefix);

/* Enumerates the laws present for a family, from both places.  Writes up to
 * `cap` names and returns how many exist. */
int  corpus_dataset_list(const char *root, const char *family,
                         char names[][40], int cap);

void corpus_dataset_free(corpus_dataset *set);

/* 0 when every name of the comma list `families` is a family of the corpus (z_nu,
 * h_nu_s); otherwise names the first unknown one on stderr and returns 1. */
int  corpus_family_check(const char *families);

#ifdef __cplusplus
}
#endif
#endif /* MATERN_CORPUS_DATASET_H */
