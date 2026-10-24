/* corpus_dataset.c -- implements corpus_dataset.h: locates the files of a law, reads
 * them whole, as a prefix or at a stride, and checks that their point counts agree. */
#define _GNU_SOURCE
#include "corpus_dataset.h"
#include "corpus_bin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

/* Reads every `stride`-th of the first `limit` records (0: all).  Returns the
 * points kept, or (size_t)-1.  `*on_disk` receives the records considered, which
 * the count-equality check compares, so that a stride cannot hide a short file. */
static size_t read_axis(const char *path, double **out, size_t stride,
                        size_t limit, size_t *on_disk)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "corpus: cannot open %s\n", path); return (size_t)-1; }
    corpus_bin_header h; char why[160] = "";
    if (corpus_bin_read_header(f, path, &h, why, sizeof why) || h.kind != CORPUS_KIND_AXIS) {
        fprintf(stderr, "corpus: %s: %s\n", path,
                h.kind != CORPUS_KIND_AXIS && !why[0] ? "not an axis file" : why);
        fclose(f); return (size_t)-1;
    }
    const size_t points = (limit && limit < h.count) ? limit : (size_t)h.count;
    const size_t kept = (points + stride - 1) / stride;
    double *v = (double *)malloc((kept ? kept : 1) * sizeof *v);
    if (!v || corpus_bin_read_array(f, &h, 0, stride, points, v)) {
        fprintf(stderr, "corpus: %s: read error\n", path);
        free(v); fclose(f); return (size_t)-1;
    }
    fclose(f);
    *on_disk = points;
    *out = v;
    return kept;
}

/* As read_axis, for a reference file: hi, lo and bits. */
static size_t read_reference(const char *path, double **hi, double **lo, int **bits,
                             size_t stride, size_t limit, size_t *on_disk)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "corpus: cannot open %s\n", path); return (size_t)-1; }
    corpus_bin_header h; char why[160] = "";
    if (corpus_bin_read_header(f, path, &h, why, sizeof why) || h.kind != CORPUS_KIND_REFERENCE) {
        fprintf(stderr, "corpus: %s: %s\n", path,
                h.kind != CORPUS_KIND_REFERENCE && !why[0] ? "not a reference file" : why);
        fclose(f); return (size_t)-1;
    }
    const size_t points = (limit && limit < h.count) ? limit : (size_t)h.count;
    const size_t kept = (points + stride - 1) / stride, m = kept ? kept : 1;
    double *H = (double *)malloc(m * sizeof *H);
    double *L = (double *)malloc(m * sizeof *L);
    int    *B = (int    *)malloc(m * sizeof *B);
    if (!H || !L || !B || corpus_bin_read_array(f, &h, 0, stride, points, H)
                       || corpus_bin_read_array(f, &h, 1, stride, points, L)
                       || corpus_bin_read_array(f, &h, 2, stride, points, B)) {
        fprintf(stderr, "corpus: %s: read error\n", path);
        free(H); free(L); free(B); fclose(f); return (size_t)-1;
    }
    fclose(f);
    *on_disk = points; *hi = H; *lo = L; *bits = B;
    return kept;
}

/* Finds <family>-<column>_<law>_<N>.bin in <root>/<family>/, else in <root>/; the
 * family prefix keeps file names unique in the flat layout. */
static int find_in(const char *dir, const char *head, char *out, size_t outn,
                   size_t *n_from_name)
{
    DIR *d = opendir(dir);
    if (!d) return -1;
    const size_t hl = strlen(head);
    int found = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, head, hl) != 0) continue;
        const char *tail = e->d_name + hl;
        char *stop;
        long n = strtol(tail, &stop, 10);
        if (n <= 0 || strcmp(stop, ".bin") != 0) continue;   /* not <N>.bin */
        snprintf(out, outn, "%s/%s", dir, e->d_name);
        if (n_from_name) *n_from_name = (size_t)n;
        found = 1;
        break;
    }
    closedir(d);
    return found ? 0 : -1;
}

static int find_file(const char *root, const char *family, const char *column,
                     const char *law, char *out, size_t outn, size_t *n_from_name,
                     int quiet)
{
    char head[96], dir[512];
    snprintf(head, sizeof head, "%s-%s_%s_", family, column, law);
    snprintf(dir, sizeof dir, "%s/%s", root, family);
    if (find_in(dir, head, out, outn, n_from_name) == 0) return 0;
    if (find_in(root, head, out, outn, n_from_name) == 0) return 0;
    if (!quiet) fprintf(stderr, "corpus: no %s<N>.bin in %s or %s\n", head, dir, root);
    return -1;
}

/* `size_random` and `size_half` are the first 2^20 points of `size_random_large` and
 * `size_half_large` (data/DATASETS.md); without files of their own they are read as
 * that prefix. */
#define CORPUS_PREFIX_POINTS ((size_t)1 << 20)

static int has_name(char names[][40], int n, const char *name)
{
    for (int i = 0; i < n; i++) if (!strcmp(names[i], name)) return 1;
    return 0;
}

int corpus_dataset_list(const char *root, const char *family,
                        char names[][40], int cap)
{
    /* <family>-ref_<law>_<N>.bin -> <law>, from <root>/<family>/ and <root>/ */
    char dir[512], head[64];
    snprintf(dir, sizeof dir, "%s/%s", root, family);
    snprintf(head, sizeof head, "%s-ref_", family);
    const size_t hl = strlen(head);
    const char *dirs[2] = { dir, root };

    int count = 0;
    for (int di = 0; di < 2; di++) {
        DIR *d = opendir(dirs[di]);
        if (!d) continue;
        struct dirent *e;
        while ((e = readdir(d))) {
            if (strncmp(e->d_name, head, hl) != 0) continue;
            const size_t nl = strlen(e->d_name);
            if (nl < hl + 4 || strcmp(e->d_name + nl - 4, ".bin") != 0) continue;
            const char *p = e->d_name + hl;
            const char *last = strrchr(p, '_');
            if (!last) continue;
            size_t len = (size_t)(last - p);
            if (len == 0 || len >= 40) continue;
            char law[40];
            memcpy(law, p, len); law[len] = '\0';
            /* a law present in both places is listed once */
            if (has_name(names, count < cap ? count : cap, law)) continue;
            if (count < cap) snprintf(names[count], 40, "%s", law);
            count++;
        }
        closedir(d);
    }
    /* a `_large` set also provides its 2^20 prefix law */
    const int listed = count < cap ? count : cap;
    for (int i = 0; i < listed; i++) {
        size_t len = strlen(names[i]);
        if (len <= 6 || strcmp(names[i] + len - 6, "_large") != 0) continue;
        char base[40];
        memcpy(base, names[i], len - 6); base[len - 6] = '\0';
        if (has_name(names, count < cap ? count : cap, base)) continue;
        if (count < cap) snprintf(names[count], 40, "%s", base);
        count++;
    }
    return count;
}

static int load_law(corpus_dataset *cs, const char *root, const char *family,
                    const char *law, size_t max_points, size_t prefix)
{
    memset(cs, 0, sizeof *cs);
    const int is_hns = (strcmp(family, "h_nu_s") == 0);

    /* Locate all files first, so that a missing one is reported before any load. */
    char p_nu[640], p_arg[640], p_scale[640], p_ref[640];
    char flaw[48];
    size_t n_name = 0, limit = 0;
    snprintf(flaw, sizeof flaw, "%s", law);
    if (find_file(root, family, "nu", flaw, p_nu, sizeof p_nu, &n_name, 1) != 0) {
        snprintf(flaw, sizeof flaw, "%s_large", law);
        if (find_file(root, family, "nu", flaw, p_nu, sizeof p_nu, &n_name, 1) != 0) {
            find_file(root, family, "nu", law, p_nu, sizeof p_nu, &n_name, 0);   /* the report */
            return 1;
        }
        limit = CORPUS_PREFIX_POINTS;
        if (n_name < limit) {
            fprintf(stderr, "corpus: %s/%s_large holds %zu points, fewer than the %zu of %s\n",
                    family, law, n_name, limit, law);
            return 1;
        }
        n_name = limit;
    }
    /* A ladder capped below the law's size reads only its prefix. */
    const size_t law_points = n_name;
    if (prefix && prefix < n_name) { limit = prefix; n_name = prefix; }
    if (find_file(root, family, is_hns ? "h" : "z", flaw, p_arg, sizeof p_arg, NULL, 0) != 0) return 1;
    if (find_file(root, family, "ref", flaw, p_ref, sizeof p_ref, NULL, 0) != 0) return 1;
    /* the scale parameter s is stored in the `s` files of h_nu_s */
    if (is_hns && find_file(root, family, "s", flaw, p_scale, sizeof p_scale, NULL, 0) != 0) return 1;

    size_t stride = 1;
    if (max_points && n_name > max_points) stride = (n_name + max_points - 1) / max_points;

    size_t d_nu = 0, d_arg = 0, d_scale = 0, d_ref = 0;
    double *nu = NULL, *arg = NULL, *scale = NULL, *hi = NULL, *lo = NULL;
    int    *bits = NULL;

    size_t k_nu  = read_axis(p_nu,  &nu,  stride, limit, &d_nu);
    size_t k_arg = read_axis(p_arg, &arg, stride, limit, &d_arg);
    size_t k_ref = read_reference(p_ref, &hi, &lo, &bits, stride, limit, &d_ref);
    size_t k_scale = k_nu;
    if (is_hns) k_scale = read_axis(p_scale, &scale, stride, limit, &d_scale); else d_scale = d_nu;

    if (k_nu == (size_t)-1 || k_arg == (size_t)-1 || k_ref == (size_t)-1 ||
        k_scale == (size_t)-1) goto fail;

    /* A short file would pair every later point with the wrong reference.  The
     * counts read are compared, not those of the subsample. */
    if (d_nu != d_arg || d_nu != d_ref || (is_hns && d_nu != d_scale)) {
        fprintf(stderr, "corpus: %s/%s POINT COUNTS DISAGREE -- nu=%zu %s=%zu ref=%zu",
                family, law, d_nu, is_hns ? "h" : "z", d_arg, d_ref);
        if (is_hns) fprintf(stderr, " s=%zu", d_scale);
        fprintf(stderr, "\n        one of these files is truncated;"
                        " refusing to pair them.\n");
        goto fail;
    }
    if (n_name && d_nu != n_name)
        fprintf(stderr, "corpus: NOTE %s/%s holds %zu points, filename says %zu\n",
                family, law, d_nu, n_name);

    int mb = 1 << 30;
    for (size_t i = 0; i < k_ref; i++) if (bits[i] < mb) mb = bits[i];

    snprintf(cs->family, sizeof cs->family, "%s", family);
    snprintf(cs->law,    sizeof cs->law,    "%s", law);
    cs->file_points   = (limit && law_points > d_nu) ? law_points : d_nu;
    cs->stride        = stride;
    cs->min_bits      = (k_ref ? mb : 0);
    cs->accuracy_bits = bits;

    wm_dataset *s = &cs->set;
    s->has_length_scale = is_hns;
    s->count            = k_nu;
    s->order            = nu;
    s->reference_high   = hi;
    s->reference_low    = lo;
    if (is_hns) { s->h = arg; s->length_scale = scale; s->argument = NULL; }
    else        { s->argument = arg; s->h = NULL;      s->length_scale = NULL; }
    return 0;

fail:
    free(nu); free(arg); free(scale); free(hi); free(lo); free(bits);
    memset(cs, 0, sizeof *cs);
    return 1;
}

int corpus_dataset_load(corpus_dataset *cs, const char *root,
                        const char *family, const char *law, size_t max_points)
{
    return load_law(cs, root, family, law, max_points, 0);
}

int corpus_dataset_load_prefix(corpus_dataset *cs, const char *root,
                               const char *family, const char *law, size_t prefix)
{
    return load_law(cs, root, family, law, 0, prefix);
}

void corpus_dataset_free(corpus_dataset *cs)
{
    if (!cs) return;
    free(cs->set.order); free(cs->set.h); free(cs->set.length_scale);
    free(cs->set.argument);
    free(cs->set.reference_high); free(cs->set.reference_low);
    free(cs->accuracy_bits);
    memset(cs, 0, sizeof *cs);
}

int corpus_family_check(const char *families)
{
    const char *p = families;
    for (;;) {
        size_t n = strcspn(p, ",");
        if (!((n == 4 && !strncmp(p, "z_nu", 4)) || (n == 6 && !strncmp(p, "h_nu_s", 6)))) {
            fprintf(stderr, "unknown family '%.*s'; the families are z_nu and h_nu_s\n",
                    (int)n, p);
            return 1;
        }
        if (!p[n]) return 0;
        p += n + 1;
    }
}
