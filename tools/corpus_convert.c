/* corpus_convert.c -- converts corpus files between their hex-float text form (one
 * value, or "hi lo bits", per line) and the binary form of benchmark/corpus_bin.h, and
 * verifies the round trip byte for byte. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "corpus_bin.h"

static int fail(const char *path, const char *what, unsigned long long line)
{
    if (line) fprintf(stderr, "corpus_convert: %s: line %llu: %s\n", path, line, what);
    else      fprintf(stderr, "corpus_convert: %s: %s\n", path, what);
    return 1;
}

static int print_double(char *buf, size_t cap, double v, uint32_t format)
{
    return snprintf(buf, cap, format == CORPUS_TEXT_SHORTEST ? "%a" : "%.13a", v);
}

/* Print one record (axis value, or hi lo bits) followed by a newline. */
static int print_record(char *buf, size_t cap, uint32_t kind, uint32_t format,
                        double a, double b, int16_t bits)
{
    int n = print_double(buf, cap, a, format);
    if (kind == CORPUS_KIND_REFERENCE) {
        buf[n++] = ' ';
        n += print_double(buf + n, cap - n, b, format);
        n += snprintf(buf + n, cap - n, " %d", (int)bits);
    }
    buf[n++] = '\n';
    buf[n] = '\0';
    return n;
}

/* %a and %.13a round-trip through strtod and print deterministically, so to_bin
 * reprints every line and refuses a file in neither format; the header records which.
 * Returns -2 when a line is not in %.13a, so that the caller retries the whole file
 * in %a; that line's number is left in first_miss for the report. */
static unsigned long long first_miss = 0;

static int to_bin(const char *in, const char *out, uint32_t format)
{
    FILE *f = fopen(in, "r");
    if (!f) return fail(in, strerror(errno), 0);

    size_t cap = 1u << 20, n = 0;
    double  *a = malloc(cap * sizeof *a), *b = NULL;
    int16_t *k = NULL;
    uint32_t kind = 0;
    char *line = NULL; size_t lcap = 0; ssize_t len;
    char re[256];

    while ((len = getline(&line, &lcap, f)) > 0) {
        const unsigned long long ln = (unsigned long long)n + 1;
        char *e1, *e2, *e3;
        /* getline keeps the newline; only a last line without one lacks it */
        if (line[len - 1] != '\n') return fail(in, "missing newline at end of file", ln);
        const double x = strtod(line, &e1);
        if (e1 == line) return fail(in, "not a number", ln);
        uint32_t this_kind = (*e1 == '\n') ? CORPUS_KIND_AXIS : CORPUS_KIND_REFERENCE;
        if (!kind) {
            kind = this_kind;
            if (kind == CORPUS_KIND_REFERENCE) {
                b = malloc(cap * sizeof *b); k = malloc(cap * sizeof *k);
            }
        }
        if (this_kind != kind) return fail(in, "column count differs from line 1", ln);
        double y = 0.0; long bits = 0;
        if (kind == CORPUS_KIND_REFERENCE) {
            y = strtod(e1, &e2);
            if (e2 == e1) return fail(in, "second field is not a number", ln);
            bits = strtol(e2, &e3, 10);
            if (e3 == e2 || *e3 != '\n') return fail(in, "third field is not an integer", ln);
            if (bits < INT16_MIN || bits > INT16_MAX) return fail(in, "bits outside int16", ln);
        }
        if (n == cap) {
            cap *= 2;
            a = realloc(a, cap * sizeof *a);
            if (kind == CORPUS_KIND_REFERENCE) {
                b = realloc(b, cap * sizeof *b); k = realloc(k, cap * sizeof *k);
            }
            if (!a || (kind == CORPUS_KIND_REFERENCE && (!b || !k)))
                return fail(in, "out of memory", 0);
        }
        a[n] = x;
        if (kind == CORPUS_KIND_REFERENCE) { b[n] = y; k[n] = (int16_t)bits; }

        /* The exactness check: the record must print back to the line as read. */
        const int m = print_record(re, sizeof re, kind, format, x, y, (int16_t)bits);
        if ((ssize_t)m != len || memcmp(re, line, (size_t)len) != 0) {
            fclose(f); free(line); free(a); free(b); free(k);
            if (format == CORPUS_TEXT_CANONICAL) { first_miss = ln; return -2; }
            fprintf(stderr, "corpus_convert: %s: line %llu is not in %%.13a form and line %llu "
                    "is not in %%a form; the conversion would not be exact\n", in, first_miss, ln);
            return 1;
        }
        n++;
    }
    if (ferror(f)) return fail(in, "read error", 0);
    fclose(f); free(line);
    if (!n) return fail(in, "empty", 0);

    corpus_bin_header h;
    corpus_bin_header_init(&h, kind, (uint64_t)n, format);
    FILE *o = fopen(out, "wb");
    if (!o) return fail(out, strerror(errno), 0);
    int bad = fwrite(&h, sizeof h, 1, o) != 1 || fwrite(a, sizeof *a, n, o) != n;
    if (kind == CORPUS_KIND_REFERENCE)
        bad |= fwrite(b, sizeof *b, n, o) != n || fwrite(k, sizeof *k, n, o) != n;
    bad |= fclose(o) != 0;
    free(a); free(b); free(k);
    if (bad) { remove(out); return fail(out, "write error", 0); }
    return 0;
}

typedef struct { corpus_bin_header h; double *a, *b; int16_t *k; } corpus_file;

static int read_bin(const char *path, corpus_file *c)
{
    memset(c, 0, sizeof *c);
    FILE *f = fopen(path, "rb");
    if (!f) return fail(path, strerror(errno), 0);
    char why[160];
    if (corpus_bin_read_header(f, path, &c->h, why, sizeof why)) {
        fclose(f); return fail(path, why, 0);
    }
    const size_t n = (size_t)c->h.count;
    c->a = malloc(n * sizeof *c->a);
    int bad = !c->a || fread(c->a, sizeof *c->a, n, f) != n;
    if (c->h.kind == CORPUS_KIND_REFERENCE) {
        c->b = malloc(n * sizeof *c->b); c->k = malloc(n * sizeof *c->k);
        bad |= !c->b || !c->k || fread(c->b, sizeof *c->b, n, f) != n
                              || fread(c->k, sizeof *c->k, n, f) != n;
    }
    fclose(f);
    if (bad) return fail(path, "truncated", 0);
    return 0;
}

static int to_text(const char *in, const char *out)
{
    corpus_file c;
    if (read_bin(in, &c)) return 1;
    FILE *o = out ? fopen(out, "w") : stdout;
    if (!o) return fail(out, strerror(errno), 0);
    char buf[256];
    for (size_t i = 0; i < c.h.count; i++) {
        const int m = print_record(buf, sizeof buf, c.h.kind, c.h.text_format, c.a[i],
                                   c.b ? c.b[i] : 0.0, c.k ? c.k[i] : 0);
        if (fwrite(buf, 1, (size_t)m, o) != (size_t)m) return fail(out ? out : "stdout", "write error", 0);
    }
    if (out && fclose(o)) return fail(out, "write error", 0);
    return 0;
}

static int verify(const char *txt, const char *bin)
{
    corpus_file c;
    if (read_bin(bin, &c)) return 1;
    FILE *f = fopen(txt, "r");
    if (!f) return fail(txt, strerror(errno), 0);
    char buf[256]; char *line = NULL; size_t lcap = 0; ssize_t len;
    size_t i = 0;
    while ((len = getline(&line, &lcap, f)) > 0) {
        if (i == c.h.count) { fclose(f); return fail(txt, "text file has more lines than the binary", i + 1); }
        const int m = print_record(buf, sizeof buf, c.h.kind, c.h.text_format, c.a[i],
                                   c.b ? c.b[i] : 0.0, c.k ? c.k[i] : 0);
        if ((ssize_t)m != len || memcmp(buf, line, (size_t)len)) {
            fclose(f); return fail(txt, "differs from the binary", i + 1);
        }
        i++;
    }
    fclose(f); free(line);
    if (i != c.h.count) return fail(txt, "text file has fewer lines than the binary", 0);
    printf("identical  %s  (%zu records, %s)\n", bin, i,
           c.h.text_format == CORPUS_TEXT_SHORTEST ? "%a" : "%.13a");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 4 && !strcmp(argv[1], "to-bin")) {
        int r = to_bin(argv[2], argv[3], CORPUS_TEXT_CANONICAL);
        if (r == -2) r = to_bin(argv[2], argv[3], CORPUS_TEXT_SHORTEST);
        return r;
    }
    if ((argc == 3 || argc == 4) && !strcmp(argv[1], "to-text"))
        return to_text(argv[2], argc == 4 ? argv[3] : NULL);
    if (argc == 4 && !strcmp(argv[1], "verify"))
        return verify(argv[2], argv[3]);
    fprintf(stderr,
            "usage: corpus_convert to-bin <in.txt> <out.bin>\n"
            "       corpus_convert to-text <in.bin> [out.txt]\n"
            "       corpus_convert verify <in.txt> <in.bin>\n");
    return 2;
}
