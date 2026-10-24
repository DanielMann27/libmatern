/* corpus_bin.h -- binary file format of the evaluation corpus and its header-only C99
 * reader, shared by the corpus loader, the boost-shortcut driver and the converter. */
#ifndef MATERN_CORPUS_BIN_H
#define MATERN_CORPUS_BIN_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "corpus_bin.h reads little-endian files in place"
#endif

/* File layout: a 32-byte header, then little-endian IEEE-754 arrays.  The header
 * fixes the file size, so a truncated or extended file is refused.
 *   offset  size  field
 *        0     8  magic "WMCORP01"
 *        8     4  version (1)
 *       12     4  kind: 1 = axis, 2 = reference
 *       16     8  count, the number of records
 *       24     4  text format of the values: 0 = %.13a, 1 = %a
 *       28     4  reserved, 0
 *       32        axis: double v[count]; reference: double hi[count], then
 *                 double lo[count], then int16 bits[count] (accurate bits of hi + lo) */
#define CORPUS_BIN_MAGIC   "WMCORP01"
#define CORPUS_BIN_VERSION 1u

enum { CORPUS_KIND_AXIS = 1, CORPUS_KIND_REFERENCE = 2 };
enum { CORPUS_TEXT_CANONICAL = 0, CORPUS_TEXT_SHORTEST = 1 };

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t kind;
    uint64_t count;
    uint32_t text_format;
    uint32_t reserved;
} corpus_bin_header;

typedef char corpus_bin_header_is_32_bytes[sizeof(corpus_bin_header) == 32 ? 1 : -1];

static inline void corpus_bin_header_init(corpus_bin_header *h, uint32_t kind,
                                          uint64_t count, uint32_t text_format)
{
    memset(h, 0, sizeof *h);
    memcpy(h->magic, CORPUS_BIN_MAGIC, 8);
    h->version     = CORPUS_BIN_VERSION;
    h->kind        = kind;
    h->count       = count;
    h->text_format = text_format;
}

/* Bytes per record after the header. */
static inline uint64_t corpus_bin_record_bytes(uint32_t kind)
{
    return kind == CORPUS_KIND_REFERENCE ? 2 * sizeof(double) + sizeof(int16_t)
                                         : sizeof(double);
}

/* Reads and checks the header of an open file, and checks the file size against
 * it.  Leaves the position after the header.  Returns 0, or 1 with the reason in
 * `why`. */
static inline int corpus_bin_read_header(FILE *f, const char *path,
                                         corpus_bin_header *h, char *why, size_t cap)
{
    (void)path;
    if (fread(h, sizeof *h, 1, f) != 1) {
        snprintf(why, cap, "shorter than the 32-byte header");
        return 1;
    }
    if (memcmp(h->magic, CORPUS_BIN_MAGIC, 8) != 0) {
        snprintf(why, cap, "not a corpus file (bad magic)");
        return 1;
    }
    if (h->version != CORPUS_BIN_VERSION) {
        snprintf(why, cap, "format version %u, this reader knows %u",
                 (unsigned)h->version, (unsigned)CORPUS_BIN_VERSION);
        return 1;
    }
    if (h->kind != CORPUS_KIND_AXIS && h->kind != CORPUS_KIND_REFERENCE) {
        snprintf(why, cap, "unknown kind %u", (unsigned)h->kind);
        return 1;
    }
    if (h->text_format > CORPUS_TEXT_SHORTEST || h->reserved != 0) {
        snprintf(why, cap, "unknown text format %u or nonzero reserved field",
                 (unsigned)h->text_format);
        return 1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { snprintf(why, cap, "cannot seek"); return 1; }
    const long long size = (long long)ftell(f);
    const long long want = (long long)sizeof *h
                         + (long long)(h->count * corpus_bin_record_bytes(h->kind));
    if (size != want) {
        snprintf(why, cap, "%lld bytes, the header's %llu records need %lld",
                 size, (unsigned long long)h->count, want);
        return 1;
    }
    if (fseek(f, (long)sizeof *h, SEEK_SET) != 0) { snprintf(why, cap, "cannot seek"); return 1; }
    return 0;
}

/* Reads every `stride`-th of the first `points` records of one array of the file
 * into dst: array 0 is the axis or hi, 1 is lo, 2 is bits (stored as int16,
 * widened to int).  dst holds ceil(points / stride) elements.  Returns 0 or 1. */
static inline int corpus_bin_read_array(FILE *f, const corpus_bin_header *h, int array,
                                        size_t stride, size_t points, void *dst)
{
    const size_t elem = (array == 2) ? sizeof(int16_t) : sizeof(double);
    const long long base = (long long)sizeof *h
                         + (long long)array * (long long)h->count * (long long)sizeof(double);
    if (fseek(f, (long)base, SEEK_SET) != 0) return 1;

    if (stride == 1 && array != 2)
        return fread(dst, elem, points, f) != points;

    /* Strided or widened: through a buffer, in chunks. */
    enum { CHUNK = 1 << 12 };
    unsigned char buf[CHUNK * sizeof(double)];
    size_t done = 0, kept = 0;
    while (done < points) {
        size_t take = points - done;
        if (take > CHUNK) take = CHUNK;
        if (fread(buf, elem, take, f) != take) return 1;
        for (size_t j = 0; j < take; j++) {
            if ((done + j) % stride != 0) continue;
            if (array == 2) {
                int16_t v; memcpy(&v, buf + j * elem, elem);
                ((int *)dst)[kept++] = v;
            } else {
                memcpy((double *)dst + kept++, buf + j * elem, elem);
            }
        }
        done += take;
    }
    return 0;
}

#endif /* MATERN_CORPUS_BIN_H */
