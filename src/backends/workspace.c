/* workspace.c -- matern_workspace: one 64-byte-aligned allocation, zeroed at creation and
 * carved into the routing arrays, and the per-thread workspace behind workspace = NULL.
 * Compiled without instruction-set flags, since it is callable before the CPU check. */
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "internal.h"

/* Every sub-array is padded to a multiple of the alignment, which keeps arrays off
 * shared cache lines and makes the total the size multiple aligned_alloc requires. */
#define MATERN_ALIGNMENT 64

static size_t round_up_to_alignment(size_t bytes)
{
    return (bytes + MATERN_ALIGNMENT - 1) / MATERN_ALIGNMENT * MATERN_ALIGNMENT;
}

int matern_workspace_create(matern_workspace **out_workspace, size_t tile)
{
    if (!out_workspace)
        return MATERN_INVALID_ARGUMENT;

    *out_workspace = NULL;

    if (tile == 0)
        tile = MATERN_TILE_DEFAULT;

    /* the region kernels count elements in an int, so a tile has to fit in one */
    if (tile > (size_t)INT_MAX)
        return MATERN_INVALID_ARGUMENT;

    /* The tag array is allocated with 16 bytes of slack after the tile. */
    const size_t tag_bytes    = round_up_to_alignment(tile + 16);
    const size_t index_bytes  = round_up_to_alignment(tile * sizeof(size_t));
    const size_t double_bytes = round_up_to_alignment(tile * sizeof(double));

    /* the continued-fraction region's own second sort; see internal.h */
    const size_t fraction_index_bytes = round_up_to_alignment(tile * sizeof(uint32_t));

    const size_t total_bytes  = tag_bytes + index_bytes + 7 * double_bytes
                              + fraction_index_bytes;

    matern_workspace *workspace = calloc(1, sizeof *workspace);
    if (!workspace)
        return MATERN_OUT_OF_MEMORY;

    void *backing = aligned_alloc(MATERN_ALIGNMENT, total_bytes);
    if (!backing) {
        free(workspace);
        return MATERN_OUT_OF_MEMORY;
    }

    unsigned char *cursor = backing;

    workspace->region_tag   = cursor;                  cursor += tag_bytes;
    workspace->source_index = (size_t *)cursor;        cursor += index_bytes;
    workspace->sorted_h     = (double *)cursor;        cursor += double_bytes;
    workspace->sorted_nu    = (double *)cursor;        cursor += double_bytes;
    workspace->sorted_scale = (double *)cursor;        cursor += double_bytes;
    workspace->sorted_out   = (double *)cursor;         cursor += double_bytes;

    workspace->fraction_argument = (double *)cursor;   cursor += double_bytes;
    workspace->fraction_order    = (double *)cursor;   cursor += double_bytes;
    workspace->fraction_result   = (double *)cursor;   cursor += double_bytes;
    workspace->fraction_index    = (uint32_t *)cursor;

    workspace->tile    = tile;
    workspace->backing = backing;

    memset(backing, 0, total_bytes);

    *out_workspace = workspace;
    return MATERN_OK;
}

void matern_workspace_destroy(matern_workspace *workspace)
{
    if (!workspace)
        return;

    free(workspace->backing);
    free(workspace);
}

/* The workspace behind `workspace = NULL`: thread-local, not locked.  C has no portable
 * thread-exit hook, so every thread that used it calls matern_release_thread_workspace()
 * before it ends; otherwise the workspace leaks. */
static _Thread_local matern_workspace *thread_workspace = NULL;

matern_workspace *matern_thread_workspace(int *status)
{
    if (!thread_workspace) {
        const int result = matern_workspace_create(&thread_workspace,
                                                   MATERN_TILE_DEFAULT);
        if (result != MATERN_OK) {
            *status = result;
            return NULL;
        }
    }

    *status = MATERN_OK;
    return thread_workspace;
}

void matern_release_thread_workspace(void)
{
    matern_workspace_destroy(thread_workspace);
    thread_workspace = NULL;
}
