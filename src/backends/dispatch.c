/* dispatch.c -- matern_covariance(), matern_covariance_z() and the introspection calls:
 * resolves the backend once from the CPU features (or MATERN_BACKEND) and forwards.
 * Compiled without instruction-set flags, since it runs before the CPU check. */
#include <stdlib.h>
#include <string.h>

#include "internal.h"

typedef int (*evaluate_function)(double *, const double *, const double *,
                                 const double *, size_t, matern_workspace *);

/* Accessed atomically: the choice is stored first and `backend_resolved` last with
 * release order, read with acquire order.  Concurrent first calls store equal values. */
static evaluate_function  chosen_backend      = NULL;
static evaluate_function  chosen_direct       = NULL;   /* the (z, nu) entry      */
static const char        *chosen_backend_name = "none";
static const char        *chosen_scalar_isa   = "none";
static int                chosen_needs_ws     = 0;
static int                backend_resolved    = 0;

typedef struct {
    evaluate_function backend, direct;
    const char *name, *scalar_isa;
    int needs_workspace;
} backend_choice;

/* Each test covers every feature its objects are compiled for (AVX512DQ and FMA as
 * well as AVX512F); otherwise a kernel can fault. */
static int cpu_has_avx512(void)
{
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512dq")
        && __builtin_cpu_supports("fma");
}

static int cpu_has_avx2(void)
{
    return __builtin_cpu_supports("avx2")
        && __builtin_cpu_supports("fma");
}

/* `avx` is tested as well: -mfma implies -mavx, so the object may use VEX encodings. */
static int cpu_has_fma(void)
{
    return __builtin_cpu_supports("fma")
        && __builtin_cpu_supports("avx");
}

static void choose_scalar(backend_choice *c)
{
    c->name = "scalar"; c->needs_workspace = 0;
    if (cpu_has_avx512()) {
        c->backend = matern_evaluate_scalar_avx512; c->direct = matern_evaluate_scalar_avx512_z;
        c->scalar_isa = "avx512";
    } else if (cpu_has_avx2()) {
        c->backend = matern_evaluate_scalar_avx2;   c->direct = matern_evaluate_scalar_avx2_z;
        c->scalar_isa = "avx2";
    } else if (cpu_has_fma()) {
        c->backend = matern_evaluate_scalar_fma;    c->direct = matern_evaluate_scalar_fma_z;
        c->scalar_isa = "fma";
    } else {
        c->backend = matern_evaluate_scalar;        c->direct = matern_evaluate_scalar_z;
        c->scalar_isa = "portable";
    }
}

static void set_vector(backend_choice *c, evaluate_function f, evaluate_function fz,
                       const char *name)
{
    c->backend = f; c->direct = fz; c->name = name;
    c->scalar_isa = "none"; c->needs_workspace = 1;
}

static void choose_backend(backend_choice *c)
{
    /* must precede every __builtin_cpu_supports() */
    __builtin_cpu_init();

    const char *forced = getenv("MATERN_BACKEND");

    if (forced) {
        if (!strcmp(forced, "avx512") && cpu_has_avx512()) {
            set_vector(c, matern_evaluate_avx512, matern_evaluate_avx512_z, "avx512");
            return;
        }
        if (!strcmp(forced, "avx2") && cpu_has_avx2()) {
            set_vector(c, matern_evaluate_avx2, matern_evaluate_avx2_z, "avx2");
            return;
        }
        /* `matched` needs only the x86-64 baseline in the default build (FMA=no). */
        if (!strcmp(forced, "matched")) {
            set_vector(c, matern_evaluate_matched, matern_evaluate_matched_z, "matched");
            return;
        }
        if (!strcmp(forced, "matched_fma") && cpu_has_fma()) {
            set_vector(c, matern_evaluate_matched_fma, matern_evaluate_matched_fma_z, "matched_fma");
            return;
        }
        /* `scalar`: the best scalar build this CPU runs; `scalar_portable`: the
         * baseline build. */
        if (!strcmp(forced, "scalar")) {
            choose_scalar(c);
            return;
        }
        if (!strcmp(forced, "scalar_portable")) {
            c->backend = matern_evaluate_scalar; c->direct = matern_evaluate_scalar_z;
            c->name = "scalar_portable"; c->scalar_isa = "portable"; c->needs_workspace = 0;
            return;
        }
        /* An unrecognised or unrunnable request falls through to the automatic choice. */
    }

    /* Automatic chain: avx512 -> avx2 -> matched_fma -> matched.  `scalar` is selected
     * only through MATERN_BACKEND. */
    if (cpu_has_avx512()) {
        set_vector(c, matern_evaluate_avx512, matern_evaluate_avx512_z, "avx512");
    } else if (cpu_has_avx2()) {
        set_vector(c, matern_evaluate_avx2, matern_evaluate_avx2_z, "avx2");
    } else if (cpu_has_fma()) {
        set_vector(c, matern_evaluate_matched_fma, matern_evaluate_matched_fma_z, "matched_fma");
    } else {
        set_vector(c, matern_evaluate_matched, matern_evaluate_matched_z, "matched");
    }
}

/* The resolved backend, choosing it on the first call. */
static evaluate_function resolve_backend(const char **name, evaluate_function *direct,
                                         int *needs_workspace)
{
    if (!__atomic_load_n(&backend_resolved, __ATOMIC_ACQUIRE)) {
        backend_choice c = { NULL, NULL, "none", "none", 1 };
        choose_backend(&c);
        __atomic_store_n(&chosen_backend, c.backend, __ATOMIC_RELAXED);
        __atomic_store_n(&chosen_direct, c.direct, __ATOMIC_RELAXED);
        __atomic_store_n(&chosen_backend_name, c.name, __ATOMIC_RELAXED);
        __atomic_store_n(&chosen_scalar_isa, c.scalar_isa, __ATOMIC_RELAXED);
        __atomic_store_n(&chosen_needs_ws, c.needs_workspace, __ATOMIC_RELAXED);
        __atomic_store_n(&backend_resolved, 1, __ATOMIC_RELEASE);
    }
    if (name)
        *name = __atomic_load_n(&chosen_backend_name, __ATOMIC_RELAXED);
    if (direct)
        *direct = __atomic_load_n(&chosen_direct, __ATOMIC_RELAXED);
    if (needs_workspace)
        *needs_workspace = __atomic_load_n(&chosen_needs_ws, __ATOMIC_RELAXED);
    return __atomic_load_n(&chosen_backend, __ATOMIC_RELAXED);
}

const char *matern_backend_name(void)
{
    const char *name;
    resolve_backend(&name, NULL, NULL);
    return name;
}

const char *matern_scalar_isa(void)
{
    resolve_backend(NULL, NULL, NULL);
    return __atomic_load_n(&chosen_scalar_isa, __ATOMIC_RELAXED);
}

int matern_covariance_z(double *out,
                        const double *z, const double *nu,
                        size_t n, matern_workspace *workspace)
{
    evaluate_function direct;
    int needs_workspace;
    const evaluate_function backend = resolve_backend(NULL, &direct, &needs_workspace);

    if (!backend || !direct)
        return MATERN_UNSUPPORTED_CPU;
    if (n == 0)
        return MATERN_OK;
    if (!out || !z || !nu)
        return MATERN_INVALID_ARGUMENT;

    if (!workspace && needs_workspace) {
        int status;
        workspace = matern_thread_workspace(&status);
        if (!workspace)
            return status;
    }

    /* `nu` fills the unused scale slot: the vector pipeline loads that pointer while
     * bucketing, so it must be a readable array of n doubles. */
    return direct(out, z, nu, nu, n, workspace);
}

int matern_covariance(double *out,
                      const double *h, const double *nu, const double *scale,
                      size_t n, matern_workspace *workspace)
{
    int needs_workspace;
    const evaluate_function backend = resolve_backend(NULL, NULL, &needs_workspace);

    if (!backend)
        return MATERN_UNSUPPORTED_CPU;
    if (n == 0)
        return MATERN_OK;
    if (!out || !h || !nu || !scale)
        return MATERN_INVALID_ARGUMENT;

    if (!workspace && needs_workspace) {
        int status;
        workspace = matern_thread_workspace(&status);
        if (!workspace)
            return status;
    }

    return backend(out, h, nu, scale, n, workspace);
}
