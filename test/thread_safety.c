/* thread_safety.c -- N threads on identical input reproduce the single-thread result bit for
 * bit, with a workspace per thread and with the per-thread cache (workspace = NULL); every
 * thread evaluates the same whole array, so the batching is identical. */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matern.h"

#define THREAD_COUNT   8
#define POINT_COUNT    200000
#define REPEAT_COUNT   6

typedef int (*evaluate_fn)(double *, const double *, const double *,
                           const double *, size_t, matern_workspace *);

static double *h_input, *nu_input, *scale_input;
static double *expected;

/* The points: log-uniform over the validated domain, so every region is hit. */
static uint64_t random_state = 0x243F6A8885A308D3ull;
static double random_unit(void)
{
    uint64_t x = (random_state += 0x9E3779B97F4A7C15ull);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (double)(x >> 11) * 0x1.0p-53;
}

static void build_points(void)
{
    for (size_t i = 0; i < POINT_COUNT; i++) {
        const double order    = 1e-3 * pow(1e3 / 1e-3, random_unit());
        const double argument = 1e-3 * pow(900.0 / 1e-3, random_unit());

        nu_input[i]    = order;
        scale_input[i] = 0.5 + 2.0 * random_unit();
        h_input[i]     = argument * scale_input[i] / sqrt(2.0 * order);
    }
}

/* One worker: the same whole array, into its own output buffer. */
typedef struct {
    double            *out;
    evaluate_fn        evaluate;
    int                own_workspace;  /* 0 selects the NULL per-thread cache path */
    matern_workspace  *shared;         /* the positive control only */
    int                status;
} worker;

/* Positive control, MATERN_THREAD_CONTROL=1: all threads share one workspace, which
 * matern.h forbids.  It fires when the program reports differences (exit non-zero) or
 * crashes, and make threads fails on any other outcome; it cannot fire on the scalar
 * backends, which use no workspace. */
static matern_workspace *control_workspace = NULL;

static void *run_worker(void *argument)
{
    worker *w = (worker *)argument;
    matern_workspace *workspace = w->shared;   /* non-NULL only in the control */

    if (!workspace && w->own_workspace
        && matern_workspace_create(&workspace, 0) != MATERN_OK) {
        w->status = -1;
        return NULL;
    }

    for (int r = 0; r < REPEAT_COUNT; r++) {
        const int status = w->evaluate(w->out, h_input, nu_input, scale_input,
                                       POINT_COUNT, workspace);
        if (status != MATERN_OK) { w->status = status; break; }
    }

    if (w->shared)
        ;                                     /* the caller owns it */
    else if (workspace)
        matern_workspace_destroy(workspace);
    else
        matern_release_thread_workspace();    /* the thread owns what it cached */

    return NULL;
}

/* Setup failures (no memory, no thread) of check_case: they fail the check, and in
 * the positive control they must not be read as the control firing. */
static int setup_failures = 0;

/* Returns the number of threads whose output differs from `expected`, or 1 when
 * the case could not be set up (counted in setup_failures as well). */
static int check_case(const char *name, evaluate_fn evaluate, int own_workspace)
{
    pthread_t thread[THREAD_COUNT];
    worker    work[THREAD_COUNT];
    double   *buffer = malloc((size_t)THREAD_COUNT * POINT_COUNT * sizeof *buffer);

    if (!buffer) { printf("  %-34s OUT OF MEMORY\n", name); setup_failures++; return 1; }

    for (int t = 0; t < THREAD_COUNT; t++) {
        work[t].out           = buffer + (size_t)t * POINT_COUNT;
        work[t].evaluate      = evaluate;
        work[t].own_workspace = own_workspace;
        work[t].shared        = control_workspace;
        work[t].status        = MATERN_OK;
        memset(work[t].out, 0xAA, POINT_COUNT * sizeof *buffer);
    }

    for (int t = 0; t < THREAD_COUNT; t++)
        if (pthread_create(&thread[t], NULL, run_worker, &work[t]) != 0) {
            printf("  %-34s pthread_create failed\n", name);
            for (int s = 0; s < t; s++)       /* the threads already started */
                pthread_join(thread[s], NULL);
            free(buffer);
            setup_failures++;
            return 1;
        }

    for (int t = 0; t < THREAD_COUNT; t++)
        pthread_join(thread[t], NULL);

    int bad_threads = 0, worst_points = 0;
    for (int t = 0; t < THREAD_COUNT; t++) {
        if (work[t].status != MATERN_OK) {
            printf("  %-34s thread %d returned status %d\n", name, t, work[t].status);
            bad_threads++;
            continue;
        }
        /* Bit comparison: identical arithmetic on identical input. */
        int differing = 0;
        for (size_t i = 0; i < POINT_COUNT; i++)
            if (memcmp(&work[t].out[i], &expected[i], sizeof(double)) != 0)
                differing++;
        if (differing) {
            if (differing > worst_points) worst_points = differing;
            bad_threads++;
        }
    }

    if (bad_threads)
        printf("  %-34s FAIL: %d of %d threads differ, worst %d of %d points\n",
               name, bad_threads, THREAD_COUNT, worst_points, POINT_COUNT);
    else
        printf("  %-34s ok: %d threads x %d reps, %d points, 0 differing bits\n",
               name, THREAD_COUNT, REPEAT_COUNT, POINT_COUNT);

    free(buffer);
    return bad_threads;
}

int main(void)
{
    h_input     = malloc(POINT_COUNT * sizeof *h_input);
    nu_input    = malloc(POINT_COUNT * sizeof *nu_input);
    scale_input = malloc(POINT_COUNT * sizeof *scale_input);
    expected    = malloc(POINT_COUNT * sizeof *expected);
    double *slice_result = malloc(POINT_COUNT * sizeof *slice_result);

    if (!h_input || !nu_input || !scale_input || !expected || !slice_result) {
        puts("out of memory");
        return 1;
    }

    /* Line-buffered, so output survives if the positive control crashes. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    build_points();

    const char *control = getenv("MATERN_THREAD_CONTROL");
    const int   is_control = control && control[0] == '1';

    printf("thread_safety: backend %s%s\n\n", matern_backend_name(),
           is_control ? "   [POSITIVE CONTROL: one workspace SHARED, must fail]" : "");

    int failures = 0;

    /* The single-thread answer every strict case is held to. */
    matern_workspace *workspace = NULL;
    if (matern_workspace_create(&workspace, 0) != MATERN_OK) {
        puts("could not create a workspace");
        return 1;
    }
    if (matern_covariance(expected, h_input, nu_input, scale_input,
                          POINT_COUNT, workspace) != MATERN_OK) {
        puts("the single-thread reference evaluation failed");
        return 1;
    }

    if (is_control && matern_workspace_create(&control_workspace, 0) != MATERN_OK) {
        puts("could not create the control workspace");
        return 1;
    }

    failures += check_case("covariance, workspace per thread",
                           matern_covariance, 1);
    failures += check_case("covariance, per-thread cache",
                           matern_covariance, 0);

    /* Batching (not a thread property): points whose bits change when one array is
     * evaluated in chunks instead of one call.  Reported, not asserted: a horizontal
     * exit test across a vector makes the trip count depend on which points share it. */
    if (matern_covariance(expected, h_input, nu_input, scale_input,
                          POINT_COUNT, workspace) == MATERN_OK) {
        const size_t chunk = 7000;    /* not a multiple of the tile size */
        for (size_t base = 0; base < POINT_COUNT; base += chunk) {
            const size_t m = (POINT_COUNT - base < chunk) ? POINT_COUNT - base : chunk;
            matern_covariance(slice_result + base, h_input + base, nu_input + base,
                              scale_input + base, m, workspace);
        }
        size_t moved = 0; double worst = 0.0;
        for (size_t i = 0; i < POINT_COUNT; i++)
            if (memcmp(&slice_result[i], &expected[i], sizeof(double)) != 0) {
                moved++;
                if (expected[i] != 0.0) {
                    const double d = fabs(slice_result[i] - expected[i]) / fabs(expected[i]);
                    if (d > worst) worst = d;
                }
            }
        printf("\n  one call vs %zu-point calls (reported) %zu of %d points differ, "
               "largest relative %.3e\n", chunk, moved, POINT_COUNT, worst);
    }

    matern_workspace_destroy(workspace);
    if (control_workspace)
        matern_workspace_destroy(control_workspace);

    if (is_control) {
        /* The control fires when the test fails: differences, a non-zero exit.  A
         * setup failure is not a firing: no shared-workspace evaluation took place,
         * so the run says so and exits non-zero without "control fired", which
         * make threads reports as a control that did not run. */
        if (setup_failures) {
            printf("\nCONTROL DID NOT RUN: %d case(s) could not be set up\n", setup_failures);
            free(h_input); free(nu_input); free(scale_input); free(expected); free(slice_result);
            return 1;
        }
        printf("\n%s\n", failures ? "control fired: the test detects a shared workspace"
                                  : "CONTROL DID NOT FIRE -- the test cannot see corruption");
        free(h_input); free(nu_input); free(scale_input); free(expected); free(slice_result);
        return failures != 0;
    }

    printf("\n%s\n", failures ? "THREAD SAFETY FAILED" : "thread safety holds");
    free(h_input); free(nu_input); free(scale_input); free(expected); free(slice_result);
    return failures != 0;
}
