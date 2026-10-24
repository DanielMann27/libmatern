/* wm_timing.h -- timing rules and primitives shared by all benchmark stages: a fixed
 * repetition count (not budget-driven), WM_WARMUP_REPS untimed passes first, the
 * nearest-rank distribution of the samples, and the raw samples to a sidecar file. */
#ifndef WM_TIMING_H
#define WM_TIMING_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

/* Default repetition count of every stage (REPS of run_benchmark.sh and run_gpu.sh
 * overrides it), so every statistic in a record is over the same number of samples. */
#define WM_REPS_MIN_DEFAULT  65
#define WM_REPS_MAX_DEFAULT  65
#define WM_BUDGET_DEFAULT   1e30      /* effectively no budget: reps are fixed */
#define WM_WARMUP_REPS        2

/* Timing covers the whole law unless a driver's --time-points N (TIME_MAX_POINTS)
 * times only the first N points, recorded in ns_points; pipeline.sh and run_gpu.sh
 * refuse it.  timing_scope records a whole call or the kernel alone. */

/* The timing distribution over the repetitions, in ns per element. */
typedef struct {
    double min, q1, med, q3, p95, max, mean, iqr, spread;
    int reps;
} wm_time;

/* Size of the largest cache of `cpu` and the number of CPUs sharing it, from sysfs. */
static inline long wm_llc_bytes_shared(int cpu, int *shared_cpus)
{
    long best = 0; int best_share = 1;
    for (int idx = 0; idx < 10; idx++) {
        char path[256]; FILE *f;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, idx);
        if (!(f = fopen(path, "r"))) continue;
        long kb = 0; char unit = 'K';
        if (fscanf(f, "%ld%c", &kb, &unit) < 1) { fclose(f); continue; }
        fclose(f);
        long bytes = kb * ((unit == 'M') ? 1024L * 1024L : 1024L);
        int share = 1;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list",
                 cpu, idx);
        if ((f = fopen(path, "r"))) {
            char buf[512] = {0};
            if (fgets(buf, sizeof buf, f)) {
                share = 0;
                for (char *q = buf; *q; ) {
                    long a = strtol(q, &q, 10), b = a;
                    if (*q == '-') { q++; b = strtol(q, &q, 10); }
                    share += (int)(b - a + 1);
                    if (*q == ',') q++; else break;
                }
                if (share < 1) share = 1;
            }
            fclose(f);
        }
        if (bytes > best) { best = bytes; best_share = share; }
    }
    if (shared_cpus) *shared_cpus = best_share;
    return best;
}

static FILE *wm_raw_fp = NULL;

/* Scheduler counters of this process (-1 where unavailable).  A row bracketed by
 * wm_sched_begin() and wm_raw_emit() records its thread count, migrations and
 * involuntary context switches in the `.sched.csv` sidecar. */
typedef struct { long threads, migrations, invol_switches; } wm_sched;

static inline wm_sched wm_sched_now(void)
{
    wm_sched s = { -1, -1, -1 };
    char line[256];
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "Threads:", 8)) { s.threads = atol(line + 8); break; }
        fclose(f);
    }
    f = fopen("/proc/self/sched", "r");
    if (f) {
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "se.nr_migrations", 16)) {
                const char *c = strchr(line, ':');
                if (c) s.migrations = atol(c + 1);
                break;
            }
        fclose(f);
    }
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) s.invol_switches = ru.ru_nivcsw;
    return s;
}

static FILE    *wm_sched_fp = NULL;
static char     wm_sched_path[4096];   /* opened at the first recorded row */
static wm_sched wm_sched_mark;
static int      wm_sched_marked = 0;
/* Set by the single-threaded drivers (benchmark_cpu, suppliers): a second
 * thread during a timed row is reported on stderr. */
static int      wm_expect_single_thread = 0;

static inline void wm_sched_begin(void)
{
    wm_sched_mark = wm_sched_now();
    wm_sched_marked = 1;
}

static inline void wm_raw_open(const char *path)
{
    if (!path || !*path) return;
    wm_raw_fp = fopen(path, "w");
    if (wm_raw_fp)
        fprintf(wm_raw_fp, "device,supplier,family,law,backend,n,rep,ns_per_elem\n");
    snprintf(wm_sched_path, sizeof wm_sched_path, "%s.sched.csv", path);
}

static inline void wm_raw_emit(const char *device, const char *supplier,
                               const char *fam, const char *law,
                               const char *backend, size_t n,
                               const double *t, int reps)
{
    if (wm_sched_marked) {
        const wm_sched e = wm_sched_now();
        const long mig = (e.migrations >= 0 && wm_sched_mark.migrations >= 0)
                         ? e.migrations - wm_sched_mark.migrations : -1;
        const long sw  = (e.invol_switches >= 0 && wm_sched_mark.invol_switches >= 0)
                         ? e.invol_switches - wm_sched_mark.invol_switches : -1;
        if (!wm_sched_fp && wm_sched_path[0]) {
            wm_sched_fp = fopen(wm_sched_path, "w");
            if (wm_sched_fp)
                fprintf(wm_sched_fp, "device,supplier,family,law,backend,n,threads,"
                                     "migrations,invol_switches\n");
        }
        if (wm_sched_fp) {
            fprintf(wm_sched_fp, "%s,%s,%s,%s,%s,%zu,%ld,%ld,%ld\n",
                    device, supplier, fam, law, backend, n, e.threads, mig, sw);
            fflush(wm_sched_fp);
        }
        static int warned = 0;
        if (wm_expect_single_thread && e.threads > 1 && !warned) {
            fprintf(stderr, "WARNING: %ld threads in this process during a timed row "
                            "(%s %s %s); the timing is meant to be single-threaded\n",
                    e.threads, fam, law, backend);
            warned = 1;
        }
        wm_sched_marked = 0;
    }
    if (!wm_raw_fp) return;
    for (int k = 0; k < reps; k++)
        fprintf(wm_raw_fp, "%s,%s,%s,%s,%s,%zu,%d,%.6f\n",
                device, supplier, fam, law, backend, n, k, t[k]);
    fflush(wm_raw_fp);
}

static inline void wm_raw_close(void)
{
    if (wm_raw_fp)   { fclose(wm_raw_fp);   wm_raw_fp = NULL; }
    if (wm_sched_fp) { fclose(wm_sched_fp); wm_sched_fp = NULL; }
}
typedef struct { int min_reps, max_reps; double budget; } wm_rep_policy;

/* The host clock of every wall-clock timing (build/detect reports it); device
 * kernel-only timings use CUDA events. */
#define WM_TIMING_CLOCK      CLOCK_MONOTONIC
#define WM_TIMING_CLOCK_NAME "CLOCK_MONOTONIC"

static inline double wm_now_s(void)
{
    struct timespec t;
    clock_gettime(WM_TIMING_CLOCK, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static inline int wm_cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Nearest rank, the rule of wm_metrics.h's accuracy quantiles: every reported value
 * occurred. */
static inline double wm_tq(const double *sorted, int m, double f)
{ return m > 0 ? sorted[(int)(f * (double)(m - 1) + 0.5)] : 0.0; }

/* Sorts the per-repetition samples and reduces them to the reported statistics. */
static inline wm_time wm_time_reduce(double *t, int i)
{
    /* memset, not `= {0}`: the header is also compiled as C++ by nvcc, which
     * warns -Wmissing-field-initializers on the brace form. */
    wm_time r; memset(&r, 0, sizeof r);
    if (i <= 0) return r;
    r.reps = i;
    qsort(t, (size_t)i, sizeof *t, wm_cmp_d);
    r.min = t[0];
    r.q1  = wm_tq(t, i, 0.25);
    r.med = wm_tq(t, i, 0.50);
    r.q3  = wm_tq(t, i, 0.75);
    r.p95 = wm_tq(t, i, 0.95);
    r.max = t[i-1];
    r.iqr = r.q3 - r.q1;
    /* spread: the excess of the maximum over the minimum, in percent */
    r.spread = (r.min > 0.0) ? 100.0 * (r.max - r.min) / r.min : 0.0;
    double sm = 0; for (int k = 0; k < i; k++) sm += t[k];
    r.mean = sm / i;
    return r;
}

#endif /* WM_TIMING_H */
