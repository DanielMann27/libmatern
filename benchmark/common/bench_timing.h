/* bench_timing.h -- timing rules and primitives shared by all benchmark stages: a fixed
 * repetition count (not budget-driven), BENCH_WARMUP_REPS untimed passes first, the
 * nearest-rank distribution of the samples, and the raw samples to a sidecar file. */
#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

/* Default repetition count of every stage (REPS of run_benchmark.sh and run_gpu.sh
 * overrides it), so every statistic in a record is over the same number of samples. */
#define BENCH_REPS_MIN_DEFAULT  65
#define BENCH_REPS_MAX_DEFAULT  65
#define BENCH_BUDGET_DEFAULT   1e30      /* effectively no budget: reps are fixed */
#define BENCH_WARMUP_REPS        2

/* Timing covers the whole law unless a driver's --time-points N (TIME_MAX_POINTS)
 * times only the first N points, recorded in ns_points; pipeline.sh and run_gpu.sh
 * refuse it.  timing_scope records a whole call or the kernel alone. */

/* The timing distribution over the repetitions, in ns per element. */
typedef struct {
    double min, q1, med, q3, p95, max, mean, iqr, spread;
    int reps;
} bench_time;

/* Size of the largest cache of `cpu` and the number of CPUs sharing it, from sysfs. */
static inline long bench_llc_bytes_shared(int cpu, int *shared_cpus)
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

static FILE *bench_raw_fp = NULL;

/* Scheduler counters of this process (-1 where unavailable).  A row bracketed by
 * bench_sched_begin() and bench_raw_emit() records its thread count, migrations,
 * involuntary context switches and the clock right after it (bench_cpu_mhz) in the
 * `.sched.csv` sidecar. */
typedef struct { long threads, migrations, invol_switches; } bench_sched;

static inline bench_sched bench_sched_now(void)
{
    bench_sched s = { -1, -1, -1 };
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

/* The clock of the CPU this process last ran on, in MHz: cpufreq's scaling_cur_freq, the
 * kernel's estimate of its recent frequency; -1 without Linux /proc or cpufreq.  The CPU is
 * field 39 of /proc/self/stat, which needs no _GNU_SOURCE (sched_getcpu() does). */
static inline long bench_cpu_mhz(void)
{
    char buf[1024], path[96];
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) return -1;
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = '\0';
    const char *q = strrchr(buf, ')');      /* the command name may hold spaces */
    if (!q) return -1;
    int field = 2;
    for (q++; *q && field < 39; q++)
        if (*q == ' ') field++;
    if (field != 39) return -1;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", atoi(q));
    f = fopen(path, "r");
    if (!f) return -1;
    long khz = -1;
    if (fscanf(f, "%ld", &khz) != 1) khz = -1;
    fclose(f);
    return khz > 0 ? khz / 1000 : -1;
}

static FILE    *bench_sched_fp = NULL;
static char     bench_sched_path[4096];   /* opened at the first recorded row */
static bench_sched bench_sched_mark;
static int      bench_sched_marked = 0;
/* Set by the single-threaded drivers (benchmark_cpu, suppliers): a second
 * thread during a timed row is reported on stderr. */
static int      bench_expect_single_thread = 0;

static inline void bench_sched_begin(void)
{
    bench_sched_mark = bench_sched_now();
    bench_sched_marked = 1;
}

static inline void bench_raw_open(const char *path)
{
    if (!path || !*path) return;
    bench_raw_fp = fopen(path, "w");
    if (bench_raw_fp)
        fprintf(bench_raw_fp, "device,supplier,family,law,backend,n,rep,ns_per_elem\n");
    snprintf(bench_sched_path, sizeof bench_sched_path, "%s.sched.csv", path);
}

static inline void bench_raw_emit(const char *device, const char *supplier,
                               const char *fam, const char *law,
                               const char *backend, size_t n,
                               const double *t, int reps)
{
    if (bench_sched_marked) {
        const bench_sched e = bench_sched_now();
        const long mig = (e.migrations >= 0 && bench_sched_mark.migrations >= 0)
                         ? e.migrations - bench_sched_mark.migrations : -1;
        const long sw  = (e.invol_switches >= 0 && bench_sched_mark.invol_switches >= 0)
                         ? e.invol_switches - bench_sched_mark.invol_switches : -1;
        const long mhz = bench_cpu_mhz();
        if (!bench_sched_fp && bench_sched_path[0]) {
            bench_sched_fp = fopen(bench_sched_path, "w");
            if (bench_sched_fp)
                fprintf(bench_sched_fp, "device,supplier,family,law,backend,n,threads,"
                                     "migrations,invol_switches,mhz_end\n");
        }
        if (bench_sched_fp) {
            fprintf(bench_sched_fp, "%s,%s,%s,%s,%s,%zu,%ld,%ld,%ld,%ld\n",
                    device, supplier, fam, law, backend, n, e.threads, mig, sw, mhz);
            fflush(bench_sched_fp);
        }
        static int warned = 0;
        if (bench_expect_single_thread && e.threads > 1 && !warned) {
            fprintf(stderr, "WARNING: %ld threads in this process during a timed row "
                            "(%s %s %s); the timing is meant to be single-threaded\n",
                    e.threads, fam, law, backend);
            warned = 1;
        }
        bench_sched_marked = 0;
    }
    if (!bench_raw_fp) return;
    for (int k = 0; k < reps; k++)
        fprintf(bench_raw_fp, "%s,%s,%s,%s,%s,%zu,%d,%.6f\n",
                device, supplier, fam, law, backend, n, k, t[k]);
    fflush(bench_raw_fp);
}

static inline void bench_raw_close(void)
{
    if (bench_raw_fp)   { fclose(bench_raw_fp);   bench_raw_fp = NULL; }
    if (bench_sched_fp) { fclose(bench_sched_fp); bench_sched_fp = NULL; }
}
typedef struct { int min_reps, max_reps; double budget; } bench_rep_policy;

/* The host clock of every wall-clock timing (build/detect reports it); device
 * kernel-only timings use CUDA events. */
#define BENCH_TIMING_CLOCK      CLOCK_MONOTONIC
#define BENCH_TIMING_CLOCK_NAME "CLOCK_MONOTONIC"

static inline double bench_now_s(void)
{
    struct timespec t;
    clock_gettime(BENCH_TIMING_CLOCK, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static inline int bench_cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Nearest rank, the rule of bench_metrics.h's accuracy quantiles: every reported value
 * occurred. */
static inline double bench_time_quantile(const double *sorted, int m, double f)
{ return m > 0 ? sorted[(int)(f * (double)(m - 1) + 0.5)] : 0.0; }

/* Sorts the per-repetition samples and reduces them to the reported statistics. */
static inline bench_time bench_time_reduce(double *t, int i)
{
    /* memset, not `= {0}`: the header is also compiled as C++ by nvcc, which
     * warns -Wmissing-field-initializers on the brace form. */
    bench_time r; memset(&r, 0, sizeof r);
    if (i <= 0) return r;
    r.reps = i;
    qsort(t, (size_t)i, sizeof *t, bench_cmp_d);
    r.min = t[0];
    r.q1  = bench_time_quantile(t, i, 0.25);
    r.med = bench_time_quantile(t, i, 0.50);
    r.q3  = bench_time_quantile(t, i, 0.75);
    r.p95 = bench_time_quantile(t, i, 0.95);
    r.max = t[i-1];
    r.iqr = r.q3 - r.q1;
    /* spread: the excess of the maximum over the minimum, in percent */
    r.spread = (r.min > 0.0) ? 100.0 * (r.max - r.min) / r.min : 0.0;
    double sm = 0; for (int k = 0; k < i; k++) sm += t[k];
    r.mean = sm / i;
    return r;
}

#endif /* BENCH_TIMING_H */
