/* threads_cpu.c -- throughput against thread count with pinned threads (weak scaling: each
 * thread evaluates its own n points), with a bit-identity check across threads and the ULP
 * of thread 0's slice (see the accompanying paper, timing methodology). */
#define _GNU_SOURCE            /* pthread_setaffinity_np, sched_getcpu */

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "matern.h"
#include "corpus_dataset.h"
#include "common/wm_metrics.h"   /* WM_SCHEMA: these rows carry the same schema number */
#include "cpu_wm.h"              /* the external suppliers, same objects suppliers.c times */
#include "cpu_wm_registry.h"     /* the supplier list, shared with suppliers.c */
#include "common/wm_timing.h"

#define MAX_THREADS       256
#define MAX_THREAD_STEPS   16

/* The CPUs this process may run on, from the affinity mask (not nproc, which honours
 * OMP_NUM_THREADS), in ascending order; physical cores are counted from the sysfs sibling
 * lists, since CPU numbering does not show which CPUs share a core. */
typedef struct {
    int cpu[MAX_THREADS];
    int count;              /* usable entries in `cpu`                      */
    int physical_cores;     /* how many of them are on distinct cores       */
    int siblings_per_core;  /* 1 = no SMT, 2 = two-way SMT                  */
} cpu_topology;

/* Whether `cpu` shares a core with one already chosen (sysfs sibling list). */
static int shares_a_core_with_any(int cpu, const int *chosen, int chosen_count)
{
    char path[128];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);

    FILE *f = fopen(path, "r");
    if (!f)
        return 0;                       /* no topology exposed: treat as distinct */

    char line[256] = {0};
    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }
    fclose(f);

    /* the list is comma- and dash-separated; only exact members matter here */
    for (const char *p = line; *p; ) {
        char *end;
        const long first = strtol(p, &end, 10);
        if (end == p) break;
        long last = first;
        if (*end == '-') { p = end + 1; last = strtol(p, &end, 10); }
        for (long s = first; s <= last; s++)
            for (int k = 0; k < chosen_count; k++)
                if (chosen[k] == (int)s)
                    return 1;
        p = (*end == ',') ? end + 1 : end;
        if (*end != ',') break;
    }
    return 0;
}

/* Pinning order: `dense` (the default) fills CPUs in ascending order; `spread` deals them
 * round-robin across the NUMA nodes, which separates a cache-capacity loss from a clock cap. */
static int numa_node_of(int cpu)
{
    for (int nd = 0; nd < 64; nd++) {
        char path[256];
        snprintf(path, sizeof path,
                 "/sys/devices/system/node/node%d/cpu%d", nd, cpu);
        if (access(path, F_OK) == 0) return nd;
    }
    return 0;
}

static void spread_topology(cpu_topology *topo)
{
    int node[MAX_THREADS], nodes[64], nnodes = 0;
    for (int i = 0; i < topo->count; i++) {
        node[i] = numa_node_of(topo->cpu[i]);
        int seen = 0;
        for (int k = 0; k < nnodes; k++) if (nodes[k] == node[i]) { seen = 1; break; }
        if (!seen && nnodes < 64) nodes[nnodes++] = node[i];
    }
    if (nnodes < 2) return;                   /* nothing to spread across */
    int out[MAX_THREADS], m = 0;
    for (int round = 0; m < topo->count; round++) {
        int placed = 0;
        for (int k = 0; k < nnodes && m < topo->count; k++) {
            int seen = 0;
            for (int i = 0; i < topo->count; i++) {
                if (node[i] != nodes[k]) continue;
                if (seen++ == round) { out[m++] = topo->cpu[i]; placed = 1; break; }
            }
        }
        if (!placed) break;
    }
    for (int i = 0; i < m; i++) topo->cpu[i] = out[i];
}

/* Mean current clock of the pinned CPUs in MHz; 0 when cpufreq is not exposed. */
static double mean_mhz(const int *cpus, int count)
{
    double sum = 0.0; int have = 0;
    for (int i = 0; i < count; i++) {
        char path[256]; FILE *f;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpus[i]);
        if (!(f = fopen(path, "r"))) continue;
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1 && khz > 0) { sum += khz / 1000.0; have++; }
        fclose(f);
    }
    return have ? sum / have : 0.0;
}

static cpu_topology discover_topology(void)
{
    cpu_topology topology;
    memset(&topology, 0, sizeof topology);

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) {
        /* Fallback only; not nproc. */
        topology.cpu[0] = 0;
        topology.count = 1;
        topology.physical_cores = 1;
        topology.siblings_per_core = 1;
        return topology;
    }

    int available[MAX_THREADS];
    int available_count = 0;
    for (int c = 0; c < CPU_SETSIZE && available_count < MAX_THREADS; c++)
        if (CPU_ISSET(c, &allowed))
            available[available_count++] = c;

    /* First pass: one CPU per physical core, in ascending order. */
    for (int i = 0; i < available_count && topology.count < MAX_THREADS; i++)
        if (!shares_a_core_with_any(available[i], topology.cpu, topology.count))
            topology.cpu[topology.count++] = available[i];

    topology.physical_cores = topology.count;
    topology.siblings_per_core =
        topology.physical_cores ? available_count / topology.physical_cores : 1;

    /* CPUs are used in affinity-mask order, so a sweep can run past the physical cores. */
    memcpy(topology.cpu, available, sizeof(int) * (size_t)available_count);
    topology.count = available_count;

    return topology;
}

/* One entry per family: (h, nu, s) and (z, nu); deriving h from z for the first would round
 * and time a different computation. */
typedef int (*evaluate_fn)(double *, const double *, const double *,
                           const double *, size_t, matern_workspace *);
typedef int (*evaluate_z_fn)(double *, const double *, const double *,
                             size_t, matern_workspace *);

typedef struct {
    const double   *h, *nu, *scale;  /* scale NULL on a (z, nu) set           */
    double         *out;
    size_t          n;
    int             reps;
    int             z_family;        /* 1 = call the (z, nu) entry            */
    evaluate_fn     evaluate;
    evaluate_z_fn   evaluate_z;
    /* An external supplier instead of the library when non-NULL, in the same harness. */
    const wmc_supplier *sup;         /* NULL = this library                   */
    int             assembly;        /* WMC_ASM_* when sup != NULL            */
    int             pin_to;          /* logical CPU, or -1 for "do not pin"   */
    int             ran_on;          /* sched_getcpu() after the warm-up      */
    double          elapsed_best;    /* seconds, min over reps                */
    double         *elapsed_rep;     /* seconds, every rep (NULL = not kept)  */
    int             ws_needed;       /* 1 = a workspace per call (library, not scalar) */
    int             status;
    pthread_barrier_t *barrier;
} worker;

static void *run_worker(void *argument)
{
    worker *w = (worker *)argument;

    if (w->pin_to >= 0) {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(w->pin_to, &one);
        if (pthread_setaffinity_np(pthread_self(), sizeof one, &one) != 0)
            w->status = -100;         /* reported as a pinning failure */
    }

    /* One workspace per call, created and destroyed inside each timed repetition and
     * warm-up pass by the thread that uses it (first touch places it on that node). */
    #define WM_THREAD_CALL(ST, OUT)                                               \
        do {                                                                      \
            if (w->sup) {                                                         \
                wmc_eval_array(w->sup, w->assembly, !w->z_family,                 \
                               w->h, w->nu, w->scale, w->h, (OUT), w->n);         \
                (ST) = MATERN_OK;                                                 \
            } else {                                                              \
                matern_workspace *ws_ = NULL;                                     \
                if (w->ws_needed && matern_workspace_create(&ws_, 0) != MATERN_OK) { \
                    (ST) = MATERN_OUT_OF_MEMORY; break;                           \
                }                                                                 \
                (ST) = w->z_family                                                \
                    ? w->evaluate_z((OUT), w->h, w->nu, w->n, ws_)                \
                    : w->evaluate  ((OUT), w->h, w->nu, w->scale, w->n, ws_);     \
                if (ws_) matern_workspace_destroy(ws_);                           \
            }                                                                     \
        } while (0)

    /* Untimed warm-up, WM_WARMUP_REPS passes as in every other stage. */
    for (int k = 0; k < WM_WARMUP_REPS; k++) {
        int warm = MATERN_OK;
        WM_THREAD_CALL(warm, w->out);
        if (warm != MATERN_OK) { w->status = warm; break; }
    }

    w->ran_on = sched_getcpu();

    w->elapsed_best = 1e300;
    for (int r = 0; r < w->reps; r++) {
        /* All threads start each repetition together. */
        pthread_barrier_wait(w->barrier);

        /* The timed pass, whose output the re-entrancy check compares. */
        int status = MATERN_OK;
        const double start = wm_now_s();
        WM_THREAD_CALL(status, w->out);
        const double elapsed = wm_now_s() - start;

        if (status != MATERN_OK) { w->status = status; break; }
        if (w->elapsed_rep) w->elapsed_rep[r] = elapsed;
        if (elapsed < w->elapsed_best) w->elapsed_best = elapsed;
    }
    #undef WM_THREAD_CALL
    return NULL;
}

static int parse_int_list(const char *text, int *out, int max)
{
    int count = 0;
    const char *p = text;
    while (*p && count < max) {
        char *end;
        const long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) out[count++] = (int)v;
        p = (*end == ',') ? end + 1 : end;
        if (*end != ',') break;
    }
    return count;
}

static void usage(FILE *o, const char *program)
{
    fprintf(o, "usage: %s [--family z_nu|h_nu_s] [--law NAME] [--backend NAME]\n"
           "          [--supplier NAME] [--assembly direct|scaled|logtrick]\n"
           "          [--threads 1,2,4,...] [--max-threads N] [--reps N] [--n POINTS_PER_THREAD]\n"
           "          --max-threads caps the thread counts: the default ladder stops at N\n"
           "            (and ends with N itself), an explicit --threads list drops larger ones\n"
           "          [--data-dir data] [--format csv|text] [--no-pin] [--list]\n"
           "          [--pin-mode dense|spread]  dense fills one socket first;\n"
           "            spread deals threads round-robin across the NUMA nodes, which\n"
           "            halves the working set resident per last-level cache -- the\n"
           "            pair discriminates a cache-capacity loss from a clock cap\n"
           "  WEAK scaling: each thread gets its own --n points, so the total work\n"
           "  grows with the thread count and the wall clock should stay FLAT.\n"
           "  speedup = ns/elem at T=1 over ns/elem at T, so perfect scaling reads T.00x.\n", program);
}

/* The parallel region at one thread count; returns 0 on success.  `wall` is the slowest
 * worker's best repetition, since the region ends when the last thread finishes. */
static int run_at(int threads, const cpu_topology *topo, int pin,
                  const wm_dataset *set, size_t n, int reps, int z_family,
                  evaluate_fn ev, evaluate_z_fn ev_z,
                  const wmc_supplier *sup, int assembly,
                  double *wall, wm_time *tm, double *rep_ns,
                  int *identical, int *status_out, int *cores_used,
                  const int *acc_bits, wm_metrics *M_out,
                  double *mhz_out)
{
    pthread_t   tid[MAX_THREADS];
    worker      work[MAX_THREADS];
    double     *out = (double *)malloc((size_t)threads * n * sizeof *out);
    if (!out) return -1;
    /* Every repetition of every thread, for the timing distribution. */
    double     *rep_s = (double *)calloc((size_t)threads * (size_t)reps, sizeof *rep_s);
    if (!rep_s) { free(out); return -1; }

    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, (unsigned)threads) != 0) { free(rep_s); free(out); return -1; }

    int distinct[MAX_THREADS]; int ndistinct = 0;

    for (int t = 0; t < threads; t++) {
        memset(&work[t], 0, sizeof work[t]);
        work[t].h        = z_family ? set->argument : set->h;
        work[t].nu       = set->order;
        work[t].scale    = set->length_scale;
        work[t].out      = out + (size_t)t * n;
        work[t].n        = n;
        work[t].reps     = reps;
        work[t].z_family = z_family;
        work[t].evaluate   = ev;
        work[t].evaluate_z = ev_z;
        work[t].sup        = sup;
        work[t].assembly   = assembly;
        /* Same rule as benchmark_cpu's backend_builds_workspace; the backend is
         * resolved and verified in main() before any thread exists. */
        work[t].ws_needed  = !sup && strncmp(matern_backend_name(), "scalar", 6) != 0;
        work[t].pin_to   = pin ? topo->cpu[t] : -1;
        work[t].ran_on   = -1;
        work[t].status   = MATERN_OK;
        work[t].barrier  = &barrier;
        work[t].elapsed_rep = rep_s + (size_t)t * (size_t)reps;
    }

    for (int t = 0; t < threads; t++)
        if (pthread_create(&tid[t], NULL, run_worker, &work[t]) != 0) {
            /* Fatal: the others would wait on a barrier sized for this thread. */
            for (int k = 0; k < t; k++) pthread_join(tid[k], NULL);
            pthread_barrier_destroy(&barrier); free(rep_s); free(out); return -1;
        }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);

    *wall = 0.0; *status_out = MATERN_OK;
    for (int t = 0; t < threads; t++) {
        if (work[t].status != MATERN_OK) *status_out = work[t].status;
        if (work[t].elapsed_best > *wall)  *wall = work[t].elapsed_best;
        int seen = 0;
        for (int k = 0; k < ndistinct; k++) if (distinct[k] == work[t].ran_on) { seen = 1; break; }
        if (!seen && work[t].ran_on >= 0) distinct[ndistinct++] = work[t].ran_on;
    }
    *cores_used = ndistinct;
    /* The region time of repetition r is the slowest thread's time in r (all start r
     * at the barrier), per element of the region; `rep_ns` keeps repetition order. */
    memset(tm, 0, sizeof *tm);
    if (*status_out == MATERN_OK && reps > 0) {
        double *slow = (double *)malloc((size_t)reps * sizeof *slow);
        if (slow) {
            const double per = 1e9 / ((double)threads * (double)n);
            for (int r = 0; r < reps; r++) {
                double m = 0.0;
                for (int t = 0; t < threads; t++)
                    if (rep_s[(size_t)t * reps + r] > m)
                        m = rep_s[(size_t)t * reps + r];
                slow[r] = m * per;
                if (rep_ns) rep_ns[r] = slow[r];
            }
            *tm = wm_time_reduce(slow, reps);
            free(slow);
        }
    }
    /* The clock of the pinned CPUs, sampled after the timed region. */
    if (mhz_out) {
        int cpus[MAX_THREADS];
        for (int i = 0; i < threads && i < MAX_THREADS; i++)
            cpus[i] = pin ? topo->cpu[i] : i;
        *mhz_out = mean_mhz(cpus, threads < MAX_THREADS ? threads : MAX_THREADS);
    }

    /* Re-entrancy: same inputs, so every slice must be bit-identical to thread 0's. */
    *identical = 1;
    for (int t = 1; t < threads && *identical; t++)
        if (memcmp(out, out + (size_t)t * n, n * sizeof *out) != 0) *identical = 0;

    /* Accuracy of thread 0's slice (the first n points of the law, acc_points), scored
     * with wm_metrics.h; not scored when a thread failed. */
    if (M_out) {
        memset(M_out, 0, sizeof *M_out);
        if (*status_out == MATERN_OK) {
            wm_metrics_compute(M_out, out, set->reference_high, set->reference_low,
                               acc_bits, set->order,
                               set->has_length_scale ? set->h : set->argument,
                               set->has_length_scale ? set->length_scale : NULL,
                               n, WM_ULP_MODE_DEFAULT);
        } else {
            M_out->n = n;
            M_out->u_max = NAN;
        }
    }

    pthread_barrier_destroy(&barrier);
    free(rep_s);
    free(out);
    return 0;
}

int main(int argc, char **argv)
{
    const char *family   = "h_nu_s";
    const char *law      = "size_random";
    const char *backend  = NULL;
    const char *supplier = NULL;         /* NULL = time this library          */
    const char *assembly_name = "direct";
    const char *root     = "data";
    const char *format   = "csv";
    const char *tlist    = NULL;
    int max_threads      = 0;          /* --max-threads; 0 = no cap */
    int   reps = WM_REPS_MAX_DEFAULT, pin = 1;   /* the pipeline's default count */
    const char *pin_mode = "dense";       /* dense | spread -- see spread_topology */
    size_t n   = 262144;   /* = run_benchmark.sh THREADS_N */

    for (int a = 1; a < argc; a++) {
        const int nx = (a + 1 < argc);
        if      (!strcmp(argv[a],"--family")   && nx) family  = argv[++a];
        else if (!strcmp(argv[a],"--law")      && nx) law     = argv[++a];
        else if (!strcmp(argv[a],"--backend")  && nx) backend = argv[++a];
        else if (!strcmp(argv[a],"--supplier") && nx) supplier = argv[++a];
        else if (!strcmp(argv[a],"--assembly") && nx) assembly_name = argv[++a];
        else if (!strcmp(argv[a],"--list")) {
            wmc_entry e[16]; const int ne = wmc_build_list(e, 16);
            printf("%d supplier(s) this build can thread:\n", ne);
            for (int i = 0; i < ne; i++)
                printf("  %-22s scaled=%s\n", e[i].label,
                       wmc_has_scaled(e[i].s) ? "yes" : "no");
            wmc_print_absent();
            return 0;
        }
        else if (!strcmp(argv[a],"--threads")  && nx) tlist   = argv[++a];
        else if (!strcmp(argv[a],"--max-threads") && nx) max_threads = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--reps")     && nx) reps    = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--n")        && nx) n       = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--data-dir") && nx) root    = argv[++a];
        else if (!strcmp(argv[a],"--format")   && nx) format  = argv[++a];
        else if (!strcmp(argv[a],"--no-pin"))         pin     = 0;
        else if (!strcmp(argv[a],"--pin-mode") && nx)  pin_mode = argv[++a];
        else if (!strcmp(argv[a],"--help")) { usage(stdout, argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); usage(stderr, argv[0]); return 2; }
    }
    if (strchr(family, ',')) { fprintf(stderr, "--family takes one family\n"); return 2; }
    if (corpus_family_check(family)) return 2;

    const int z_family = !strcmp(family, "z_nu");
    /* Per-repetition samples; run_benchmark.sh gives each invocation its own path. */
    wm_raw_open(getenv("WM_RAW_REPS"));

    /* An unknown supplier name is refused; the library does not stand in for it. */
    wmc_entry sups[16]; const int nsup = wmc_build_list(sups, 16);
    const wmc_supplier *sup = NULL; const char *sup_label = "-";
    int assembly = WMC_ASM_DIRECT;
    if (supplier) {
        const wmc_entry *found = wmc_find(sups, nsup, supplier);
        if (!found) {
            fprintf(stderr, "threads_cpu: no supplier '%s' in this build."
                            "  --list shows the %d there are.\n", supplier, nsup);
            return 3;
        }
        sup = found->s; sup_label = found->label;
        if      (!strcmp(assembly_name, "direct"))   assembly = WMC_ASM_DIRECT;
        else if (!strcmp(assembly_name, "scaled"))   assembly = WMC_ASM_SCALED;
        else if (!strcmp(assembly_name, "logtrick")) assembly = WMC_ASM_LOGTRICK;
        else { fprintf(stderr, "threads_cpu: --assembly must be direct, scaled "
                               "or logtrick (got '%s')\n", assembly_name); return 3; }
        /* A supplier without a scaled K_nu has no scaled row. */
        if (assembly == WMC_ASM_SCALED && !wmc_has_scaled(sup)) {
            fprintf(stderr, "threads_cpu: %s publishes no scaled K_nu; there is "
                            "no scaled row for it.\n", sup_label);
            return 3;
        }
    }

    /* Dispatch resolves once per process and falls back for an unknown name, so the live
     * backend is checked here, before any thread exists. */
    if (sup) {
        /* For a supplier row the backend column holds the assembly. */
        backend = wmc_asm_name(assembly);
    } else if (backend) {
        setenv("MATERN_BACKEND", backend, 1);
        double probe_o, probe_z = 1.0, probe_nu = 1.5;
        matern_covariance_z(&probe_o, &probe_z, &probe_nu, 1, NULL);
        const char *live = matern_backend_name();
        if (strcmp(live, backend) != 0) {
            fprintf(stderr,
                "threads_cpu: asked for backend '%s' but '%s' is live.\n"
                "  Either the name is wrong, or this CPU cannot run it.\n"
                "  Refusing to label %s's numbers as '%s'.\n", backend, live, live, backend);
            return 3;
        }
    } else {
        double probe_o, probe_z = 1.0, probe_nu = 1.5;
        matern_covariance_z(&probe_o, &probe_z, &probe_nu, 1, NULL);
        backend = matern_backend_name();
    }

    corpus_dataset cs;
    /* only the first n points are evaluated, so only they are read */
    if (corpus_dataset_load_prefix(&cs, root, family, law, n) != 0) {
        fprintf(stderr, "threads_cpu: could not load %s/%s from %s\n", family, law, root);
        return 1;
    }
    if (n > cs.set.count) n = cs.set.count;

    cpu_topology topo = discover_topology();
    if (topo.count < 1) { fprintf(stderr, "threads_cpu: empty affinity mask\n"); return 1; }
    if (!strcmp(pin_mode, "spread")) spread_topology(&topo);
    else if (strcmp(pin_mode, "dense")) {
        fprintf(stderr, "threads_cpu: --pin-mode must be dense or spread\n");
        return 2;
    }
    /* the last-level cache of the first pinned CPU, read once */
    int  llc_share = 1;
    long llc_bytes = wm_llc_bytes_shared(topo.cpu[0], &llc_share);

    int steps[MAX_THREAD_STEPS]; int nsteps = 0;
    if (tlist) nsteps = parse_int_list(tlist, steps, MAX_THREAD_STEPS);
    if (nsteps == 0) {
        /* Default: powers of two up to the physical core count (or --max-threads),
         * then that count itself. */
        int top = topo.physical_cores;
        if (max_threads > 0 && max_threads < top) top = max_threads;
        for (int t = 1; t <= top && nsteps < MAX_THREAD_STEPS; t *= 2)
            steps[nsteps++] = t;
        if (nsteps > 0 && steps[nsteps-1] != top && nsteps < MAX_THREAD_STEPS)
            steps[nsteps++] = top;
    } else if (max_threads > 0) {
        /* An explicit list keeps its counts up to the cap. */
        int k = 0;
        for (int i = 0; i < nsteps; i++) if (steps[i] <= max_threads) steps[k++] = steps[i];
        nsteps = k;
        if (nsteps == 0) {
            fprintf(stderr, "threads_cpu: --max-threads %d leaves no count of --threads %s\n",
                    max_threads, tlist);
            return 2;
        }
    }

    evaluate_fn   ev   = matern_covariance;
    evaluate_z_fn ev_z = matern_covariance_z;

    /* The load average is recorded: load biases the high thread counts. */
    double load1 = 0.0;
    { FILE *f = fopen("/proc/loadavg", "r");
      if (f) { if (fscanf(f, "%lf", &load1) != 1) load1 = 0.0; fclose(f); } }

    const int csv = !strcmp(format, "csv");
    if (csv) {
        /* Columns: docs/benchmark.md, "Output".  Must match the row printf below. */
        printf("schema,device,supplier,family,law,backend,threads,pinned,cores_used,"
               "physical_cores,siblings_per_core,n_per_thread,n_total,reps,wall_s,"
               "ns_per_elem,speedup,efficiency,identical,load1,status,"
               "values_per_s,u_max,u_scored,"
               "pin_mode,mhz_end,llc_bytes,llc_shared_cpus,ws_bytes,"
               "ns_min,ns_q1,ns_med,ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,"
               "ns_spread,"
               "ulp_mode,acc_points,n_excluded,nonfinite,ref_zero,ref_subnormal,"
               "exact,min_bits,u_min,u_q1,u_med,u_q3,u_p99,u_p999,u_mean,u_iqr,"
               "u_fence,beyond_fence,u_med_nontrivial,rel_max,abs_max,worst_nu,"
               "worst_arg\n");
    } else {
        printf("threads_cpu: %s %s, %s %s, %zu points/thread, %d reps, pin=%s\n",
               family, law, sup ? sup_label : "backend", backend, n, reps,
               pin ? "yes" : "no");
        printf("  %-8s %-10s %-11s %-11s %-13s %-9s %-11s %s\n",
               "threads", "wall_s", "ns/elem", "med ns/el", "values/s",
               "speedup", "efficiency", "identical");
    }

    double base_ns = 0.0;
    int rc = 0;

    for (int i = 0; i < nsteps; i++) {
        const int t = steps[i];
        if (t > topo.count) continue;          /* more threads than usable CPUs */

        double wall = 0.0;
        wm_time tm;
        double *rep_ns = (double *)calloc((size_t)(reps > 0 ? reps : 1), sizeof *rep_ns);
        if (!rep_ns) { rc = 1; continue; }
        int identical = 0, status = 0, cores = 0;
        wm_metrics M; double mhz = 0.0;
        if (run_at(t, &topo, pin, &cs.set, n, reps, z_family, ev, ev_z,
                   sup, assembly,
                   &wall, &tm, rep_ns, &identical, &status, &cores,
                   cs.accuracy_bits, &M, &mhz) != 0) {
            fprintf(stderr, "threads_cpu: run at %d threads failed to start\n", t);
            free(rep_ns);
            rc = 1; continue;
        }

        const double total  = (double)t * (double)n;
        const double ns_pe  = wall * 1e9 / total;
        /* Raw samples, keyed by the whole region's point count. */
        if (tm.reps > 0)
            wm_raw_emit("cpu", sup_label, family, law, backend, (size_t)total,
                        rep_ns, tm.reps);
        free(rep_ns);
        const double vps    = wall > 0.0 ? total / wall : 0.0;
        /* base_ns: this invocation's first thread count; run_benchmark.sh starts one
         * process per (family, backend or supplier). */
        if (i == 0 || base_ns == 0.0) base_ns = ns_pe;
        const double speedup = base_ns > 0.0 ? base_ns / ns_pe : 0.0;

        if (csv)
            printf("%d,cpu,%s,%s,%s,%s,%d,%s,%d,%d,%d,%zu,%.0f,%d,%.9g,%.6f,%.4f,%.4f,%s,%.2f,%d,"
                   "%.6g,%.17g,%zu,%s,%.1f,%ld,%d,%.0f,"
                   "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "up,%zu,%zu,%zu,%zu,%zu,%zu,%d,"
                   "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,%.17g,"
                   "%.17g,%.17g,%.17g,%.17g\n",
                   WM_SCHEMA, sup_label, family, law, backend, t, pin ? "yes" : "no", cores,
                   topo.physical_cores, topo.siblings_per_core, n, total, reps,
                   wall, ns_pe, speedup, speedup / (double)t,
                   /* At T=1 there is nothing to compare, so `identical` is n/a. */
                   t > 1 ? (identical ? "yes" : "no") : "n/a", load1, status,
                   vps, M.u_max, M.n_scored,
                   pin_mode, mhz, llc_bytes, llc_share,
                   (double)total * (double)sizeof(double) * (z_family ? 3.0 : 4.0),
                   tm.min, tm.q1, tm.med, tm.q3, tm.p95, tm.max, tm.mean, tm.iqr,
                   tm.spread,
                   M.n, M.n_excluded, M.n_nonfinite, M.n_ref_zero, M.n_ref_subnormal,
                   M.n_exact, M.min_bits,
                   M.u_min, M.u_q1, M.u_med, M.u_q3, M.u_p99, M.u_p999, M.u_mean,
                   M.u_iqr, M.u_fence_hi, M.n_beyond_fence, M.u_med_nontrivial,
                   M.rel_max, M.abs_max,
                   M.n_worst ? M.worst[0].nu : NAN, M.n_worst ? M.worst[0].arg : NAN);
        else
            printf("  %-8d %-10.6f %-11.4f %-11.4f %-13.4g %-9.3f %-11.3f %s%s\n",
                   t, wall, ns_pe, tm.med, vps, speedup, speedup / (double)t,
                   t > 1 ? (identical ? "yes" : "NO -- RE-ENTRANCY FAILURE") : "n/a",
                   status == MATERN_OK ? "" : " (status != OK)");

        if (!identical) rc = 4;                /* re-entrancy failure */
    }

    wm_raw_close();
    corpus_dataset_free(&cs);
    return rc;
}
