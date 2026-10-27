/* detect.c -- stage 1: machine, toolchain, suppliers and corpus; fails only on a corpus that
 * cannot be read.  --fpenv, --build-info, --ulp and --timer print `key value` lines for
 * PROVENANCE.txt; --scalar-isa and --live-backend report the backend the library selects. */
#include "common/numa_bind.h"      /* first: sets _GNU_SOURCE */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#include "corpus_dataset.h"
#include "matern.h"
#include "common/ulpmodes.h"
#include "common/bench_timing.h"
#include "../references/common/build_info.h"

static void line(void) { puts("--------------------------------------------------------------------------"); }

static int read_first_line(const char *path, char *buf, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int ok = fgets(buf, (int)n, f) != NULL;
    fclose(f);
    if (ok) buf[strcspn(buf, "\n")] = '\0';
    return ok ? 0 : -1;
}

static int cpuinfo_field(const char *key, char *out, size_t n)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return -1;
    char b[512]; int found = -1;
    const size_t kl = strlen(key);
    while (fgets(b, sizeof b, f)) {
        if (strncmp(b, key, kl) == 0) {
            char *c = strchr(b, ':');
            if (c) { c++; while (*c == ' ' || *c == '\t') c++;
                     c[strcspn(c, "\n")] = '\0';
                     snprintf(out, n, "%s", c); found = 0; }
            break;
        }
    }
    fclose(f);
    return found;
}

/* Whether `a*b + c` is contracted into an FMA.  GCC's default follows the -std dialect: a
 * -std=c11 build without -ffp-contract=fast does not fuse.  The volatile input keeps the
 * probe from being evaluated at compile time. */
static volatile double bench_fma_probe_in = 1.0 + 0x1p-27;

static int fma_is_contracted(void)
{
    /* The arithmetic below is not volatile, so the build's contraction rule applies. */
    double a = bench_fma_probe_in;
    double r = a * a - 1.0 - 0x1p-26;
    /* unfused: a*a rounds to 1 + 2^-26, so r == 0.
     * fused:   fma(a, a, -1.0) is exact (2^-26 + 2^-54), so r == 2^-54. */
    return r != 0.0;
}

/* One ISA row: compiled for, and executable on this CPU. */
static void isa_row(const char *name, int compiled, int runtime, int *disagree)
{
    const char *verdict;
    if (compiled && runtime)        verdict = "compiled and executable";
    else if (!compiled && runtime)  { verdict = "CPU HAS IT, this build does not use it"; *disagree = 1; }
    else if (compiled && !runtime)  verdict = "compiled but NOT executable here";
    else                            verdict = "not available";
    printf("  %-10s %-8s %s\n", name, runtime ? "yes" : "no", verdict);
}

/* Boost version, from the environment and the standard include paths. */
static int boost_version_at(const char *inc)
{
    char path[600], b[512];
    snprintf(path, sizeof path, "%s/boost/version.hpp", inc);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int v = 0;
    while (fgets(b, sizeof b, f))
        if (sscanf(b, "#define BOOST_VERSION %d", &v) == 1) break;
    fclose(f);
    return v;
}

/* Every Boost found, one line each with the rows it can serve: the copies of make externals,
 * BOOST_DIR, BOOST_ROOT, the conda prefix and the system include directories.  The build
 * selects among them (benchmark/tools/boost_detect.sh); the "built with" list says which. */
static int list_boost(void)
{
    char conda[512] = "";
    if (getenv("CONDA_PREFIX")) snprintf(conda, sizeof conda, "%s/include", getenv("CONDA_PREFIX"));
    /* Sized, not NULL-terminated: an unset variable yields a NULL entry. */
    const char *where[] = { "references/boost/1.75.0", "references/boost/1.87.0",
                            getenv("BOOST_DIR"), getenv("BOOST_ROOT"), conda[0] ? conda : NULL,
                            "/usr/include", "/usr/local/include",
                            "/opt/homebrew/include", "/opt/local/include" };
    const size_t n_where = sizeof where / sizeof where[0];
    int found = 0;
    for (size_t i = 0; i < n_where; i++) {
        if (!where[i]) continue;
        int dup = 0;
        for (size_t j = 0; j < i; j++) dup |= where[j] && strcmp(where[j], where[i]) == 0;
        int v = dup ? 0 : boost_version_at(where[i]);
        if (!v) continue;
        printf("  %-10s %d.%d.%d at %s -- %s\n", found ? "" : "boost",
               v / 100000, v / 100 % 1000, v % 100, where[i],
               v == 107500 ? "the CPU row's version"
               : v >= 108700 ? "a GPU row version (1.87 or later)"
                             : "neither row (the CPU row is 1.75.0, the GPU row 1.87 or later)");
        found++;
    }
    return found;
}

/* GSL version from the header the row compiles against, not from libgsl.so, which can
 * differ from it. */
static int gsl_version_at(const char *inc, char *ver, size_t vn)
{
    char path[512]; snprintf(path, sizeof path, "%s/gsl/gsl_version.h", inc);
    FILE *f = fopen(path, "r"); if (!f) return 0;
    char b[512]; int got = 0;
    while (fgets(b, sizeof b, f))
        if (sscanf(b, "#define GSL_VERSION \"%31[^\"]\"", ver) == 1) { got = 1; break; }
    fclose(f); (void)vn; return got;
}

static int find_gsl(char *out, size_t n, char *ver, size_t vn)
{
    /* the copy of make externals first: the build prefers it (benchmark/tools/gsl_flags.sh) */
    const char *fixed[] = { "references/gsl/2.6/include", "/usr/include", "/usr/local/include",
                            "/opt/homebrew/include", "/opt/local/include", NULL };
    const char *conda = getenv("CONDA_PREFIX");
    if (conda && !gsl_version_at(fixed[0], ver, vn)) {
        char p[512]; snprintf(p, sizeof p, "%s/include", conda);
        if (gsl_version_at(p, ver, vn)) { snprintf(out, n, "%s", p); return 1; }
    }
    for (int i = 0; fixed[i]; i++)
        if (gsl_version_at(fixed[i], ver, vn)) { snprintf(out, n, "%s", fixed[i]); return 1; }
    return 0;
}

static int file_exists(const char *p) { struct stat st; return stat(p, &st) == 0; }

static int corpus_stage(const char *root)
{
    const char *fams[2] = { "z_nu", "h_nu_s" };
    int total = 0, bad = 0;
    for (int f = 0; f < 2; f++) {
        char names[40][40];
        int c = corpus_dataset_list(root, fams[f], names, 40);
        if (c == 0) { printf("  %-10s MISSING -- no datasets found under %s/%s\n",
                             fams[f], root, fams[f]); bad++; continue; }
        printf("  %-10s %2d datasets:", fams[f], c);
        for (int i = 0; i < c && i < 40; i++) printf(" %s", names[i]);
        printf("\n");
        total += c;
    }
    return bad ? -1 : total;
}

/* The floating-point environment.  Every ULP figure assumes round-to-nearest-even and honoured
 * subnormals (FTZ/DAZ removes the subnormal range, where the extended-range assemblies work).
 * The probes use volatile operands. */
static void fp_environment(void)
{
    int r = fegetround();
    printf("fp_rounding        %s\n",
           r == FE_TONEAREST  ? "to-nearest-even (the assumed mode)" :
           r == FE_UPWARD     ? "UPWARD -- every result differs from the reference" :
           r == FE_DOWNWARD   ? "DOWNWARD -- every result differs from the reference" :
           r == FE_TOWARDZERO ? "TOWARD-ZERO -- every result differs from the reference" :
                                "unknown mode");

    /* Subnormals: with flush-to-zero the product of a subnormal and 0.5 is 0. */
    volatile double tiny = 0x1p-1060;      /* a subnormal double */
    volatile double half = 0.5;
    volatile double prod = tiny * half;
    printf("fp_subnormals      %s\n", prod == 0.0
           ? "FLUSHED TO ZERO (FTZ/DAZ active) -- the subnormal range is not measurable here"
           : "honoured");

    printf("fp_iec559          %s\n",
#ifdef __STDC_IEC_559__
           "yes (__STDC_IEC_559__: IEEE 754 semantics promised)"
#else
           "NOT promised (__STDC_IEC_559__ undefined)"
#endif
    );
    printf("fp_fast_math       %s\n",
#ifdef __FAST_MATH__
           "ON -- -ffast-math was used; accuracy claims do not hold"
#else
           "off"
#endif
    );
    printf("fp_eval_method     %d%s\n", (int)FLT_EVAL_METHOD,
           FLT_EVAL_METHOD == 0 ? " (each type evaluated in its own precision)" :
           FLT_EVAL_METHOD == 1 ? " (float evaluated as double)" :
           FLT_EVAL_METHOD == 2 ? " (all evaluated as long double -- x87; excess precision)" :
                                  " (implementation-defined)");
    printf("fp_long_double     %zu bytes, %d mantissa bits\n",
           sizeof(long double), (int)LDBL_MANT_DIG);
    printf("fp_double_digits   %d mantissa bits, radix %d\n", (int)DBL_MANT_DIG, (int)FLT_RADIX);
    /* Contraction and FP_FAST_FMA are not reported: they describe this program's translation
     * unit, not the library, whose fused instructions PROVENANCE.txt counts (fma_insns_*). */
}

/* The scalar build MATERN_BACKEND=scalar runs (matern_scalar_isa(), src/backends/internal.h).
 * The backend is chosen on the first call, so this runs before any other library use. */
const char *matern_scalar_isa(void);

static const char *scalar_isa_selected(void)
{
    setenv("MATERN_BACKEND", "scalar", 1);
    double o, z = 1.0, nu = 1.5;
    matern_covariance_z(&o, &z, &nu, 1, NULL);
    const char *isa = matern_scalar_isa();
    unsetenv("MATERN_BACKEND");
    return isa;
}

/* `detect --build-info [builddir]`: `build_<artefact>_<key>  <value>` lines read back from
 * each artefact: the `.comment` compiler stamps of libmatern.a (one `ld -r` object keeping
 * every input's stamp) and cuda.o, and each driver's own `--build-info`. */

/* The distinct `.comment` stamps of an object or archive, in the order of
 * their first appearance, with the number of archive members carrying each. */
typedef struct { char stamp[16][160]; int count[16]; int ns, members; } stamp_set;

/* Reads the stamps of `path`; returns 0 when readelf could not be run.  A stamp is
 * counted once per archive member, since an `ld -r` object carries one per input. */
static int read_stamps(const char *path, stamp_set *st)
{
    memset(st, 0, sizeof *st);
    char cmd[PATH_MAX + 64];
    snprintf(cmd, sizeof cmd, "readelf -p .comment '%s' 2>/dev/null", path);
    FILE *p = popen(cmd, "r");
    if (!p) return 0;
    int seen[16] = {0};
    char buf[512];
    while (fgets(buf, sizeof buf, p)) {
        if (strstr(buf, "File: ")) { st->members++; memset(seen, 0, sizeof seen); continue; }
        char *b = strchr(buf, ']');
        if (!b) continue;
        b++; while (*b == ' ') b++;
        b[strcspn(b, "\n")] = '\0';
        if (!*b) continue;
        int k = 0;
        for (; k < st->ns; k++) if (!strcmp(st->stamp[k], b)) break;
        if (k == st->ns && st->ns < 16) {
            snprintf(st->stamp[st->ns], sizeof st->stamp[st->ns], "%s", b);
            st->ns++;
        }
        if (k < 16 && !seen[k]) { st->count[k]++; seen[k] = 1; }
    }
    pclose(p);
    /* readelf prints a `File:` header per archive member, none for a lone object. */
    if (!st->members) st->members = 1;
    return 1;
}

/* The two sets hold the same stamps, in any order. */
static int same_stamps(const stamp_set *a, const stamp_set *b)
{
    if (a->ns != b->ns) return 0;
    for (int i = 0; i < a->ns; i++) {
        int k = 0;
        for (; k < b->ns; k++) if (!strcmp(a->stamp[i], b->stamp[k])) break;
        if (k == b->ns) return 0;
    }
    return 1;
}

/* Prints the stamps of `path`.  `count_path`, when not NULL, names an archive
 * of the same objects as separate members: if it exists and carries the same
 * stamps, the counts are taken from it, so that they count objects. */
static void comment_stamps(const char *path, const char *count_path,
                           const char *side, const char *unit)
{
    char key[64];
    snprintf(key, sizeof key, "%s_compiler", unit);
    if (!file_exists(path)) {
        bench_build_info_kv(stdout, side, key, "not built");
        return;
    }
    static stamp_set st, per;
    if (!read_stamps(path, &st)) {
        bench_build_info_kv(stdout, side, key, "unknown (readelf could not be run)");
        return;
    }
    if (!st.ns) {
        bench_build_info_kv(stdout, side, key,
                         "unknown (no .comment stamp: not GCC or Clang, stripped, or no readelf)");
        return;
    }
    const stamp_set *shown = &st;
    if (count_path && file_exists(count_path) && read_stamps(count_path, &per)
        && same_stamps(&st, &per))
        shown = &per;
    char val[320];
    for (int k = 0; k < shown->ns; k++) {
        snprintf(key, sizeof key, shown->ns == 1 ? "%s_compiler" : "%s_compiler_%d", unit, k + 1);
        snprintf(val, sizeof val, "%.159s  [%d of %d object(s)]",
                 shown->stamp[k], shown->count[k], shown->members);
        bench_build_info_kv(stdout, side, key, val);
    }
}

/* Copies `<exe> --build-info` to stdout; returns the number of lines copied. */
static int relay(const char *exe)
{
    char cmd[PATH_MAX + 64];
    snprintf(cmd, sizeof cmd, "'%s' --build-info 2>/dev/null", exe);
    FILE *p = popen(cmd, "r");
    if (!p) return 0;
    char buf[512]; int n = 0;
    while (fgets(buf, sizeof buf, p)) { fputs(buf, stdout); n++; }
    pclose(p);
    return n;
}

static void build_info(const char *bdir)
{
    char path[PATH_MAX], internal[PATH_MAX];
    fflush(stdout);
    snprintf(path, sizeof path, "%s/libmatern.a", bdir);
    snprintf(internal, sizeof internal, "%s/libmatern_internal.a", bdir);
    comment_stamps(path, internal, "cpu", "library");
    bench_build_info_kv(stdout, "cpu", "library_version", matern_version());
    snprintf(path, sizeof path, "%s/cuda.o", bdir);
    /* An nvcc object carries the host compiler's stamp; nvcc's is in the GPU drivers' records. */
    if (file_exists(path)) comment_stamps(path, NULL, "gpu", "library_host");
    static const struct { const char *exe, *unit; } drivers[] = {
        { "benchmark_cpu", "cpu_benchmark" },  { "suppliers",     "cpu_suppliers" },
        { "benchmark_gpu", "gpu_benchmark" },  { "suppliers_gpu", "gpu_suppliers" },
        { "boost_shortcut_gpu", "gpu_boost_shortcut" },
        { "fold_cpu",      "cpu_fold" },
    };
    for (size_t i = 0; i < sizeof drivers / sizeof drivers[0]; i++) {
        snprintf(path, sizeof path, "%s/%s", bdir, drivers[i].exe);
        fflush(stdout);
        if (access(path, X_OK) != 0)
            bench_build_info_kv(stdout, drivers[i].unit, "compiler",
                             "not built -- no rows from this driver");
        else if (relay(path) == 0)
            bench_build_info_kv(stdout, drivers[i].unit, "compiler",
                             "unknown -- the binary printed no --build-info (built before it had one, "
                             "or it cannot start here)");
    }
}

/* The directory of this executable: the build directory the report describes. */
static void self_dir(char *out, size_t n)
{
    char exe[PATH_MAX];
    ssize_t k = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (k > 0) {
        exe[k] = '\0';
        char *slash = strrchr(exe, '/');
        if (slash) { *slash = '\0'; snprintf(out, n, "%s", exe); return; }
    }
    snprintf(out, n, "build");
}

/* Host timings read BENCH_TIMING_CLOCK (bench_timing.h).  With clocksource `tsc` and an invariant
 * TSC a tick has the same length at every frequency; otherwise a warning goes to stderr. */
static int cpuinfo_has_flag(const char *flags, const char *name)
{
    const size_t nl = strlen(name);
    for (const char *p = flags; (p = strstr(p, name)) != NULL; p += nl) {
        const int start = (p == flags) || p[-1] == ' ' || p[-1] == '\t';
        const int end   = p[nl] == '\0' || p[nl] == ' ' || p[nl] == '\n';
        if (start && end) return 1;
    }
    return 0;
}

static int timer_info(const char *indent)
{
    printf("%s%-18s %s via clock_gettime (host timings; device kernel-only timings use CUDA events)\n",
           indent, "timer_clock", BENCH_TIMING_CLOCK_NAME);
    struct timespec res;
    if (clock_getres(BENCH_TIMING_CLOCK, &res) == 0)
        printf("%s%-18s %.0f ns\n", indent, "timer_resolution",
               (double)res.tv_sec * 1e9 + (double)res.tv_nsec);
    else
        printf("%s%-18s unavailable (clock_getres failed)\n", indent, "timer_resolution");

    char cur[128] = "", avail[512] = "";
    int have_cs = read_first_line("/sys/devices/system/clocksource/clocksource0/current_clocksource",
                                  cur, sizeof cur) == 0;
    if (read_first_line("/sys/devices/system/clocksource/clocksource0/available_clocksource",
                        avail, sizeof avail) != 0)
        snprintf(avail, sizeof avail, "unavailable");
    for (size_t k = strlen(avail); k > 0 && avail[k - 1] == ' '; k--) avail[k - 1] = '\0';
    printf("%s%-18s %s\n", indent, "timer_clocksource",
           have_cs ? cur : "unavailable (no clocksource sysfs)");
    printf("%s%-18s %s\n", indent, "timer_clocksources", avail);

    /* Invariant TSC: CPUID 0x80000007 EDX bit 8, or cpuinfo constant_tsc and nonstop_tsc. */
    int cpuid_known = 0, cpuid_inv = 0;
#if defined(__x86_64__) || defined(__i386__)
    {
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (__get_cpuid_max(0x80000000u, NULL) >= 0x80000007u
            && __get_cpuid(0x80000007u, &a, &b, &c, &d)) {
            cpuid_known = 1;
            cpuid_inv = (d >> 8) & 1u;
        }
    }
#endif
    char flags[8192] = "";
    int have_flags = cpuinfo_field("flags", flags, sizeof flags) == 0;
    const int f_const = have_flags && cpuinfo_has_flag(flags, "constant_tsc");
    const int f_nonstop = have_flags && cpuinfo_has_flag(flags, "nonstop_tsc");
    const int invariant = cpuid_inv || (f_const && f_nonstop);
    char cpuid_s[64], flag_s[96];
    snprintf(cpuid_s, sizeof cpuid_s, "%s", !cpuid_known ? "cpuid leaf 0x80000007 unavailable"
             : cpuid_inv ? "cpuid 0x80000007 EDX bit 8 set" : "cpuid 0x80000007 EDX bit 8 clear");
    snprintf(flag_s, sizeof flag_s, "%s", !have_flags ? "no /proc/cpuinfo flags"
             : (f_const && f_nonstop) ? "flags constant_tsc nonstop_tsc"
             : f_const ? "flag constant_tsc only" : f_nonstop ? "flag nonstop_tsc only"
             : "neither constant_tsc nor nonstop_tsc");
    printf("%s%-18s %s (%s; %s)\n", indent, "timer_tsc_invariant",
           invariant ? "yes" : "no", cpuid_s, flag_s);

    const int cs_tsc = have_cs && strcmp(cur, "tsc") == 0;
    if (cs_tsc && invariant) {
        printf("%s%-18s none\n", indent, "timer_warning");
        return 0;
    }
    char why[256];
    if (!cs_tsc && !invariant)
        snprintf(why, sizeof why, "the clocksource is %s, not tsc, and the TSC is not invariant",
                 have_cs ? cur : "unknown");
    else if (!cs_tsc)
        snprintf(why, sizeof why, "the clocksource is %s, not tsc", have_cs ? cur : "unknown");
    else
        snprintf(why, sizeof why, "the TSC is not invariant");
    printf("%s%-18s %s; timings may drift with frequency changes\n", indent, "timer_warning", why);
    fflush(stdout);
    fprintf(stderr,
            "WARNING: timer: %s.  " BENCH_TIMING_CLOCK_NAME " may then not advance at a constant\n"
            "WARNING: rate, and timings may drift with frequency changes (turbo, power states).\n"
            "WARNING: Treat the timings of this run with caution; see timer_* in PROVENANCE.txt.\n",
            why);
    return 1;
}

int main(int argc, char **argv)
{
    /* `--fpenv`: the floating-point environment as `key value` lines. */
    if (argc > 1 && strcmp(argv[1], "--fpenv") == 0) { fp_environment(); return 0; }
    /* `--build-info [builddir]`: the build record of every linked artefact (build_info()). */
    if (argc > 1 && strcmp(argv[1], "--build-info") == 0) {
        char bdir[PATH_MAX];
        if (argc > 2) snprintf(bdir, sizeof bdir, "%s", argv[2]);
        else self_dir(bdir, sizeof bdir);
        build_info(bdir);
        return 0;
    }
    /* `--scalar-isa`: the scalar build MATERN_BACKEND=scalar runs on this CPU. */
    if (argc > 1 && strcmp(argv[1], "--scalar-isa") == 0) {
        puts(scalar_isa_selected());
        return 0;
    }
    /* `--ulp`: the ULP metric the drivers score with, as `key value` lines. */
    if (argc > 1 && strcmp(argv[1], "--ulp") == 0) {
        bench_ulp_mode_describe(stdout, "", BENCH_ULP_MODE_DEFAULT);
        return 0;
    }
    /* `--timer`: the timer clock as `key value` lines. */
    if (argc > 1 && strcmp(argv[1], "--timer") == 0) {
        timer_info("");
        return 0;
    }
    /* `--live-backend <name>`: the backend the library runs when <name> is requested; it
     * differs from <name> when this CPU cannot run it (run_benchmark.sh's default BACKENDS). */
    if (argc > 2 && strcmp(argv[1], "--live-backend") == 0) {
        setenv("MATERN_BACKEND", argv[2], 1);
        double o, z = 1.0, nu = 1.5;
        matern_covariance_z(&o, &z, &nu, 1, NULL);
        puts(matern_backend_name());
        return 0;
    }

    /* Any other option is refused, so that a mistyped one is not read as the corpus root. */
    if (argc > 1 && argv[1][0] == '-') {
        const char *p = argv[0];
        const int help = strcmp(argv[1], "--help") == 0;
        if (!help)
            fprintf(stderr, strcmp(argv[1], "--live-backend") == 0
                            ? "detect: %s needs a backend name\n"
                            : "detect: unknown option %s\n", argv[1]);
        fprintf(help ? stdout : stderr,
                "usage: %s [DATA]\n"
                "         the environment report; DATA is the corpus root (default data)\n"
                "       %s --build-info [BUILD]\n"
                "         the build record of every artefact in BUILD (default: this\n"
                "         binary's directory), as key-value lines\n"
                "       %s --fpenv | --ulp | --timer\n"
                "         the floating-point environment, the ULP metric or the timer\n"
                "         clock, as key-value lines\n"
                "       %s --scalar-isa\n"
                "         the scalar build that MATERN_BACKEND=scalar runs on this CPU\n"
                "       %s --live-backend NAME\n"
                "         the backend the library runs when NAME is requested\n"
                "       %s --help\n"
                "         this text\n", p, p, p, p, p, p);
        return help ? 0 : 2;
    }

    const char *root = (argc > 1) ? argv[1] : "data";
    int warn = 0;

    line(); puts("  STAGE 1 -- ENVIRONMENT"); line();

    /* 0. date: local time with the zone, and UTC, which the run id uses. */
    {
        time_t now = time(NULL);
        struct tm lt, gt;
        char ls[64] = "", gs[64] = "", zone[16] = "", off[8] = "";
        if (localtime_r(&now, &lt)) {
            strftime(ls, sizeof ls, "%d.%m.%Y %H:%M:%S", &lt);
            strftime(zone, sizeof zone, "%Z", &lt);
            strftime(off, sizeof off, "%z", &lt);
        }
        if (gmtime_r(&now, &gt))
            strftime(gs, sizeof gs, "%d.%m.%Y %H:%M:%S", &gt);
        if (*ls)
            printf("  date       %s %s (UTC%s)\n", ls, zone, off);
        if (*gs)
            printf("             %s UTC  -- the run id's clock\n", gs);
    }

    /* 1. machine */
    char buf[512];
    if (cpuinfo_field("model name", buf, sizeof buf) == 0) printf("  cpu        %s\n", buf);
    printf("  cpus       %ld online (hardware threads; cpus_physical in PROVENANCE.txt counts "
           "the cores)\n", sysconf(_SC_NPROCESSORS_ONLN));
    if (read_first_line("/proc/loadavg", buf, sizeof buf) == 0) {
        char *sp = buf; int k = 0; for (char *p = buf; *p; p++) if (*p == ' ' && ++k == 3) { *p = '\0'; break; }
        printf("  load       %s\n", sp);
    }

    /* 2. NUMA topology and the placement obtained */
    puts("");
    int nodes = 0, best = bench_best_node(&nodes);
    if (nodes <= 0) {
        puts("  numa       no /sys topology -- placement policy UNAVAILABLE, running unpinned");
        fflush(stdout);
        bench_bind_warn(0, "no NUMA topology or CPU-affinity API on this platform; running UNPINNED");
        warn++;
    } else {
        printf("  numa       %d node(s)\n", nodes);
        for (int i = 0; i < nodes && i < 8; i++) {
            char p[128], cl[512], di[512];
            snprintf(p, sizeof p, "/sys/devices/system/node/node%d/cpulist", i);
            if (read_first_line(p, cl, sizeof cl) != 0) continue;
            snprintf(p, sizeof p, "/sys/devices/system/node/node%d/distance", i);
            if (read_first_line(p, di, sizeof di) != 0) di[0] = '\0';
            printf("             node%d cpus %-22s distance %s\n", i, cl, di);
        }
        char why[256];
        int st = bench_bind_node(best < 0 ? 0 : best, why, sizeof why);
        printf("  placement  %s\n", why);
        fflush(stdout);
        bench_bind_warn(st, why);
        /* The mask after binding (PROVENANCE's `affinity_mask` is the mask before it). */
        {
            cpu_set_t got;
            if (sched_getaffinity(0, sizeof got, &got) == 0) {
                printf("  affinity_after_bind ");
                int first = 1, run_start = -1, prev = -2, n = 0;
                for (int c = 0; c <= CPU_SETSIZE; c++) {
                    int in = (c < CPU_SETSIZE) && CPU_ISSET(c, &got);
                    if (in) { n++; if (prev + 1 != c) run_start = c; prev = c; }
                    else if (run_start >= 0) {
                        if (run_start == prev) printf("%s%d", first ? "" : ",", run_start);
                        else printf("%s%d-%d", first ? "" : ",", run_start, prev);
                        first = 0; run_start = -1;
                    }
                }
                printf("  (%d cpus)\n", n);
            }
        }
        if (st != 2) warn++;
    }

    /* 3. toolchain: the compiler of this program; the library's is in the "built with" list. */
    puts("");
#if defined(__clang__)
    printf("  compiler   clang %d.%d.%d (this program)\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    printf("  compiler   gcc %d.%d.%d (this program)\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    puts("  compiler   unknown (this program)");
#endif
    /* "Not fused" has two causes: no FMA target, or no contraction flag. */
    int contracted = fma_is_contracted();
#if defined(__FMA__)
    const int target_has_fma = 1;
#else
    const int target_has_fma = 0;
#endif
    if (!target_has_fma) {
        /* This program has no FMA target, so the probe cannot speak for the library; the
         * fused instructions of its objects are counted in PROVENANCE.txt (fma_insns_*). */
    } else if (contracted) {
        puts("  fp-contract FUSED (a*b+c becomes one FMA) -- as required");
    } else {
        puts("  fp-contract *** NOT FUSED, and the target HAS fma ***");
        puts("              The build lost -ffp-contract=fast.  GCC's default follows the");
        puts("              -std dialect: gnu11 fuses, c11 does not.  Worth one ulp on ~2 %");
        puts("              of covariance outputs -- every measured number moves.");
        warn++;
    }

    /* 3b. the timer clock and the ULP metric */
    puts("");
    warn += timer_info("  ");
    puts("");
    /* the definition of the mode is in PROVENANCE.txt (ulp_formula, ulp_spacing, ...) */
    printf("  ulp_mode           %s\n", bench_ulp_mode_name(BENCH_ULP_MODE_DEFAULT));

    /* 4. ISA: compiled-for vs executable */
    puts("");
    int dis = 0;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    int r512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
             && __builtin_cpu_supports("fma");
    int r2   = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    puts("  isa        runtime  status");
    isa_row("avx512", 1, r512, &dis);
    isa_row("avx2",   1, r2,   &dis);
    /* Same columns as isa_row(). */
    printf("  %-10s %-8s %s\n", "scalar", "yes", "always available");
    printf("  scalar_isa %s  (the scalar build MATERN_BACKEND=scalar runs here; "
           "scalar_portable is the x86-64 baseline)\n", scalar_isa_selected());
    if (!r512 && !r2)
        puts("             NOTE: no vector ISA here -- the machines that run the benchmark may\n"
             "             differ.  Run the benchmark where you will quote it.");
#else
    puts("  isa        not x86 -- the vector backends are x86-only; scalar builds anywhere.");
    dis = 1;
#endif
    if (dis) warn++;

    /* 5. suppliers installed; the "built with" list below records what each artefact used. */
    puts("");
    puts("  installed  (what a build could use; \"built with\" below says what each artefact used)");
    if (!list_boost())
        puts("  boost      none found (references/boost/, BOOST_DIR, BOOST_ROOT, the system)");
    char ginc[512] = "", gver[32] = "";
    if (!find_gsl(ginc, sizeof ginc, gver, sizeof gver)) {
        puts("  gsl        none found (references/gsl/2.6, the system include directories)");
    } else {
        printf("  gsl        %s at %s\n", gver, ginc);
        puts("             enables ONE CPU supplier row in all three assemblies");
        puts("             (direct, scaled, logtrick -- it publishes all three natively).");
        puts("             NOT a GPU row: separately compiled, no __device__ path.");
    }
    /* the port, generated only by `make kaust DOWNLOAD=yes`; the hint is `make kaust`,
     * whose notice states the copyright situation first */
    int kaust = file_exists("references/build/kaust/kaust_derived.h");
    printf("  kaust      %s\n", kaust
        ? "CuBesselK ported -- the kaust-128, kaust-40 and kaust-adaptive rows"
        : "CuBesselK not ported (not distributed: all rights reserved) -- no KAUST\n"
          "             rows.  make kaust: the copyright situation and how to obtain it");

    /* 5b. "built with": what every linked artefact used, read from the artefacts */
    puts("");
    puts("  built with (read back from each artefact; this is what PROVENANCE.txt records)");
    {
        char bdir[PATH_MAX];
        self_dir(bdir, sizeof bdir);
        build_info(bdir);
    }

    /* 6. corpus */
    puts("");
    int laws = corpus_stage(root);
    if (laws < 0) { puts("  corpus     UNUSABLE -- cannot proceed"); warn += 100; }

    /* 7. what this supports */
    puts(""); line();
    printf("  %d warning(s).\n", warn);
    puts("  Accuracy figures do not depend on the load.  They are expected to agree");
    puts("  between machines with the same build (compiler, flags, libm and the");
    puts("  supplier libraries), whatever the processor model or vendor.  Speed is a");
    puts("  property of this host and");
    puts("  its load: on a machine shared with other processes, timings can be slower.");
    line();
    return (laws < 0) ? 1 : 0;
}
