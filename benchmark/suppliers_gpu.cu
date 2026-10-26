/* suppliers_gpu.cu -- external device K_nu kernels (KAUST, Boost >= 1.87) beside the
 * library's kernel: kernel-scope and whole-call timings, scored like the CPU rows.
 * KAUST rows use the generated port (licence: references/kaust_logbesselk/PROVENANCE.md). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <cuda_runtime.h>

#include "matern.h"
#include "corpus_dataset.h"
#include "common/bench_metrics.h"

/* The ULP mode of the scoring call and of the `ulp_mode` column. */
#define BENCH_ACCURACY_ULP_MODE BENCH_ULP_MODE_DEFAULT
#include "common/bench_timing.h"
#include "common/bench_cuda_call.h"
#include "common/build_info_cuda.h"

#ifdef GPU_SUPPLIER_WITH_BOOST
#include <boost/math/special_functions/bessel.hpp>
#endif

#ifdef GPU_SUPPLIER_WITH_KAUST
/* KAUST's device entries from kaust_port.c (build/kaust_dev.o, -rdc=true), with the
 * host suppliers' (fix, iv, adapt) settings; gpu_supplier_kaust_logkv gives the native ln K. */
extern __device__ double gpu_supplier_kaust_kv   (double nu, double x, int fix, int iv, int adapt);
extern __device__ double gpu_supplier_kaust_logkv(double nu, double x, int fix, int iv, int adapt);
#endif

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"cuda: %s (%s:%d)\n",cudaGetErrorString(e_),__FILE__,__LINE__); \
    return -1; } }while(0)

/* The assemblies of M from K: 2^(1-nu)/Gamma(nu) * z^nu * K_nu(z), and the same
 * through one exponential, with the expressions of the CPU rows. */
#if defined(GPU_SUPPLIER_WITH_KAUST) || defined(GPU_SUPPLIER_WITH_BOOST)
__device__ static double gpu_supplier_direct(double k, double nu, double z)
{
    if (!(z > 0.0) || !(nu > 0.0)) return (z == 0.0 && nu > 0.0) ? 1.0 : NAN;
    /* The expression of cpu_supplier_direct (pow/tgamma; exp/lgamma differs in accuracy),
     * so that CPU and GPU rows differ by device, not by assembly. */
    return pow(2.0, 1.0 - nu) / tgamma(nu) * pow(z, nu) * k;
}
#endif
/* Boost kernels only: the KAUST kernel forms its exponent from the native ln K. */
#ifdef GPU_SUPPLIER_WITH_BOOST
__device__ static double gpu_supplier_logtrick(double k, double nu, double z)
{
    if (!(z > 0.0) || !(nu > 0.0)) return (z == 0.0 && nu > 0.0) ? 1.0 : NAN;
    if (!(k > 0.0)) return k;          /* 0 or non-finite: pass it through */
    return exp((1.0 - nu) * M_LN2 - lgamma(nu) + nu * log(z) + log(k));
}
#endif

/* The library's kernels through its launcher (build/cuda_bench.o), so that kernel-only
 * timings run the kernels the entries run. */
extern "C" void matern_cuda_launch_external(int direct,
                                            const double *a, const double *b,
                                            const double *c, double *out,
                                            long n, long base,
                                            long blocks, int threads);

enum { ASM_DIRECT = 0, ASM_LOGTRICK = 1, ASM_COUNT = 2 };
static const char *asm_name(int a){ return a == ASM_DIRECT ? "direct" : "logtrick"; }
static const char *row_asm(int which, int a)
{ return which == 2 ? "direct" : asm_name(a); }

/* On h_nu_s, z = sqrt(2 nu) h / s is derived on the device, as in the library,
 * so that the derivation is timed and scored. */
#if defined(GPU_SUPPLIER_WITH_KAUST) || defined(GPU_SUPPLIER_WITH_BOOST)
__device__ static inline double gpu_supplier_z_of(const double *h, const double *nu,
                                         const double *scale, const double *z,
                                         int i)
{
    return scale ? sqrt(2.0 * nu[i]) * h[i] / scale[i] : z[i];
}
#endif

/* Every supplier kernel computes [base, n), `n` one past the last index: the
 * contract of the library's kernels (src/backends/cuda.cu). */
#ifdef GPU_SUPPLIER_WITH_KAUST
/* Every KAUST variant, selected by kaust_port.c's (fix, iv, adapt) triple. */
__global__ void k_kaust(const double *z, const double *nu, double *out,
                        int n, int base, int asm_, const double *h,
                        const double *scale, int fix, int iv, int adapt)
{
    int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double zz = gpu_supplier_z_of(h, nu, scale, z, i);
    if (asm_ == ASM_DIRECT) {
        out[i] = gpu_supplier_direct(gpu_supplier_kaust_kv(nu[i], zz, fix, iv, adapt), nu[i], zz);
    } else {
        /* the native ln K */
        double lk = gpu_supplier_kaust_logkv(nu[i], zz, fix, iv, adapt);
        out[i] = (!(zz > 0.0) || !(nu[i] > 0.0))
                   ? ((zz == 0.0 && nu[i] > 0.0) ? 1.0 : NAN)
                   : exp((1.0 - nu[i]) * M_LN2 - lgamma(nu[i])
                         + nu[i] * log(zz) + lk);
    }
}
#endif

#ifdef GPU_SUPPLIER_WITH_BOOST
#include <boost/version.hpp>          /* BOOST_VERSION, part of the row label */
#endif


#ifdef GPU_SUPPLIER_WITH_BOOST
__global__ void k_boost(const double *z, const double *nu, double *out,
                        int n, int base, int asm_, const double *h,
                        const double *scale)
{
    int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double zz = gpu_supplier_z_of(h, nu, scale, z, i);
    double k = boost::math::cyl_bessel_k(nu[i], zz);
    out[i] = (asm_ == ASM_DIRECT) ? gpu_supplier_direct(k, nu[i], zz)
                                  : gpu_supplier_logtrick(k, nu[i], zz);
}

/* `boost-gpu-<ver>-adapted`: the rule of common/boost_adapted.h, shared with
 * boost_shortcut_gpu.cu. */
#include "boost_adapted.h"

__global__ void k_boost_adapted(const double *z, const double *nu, double *out,
                                int n, int base, int asm_, const double *h,
                                const double *scale)
{
    int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double zz = gpu_supplier_z_of(h, nu, scale, z, i);
    double k = bench_boost_kv_adapted(nu[i], zz);
    out[i] = (asm_ == ASM_DIRECT) ? gpu_supplier_direct(k, nu[i], zz)
                                  : gpu_supplier_logtrick(k, nu[i], zz);
}
#endif

/* Call scope: the external kernel replaces the library's through
 * matern_cuda_set_kernel_override(), so that only the kernel differs between rows. */
extern "C" void matern_cuda_set_kernel_override(void *fn);

enum { TM_SYNC = 0, TM_ADIRECT = 1, TM_ARING = 2, TM_AMAPPED = 3, TM_COUNT = 4 };
static const char *tm_name(int m)
{
    switch (m) {
        case TM_SYNC:    return "sync";
        case TM_ADIRECT: return "async-direct";
        case TM_ARING:   return "async-ring";
        default:         return "async-mapped";
    }
}

/* which: 0 kaust-128, 1 boost, 2 libmatern's own device kernel,
 *        3 kaust-40, 4 kaust-adaptive, 5 boost-adapted. */
typedef struct { const char *label; int which; } gsup;

/* set per dataset: the (h,s) pair for the h_nu_s family, NULL for z_nu */
static const double *g_dh = NULL, *g_dscale = NULL;

static int launch_s(int which, const double *dz, const double *dnu, double *dout,
                    int n, int base, int a, const double *dh, const double *dscale,
                    cudaStream_t stream, int B, int T);

/* which supplier the override should run, and with which assembly; g_ov_err is
 * set when its launch fails, which the library does not see in every mode */
static int g_ov_which = -1, g_ov_asm = 0, g_ov_err = 0;

/* Called once per library launch: over [0, n) in sync and mapped, per chunk
 * [base, n) in async-direct and async-ring (`n` is the end, not the length).  The
 * kernel runs on that geometry, so it reads only inputs its stream has uploaded. */
static void gpu_supplier_override(int direct, const double *a1, const double *b1,
                         const double *c1, double *out, long n, long base,
                         long blocks, int threads, cudaStream_t stream)
{
    int T = threads, B = (int)blocks;
    if (T <= 0 || B <= 0) {   /* no geometry given: cover [base, n) */
        T = 256;
        B = (int)((n - base + T - 1) / T);
    }
    /* the library passes (h,nu,s) when !direct and (z,nu) when direct, the
     * two shapes the external kernels accept */
    if (launch_s(g_ov_which, direct ? a1 : NULL, b1, out, (int)n, (int)base, g_ov_asm,
                 direct ? NULL : a1, direct ? NULL : c1, stream, B, T) != 0
            || cudaPeekAtLastError() != cudaSuccess)
        g_ov_err = 1;
}

static int timed_call(int which, int a, int mode, size_t cn, int has_length_scale,
                      const double *hz, const double *hnu, const double *hh,
                      const double *hr, double *hout)
{
    /* The external kernels index with int; a larger call is refused (row omitted). */
    if (which != 2 && cn > (size_t)INT_MAX) return 1;
    g_ov_which = which; g_ov_asm = a; g_ov_err = 0;
    (void)cudaGetLastError();   /* from a clean error state, so g_ov_err is this call's */
    /* The same two override calls in every timed region, the library's row included. */
    matern_cuda_set_kernel_override(which != 2 ? (void *)gpu_supplier_override : NULL);
    int rc = has_length_scale
        ? matern_covariance_cuda(hout, hh, hnu, hr, cn, mode)
        : matern_covariance_cuda_z(hout, hz, hnu, cn, mode);
    matern_cuda_set_kernel_override(NULL);
    /* The sync and mapped modes clear a launch error before returning, so a
     * kernel that never ran would otherwise be timed as a call. */
    return rc != MATERN_OK || g_ov_err;
}

/* timed_call's arguments, for bench_time_cuda_call. */
typedef struct {
    int which, a, mode, has_length_scale; size_t n;
    const double *hz, *hnu, *hh, *hr; double *hout;
} sg_call;
static int sg_call_once(void *p)
{
    const sg_call *c = (const sg_call *)p;
    return timed_call(c->which, c->a, c->mode, c->n, c->has_length_scale,
                      c->hz, c->hnu, c->hh, c->hr, c->hout);
}

/* The device copies of one dataset for the kernel-only launches: allocated and uploaded by
 * dev_acquire, freed by dev_release, so that none is held while the public call is timed
 * (common/bench_cuda_call.h).  Returns 0, or -1 after printing a CUDA error. */
typedef struct { double *z, *nu, *out, *h, *scale; } dev_set;
static int dev_acquire(dev_set *d, const corpus_dataset *cs, const double *zh, size_t n)
{
    CK(cudaMalloc(&d->z,   n*sizeof(double)));
    CK(cudaMalloc(&d->nu,  n*sizeof(double)));
    CK(cudaMalloc(&d->out, n*sizeof(double)));
    CK(cudaMemcpy(d->z,  zh,            n*sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d->nu, cs->set.order, n*sizeof(double), cudaMemcpyHostToDevice));
    /* h and s: the external kernels derive z in double, the library's kernel in
     * double-double. */
    if (cs->set.has_length_scale) {
        CK(cudaMalloc(&d->h,     n*sizeof(double)));
        CK(cudaMalloc(&d->scale, n*sizeof(double)));
        CK(cudaMemcpy(d->h,     cs->set.h,            n*sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d->scale, cs->set.length_scale, n*sizeof(double), cudaMemcpyHostToDevice));
    }
    g_dh = d->h; g_dscale = d->scale;
    return 0;
}
static void dev_release(dev_set *d)
{
    if (d->h) cudaFree(d->h);
    if (d->scale) cudaFree(d->scale);
    if (d->z) cudaFree(d->z);
    if (d->nu) cudaFree(d->nu);
    if (d->out) cudaFree(d->out);
    memset(d, 0, sizeof *d);
    g_dh = NULL; g_dscale = NULL;
}

static int launch_s(int which, const double *dz, const double *dnu, double *dout,
                    int n, int base, int a, const double *dh, const double *dscale,
                    cudaStream_t stream, int B, int T)
{
    if (which == 2) {  /* libmatern returns M directly; no assembly applies */
        if (dh && dscale)     /* h_nu_s: its own (h,nu,s) path, including the
                               * double-double derivation of z */
            matern_cuda_launch_external(0, dh, dnu, dscale, dout,
                                        (long)n, (long)base, (long)B, T);
        else
            matern_cuda_launch_external(1, dz, dnu, NULL, dout,
                                        (long)n, (long)base, (long)B, T);
        return 0;
    }
#ifdef GPU_SUPPLIER_WITH_KAUST
    /* (fix, iv, adapt): the CPU rows' triples, as kernel arguments */
    if (which == 0) { k_kaust<<<B,T,0,stream>>>(dz,dnu,dout,n,base,a,dh,dscale, 0,128,0); return 0; }
    if (which == 3) { k_kaust<<<B,T,0,stream>>>(dz,dnu,dout,n,base,a,dh,dscale, 0, 40,0); return 0; }
    if (which == 4) { k_kaust<<<B,T,0,stream>>>(dz,dnu,dout,n,base,a,dh,dscale, 1,128,1); return 0; }
#endif
#ifdef GPU_SUPPLIER_WITH_BOOST
    if (which == 1) { k_boost<<<B,T,0,stream>>>(dz, dnu, dout, n, base, a, dh, dscale); return 0; }
    if (which == 5) { k_boost_adapted<<<B,T,0,stream>>>(dz, dnu, dout, n, base, a, dh, dscale); return 0; }
#endif
    (void)dz; (void)dnu; (void)dout; (void)n; (void)base; (void)a; (void)which;
    (void)dh; (void)dscale; (void)stream;
    return -1;
}

/* kernel-scope launch over the whole array, on the default stream with 256
 * threads per block */
static int launch(int which, const double *dz, const double *dnu, double *dout,
                  int n, int a)
{ int T = 256, B = (n + T - 1) / T;
  return launch_s(which, dz, dnu, dout, n, 0, a, g_dh, g_dscale, 0, B, T); }

/* Every corpus law, the size (prefix-ladder) laws included so that they can be named. */
static const char *ALL_LAWS[] = { "grid","random","small_z","near_int","half_int",
                                  "large_z","large_nu","seam",
                                  "size_random","size_half",
                                  "size_random_large","size_half_large" };
static const int N_LAWS = (int)(sizeof ALL_LAWS / sizeof *ALL_LAWS);

/* No comment may stand between adjacent printf string literals in this file:
 * benchmark/tests/run_tests.sh counts header and row fields from them. */
static void csv_header(void)
{
    printf("schema,device,supplier,family,law,backend,ulp_mode,n,n_scored,n_excluded,"
           "nonfinite,ref_zero,ref_subnormal,exact,min_bits,u_min,u_q1,u_med,u_q3,"
           "u_p99,u_p999,u_max,u_mean,u_iqr,u_fence,beyond_fence,u_med_nontrivial,"
           "rel_max,abs_max,worst_nu,worst_arg,ns_min,ns_q1,ns_med,ns_q3,ns_p95,"
           "ns_max,ns_mean,ns_iqr,ns_spread,reps,ns_points,timing_scope,n_file\n");
}

static void csv_row_scope(const char *sup, const char *fam, const char *law,
                          const char *assembly, const bench_metrics *M,
                          const bench_time *T, size_t ns_points, const char *scope)
{
    printf("%d,cuda,%s,%s,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%zu,"
           "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
           "%.17g,%.17g,%.17g,%d,%zu,%s,%zu\n",
           BENCH_SCHEMA, sup, fam, law, assembly, bench_ulp_mode_name(BENCH_ACCURACY_ULP_MODE),
           M->n, M->n_scored, M->n_excluded, M->n_nonfinite,
           M->n_ref_zero, M->n_ref_subnormal, M->n_exact, M->min_bits,
           M->u_min, M->u_q1, M->u_med, M->u_q3, M->u_p99, M->u_p999, M->u_max,
           M->u_mean, M->u_iqr, M->u_fence_hi, M->n_beyond_fence,
           M->u_med_nontrivial, M->rel_max, M->abs_max,
           M->n_worst ? M->worst[0].nu : 0.0, M->n_worst ? M->worst[0].arg : 0.0,
           T->min, T->q1, T->med, T->q3, T->p95, T->max, T->mean, T->iqr,
           T->spread, T->reps, ns_points, scope, M->n_file);
}

static void csv_row(const char *sup, const char *fam, const char *law,
                    const char *assembly, const bench_metrics *M,
                    const bench_time *T, size_t ns_points)
{ csv_row_scope(sup, fam, law, assembly, M, T, ns_points, "kernel"); }

/* `--build-info`, without a device call: build record, ULP mode, and the Boost version,
 * include directory and origin (GPU_SUPPLIER_BOOST_SOURCE, "<dir> (<origin>)" from
 * boost_detect.sh --where gpu) and evaluation type of the device rows (nvcc disables
 * promote_double). */
static void print_build_info(void)
{
    bench_build_info_cuda("gpu_suppliers");
    bench_build_info_kv(stdout, "gpu_suppliers", "ulp_mode", bench_ulp_mode_name(BENCH_ACCURACY_ULP_MODE));
#ifdef GPU_SUPPLIER_WITH_BOOST
    char v[96];
    snprintf(v, sizeof v, "%d.%d.%d", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000,
             BOOST_VERSION % 100);
    bench_build_info_kv(stdout, "gpu_suppliers", "boost", v);
#ifdef GPU_SUPPLIER_BOOST_SOURCE
    bench_build_info_kv(stdout, "gpu_suppliers", "boost_source", GPU_SUPPLIER_BOOST_SOURCE);
#endif
    typedef boost::math::policies::evaluation<double, boost::math::policies::policy<> >::type eval_t;
    snprintf(v, sizeof v, "%s", sizeof(eval_t) > sizeof(double) ? "long double" : "double");
    bench_build_info_kv(stdout, "gpu_suppliers", "boost_eval_type", v);
#else
    bench_build_info_kv(stdout, "gpu_suppliers", "boost", "none (no Boost 1.87 or later with a device path at build time)");
#endif
#ifdef GPU_SUPPLIER_WITH_KAUST
    bench_build_info_kv(stdout, "gpu_suppliers", "kaust", "built (kaust_port.c compiled as CUDA)");
#else
    bench_build_info_kv(stdout, "gpu_suppliers", "kaust", "not built (CuBesselK not ported; make kaust)");
#endif
}

static void usage(FILE *o, const char *p)
{
    fprintf(o, "usage: %s [--family z_nu,h_nu_s] [--law all|<law>,...] [--supplier NAME]\n"
           "          [--format text|csv] [--data-dir data] [--max-points N]\n"
           "          [--time-points N] [--reps N] [--list] [--build-info]\n"
           "          [--ladder [--ladder-max N]] [--call-only]\n"
           "  --ladder times each device kernel on prefix rungs 2^8 .. the corpus\n"
           "          size (2^25 for the large datasets); --ladder-max caps the top rung\n"
           "  --call-only times the whole calls alone (four transfer modes, every\n"
           "          kernel): no kernel-only row, no scoring, no data held on the\n"
           "          device; rows in the --ladder layout, timing_scope call\n", p);
}

int main(int argc, char **argv)
{
    for (int a = 1; a < argc; a++)
        if (!strcmp(argv[a], "--build-info")) { print_build_info(); return 0; }

    const char *families = "z_nu,h_nu_s", *laws = "all", *root = "data";
    const char *format = "text", *only = NULL;
    size_t maxpts = 0, time_points = 0;
    int reps = BENCH_REPS_MIN_DEFAULT;
    int ladder = 0, call_only = 0;
    size_t ladder_max = (size_t)-1;  /* no cap: the corpus size; --ladder-max caps it */

    for (int a = 1; a < argc; a++) {
        const char *nx = (a + 1 < argc) ? argv[a+1] : NULL;
        if      (!strcmp(argv[a],"--family")   && nx) families = argv[++a];
        else if (!strcmp(argv[a],"--law")      && nx) laws     = argv[++a];
        else if (!strcmp(argv[a],"--supplier") && nx) only     = argv[++a];
        else if (!strcmp(argv[a],"--format")   && nx) format   = argv[++a];
        else if (!strcmp(argv[a],"--data-dir") && nx) root     = argv[++a];
        else if (!strcmp(argv[a],"--max-points") && nx) maxpts = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--time-points") && nx) time_points = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--reps") && nx) reps = atoi(argv[++a]);
        else if (!strcmp(argv[a],"--ladder")) ladder = 1;
        else if (!strcmp(argv[a],"--ladder-max") && nx) ladder_max = (size_t)atol(argv[++a]);
        else if (!strcmp(argv[a],"--call-only")) call_only = 1;
        else if (!strcmp(argv[a],"--help")) { usage(stdout, argv[0]); return 0; }
        else if (!strcmp(argv[a],"--list")) {
            printf("GPU kernels in this build:\n");
            printf("  libmatern-cuda    this library's own device kernel, "
                   "launched directly on device pointers\n"
                   "                    (its public CUDA entry does its own "
                   "transfers and cannot be timed kernel-only)\n");
#ifdef GPU_SUPPLIER_WITH_KAUST
            printf("  kaust-128         kaust_port.c gpu_supplier_kaust_kv/logkv "
                   "(fix=0,iv=128,adapt=0) -- the SAME code and settings as "
                   "the CPU row of this name\n");
            printf("  kaust-40          (fix=0,iv=40,adapt=0)\n");
            printf("  kaust-adaptive    (fix=1,iv=128,adapt=1) -- 128 panels, e10 applied,\n"
                   "                    non-overflowing log(cosh), and Geng et al.'s Eq. (7)\n"
                   "                    range instead of the released fixed t in [0,9]\n");
#else
            printf("  (kaust-gpu absent: CuBesselK not ported; make kaust)\n");
#endif
#ifdef GPU_SUPPLIER_WITH_BOOST
            printf("  boost-gpu         boost::math::cyl_bessel_k, "
                   "BOOST_MATH_GPU_ENABLED\n");
            printf("  boost-gpu-adapted the same call, with K_v = K_{v-2} + 2(v-1)/x K_{v-1}\n"
                   "                    wherever boost's |4v^2-25|/(8x) < eps^(1/4) Hankel\n"
                   "                    short-cut (>= 1.86, wrong near v = 5/2) could fire\n");
#else
            printf("  (boost-gpu absent: needs Boost >= 1.87 with a device path; "
                   "make externals, or make boost-headers BOOST_VERSIONS=1.87.0)\n");
#endif
            return 0;
        }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); usage(stderr, argv[0]); return 2; }
    }
    if (corpus_family_check(families)) return 2;

    /* Refused: --ladder (kernels alone) with --call-only (whole calls), and
     * --max-points (a stride over the file) with the ladder (prefixes). */
    if (ladder && call_only) {
        fprintf(stderr, "--call-only times whole calls and --ladder times kernels alone; "
                        "they exclude each other.  Refusing.\n");
        return 2;
    }
    if (ladder && maxpts) {
        fprintf(stderr, "--max-points decimates the corpus (it is a STRIDE), which breaks the\n"
                        "ladder's prefix property and changes the input mixture.  Refusing.\n"
                        "  drop --max-points, or drop --ladder.\n");
        return 2;
    }

    gsup sup[8]; int nsup = 0;
    sup[nsup].label = "libmatern-cuda"; sup[nsup].which = 2; nsup++;
#ifdef GPU_SUPPLIER_WITH_KAUST
    /* the same three KAUST rows as on the CPU, with the same labels */
    sup[nsup].label = "kaust-128";      sup[nsup].which = 0; nsup++;
    sup[nsup].label = "kaust-40";       sup[nsup].which = 3; nsup++;
    sup[nsup].label = "kaust-adaptive"; sup[nsup].which = 4; nsup++;
#endif
#ifdef GPU_SUPPLIER_WITH_BOOST
    /* The label carries this TU's BOOST_VERSION, as the CPU rows carry theirs. */
    static char boostlab[64];
    snprintf(boostlab, sizeof boostlab, "boost-gpu-%d.%d.%d",
             BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000,
             BOOST_VERSION % 100);
    sup[nsup].label = boostlab; sup[nsup].which = 1; nsup++;
    static char boostlab_ad[80];
    snprintf(boostlab_ad, sizeof boostlab_ad, "%s-adapted", boostlab);
    sup[nsup].label = boostlab_ad; sup[nsup].which = 5; nsup++;
#endif
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        fprintf(stderr, "suppliers_gpu: no CUDA device visible -- rows are absent "
                        "by design, not by failure.\n");
        return 0;
    }

    /* No KAUST __constant__ upload: its settings are kernel arguments (launch_s). */
    bench_raw_open(getenv("BENCH_TIMING_RAW"));
    bench_detail_open(getenv("BENCH_ACCURACY_DETAIL"));
    if (!strcmp(format,"csv")) {
        if (ladder || call_only)
            printf("schema,device,supplier,family,law,backend,n,ns_min,ns_q1,"
                   "ns_med,ns_q3,ns_p95,ns_max,ns_mean,ns_iqr,ns_spread,reps,"
                   "ns_points,timing_scope\n");
        else csv_header();
    }
    size_t rows = 0;

    char famb[64]; snprintf(famb, sizeof famb, "%s", families);
    char *fsave = NULL;
    for (char *fs = strtok_r(famb, ",", &fsave); fs; fs = strtok_r(NULL, ",", &fsave)) {
        for (int li = 0; li < N_LAWS; li++) {
            const char *law = ALL_LAWS[li];
            /* `all` excludes the size laws, which repeat the distributions of
             * `random` and `half_int`; they stay reachable by name for --ladder. */
            {   const size_t _ln = strlen(law);
                const int _large = _ln > 6 && !strcmp(law + _ln - 6, "_large");
                if (!strcmp(laws,"all") && (_large || !strcmp(law,"size_random") ||
                                            !strcmp(law,"size_half")))
                    continue;
            }
            if (strcmp(laws,"all") != 0) {
                char tmp[512]; snprintf(tmp,sizeof tmp,"%s",laws);
                int hit = 0; char *ls = NULL;
                for (char *t = strtok_r(tmp,",",&ls); t; t = strtok_r(NULL,",",&ls))
                    if (!strcmp(t,law)) { hit = 1; break; }
                if (!hit) continue;
            }
            corpus_dataset cs;
            if ((ladder && ladder_max != (size_t)-1)
                    ? corpus_dataset_load_prefix(&cs, root, fs, law, ladder_max)   /* the rungs need no more */
                    : corpus_dataset_load(&cs, root, fs, law, maxpts))
                { fprintf(stderr, "corpus: no data for %s/%s under %s -- dataset skipped\n", fs, law, root); continue; }
            const size_t n = cs.set.count;
            const int has_length_scale = cs.set.has_length_scale;

            /* Host z for the uploads; on h_nu_s both paths pass (h, nu, s) and
             * the kernels derive z themselves (gpu_supplier_z_of). */
            double *zh = (double *)malloc(n * sizeof *zh);
            for (size_t i = 0; i < n; i++)
                zh[i] = has_length_scale ? sqrt(2.0*cs.set.order[i]) * cs.set.h[i]
                                           / cs.set.length_scale[i]
                                         : cs.set.argument[i];
            double *out  = (double *)malloc(n * sizeof *out);
            /* Device copies for the kernel-only launches; none with --call-only,
             * where every timed call moves its own data, and none while a call is
             * timed (released around the call-scope loop below). */
            dev_set D; memset(&D, 0, sizeof D);
            if (!call_only && dev_acquire(&D, &cs, zh, n)) return -1;

            for (int si = 0; si < nsup; si++) {
                if (only && strcmp(only, sup[si].label)) continue;
                /* The library returns M, so it has one row, not one per assembly. */
                const int nasm = (sup[si].which == 2) ? 1 : ASM_COUNT;
                for (int a = 0; a < nasm; a++) {
                    /* The kernel-only launch, scored for the kernel-scope row. */
                    bench_metrics M; memset(&M, 0, sizeof M);
                    if (!call_only) {
                        /* NaN first (all-ones bytes) and the launch error checked:
                         * `D.out` is reused across suppliers and assemblies. */
                        CK(cudaMemset(D.out, 0xFF, n*sizeof(double)));
                        if (launch(sup[si].which, D.z, D.nu, D.out, (int)n, a)) continue;
                        { cudaError_t le = cudaGetLastError();
                          if (le != cudaSuccess) {
                              fprintf(stderr, "suppliers_gpu: %s %s/%s launch failed: %s"
                                      " -- row omitted\n", sup[si].label, fs, law,
                                      cudaGetErrorString(le));
                              continue;
                          } }
                        CK(cudaDeviceSynchronize());
                        CK(cudaMemcpy(out, D.out, n*sizeof(double), cudaMemcpyDeviceToHost));

                        bench_metrics_compute(&M, out, cs.set.reference_high,
                                           cs.set.reference_low, cs.accuracy_bits,
                                           cs.set.order,
                                           has_length_scale ? cs.set.h : cs.set.argument,
                                           has_length_scale ? cs.set.length_scale : NULL,
                                           n, BENCH_ACCURACY_ULP_MODE);
                        M.n_file = cs.file_points;
                        /* the detail sidecar (bench_metrics.h), before the timed loop reuses `out` */
                        bench_detail_emit("cuda", sup[si].label, fs, law,
                                       row_asm(sup[si].which, a), &M, out, cs.set.order,
                                       has_length_scale ? cs.set.h : cs.set.argument,
                                       has_length_scale ? cs.set.length_scale : NULL, n);
                    }

                    /* Call scope, under the four transfer modes.  Not in ladder
                     * mode: ladder rows (19 fields) must not mix with accuracy rows
                     * (44 fields) in one file; --call-only writes the ladder's 19.
                     * No device copy of this driver is held while a call is timed. */
                    if (!ladder) dev_release(&D);
                    for (int tm = 0; !ladder && tm < TM_COUNT; tm++) {
                        size_t cn = n;
                        if (time_points && cn > time_points) cn = time_points;
                        double *ct = (double *)malloc((size_t)reps * sizeof *ct);
                        int ok = (ct != NULL);
                        sg_call c = { sup[si].which, a, tm, has_length_scale, cn,
                                      zh, cs.set.order, cs.set.h, cs.set.length_scale, out };
                        /* A failed call (e.g. refused host registration) makes the mode
                         * unavailable; the row is omitted. */
                        if (ok && bench_time_cuda_call(sg_call_once, &c, cn, reps, ct)) {
                            ok = 0; cudaGetLastError();
                        }
                        /* The call-scope row is scored on its own path's output: one
                         * untimed call over the whole law into a NaN-filled buffer. */
                        bench_metrics MC; memset(&MC, 0, sizeof MC);
                        if (ok && !call_only) {
                            for (size_t k = 0; k < n; k++) out[k] = NAN;
                            if (timed_call(sup[si].which, a, tm, n, has_length_scale,
                                           zh, cs.set.order, cs.set.h,
                                           cs.set.length_scale, out)) {
                                ok = 0; cudaGetLastError();
                            } else {
                                bench_metrics_compute(&MC, out, cs.set.reference_high,
                                                   cs.set.reference_low, cs.accuracy_bits,
                                                   cs.set.order,
                                                   has_length_scale ? cs.set.h : cs.set.argument,
                                                   has_length_scale ? cs.set.length_scale : NULL,
                                                   n, BENCH_ACCURACY_ULP_MODE);
                                MC.n_file = cs.file_points;
                                { char dl[96];
                                  snprintf(dl, sizeof dl, "%s/%s",
                                           row_asm(sup[si].which, a), tm_name(tm));
                                  bench_detail_emit("cuda", sup[si].label, fs, law, dl,
                                                 &MC, out, cs.set.order,
                                                 has_length_scale ? cs.set.h : cs.set.argument,
                                                 has_length_scale ? cs.set.length_scale : NULL, n); }
                            }
                        }
                        if (ok) {
                            char lab[96];
                            snprintf(lab, sizeof lab, "%s/%s",
                                     row_asm(sup[si].which, a), tm_name(tm));
                            bench_raw_emit("cuda", sup[si].label, fs, law, lab,
                                        cn, ct, reps);
                            bench_time CT = bench_time_reduce(ct, reps);
                            rows++;
                            if (call_only && !strcmp(format,"csv"))
                                printf("%d,cuda,%s,%s,%s,%s,%zu,%.17g,%.17g,%.17g,"
                                       "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%zu,call\n",
                                       BENCH_SCHEMA, sup[si].label, fs, law, lab, cn,
                                       CT.min, CT.q1, CT.med, CT.q3, CT.p95,
                                       CT.max, CT.mean, CT.iqr, CT.spread,
                                       CT.reps, cn);
                            else if (call_only)
                                printf("%-11s %-9s %-17s %-22s n=%-8zu ns/elem min=%.4f "
                                       "med=%.4f (call)\n",
                                       sup[si].label, fs, law, lab, cn, CT.min, CT.med);
                            else if (!strcmp(format,"csv"))
                                csv_row_scope(sup[si].label, fs, law, lab,
                                              &MC, &CT, cn, "call");
                        } else if (!strcmp(format,"csv")) {
                            fprintf(stderr, "suppliers_gpu: %s %s/%s %s mode %s "
                                    "unavailable -- row omitted\n",
                                    sup[si].label, fs, law,
                                    row_asm(sup[si].which, a), tm_name(tm));
                        }
                        free(ct);
                    }
                    if (call_only) continue;   /* no kernel-only row */
                    if (!ladder && dev_acquire(&D, &cs, zh, n)) return -1;

                    /* Kernel-only timing, also on the ladder's prefix rungs:
                     * (1) the device is drained before each repetition; (2) CUDA
                     * events give device-side time without host launch overhead;
                     * (3) transfers are excluded, the data being resident. */
                    if (ladder) {
                        cudaEvent_t e0, e1;
                        cudaEventCreate(&e0); cudaEventCreate(&e1);
                        for (size_t nn = 256; nn <= n && nn <= ladder_max; nn <<= 1) {
                            double *lt = (double *)malloc((size_t)reps * sizeof *lt);
                            if (!lt) break;
                            for (int w = 0; w < BENCH_WARMUP_REPS; w++) {
                                launch(sup[si].which, D.z, D.nu, D.out, (int)nn, a);
                                cudaDeviceSynchronize();
                            }
                            for (int r = 0; r < reps; r++) {
                                cudaDeviceSynchronize();
                                cudaEventRecord(e0);
                                launch(sup[si].which, D.z, D.nu, D.out, (int)nn, a);
                                cudaEventRecord(e1);
                                cudaEventSynchronize(e1);
                                float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
                                lt[r] = (double)ms * 1e6 / (double)nn;
                            }
                            bench_raw_emit("cuda", sup[si].label, fs, law,
                                        row_asm(sup[si].which, a), nn, lt, reps);
                            bench_time LT = bench_time_reduce(lt, reps);
                            free(lt);
                            rows++;
                            if (!strcmp(format,"csv"))
                                printf("%d,cuda,%s,%s,%s,%s,%zu,%.17g,%.17g,%.17g,"
                                       "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%zu,kernel\n",
                                       BENCH_SCHEMA, sup[si].label, fs, law,
                                       row_asm(sup[si].which, a), nn,
                                       LT.min, LT.q1, LT.med, LT.q3, LT.p95,
                                       LT.max, LT.mean, LT.iqr, LT.spread,
                                       LT.reps, nn);
                        }
                        cudaEventDestroy(e0); cudaEventDestroy(e1);
                        continue;
                    }
                    size_t tn = n; if (time_points && tn > time_points) tn = time_points;
                    double *t = (double *)malloc((size_t)reps * sizeof *t);
                    cudaEvent_t ev0, ev1;
                    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                    for (int w = 0; w < BENCH_WARMUP_REPS; w++) {
                        launch(sup[si].which, D.z, D.nu, D.out, (int)tn, a);
                        cudaDeviceSynchronize();
                    }
                    for (int r = 0; r < reps; r++) {
                        cudaDeviceSynchronize();          /* (1) drain */
                        cudaEventRecord(ev0);
                        launch(sup[si].which, D.z, D.nu, D.out, (int)tn, a);
                        cudaEventRecord(ev1);
                        cudaEventSynchronize(ev1);
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, ev0, ev1);   /* (2) device-side */
                        t[r] = (double)ms * 1e6 / (double)tn;  /* ms to ns per element */
                    }
                    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
                    bench_raw_emit("cuda", sup[si].label, fs, law,
                                row_asm(sup[si].which, a), tn, t, reps);
                    bench_time T = bench_time_reduce(t, reps);
                    free(t);

                    rows++;
                    if (!strcmp(format,"csv"))
                        csv_row(sup[si].label, fs, law, row_asm(sup[si].which, a), &M, &T, tn);
                    else
                        printf("%-11s %-9s %-12s %-9s n=%-8zu ULPmax=%-12.4g "
                               "relmax=%-11.3e nonfin=%-7zu ns/elem=%.4f\n",
                               sup[si].label, fs, law,
                               row_asm(sup[si].which, a), M.n, M.u_max,
                               M.rel_max, M.n_nonfinite, T.min);
                }
            }
            dev_release(&D);
            free(zh); free(out); corpus_dataset_free(&cs);
        }
    }
    bench_raw_close();
    bench_detail_close();
    if (rows == 0) {
        fprintf(stderr, "suppliers_gpu: nothing matched --family '%s' --law '%s'.\n",
                families, laws);
        return 4;
    }
    return 0;
}
