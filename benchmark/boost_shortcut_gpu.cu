/* boost_shortcut_gpu.cu -- stage `gpu-boost-shortcut`: Boost's K_nu at nu = 2.5 + delta
 * against z (data/boost_shortcut/), as a host long-double row and plain and adapted
 * device rows on one Boost version, scored on K itself against the arb reference. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <boost/version.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include "boost_adapted.h"
#include "bench_metrics.h"          /* BENCH_SCHEMA, and the pipeline's ULP (ulpmodes.h) */
#include "corpus_bin.h"
#include "common/build_info_cuda.h"

extern "C" void bench_boost_shortcut_host(const double *, const double *, double *, int);
extern "C" int  bench_boost_shortcut_host_promoted(void);
extern "C" int  bench_boost_shortcut_host_version(void);
extern "C" void bench_boost_shortcut_host_build_info(FILE *);

/* The record's schema number, from bench_metrics.h. */
#define SCHEMA BENCH_SCHEMA

__global__ void k_plain(const double *nu, const double *z, double *o, int n)
{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = boost::math::cyl_bessel_k(nu[i], z[i]); }
__global__ void k_adapted(const double *nu, const double *z, double *o, int n)
{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = bench_boost_kv_adapted(nu[i], z[i]); }

/* One array of a corpus file (benchmark/corpus_bin.h); empty on any error. */
static std::vector<double> column(const std::string &p, uint32_t kind, int array)
{
    std::vector<double> v; FILE *f = fopen(p.c_str(), "rb");
    if (!f) return v;
    corpus_bin_header h; char why[160] = "";
    if (corpus_bin_read_header(f, p.c_str(), &h, why, sizeof why) == 0 && h.kind == kind) {
        v.resize(h.count);
        if (array == 2) {
            std::vector<int> b(h.count);
            if (corpus_bin_read_array(f, &h, 2, 1, h.count, b.data())) v.clear();
            else for (size_t i = 0; i < b.size(); i++) v[i] = b[i];
        } else if (corpus_bin_read_array(f, &h, array, 1, h.count, v.data())) v.clear();
    } else if (why[0]) fprintf(stderr, "boost_shortcut_gpu: %s: %s\n", p.c_str(), why);
    fclose(f); return v;
}

/* `--build-info`: this TU's record and that of boost_shortcut_host.o (the cpu row). */
static void print_build_info(void)
{
    const char *unit = "gpu_boost_shortcut";
    char v[32];
    bench_build_info_cuda(unit);
    bench_build_info_kv(stdout, unit, "ulp_mode", bench_ulp_mode_name(BENCH_ULP_MODE_DEFAULT));
    snprintf(v, sizeof v, "%d.%d.%d", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000,
             BOOST_VERSION % 100);
    bench_build_info_kv(stdout, unit, "boost", v);
#ifdef GPU_SUPPLIER_BOOST_SOURCE
    bench_build_info_kv(stdout, unit, "boost_source", GPU_SUPPLIER_BOOST_SOURCE);
#endif
    typedef boost::math::policies::evaluation<double, boost::math::policies::policy<> >::type eval_t;
    bench_build_info_kv(stdout, unit, "boost_eval_type",
                     sizeof(eval_t) > sizeof(double) ? "long double" : "double");
    bench_boost_shortcut_host_build_info(stdout);
}

int main(int argc, char **argv)
{
    for (int a = 1; a < argc; a++)
        if (!strcmp(argv[a], "--build-info")) { print_build_info(); return 0; }
    std::string root = "data";
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--data-dir") && a + 1 < argc) { root = argv[++a]; continue; }
        const int help = !strcmp(argv[a], "--help");
        if (!help) fprintf(stderr, "unknown option %s\n", argv[a]);
        fprintf(help ? stdout : stderr,
                "usage: %s [--data-dir DIR] [--build-info] [--help]\n"
                "  Boost's K_nu at nu = 2.5 + delta against the reference in\n"
                "  DIR/boost_shortcut/ or DIR (DIR: the corpus root, default data), as CSV\n"
                "  on stdout; without a CUDA device it writes nothing and says so.\n"
                "  --build-info prints this binary's build record.\n", argv[0]);
        return help ? 0 : 2;
    }
    const int N = 873;                 /* 9 delta x 97 z */
    char suf[32]; snprintf(suf, sizeof suf, "_%d.bin", N);
    /* data/boost_shortcut/, or the files directly in data/ (data/README.md) */
    std::string d = root + "/boost_shortcut/";
    if (FILE *t = fopen((d + "nu_boost_shortcut" + suf).c_str(), "rb")) fclose(t);
    else d = root + "/";
    std::vector<double> nu = column(d + "nu_boost_shortcut" + suf, CORPUS_KIND_AXIS, 0),
                        z  = column(d + "z_boost_shortcut" + suf, CORPUS_KIND_AXIS, 0),
                        hi = column(d + "refK_boost_shortcut" + suf, CORPUS_KIND_REFERENCE, 0),
                        lo = column(d + "refK_boost_shortcut" + suf, CORPUS_KIND_REFERENCE, 1),
                        bd = column(d + "refK_boost_shortcut" + suf, CORPUS_KIND_REFERENCE, 2);
    if ((int)nu.size() != N || (int)z.size() != N || (int)hi.size() != N) {
        fprintf(stderr, "boost_shortcut_gpu: %s incomplete (%zu/%zu/%zu of %d) -- "
                        "the corpus files are expected (sha256 in data/MANIFEST.md; make corpus DOWNLOAD=yes); "
                        "--data-dir names another data root\n", d.c_str(), nu.size(), z.size(), hi.size(), N);
        return 1;
    }
    std::vector<long> bits(N);
    for (int i = 0; i < N; i++) bits[i] = (long)bd[i];
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        fprintf(stderr, "boost_shortcut_gpu: no CUDA device -- rows absent by design\n");
        return 0;
    }
    if (bench_boost_shortcut_host_version() != BOOST_VERSION) {
        fprintf(stderr, "boost_shortcut_gpu: host object built on Boost %d, device TU on %d -- "
                        "the rows would differ in VERSION, which is not what they compare\n",
                bench_boost_shortcut_host_version(), BOOST_VERSION);
        return 1;
    }
    std::vector<double> kh(N), kp(N), ka(N);
    bench_boost_shortcut_host(nu.data(), z.data(), kh.data(), N);
    double *dn, *dz, *dk;
    cudaMalloc(&dn, N * 8); cudaMalloc(&dz, N * 8); cudaMalloc(&dk, N * 8);
    cudaMemcpy(dn, nu.data(), N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, z.data(), N * 8, cudaMemcpyHostToDevice);
    /* NaN fill (all bytes 0xFF) before each launch; each launch and copy checked. */
    cudaError_t err = cudaSuccess;
    cudaMemset(dk, 0xFF, N * 8);
    k_plain<<<(N + 127) / 128, 128>>>(dn, dz, dk, N);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpy(kp.data(), dk, N * 8, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemset(dk, 0xFF, N * 8);
    k_adapted<<<(N + 127) / 128, 128>>>(dn, dz, dk, N);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpy(ka.data(), dk, N * 8, cudaMemcpyDeviceToHost);
    cudaFree(dn); cudaFree(dz); cudaFree(dk);
    if (err != cudaSuccess) { fprintf(stderr, "boost_shortcut_gpu: %s\n", cudaGetErrorString(err)); return 1; }

    char ver[32];
    snprintf(ver, sizeof ver, "%d.%d.%d", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000, BOOST_VERSION % 100);
    const std::string s_h = std::string("boost-") + ver, s_p = std::string("boost-gpu-") + ver,
                      s_a = s_p + "-adapted";
    const char *ph = bench_boost_shortcut_host_promoted() ? "long-double" : "double";
    struct { const char *dev, *prec; const std::string *sup; const std::vector<double> *k; } R[] = {
        {"cpu", ph, &s_h, &kh}, {"cuda", "double", &s_p, &kp}, {"cuda", "double", &s_a, &ka}};
    /* Columns: docs/benchmark.md, "Output"; `ulp` is bench_ulp_signed, as in the
     * accuracy rows. */
    printf("schema,device,supplier,precision,law,nu,delta,z,k,rel,ref_bits,ulp_mode,ulp\n");
    for (auto &r : R)
        for (int i = 0; i < N; i++) {
            const double k = (*r.k)[i];
            const double rel = std::isfinite(k) ? fabs((k - hi[i]) - lo[i]) / fabs(hi[i]) : NAN;
            const double ulp = std::isfinite(k) ? bench_ulp_signed(k, hi[i], lo[i], BENCH_ULP_MODE_DEFAULT) : NAN;
            printf("%d,%s,%s,%s,boost_shortcut,%a,%a,%a,%a,%.17g,%ld,%s,%.17g\n", SCHEMA, r.dev,
                   r.sup->c_str(), r.prec, nu[i], nu[i] - 2.5, z[i], k, rel, bits[i],
                   bench_ulp_mode_name(BENCH_ULP_MODE_DEFAULT), ulp);
        }
    return 0;
}
