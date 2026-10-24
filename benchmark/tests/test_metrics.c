/* test_metrics.c -- unit test of the accuracy statistics in wm_metrics.h on fixtures with
 * known errors.  Run by benchmark/tests/run_tests.sh; exit status 1 on any failure. */
#include <stdio.h>
#include <math.h>
#include "wm_metrics.h"
static int fails=0;
static void chk(const char*w,double got,double want){
    int ok=(got==want)||(want!=0&&fabs(got-want)/fabs(want)<1e-12);
    printf("  %-4s %-48s %.17g\n", ok?"ok":"FAIL", w, got); if(!ok){printf("        wanted %.17g\n",want);fails++;}
}
static void chkl(const char*w,long got,long want){
    int ok=(got==want); printf("  %-4s %-48s %ld\n", ok?"ok":"FAIL", w, got);
    if(!ok){printf("        wanted %ld\n",want);fails++;}
}
int main(void){
    /* 1..9 ulp above the reference, plus one exact, one non-finite, one zero-ref */
    enum { N = 12 };
    double v[N], hi[N], lo[N], nu[N], arg[N]; int bits[N];
    for (int i=0;i<N;i++){ hi[i]=1.0; lo[i]=0.0; nu[i]=1.5+i; arg[i]=10.0+i; bits[i]=100+i; }
    for (int i=0;i<9;i++){ v[i]=1.0; for(int k=0;k<=i;k++) v[i]=nextafter(v[i],2.0); }  /* +1..+9 ulp */
    v[9]  = 1.0;                       /* exact                       */
    v[10] = NAN;                       /* non-finite -> excluded      */
    hi[11]= 0.0; lo[11]=0.0; v[11]=0.0;/* zero reference, value 0     */

    wm_metrics M;
    wm_metrics_compute(&M, v, hi, lo, bits, nu, arg, NULL, N, WM_ULP_UP);

    puts("[population]");
    chkl("n",              (long)M.n, N);
    chkl("n_scored",       (long)M.n_scored, 11);        /* 12 - 1 nonfinite */
    chkl("n_nonfinite",    (long)M.n_nonfinite, 1);
    chkl("n_excluded",     (long)M.n_excluded, 1);
    chkl("n_ref_zero",     (long)M.n_ref_zero, 1);
    chkl("n_exact",        (long)M.n_exact, 2);          /* v[9] and the zero-ref pt */
    chkl("min_bits",       (long)M.min_bits, 100);

    puts("[ULP quantiles over |0,0,1..9|]");
    chk("u_min",  M.u_min, 0.0);
    chk("u_max",  M.u_max, 9.0);
    chk("u_med (nearest-rank of 11 pts)", M.u_med, 4.0);
    chk("u_mean (0+0+1..9)/11",           M.u_mean, 45.0/11.0);

    puts("[non-trivial median excludes the zero-reference point]");
    /* 10 non-trivial points {0,1,...,9}: nearest-rank median is index
     * round(0.5*(10-1)) = 5, so 5.0; the 11-point median including the
     * zero-reference point is 4.0. */
    chk("u_med_nontrivial (10 pts -> index 5)", M.u_med_nontrivial, 5.0);

    puts("[histogram: signed, exact bucket separate]");
    long pos=0,neg=0; for(int b=0;b<WM_HIST_BINS+2;b++){pos+=M.hist_pos[b];neg+=M.hist_neg[b];}
    chkl("total histogram entries == n_scored", pos+neg, (long)M.n_scored);
    chkl("exact bucket (bin 0) holds the 2 zeros", M.hist_pos[0], 2);
    chkl("nothing negative here", neg, 0);

    puts("[worst list is sorted by |ulp|, descending, and located]");
    chk("worst[0].ulp", fabs(M.worst[0].ulp), 9.0);
    chk("worst[1].ulp", fabs(M.worst[1].ulp), 8.0);
    chk("worst[0].nu tracks the point", M.worst[0].nu, 1.5+8);

    puts("[sign is kept]");
    double v2[3]={0,0,0}, h2[3]={1.0,1.0,1.0}, l2[3]={0,0,0}, n2[3]={1,1,1}, a2[3]={1,1,1};
    v2[0]=nextafter(1.0,0.0); v2[1]=nextafter(1.0,2.0); v2[2]=1.0;
    wm_metrics M2; wm_metrics_compute(&M2,v2,h2,l2,NULL,n2,a2,NULL,3,WM_ULP_UP);
    long n2neg=0,n2pos=0; for(int b=1;b<WM_HIST_BINS+2;b++){n2neg+=M2.hist_neg[b];n2pos+=M2.hist_pos[b];}
    chkl("one below the reference -> hist_neg", n2neg, 1);
    chkl("one above the reference -> hist_pos", n2pos, 1);

    puts("[worst list is the true top-64 when errors are NEGATIVE]");
    /* 64 points at -1..-64 steps, then one exact point.  Eviction compares |ulp|
     * with |ulp| of the last kept entry, so the exact point must not displace the
     * smallest negative error. */
    enum { NW = WM_WORST_KEEP + 1 };
    static double vw[NW], hw[NW], lw[NW], nw[NW], aw[NW];
    for (int i=0;i<NW;i++){ hw[i]=1.0; lw[i]=0.0; nw[i]=i; aw[i]=i; }
    for (int i=0;i<WM_WORST_KEEP;i++){ vw[i]=1.0; for(int k=0;k<=i;k++) vw[i]=nextafter(vw[i],0.0); }
    vw[WM_WORST_KEEP]=1.0;             /* exact, arrives last */
    wm_metrics MW; wm_metrics_compute(&MW,vw,hw,lw,NULL,nw,aw,NULL,NW,WM_ULP_UP);
    chkl("n_worst", (long)MW.n_worst, WM_WORST_KEEP);
    /* one step below 1.0 is half an ulp of the reference (the binade below) */
    chk("worst[63] is |-1 step| = 0.5 ulp, not the exact point", fabs(MW.worst[WM_WORST_KEEP-1].ulp), 0.5);
    chk("worst[0] is |-64 steps| = 32 ulp", fabs(MW.worst[0].ulp), 32.0);

    puts("[empty input does not crash or divide by zero]");
    wm_metrics M3; wm_metrics_compute(&M3,v,hi,lo,bits,nu,arg,NULL,0,WM_ULP_UP);
    chkl("n_scored", (long)M3.n_scored, 0);

    printf("\n%s (%d failures)\n", fails?"FAILURES":"ALL PASS", fails);
    return fails!=0;
}
