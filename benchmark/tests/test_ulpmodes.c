/* test_ulpmodes.c -- unit test of the ULP-spacing conventions in ulpmodes.h.  Run by
 * benchmark/tests/run_tests.sh; exit status 1 on any failure. */
#include <stdio.h>
#include <math.h>
#include "ulpmodes.h"
static int fails=0;
static void chk(const char*what,double got,double want){
    int ok = (got==want) || (want!=0 && fabs(got-want)/fabs(want)<1e-15);
    if(!ok){printf("  FAIL %-46s got %.17g want %.17g\n",what,got,want);fails++;}
    else    printf("  ok   %-46s %.17g\n",what,got);
}
int main(void){
    puts("[1] spacing at ordinary values: all modes must agree");
    for(int m=0;m<WM_ULP_MODE_COUNT;m++){
        char b[80]; snprintf(b,sizeof b,"ulp(1.5) mode=%s",wm_ulp_mode_name(m));
        chk(b, wm_ulp_at_mode(1.5,0.0,(wm_ulp_mode)m), ldexp(1.0,-52));
    }
    puts("[2] AT a power of two the modes must DIFFER (this is the whole point)");
    chk("ulp(1.0) up      = 2^-52", wm_ulp_at_mode(1.0,0.0,WM_ULP_UP),      ldexp(1.0,-52));
    chk("ulp(1.0) down    = 2^-53", wm_ulp_at_mode(1.0,0.0,WM_ULP_DOWN),    ldexp(1.0,-53));
    chk("ulp(1.0) muller  = 2^-52", wm_ulp_at_mode(1.0,0.0,WM_ULP_MULLER),  ldexp(1.0,-52));
    chk("ulp(1.0) straddle(lo<0)=2^-53", wm_ulp_at_mode(1.0,-1e-20,WM_ULP_STRADDLE), ldexp(1.0,-53));
    chk("ulp(1.0) straddle(lo>0)=2^-52", wm_ulp_at_mode(1.0, 1e-20,WM_ULP_STRADDLE), ldexp(1.0,-52));
    puts("[3] UP must be wmq_ulp_at VERBATIM (delegation, not a copy)");
    double xs[]={1.0,1.5,7.059e-4,1.7e-307,9.881e-324,0x1p-1074,0.0};
    int d=0; for(int i=0;i<7;i++) if(wm_ulp_at_mode(xs[i],0.0,WM_ULP_UP)!=wmq_ulp_at(xs[i])) d++;
    chk("UP == wmq_ulp_at over 7 magnitudes (0 diffs)", (double)d, 0.0);
    puts("[4] subnormal floor, every mode");
    for(int m=0;m<WM_ULP_MODE_COUNT;m++){
        char b[80]; snprintf(b,sizeof b,"ulp(2^-1074) mode=%s",wm_ulp_mode_name(m));
        chk(b, wm_ulp_at_mode(0x1p-1074,0.0,(wm_ulp_mode)m), 0x1p-1074);
    }
    puts("[5] signed error keeps its sign");
    chk("v one ulp ABOVE ref -> +1", wm_ulp_signed(nextafter(1.5,2),1.5,0.0,WM_ULP_UP), 1.0);
    chk("v one ulp BELOW ref -> -1", wm_ulp_signed(nextafter(1.5,1),1.5,0.0,WM_ULP_UP),-1.0);
    chk("exact -> 0",                wm_ulp_signed(1.5,1.5,0.0,WM_ULP_UP), 0.0);
    puts("[6] dd reference half an ulp above 1.0 -> both neighbours 0.5");
    chk("v=1.0",        fabs(wm_ulp_signed(1.0,1.0,ldexp(1.0,-53),WM_ULP_UP)), 0.5);
    chk("v=nextafter",  fabs(wm_ulp_signed(nextafter(1.0,2),1.0,ldexp(1.0,-53),WM_ULP_UP)), 0.5);
    printf("\n%s (%d failures)\n", fails?"FAILURES":"ALL PASS", fails);
    return fails!=0;
}
