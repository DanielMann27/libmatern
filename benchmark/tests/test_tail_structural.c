/* test_tail_structural.c -- short calls (n = 1..NMAX) through both public entries: no
 * output may be non-finite; agreement with one bulk call is reported, not asserted.
 * Run by benchmark/tests/run_tests.sh; exit status 1 on a non-finite output. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matern.h"
#define NMAX 24
#define BULK 256
static double NU[BULK],Z[BULK],H[BULK],SCALE[BULK],OB[BULK],OT[BULK];

/* Input set spanning the corpus, including z beyond the exponential limit (z > 708). */
static void build(void){
    const double nus[]={1e-3,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.7,8.0,15.0,29.5,30.0,35.0,100.0,1000.0};
    const double zs []={1e-6,1e-3,0.5,1.0,5.0,17.0,22.0,35.0,140.0,300.0,700.0,708.0,760.0,2000.0,1e4,5.0};
    for(int i=0;i<BULK;i++){
        double nu=nus[i%16], z=zs[(i/16)%16];
        NU[i]=nu; Z[i]=z;
        SCALE[i]=1.0; H[i]=z/sqrt(2.0*nu);   /* so the derived z is this z */
    }
}
int main(void){
    build();
    const char *bk = getenv("MATERN_BACKEND"); if(!bk) bk="auto";
    long nf_z=0, nf_h=0, dis_z=0, dis_h=0, tot=0;
    double worst_z=0, worst_h=0;

    /* bulk reference: one big call, which the vector classifier handles */
    matern_covariance_z(OB,Z,NU,BULK,NULL);
    static double OBH[BULK]; matern_covariance(OBH,H,NU,SCALE,BULK,NULL);

    for(int n=1;n<=NMAX;n++){
        /* slide the window so tails land at every alignment */
        for(int start=0; start+n<=BULK; start+=7){
            memset(OT,0,sizeof OT);
            matern_covariance_z(OT,Z+start,NU+start,(size_t)n,NULL);
            for(int i=0;i<n;i++){ tot++;
                if(!isfinite(OT[i])) nf_z++;
                else { double r=OB[start+i]!=0?fabs(OT[i]-OB[start+i])/fabs(OB[start+i]):0;
                       if(OT[i]!=OB[start+i]){dis_z++; if(r>worst_z)worst_z=r;} } }
            memset(OT,0,sizeof OT);
            matern_covariance(OT,H+start,NU+start,SCALE+start,(size_t)n,NULL);
            for(int i=0;i<n;i++){
                if(!isfinite(OT[i])) nf_h++;
                else { double r=OBH[start+i]!=0?fabs(OT[i]-OBH[start+i])/fabs(OBH[start+i]):0;
                       if(OT[i]!=OBH[start+i]){dis_h++; if(r>worst_h)worst_h=r;} } }
        }
    }
    printf("%-8s n=1..%d, %ld point-evaluations per entry\n", bk, NMAX, tot);
    printf("   covariance_z : non-finite %-6ld   differs-from-bulk %-7ld  worst rel %.2e\n",nf_z,dis_z,worst_z);
    printf("   covariance   : non-finite %-6ld   differs-from-bulk %-7ld  worst rel %.2e\n",nf_h,dis_h,worst_h);
    return (nf_z||nf_h)?1:0;
}
