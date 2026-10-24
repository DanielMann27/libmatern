/* test_entry_z_validity.c -- matern_covariance_z returns NaN for nu <= 0, z < 0 and NaN
 * inputs and 1 for z == 0 (matern.h), at n = 1 and n = 9.  Run by
 * benchmark/tests/run_tests.sh; exit status 1 on any wrong case. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matern.h"
int main(void){
  struct { double z, nu; const char *want; } c[] = {
    { 0.0,  1.5, "1"    }, { -1.0, 1.5, "NaN"  }, { 1.0, -1.5, "NaN"  },
    { 1.0,  0.0, "NaN"  }, { NAN,  1.5, "NaN"  }, { 1.0, NAN,  "NaN"  },
    { INFINITY, 1.5, "0 or NaN" }, { 1.0, 1.5, "finite" },
  };
  int bad = 0;
  for (int k = 0; k < 8; k++)
    for (int n = 1; n <= 9; n += 8) {          /* n=1 (short) and n=9 (vector) */
      double z[9], nu[9], o[9];
      for (int i = 0; i < n; i++) { z[i]=c[k].z; nu[i]=c[k].nu; o[i]=-1; }
      matern_covariance_z(o, z, nu, (size_t)n, NULL);
      const char *got = isnan(o[0]) ? "NaN" : (o[0]==1.0 ? "1" : (o[0]==0.0 ? "0" : "finite"));
      int ok = !strcmp(got, c[k].want) ||
               (!strcmp(c[k].want,"0 or NaN") && (!strcmp(got,"0")||!strcmp(got,"NaN")));
      if (!ok) bad++;
      printf("  n=%-2d z=%-9.3g nu=%-7.3g -> %-7s want %-9s %s\n",
             n, c[k].z, c[k].nu, got, c[k].want, ok?"":"  <<< WRONG");
    }
  printf("\n  %s (%d wrong)\n", bad?"FAILURES":"all validity cases correct", bad);
  return bad!=0;
}
