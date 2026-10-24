/* region_map.h -- the trivial bucket, matern_trivial_run(): invalid input, z = 0, z = inf
 * and z past the exponential limit, per element (see the accompanying paper, treatment of
 * limiting and exceptional cases, and large-argument formulation). */
#ifndef MATERN_REGION_MAP_H
#define MATERN_REGION_MAP_H

#include <float.h>
#include <math.h>
#include "vector_current.h"

/* the trivial bucket evaluates the extended range, so it needs the two tail kernels */
#include "asymptotic_large_argument.h"
#include "debye_uniform_order.h"
#include "temme_small_argument.h"   /* z = 0 with h > 0: ln z from the inputs */

#include "region_partition.h"


static void matern_trivial_run(double *restrict out,const double *restrict h,
                          const double *restrict nu,const double *restrict scale,size_t m){
    for(size_t i=0;i<m;i++){
        const double H=h[i], NU=nu[i];
#ifdef MATERN_ENTRY_Z
        /* z is the input; the guard is on (z, nu). */
        const double z = H;
        if(!(NU>0.0 && NU<=DBL_MAX) || !(z>=0.0)) out[i] = (double)NAN;  /* invalid / NaN / inf nu */
#else
        const double R=scale[i];
        double z = sqrt(2.0*NU)*H/R;
        if(!(NU>0.0 && NU<=DBL_MAX) || !(R>0.0 && R<=DBL_MAX) || !(H>=0.0))
            out[i] = (double)NAN;                    /* invalid, NaN, or infinite nu or s */
#endif
        /* h = 0: M = 1 exactly.  H is tested as well: past nu = DBL_MAX/2, 2 nu
         * overflows and the derived z is sqrt(inf) * 0 = NaN. */
        else if(z==0.0 || H==0.0){
#ifndef MATERN_ENTRY_Z
            /* h > 0 with z = 0: sqrt(2 nu) h / s underflowed.  Below order 1/2 M = 1 -
             * O((z/2)^(2 nu)) can differ from 1; the small-argument cell takes ln z from
             * the inputs. */
            if (H > 0.0 && NU < 0.5){
                const double v = vec_first_lane(small_argument_cell(VEC_VSET(0.0), VEC_VSET(NU),
                                                                   VEC_VSET(H), VEC_VSET(R)));
                out[i] = (v > 1.0) ? 1.0 : (v > 0.0) ? v : 0.0;
            } else
#endif
            out[i] = 1.0;
        }
        /* Past the exponential limit every prefactor is folded into one exponent, so
         * subnormal results are delivered.  z = inf returns 0: the fold's
         * -z + (nu - 1/2) ln z is inf - inf there.  A value outside (0, 1] becomes 0. */
        else if(!(z<=MATERN_EXP_RANGE_LIMIT)){
            if(!isfinite(z)) out[i] = 0.0;
            /* per element: large-argument tail below EXTENDED_RANGE_ORDER_SPLIT, Debye above */
            else { double v;
                   if (NU < EXTENDED_RANGE_ORDER_SPLIT)
                        matern_large_argument_extended_x1 (&v,(double*)&h[i],(double*)&nu[i],(double*)&scale[i],1);
                   else matern_debye_extended_x1(&v,(double*)&h[i],(double*)&nu[i],(double*)&scale[i],1);
                   out[i] = (v > 0.0 && v <= 1.0) ? v : 0.0; }
        }
        else                 out[i] = (double)NAN;                    /* unreachable */
    }
}

#endif /* MATERN_REGION_MAP_H */
