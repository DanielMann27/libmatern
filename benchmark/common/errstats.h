/* errstats.h -- `wmq_ulp_at`, the ulp of a reference value, on which ulpmodes.h builds.
 * Taken at the reference, so a result cannot gain a larger ulp by landing in a higher
 * binade; nextafter covers the subnormal range, which the corpus reaches. */
#ifndef CPU_ERRSTATS_H
#define CPU_ERRSTATS_H
#include <math.h>

/* ulp at a double magnitude; the subnormal floor is the smallest positive double. */
static inline double wmq_ulp_at(double r)
{
    double u;
    r = fabs(r);
    if (r > 0.0 && isfinite(r)) u = nextafter(r, INFINITY) - r;
    else                        u = 4.9406564584124654e-324;
    if (!(u > 0.0))             u = 4.9406564584124654e-324;
    return u;
}

#endif /* CPU_ERRSTATS_H */
