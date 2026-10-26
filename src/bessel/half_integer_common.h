/* half_integer_common.h -- ISA-free parts of the half-integer closed form, shared with the
 * scalar backend: degree caps, the exact half-integer test, the coefficient recurrence and
 * the constant-run scan. */
#ifndef MATERN_HALF_INTEGER_COMMON_H
#define MATERN_HALF_INTEGER_COMMON_H

#include <stddef.h>

#define HALF_INTEGER_MAX_DEGREE 40

/* Each backend's cap is where its general router becomes the more accurate kernel. */
#define HALF_INTEGER_SCALAR_CAP HALF_INTEGER_MAX_DEGREE   /* scalar: no narrowing */
#define HALF_INTEGER_VECTOR_CAP 29                        /* vector: nu <= 29.5   */

/* AVX-512 and AVX2 entries scan the batch from its start: a half-integer point within the
 * cap takes the closed form inside a constant-order run or a mixed half-integer segment of
 * at least this many points, or one that reaches the end of the batch; otherwise it goes to
 * the router, as does the rest of the batch from the first shorter run that takes no closed
 * form.  The scalar and width-1 entries have no threshold. */
#define HALF_INTEGER_MIN_RUN 256

/* p if nu is exactly p + 1/2 with 0 <= p <= cap, else -1.  The test has no tolerance,
 * since the closed form holds only at half-integers; p + 0.5 is representable, so a
 * caller writing 1.5 or 2.5 always matches. */
static inline int half_integer_degree(double nu, int cap){
    const double t = 2.0 * nu;                    /* exact: multiply by 2 */
    if(!(t >= 1.0) || t > (2.0*(double)cap + 1.0)) return -1;
    if(t != (double)(long long)t) return -1;       /* 2nu must be an integer */
    const long long i = (long long)t;
    if((i & 1) == 0) return -1;                    /* ... and odd */
    return (int)((i - 1) >> 1);
}

/* c_0 .. c_p, built once per run by the ratio recurrence instead of factorials; every
 * c_k <= 1 carries only the roundings of its k steps. */
static inline void half_integer_coefficients(int p, double *c){
    c[0] = 1.0;
    for(int k = 1; k <= p; k++)
        c[k] = c[k-1] * 2.0 * (double)(p - k + 1) / ((double)k * (double)(2*p - k + 1));
}

/* Length of the leading run of identical nu, capped at n (n >= 1). */
static inline size_t constant_run_length(const double *nu, size_t n){
    size_t i = 1;
    while(i < n && nu[i] == nu[0]) i++;
    return i;
}

#endif /* MATERN_HALF_INTEGER_COMMON_H */
