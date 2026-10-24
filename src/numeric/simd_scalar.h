/* simd_scalar.h -- width-1 type and width names under -DMATERN_TARGET_SCALAR
 * (SIMD_TYPE_FLOAT is double, every width constant 1); supplies no arithmetic. */
#ifndef MATERN_SIMD_SCALAR_H
#define MATERN_SIMD_SCALAR_H

#include <float.h>
#include <math.h>
#include <stdint.h>

/* Defines what fma() means in this object; must precede any kernel text. */
#include "fused_multiply_add_scalar.h"

#define DATA_TYPE_FLOAT  double
#define SIMD_TYPE_FLOAT  double
#define MASK_FLOAT       int

#define INDEX_FACTOR  1

/* MASK_REDUCE of a mask with every lane set: the all-lanes convergence test. */
#define WHILE_FACTOR  1

#define LOAD_FLOAT(X)   (*(const double *)(X))
#define SET_FLOAT(X)    ((double)(X))

#endif /* MATERN_SIMD_SCALAR_H */
