/* half_integer_table_device.c -- prints the coefficients of the device half-integer table
 * (matern_tier_a_coefficients_device): c_0 .. c_p of every degree p <= 32, built by the
 * ratio recurrence in long double and rounded once to double, as hexadecimal literals.
 * The first line states long double's mantissa width; the table was generated where it is
 * 64 (x86-64), and `make audit` compares the header against this output there.
 *   cc -O2 -o build/half_integer_table_device tools/half_integer_table_device.c */
#include <float.h>
#include <stdio.h>

#define PMAX 32

int main(void)
{
    printf("/* LDBL_MANT_DIG %d */\n", LDBL_MANT_DIG);
    for (int p = 0; p <= PMAX; p++) {
        long double c = 1.0L;
        printf("    /* p = %d */\n    %a,", p, 1.0);
        for (int j = 1; j <= p; j++) {
            c *= (long double)(2*(p-j+1)) / (long double)(j*(2*p-j+1));
            printf("%s%a,", (j % 4 == 0) ? "\n    " : " ", (double)c);
        }
        printf("\n");
    }
    return 0;
}
