/* version.c -- matern_version(): the MATERN_VERSION_STRING of include/matern.h as
 * compiled into the library, for comparison with the header a program is compiled
 * against. */
#include "matern.h"

const char *matern_version(void)
{
    return MATERN_VERSION_STRING;
}
