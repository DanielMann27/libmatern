/* backend_name.c -- prints the backend the library runs in this process.  The per-backend
 * targets of make check compare it with MATERN_BACKEND, since the dispatcher falls back
 * without a message for a backend this CPU cannot run, and report such a backend as skipped. */
#include <stdio.h>
#include "matern.h"

int main(void)
{
    puts(matern_backend_name());
    return 0;
}
