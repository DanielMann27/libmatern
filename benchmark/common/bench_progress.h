/* bench_progress.h -- bench_progress(fmt, ...) writes one time-stamped line, in one write(2),
 * to the descriptor named by BENCH_PROGRESS_FD (none: no output).  Drivers call it only
 * between measurements, never inside a timed interval. */
#ifndef BENCH_PROGRESS_H
#define BENCH_PROGRESS_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static inline void bench_progress(const char *fmt, ...)
{
    static int fd = -2;
    if (fd == -2) {
        const char *e = getenv("BENCH_PROGRESS_FD");
        fd = (e && *e) ? atoi(e) : -1;
    }
    if (fd < 0) return;
    char buf[256];
    const time_t now = time(NULL);
    struct tm tm;
    size_t n = localtime_r(&now, &tm) ? strftime(buf, sizeof buf, "%H:%M:%S  ", &tm) : 0;
    va_list ap;
    va_start(ap, fmt);
    const int m = vsnprintf(buf + n, sizeof buf - n - 1, fmt, ap);
    va_end(ap);
    if (m < 0) return;
    n += (size_t)m < sizeof buf - n - 1 ? (size_t)m : sizeof buf - n - 2;
    buf[n++] = '\n';
    if (write(fd, buf, n) < 0) fd = -1;     /* not open: stop trying */
}

#endif /* BENCH_PROGRESS_H */
