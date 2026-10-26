/* numa_bind.h -- binds the process to a NUMA node without libnuma, then pins it to one
 * CPU of the node (BENCH_PIN=node keeps the node); otherwise runs unpinned, and reports
 * either way.  Linux only; elsewhere bench_bind_node() reports a no-op. */
/* _GNU_SOURCE must precede every include, not only <sched.h>: <stdio.h> pulls in
 * features.h, which fixes the feature-test macros. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#ifndef BENCH_NUMA_BIND_H
#define BENCH_NUMA_BIND_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#if defined(__linux__)
#include <sched.h>
static inline int bench_parse_cpulist(const char *s, cpu_set_t *set){
    int n=0; CPU_ZERO(set);
    while(*s){ char *e; long a=strtol(s,&e,10); if(e==s) break; long b=a;
        if(*e=='-'){ s=e+1; b=strtol(s,&e,10); }
        for(long c=a;c<=b;c++){ CPU_SET((int)c,set); n++; }
        s = (*e==',') ? e+1 : e; if(*e!=',') break; }
    return n;
}
/* returns: 2 = bound to node, 1 = node found but no usable CPUs, 0 = unavailable,
 * 3 = bound, but something else narrows the effective mask */
static inline int bench_bind_node(int node, char *why, size_t wn){
    char path[128], buf[4096];
    snprintf(path,sizeof path,"/sys/devices/system/node/node%d/cpulist",node);
    FILE *f=fopen(path,"r");
    if(!f){ snprintf(why,wn,"no /sys NUMA topology (node%d); running UNPINNED",node); return 0; }
    if(!fgets(buf,sizeof buf,f)){ fclose(f); snprintf(why,wn,"empty cpulist; UNPINNED"); return 0; }
    fclose(f);
    cpu_set_t want, allowed, use;
    if(!bench_parse_cpulist(buf,&want)){ snprintf(why,wn,"unparsable cpulist; UNPINNED"); return 0; }
    if(sched_getaffinity(0,sizeof allowed,&allowed)!=0){ snprintf(why,wn,"sched_getaffinity failed; UNPINNED"); return 0; }
    CPU_ZERO(&use); int k=0;
    for(int c=0;c<CPU_SETSIZE;c++) if(CPU_ISSET(c,&want)&&CPU_ISSET(c,&allowed)){ CPU_SET(c,&use); k++; }
    if(k==0){ snprintf(why,wn,"node%d has no CPUs in this process's affinity mask; UNPINNED",node); return 1; }
    if(sched_setaffinity(0,sizeof use,&use)!=0){ snprintf(why,wn,"sched_setaffinity denied (cgroup?); UNPINNED"); return 0; }
    /* Read the mask back: sched_setaffinity can succeed while a cpuset cgroup
     * narrows the effective set, so success is reported only for the verified mask. */
    cpu_set_t got; int g=0;
    if(sched_getaffinity(0,sizeof got,&got)!=0){
        snprintf(why,wn,"bound to NUMA node%d (%d CPUs requested) but the mask "
                        "could NOT be read back; treat placement as UNVERIFIED",node,k);
        return 2;
    }
    for(int c=0;c<CPU_SETSIZE;c++) if(CPU_ISSET(c,&got)) g++;
    if(g!=k){
        snprintf(why,wn,"asked for node%d (%d CPUs) but the EFFECTIVE mask has %d; "
                        "something else is narrowing it -- placement NOT as requested",
                 node,k,g);
        return 3;
    }
    /* One CPU, so that the timed thread cannot migrate within the node. */
    const char *pin = getenv("BENCH_PIN");
    if(!pin || strcmp(pin,"node")!=0){
        int c = sched_getcpu();
        if(c<0 || c>=CPU_SETSIZE || !CPU_ISSET(c,&got))
            for(c=0;c<CPU_SETSIZE && !CPU_ISSET(c,&got);c++) ;
        cpu_set_t one, back; CPU_ZERO(&one); CPU_SET(c,&one);
        if(sched_setaffinity(0,sizeof one,&one)==0 && sched_getaffinity(0,sizeof back,&back)==0
           && CPU_EQUAL(&one,&back)){
            snprintf(why,wn,"pinned to CPU %d of NUMA node%d (%d CPUs of the affinity mask on "
                            "the node; VERIFIED by read-back; BENCH_PIN=node binds the whole "
                            "node); memory follows by first touch",c,node,k);
            return 2;
        }
        sched_setaffinity(0,sizeof got,&got);
        snprintf(why,wn,"bound to NUMA node%d (%d CPUs, VERIFIED by read-back); pinning to "
                        "one CPU failed, so the thread may migrate within the node",node,k);
        return 2;
    }
    snprintf(why,wn,"bound to NUMA node%d (%d CPUs of the affinity mask, VERIFIED by "
                    "read-back; BENCH_PIN=node); memory follows by first touch",node,k);
    return 2;
}
/* The node holding the most CPUs of this process's affinity mask, or -1 (no sysfs
 * topology, or no node with allowed CPUs). */
static inline int bench_best_node(int *n_nodes_out)
{
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return -1;
    int best = -1, best_k = 0, seen = 0;
    for (int node = 0; node < 64; node++) {
        char path[128], buf[4096];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        seen++;
        if (fgets(buf, sizeof buf, f)) {
            cpu_set_t want; int k = 0;
            if (bench_parse_cpulist(buf, &want))
                for (int c = 0; c < CPU_SETSIZE; c++)
                    if (CPU_ISSET(c, &want) && CPU_ISSET(c, &allowed)) k++;
            if (k > best_k) { best_k = k; best = node; }
        }
        fclose(f);
    }
    if (n_nodes_out) *n_nodes_out = seen;
    return best;
}
#else
static inline int bench_bind_node(int node,char *why,size_t wn){ (void)node;
    snprintf(why,wn,"no CPU-affinity API on this platform; running UNPINNED"); return 0; }
static inline int bench_best_node(int *n){ if(n)*n=0; return -1; }
#endif

/* Warns on stderr unless the process is pinned to one CPU or bound by BENCH_PIN=node.
 * Matches bench_bind_node's `why` texts ("pinned to CPU", "failed"): change them together. */
static inline void bench_bind_warn(int st, const char *why)
{
    const char *pin = getenv("BENCH_PIN");
    if (strstr(why, "pinned to CPU")) return;
    if (st == 2 && pin && !strcmp(pin, "node") && !strstr(why, "failed")) return;
    fprintf(stderr, "WARNING: the timing process is not pinned to one CPU: %s\n"
                    "WARNING: it may migrate between CPUs or share one, so timings are less "
                    "repeatable.\n", why);
}
#endif /* BENCH_NUMA_BIND_H */
