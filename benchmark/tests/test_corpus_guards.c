/* test_corpus_guards.c -- test_corpus_guards <root> <family> <law> [max_points]: loads one
 * corpus law and prints whether corpus_dataset_load accepts it; the exit status is the load
 * status.  Run by benchmark/tests/run_tests.sh on intact and damaged laws. */
#include <stdio.h>
#include <stdlib.h>
#include "corpus_dataset.h"
int main(int c,char**v){
    corpus_dataset cs;
    int r = corpus_dataset_load(&cs, v[1], v[2], v[3], (size_t)(c>4?atol(v[4]):0));
    printf("  -> load returned %d%s", r, r?"  (REFUSED)\n":"  (accepted)\n");
    if(!r){ printf("     n=%zu stride=%zu file_points=%zu min_bits=%d\n",
                   cs.set.count, cs.stride, cs.file_points, cs.min_bits);
            corpus_dataset_free(&cs); }
    return r;
}
