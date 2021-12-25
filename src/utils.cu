# include "utils.h"


double init_rand(void) {
    return 0.5f - double(rand()) / double(RAND_MAX);
}

