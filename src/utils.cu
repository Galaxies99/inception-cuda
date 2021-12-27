# include "utils.h"


double init_rand(void) {
    return 0.05f - double(rand()) / double(RAND_MAX) / 10;
}

