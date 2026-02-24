#pragma once

#include <stdint.h>


#define Knuth_a 6364136223846793005ull
#define Knuth_c 1442695040888963407ull

// Basic RNG helpers used by RMCTS core sampling routines.
void Knuth_init(uint64_t seed);
uint32_t Knuth_lrand(void);
float Knuth_drand(void);

int random_index(const float* const p, int const len_p);

