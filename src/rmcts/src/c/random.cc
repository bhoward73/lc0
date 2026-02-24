#include <cstdint>
#include "random.h"

uint64_t Knuth_seed = 0ull;

void Knuth_init(uint64_t seed) {
  int i;
  Knuth_seed = seed;
  for(i=0;i<20;i++) {
    Knuth_seed *= Knuth_a;
    Knuth_seed += Knuth_c;
  }
}

uint32_t Knuth_lrand(void) {
  Knuth_seed *= Knuth_a;
  Knuth_seed += Knuth_c;
  return (uint32_t) (Knuth_seed >> 32);
}

float Knuth_drand(void) {
  Knuth_seed *= Knuth_a;
  Knuth_seed += Knuth_c;
  return ((float) (Knuth_seed >> 40)) / ((float) (1u << 24));
}

// Assumes p is a normalized probability distribution.
int random_index(const float* const p, int const len_p) {
  float s = 0.0;
  int i;
  float x;  
  x = Knuth_drand();
  for(i=0;i<len_p;i++) {
    s += p[i];
    if(s >= x) return i;
  }
  return len_p - 1;
}
