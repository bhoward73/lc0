#pragma once

#include <stdint.h>
#include <math.h>

#include "game.h"

void assign_simulations(int* sims_per_action, int budget, float* pi, int n);
void new_policy_common_ucb_Newton(float* pi1, int n, float* Q, float c, float* pi0, int T);

typedef struct {
	int num_lanes;
	int numSims;
	float c_puct;

	float* new_policy;
	float* new_value;	
	
	GameStateHandle* G;
	float* policy;
	float* value;
	float* Q;
	float* N;
	int32_t* child;
	int32_t* parent;
	int32_t* a0;
	int32_t* sims;
	int32_t* sims_remaining;
	int32_t* inference_stack;
	int32_t* inference_stack_size;
	int32_t* new_stack;
	int32_t* new_stack_size;
	int32_t* num_completed;
	int32_t* row_count;
} MCTS_new_t;

// Caller-owned scratch/state buffers. The RMCTS core stores pointers to these
// arrays and mutates them in place.
void* MCTS_init(int const num_lanes,  
				int const numSims,
				float const c_puct,
				float* new_policy,
				float* new_value,
				GameStateHandle* G,
				float* policy,
				float* value,
				float* Q,
				float* N,
				int32_t* child,
				int32_t* parent,
				int32_t* a0,
				int32_t* sims,
				int32_t* sims_remaining,
				int32_t* inference_stack,
				int32_t* inference_stack_size,
				int32_t* new_stack,
				int32_t* new_stack_size,
				int32_t* num_completed,
				int32_t* row_count);


void MCTS_free(void* mcts);


// Drains new_stack and appends newly discovered rows to inference_stack.
void MCTS_flush_new_stack(void* const mcts);


