#pragma once

#include <cstdint>

// Opaque state id managed by the lc0 RMCTS adapter.
using GameStateHandle = int32_t;

int numActions(void);

GameStateHandle rootState(void);

float playerId(GameStateHandle state);

int gameEnded(float* const terminal_score, GameStateHandle state);

int isValidAction(GameStateHandle state, int const action);

int getValidActions(int* const actions, GameStateHandle state);

int nextState(GameStateHandle* const child_state, GameStateHandle state,
			  int const action);

void printGame(GameStateHandle state);

