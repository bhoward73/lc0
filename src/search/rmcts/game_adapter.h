#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "chess/gamestate.h"
#include "chess/position.h"

namespace lczero::rmcts {

constexpr int kPolicySize = 1858;
// Stable index into the adapter-owned state arena for one search.
using GameStateHandle = int32_t;

void InitializeRootState(const GameState& root_state);

GameStateHandle RootHandle();
const GameState& GetStateByHandle(GameStateHandle handle);

std::vector<int> GetValidActionIds(const GameState& state);
std::optional<Move> FindMoveForAction(const GameState& state, int action_id);

GameStateHandle CreateChildState(GameStateHandle parent_handle, Move move);

float ScoreFromWhitePerspective(GameResult result);

}  // namespace lczero::rmcts
