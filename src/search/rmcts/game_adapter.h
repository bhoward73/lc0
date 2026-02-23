#pragma once

#include <optional>
#include <vector>

#include "chess/gamestate.h"
#include "chess/position.h"

namespace lczero::rmcts {

constexpr int kPolicySize = 1858;
constexpr int kEncodedStateSize = 1;

void InitializeRootState(const GameState& root_state);

int DecodeHandle(const float* g);
const GameState& GetStateByHandle(int handle);

std::vector<int> GetValidActionIds(const GameState& state);
std::optional<Move> FindMoveForAction(const GameState& state, int action_id);

int CreateChildState(int parent_handle, Move move);

float ScoreFromWhitePerspective(GameResult result);

}  // namespace lczero::rmcts
