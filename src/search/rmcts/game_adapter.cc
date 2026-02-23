#include "search/rmcts/game_adapter.h"

#include <cassert>
#include <cstdio>
#include <vector>

#include "neural/encoder.h"

namespace lczero::rmcts {
namespace {

struct AdapterStore {
  std::vector<GameState> states;
  int root_handle = 0;
};

AdapterStore& Store() {
  static AdapterStore store;
  return store;
}

std::vector<Move> GenerateLegalMoves(const GameState& state) {
  return state.CurrentPosition().GetBoard().GenerateLegalMoves();
}

}  // namespace

void InitializeRootState(const GameState& root_state) {
  auto& store = Store();
  store.states.clear();
  store.states.push_back(root_state);
  store.root_handle = 0;
}

int DecodeHandle(const float* g) {
  return static_cast<int>(g[0]);
}

const GameState& GetStateByHandle(int handle) {
  const auto& store = Store();
  assert(handle >= 0 && handle < static_cast<int>(store.states.size()));
  return store.states[handle];
}

std::vector<int> GetValidActionIds(const GameState& state) {
  std::vector<int> actions;
  const auto legal_moves = GenerateLegalMoves(state);
  actions.reserve(legal_moves.size());
  for (const Move move : legal_moves) {
    actions.push_back(MoveToNNIndex(move, 0));
  }
  return actions;
}

std::optional<Move> FindMoveForAction(const GameState& state, int action_id) {
  const auto legal_moves = GenerateLegalMoves(state);
  for (const Move move : legal_moves) {
    if (MoveToNNIndex(move, 0) == action_id) return move;
  }
  return std::nullopt;
}

int CreateChildState(int parent_handle, Move move) {
  auto& store = Store();
  assert(parent_handle >= 0 && parent_handle < static_cast<int>(store.states.size()));
  GameState child = store.states[parent_handle];
  child.moves.push_back(move);
  store.states.push_back(std::move(child));
  return static_cast<int>(store.states.size()) - 1;
}

float ScoreFromWhitePerspective(GameResult result) {
  if (result == GameResult::WHITE_WON) return 1.0f;
  if (result == GameResult::BLACK_WON) return -1.0f;
  return 0.0f;
}

}  // namespace lczero::rmcts

int numActions(void) { return lczero::rmcts::kPolicySize; }

int gameLength(void) { return lczero::rmcts::kEncodedStateSize; }

int inputLength(void) { return lczero::rmcts::kEncodedStateSize; }

void rootState(float* const g) {
  g[0] = 0.0f;
}

float playerId(const float* const g) {
  const int handle = lczero::rmcts::DecodeHandle(g);
  const auto& state = lczero::rmcts::GetStateByHandle(handle);
  return state.CurrentPosition().IsBlackToMove() ? -1.0f : 1.0f;
}

void inputNetwork(float* const x, const float* const g) {
  x[0] = g[0];
}

int gameEnded(float* const terminal_score, const float* const g) {
  const int handle = lczero::rmcts::DecodeHandle(g);
  const auto& state = lczero::rmcts::GetStateByHandle(handle);
  lczero::PositionHistory history(state.GetPositions());
  const lczero::GameResult result = history.ComputeGameResult();
  if (result == lczero::GameResult::UNDECIDED) {
    *terminal_score = 0.0f;
    return 0;
  }
  *terminal_score = lczero::rmcts::ScoreFromWhitePerspective(result);
  return 1;
}

int isValidAction(const float* const g, int const a) {
  const int handle = lczero::rmcts::DecodeHandle(g);
  const auto& state = lczero::rmcts::GetStateByHandle(handle);
  return lczero::rmcts::FindMoveForAction(state, a).has_value() ? 1 : 0;
}

int getValidActions(int* const actions, const float* const g) {
  const int handle = lczero::rmcts::DecodeHandle(g);
  const auto& state = lczero::rmcts::GetStateByHandle(handle);
  const auto ids = lczero::rmcts::GetValidActionIds(state);
  for (size_t i = 0; i < ids.size(); ++i) {
    actions[i] = ids[i];
  }
  return static_cast<int>(ids.size());
}

int nextState(float* const ga, const float* const g, const int a) {
  const int handle = lczero::rmcts::DecodeHandle(g);
  const auto& state = lczero::rmcts::GetStateByHandle(handle);
  const auto move = lczero::rmcts::FindMoveForAction(state, a);
  if (!move.has_value()) return -1;
  const int child_handle = lczero::rmcts::CreateChildState(handle, *move);
  ga[0] = static_cast<float>(child_handle);
  float terminal_score = 0.0f;
  return gameEnded(&terminal_score, ga);
}

void printGame(const float* const g) {
  const int handle = lczero::rmcts::DecodeHandle(g);
  const auto& state = lczero::rmcts::GetStateByHandle(handle);
  std::printf("%s\n", PositionToFen(state.CurrentPosition()).c_str());
}
