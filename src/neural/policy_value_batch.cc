/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "neural/policy_value_batch.h"

#include "neural/encoder.h"
#include "neural/register.h"

namespace lczero {

PolicyValueBatchEvaluator::PolicyValueBatchEvaluator(const OptionsDict& options,
                                                     bool white_pov_values)
    : white_pov_values_(white_pov_values),
      backend_(CreateMemCache(BackendManager::Get()->CreateFromParams(options),
                              options)) {}

PolicyValueBatch PolicyValueBatchEvaluator::Evaluate(
  const std::vector<GameState>& batch) {
  std::vector<std::vector<Position>> positions_storage;
  positions_storage.reserve(batch.size());

  std::vector<MoveList> legal_moves_storage;
  legal_moves_storage.reserve(batch.size());

  std::vector<EvalPosition> eval_positions;
  eval_positions.reserve(batch.size());

  for (const GameState& state : batch) {
    positions_storage.emplace_back(state.GetPositions());
    const ChessBoard& board = positions_storage.back().back().GetBoard();
    legal_moves_storage.emplace_back(board.GenerateLegalMoves());
    eval_positions.emplace_back(
        EvalPosition{positions_storage.back(), legal_moves_storage.back()});
  }

  std::vector<EvalResult> eval_results = backend_->EvaluateBatch(eval_positions);

  PolicyValueBatch output;
  output.pi.assign(batch.size(),
                   std::vector<float>(PolicyValueBatch::kPolicySize, 0.0f));
  output.v.resize(batch.size());

  for (size_t state_idx = 0; state_idx < batch.size(); ++state_idx) {
    const Position& current_position = positions_storage[state_idx].back();
    float value = eval_results[state_idx].q;
    if (white_pov_values_ && current_position.IsBlackToMove()) value = -value;
    output.v[state_idx] = value;

    const auto& legal_moves = legal_moves_storage[state_idx];
    const auto& legal_policy = eval_results[state_idx].p;
    for (size_t move_idx = 0; move_idx < legal_moves.size(); ++move_idx) {
      const uint16_t policy_idx = MoveToNNIndex(legal_moves[move_idx], 0);
      output.pi[state_idx][policy_idx] = legal_policy[move_idx];
    }
  }

  return output;
}

}  // namespace lczero
