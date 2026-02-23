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

#pragma once

#include <memory>
#include <vector>

#include "chess/gamestate.h"
#include "neural/memcache.h"
#include "utils/optionsdict.h"

namespace lczero {

// Batch policy/value output.
// - pi: N x 1858, where rows are states and columns are MoveToNNIndex IDs.
// - v:  N x 1, white-relative value in [-1, 1] when white_pov_values=true,
//       otherwise side-to-move-relative value.
struct PolicyValueBatch {
  static constexpr int kPolicySize = 1858;

  std::vector<std::vector<float>> pi;
  std::vector<float> v;
};

// Reusable evaluator for batching NN policy/value queries.
class PolicyValueBatchEvaluator {
 public:
  explicit PolicyValueBatchEvaluator(const OptionsDict& options,
                                     bool white_pov_values = true);

  PolicyValueBatch Evaluate(const std::vector<GameState>& batch);
  size_t GetBackendMaxBatchSize() const;

 private:
  const bool white_pov_values_;
  std::unique_ptr<CachingBackend> backend_;
};

}  // namespace lczero
