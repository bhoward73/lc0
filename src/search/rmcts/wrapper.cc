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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/batchsplit.h"
#include "neural/encoder.h"
#include "search/register.h"
#include "search/rmcts/game_adapter.h"
#include "search/search.h"
#include "src/rmcts/src/c/RMCTS.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kRmctsNumSimsId{
    "rmcts-num-sims", "RMCTSNumSims", "Number of RMCTS simulations."};
const OptionId kRmctsCpuctId{
    "rmcts-cpuct", "RMCTSCpuct", "RMCTS cpuct parameter."};
const OptionId kRmctsEpochsId{
  "rmcts-epochs", "RMCTSEpochs",
  "Number of RMCTS epochs to run (splits RMCTSNumSims across epochs)."};
const OptionId kRmctsPosteriorWeightId{
  "rmcts-posterior-weight", "RMCTSPosteriorWeight",
  "Blend weight in [0,1] for posterior vs prior policy across epochs."};
const OptionId kRmctsChunkSizeId{
  "rmcts-chunk-sims", "RMCTSChunkSims",
  "RMCTS simulations per checkpoint chunk for time-aware stopping."};
const OptionId kRmctsPrewarmOnNewGameId{
  "rmcts-prewarm-on-newgame", "RMCTSPrewarmOnNewGame",
  "Run one backend warmup evaluation on ucinewgame before first search."};

struct RmctsRunResult {
  Move best_move;
  float value = 0.0f;
  int64_t nodes = 0;
  int64_t elapsed_ms = 0;
  std::vector<ThinkingInfo> policy_infos;
};

struct CachedEval {
  std::vector<std::pair<uint16_t, float>> sparse_policy;
  float value = 0.0f;
};

class RmctsSearch : public SearchBase {
 public:
  RmctsSearch(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}

  ~RmctsSearch() override {
    stop_requested_.store(true, std::memory_order_relaxed);
    JoinWorker();
  }

  void SetPosition(const GameState& game_state) final { game_state_ = game_state; }

  void NewGame() final {
    JoinWorker();
    if (!options_->Get<bool>(kRmctsPrewarmOnNewGameId)) return;
    TryPrewarmBackend(GameState{});
  }

  void StartSearch(const GoParams& go_params) final {
    JoinWorker();
    responded_bestmove_.store(false, std::memory_order_relaxed);
    stop_requested_.store(false, std::memory_order_relaxed);
    abort_requested_.store(false, std::memory_order_relaxed);
    force_respond_.store(false, std::memory_order_relaxed);
    search_done_.store(false, std::memory_order_relaxed);

    const GameState search_state = game_state_;
    worker_ = std::thread([this, search_state, go_params]() {
      const RmctsRunResult result = RunRmcts(search_state, go_params);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        bestmove_ = result.best_move;
      }
      const int elapsed_nps =
          (result.elapsed_ms > 0)
              ? static_cast<int>((result.nodes * 1000LL) / result.elapsed_ms)
              : -1;
      std::vector<ThinkingInfo> infos = {{
          .depth = 1,
          .seldepth = 1,
          .time = result.elapsed_ms,
          .nodes = result.nodes,
          .nps = elapsed_nps,
          .score = 90 * std::tan(1.5637541897 * result.value),
        .wdl = ThinkingInfo::WDL{
          static_cast<int>(std::round(500 * (1 + result.value))),
          0,
          static_cast<int>(std::round(500 * (1 - result.value)))},
      }};
      infos.insert(infos.end(), result.policy_infos.begin(), result.policy_infos.end());
      uci_responder_->OutputThinkingInfo(&infos);

      if (!abort_requested_.load(std::memory_order_relaxed) &&
          (force_respond_.load(std::memory_order_relaxed) ||
           (!go_params.infinite && !go_params.ponder))) {
        RespondBestMove();
      }

      search_done_.store(true, std::memory_order_relaxed);
      cv_.notify_all();
    });
  }

  void StartClock() final {}

  void WaitSearch() final {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() {
      return search_done_.load(std::memory_order_relaxed);
    });
    lock.unlock();
    JoinWorker();
  }

  void StopSearch() final {
    force_respond_.store(true, std::memory_order_relaxed);
    stop_requested_.store(true, std::memory_order_relaxed);
    if (search_done_.load(std::memory_order_relaxed)) {
      RespondBestMove();
    }
  }

  void AbortSearch() final {
    abort_requested_.store(true, std::memory_order_relaxed);
    stop_requested_.store(true, std::memory_order_relaxed);
  }

  void SetBackend(Backend* backend) override {
    batchsplit_backend_ = CreateBatchSplitingBackend(backend);
    backend_ = batchsplit_backend_.get();
    prewarmed_.store(false, std::memory_order_relaxed);
  }

 private:
  void TryPrewarmBackend(const GameState& state) {
    if (backend_ == nullptr) return;
    if (prewarmed_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(prewarm_mutex_);
    if (prewarmed_.load(std::memory_order_relaxed)) return;

    const std::vector<Position> positions = state.GetPositions();
    if (positions.empty()) return;
    const MoveList legal_moves = positions.back().GetBoard().GenerateLegalMoves();
    if (legal_moves.empty()) return;

    std::vector<EvalPosition> eval_positions;
    eval_positions.push_back(EvalPosition{positions, legal_moves});
    backend_->EvaluateBatch(eval_positions);
    prewarmed_.store(true, std::memory_order_release);
  }

  std::optional<int64_t> ComputeTimeBudgetMs(const GoParams& go_params,
                                             const GameState& game_state) const {
    if (go_params.infinite || go_params.ponder) return std::nullopt;

    constexpr int64_t kSafetyMarginMs = 5;

    if (go_params.movetime) {
      return std::max<int64_t>(1, *go_params.movetime - kSafetyMarginMs);
    }

    const bool is_black = game_state.CurrentPosition().IsBlackToMove();
    const auto side_time = is_black ? go_params.btime : go_params.wtime;
    const auto side_inc = is_black ? go_params.binc : go_params.winc;
    if (!side_time) return std::nullopt;

    const int64_t increment = side_inc.value_or(0);
    const int moves_to_go = std::max(1, go_params.movestogo.value_or(30));
    int64_t budget = *side_time / moves_to_go + increment / 2;
    budget = std::min(budget, *side_time - kSafetyMarginMs);
    return std::max<int64_t>(1, budget);
  }

  bool DeadlineReached(
      const std::optional<std::chrono::steady_clock::time_point>& deadline)
      const {
    return deadline && std::chrono::steady_clock::now() >= *deadline;
  }

  void JoinWorker() {
    if (worker_.joinable()) worker_.join();
  }

  static std::string StateKey(const GameState& state) {
    return PositionToFen(state.CurrentPosition());
  }

  static void NormalizePolicy(std::vector<float>* pi) {
    float sum = 0.0f;
    for (const float p : *pi) sum += p;
    if (!(sum > 0.0f) || !std::isfinite(sum)) return;
    for (float& p : *pi) p /= sum;
  }

  static std::vector<std::pair<uint16_t, float>> DenseToSparse(
      const std::vector<float>& dense) {
    std::vector<std::pair<uint16_t, float>> sparse;
    sparse.reserve(64);
    for (size_t i = 0; i < dense.size(); ++i) {
      if (dense[i] > 0.0f && std::isfinite(dense[i])) {
        sparse.emplace_back(static_cast<uint16_t>(i), dense[i]);
      }
    }
    return sparse;
  }

  static void FillRowFromSparse(
      const std::vector<std::pair<uint16_t, float>>& sparse, int n,
      float* row_policy) {
    std::fill(row_policy, row_policy + n, 0.0f);
    for (const auto& [idx, val] : sparse) {
      if (idx < n) row_policy[idx] = val;
    }
  }

  static void FillUniformLegal(const MoveList& legal_moves, int n,
                               float* row_policy) {
    std::fill(row_policy, row_policy + n, 0.0f);
    if (legal_moves.empty()) return;
    const float p = 1.0f / static_cast<float>(legal_moves.size());
    for (const Move move : legal_moves) {
      row_policy[MoveToNNIndex(move, 0)] = p;
    }
  }

  void EvaluateRows(const std::vector<int32_t>& rows, int gamesize,
                    int n, std::vector<float>* G, std::vector<float>* policy,
                    std::vector<float>* value,
                    std::unordered_map<std::string, CachedEval>* eval_cache,
                    int* expensive_events_remaining) {
    std::vector<std::vector<Position>> positions_storage;
    positions_storage.reserve(rows.size());
    std::vector<MoveList> legal_moves_storage;
    legal_moves_storage.reserve(rows.size());
    std::vector<EvalPosition> eval_positions;
    eval_positions.reserve(rows.size());
    std::vector<size_t> pending_indices;
    pending_indices.reserve(rows.size());
    std::vector<std::string> state_keys(rows.size());

    for (size_t sample = 0; sample < rows.size(); ++sample) {
      const int32_t row = rows[sample];
      const int handle = rmcts::DecodeHandle(G->data() + row * gamesize);
      const GameState& state = rmcts::GetStateByHandle(handle);
      state_keys[sample] = StateKey(state);
      positions_storage.emplace_back(state.GetPositions());
      legal_moves_storage.emplace_back(
          positions_storage.back().back().GetBoard().GenerateLegalMoves());

      float* const row_policy = policy->data() + row * n;
      const auto cache_it = eval_cache->find(state_keys[sample]);
      if (cache_it != eval_cache->end()) {
        FillRowFromSparse(cache_it->second.sparse_policy, n, row_policy);
        value->at(row) = cache_it->second.value;
        continue;
      }

      if (*expensive_events_remaining <= 0) {
        FillUniformLegal(legal_moves_storage.back(), n, row_policy);
        value->at(row) = 0.0f;
        continue;
      }

      eval_positions.emplace_back(
          EvalPosition{positions_storage.back(), legal_moves_storage.back()});
      pending_indices.push_back(sample);
    }

    if (eval_positions.empty()) return;

    auto eval_results = backend_->EvaluateBatch(eval_positions);
    for (size_t eval_idx = 0; eval_idx < eval_results.size(); ++eval_idx) {
      const size_t sample = pending_indices[eval_idx];
      const int32_t row = rows[sample];
      float* const row_policy = policy->data() + row * n;
      std::fill(row_policy, row_policy + n, 0.0f);
      for (size_t move_idx = 0; move_idx < legal_moves_storage[sample].size();
           ++move_idx) {
        const uint16_t idx = MoveToNNIndex(legal_moves_storage[sample][move_idx], 0);
        row_policy[idx] = eval_results[eval_idx].p[move_idx];
      }

      value->at(row) = eval_results[eval_idx].q;
      (*eval_cache)[state_keys[sample]] = CachedEval{
          .sparse_policy = DenseToSparse(std::vector<float>(row_policy, row_policy + n)),
          .value = value->at(row),
      };
      (*expensive_events_remaining)--;
    }
  }

  RmctsRunResult RunRmcts(const GameState& game_state, const GoParams& go_params) {
    const auto search_start = std::chrono::steady_clock::now();
    constexpr int kNumLanes = 1;
    constexpr int gamesize = rmcts::kEncodedStateSize;
    constexpr int n = rmcts::kPolicySize;

    const int num_sims = std::max(1, options_->Get<int>(kRmctsNumSimsId));
    const int chunk_sims = std::max(1, options_->Get<int>(kRmctsChunkSizeId));
    const float c_puct = options_->Get<float>(kRmctsCpuctId);
    const int epochs = std::max(1, options_->Get<int>(kRmctsEpochsId));
    const float posterior_weight = std::clamp(
        options_->Get<float>(kRmctsPosteriorWeightId), 0.0f, 1.0f);
    const int effective_epochs = std::max(epochs, (num_sims + chunk_sims - 1) / chunk_sims);
    const int epoch_sims = std::max(1, (num_sims + effective_epochs - 1) / effective_epochs);
    const int capacity = kNumLanes * epoch_sims;
    int expensive_events_remaining = num_sims;
    int sims_remaining_global = num_sims;

    const auto budget_ms = ComputeTimeBudgetMs(go_params, game_state);
    const auto deadline = budget_ms
                              ? std::optional<std::chrono::steady_clock::time_point>(
                                    search_start + std::chrono::milliseconds(*budget_ms))
                              : std::nullopt;

    std::unordered_map<std::string, CachedEval> eval_cache;
    std::vector<float> root_posterior_running(n, 0.0f);
    std::vector<float> root_prior(n, 0.0f);
    std::vector<float> final_root_policy(n, 0.0f);
    float final_root_value = 0.0f;

    rmcts::InitializeRootState(game_state);
    const std::string root_key = StateKey(game_state);

    int completed_epochs = 0;
    int32_t last_total_sims = 0;
    for (int epoch = 0; epoch < effective_epochs && sims_remaining_global > 0;
         ++epoch) {
      if (stop_requested_.load(std::memory_order_relaxed) ||
          DeadlineReached(deadline)) {
        break;
      }

      const int this_epoch_sims = std::min(epoch_sims, sims_remaining_global);
      sims_remaining_global -= this_epoch_sims;
      std::vector<float> new_policy(kNumLanes * n, 0.0f);
      std::vector<float> new_value(kNumLanes, 0.0f);
      std::vector<float> G(capacity * gamesize, 0.0f);
      std::vector<float> policy(capacity * n, 0.0f);
      std::vector<float> value(capacity, 0.0f);
      std::vector<float> Q(capacity * n, 0.0f);
      std::vector<float> N(capacity * n, 0.0f);

      std::vector<int32_t> parent(capacity, 0);
      std::vector<int32_t> a0(capacity, 0);
      std::vector<int32_t> sims(capacity, 0);
      std::vector<int32_t> sims_remaining(capacity, 0);
      std::vector<int32_t> inference_stack(capacity, 0);
      int32_t inference_stack_size = 0;
      std::vector<int32_t> new_stack(capacity, 0);
      int32_t new_stack_size = 0;
      int32_t num_completed = 0;
      int32_t row_count = 1;

      G[0] = 0.0f;
      parent[0] = -1;
      a0[0] = -1;
      sims[0] = this_epoch_sims;
      sims_remaining[0] = this_epoch_sims;

      EvaluateRows({0}, gamesize, n, &G, &policy, &value, &eval_cache,
                   &expensive_events_remaining);

      if (epoch == 0) {
        for (int i = 0; i < n; ++i) root_prior[i] = policy[i];
      }

      if (epoch > 0) {
        auto it = eval_cache.find(root_key);
        if (it != eval_cache.end()) {
          std::vector<float> root_prior(n, 0.0f);
          FillRowFromSparse(it->second.sparse_policy, n, root_prior.data());
          for (int i = 0; i < n; ++i) {
            policy[i] = (1.0f - posterior_weight) * root_prior[i] +
                        posterior_weight * root_posterior_running[i];
          }
          NormalizePolicy(&policy);
        }
      }

      new_stack[0] = 0;
      new_stack_size = 1;

      void* mcts = MCTS_init(kNumLanes, epoch_sims, c_puct, new_policy.data(),
                             new_value.data(), G.data(), policy.data(),
                             value.data(), Q.data(), N.data(), parent.data(),
                             a0.data(), sims.data(), sims_remaining.data(),
                             inference_stack.data(), &inference_stack_size,
                             new_stack.data(), &new_stack_size, &num_completed,
                             &row_count);

      while (num_completed < kNumLanes) {
        if (stop_requested_.load(std::memory_order_relaxed) ||
            DeadlineReached(deadline)) {
          break;
        }
        MCTS_flush_new_stack(mcts);
        if (inference_stack_size == 0) break;
        std::vector<int32_t> rows(inference_stack.begin(),
                                  inference_stack.begin() + inference_stack_size);
        inference_stack_size = 0;
        EvaluateRows(rows, gamesize, n, &G, &policy, &value, &eval_cache,
                     &expensive_events_remaining);
        for (const int32_t row : rows) {
          new_stack[new_stack_size++] = row;
        }
      }

      MCTS_free(mcts);

      last_total_sims = sims[0] - sims_remaining[0];
      if (last_total_sims <= 0) break;

      ++completed_epochs;
      for (int i = 0; i < n; ++i) {
        root_posterior_running[i] =
            (root_posterior_running[i] * static_cast<float>(completed_epochs - 1) +
             new_policy[i]) /
            static_cast<float>(completed_epochs);
      }
      NormalizePolicy(&root_posterior_running);

      final_root_policy = new_policy;
      final_root_value = new_value[0];

      {
        std::lock_guard<std::mutex> lock(mutex_);
        Move best_so_far;
        float best_prob = -1.0f;
        const MoveList legal_moves =
            game_state.CurrentPosition().GetBoard().GenerateLegalMoves();
        for (const Move move : legal_moves) {
          const int idx = MoveToNNIndex(move, 0);
          if (final_root_policy[idx] > best_prob) {
            best_prob = final_root_policy[idx];
            best_so_far = move;
          }
        }
        if (!best_so_far.is_null()) bestmove_ = best_so_far;
      }
    }

    MoveList legal_moves = game_state.CurrentPosition().GetBoard().GenerateLegalMoves();
    Move best_move;
    float best_p = -1.0f;
    std::vector<ThinkingInfo> policy_infos;
    policy_infos.reserve(legal_moves.size());

    struct Row {
      Move move;
      float prior;
      float posterior;
      int visits;
    };
    std::vector<Row> rows;
    rows.reserve(legal_moves.size());

    int total_visits = 0;
    for (const Move move : legal_moves) {
      const int idx = MoveToNNIndex(move, 0);
      const float p = std::max(0.0f, final_root_policy[idx]);
      const int visits = static_cast<int>(std::round(p * std::max<int64_t>(1, num_sims)));
      rows.push_back(Row{move, std::max(0.0f, root_prior[idx]), p, visits});
      total_visits += visits;
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
      if (a.prior != b.prior) return a.prior > b.prior;
      return a.move.ToString(false) < b.move.ToString(false);
    });

    for (const auto& row : rows) {
      std::ostringstream oss;
      oss << row.move.ToString(false)
          << " N: " << row.visits
          << " (P: " << std::fixed << std::setprecision(2)
          << row.prior * 100.0f << "%)";
      policy_infos.push_back(ThinkingInfo{.comment = oss.str()});
    }

    for (const Move move : legal_moves) {
      const int idx = MoveToNNIndex(move, 0);
      const float p = final_root_policy[idx];
      if (p > best_p) {
        best_p = p;
        best_move = move;
      }
    }

    if (best_move.is_null() && !legal_moves.empty()) {
      best_move = legal_moves.front();
    }

    const auto elapsed_ms = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - search_start)
            .count());

    return RmctsRunResult{best_move, final_root_value,
                          static_cast<int64_t>(num_sims - expensive_events_remaining),
                          elapsed_ms, std::move(policy_infos)};
  }

  void RespondBestMove() {
    if (responded_bestmove_.exchange(true, std::memory_order_relaxed)) return;
    Move move;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      move = bestmove_;
    }
    if (move.is_null()) {
      const MoveList legal = game_state_.CurrentPosition().GetBoard().GenerateLegalMoves();
      if (!legal.empty()) move = legal.front();
    }
    BestMoveInfo info{move};
    if (game_state_.CurrentPosition().IsBlackToMove()) {
      info.bestmove.Flip();
    } else if (!info.ponder.is_null()) {
      info.ponder.Flip();
    }
    uci_responder_->OutputBestMove(&info);
  }

  std::atomic<bool> responded_bestmove_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> abort_requested_{false};
  std::atomic<bool> force_respond_{false};
  std::atomic<bool> search_done_{true};
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> prewarmed_{false};
  std::mutex prewarm_mutex_;
  std::unique_ptr<Backend> batchsplit_backend_;
  const OptionsDict* options_ = nullptr;
  GameState game_state_;
  Move bestmove_;
};

class RmctsFactory : public SearchFactory {
 public:
  std::string_view GetName() const override { return "rmcts"; }

  void PopulateParams(OptionsParser* parser) const override {
    parser->Add<IntOption>(kRmctsNumSimsId, 1, 1000000) = 800;
    parser->Add<FloatOption>(kRmctsCpuctId, 0.01f, 1000.0f) = 1.2f;
    parser->Add<IntOption>(kRmctsEpochsId, 1, 1024) = 2;
    parser->Add<IntOption>(kRmctsChunkSizeId, 1, 1000000) = 64;
    parser->Add<BoolOption>(kRmctsPrewarmOnNewGameId) = true;
    parser->Add<FloatOption>(kRmctsPosteriorWeightId, 0.0f, 1.0f) = 0.5f;
  }

  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict* options) const override {
    return std::make_unique<RmctsSearch>(responder, options);
  }
};

REGISTER_SEARCH(RmctsFactory)

}  // namespace
}  // namespace lczero
