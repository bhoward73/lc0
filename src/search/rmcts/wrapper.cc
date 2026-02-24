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
#include <limits>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>

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
    PrintAndResetReuseGameSummary("shutdown");
  }

  void SetPosition(const GameState& game_state) final { game_state_ = game_state; }

  void NewGame() final {
    JoinWorker();
    PrintAndResetReuseGameSummary("ucinewgame");
    ClearPersistentTree();
    if (!options_->Get<bool>(kRmctsPrewarmOnNewGameId)) return;
    TryPrewarmBackend(GameState{});
  }

  void PrintAndResetReuseGameSummary(const char* reason) {
    if (reuse_game_searches_ == 0) return;
    const double branch_avg_pct =
        (reuse_game_branch_ratio_count_ > 0)
            ? (100.0 * reuse_game_branch_ratio_sum_ /
               static_cast<double>(reuse_game_branch_ratio_count_))
            : 0.0;
    const double branch_weighted_pct =
        (reuse_game_branch_old_rows_sum_ > 0.0)
            ? (100.0 * reuse_game_branch_new_rows_sum_ /
               reuse_game_branch_old_rows_sum_)
            : 0.0;
    std::cerr << "RMCTS_REUSE_SUMMARY reason=" << reason
              << " searches=" << reuse_game_searches_
              << " exact=" << reuse_game_exact_hits_
              << " branch=" << reuse_game_branch_hits_
              << " reset=" << reuse_game_resets_;
    if (reuse_game_branch_ratio_count_ > 0) {
      std::cerr << std::fixed << std::setprecision(1)
                << " branch_retain_avg_pct=" << branch_avg_pct
                << " branch_retain_weighted_pct=" << branch_weighted_pct
                << " branch_rows_old_sum=" << reuse_game_branch_old_rows_sum_
                << " branch_rows_new_sum=" << reuse_game_branch_new_rows_sum_
                << std::defaultfloat;
    }
    std::cerr << '\n';
    reuse_game_searches_ = 0;
    reuse_game_exact_hits_ = 0;
    reuse_game_branch_hits_ = 0;
    reuse_game_resets_ = 0;
    reuse_game_branch_ratio_sum_ = 0.0;
    reuse_game_branch_ratio_count_ = 0;
    reuse_game_branch_old_rows_sum_ = 0.0;
    reuse_game_branch_new_rows_sum_ = 0.0;
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

  static void NormalizePolicyRow(float* row_policy, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += row_policy[i];
    if (!(sum > 0.0f) || !std::isfinite(sum)) return;
    for (int i = 0; i < n; ++i) row_policy[i] /= sum;
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

  static std::string EscapeCsv(const std::string& text) {
    if (text.find_first_of(",\"\n") == std::string::npos) return text;
    std::string out;
    out.reserve(text.size() + 8);
    out.push_back('"');
    for (char c : text) {
      if (c == '"') out.push_back('"');
      out.push_back(c);
    }
    out.push_back('"');
    return out;
  }

  void AppendChunkTraceCsv(int chunk_idx, const Position& root_pos,
                           const std::vector<float>& root_prior,
                           const std::vector<float>& posterior,
                           const std::vector<float>& root_q,
                           const std::vector<float>& root_n) const {
    const char* trace_path = std::getenv("RMCTS_TRACE_CSV");
    if (!trace_path || !trace_path[0]) return;

    const bool write_header = (chunk_idx == 0);
    std::ofstream out(trace_path, write_header ? std::ios::out : std::ios::app);
    if (!out.is_open()) return;

    if (write_header) {
      out << "chunk,fen,move,prior,posterior,q,n_total\n";
    }

    const MoveList legal_moves = root_pos.GetBoard().GenerateLegalMoves();
    for (const Move move : legal_moves) {
      const int idx = MoveToNNIndex(move, 0);
      Move display_move = move;
      if (root_pos.IsBlackToMove()) {
        display_move.Flip();
      }
      out << chunk_idx << ','
          << EscapeCsv(PositionToFen(root_pos)) << ','
          << display_move.ToString(false) << ','
          << std::fixed << std::setprecision(6)
          << std::max(0.0f, root_prior[idx]) << ','
          << std::max(0.0f, posterior[idx]) << ','
          << root_q[idx] << ','
          << std::max(0.0f, root_n[idx]) << '\n';
    }
  }

  void EvaluateRows(const std::vector<int32_t>& rows, int n,
                    std::vector<rmcts::GameStateHandle>* state_handles,
                    std::vector<float>* policy,
                    std::vector<float>* value,
                    std::unordered_map<std::string, CachedEval>* eval_cache,
                    int* expensive_events_remaining) {
    // For each requested row:
    // - reuse cached eval when available,
    // - otherwise run backend eval if budget remains,
    // - otherwise fall back to uniform over legal moves.
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
      const int handle = state_handles->at(row);
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

  void ClearPersistentTree() {
    has_persistent_tree_ = false;
    current_root_key_.clear();
    eval_cache_.clear();
    state_handles_.clear();
    policy_.clear();
    value_.clear();
    Q_.clear();
    N_.clear();
    child_.clear();
    parent_.clear();
    a0_.clear();
    sims_.clear();
    sims_remaining_.clear();
    inference_stack_.clear();
    new_stack_.clear();
    capacity_ = 0;
    row_count_ = 0;
  }

  void EnsurePersistentCapacity(int required_rows, int n) {
    if (required_rows <= capacity_) return;
    int new_capacity = std::max(2, capacity_);
    while (new_capacity < required_rows) new_capacity *= 2;

    state_handles_.resize(new_capacity, 0);
    policy_.resize(new_capacity * n, 0.0f);
    value_.resize(new_capacity, 0.0f);
    Q_.resize(new_capacity * n, 0.0f);
    N_.resize(new_capacity * n, 0.0f);
    child_.resize(new_capacity * n, -1);
    parent_.resize(new_capacity, 0);
    a0_.resize(new_capacity, 0);
    sims_.resize(new_capacity, 0);
    sims_remaining_.resize(new_capacity, 0);
    inference_stack_.resize(new_capacity, 0);
    new_stack_.resize(new_capacity, 0);
    capacity_ = new_capacity;
  }

  void ResetPersistentTree(const GameState& root_state, int initial_capacity,
                           int n) {
    rmcts::InitializeRootState(root_state);

    capacity_ = std::max(2, initial_capacity);
    state_handles_.assign(capacity_, 0);
    policy_.assign(capacity_ * n, 0.0f);
    value_.assign(capacity_, 0.0f);
    Q_.assign(capacity_ * n, 0.0f);
    N_.assign(capacity_ * n, 0.0f);
    child_.assign(capacity_ * n, -1);
    parent_.assign(capacity_, 0);
    a0_.assign(capacity_, 0);
    sims_.assign(capacity_, 0);
    sims_remaining_.assign(capacity_, 0);
    inference_stack_.assign(capacity_, 0);
    new_stack_.assign(capacity_, 0);

    row_count_ = 1;
    state_handles_[0] = rmcts::RootHandle();
    parent_[0] = -1;
    a0_[0] = -1;

    current_root_state_ = root_state;
    current_root_key_ = StateKey(root_state);
    eval_cache_.clear();
    has_persistent_tree_ = true;
  }

  bool RestrictToBranch(int action_id, const GameState& new_root_state, int n) {
    if (!has_persistent_tree_ || row_count_ <= 0) return false;
    if (action_id < 0 || action_id >= n) return false;

    const int32_t old_total_rows = row_count_;

    const int32_t old_root_child = child_[action_id];
    if (old_root_child < 0 || old_root_child >= row_count_) return false;

    std::vector<int32_t> order;
    order.reserve(row_count_);
    std::unordered_map<int32_t, int32_t> remap;
    remap.reserve(static_cast<size_t>(row_count_));
    std::vector<int32_t> stack;
    stack.push_back(old_root_child);

    while (!stack.empty()) {
      const int32_t old_row = stack.back();
      stack.pop_back();
      if (old_row < 0 || old_row >= row_count_) continue;
      if (remap.find(old_row) != remap.end()) continue;
      const int32_t new_row = static_cast<int32_t>(order.size());
      remap.emplace(old_row, new_row);
      order.push_back(old_row);

      const int base = old_row * n;
      for (int a = 0; a < n; ++a) {
        const int32_t c = child_[base + a];
        if (c >= 0 && c < row_count_ && remap.find(c) == remap.end()) {
          stack.push_back(c);
        }
      }
    }

    if (order.empty()) return false;

    const int32_t new_row_count = static_cast<int32_t>(order.size());
    const int new_capacity = std::max(capacity_, std::max(2, new_row_count * 2));

    std::vector<rmcts::GameStateHandle> state_handles_new(new_capacity, 0);
    std::vector<float> policy_new(new_capacity * n, 0.0f);
    std::vector<float> value_new(new_capacity, 0.0f);
    std::vector<float> Q_new(new_capacity * n, 0.0f);
    std::vector<float> N_new(new_capacity * n, 0.0f);
    std::vector<int32_t> child_new(new_capacity * n, -1);
    std::vector<int32_t> parent_new(new_capacity, 0);
    std::vector<int32_t> a0_new(new_capacity, 0);
    std::vector<int32_t> sims_new(new_capacity, 0);
    std::vector<int32_t> sims_remaining_new(new_capacity, 0);
    std::vector<int32_t> inference_stack_new(new_capacity, 0);
    std::vector<int32_t> new_stack_new(new_capacity, 0);

    for (int32_t new_row = 0; new_row < new_row_count; ++new_row) {
      const int32_t old_row = order[new_row];
      state_handles_new[new_row] = state_handles_[old_row];
      value_new[new_row] = value_[old_row];
      sims_new[new_row] = sims_[old_row];
      sims_remaining_new[new_row] = sims_remaining_[old_row];

      std::copy_n(policy_.begin() + old_row * n, n,
                  policy_new.begin() + new_row * n);
      std::copy_n(Q_.begin() + old_row * n, n, Q_new.begin() + new_row * n);
      std::copy_n(N_.begin() + old_row * n, n, N_new.begin() + new_row * n);

      const int32_t old_parent = parent_[old_row];
      const auto p_it = remap.find(old_parent);
      parent_new[new_row] = (p_it != remap.end()) ? p_it->second : -1;
      a0_new[new_row] = a0_[old_row];

      for (int a = 0; a < n; ++a) {
        const int32_t old_child = child_[old_row * n + a];
        const auto c_it = remap.find(old_child);
        child_new[new_row * n + a] =
            (c_it != remap.end()) ? c_it->second : -1;
      }
    }

    parent_new[0] = -1;
    a0_new[0] = -1;

    state_handles_.swap(state_handles_new);
    policy_.swap(policy_new);
    value_.swap(value_new);
    Q_.swap(Q_new);
    N_.swap(N_new);
    child_.swap(child_new);
    parent_.swap(parent_new);
    a0_.swap(a0_new);
    sims_.swap(sims_new);
    sims_remaining_.swap(sims_remaining_new);
    inference_stack_.swap(inference_stack_new);
    new_stack_.swap(new_stack_new);
    capacity_ = new_capacity;
    row_count_ = new_row_count;

    current_root_state_ = new_root_state;
    current_root_key_ = StateKey(new_root_state);

    last_restrict_action_id_ = action_id;
    last_restrict_old_rows_ = old_total_rows;
    last_restrict_new_rows_ = new_row_count;
    return true;
  }

  enum class ReuseKind {
    kReset,
    kExact,
    kBranch,
  };

  struct ReuseInfo {
    ReuseKind kind = ReuseKind::kReset;
    int action_id = -1;
    int32_t old_rows = 0;
    int32_t new_rows = 0;
    int steps = 0;
  };

  ReuseInfo TryReuseForRoot(const GameState& new_root_state, int n) {
    ReuseInfo info;
    if (!has_persistent_tree_ || row_count_ <= 0) return info;

    const std::string next_key = StateKey(new_root_state);
    if (next_key == current_root_key_) {
      info.kind = ReuseKind::kExact;
      info.old_rows = row_count_;
      info.new_rows = row_count_;
      return info;
    }

    const auto& old_moves = current_root_state_.moves;
    const auto& new_moves = new_root_state.moves;
    if (new_moves.size() < old_moves.size()) return info;
    if (!std::equal(old_moves.begin(), old_moves.end(), new_moves.begin())) {
      return info;
    }

    if (new_moves.size() == old_moves.size()) {
      return info;
    }

    GameState rolling_state = current_root_state_;
    const int32_t initial_rows = row_count_;
    for (size_t i = old_moves.size(); i < new_moves.size(); ++i) {
      const Move step_move = new_moves[i];
      const int action_id = MoveToNNIndex(step_move, 0);
      rolling_state.moves.push_back(step_move);
      if (!RestrictToBranch(action_id, rolling_state, n)) {
        return ReuseInfo{};
      }
      info.steps++;
      info.action_id = action_id;
    }

    info.kind = ReuseKind::kBranch;
    info.old_rows = initial_rows;
    info.new_rows = row_count_;
    return info;
  }

  RmctsRunResult RunRmcts(const GameState& game_state, const GoParams& go_params) {
    const auto search_start = std::chrono::steady_clock::now();
    constexpr int kNumLanes = 1;
    constexpr int n = rmcts::kPolicySize;

    const int num_sims = std::max(1, options_->Get<int>(kRmctsNumSimsId));
    const int chunk_sims = std::max(1, options_->Get<int>(kRmctsChunkSizeId));
    const float c_puct = options_->Get<float>(kRmctsCpuctId);
    const int epochs = std::max(1, options_->Get<int>(kRmctsEpochsId));
    const float posterior_weight = std::clamp(
        options_->Get<float>(kRmctsPosteriorWeightId), 0.0f, 1.0f);
    const auto budget_ms = ComputeTimeBudgetMs(go_params, game_state);
    const bool time_limited = budget_ms.has_value();

    const int effective_epochs =
      time_limited
        ? std::numeric_limits<int>::max()
        : std::max(epochs, (num_sims + chunk_sims - 1) / chunk_sims);
    const int epoch_sims =
      time_limited
        ? chunk_sims
        : std::max(1, (num_sims + effective_epochs - 1) / effective_epochs);
    const int total_sims_budget = std::max(num_sims, chunk_sims);
    const int initial_capacity =
        2 * kNumLanes * std::max(1, total_sims_budget);
    int expensive_events_remaining =
      time_limited ? std::numeric_limits<int>::max() / 4 : num_sims;
    int sims_remaining_global = num_sims;

    const auto deadline = budget_ms
                              ? std::optional<std::chrono::steady_clock::time_point>(
                                    search_start + std::chrono::milliseconds(*budget_ms))
                              : std::nullopt;

    std::vector<float> root_posterior_running(n, 0.0f);
    std::vector<float> root_prior(n, 0.0f);
    std::vector<float> final_root_policy(n, 0.0f);
    std::vector<float> final_root_q(n, 0.0f);
    std::vector<float> final_root_n(n, 0.0f);
    float final_root_value = 0.0f;

    const ReuseInfo reuse_info = TryReuseForRoot(game_state, n);
    ++reuse_game_searches_;
    if (reuse_info.kind == ReuseKind::kReset) {
      ResetPersistentTree(game_state, initial_capacity, n);
      ++reuse_resets_;
      ++reuse_game_resets_;
    } else if (reuse_info.kind == ReuseKind::kBranch) {
      ++reuse_branch_hits_;
      ++reuse_game_branch_hits_;
      if (reuse_info.old_rows > 0) {
        const double ratio = static_cast<double>(reuse_info.new_rows) /
                             static_cast<double>(reuse_info.old_rows);
        reuse_game_branch_ratio_sum_ += ratio;
        ++reuse_game_branch_ratio_count_;
        reuse_game_branch_old_rows_sum_ +=
            static_cast<double>(reuse_info.old_rows);
        reuse_game_branch_new_rows_sum_ +=
            static_cast<double>(reuse_info.new_rows);
      }
    } else {
      ++reuse_exact_hits_;
      ++reuse_game_exact_hits_;
      EnsurePersistentCapacity(
          row_count_ + 2 * std::max(1, std::max(chunk_sims, epoch_sims)), n);
    }

    const std::string root_key = StateKey(game_state);

    int completed_epochs = 0;
    int64_t total_completed_sims = 0;
    int32_t last_total_sims = 0;
    for (int epoch = 0;
         epoch < effective_epochs && (time_limited || sims_remaining_global > 0);
         ++epoch) {
      if (stop_requested_.load(std::memory_order_relaxed) ||
          DeadlineReached(deadline)) {
        break;
      }

      const int this_epoch_sims =
          time_limited ? epoch_sims : std::min(epoch_sims, sims_remaining_global);
      if (!time_limited) sims_remaining_global -= this_epoch_sims;

        // Worst-case growth per chunk is proportional to chunk sims.
        // Grow buffers ahead of this epoch when needed.
        EnsurePersistentCapacity(row_count_ + 2 * std::max(1, this_epoch_sims),
                                n);

      std::vector<float> new_policy(kNumLanes * n, 0.0f);
      std::vector<float> new_value(kNumLanes, 0.0f);

        inference_stack_size_ = 0;
        new_stack_size_ = 0;
        num_completed_ = 0;
      sims_[0] = this_epoch_sims;
      sims_remaining_[0] = this_epoch_sims;

      EvaluateRows({0}, n, &state_handles_, &policy_, &value_, &eval_cache_,
                   &expensive_events_remaining);

      if (epoch == 0) {
        for (int i = 0; i < n; ++i) root_prior[i] = policy_[i];
        // Keep a valid fallback posterior from the first evaluated root prior.
        final_root_policy = root_prior;
        NormalizePolicy(&final_root_policy);
      }

      if (epoch > 0) {
        auto it = eval_cache_.find(root_key);
        if (it != eval_cache_.end()) {
          // Re-seed the root prior using current-network prior blended with the
          // running posterior from previous epochs.
          std::vector<float> root_prior(n, 0.0f);
          FillRowFromSparse(it->second.sparse_policy, n, root_prior.data());
          for (int i = 0; i < n; ++i) {
            policy_[i] = (1.0f - posterior_weight) * root_prior[i] +
                         posterior_weight * root_posterior_running[i];
          }
          NormalizePolicyRow(policy_.data(), n);
        }
      }

      new_stack_[0] = 0;
      new_stack_size_ = 1;

      void* mcts = MCTS_init(kNumLanes, capacity_ / kNumLanes, c_puct,
                             new_policy.data(), new_value.data(),
                             state_handles_.data(), policy_.data(), value_.data(),
                             Q_.data(), N_.data(), child_.data(), parent_.data(),
                             a0_.data(), sims_.data(), sims_remaining_.data(),
                             inference_stack_.data(), &inference_stack_size_,
                             new_stack_.data(), &new_stack_size_, &num_completed_,
                             &row_count_);

      while (num_completed_ < kNumLanes) {
        if (stop_requested_.load(std::memory_order_relaxed) ||
            DeadlineReached(deadline)) {
          break;
        }
        MCTS_flush_new_stack(mcts);
        if (inference_stack_size_ == 0) break;
        std::vector<int32_t> rows(inference_stack_.begin(),
                                  inference_stack_.begin() + inference_stack_size_);
        inference_stack_size_ = 0;
        EvaluateRows(rows, n, &state_handles_, &policy_, &value_, &eval_cache_,
                     &expensive_events_remaining);
        for (const int32_t row : rows) {
          new_stack_[new_stack_size_++] = row;
        }
      }

      MCTS_free(mcts);

      last_total_sims = sims_[0] - sims_remaining_[0];
      if (last_total_sims <= 0) break;
      total_completed_sims += last_total_sims;

      float new_policy_sum = 0.0f;
      for (const float p : new_policy) new_policy_sum += p;
      const bool has_valid_posterior =
          std::isfinite(new_policy_sum) && new_policy_sum > 0.0f;
      if (!has_valid_posterior) {
        // Deadline/stop can interrupt before root posterior finalization.
        // Keep the last valid posterior instead of replacing it with zeros.
        continue;
      }

      std::copy(Q_.begin(), Q_.begin() + n, final_root_q.begin());
      std::copy(N_.begin(), N_.begin() + n, final_root_n.begin());

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

      AppendChunkTraceCsv(completed_epochs - 1, game_state.CurrentPosition(),
              root_prior, final_root_policy, final_root_q,
              final_root_n);

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

    float posterior_sum = 0.0f;
    for (const float p : final_root_policy) posterior_sum += p;
    if (!(posterior_sum > 0.0f) || !std::isfinite(posterior_sum)) {
      final_root_policy = root_prior;
      NormalizePolicy(&final_root_policy);
    }

    struct Row {
      Move move;
      int action_id;
      float prior;
      float posterior;
      int visits;
      float q;
      float root_n;
    };
    std::vector<Row> rows;
    rows.reserve(legal_moves.size());

    int total_visits = 0;
    for (const Move move : legal_moves) {
      const int idx = MoveToNNIndex(move, 0);
      const float p = std::max(0.0f, final_root_policy[idx]);
        const int visits =
          static_cast<int>(std::lround(std::max(0.0f, final_root_n[idx])));
      rows.push_back(Row{move, idx, std::max(0.0f, root_prior[idx]), p, visits,
           final_root_q[idx], final_root_n[idx]});
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
          << row.prior * 100.0f << "%)"
          << " Post: " << std::fixed << std::setprecision(2)
          << row.posterior * 100.0f << "%"
          << " Q: " << std::fixed << std::setprecision(3)
          << row.q
          << " A: " << row.action_id
          << " RN: " << static_cast<int>(std::lround(row.root_n));
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

    return RmctsRunResult{best_move, final_root_value, total_completed_sims,
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

  bool has_persistent_tree_ = false;
  std::string current_root_key_;
  GameState current_root_state_;
  std::unordered_map<std::string, CachedEval> eval_cache_;

  int capacity_ = 0;
  int32_t row_count_ = 0;
  std::vector<rmcts::GameStateHandle> state_handles_;
  std::vector<float> policy_;
  std::vector<float> value_;
  std::vector<float> Q_;
  std::vector<float> N_;
  std::vector<int32_t> child_;
  std::vector<int32_t> parent_;
  std::vector<int32_t> a0_;
  std::vector<int32_t> sims_;
  std::vector<int32_t> sims_remaining_;
  std::vector<int32_t> inference_stack_;
  int32_t inference_stack_size_ = 0;
  std::vector<int32_t> new_stack_;
  int32_t new_stack_size_ = 0;
  int32_t num_completed_ = 0;

  int64_t reuse_exact_hits_ = 0;
  int64_t reuse_branch_hits_ = 0;
  int64_t reuse_resets_ = 0;
  int64_t reuse_game_searches_ = 0;
  int64_t reuse_game_exact_hits_ = 0;
  int64_t reuse_game_branch_hits_ = 0;
  int64_t reuse_game_resets_ = 0;
  double reuse_game_branch_ratio_sum_ = 0.0;
  int64_t reuse_game_branch_ratio_count_ = 0;
  double reuse_game_branch_old_rows_sum_ = 0.0;
  double reuse_game_branch_new_rows_sum_ = 0.0;
  int last_restrict_action_id_ = -1;
  int32_t last_restrict_old_rows_ = 0;
  int32_t last_restrict_new_rows_ = 0;
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
