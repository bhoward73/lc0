#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "chess/bitboard.h"
#include "chess/gamestate.h"
#include "chess/position.h"
#include "neural/policy_value_batch.h"
#include "neural/shared_params.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace {

std::optional<std::string> ParseStringFlag(int argc, const char** argv,
                                           std::string_view prefix) {
  for (int idx = 1; idx < argc; ++idx) {
    const std::string_view arg(argv[idx]);
    if (arg.size() >= prefix.size() && arg.substr(0, prefix.size()) == prefix) {
      return std::string(arg.substr(prefix.size()));
    }
  }
  return std::nullopt;
}

std::optional<int> ParseIntFlag(int argc, const char** argv,
                                std::string_view prefix) {
  const auto value = ParseStringFlag(argc, argv, prefix);
  if (!value) return std::nullopt;
  try {
    return std::stoi(*value);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bool HasFlag(int argc, const char** argv, std::string_view flag) {
  for (int idx = 1; idx < argc; ++idx) {
    if (std::string_view(argv[idx]) == flag) return true;
  }
  return false;
}

void PrintHelp(const char* binary_name) {
  std::cout << "Usage: " << binary_name << " [options]\n\n"
            << "Options:\n"
            << "  --help, -h            Show this help and exit\n"
            << "  --num-states=N        Number of states to generate (required)\n"
            << "  --warmup-states=W     Warmup size (default: 100, clamped to N)\n"
            << "  --chunk-size=C        States per inference call (default: 4096, 0 = all)\n"
            << "  --weights=<path>      Path to network weights file\n"
            << "  --backend=<name>      Backend name (default: onnx-trt)\n"
            << "  --backend-opts=<s>    Backend options string\n"
            << "  --strict-output       Exit non-zero if output validity checks fail\n"
            << "  --max-bad-rows=N      Max bad rows to print (default: 8)\n\n"
            << "Checks:\n"
            << "  - Value should be finite and in [-1, 1].\n"
            << "  - Policy entries should be finite and in [0, 1].\n"
            << "  - Policy row sum should be close to 1.0 (tol=1e-3).\n\n"
            << "Notes:\n"
            << "  - If --backend-opts is omitted, defaults to dynamic batching:\n"
            << "    gpu=0,optimize=6,batch=0,min_batch=1,steps=1\n"
            << "  - States are sampled from uniformly random legal-move playouts\n"
            << "    over many games and stored as GameState(startpos=current, moves=[]).\n"
            << "  - Chunking limits peak memory use for large N.\n";
}

std::vector<lczero::GameState> BuildRandomStates(std::int64_t num_states) {
  std::vector<lczero::GameState> states;
  states.reserve(static_cast<size_t>(num_states));

  lczero::PositionHistory history;
  history.Reset(lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));

  while (static_cast<std::int64_t>(states.size()) < num_states) {
    if (history.ComputeGameResult() != lczero::GameResult::UNDECIDED) {
      history.Reset(lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));
      continue;
    }

    const lczero::Position& current = history.Last();
    auto legal_moves = current.GetBoard().GenerateLegalMoves();
    if (legal_moves.empty()) continue;

    states.push_back(lczero::GameState{.startpos = current, .moves = {}});

    const int idx = lczero::Random::Get().GetInt(
        0, static_cast<int>(legal_moves.size()) - 1);
    lczero::Move next = legal_moves[static_cast<size_t>(idx)];
    history.Append(next);
  }

  return states;
}

}  // namespace

int main(int argc, const char** argv) {
  if (HasFlag(argc, argv, "--help") || HasFlag(argc, argv, "-h")) {
    PrintHelp(argc > 0 ? argv[0] : "state_batch_bench");
    return 0;
  }

  const auto num_states_opt = ParseIntFlag(argc, argv, "--num-states=");
  if (!num_states_opt || *num_states_opt <= 0) {
    std::cerr << "Invalid or missing --num-states. Use a positive integer."
              << std::endl;
    return 1;
  }
  const std::int64_t num_states = *num_states_opt;

  int warmup_states = 100;
  if (const auto warmup_opt = ParseIntFlag(argc, argv, "--warmup-states=");
      warmup_opt) {
    if (*warmup_opt < 0) {
      std::cerr << "Invalid --warmup-states. Use a non-negative integer."
                << std::endl;
      return 1;
    }
    warmup_states = *warmup_opt;
  }

  int chunk_size = 4096;
  if (const auto chunk_opt = ParseIntFlag(argc, argv, "--chunk-size=");
      chunk_opt) {
    if (*chunk_opt < 0) {
      std::cerr << "Invalid --chunk-size. Use a non-negative integer."
                << std::endl;
      return 1;
    }
    chunk_size = *chunk_opt;
  }

  const bool strict_output = HasFlag(argc, argv, "--strict-output");

  int max_bad_rows = 8;
  if (const auto parsed = ParseIntFlag(argc, argv, "--max-bad-rows=");
      parsed) {
    if (*parsed < 0) {
      std::cerr << "Invalid --max-bad-rows. Use a non-negative integer."
                << std::endl;
      return 1;
    }
    max_bad_rows = *parsed;
  }

  lczero::InitializeMagicBitboards();

  lczero::OptionsParser options_parser;
  lczero::SharedBackendParams::Populate(&options_parser);

  if (const auto weights = ParseStringFlag(argc, argv, "--weights="); weights) {
    options_parser.GetMutableDefaultsOptions()->Set<std::string>(
        lczero::SharedBackendParams::kWeightsId, *weights);
  }

  std::string backend = "onnx-trt";
  if (const auto backend_flag = ParseStringFlag(argc, argv, "--backend=");
      backend_flag) {
    backend = *backend_flag;
  }
  options_parser.GetMutableDefaultsOptions()->Set<std::string>(
      lczero::SharedBackendParams::kBackendId, backend);

  std::string backend_opts = "gpu=0,optimize=6,batch=0,min_batch=1,steps=1";
  if (const auto backend_opts_flag =
          ParseStringFlag(argc, argv, "--backend-opts=");
      backend_opts_flag) {
    backend_opts = *backend_opts_flag;
  }
  options_parser.GetMutableDefaultsOptions()->Set<std::string>(
      lczero::SharedBackendParams::kBackendOptionsId, backend_opts);

  const lczero::OptionsDict& options = options_parser.GetOptionsDict();

  std::cout << "Benchmark config: N=" << num_states
            << " warmup=" << std::min<std::int64_t>(warmup_states, num_states)
        << " chunk_size=" << (chunk_size == 0 ? num_states : chunk_size)
            << " backend="
            << options.Get<std::string>(lczero::SharedBackendParams::kBackendId)
            << " backend_opts="
            << options.Get<std::string>(
                   lczero::SharedBackendParams::kBackendOptionsId)
            << std::endl;

    const double est_policy_gib =
      static_cast<double>(num_states) *
      static_cast<double>(lczero::PolicyValueBatch::kPolicySize) *
      sizeof(float) / (1024.0 * 1024.0 * 1024.0);
    const double est_value_gib =
      static_cast<double>(num_states) * sizeof(float) /
      (1024.0 * 1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(2)
        << "Estimated output memory if kept all-at-once: policy="
        << est_policy_gib << " GiB value=" << est_value_gib << " GiB"
        << std::defaultfloat << std::endl;

  std::cout << "Initializing backend (TensorRT engine build may take minutes on"
            << " first run with large models)..." << std::endl;
  lczero::PolicyValueBatchEvaluator evaluator(options, true);
  const size_t backend_max_batch =
      std::max<size_t>(1, evaluator.GetBackendMaxBatchSize());
  std::cout << "Backend max batch size: " << backend_max_batch << std::endl;
  if (chunk_size > 0 && static_cast<size_t>(chunk_size) > backend_max_batch) {
    std::cout << "Note: chunk_size exceeds backend max batch; evaluator will "
              << "internally split each chunk." << std::endl;
  }

  std::vector<lczero::GameState> states;
  {
    const auto t0 = std::chrono::steady_clock::now();
    states = BuildRandomStates(num_states);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU state generation: " << ms << " ms"
          << " (" << static_cast<double>(num_states) * 1000.0 /
             std::max(ms, 1e-9)
              << " states/s)" << std::endl;
  }

  const int warmup_count =
      static_cast<int>(std::min<std::int64_t>(warmup_states, num_states));
  if (warmup_count > 0) {
    std::vector<lczero::GameState> warmup_batch;
    warmup_batch.reserve(static_cast<size_t>(warmup_count));
    warmup_batch.insert(warmup_batch.end(), states.begin(),
                        states.begin() + warmup_count);

    const auto warm_t0 = std::chrono::steady_clock::now();
    auto warmup_result = evaluator.Evaluate(warmup_batch);
    const auto warm_t1 = std::chrono::steady_clock::now();
    (void)warmup_result;
    const double warm_ms =
      std::chrono::duration<double, std::milli>(warm_t1 - warm_t0).count();
    std::cout << "GPU warmup (" << warmup_count << " states): " << warm_ms
              << " ms" << std::endl;
  }

  try {
    const auto gpu_t0 = std::chrono::steady_clock::now();

    const std::int64_t effective_chunk =
        chunk_size == 0 ? num_states
                        : std::max<std::int64_t>(1, static_cast<std::int64_t>(chunk_size));
    std::int64_t processed = 0;
    double value_sum = 0.0;
    double policy_sum = 0.0;
    std::int64_t nonfinite_values = 0;
    std::int64_t nonfinite_policy = 0;
    std::int64_t out_of_range_values = 0;
    std::int64_t negative_policy = 0;
    std::int64_t policy_over_one = 0;
    std::int64_t bad_policy_rows = 0;
    constexpr double kPolicyRowSumTolerance = 1e-3;
    constexpr double kPolicyEntryTolerance = 1e-6;

    struct BadRowExample {
      std::int64_t state_idx;
      double row_sum;
      float min_policy;
      float max_policy;
      int nonfinite_count;
      int negative_count;
      int gt_one_count;
    };
    std::vector<BadRowExample> bad_row_examples;

    while (processed < num_states) {
      const std::int64_t remain = num_states - processed;
      const std::int64_t take = std::min(remain, effective_chunk);

      std::vector<lczero::GameState> chunk;
      chunk.reserve(static_cast<size_t>(take));
      chunk.insert(chunk.end(), states.begin() + processed,
                   states.begin() + processed + take);

      auto output = evaluator.Evaluate(chunk);
      for (size_t i = 0; i < output.v.size(); ++i) {
        if (std::isfinite(output.v[i])) {
          value_sum += output.v[i];
          if (output.v[i] < -1.0f || output.v[i] > 1.0f) {
            ++out_of_range_values;
          }
        } else {
          ++nonfinite_values;
        }

        double row_sum = 0.0;
        float row_min = std::numeric_limits<float>::infinity();
        float row_max = -std::numeric_limits<float>::infinity();
        int row_nonfinite = 0;
        int row_negative = 0;
        int row_gt_one = 0;
        for (float p : output.pi[i]) {
          if (std::isfinite(p)) {
            policy_sum += p;
            row_sum += p;
            row_min = std::min(row_min, p);
            row_max = std::max(row_max, p);
            if (p < -kPolicyEntryTolerance) {
              ++negative_policy;
              ++row_negative;
            }
            if (p > 1.0f + kPolicyEntryTolerance) {
              ++policy_over_one;
              ++row_gt_one;
            }
          } else {
            ++nonfinite_policy;
            ++row_nonfinite;
          }
        }
        if (std::isinf(row_min)) row_min = 0.0f;
        if (std::isinf(-row_max)) row_max = 0.0f;

        const bool row_bad_sum =
            std::fabs(row_sum - 1.0) > kPolicyRowSumTolerance;
        if (row_bad_sum) {
          ++bad_policy_rows;
        }

        if ((row_bad_sum || row_nonfinite > 0 || row_negative > 0 ||
             row_gt_one > 0) &&
            static_cast<int>(bad_row_examples.size()) < max_bad_rows) {
          bad_row_examples.push_back(BadRowExample{
              .state_idx = processed + static_cast<std::int64_t>(i),
              .row_sum = row_sum,
              .min_policy = row_min,
              .max_policy = row_max,
              .nonfinite_count = row_nonfinite,
              .negative_count = row_negative,
              .gt_one_count = row_gt_one,
          });
        }
      }

      processed += take;
    }

    const auto gpu_t1 = std::chrono::steady_clock::now();
    const double gpu_ms =
        std::chrono::duration<double, std::milli>(gpu_t1 - gpu_t0).count();

    std::cout << "GPU full inference (chunked): " << gpu_ms << " ms"
              << " (" << static_cast<double>(num_states) * 1000.0 /
                     std::max(gpu_ms, 1e-9)
              << " states/s)" << std::endl;
    std::cout << "Processed states: " << processed << " / " << num_states
              << " policy_size=" << lczero::PolicyValueBatch::kPolicySize
              << " value_checksum=" << value_sum
              << " policy_checksum=" << policy_sum << std::endl;

    std::cout << "Validity: nonfinite_value=" << nonfinite_values
              << " value_out_of_range=" << out_of_range_values
              << " nonfinite_policy=" << nonfinite_policy
              << " negative_policy=" << negative_policy
              << " policy_gt_1=" << policy_over_one
              << " bad_policy_rows=" << bad_policy_rows
              << " (row_tol=" << kPolicyRowSumTolerance
              << ", entry_tol=" << kPolicyEntryTolerance << ")"
              << std::endl;

    if (!bad_row_examples.empty()) {
      std::cout << "Bad policy rows (state_idx sum min max nonfinite neg gt1):"
                << std::endl;
      for (const auto& row : bad_row_examples) {
        std::cout << "  " << row.state_idx << " " << row.row_sum << " "
                  << row.min_policy << " " << row.max_policy << " "
                  << row.nonfinite_count << " " << row.negative_count << " "
                  << row.gt_one_count << std::endl;
      }
    }

    if (nonfinite_values > 0 || nonfinite_policy > 0) {
      std::cout << "Non-finite outputs detected: value=" << nonfinite_values
                << " policy=" << nonfinite_policy << std::endl;
    }

    if (strict_output &&
        (nonfinite_values > 0 || nonfinite_policy > 0 ||
         out_of_range_values > 0 || negative_policy > 0 ||
         policy_over_one > 0 || bad_policy_rows > 0)) {
      std::cerr << "Strict output validation failed." << std::endl;
      return 3;
    }
  } catch (const std::bad_alloc&) {
    std::cerr << "Out of memory during inference/output allocation."
              << " Try smaller --chunk-size and/or --num-states." << std::endl;
    return 2;
  }

  return 0;
}
