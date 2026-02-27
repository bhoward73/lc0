
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "engine.h"
#include "neural/shared_params.h"
#include "chess/position.h"
#include "search/register.h"
#include "search/classic/params.h"
#include "utils/optionsparser.h"

namespace {

char GetPieceAt(const lczero::ChessBoard& board, lczero::Square square) {
    char piece = '\0';
    if (board.ours().get(square) || board.theirs().get(square)) {
        if (board.pawns().get(square)) {
            piece = 'P';
        } else if (board.kings().get(square)) {
            piece = 'K';
        } else if (board.bishops().get(square)) {
            piece = 'B';
        } else if (board.queens().get(square)) {
            piece = 'Q';
        } else if (board.rooks().get(square)) {
            piece = 'R';
        } else {
            piece = 'N';
        }
        if (board.theirs().get(square)) {
            piece = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
        }
    }
    return piece;
}

std::string_view PieceToUnicode(char piece) {
    switch (piece) {
        case 'P': return "♙";
        case 'N': return "♘";
        case 'B': return "♗";
        case 'R': return "♖";
        case 'Q': return "♕";
        case 'K': return "♔";
        case 'p': return "♟";
        case 'n': return "♞";
        case 'b': return "♝";
        case 'r': return "♜";
        case 'q': return "♛";
        case 'k': return "♚";
        default: return "·";
    }
}

std::string_view PieceToUnicodeSolid(char piece) {
    switch (std::tolower(static_cast<unsigned char>(piece))) {
        case 'p': return "♟";
        case 'n': return "♞";
        case 'b': return "♝";
        case 'r': return "♜";
        case 'q': return "♛";
        case 'k': return "♚";
        default: return "·";
    }
}

bool SupportsAnsiColors() {
    if (!isatty(fileno(stdout))) return false;
    if (const char* no_color = std::getenv("NO_COLOR"); no_color && *no_color) {
        return false;
    }
    const char* term = std::getenv("TERM");
    if (!term || std::string_view(term) == "dumb") return false;
    return true;
}

bool IsWhitePiece(char piece) {
    return piece >= 'A' && piece <= 'Z';
}

void PrintBoardCell(char piece, bool is_light_square, bool use_ansi_colors) {
    if (!use_ansi_colors) {
        std::cout << PieceToUnicode(piece);
        return;
    }

    constexpr std::string_view kReset = "\x1b[0m";
    constexpr std::string_view kLightSquareBg = "\x1b[48;5;247m";
    constexpr std::string_view kDarkSquareBg = "\x1b[48;5;240m";
    constexpr std::string_view kWhitePieceFg = "\x1b[38;5;231m";
    constexpr std::string_view kBlackPieceFg = "\x1b[38;5;16m";
    constexpr std::string_view kEmptyFg = "\x1b[38;5;244m";

    const std::string_view bg = is_light_square ? kLightSquareBg : kDarkSquareBg;
    const std::string_view fg =
        piece == '\0' ? kEmptyFg : (IsWhitePiece(piece) ? kWhitePieceFg : kBlackPieceFg);
    const std::string_view glyph = piece == '\0' ? PieceToUnicode(piece)
                                                  : PieceToUnicodeSolid(piece);

    std::cout << bg << fg << ' ' << glyph << ' ' << kReset;
}

void PrintUnicodeBoard(const lczero::Position& pos) {
    lczero::ChessBoard board = pos.GetBoard();
    if (board.flipped()) {
        board.Mirror();
    }
    const bool use_ansi_colors = SupportsAnsiColors();

    if (use_ansi_colors) {
        std::cout << "   a  b  c  d  e  f  g  h\n";
    } else {
        std::cout << "  a b c d e f g h\n";
    }
    for (lczero::Rank rank = lczero::kRank8; rank.IsValid(); --rank) {
        std::cout << static_cast<int>(rank.idx) + 1 << ' ';
        for (int file_idx = lczero::kFileA.idx; file_idx <= lczero::kFileH.idx;
             ++file_idx) {
            const lczero::File file = lczero::File::FromIdx(file_idx);
            const lczero::Square square(file, rank);
            const char piece = GetPieceAt(board, square);
            const bool is_light_square = ((file.idx + rank.idx) % 2) == 1;
            PrintBoardCell(piece, is_light_square, use_ansi_colors);
            if (!use_ansi_colors && file_idx != lczero::kFileH.idx) std::cout << ' ';
        }
        std::cout << ' ' << static_cast<int>(rank.idx) + 1 << '\n';
    }
    if (use_ansi_colors) {
        std::cout << "   a  b  c  d  e  f  g  h\n";
    } else {
        std::cout << "  a b c d e f g h\n";
    }
}

class BestMoveCaptureResponder : public lczero::UciResponder {
 public:
    struct PolicyStat {
        std::string move;
        std::uint32_t visits;
        double prior;
        std::optional<double> posterior;
        std::optional<double> q;
        std::optional<int> action_id;
        std::optional<int> root_n;
    };

    void OutputBestMove(lczero::BestMoveInfo* info) override {
        std::lock_guard<std::mutex> lock(mutex_);
        best_move_ = info->bestmove;
    }

    void OutputThinkingInfo(std::vector<lczero::ThinkingInfo>* infos) override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& info : *infos) {
            if (info.multipv > 1) continue;
            if (info.nodes >= 0) nodes_ = info.nodes;
            if (info.nps >= 0) nps_ = info.nps;
            if (info.time >= 0) time_ms_ = info.time;
            if (info.wdl) {
                eval_stm_ = static_cast<double>(info.wdl->w - info.wdl->l) / 1000.0;
            }
            if (!info.comment.empty()) {
                const auto parsed = ParsePolicyLine(info.comment);
                if (parsed) {
                    policy_by_move_[parsed->move] = *parsed;
                }
            }
        }
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        best_move_.reset();
        eval_stm_.reset();
        nodes_.reset();
        nps_.reset();
        time_ms_.reset();
        policy_by_move_.clear();
    }

    std::optional<lczero::Move> GetBestMove() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return best_move_;
    }

    std::optional<double> GetEvalStm() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return eval_stm_;
    }

    std::vector<PolicyStat> GetPolicyStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<PolicyStat> stats;
        stats.reserve(policy_by_move_.size());
        for (const auto& [_, value] : policy_by_move_) {
            stats.push_back(value);
        }
        return stats;
    }

    std::optional<std::int64_t> GetNodes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_;
    }

    std::optional<int> GetNps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nps_;
    }

    std::optional<std::int64_t> GetTimeMs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return time_ms_;
    }

 private:
    static std::optional<PolicyStat> ParsePolicyLine(const std::string& line) {
        PolicyStat stat;

        auto is_move_text = [](std::string_view move) {
            if (move.size() != 4 && move.size() != 5) return false;
            const auto lower = [](char c) {
                return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            };
            if (lower(move[0]) < 'a' || lower(move[0]) > 'h') return false;
            if (move[1] < '1' || move[1] > '8') return false;
            if (lower(move[2]) < 'a' || lower(move[2]) > 'h') return false;
            if (move[3] < '1' || move[3] > '8') return false;
            if (move.size() == 5) {
                const char p = lower(move[4]);
                if (p != 'n' && p != 'b' && p != 'r' && p != 'q') return false;
            }
            return true;
        };

        const auto first_non_space = line.find_first_not_of(' ');
        if (first_non_space == std::string::npos) return std::nullopt;
        const auto token_end = line.find(' ', first_non_space);
        const std::string move =
            line.substr(first_non_space, token_end - first_non_space);
        if (!is_move_text(move)) return std::nullopt;

        const auto n_pos = line.find("N:");
        if (n_pos == std::string::npos) return std::nullopt;
        std::size_t n_value_start = n_pos + 2;
        while (n_value_start < line.size() && line[n_value_start] == ' ') {
            ++n_value_start;
        }
        std::size_t n_value_end = n_value_start;
        while (n_value_end < line.size() && std::isdigit(static_cast<unsigned char>(line[n_value_end]))) {
            ++n_value_end;
        }
        if (n_value_start == n_value_end) return std::nullopt;

        const auto p_pos = line.find("(P:");
        if (p_pos == std::string::npos) return std::nullopt;
        std::size_t p_value_start = p_pos + 3;
        while (p_value_start < line.size() && line[p_value_start] == ' ') {
            ++p_value_start;
        }
        const auto percent_pos = line.find('%', p_value_start);
        if (percent_pos == std::string::npos) return std::nullopt;
        const std::string p_text = line.substr(p_value_start, percent_pos - p_value_start);

        stat.move = move;
        stat.visits = static_cast<std::uint32_t>(std::stoul(line.substr(n_value_start, n_value_end - n_value_start)));
        stat.prior = std::stod(p_text) / 100.0;

        const auto post_pos = line.find("Post:");
        if (post_pos != std::string::npos) {
            std::size_t post_value_start = post_pos + 5;
            while (post_value_start < line.size() && line[post_value_start] == ' ') {
                ++post_value_start;
            }
            const auto post_percent_pos = line.find('%', post_value_start);
            if (post_percent_pos != std::string::npos) {
                const std::string post_text =
                    line.substr(post_value_start, post_percent_pos - post_value_start);
                try {
                    stat.posterior = std::stod(post_text) / 100.0;
                } catch (const std::exception&) {
                    stat.posterior.reset();
                }
            }
        }

        const auto q_pos = line.find("Q:");
        if (q_pos != std::string::npos) {
            std::size_t q_value_start = q_pos + 2;
            while (q_value_start < line.size() && line[q_value_start] == ' ') {
                ++q_value_start;
            }
            std::size_t q_value_end = q_value_start;
            while (q_value_end < line.size()) {
                const char c = line[q_value_end];
                if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '+')) {
                    break;
                }
                ++q_value_end;
            }
            if (q_value_end > q_value_start) {
                try {
                    stat.q = std::stod(line.substr(q_value_start, q_value_end - q_value_start));
                } catch (const std::exception&) {
                    stat.q.reset();
                }
            }
        }

        const auto a_pos = line.find("A:");
        if (a_pos != std::string::npos) {
            std::size_t a_value_start = a_pos + 2;
            while (a_value_start < line.size() && line[a_value_start] == ' ') {
                ++a_value_start;
            }
            std::size_t a_value_end = a_value_start;
            while (a_value_end < line.size() &&
                   std::isdigit(static_cast<unsigned char>(line[a_value_end]))) {
                ++a_value_end;
            }
            if (a_value_end > a_value_start) {
                try {
                    stat.action_id = std::stoi(
                        line.substr(a_value_start, a_value_end - a_value_start));
                } catch (const std::exception&) {
                    stat.action_id.reset();
                }
            }
        }

        const auto rn_pos = line.find("RN:");
        if (rn_pos != std::string::npos) {
            std::size_t rn_value_start = rn_pos + 3;
            while (rn_value_start < line.size() && line[rn_value_start] == ' ') {
                ++rn_value_start;
            }
            std::size_t rn_value_end = rn_value_start;
            while (rn_value_end < line.size() &&
                   std::isdigit(static_cast<unsigned char>(line[rn_value_end]))) {
                ++rn_value_end;
            }
            if (rn_value_end > rn_value_start) {
                try {
                    stat.root_n = std::stoi(
                        line.substr(rn_value_start, rn_value_end - rn_value_start));
                } catch (const std::exception&) {
                    stat.root_n.reset();
                }
            }
        }
        return stat;
    }

    mutable std::mutex mutex_;
    std::optional<lczero::Move> best_move_;
    std::optional<double> eval_stm_;
    std::optional<std::int64_t> nodes_;
    std::optional<int> nps_;
    std::optional<std::int64_t> time_ms_;
    std::unordered_map<std::string, PolicyStat> policy_by_move_;
};

std::optional<std::string> ParseWeightsFlag(int argc, const char** argv) {
    constexpr std::string_view kPrefix = "--weights=";
    for (int idx = 1; idx < argc; ++idx) {
        const std::string_view arg(argv[idx]);
        if (arg.size() >= kPrefix.size() &&
            arg.substr(0, kPrefix.size()) == kPrefix) {
            return std::string(arg.substr(kPrefix.size()));
        }
    }
    return std::nullopt;
}

std::optional<std::string> ParseStringFlag(int argc, const char** argv,
                                           std::string_view prefix) {
    for (int idx = 1; idx < argc; ++idx) {
        const std::string_view arg(argv[idx]);
        if (arg.size() >= prefix.size() &&
            arg.substr(0, prefix.size()) == prefix) {
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

std::optional<float> ParseFloatFlag(int argc, const char** argv,
                                    std::string_view prefix) {
    const auto value = ParseStringFlag(argc, argv, prefix);
    if (!value) return std::nullopt;
    try {
        return std::stof(*value);
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
              << "  --help, -h          Show this help and exit\n"
              << "  --weights=<path>    Path to network weights file\n"
              << "  --backend=<name>    Backend name (e.g. cuda, onnx-trt, onnx-cuda)\n"
              << "  --backend-opts=<s>  Backend options string (key=value,...)\n"
              << "  --movetime-ms=N     Search time per move in milliseconds\n"
              << "  --move-overhead-ms=N  Time safety margin subtracted from movetime\n"
              << "  --threads=N         Number of search worker threads (0 = backend default)\n"
              << "  --cpuct=X           Classic CPuct value (alias of --classic-cpuct)\n"
              << "  --classic-cpuct=X   Classic CPuct value (exploration constant)\n"
              << "  --rmcts-cpuct=X     RMCTS c_puct value\n"
              << "  --minibatch-size=N  Search minibatch size (0 = auto)\n"
              << "  --max-prefetch=N    Search prefetch batch size\n"
              << "  --max-half-moves=N  Stop after N half-moves (plies)\n"
              << "  --human             Enable human-vs-human mode (manual moves for both sides)\n"
              << "  --no-pause          In bot mode, do not pause between moves\n\n"
              << "Notes:\n"
              << "  - Default search movetime is 300 ms per move.\n"
              << "  - Default move overhead is 0 ms in this app.\n"
              << "  - Default backend is onnx-trt with backend opts 'batch=136,steps=1' (unless overridden).\n"
              << "  - Default classic minibatch/max-prefetch are 136/136 (unless overridden).\n"
              << "  - In human mode, type moves in white-oriented UCI coordinates for both sides (e.g. e2e4, e7e5).\n"
              << "  - You can type 'show policy' on either side to inspect classic and rmcts root policy.\n"
              << "  - Type 'quit' (or 'q'/'exit') to leave the program.\n";
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string NormalizeMoveToWhitePov(std::string move_text,
                                    bool black_to_move) {
    if (!black_to_move) return move_text;
    if (move_text.size() < 4) return move_text;

    auto flip_rank = [](char rank) {
        if (rank < '1' || rank > '8') return rank;
        return static_cast<char>('1' + ('8' - rank));
    };

    move_text[1] = flip_rank(move_text[1]);
    move_text[3] = flip_rank(move_text[3]);
    return move_text;
}

std::string CanonicalizeMoveToWhitePov(const lczero::Position& pos,
                                       const std::string& move_text) {
    const lczero::ChessBoard& board = pos.GetBoard();

    auto to_display = [&](const lczero::Move& parsed) {
        lczero::Move display = parsed;
        if (pos.IsBlackToMove()) {
            display.Flip();
        }
        return display.ToString(false);
    };

    try {
        const lczero::Move parsed = board.ParseMove(move_text);
        return to_display(parsed);
    } catch (const std::exception&) {
    }

    if (pos.IsBlackToMove()) {
        try {
            const std::string flipped = NormalizeMoveToWhitePov(move_text, true);
            const lczero::Move parsed = board.ParseMove(flipped);
            return to_display(parsed);
        } catch (const std::exception&) {
        }
    }

    return move_text;
}

void PrintPolicyComparisonTable(
    const std::vector<BestMoveCaptureResponder::PolicyStat>& classic_stats,
    const std::vector<BestMoveCaptureResponder::PolicyStat>& rmcts_stats) {
    if (classic_stats.empty() && rmcts_stats.empty()) {
        std::cout << "No policy stats available yet.\n";
        return;
    }

    std::unordered_map<std::string, double> classic_prior;
    std::unordered_map<std::string, double> rmcts_prior;
    std::unordered_map<std::string, double> classic_post;
    std::unordered_map<std::string, double> rmcts_post;
    std::unordered_map<std::string, double> classic_q;
    std::unordered_map<std::string, double> rmcts_q;
    std::unordered_map<std::string, int> classic_visits;
    std::unordered_map<std::string, int> rmcts_visits;

    std::uint64_t classic_total_visits = 0;
    std::uint64_t rmcts_total_visits = 0;
    for (const auto& row : classic_stats) classic_total_visits += row.visits;
    for (const auto& row : rmcts_stats) rmcts_total_visits += row.visits;

    for (const auto& row : classic_stats) {
        classic_prior[row.move] = row.prior;
        classic_visits[row.move] = static_cast<int>(row.visits);
        classic_post[row.move] = row.posterior.value_or(
            classic_total_visits == 0
                ? 0.0
                : static_cast<double>(row.visits) /
                      static_cast<double>(classic_total_visits));
        if (row.q.has_value()) classic_q[row.move] = *row.q;
    }
    for (const auto& row : rmcts_stats) {
        rmcts_prior[row.move] = row.prior;
        rmcts_visits[row.move] = row.root_n.value_or(static_cast<int>(row.visits));
        rmcts_post[row.move] = row.posterior.value_or(
            rmcts_total_visits == 0
                ? 0.0
                : static_cast<double>(row.visits) /
                      static_cast<double>(rmcts_total_visits));
        if (row.q.has_value()) rmcts_q[row.move] = *row.q;
    }

    std::unordered_map<std::string, double> network_prior = classic_prior;
    if (network_prior.empty()) network_prior = rmcts_prior;

    auto weighted_value = [](const std::unordered_map<std::string, double>& policy,
                             const std::unordered_map<std::string, double>& q_by_move)
        -> std::optional<double> {
        double sum = 0.0;
        bool has_term = false;
        for (const auto& [move, prob] : policy) {
            const auto it = q_by_move.find(move);
            if (it == q_by_move.end()) continue;
            sum += prob * it->second;
            has_term = true;
        }
        if (!has_term) return std::nullopt;
        return sum;
    };

    std::unordered_map<std::string, double> prior_q_source = classic_q;
    for (const auto& [move, q] : rmcts_q) {
        if (!prior_q_source.count(move)) prior_q_source[move] = q;
    }

    const auto prior_v = weighted_value(network_prior, prior_q_source);
    const auto classic_post_v = weighted_value(classic_post, classic_q);
    const auto rmcts_post_v = weighted_value(rmcts_post, rmcts_q);

    struct Row {
        std::string move;
        double rank_prior = 0.0;
        double classic_post = 0.0;
        double rmcts_post = 0.0;
        int classic_visits = 0;
        int rmcts_visits = 0;
        std::optional<double> classic_q;
        std::optional<double> rmcts_q;
    };

    std::vector<Row> rows;
    rows.reserve(classic_prior.size() + rmcts_prior.size());

    std::unordered_map<std::string, bool> seen;
    for (const auto& [move, _] : classic_prior) seen[move] = true;
    for (const auto& [move, _] : rmcts_prior) seen[move] = true;

    for (const auto& [move, _] : seen) {
        const double c_prior = classic_prior.count(move) ? classic_prior[move] : 0.0;
        const double r_prior = rmcts_prior.count(move) ? rmcts_prior[move] : 0.0;
        rows.push_back(Row{
            .move = move,
            .rank_prior = std::max(c_prior, r_prior),
            .classic_post = classic_post.count(move) ? classic_post[move] : 0.0,
            .rmcts_post = rmcts_post.count(move) ? rmcts_post[move] : 0.0,
            .classic_visits = classic_visits.count(move) ? classic_visits[move] : 0,
            .rmcts_visits = rmcts_visits.count(move) ? rmcts_visits[move] : 0,
            .classic_q = classic_q.count(move) ? std::optional<double>(classic_q[move]) : std::nullopt,
            .rmcts_q = rmcts_q.count(move) ? std::optional<double>(rmcts_q[move]) : std::nullopt,
        });
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.rank_prior != b.rank_prior) return a.rank_prior > b.rank_prior;
        return a.move < b.move;
    });

    auto print_opt_value = [](const std::optional<double>& value) {
        if (value.has_value()) {
            std::cout << std::fixed << std::setprecision(4) << *value
                      << std::defaultfloat;
        } else {
            std::cout << "n/a";
        }
    };

    std::cout << "Values: prior_v=";
    print_opt_value(prior_v);
    std::cout << " classic_post_v=";
    print_opt_value(classic_post_v);
    std::cout << " rmcts_post_v=";
    print_opt_value(rmcts_post_v);
    std::cout << '\n';
    std::cout << "Policy comparison (posterior, ranked by prior):\n";
    std::cout << std::left << std::setw(8) << "move"
              << std::right << std::setw(10) << "prior"
              << std::right << std::setw(10) << "classic"
              << std::setw(10) << "rmcts"
              << std::setw(10) << "N_classic"
              << std::setw(10) << "N_rmcts"
              << std::setw(10) << "Q_classic"
              << std::setw(10) << "Q_rmcts" << '\n';
    for (const auto& row : rows) {
        std::cout << std::left << std::setw(8) << row.move
                  << std::right << std::setw(10) << std::fixed
                  << std::setprecision(4) << row.rank_prior
                  << std::right << std::setw(10) << std::fixed
                  << std::setprecision(4) << row.classic_post
                  << std::setw(10) << row.rmcts_post
                  << std::setw(10) << row.classic_visits
                  << std::setw(10) << row.rmcts_visits;

        if (row.classic_q.has_value()) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << *row.classic_q;
        } else {
            std::cout << std::setw(10) << "-";
        }
        if (row.rmcts_q.has_value()) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << *row.rmcts_q;
        } else {
            std::cout << std::setw(10) << "-";
        }
        std::cout << std::defaultfloat << '\n';
    }
}

void PrintRmctsDiagnostics(
    const std::vector<BestMoveCaptureResponder::PolicyStat>& rmcts_stats) {
    struct Row {
        std::string move;
        int action_id = -1;
        double prior = 0.0;
        double posterior = 0.0;
        double q = 0.0;
        int root_n = 0;
    };

    std::vector<Row> rows;
    rows.reserve(rmcts_stats.size());
    for (const auto& stat : rmcts_stats) {
        if (!stat.q.has_value()) continue;
        rows.push_back(Row{
            .move = stat.move,
            .action_id = stat.action_id.value_or(-1),
            .prior = stat.prior,
            .posterior = stat.posterior.value_or(0.0),
            .q = *stat.q,
            .root_n = stat.root_n.value_or(0),
        });
    }
    if (rows.empty()) return;

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.posterior != b.posterior) return a.posterior > b.posterior;
        return a.move < b.move;
    });

    std::cout << "\nRMCTS diagnostics (top by posterior):\n";
    std::cout << std::left << std::setw(8) << "move"
              << std::right << std::setw(6) << "a"
              << std::right << std::setw(10) << "prior"
              << std::setw(10) << "post"
              << std::setw(10) << "q"
              << std::setw(8) << "n" << '\n';
    const size_t limit = std::min<size_t>(12, rows.size());
    for (size_t i = 0; i < limit; ++i) {
        const auto& row = rows[i];
        std::cout << std::left << std::setw(8) << row.move
                  << std::right << std::setw(6) << row.action_id
                  << std::right << std::setw(10) << std::fixed
                  << std::setprecision(4) << row.prior
                  << std::setw(10) << row.posterior
                  << std::setw(10) << std::setprecision(3) << row.q
                  << std::setw(8) << row.root_n
                  << std::defaultfloat << '\n';
    }
}

std::string GameResultMessage(const lczero::PositionHistory& history) {
    const lczero::GameResult result = history.ComputeGameResult();
    switch (result) {
        case lczero::GameResult::WHITE_WON:
            return "Game Over: White wins.";
        case lczero::GameResult::BLACK_WON:
            return "Game Over: Black wins.";
        case lczero::GameResult::DRAW: {
            const lczero::Position& pos = history.Last();
            if (pos.GetRule50Ply() >= 100) {
                return "Game Over: Draw by 50-move rule.";
            }
            if (pos.GetRepetitions() >= 2) {
                return "Game Over: Draw by threefold repetition.";
            }
            if (!pos.GetBoard().HasMatingMaterial()) {
                return "Game Over: Draw by insufficient mating material.";
            }
            return "Game Over: Draw.";
        }
        case lczero::GameResult::UNDECIDED:
            break;
    }
    return "Game Over.";
}

}  // namespace

int main(int argc, const char** argv) {
    if (HasFlag(argc, argv, "--help") || HasFlag(argc, argv, "-h")) {
        PrintHelp(argc > 0 ? argv[0] : "play");
        return 0;
    }

    lczero::InitializeMagicBitboards();

    lczero::PositionHistory history;
    history.Reset(lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));

    std::string search_name = "classic";
    if (const auto parsed_search = ParseStringFlag(argc, argv, "--search=");
        parsed_search) {
        search_name = *parsed_search;
    }

    auto* factory = lczero::SearchManager::Get()->GetFactoryByName(search_name);
    auto* classic_factory = lczero::SearchManager::Get()->GetFactoryByName("classic");
    auto* rmcts_factory = lczero::SearchManager::Get()->GetFactoryByName("rmcts");
    if (!factory) {
        std::cerr << "Search algorithm '" << search_name
                  << "' is not available." << std::endl;
        return 1;
    }

    lczero::OptionsParser options_parser;
    lczero::Engine::PopulateOptions(&options_parser);
    if (classic_factory) classic_factory->PopulateParams(&options_parser);
    if (rmcts_factory) rmcts_factory->PopulateParams(&options_parser);
    factory->PopulateParams(&options_parser);
    lczero::SharedBackendParams::Populate(&options_parser);
    if (classic_factory) {
        options_parser.GetMutableDefaultsOptions()->Set<bool>(
            lczero::classic::BaseSearchParams::kVerboseStatsId, true);
    }

    const bool is_classic_search = (search_name == "classic");

    int move_overhead_ms = 0;
    if (const auto parsed_move_overhead =
            ParseIntFlag(argc, argv, "--move-overhead-ms=");
        parsed_move_overhead) {
        if (*parsed_move_overhead < 0) {
            std::cerr << "Invalid value for --move-overhead-ms."
                      << " Expected a non-negative integer." << std::endl;
            return 1;
        }
        move_overhead_ms = *parsed_move_overhead;
    }
    if (is_classic_search) {
        options_parser.GetMutableDefaultsOptions()->Set<int>("move-overhead",
                                                             move_overhead_ms);
        options_parser.SetUciOption("MoveOverheadMs",
                                    std::to_string(move_overhead_ms));
        options_parser.GetMutableDefaultsOptions()->Set<bool>(
            lczero::classic::BaseSearchParams::kVerboseStatsId, true);
    }
    if (const auto weights_path = ParseWeightsFlag(argc, argv); weights_path) {
        options_parser.GetMutableDefaultsOptions()->Set<std::string>(
            lczero::SharedBackendParams::kWeightsId, *weights_path);
    }
        const auto backend_flag = ParseStringFlag(argc, argv, "--backend=");
        const auto backend_opts_flag = ParseStringFlag(argc, argv, "--backend-opts=");

        const std::string backend_value = backend_flag.value_or("onnx-trt");
        options_parser.GetMutableDefaultsOptions()->Set<std::string>(
        lczero::SharedBackendParams::kBackendId, backend_value);

        if (backend_opts_flag) {
        options_parser.GetMutableDefaultsOptions()->Set<std::string>(
            lczero::SharedBackendParams::kBackendOptionsId, *backend_opts_flag);
        } else if (backend_value == "onnx-trt") {
        options_parser.GetMutableDefaultsOptions()->Set<std::string>(
            lczero::SharedBackendParams::kBackendOptionsId,
            "batch=136,steps=1");
    }

    int move_time_ms = 300;
    if (const auto parsed_movetime = ParseIntFlag(argc, argv, "--movetime-ms=");
        parsed_movetime) {
        if (*parsed_movetime <= 0) {
            std::cerr << "Invalid value for --movetime-ms."
                      << " Expected a positive integer." << std::endl;
            return 1;
        }
        move_time_ms = *parsed_movetime;
    }

    int threads = 0;
    if (const auto parsed_threads = ParseIntFlag(argc, argv, "--threads=");
        parsed_threads) {
        if (*parsed_threads < 0) {
            std::cerr << "Invalid value for --threads."
                      << " Expected a non-negative integer." << std::endl;
            return 1;
        }
        threads = *parsed_threads;
    }
    if (is_classic_search) {
        options_parser.GetMutableDefaultsOptions()->Set<int>("threads", threads);
        options_parser.SetUciOption("Threads", std::to_string(threads));
    }

    std::optional<float> classic_cpuct =
        ParseFloatFlag(argc, argv, "--classic-cpuct=");
    if (!classic_cpuct) {
        classic_cpuct = ParseFloatFlag(argc, argv, "--cpuct=");
    }
    if (classic_cpuct) {
        if (!std::isfinite(*classic_cpuct) || *classic_cpuct <= 0.0f) {
            std::cerr << "Invalid value for --classic-cpuct/--cpuct."
                      << " Expected a positive finite number." << std::endl;
            return 1;
        }
        if (!classic_factory) {
            std::cerr << "Classic search is not available; cannot set --classic-cpuct."
                      << std::endl;
            return 1;
        }
        options_parser.GetMutableDefaultsOptions()->Set<float>(
            lczero::classic::BaseSearchParams::kCpuctId, *classic_cpuct);
        options_parser.SetUciOption("CPuct", std::to_string(*classic_cpuct));
    }

    if (const auto rmcts_cpuct = ParseFloatFlag(argc, argv, "--rmcts-cpuct=");
        rmcts_cpuct) {
        if (!std::isfinite(*rmcts_cpuct) || *rmcts_cpuct <= 0.0f) {
            std::cerr << "Invalid value for --rmcts-cpuct."
                      << " Expected a positive finite number." << std::endl;
            return 1;
        }
        if (!rmcts_factory) {
            std::cerr << "RMCTS search is not available; cannot set --rmcts-cpuct."
                      << std::endl;
            return 1;
        }
        options_parser.SetUciOption("RMCTSCpuct", std::to_string(*rmcts_cpuct));
    }

    if (is_classic_search) {
        if (const auto minibatch_size =
                ParseIntFlag(argc, argv, "--minibatch-size=");
            minibatch_size) {
            if (*minibatch_size < 0) {
                std::cerr << "Invalid value for --minibatch-size."
                          << " Expected a non-negative integer." << std::endl;
                return 1;
            }
            options_parser.GetMutableDefaultsOptions()->Set<int>(
                lczero::classic::BaseSearchParams::kMiniBatchSizeId,
                *minibatch_size);
        } else {
            options_parser.GetMutableDefaultsOptions()->Set<int>(
                lczero::classic::BaseSearchParams::kMiniBatchSizeId, 136);
        }

        if (const auto max_prefetch = ParseIntFlag(argc, argv, "--max-prefetch=");
            max_prefetch) {
            if (*max_prefetch < 0) {
                std::cerr << "Invalid value for --max-prefetch."
                          << " Expected a non-negative integer." << std::endl;
                return 1;
            }
            options_parser.GetMutableDefaultsOptions()->Set<int>(
                lczero::classic::SearchParams::kMaxPrefetchBatchId,
                *max_prefetch);
        } else {
            options_parser.GetMutableDefaultsOptions()->Set<int>(
                lczero::classic::SearchParams::kMaxPrefetchBatchId, 136);
        }
    }

    int minibatch_size_effective = -1;
    int max_prefetch_effective = -1;
    if (is_classic_search) {
        minibatch_size_effective = options_parser.GetOptionsDict().Get<int>(
            lczero::classic::BaseSearchParams::kMiniBatchSizeId);
        max_prefetch_effective = options_parser.GetOptionsDict().Get<int>(
            lczero::classic::SearchParams::kMaxPrefetchBatchId);
    }

    int max_half_moves = 400;
    if (const auto parsed_max_half_moves =
            ParseIntFlag(argc, argv, "--max-half-moves=");
        parsed_max_half_moves) {
        if (*parsed_max_half_moves <= 0) {
            std::cerr << "Invalid value for --max-half-moves."
                      << " Expected a positive integer." << std::endl;
            return 1;
        }
        max_half_moves = *parsed_max_half_moves;
    }

    const lczero::OptionsDict& options = options_parser.GetOptionsDict();

    const bool human_mode = HasFlag(argc, argv, "--human");
    const bool no_pause = HasFlag(argc, argv, "--no-pause");

    BestMoveCaptureResponder responder;
    std::unique_ptr<lczero::Engine> engine;
    if (!human_mode) {
        engine = std::make_unique<lczero::Engine>(*factory, options);
        engine->RegisterUciResponder(&responder);
    }

    auto fetch_policy_stats = [&](std::string_view target_search,
                                  const lczero::Position& current_pos)
        -> std::optional<std::vector<BestMoveCaptureResponder::PolicyStat>> {
        if (target_search == search_name && engine) {
            auto run_probe_once = [&]() {
                responder.Reset();
                engine->SetPosition(lczero::PositionToFen(current_pos), {});
                lczero::GoParams params;
                params.movetime = move_time_ms;
                engine->Go(params);
                engine->Wait();
                auto stats = responder.GetPolicyStats();
                for (auto& row : stats) {
                    row.move = CanonicalizeMoveToWhitePov(current_pos, row.move);
                }
                return stats;
            };

            auto stats = run_probe_once();
            auto all_zero_posterior = [](const std::vector<BestMoveCaptureResponder::PolicyStat>& rows) {
                if (rows.empty()) return true;
                for (const auto& row : rows) {
                    const double post = row.posterior.value_or(static_cast<double>(row.visits));
                    if (post > 0.0) return false;
                }
                return true;
            };

            if (all_zero_posterior(stats)) {
                stats = run_probe_once();
            }
            return stats;
        }

        const lczero::SearchFactory* probe_factory = nullptr;
        if (target_search == "classic") {
            probe_factory = classic_factory;
        } else if (target_search == "rmcts") {
            probe_factory = rmcts_factory;
        } else {
            probe_factory = lczero::SearchManager::Get()->GetFactoryByName(
                std::string(target_search));
        }
        if (!probe_factory) return std::nullopt;
        try {
            BestMoveCaptureResponder probe_responder;
            lczero::Engine probe_engine(*probe_factory, options);
            probe_engine.RegisterUciResponder(&probe_responder);

            auto run_probe_once = [&]() {
                probe_responder.Reset();
                probe_engine.SetPosition(lczero::PositionToFen(current_pos), {});
                lczero::GoParams params;
                params.movetime = move_time_ms;
                probe_engine.Go(params);
                probe_engine.Wait();
                auto stats = probe_responder.GetPolicyStats();
                for (auto& row : stats) {
                    row.move = CanonicalizeMoveToWhitePov(current_pos, row.move);
                }
                return stats;
            };

            auto stats = run_probe_once();
            auto all_zero_posterior = [](const std::vector<BestMoveCaptureResponder::PolicyStat>& rows) {
                if (rows.empty()) return true;
                for (const auto& row : rows) {
                    const double post = row.posterior.value_or(static_cast<double>(row.visits));
                    if (post > 0.0) return false;
                }
                return true;
            };

            // First probe for a just-created engine can spend most time on warmup.
            // Retry once if posterior is entirely zero to get a meaningful table.
            if (all_zero_posterior(stats)) {
                stats = run_probe_once();
            }

            probe_engine.UnregisterUciResponder(&probe_responder);
            return stats;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    };

    if (human_mode) {
        std::cout << "--- STARTING HUMAN VS HUMAN ---" << std::endl;
        std::cout << "Enter moves in white-oriented UCI format (e.g. e2e4, e7e5, e7e8q).\n";
    } else {
        std::cout << "--- STARTING BOT PLAYOUT ---" << std::endl;
    }
    std::cout << "Config: movetime_ms=" << move_time_ms
              << " search=" << search_name
              << " move_overhead_ms=" << move_overhead_ms
              << " threads=" << threads
              << " minibatch_size=" << minibatch_size_effective
              << " max_prefetch=" << max_prefetch_effective << std::endl;

    int half_moves = 0;
    constexpr bool is_chess960 = false;

    while (half_moves < max_half_moves) {
        const lczero::GameResult game_result = history.ComputeGameResult();
        if (game_result != lczero::GameResult::UNDECIDED) {
            std::cout << GameResultMessage(history) << std::endl;
            break;
        }

        const lczero::Position& pos = history.Last();
        const lczero::ChessBoard& board = pos.GetBoard();

        PrintUnicodeBoard(pos);
        std::cout << "Position: " << lczero::PositionToFen(pos) << std::endl;
        std::cout << "Counters: rule50=" << pos.GetRule50Ply()
              << " repetitions=" << pos.GetRepetitions() << std::endl;

        auto legal_moves = board.GenerateLegalMoves();
        if (legal_moves.empty()) {
            std::cout << "Game Over: No legal moves remaining." << std::endl;
            break;
        }

        lczero::Move chosen_move;
        if (human_mode) {
            while (true) {
                const bool white_to_move = !pos.IsBlackToMove();
                std::cout << (white_to_move ? "White move: " : "Black move: ");
                std::string move_text;
                if (!std::getline(std::cin, move_text)) {
                    std::cout << "\nInput closed. Exiting." << std::endl;
                    if (engine) {
                        engine->UnregisterUciResponder(&responder);
                    }
                    return 0;
                }
                const std::string normalized = ToLower(move_text);
                if (normalized == "quit" || normalized == "q" || normalized == "exit") {
                    std::cout << "Exiting." << std::endl;
                    if (engine) {
                        engine->UnregisterUciResponder(&responder);
                    }
                    return 0;
                }
                if (normalized == "show policy") {
                    const auto classic_stats = fetch_policy_stats("classic", pos);
                    const auto rmcts_stats = fetch_policy_stats("rmcts", pos);

                    if (!classic_stats) {
                        std::cout << "Classic search is not available.\n";
                    } else if (!rmcts_stats) {
                        std::cout << "RMCTS search is not available.\n";
                    } else {
                        PrintPolicyComparisonTable(*classic_stats, *rmcts_stats);
                        PrintRmctsDiagnostics(*rmcts_stats);
                    }
                    continue;
                }
                try {
                    auto try_parse = [&](const std::string& text)
                        -> std::optional<lczero::Move> {
                        try {
                            return board.ParseMove(text);
                        } catch (const std::exception&) {
                            return std::nullopt;
                        }
                    };

                    auto parsed = try_parse(move_text);
                    if (!parsed && pos.IsBlackToMove()) {
                        parsed = try_parse(NormalizeMoveToWhitePov(move_text, true));
                    }
                    if (!parsed) {
                        std::cout << "Invalid move format. Use UCI style like e2e4." << std::endl;
                        continue;
                    }
                    chosen_move = *parsed;
                } catch (const std::exception&) {
                    std::cout << "Invalid move format. Use UCI style like e2e4." << std::endl;
                    continue;
                }
                if (std::find(legal_moves.begin(), legal_moves.end(), chosen_move) ==
                    legal_moves.end()) {
                    std::cout << "Illegal move in this position. Try again." << std::endl;
                    continue;
                }
                break;
            }
        } else {
            if (!engine) {
                std::cout << "Game Over: Engine is not available in this mode." << std::endl;
                break;
            }
            responder.Reset();
            engine->SetPosition(lczero::PositionToFen(pos), {});
            lczero::GoParams params;
            params.movetime = move_time_ms;
            engine->Go(params);
            engine->Wait();

            auto best_move = responder.GetBestMove();
            if (!best_move || best_move->is_null()) {
                std::cout << "Game Over: Engine did not return a best move." << std::endl;
                break;
            }
            if (const auto eval_stm = responder.GetEvalStm(); eval_stm) {
                const double eval_white = pos.IsBlackToMove() ? -*eval_stm : *eval_stm;
                std::cout << "Eval (white): " << std::showpos << std::fixed
                          << std::setprecision(3) << eval_white << std::noshowpos
                          << std::defaultfloat << '\n';
            }
            const auto nodes = responder.GetNodes();
            const auto nps = responder.GetNps();
            const auto elapsed_ms = responder.GetTimeMs();
            if (nodes || nps || elapsed_ms) {
                std::cout << "Search: "
                          << "time_ms=" << (elapsed_ms ? std::to_string(*elapsed_ms) : "?")
                          << " nodes=" << (nodes ? std::to_string(*nodes) : "?")
                          << " nps=" << (nps ? std::to_string(*nps) : "?")
                          << '\n';
            }
            chosen_move = *best_move;
            if (pos.IsBlackToMove()) {
                chosen_move.Flip();
            }
        }

        if (std::find(legal_moves.begin(), legal_moves.end(), chosen_move) ==
            legal_moves.end()) {
            std::cout << "Game Over: Engine returned an illegal move." << std::endl;
            break;
        }

        auto display_move = chosen_move;
        if (pos.IsBlackToMove()) {
            display_move.Flip();
        }

        std::cout << "Move played: " << display_move.ToString(is_chess960)
                  << "\n\n";

        if (!human_mode && !no_pause) {
            std::cout << "Press Enter for the next move...";
            std::cin.get();
        }

        history.Append(chosen_move);
        half_moves++;
    }

    if (half_moves >= max_half_moves) {
        std::cout << "Game Over: Reached " << max_half_moves
                  << " half-moves limit." << std::endl;
    }

    if (engine) {
        engine->UnregisterUciResponder(&responder);
    }
    return 0;
}
