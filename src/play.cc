
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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
              << "  --minibatch-size=N  Search minibatch size (0 = auto)\n"
              << "  --max-prefetch=N    Search prefetch batch size\n"
              << "  --max-half-moves=N  Stop after N half-moves (plies)\n"
              << "  --human             Enable human-vs-bot mode\n"
              << "  --no-pause          In bot mode, do not pause between moves\n\n"
              << "Notes:\n"
              << "  - Default search movetime is 300 ms per move.\n"
              << "  - Default move overhead is 0 ms in this app.\n"
              << "  - In human mode, type moves in UCI format (e.g. e2e4, e7e8q).\n"
              << "  - You can type 'show policy' on your turn to inspect root policy.\n";
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::optional<bool> PromptHumanSide() {
    while (true) {
        std::cout << "Play as which side? (white/black): ";
        std::string side;
        if (!std::getline(std::cin, side)) return std::nullopt;
        side = ToLower(side);
        if (side == "white" || side == "w") return true;
        if (side == "black" || side == "b") return false;
        std::cout << "Please enter 'white' or 'black'.\n";
    }
}

bool IsHumanTurn(const lczero::Position& pos, bool human_is_white) {
    const bool white_to_move = !pos.IsBlackToMove();
    return human_is_white ? white_to_move : !white_to_move;
}

void PrintPolicyTable(std::vector<BestMoveCaptureResponder::PolicyStat> stats) {
    if (stats.empty()) {
        std::cout << "No policy stats available yet.\n";
        return;
    }
    std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
        if (a.prior != b.prior) return a.prior > b.prior;
        return a.move < b.move;
    });

    std::uint64_t total_visits = 0;
    for (const auto& row : stats) total_visits += row.visits;

    std::cout << "Policy (ranked by prior):\n";
    std::cout << std::left << std::setw(8) << "move"
              << std::right << std::setw(10) << "prior"
              << std::setw(10) << "post"
              << std::setw(10) << "visits" << '\n';
    for (const auto& row : stats) {
        const double posterior = total_visits == 0
                                     ? 0.0
                                     : static_cast<double>(row.visits) /
                                           static_cast<double>(total_visits);
        std::cout << std::left << std::setw(8) << row.move
                  << std::right << std::setw(10) << std::fixed
                  << std::setprecision(4) << row.prior
                  << std::setw(10) << posterior
                  << std::setw(10) << row.visits << std::defaultfloat << '\n';
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
    if (!factory) {
        std::cerr << "Search algorithm '" << search_name
                  << "' is not available." << std::endl;
        return 1;
    }

    lczero::OptionsParser options_parser;
    lczero::Engine::PopulateOptions(&options_parser);
    factory->PopulateParams(&options_parser);
    lczero::SharedBackendParams::Populate(&options_parser);

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
    if (const auto backend =
            ParseStringFlag(argc, argv, "--backend=");
        backend) {
        options_parser.GetMutableDefaultsOptions()->Set<std::string>(
            lczero::SharedBackendParams::kBackendId, *backend);
    }
    if (const auto backend_opts =
            ParseStringFlag(argc, argv, "--backend-opts=");
        backend_opts) {
        options_parser.GetMutableDefaultsOptions()->Set<std::string>(
            lczero::SharedBackendParams::kBackendOptionsId, *backend_opts);
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
    bool human_is_white = true;
    if (human_mode) {
        auto selected_side = PromptHumanSide();
        if (!selected_side) return 0;
        human_is_white = *selected_side;
    }

    BestMoveCaptureResponder responder;
    lczero::Engine engine(*factory, options);
    engine.RegisterUciResponder(&responder);

    if (human_mode) {
        std::cout << "--- STARTING HUMAN VS BOT ---" << std::endl;
        std::cout << "Enter moves in UCI format (e.g. e2e4, e7e8q).\n";
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
        if (human_mode && IsHumanTurn(pos, human_is_white)) {
            while (true) {
                std::cout << "Your move: ";
                std::string move_text;
                if (!std::getline(std::cin, move_text)) {
                    std::cout << "\nInput closed. Exiting." << std::endl;
                    engine.UnregisterUciResponder(&responder);
                    return 0;
                }
                const std::string normalized = ToLower(move_text);
                if (normalized == "show policy") {
                    responder.Reset();
                    engine.SetPosition(lczero::PositionToFen(pos), {});
                    lczero::GoParams params;
                    params.movetime = move_time_ms;
                    engine.Go(params);
                    engine.Wait();
                    PrintPolicyTable(responder.GetPolicyStats());
                    continue;
                }
                try {
                    chosen_move = board.ParseMove(move_text);
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
            responder.Reset();
            engine.SetPosition(lczero::PositionToFen(pos), {});
            lczero::GoParams params;
            params.movetime = move_time_ms;
            engine.Go(params);
            engine.Wait();

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

    engine.UnregisterUciResponder(&responder);
    return 0;
}
