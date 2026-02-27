#!/usr/bin/env python3
import argparse
import random
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rmcts_params import (
    apply_known_defaults,
    load_config,
    parse_config_path,
    section_defaults,
)

try:
    import chess
    import chess.engine
except Exception as exc:  # pragma: no cover
    print(
        "Missing dependency: python-chess. Install with: pip install chess\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    sys.exit(2)


@dataclass
class MatchConfig:
    positions: int
    movetime_ms: int
    opening_plies_min: int
    opening_plies_max: int
    max_plies: int
    seed: int


@dataclass
class Tally:
    a_wins: int = 0
    draws: int = 0
    b_wins: int = 0
    a_sims: int = 0
    b_sims: int = 0

    def add(self, score_from_a: float) -> None:
        if score_from_a > 0:
            self.a_wins += 1
        elif score_from_a < 0:
            self.b_wins += 1
        else:
            self.draws += 1

    @property
    def games(self) -> int:
        return self.a_wins + self.draws + self.b_wins

    @property
    def score(self) -> float:
        return self.a_wins + 0.5 * self.draws


class EnginePair:
    def __init__(self, engine_path: str, a_args: list[str], b_args: list[str]) -> None:
        self.a = chess.engine.SimpleEngine.popen_uci([engine_path, *a_args])
        self.b = chess.engine.SimpleEngine.popen_uci([engine_path, *b_args])

    def close(self) -> None:
        for engine in (self.a, self.b):
            try:
                engine.quit()
            except Exception:
                try:
                    engine.close()
                except Exception:
                    pass


def random_start_fen(rng: random.Random, min_plies: int, max_plies: int) -> str:
    while True:
        board = chess.Board()
        target = rng.randint(min_plies, max_plies)
        ok = True
        for ply in range(target):
            if board.is_game_over(claim_draw=True):
                ok = False
                break
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if ok and not board.is_game_over(claim_draw=True):
            return board.fen()


def score_from_white_result(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner else -1.0


def play_one_game(
    white: chess.engine.SimpleEngine,
    black: chess.engine.SimpleEngine,
    start_fen: str,
    movetime_ms: int,
    max_plies: int,
) -> tuple[float, int, int]:
    board = chess.Board(start_fen)
    white.protocol.send_line("ucinewgame")
    black.protocol.send_line("ucinewgame")
    white_sims = 0
    black_sims = 0

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white if board.turn == chess.WHITE else black
        limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        try:
            result = engine.play(board, limit, info=chess.engine.INFO_BASIC)
        except Exception:
            return (
                (-1.0 if board.turn == chess.WHITE else 1.0),
                white_sims,
                black_sims,
            )
        nodes = int(result.info.get("nodes", 0)) if getattr(result, "info", None) else 0
        if board.turn == chess.WHITE:
            white_sims += nodes
        else:
            black_sims += nodes
        move = result.move
        if move is None or move not in board.legal_moves:
            return (
                (-1.0 if board.turn == chess.WHITE else 1.0),
                white_sims,
                black_sims,
            )
        board.push(move)

    if not board.is_game_over(claim_draw=True):
        return 0.0, white_sims, black_sims
    return score_from_white_result(board), white_sims, black_sims


def parse_args(argv: list[str]) -> argparse.Namespace:
    config_path = parse_config_path(argv)
    config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description=(
            "Run paired bot-vs-bot matches from random starting positions. "
            "For each start, play two games with colors swapped."
        )
    )
    parser.add_argument(
        "--config",
        default=str(config_path),
        help="Path to shared RMCTS TOML config file.",
    )
    parser.add_argument("--engine", default="build/onnxtrt/lc0")
    parser.add_argument(
        "--shared-args",
        default="--backend=onnx-trt --backend-opts=batch=136,steps=1",
    )
    parser.add_argument("--a-args", default="rmcts")
    parser.add_argument("--b-args", default="classic --minibatch-size=136 --max-prefetch=136")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--movetime-ms", type=int, default=300)
    parser.add_argument("--opening-plies-min", type=int, default=6)
    parser.add_argument("--opening-plies-max", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=220)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label-a", default="rmcts")
    parser.add_argument("--label-b", default="classic")

    apply_known_defaults(parser, section_defaults(config, "bot_match"))
    args = parser.parse_args(argv)
    args.config = str(Path(args.config))
    return args


def main() -> int:
    args = parse_args(sys.argv[1:])

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.positions <= 0:
        print("--positions must be > 0", file=sys.stderr)
        return 2
    if args.movetime_ms <= 0:
        print("--movetime-ms must be > 0", file=sys.stderr)
        return 2
    if args.opening_plies_min < 0 or args.opening_plies_max < 0:
        print("opening ply limits must be >= 0", file=sys.stderr)
        return 2
    if args.opening_plies_min > args.opening_plies_max:
        print("--opening-plies-min cannot exceed --opening-plies-max", file=sys.stderr)
        return 2

    cfg = MatchConfig(
        positions=args.positions,
        movetime_ms=args.movetime_ms,
        opening_plies_min=args.opening_plies_min,
        opening_plies_max=args.opening_plies_max,
        max_plies=args.max_plies,
        seed=args.seed,
    )

    shared = shlex.split(args.shared_args)
    a_cmd = [*shlex.split(args.a_args), *shared]
    b_cmd = [*shlex.split(args.b_args), *shared]

    rng = random.Random(cfg.seed)
    tally = Tally()

    print(f"Engine: {args.engine}", flush=True)
    print(f"Config file: {args.config}", flush=True)
    print(f"A ({args.label_a}) args: {' '.join(a_cmd)}", flush=True)
    print(f"B ({args.label_b}) args: {' '.join(b_cmd)}", flush=True)
    print(
        f"Config: positions={cfg.positions} movetime_ms={cfg.movetime_ms} "
        f"opening_plies=[{cfg.opening_plies_min},{cfg.opening_plies_max}] "
        f"max_plies={cfg.max_plies} seed={cfg.seed}"
    , flush=True)

    pair_idx = 0
    engines: Optional[EnginePair] = None
    try:
        engines = EnginePair(args.engine, a_cmd, b_cmd)
        for i in range(cfg.positions):
            start_fen = random_start_fen(
                rng, cfg.opening_plies_min, cfg.opening_plies_max
            )
            pair_idx = i + 1

            g1_white, g1_white_sims, g1_black_sims = play_one_game(
                engines.a, engines.b, start_fen, cfg.movetime_ms, cfg.max_plies
            )
            tally.add(g1_white)
            tally.a_sims += g1_white_sims
            tally.b_sims += g1_black_sims

            g2_white, g2_white_sims, g2_black_sims = play_one_game(
                engines.b, engines.a, start_fen, cfg.movetime_ms, cfg.max_plies
            )
            tally.add(-g2_white)
            tally.a_sims += g2_black_sims
            tally.b_sims += g2_white_sims

            pair_score = (1.0 if g1_white > 0 else 0.5 if g1_white == 0 else 0.0) + (
                1.0 if -g2_white > 0 else 0.5 if -g2_white == 0 else 0.0
            )
            print(
                f"pair {pair_idx:03d}: score({args.label_a})={pair_score:.1f}/2 "
                f"running={tally.score:.1f}/{tally.games} "
                f"wdl={tally.a_wins}/{tally.draws}/{tally.b_wins} "
                f"sims({args.label_a}/{args.label_b})={tally.a_sims}/{tally.b_sims}"
            , flush=True)

    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    finally:
        if engines is not None:
            engines.close()

    print("\nFinal tally", flush=True)
    print(f"{args.label_a} wins: {tally.a_wins}", flush=True)
    print(f"draws: {tally.draws}", flush=True)
    print(f"{args.label_b} wins: {tally.b_wins}", flush=True)
    print(f"{args.label_a} total sims: {tally.a_sims}", flush=True)
    print(f"{args.label_b} total sims: {tally.b_sims}", flush=True)
    if tally.b_sims > 0:
        print(
            f"sim ratio ({args.label_a}/{args.label_b}) = {tally.a_sims / tally.b_sims:.3f}",
            flush=True,
        )
    if tally.games > 0:
        pct = 100.0 * tally.score / tally.games
        print(
            f"score({args.label_a}) = {tally.score:.1f}/{tally.games} "
            f"({pct:.2f}%)"
        , flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
