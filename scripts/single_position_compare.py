#!/usr/bin/env python3
import argparse
import csv
from contextlib import suppress
import math
import random
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import matplotlib.pyplot as plt


@dataclass
class AnalysisRow:
    ply: int
    fen: str
    turn: str
    chosen_move: str
    rmcts_move: str
    classic_move: str
    policy_move: str
    rmcts_value_white: float
    classic_value_white: float
    rmcts_value_black: float
    classic_value_black: float
    disagree_rmcts_classic: int
    disagree_rmcts_policy: int
    disagree_classic_policy: int


def random_start_fen(rng: random.Random, min_plies: int, max_plies: int) -> str:
    while True:
        board = chess.Board()
        target = rng.randint(min_plies, max_plies)
        ok = True
        for _ in range(target):
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


def score_to_value_white(score: Optional[chess.engine.PovScore]) -> float:
    if score is None:
        return float("nan")
    expectation = score.white().wdl().expectation()
    return 2.0 * expectation - 1.0


def analyze_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    movetime_ms: int,
) -> tuple[chess.Move, float]:
    result = engine.play(
        board,
        chess.engine.Limit(time=movetime_ms / 1000.0),
        info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC,
    )
    if result.move is None or result.move not in board.legal_moves:
        raise RuntimeError("Engine returned invalid move.")
    value_white = score_to_value_white(result.info.get("score"))
    return result.move, value_white


def format_move(move: chess.Move) -> str:
    return move.uci()


def write_csv(path: Path, rows: list[AnalysisRow]) -> None:
    header = (
        "ply,turn,chosen_move,rmcts_move,classic_move,policy_move,"
        "rmcts_value_white,classic_value_white,rmcts_value_black,classic_value_black,"
        "disagree_rmcts_classic,disagree_rmcts_policy,disagree_classic_policy,fen"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(
                f"{r.ply},{r.turn},{r.chosen_move},{r.rmcts_move},{r.classic_move},{r.policy_move},"
                f"{r.rmcts_value_white:.6f},{r.classic_value_white:.6f},"
                f"{r.rmcts_value_black:.6f},{r.classic_value_black:.6f},"
                f"{r.disagree_rmcts_classic},{r.disagree_rmcts_policy},{r.disagree_classic_policy},"
                f'"{r.fen}"\n'
            )


def plot_rows(path: Path, rows: list[AnalysisRow]) -> None:
    if not rows:
        return

    x = [r.ply for r in rows]
    rmcts_w = [r.rmcts_value_white for r in rows]
    classic_w = [r.classic_value_white for r in rows]
    rmcts_b = [r.rmcts_value_black for r in rows]
    classic_b = [r.classic_value_black for r in rows]

    rc_idx = [i for i, r in enumerate(rows) if r.disagree_rmcts_classic]
    rp_idx = [i for i, r in enumerate(rows) if r.disagree_rmcts_policy]
    cp_idx = [i for i, r in enumerate(rows) if r.disagree_classic_policy]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax1.plot(x, rmcts_w, label="RMCTS value (white POV)", color="tab:blue")
    ax1.plot(x, classic_w, label="Classic value (white POV)", color="tab:green")
    for i in rc_idx:
        ax1.axvline(x=x[i], color="tab:orange", alpha=0.18, linewidth=1.2)
    if rp_idx:
        ax1.scatter(
            [x[i] for i in rp_idx],
            [rmcts_w[i] for i in rp_idx],
            marker="x",
            color="tab:red",
            s=45,
            label="RMCTS ≠ policy argmax",
        )
    if cp_idx:
        ax1.scatter(
            [x[i] for i in cp_idx],
            [classic_w[i] for i in cp_idx],
            marker="o",
            facecolors="none",
            edgecolors="tab:purple",
            s=45,
            label="Classic ≠ policy argmax",
        )
    ax1.set_ylabel("Projected value [-1, 1]")
    ax1.set_title("White perspective")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="best")

    ax2.plot(x, rmcts_b, label="RMCTS value (black POV)", color="tab:blue")
    ax2.plot(x, classic_b, label="Classic value (black POV)", color="tab:green")
    for i in rc_idx:
        ax2.axvline(x=x[i], color="tab:orange", alpha=0.18, linewidth=1.2)
    if rp_idx:
        ax2.scatter(
            [x[i] for i in rp_idx],
            [rmcts_b[i] for i in rp_idx],
            marker="x",
            color="tab:red",
            s=45,
            label="RMCTS ≠ policy argmax",
        )
    if cp_idx:
        ax2.scatter(
            [x[i] for i in cp_idx],
            [classic_b[i] for i in cp_idx],
            marker="o",
            facecolors="none",
            edgecolors="tab:purple",
            s=45,
            label="Classic ≠ policy argmax",
        )
    ax2.set_ylabel("Projected value [-1, 1]")
    ax2.set_xlabel("Total plies")
    ax2.set_title("Black perspective")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="best")

    fig.suptitle("Single-position RMCTS vs Classic value trace")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_rows_from_csv(path: Path) -> list[AnalysisRow]:
    rows: list[AnalysisRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                AnalysisRow(
                    ply=int(row["ply"]),
                    fen=row["fen"],
                    turn=row["turn"],
                    chosen_move=row["chosen_move"],
                    rmcts_move=row["rmcts_move"],
                    classic_move=row["classic_move"],
                    policy_move=row["policy_move"],
                    rmcts_value_white=float(row["rmcts_value_white"]),
                    classic_value_white=float(row["classic_value_white"]),
                    rmcts_value_black=float(row["rmcts_value_black"]),
                    classic_value_black=float(row["classic_value_black"]),
                    disagree_rmcts_classic=int(row["disagree_rmcts_classic"]),
                    disagree_rmcts_policy=int(row["disagree_rmcts_policy"]),
                    disagree_classic_policy=int(row["disagree_classic_policy"]),
                )
            )
    return rows


def plot_overlay(path: Path, rmcts_rows: list[AnalysisRow], classic_rows: list[AnalysisRow]) -> None:
    if not rmcts_rows or not classic_rows:
        raise SystemExit("Both CSV inputs must contain at least one row.")

    x_r = [r.ply for r in rmcts_rows]
    x_c = [r.ply for r in classic_rows]
    r_rp = [i for i, r in enumerate(rmcts_rows) if r.disagree_rmcts_policy]
    r_cp = [i for i, r in enumerate(rmcts_rows) if r.disagree_classic_policy]
    c_rp = [i for i, r in enumerate(classic_rows) if r.disagree_rmcts_policy]
    c_cp = [i for i, r in enumerate(classic_rows) if r.disagree_classic_policy]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

    ax1.plot(x_r, [r.rmcts_value_white for r in rmcts_rows], color="tab:blue", label="RMCTS-driven: RMCTS value")
    ax1.plot(x_r, [r.classic_value_white for r in rmcts_rows], color="tab:green", label="RMCTS-driven: Classic value")
    ax1.plot(x_c, [r.rmcts_value_white for r in classic_rows], color="tab:blue", linestyle="--", label="Classic-driven: RMCTS value")
    ax1.plot(x_c, [r.classic_value_white for r in classic_rows], color="tab:green", linestyle="--", label="Classic-driven: Classic value")
    for i, r in enumerate(rmcts_rows):
        if r.disagree_rmcts_classic:
            ax1.axvline(x=x_r[i], color="tab:orange", alpha=0.08, linewidth=1.0)
    for i, r in enumerate(classic_rows):
        if r.disagree_rmcts_classic:
            ax1.axvline(x=x_c[i], color="tab:red", alpha=0.08, linewidth=1.0)
    if r_rp:
        ax1.scatter(
            [x_r[i] for i in r_rp],
            [rmcts_rows[i].rmcts_value_white for i in r_rp],
            marker="x",
            color="tab:red",
            s=35,
            label="RMCTS-driven: RMCTS ≠ policy",
        )
    if r_cp:
        ax1.scatter(
            [x_r[i] for i in r_cp],
            [rmcts_rows[i].classic_value_white for i in r_cp],
            marker="o",
            facecolors="none",
            edgecolors="tab:purple",
            s=35,
            label="RMCTS-driven: Classic ≠ policy",
        )
    if c_rp:
        ax1.scatter(
            [x_c[i] for i in c_rp],
            [classic_rows[i].rmcts_value_white for i in c_rp],
            marker="+",
            color="tab:brown",
            s=35,
            label="Classic-driven: RMCTS ≠ policy",
        )
    if c_cp:
        ax1.scatter(
            [x_c[i] for i in c_cp],
            [classic_rows[i].classic_value_white for i in c_cp],
            marker="s",
            facecolors="none",
            edgecolors="tab:gray",
            s=35,
            label="Classic-driven: Classic ≠ policy",
        )
    ax1.set_ylabel("Projected value [-1, 1]")
    ax1.set_title("White perspective (solid=RMCTS-driven, dashed=Classic-driven)")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="best")

    ax2.plot(x_r, [r.rmcts_value_black for r in rmcts_rows], color="tab:blue", label="RMCTS-driven: RMCTS value")
    ax2.plot(x_r, [r.classic_value_black for r in rmcts_rows], color="tab:green", label="RMCTS-driven: Classic value")
    ax2.plot(x_c, [r.rmcts_value_black for r in classic_rows], color="tab:blue", linestyle="--", label="Classic-driven: RMCTS value")
    ax2.plot(x_c, [r.classic_value_black for r in classic_rows], color="tab:green", linestyle="--", label="Classic-driven: Classic value")
    for i, r in enumerate(rmcts_rows):
        if r.disagree_rmcts_classic:
            ax2.axvline(x=x_r[i], color="tab:orange", alpha=0.08, linewidth=1.0)
    for i, r in enumerate(classic_rows):
        if r.disagree_rmcts_classic:
            ax2.axvline(x=x_c[i], color="tab:red", alpha=0.08, linewidth=1.0)
    if r_rp:
        ax2.scatter(
            [x_r[i] for i in r_rp],
            [rmcts_rows[i].rmcts_value_black for i in r_rp],
            marker="x",
            color="tab:red",
            s=35,
            label="RMCTS-driven: RMCTS ≠ policy",
        )
    if r_cp:
        ax2.scatter(
            [x_r[i] for i in r_cp],
            [rmcts_rows[i].classic_value_black for i in r_cp],
            marker="o",
            facecolors="none",
            edgecolors="tab:purple",
            s=35,
            label="RMCTS-driven: Classic ≠ policy",
        )
    if c_rp:
        ax2.scatter(
            [x_c[i] for i in c_rp],
            [classic_rows[i].rmcts_value_black for i in c_rp],
            marker="+",
            color="tab:brown",
            s=35,
            label="Classic-driven: RMCTS ≠ policy",
        )
    if c_cp:
        ax2.scatter(
            [x_c[i] for i in c_cp],
            [classic_rows[i].classic_value_black for i in c_cp],
            marker="s",
            facecolors="none",
            edgecolors="tab:gray",
            s=35,
            label="Classic-driven: Classic ≠ policy",
        )
    ax2.set_ylabel("Projected value [-1, 1]")
    ax2.set_xlabel("Total plies")
    ax2.set_title("Black perspective (solid=RMCTS-driven, dashed=Classic-driven)")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="best")

    fig.suptitle("Overlay: RMCTS-driven vs Classic-driven single-position traces")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one game line while querying RMCTS, classic, and policyhead "
            "on every position; save per-ply values and disagreement markers."
        )
    )
    parser.add_argument("--engine", default="build/onnxtrt/lc0")
    parser.add_argument("--mode", choices=["analyze", "overlay"], default="analyze")
    parser.add_argument(
        "--shared-args",
        default=(
            "--backend=onnx-trt "
            "--weights=weights/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
        ),
    )
    parser.add_argument(
        "--rmcts-args",
        default="rmcts --rmcts-chunk-sims=128 --backend-opts=gpu=1,batch=16,steps=1",
    )
    parser.add_argument(
        "--classic-args",
        default="classic --minibatch-size=136 --max-prefetch=136 --backend-opts=gpu=0,batch=136,steps=1",
    )
    parser.add_argument(
        "--policy-args",
        default="policyhead --backend-opts=gpu=0,batch=32,steps=1",
    )
    parser.add_argument("--driver", choices=["rmcts", "classic"], default="rmcts")
    parser.add_argument("--movetime-ms", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--opening-plies-min", type=int, default=2)
    parser.add_argument("--opening-plies-max", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-fen", default="")
    parser.add_argument("--out-csv", default="/tmp/single_position_compare.csv")
    parser.add_argument("--out-plot", default="/tmp/single_position_compare.png")
    parser.add_argument("--uci-timeout-s", type=float, default=120.0)
    parser.add_argument("--overlay-rmcts-csv", default="")
    parser.add_argument("--overlay-classic-csv", default="")
    parser.add_argument("--overlay-out-plot", default="/tmp/single_position_compare_overlay.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "overlay":
        if not args.overlay_rmcts_csv or not args.overlay_classic_csv:
            raise SystemExit("--overlay-rmcts-csv and --overlay-classic-csv are required in overlay mode")
        rmcts_rows = read_rows_from_csv(Path(args.overlay_rmcts_csv))
        classic_rows = read_rows_from_csv(Path(args.overlay_classic_csv))
        out_plot = Path(args.overlay_out_plot)
        plot_overlay(out_plot, rmcts_rows, classic_rows)
        print(f"Overlay plot: {out_plot}")
        print(f"RMCTS CSV rows: {len(rmcts_rows)}")
        print(f"Classic CSV rows: {len(classic_rows)}")
        return 0

    if args.movetime_ms <= 0:
        raise SystemExit("--movetime-ms must be > 0")
    if args.max_plies <= 0:
        raise SystemExit("--max-plies must be > 0")
    if args.uci_timeout_s <= 0:
        raise SystemExit("--uci-timeout-s must be > 0")
    if args.opening_plies_min < 0 or args.opening_plies_max < 0:
        raise SystemExit("opening plies must be >= 0")
    if args.opening_plies_min > args.opening_plies_max:
        raise SystemExit("--opening-plies-min cannot exceed --opening-plies-max")

    shared = shlex.split(args.shared_args)
    rmcts_cmd = [*shlex.split(args.rmcts_args), *shared]
    classic_cmd = [*shlex.split(args.classic_args), *shared]
    policy_cmd = [*shlex.split(args.policy_args), *shared]

    rng = random.Random(args.seed)
    start_fen = args.start_fen.strip() or random_start_fen(
        rng, args.opening_plies_min, args.opening_plies_max
    )
    board = chess.Board(start_fen)
    rows: list[AnalysisRow] = []

    print(f"Engine: {args.engine}")
    print(f"RMCTS args: {' '.join(rmcts_cmd)}")
    print(f"Classic args: {' '.join(classic_cmd)}")
    print(f"Policy args: {' '.join(policy_cmd)}")
    print(f"Start FEN: {start_fen}")
    print(f"Driver: {args.driver}")

    rmcts = chess.engine.SimpleEngine.popen_uci(
        [args.engine, *rmcts_cmd], timeout=args.uci_timeout_s
    )
    classic = chess.engine.SimpleEngine.popen_uci(
        [args.engine, *classic_cmd], timeout=args.uci_timeout_s
    )
    policy = chess.engine.SimpleEngine.popen_uci(
        [args.engine, *policy_cmd], timeout=args.uci_timeout_s
    )
    try:
        for eng in (rmcts, classic, policy):
            eng.protocol.send_line("ucinewgame")

        for ply in range(args.max_plies):
            if board.is_game_over(claim_draw=True):
                break

            rmcts_move, rmcts_w = analyze_position(rmcts, board, args.movetime_ms)
            classic_move, classic_w = analyze_position(classic, board, args.movetime_ms)
            policy_move, _policy_w = analyze_position(policy, board, args.movetime_ms)

            chosen = rmcts_move if args.driver == "rmcts" else classic_move
            if chosen not in board.legal_moves:
                raise RuntimeError("Chosen move is illegal.")

            row = AnalysisRow(
                ply=ply,
                fen=board.fen(),
                turn="w" if board.turn == chess.WHITE else "b",
                chosen_move=format_move(chosen),
                rmcts_move=format_move(rmcts_move),
                classic_move=format_move(classic_move),
                policy_move=format_move(policy_move),
                rmcts_value_white=rmcts_w,
                classic_value_white=classic_w,
                rmcts_value_black=-rmcts_w if not math.isnan(rmcts_w) else float("nan"),
                classic_value_black=-classic_w if not math.isnan(classic_w) else float("nan"),
                disagree_rmcts_classic=int(rmcts_move != classic_move),
                disagree_rmcts_policy=int(rmcts_move != policy_move),
                disagree_classic_policy=int(classic_move != policy_move),
            )
            rows.append(row)
            board.push(chosen)

        out_csv = Path(args.out_csv)
        out_plot = Path(args.out_plot)
        write_csv(out_csv, rows)
        plot_rows(out_plot, rows)

        disagree_rc = sum(r.disagree_rmcts_classic for r in rows)
        disagree_rp = sum(r.disagree_rmcts_policy for r in rows)
        disagree_cp = sum(r.disagree_classic_policy for r in rows)

        print(f"Plies analyzed: {len(rows)}")
        print(f"RMCTS vs Classic disagreements: {disagree_rc}")
        print(f"RMCTS vs Policy disagreements: {disagree_rp}")
        print(f"Classic vs Policy disagreements: {disagree_cp}")
        print(f"CSV: {out_csv}")
        print(f"Plot: {out_plot}")
        if board.is_game_over(claim_draw=True):
            print(f"Game result: {board.result(claim_draw=True)}")
        else:
            print("Game result: unfinished (max plies reached)")
    finally:
        for eng in (rmcts, classic, policy):
            with suppress(chess.engine.EngineTerminatedError):
                eng.quit()
            eng.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
