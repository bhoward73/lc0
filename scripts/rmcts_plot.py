#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
import colorsys
import sys

import matplotlib.pyplot as plt

from rmcts_params import (
    apply_known_defaults,
    load_config,
    parse_config_path,
    section_defaults,
)


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "chunk": int(row["chunk"]),
                    "fen": row["fen"],
                    "move": row["move"],
                    "prior": float(row["prior"]),
                    "posterior": float(row["posterior"]),
                    "q": float(row["q"]),
                    "n_total": float(row["n_total"]),
                    "classic_posterior": float(row["classic_posterior"]) if "classic_posterior" in row and row["classic_posterior"] else None,
                }
            )
    return rows


def select_fen(rows, fen_filter=None, latest_fen=False):
    if fen_filter is not None:
        return fen_filter
    if not latest_fen:
        return None
    # Default to the most recent root position encountered in the trace.
    return rows[-1]["fen"]


def select_focus_moves(rows):
    by_move = defaultdict(list)
    for row in rows:
        by_move[row["move"]].append(row)

    latest_chunk = max(r["chunk"] for r in rows)
    latest_rows = [r for r in rows if r["chunk"] == latest_chunk]

    top_prior = max(by_move.keys(), key=lambda m: max(r["prior"] for r in by_move[m]))
    top_rmcts = max(latest_rows, key=lambda r: r["posterior"])["move"]

    selected = [top_prior]
    if top_rmcts not in selected:
        selected.append(top_rmcts)

    classic_rows = [r for r in latest_rows if r.get("classic_posterior") is not None]
    if classic_rows:
        top_classic = max(classic_rows, key=lambda r: r["classic_posterior"])["move"]
        if top_classic not in selected:
            selected.append(top_classic)

    return selected[:3]


def plot_metric(
    rows,
    metric,
    title,
    ylabel,
    output_path=None,
    show_initial_prior=False,
    focus_moves=None,
):
    by_move = defaultdict(list)
    prior0_by_move = {}
    for row in rows:
        if focus_moves is not None and row["move"] not in focus_moves:
            continue
        by_move[row["move"]].append((row["chunk"], row[metric]))
        if row["chunk"] == 0 and row["move"] not in prior0_by_move:
            prior0_by_move[row["move"]] = row["prior"]

    plt.figure(figsize=(12, 8))

    moves_sorted = sorted(by_move.keys())

    # Build a larger distinct palette than matplotlib's default cycle.
    # First use tab20 (good categorical contrast), then extend with evenly
    # spaced HSV hues for additional series.
    tab20 = list(plt.get_cmap("tab20").colors)
    extra_needed = max(0, len(moves_sorted) - len(tab20))
    hsv_extra = []
    if extra_needed > 0:
      for i in range(extra_needed):
          h = (i / max(1, extra_needed)) % 1.0
          s = 0.75
          v = 0.85
          hsv_extra.append(colorsys.hsv_to_rgb(h, s, v))
    color_palette = tab20 + hsv_extra

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

    for idx, move in enumerate(moves_sorted):
        pts = by_move[move]
        pts.sort(key=lambda x: x[0])
        xs = [p[0] + 1 for p in pts]
        ys = [p[1] for p in pts]
        color = color_palette[idx]
        linestyle = line_styles[(idx // len(markers)) % len(line_styles)]
        marker = markers[idx % len(markers)]
        line, = plt.plot(
            xs,
            ys,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4,
            color=color,
            label=move,
        )
        color = line.get_color()

        if show_initial_prior and metric == "posterior":
            initial_prior = prior0_by_move.get(move)
            if initial_prior is not None:
                px = [0] + xs
                py = [initial_prior] + ys
                plt.plot(
                    px,
                    py,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=4,
                    color=color,
                )

    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=140)
    else:
        plt.show()


def main():
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Plot RMCTS per-chunk trajectories from RMCTS_TRACE_CSV output"
    )
    parser.add_argument(
        "--config",
        default=str(config_path),
        help="Path to shared RMCTS TOML config file.",
    )
    parser.add_argument("csv", nargs="?", default=None, help="Path to RMCTS trace CSV")
    parser.add_argument(
        "--fen",
        default=None,
        help="Optional exact FEN filter (plots only rows matching this root position)",
    )
    parser.add_argument(
        "--all-fens",
        action="store_true",
        help="Plot rows from all FENs in the CSV (default is latest FEN only).",
    )
    parser.add_argument(
        "--all-moves",
        action="store_true",
        help="Plot all moves instead of default focus view (max 3 key moves).",
    )
    parser.add_argument(
        "--metric",
        choices=["posterior", "q", "n_total", "prior", "all"],
        default="posterior",
        help="Metric to plot",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="If set, save PNG(s) with this prefix instead of showing interactively",
    )
    parser.add_argument(
        "--no-initial-prior",
        action="store_true",
        help="For posterior plots, disable plotting initial prior point at chunk 0.",
    )

    apply_known_defaults(parser, section_defaults(config, "rmcts_plot"))
    args = parser.parse_args()

    if not args.csv:
        raise SystemExit("CSV path is required. Set it in config [rmcts_plot].csv or pass it on CLI.")

    rows = load_rows(args.csv)
    selected_fen = select_fen(
        rows,
        fen_filter=args.fen,
        latest_fen=not args.all_fens,
    )
    if selected_fen is not None:
        rows = [r for r in rows if r["fen"] == selected_fen]
        print(f"Plotting FEN: {selected_fen}")
    else:
        print("Plotting all FENs in trace.")

    if not rows:
        raise SystemExit("No rows to plot. Check CSV path/FEN filter.")

    show_initial_prior = not args.no_initial_prior
    focus_moves = None if args.all_moves else select_focus_moves(rows)
    if focus_moves is not None:
        print("Plotting focus moves:", ", ".join(focus_moves))
        if not any(r.get("classic_posterior") is not None for r in rows):
            print("Note: classic posterior is not present in this CSV; focus uses prior + rmcts posterior.")

    if args.metric == "all":
        plot_metric(
            rows,
            "posterior",
            "RMCTS Posterior by Chunk (all legal moves)",
            "Posterior",
            f"{args.out_prefix}_posterior.png" if args.out_prefix else None,
            show_initial_prior=show_initial_prior,
            focus_moves=focus_moves,
        )
        plot_metric(
            rows,
            "q",
            "RMCTS Q by Chunk (all legal moves)",
            "Q",
            f"{args.out_prefix}_q.png" if args.out_prefix else None,
            focus_moves=focus_moves,
        )
        plot_metric(
            rows,
            "n_total",
            "RMCTS Total Root Visits by Chunk (all legal moves)",
            "N total",
            f"{args.out_prefix}_n_total.png" if args.out_prefix else None,
            focus_moves=focus_moves,
        )
    else:
        titles = {
            "posterior": "RMCTS Posterior by Chunk (all legal moves)",
            "q": "RMCTS Q by Chunk (all legal moves)",
            "n_total": "RMCTS Total Root Visits by Chunk (all legal moves)",
            "prior": "RMCTS Prior by Chunk (all legal moves)",
        }
        ylabels = {
            "posterior": "Posterior",
            "q": "Q",
            "n_total": "N total",
            "prior": "Prior",
        }
        out = f"{args.out_prefix}_{args.metric}.png" if args.out_prefix else None
        plot_metric(
            rows,
            args.metric,
            titles[args.metric],
            ylabels[args.metric],
            out,
            show_initial_prior=show_initial_prior,
            focus_moves=focus_moves,
        )


if __name__ == "__main__":
    main()
