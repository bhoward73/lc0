#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ExperimentConfig:
    weight: Path
    movetime_ms: int
    rmcts_gpu: int
    classic_gpu: int

    @property
    def key(self) -> str:
        return (
            f"w={self.weight.name}|t={self.movetime_ms}ms|"
            f"rmcts_gpu={self.rmcts_gpu}|classic_gpu={self.classic_gpu}"
        )


@dataclass
class RunResult:
    rmcts_wins: int
    draws: int
    classic_wins: int
    rmcts_sims: int
    classic_sims: int

    @property
    def games(self) -> int:
        return self.rmcts_wins + self.draws + self.classic_wins


@dataclass
class Aggregate:
    runs: int = 0
    games: int = 0
    rmcts_wins: int = 0
    draws: int = 0
    classic_wins: int = 0
    rmcts_sims: int = 0
    classic_sims: int = 0

    def add(self, run: RunResult) -> None:
        self.runs += 1
        self.games += run.games
        self.rmcts_wins += run.rmcts_wins
        self.draws += run.draws
        self.classic_wins += run.classic_wins
        self.rmcts_sims += run.rmcts_sims
        self.classic_sims += run.classic_sims


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("No integer values provided")
    return values


def parse_gpu_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            left, right = chunk.split(",")
            pairs.append((int(left.strip()), int(right.strip())))
        except Exception as exc:
            raise ValueError(
                "GPU pairs must look like '0,1;1,0'"
            ) from exc
    if not pairs:
        raise ValueError("No GPU pairs provided")
    return pairs


def resolve_weights(weights_dir: Path, explicit: str | None) -> list[Path]:
    if explicit:
        files = [Path(p.strip()) for p in explicit.split(",") if p.strip()]
    else:
        files = sorted(weights_dir.glob("*.pb.gz"))

    resolved: list[Path] = []
    for f in files:
        if not f.is_absolute():
            f = (weights_dir / f).resolve() if not str(f).startswith(str(weights_dir)) else f
        if not f.exists():
            raise FileNotFoundError(f"Weights file not found: {f}")
        resolved.append(f)

    if len(resolved) < 2:
        raise ValueError(
            "Need at least two weights files. Pass --weights or add files to weights/."
        )
    return resolved


def next_deadline_et(stop_hour: int) -> dt.datetime:
    now_et = dt.datetime.now(ET)
    deadline = now_et.replace(hour=stop_hour, minute=0, second=0, microsecond=0)
    if now_et >= deadline:
        deadline += dt.timedelta(days=1)
    return deadline


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(fp, message: str) -> None:
    line = f"[{now_stamp()}] {message}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def parse_bot_match_result(output_lines: Iterable[str]) -> RunResult | None:
    text = "\n".join(output_lines)

    patterns = {
        "rmcts_wins": r"^rmcts wins:\s*(\d+)\s*$",
        "draws": r"^draws:\s*(\d+)\s*$",
        "classic_wins": r"^classic wins:\s*(\d+)\s*$",
        "rmcts_sims": r"^rmcts total sims:\s*(\d+)\s*$",
        "classic_sims": r"^classic total sims:\s*(\d+)\s*$",
    }

    values: dict[str, int] = {}
    for key, pat in patterns.items():
        matches = re.findall(pat, text, flags=re.MULTILINE)
        if not matches:
            return None
        values[key] = int(matches[-1])

    return RunResult(
        rmcts_wins=values["rmcts_wins"],
        draws=values["draws"],
        classic_wins=values["classic_wins"],
        rmcts_sims=values["rmcts_sims"],
        classic_sims=values["classic_sims"],
    )


def build_bot_match_cmd(
    python_exe: str,
    script_path: Path,
    engine_path: str,
    cfg: ExperimentConfig,
    rmcts_cpuct: float,
    positions_per_run: int,
    max_plies: int,
    opening_min: int,
    opening_max: int,
    seed: int,
) -> list[str]:
    shared_args = f"--backend=onnx-trt --weights={cfg.weight}"
    a_args = (
        "rmcts "
        "--rmcts-chunk-sims=128 "
        f"--rmcts-cpuct={rmcts_cpuct:g} "
        f"--backend-opts=gpu={cfg.rmcts_gpu},batch=16,steps=1"
    )
    b_args = (
        "classic "
        "--minibatch-size=136 "
        "--max-prefetch=136 "
        f"--backend-opts=gpu={cfg.classic_gpu},batch=136,steps=1"
    )

    return [
        python_exe,
        str(script_path),
        "--engine",
        engine_path,
        "--shared-args",
        shared_args,
        "--a-args",
        a_args,
        "--b-args",
        b_args,
        "--positions",
        str(positions_per_run),
        "--movetime-ms",
        str(cfg.movetime_ms),
        "--opening-plies-min",
        str(opening_min),
        "--opening-plies-max",
        str(opening_max),
        "--max-plies",
        str(max_plies),
        "--seed",
        str(seed),
        "--label-a",
        "rmcts",
        "--label-b",
        "classic",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Overnight RMCTS-vs-classic orchestrator. Cycles weight/time/GPU settings, "
            "logs verbosely, tracks WDL aggregates, and stops at a given ET hour."
        )
    )
    parser.add_argument("--engine", default="build/release/lc0")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Comma-separated weights files (absolute or relative to weights dir). "
            "If omitted, uses all *.pb.gz in weights dir."
        ),
    )
    parser.add_argument("--movetimes", default="10000")
    parser.add_argument(
        "--rmcts-cpuct",
        type=float,
        default=10.0,
        help="RMCTS c_puct to pass to engine (default: 10.0).",
    )
    parser.add_argument(
        "--gpu-pairs",
        default="0,1;1,0",
        help="Semicolon-separated pairs rmcts_gpu,classic_gpu (e.g. '0,1;1,0').",
    )
    parser.add_argument("--positions-per-run", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=220)
    parser.add_argument("--opening-plies-min", type=int, default=6)
    parser.add_argument("--opening-plies-max", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--stop-hour-et", type=int, default=6)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Optional max full cycles through all configs (0 = until deadline).",
    )
    parser.add_argument("--log-file", default="")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    bot_match_script = repo_root / "scripts" / "bot_match.py"
    if not bot_match_script.exists():
        raise FileNotFoundError(f"Missing script: {bot_match_script}")

    weights_dir = (repo_root / args.weights_dir).resolve()
    if not weights_dir.exists():
        raise FileNotFoundError(f"Missing weights dir: {weights_dir}")

    movetimes = parse_int_list(args.movetimes)
    gpu_pairs = parse_gpu_pairs(args.gpu_pairs)
    weights = resolve_weights(weights_dir, args.weights)

    configs: list[ExperimentConfig] = [
        ExperimentConfig(weight=w, movetime_ms=t, rmcts_gpu=g0, classic_gpu=g1)
        for w, t, (g0, g1) in itertools.product(weights, movetimes, gpu_pairs)
    ]

    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_file) if args.log_file else Path(f"/tmp/rmcts_overnight_{stamp}.log")
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path(f"/tmp/rmcts_overnight_summary_{stamp}.csv")
    )

    deadline = next_deadline_et(args.stop_hour_et)

    agg: dict[str, Aggregate] = {cfg.key: Aggregate() for cfg in configs}

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(
            [
                "timestamp",
                "cycle",
                "run_index",
                "weight",
                "movetime_ms",
                "rmcts_gpu",
                "classic_gpu",
                "seed",
                "status",
                "rmcts_wins",
                "draws",
                "classic_wins",
                "rmcts_sims",
                "classic_sims",
                "games",
            ]
        )

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log_fp:
        log_line(log_fp, "=== RMCTS overnight run starting ===")
        log_line(log_fp, f"Log file: {log_file}")
        log_line(log_fp, f"Summary CSV: {summary_csv}")
        log_line(log_fp, f"Deadline (ET): {deadline.isoformat()}")
        log_line(log_fp, f"Configs: {len(configs)}")

        for cfg in configs:
            log_line(log_fp, f"  - {cfg.key}")

        if args.dry_run:
            log_line(log_fp, "Dry-run mode enabled; commands will not execute.")

        run_index = 0
        cycle = 0
        stop = False

        while not stop:
            cycle += 1
            if args.max_cycles > 0 and cycle > args.max_cycles:
                log_line(log_fp, "Reached max cycles; stopping.")
                break

            for cfg in configs:
                now_et = dt.datetime.now(ET)
                if now_et >= deadline:
                    log_line(log_fp, "Reached deadline before next config; stopping.")
                    stop = True
                    break

                run_index += 1
                seed = args.seed_base + run_index
                cmd = build_bot_match_cmd(
                    python_exe=sys.executable,
                    script_path=bot_match_script,
                    engine_path=args.engine,
                    cfg=cfg,
                    rmcts_cpuct=args.rmcts_cpuct,
                    positions_per_run=args.positions_per_run,
                    max_plies=args.max_plies,
                    opening_min=args.opening_plies_min,
                    opening_max=args.opening_plies_max,
                    seed=seed,
                )

                log_line(
                    log_fp,
                    f"START cycle={cycle} run={run_index} cfg={cfg.key} seed={seed}",
                )
                log_line(log_fp, "CMD: " + " ".join(shlex.quote(x) for x in cmd))

                if args.dry_run:
                    status = "dry-run"
                    result = None
                else:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    captured: list[str] = []
                    status = "ok"
                    terminated_for_deadline = False

                    assert proc.stdout is not None
                    while True:
                        line = proc.stdout.readline()
                        if line:
                            line = line.rstrip("\n")
                            captured.append(line)
                            log_line(log_fp, f"[run {run_index}] {line}")

                        if proc.poll() is not None:
                            for tail in proc.stdout.readlines():
                                tail = tail.rstrip("\n")
                                captured.append(tail)
                                log_line(log_fp, f"[run {run_index}] {tail}")
                            break

                        if dt.datetime.now(ET) >= deadline:
                            terminated_for_deadline = True
                            status = "deadline-stop"
                            log_line(log_fp, "Deadline reached during run; terminating subprocess.")
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                log_line(log_fp, "Subprocess did not exit after terminate; killing.")
                                proc.kill()
                            break

                        time.sleep(0.2)

                    if terminated_for_deadline:
                        stop = True

                    rc = proc.wait()
                    if rc != 0 and status == "ok":
                        status = f"rc={rc}"

                    result = parse_bot_match_result(captured)

                if result and status == "ok":
                    agg[cfg.key].add(result)
                    log_line(
                        log_fp,
                        (
                            f"RESULT run={run_index} cfg={cfg.key} "
                            f"wdl={result.rmcts_wins}/{result.draws}/{result.classic_wins} "
                            f"sims={result.rmcts_sims}/{result.classic_sims}"
                        ),
                    )
                else:
                    log_line(log_fp, f"RESULT run={run_index} cfg={cfg.key} status={status}")

                with summary_csv.open("a", newline="", encoding="utf-8") as sf:
                    writer = csv.writer(sf)
                    writer.writerow(
                        [
                            now_stamp(),
                            cycle,
                            run_index,
                            cfg.weight.name,
                            cfg.movetime_ms,
                            cfg.rmcts_gpu,
                            cfg.classic_gpu,
                            seed,
                            status,
                            result.rmcts_wins if result else "",
                            result.draws if result else "",
                            result.classic_wins if result else "",
                            result.rmcts_sims if result else "",
                            result.classic_sims if result else "",
                            result.games if result else "",
                        ]
                    )

                total_agg_games = sum(a.games for a in agg.values())
                total_agg_rmcts = sum(a.rmcts_wins for a in agg.values())
                total_agg_draws = sum(a.draws for a in agg.values())
                total_agg_classic = sum(a.classic_wins for a in agg.values())
                log_line(
                    log_fp,
                    (
                        f"AGGREGATE after run={run_index}: games={total_agg_games} "
                        f"wdl={total_agg_rmcts}/{total_agg_draws}/{total_agg_classic}"
                    ),
                )

                if stop:
                    break

        log_line(log_fp, "=== FINAL PER-CONFIG AGGREGATES ===")
        for cfg in configs:
            a = agg[cfg.key]
            log_line(
                log_fp,
                (
                    f"{cfg.key} | runs={a.runs} games={a.games} "
                    f"wdl={a.rmcts_wins}/{a.draws}/{a.classic_wins} "
                    f"sims={a.rmcts_sims}/{a.classic_sims}"
                ),
            )

        log_line(log_fp, "=== RMCTS overnight run finished ===")

    print(f"Log written to: {log_file}")
    print(f"Summary CSV written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
