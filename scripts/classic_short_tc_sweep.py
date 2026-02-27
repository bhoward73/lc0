#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepResult:
    movetime_ms: int
    classic_batch: int
    classic_prefetch: int
    seed: int
    status: str
    rmcts_wins: int = 0
    draws: int = 0
    classic_wins: int = 0
    rmcts_sims: int = 0
    classic_sims: int = 0

    @property
    def games(self) -> int:
        return self.rmcts_wins + self.draws + self.classic_wins

    @property
    def rmcts_score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.rmcts_wins + 0.5 * self.draws) / self.games


def parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Expected at least one integer value")
    return out


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(fp, msg: str) -> None:
    line = f"[{now_stamp()}] {msg}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def parse_final(lines: str) -> tuple[int, int, int, int, int] | None:
    pats = {
        "rw": r"^rmcts wins:\s*(\d+)\s*$",
        "d": r"^draws:\s*(\d+)\s*$",
        "cw": r"^classic wins:\s*(\d+)\s*$",
        "rs": r"^rmcts total sims:\s*(\d+)\s*$",
        "cs": r"^classic total sims:\s*(\d+)\s*$",
    }
    vals: dict[str, int] = {}
    for k, p in pats.items():
        m = re.findall(p, lines, flags=re.MULTILINE)
        if not m:
            return None
        vals[k] = int(m[-1])
    return vals["rw"], vals["d"], vals["cw"], vals["rs"], vals["cs"]


def run_bot_match(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def build_cmd(
    python_exe: str,
    bot_match: Path,
    engine: str,
    weights: str,
    movetime_ms: int,
    positions: int,
    opening_min: int,
    opening_max: int,
    max_plies: int,
    seed: int,
    rmcts_gpu: int,
    classic_gpu: int,
    rmcts_chunk: int,
    rmcts_cpuct: float,
    classic_batch: int,
    classic_prefetch: int,
) -> list[str]:
    shared = f"--backend=onnx-trt --weights={weights}"
    a_args = (
        f"rmcts --rmcts-chunk-sims={rmcts_chunk} --rmcts-cpuct={rmcts_cpuct} "
        f"--backend-opts=gpu={rmcts_gpu},batch=16,steps=1"
    )
    b_args = (
        f"classic --minibatch-size={classic_batch} --max-prefetch={classic_prefetch} "
        f"--backend-opts=gpu={classic_gpu},batch={classic_batch},steps=1"
    )
    return [
        python_exe,
        str(bot_match),
        "--engine", engine,
        "--shared-args", shared,
        "--a-args", a_args,
        "--b-args", b_args,
        "--positions", str(positions),
        "--movetime-ms", str(movetime_ms),
        "--opening-plies-min", str(opening_min),
        "--opening-plies-max", str(opening_max),
        "--max-plies", str(max_plies),
        "--seed", str(seed),
        "--label-a", "rmcts",
        "--label-b", "classic",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep classic minibatch/prefetch at short time controls vs fixed RMCTS settings."
    )
    parser.add_argument("--engine", default="build/release/lc0")
    parser.add_argument("--weights", required=True, help="Absolute or repo-relative path to .pb.gz weights")
    parser.add_argument("--movetimes", default="100,200")
    parser.add_argument("--classic-batches", default="16,32,64,96,136")
    parser.add_argument("--positions", type=int, default=2)
    parser.add_argument("--opening-plies-min", type=int, default=6)
    parser.add_argument("--opening-plies-max", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=220)
    parser.add_argument("--seed-base", type=int, default=3000)
    parser.add_argument("--rmcts-gpu", type=int, default=0)
    parser.add_argument("--classic-gpu", type=int, default=1)
    parser.add_argument("--rmcts-chunk", type=int, default=128)
    parser.add_argument("--rmcts-cpuct", type=float, default=2.0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-positions", type=int, default=1)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    bot_match = repo / "scripts" / "bot_match.py"
    if not bot_match.exists():
        raise FileNotFoundError(bot_match)

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (repo / weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)

    movetimes = parse_int_list(args.movetimes)
    batches = parse_int_list(args.classic_batches)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_file) if args.log_file else Path(f"/tmp/classic_short_tc_sweep_{ts}.log")
    csv_file = Path(args.csv) if args.csv else Path(f"/tmp/classic_short_tc_sweep_{ts}.csv")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with csv_file.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "timestamp",
            "movetime_ms",
            "classic_batch",
            "classic_prefetch",
            "seed",
            "status",
            "rmcts_wins",
            "draws",
            "classic_wins",
            "games",
            "rmcts_score",
            "rmcts_sims",
            "classic_sims",
            "sim_ratio_rmcts_over_classic",
        ])

    with log_file.open("a", encoding="utf-8") as lf:
        log(lf, "=== classic short-TC sweep start ===")
        log(lf, f"Weights: {weights}")
        log(lf, f"Movetimes: {movetimes}")
        log(lf, f"Classic batches/prefetch: {batches}")
        log(lf, f"RMCTS fixed: chunk={args.rmcts_chunk}, cpuct={args.rmcts_cpuct}")
        log(lf, f"GPU assignment: rmcts={args.rmcts_gpu}, classic={args.classic_gpu}")
        log(lf, f"Log: {log_file}")
        log(lf, f"CSV: {csv_file}")

        idx = 0
        for mt in movetimes:
            for b in batches:
                idx += 1
                seed = args.seed_base + idx
                if args.warmup:
                    warm = build_cmd(
                        sys.executable,
                        bot_match,
                        args.engine,
                        str(weights),
                        mt,
                        args.warmup_positions,
                        args.opening_plies_min,
                        args.opening_plies_max,
                        args.max_plies,
                        seed + 100000,
                        args.rmcts_gpu,
                        args.classic_gpu,
                        args.rmcts_chunk,
                        args.rmcts_cpuct,
                        b,
                        b,
                    )
                    log(lf, f"WARMUP mt={mt} batch={b} cmd={' '.join(shlex.quote(x) for x in warm)}")
                    if not args.dry_run:
                        _, out = run_bot_match(warm)
                        for line in out.splitlines():
                            log(lf, f"[warmup] {line}")

                cmd = build_cmd(
                    sys.executable,
                    bot_match,
                    args.engine,
                    str(weights),
                    mt,
                    args.positions,
                    args.opening_plies_min,
                    args.opening_plies_max,
                    args.max_plies,
                    seed,
                    args.rmcts_gpu,
                    args.classic_gpu,
                    args.rmcts_chunk,
                    args.rmcts_cpuct,
                    b,
                    b,
                )
                log(lf, f"RUN mt={mt} batch={b} seed={seed}")
                log(lf, f"CMD {' '.join(shlex.quote(x) for x in cmd)}")

                status = "dry-run"
                result = None
                output = ""
                if not args.dry_run:
                    rc, output = run_bot_match(cmd)
                    for line in output.splitlines():
                        log(lf, f"[run] {line}")
                    parsed = parse_final(output)
                    if rc == 0 and parsed is not None:
                        rw, d, cw, rs, cs = parsed
                        result = SweepResult(
                            movetime_ms=mt,
                            classic_batch=b,
                            classic_prefetch=b,
                            seed=seed,
                            status="ok",
                            rmcts_wins=rw,
                            draws=d,
                            classic_wins=cw,
                            rmcts_sims=rs,
                            classic_sims=cs,
                        )
                        status = "ok"
                    else:
                        status = f"rc={rc}"
                        result = SweepResult(
                            movetime_ms=mt,
                            classic_batch=b,
                            classic_prefetch=b,
                            seed=seed,
                            status=status,
                        )
                else:
                    result = SweepResult(
                        movetime_ms=mt,
                        classic_batch=b,
                        classic_prefetch=b,
                        seed=seed,
                        status=status,
                    )

                ratio = ""
                if result.classic_sims > 0:
                    ratio = f"{result.rmcts_sims / result.classic_sims:.6f}"
                elif result.rmcts_sims > 0:
                    ratio = "inf"

                with csv_file.open("a", newline="", encoding="utf-8") as cf:
                    w = csv.writer(cf)
                    w.writerow([
                        now_stamp(),
                        result.movetime_ms,
                        result.classic_batch,
                        result.classic_prefetch,
                        result.seed,
                        result.status,
                        result.rmcts_wins,
                        result.draws,
                        result.classic_wins,
                        result.games,
                        f"{result.rmcts_score:.6f}",
                        result.rmcts_sims,
                        result.classic_sims,
                        ratio,
                    ])

                log(
                    lf,
                    (
                        f"RESULT mt={mt} batch={b} status={status} "
                        f"wdl={result.rmcts_wins}/{result.draws}/{result.classic_wins} "
                        f"score={result.rmcts_score:.3f} sims={result.rmcts_sims}/{result.classic_sims}"
                    ),
                )

        log(lf, "=== classic short-TC sweep done ===")

    print(f"Log written to: {log_file}")
    print(f"CSV written to: {csv_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
