#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from rmcts_params import DEFAULT_CONFIG_PATH, load_config


def run_command(argv: list[str], dry_run: bool) -> int:
    print("Command:", " ".join(shlex.quote(x) for x in argv))
    if dry_run:
        return 0
    return subprocess.call(argv)


def cfg_section(config: dict, section: str) -> dict:
    value = config.get(section, {})
    if isinstance(value, dict):
        return value
    return {}


def ensure_trt_cache_dirs(paths: list[str], dry_run: bool) -> None:
    seen: set[Path] = set()
    created_any = False
    for binary_path in paths:
        if not binary_path:
            continue
        cache_dir = Path(binary_path).expanduser().resolve().parent / "trt_cache"
        if cache_dir in seen:
            continue
        seen.add(cache_dir)
        if dry_run:
            print(f"Will ensure TensorRT cache dir: {cache_dir}")
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        created_any = True
    if created_any:
        print("TensorRT cache directories ensured.")


def has_weights_arg(args: list[str]) -> bool:
    for i, token in enumerate(args):
        if token.startswith("--weights=") and token != "--weights=":
            return True
        if token == "--weights" and i + 1 < len(args) and not args[i + 1].startswith("-"):
            return True
    return False


def uses_onnx_backend(args: list[str]) -> bool:
    for i, token in enumerate(args):
        if token.startswith("--backend="):
            backend = token.split("=", 1)[1].strip().lower()
            if backend.startswith("onnx"):
                return True
        if token == "--backend" and i + 1 < len(args):
            backend = args[i + 1].strip().lower()
            if backend.startswith("onnx"):
                return True
    return False


def validate_onnx_weights(command_label: str, args: list[str]) -> bool:
    if not uses_onnx_backend(args):
        return True
    if has_weights_arg(args):
        return True
    print(
        f"Configuration error for {command_label}: ONNX backend selected but no --weights provided.",
        file=sys.stderr,
    )
    print(
        "Set --weights in src/rmcts/params.toml shared_args (or pass it on the CLI).",
        file=sys.stderr,
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RMCTS tools using shared config in src/rmcts/params.toml"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to shared RMCTS TOML config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_uci_r = sub.add_parser("uci-rmcts", help="Run lc0 in UCI RMCTS mode.")
    p_uci_r.add_argument("extra", nargs=argparse.REMAINDER)

    p_uci_c = sub.add_parser("uci-classic", help="Run lc0 in UCI classic mode.")
    p_uci_c.add_argument("extra", nargs=argparse.REMAINDER)

    p_play = sub.add_parser("play", help="Run play utility with configured defaults.")
    p_play.add_argument("extra", nargs=argparse.REMAINDER)

    p_bm = sub.add_parser("bot-match", help="Run scripts/bot_match.py with config defaults.")
    p_bm.add_argument("extra", nargs=argparse.REMAINDER)

    p_sp = sub.add_parser(
        "single-compare",
        help="Run scripts/single_position_compare.py with config defaults.",
    )
    p_sp.add_argument("extra", nargs=argparse.REMAINDER)

    p_pl = sub.add_parser("plot", help="Run scripts/rmcts_plot.py with config defaults.")
    p_pl.add_argument("extra", nargs=argparse.REMAINDER)

    sub.add_parser(
        "check-config",
        help="Validate RMCTS config (including ONNX backend weights requirements).",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    common = cfg_section(config, "common")
    commands = cfg_section(config, "commands")

    lc0_binary = commands.get("lc0_binary", "build/release/lc0")
    play_binary = commands.get("play_binary", "build/release/play")
    script_engine = str(common.get("engine", ""))
    shared_args = shlex.split(str(common.get("shared_args", "")))
    bot_match = cfg_section(config, "bot_match")
    single_compare = cfg_section(config, "single_position_compare")

    if args.cmd in {"uci-rmcts", "uci-classic", "play", "bot-match", "single-compare"}:
        ensure_trt_cache_dirs([str(lc0_binary), str(play_binary), script_engine], args.dry_run)

    if args.cmd == "check-config":
        checks_ok = True

        if not validate_onnx_weights("common.shared_args", shared_args):
            checks_ok = False

        single_shared_args = shlex.split(str(single_compare.get("shared_args", "")))
        if single_shared_args and not validate_onnx_weights(
            "single_position_compare.shared_args", single_shared_args
        ):
            checks_ok = False

        if checks_ok:
            print("Config check passed.")
            return 0
        return 2

    if args.cmd == "uci-rmcts":
        mode_args = shlex.split(str(commands.get("uci_rmcts_args", "rmcts")))
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        merged = [*mode_args, *shared_args, *extra]
        if not validate_onnx_weights("uci-rmcts", merged):
            return 2
        argv = [lc0_binary, *merged]
        return run_command(argv, args.dry_run)

    if args.cmd == "uci-classic":
        mode_args = shlex.split(str(commands.get("uci_classic_args", "classic")))
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        merged = [*mode_args, *shared_args, *extra]
        if not validate_onnx_weights("uci-classic", merged):
            return 2
        argv = [lc0_binary, *merged]
        return run_command(argv, args.dry_run)

    if args.cmd == "play":
        play_args = shlex.split(str(commands.get("play_args", "")))
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        merged = [*play_args, *shared_args, *extra]
        if not validate_onnx_weights("play", merged):
            return 2
        argv = [play_binary, *merged]
        return run_command(argv, args.dry_run)

    if args.cmd == "bot-match":
        bot_shared_args = shlex.split(str(bot_match.get("shared_args", "")))
        if bot_shared_args:
            if not validate_onnx_weights("bot_match.shared_args", bot_shared_args):
                return 2
        elif not validate_onnx_weights("common.shared_args", shared_args):
            return 2

        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        argv = [
            sys.executable,
            "scripts/bot_match.py",
            "--config",
            str(config_path),
            *extra,
        ]
        return run_command(argv, args.dry_run)

    if args.cmd == "single-compare":
        single_shared_args = shlex.split(str(single_compare.get("shared_args", "")))
        if single_shared_args:
            if not validate_onnx_weights("single_position_compare.shared_args", single_shared_args):
                return 2
        elif not validate_onnx_weights("common.shared_args", shared_args):
            return 2

        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        argv = [
            sys.executable,
            "scripts/single_position_compare.py",
            "--config",
            str(config_path),
            *extra,
        ]
        return run_command(argv, args.dry_run)

    if args.cmd == "plot":
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        argv = [
            sys.executable,
            "scripts/rmcts_plot.py",
            "--config",
            str(config_path),
            *extra,
        ]
        return run_command(argv, args.dry_run)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())