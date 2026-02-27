from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "rmcts" / "params.toml"
)


def parse_config_path(argv: list[str], default_path: Path | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(default_path or DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args(argv)
    return Path(args.config)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def section_defaults(config: dict[str, Any], section: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    common = config.get("common", {})
    specific = config.get(section, {})
    if isinstance(common, dict):
        defaults.update(common)
    if isinstance(specific, dict):
        defaults.update(specific)
    return defaults


def apply_known_defaults(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    valid = {action.dest for action in parser._actions}
    filtered = {key: value for key, value in defaults.items() if key in valid}
    parser.set_defaults(**filtered)
