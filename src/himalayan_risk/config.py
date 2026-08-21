"""Configuration loading.

A single YAML file holds every knob that can change a reported number.
`load_config` reads it, applies dotted-key overrides from the command
line, and returns a plain dict wrapped so that `cfg["a.b.c"]` works.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """Dict with dotted-path access and resolved, repo-relative paths."""

    def get_path(self, dotted: str) -> Path:
        value = self.get_dotted(dotted)
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    def get_dotted(self, dotted: str, default: Any = "__RAISE__") -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__RAISE__":
                    raise KeyError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node

    def set_dotted(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dict(self), sort_keys=False))


def _coerce(text: str) -> Any:
    """Turn a command-line string into the obvious Python value."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> Config:
    """Load the YAML config and apply `key.subkey=value` overrides."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = Config(copy.deepcopy(raw))

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got: {item!r}")
        key, _, value = item.partition("=")
        cfg.set_dotted(key.strip(), _coerce(value))

    return cfg


def ensure_output_dirs(cfg: Config) -> None:
    for key in ("processed_dir", "results_dir", "figures_dir", "artifacts_dir"):
        cfg.get_path(f"paths.{key}").mkdir(parents=True, exist_ok=True)
