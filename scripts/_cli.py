"""Shared argument parsing and logging setup for the stage scripts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from himalayan_risk.config import ensure_output_dirs, load_config  # noqa: E402
from himalayan_risk.seeds import resolve_device, set_seeds  # noqa: E402


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set seed=7",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="rebuild the modelling table instead of reusing the cached one",
    )
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only")
    return parser


def setup(args) -> tuple:
    """Configure logging, load config, seed everything, resolve the device."""
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config, args.overrides)
    ensure_output_dirs(cfg)
    set_seeds(cfg.get_dotted("seed"))

    device = resolve_device(cfg.get_dotted("compute.device"))
    cfg.set_dotted("compute.device", device)

    return cfg, device


def require_frozen_features(cfg) -> dict:
    """Load the frozen feature subsets, or exit with a readable message.

    A stack trace is the wrong response to a setup step the user simply
    has not done yet.
    """
    from himalayan_risk.selection import load_frozen_features

    try:
        feature_sets = load_frozen_features(cfg.get_path("paths.artifacts_dir"))
    except RuntimeError as error:
        raise SystemExit(f"\n{error}\n") from None

    if feature_sets is None:
        raise SystemExit(
            "\nNo frozen feature sets found.\n"
            "Run: python scripts/03_feature_selection.py\n"
        )

    return feature_sets
