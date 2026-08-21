#!/usr/bin/env python3
"""Stage 0 - fetch both source archives into data/raw/.

Requires Kaggle credentials. If you would rather download by hand, see
docs/DATA.md for the expected directory layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import REPO_ROOT, build_parser, setup  # noqa: E402

from himalayan_risk import data  # noqa: E402


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    cfg, _ = setup(args)

    destination = REPO_ROOT / "data" / "raw"
    resolved = data.download_from_kaggle(destination)

    print("\nDownloaded:")
    for name, path in resolved.items():
        print(f"  {name}: {path}")
    print("\nThese match the paths already set in config/default.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
