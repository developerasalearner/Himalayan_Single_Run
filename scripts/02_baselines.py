#!/usr/bin/env python3
"""Stage 2 - Table I: untuned classifiers on the full feature set.

The reference point. Accuracy above 0.94 is available to anything that
learns to answer "no", so the interesting column is how many of the fatal
expeditions each model actually recovers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import build_parser, setup  # noqa: E402

from himalayan_risk.evaluate import run_untuned_classifiers  # noqa: E402
from himalayan_risk.pipeline import build_splits, save_table  # noqa: E402


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    cfg, device = setup(args)

    X_train, X_test, y_train, y_test = build_splits(cfg, use_cache=not args.no_cache)

    table, matrices = run_untuned_classifiers(
        X_train, X_test, y_train, y_test,
        seed=cfg.get_dotted("seed"), device=device,
    )

    save_table(table, cfg.get_path("paths.results_dir"), "table1_untuned_classifiers")

    print(f"\nTest-set positives: {int(y_test.sum())}")
    for name, matrix in matrices.items():
        print(f"{name}: recovered {matrix[1, 1]} of {int(y_test.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
