#!/usr/bin/env python3
"""Stage 1 - build the modelling table and report the split.

Joins expedition records to hourly weather through peak codes and recorded
intervals, engineers the climate and vulnerability descriptors, applies
the outcome-field exclusion rule, imputes, encodes and screens for
redundancy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import build_parser, setup  # noqa: E402

import pandas as pd  # noqa: E402

from himalayan_risk.pipeline import build_splits, save_table  # noqa: E402


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    cfg, _ = setup(args)

    X_train, X_test, y_train, y_test = build_splits(cfg, use_cache=not args.no_cache)

    summary = pd.DataFrame([
        {
            "Partition": "train",
            "Rows": len(X_train),
            "Positives": int(y_train.sum()),
            "PositiveRate": round(float(y_train.mean()), 5),
        },
        {
            "Partition": "test",
            "Rows": len(X_test),
            "Positives": int(y_test.sum()),
            "PositiveRate": round(float(y_test.mean()), 5),
        },
    ])

    save_table(summary, cfg.get_path("paths.results_dir"), "00_split_summary")

    features = pd.DataFrame({"Feature": sorted(X_train.columns)})
    save_table(features, cfg.get_path("paths.results_dir"), "00_screened_features")

    print(f"\nFeatures after redundancy screen: {X_train.shape[1]}")
    print(f"termreason excluded: {cfg.get_dotted('leakage.exclude_termreason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
