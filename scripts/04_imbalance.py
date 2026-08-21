#!/usr/bin/env python3
"""Stage 4 - Table III: imbalance treatments on the hybrid subset.

Calibration is held fixed across all six treatments, so the only thing
varying is the sampling strategy. Each treatment gets its own operating
point, chosen by maximising minority F1 over a grid on [0.10, 0.30].

Note the difference from Stage 5: that stage uses a narrower grid and a
weighted recall/precision objective, which is why a method can appear in
both tables with different numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import build_parser, require_frozen_features, setup  # noqa: E402

from himalayan_risk.imbalance import flag_degenerate, run_imbalance_comparison  # noqa: E402
from himalayan_risk.pipeline import build_splits, save_table  # noqa: E402


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    cfg, device = setup(args)

    X_train, X_test, y_train, y_test = build_splits(cfg, use_cache=not args.no_cache)

    feature_sets = require_frozen_features(cfg)

    selector = cfg.get_dotted("final.selector")
    features = [f for f in feature_sets[selector] if f in X_train.columns]
    print(f"Using the {selector} subset: {len(features)} features")

    table = run_imbalance_comparison(
        X_train[features], X_test[features], y_train, y_test,
        methods=cfg.get_dotted("imbalance.methods"),
        seed=cfg.get_dotted("seed"),
        device=device,
        calibration_method=cfg.get_dotted("calibration.method"),
        calibration_cv=cfg.get_dotted("calibration.cv"),
        threshold_grid=cfg.get_dotted("imbalance.threshold_grid"),
    )
    table = flag_degenerate(table, base_rate=float(y_test.mean()))

    save_table(table, cfg.get_path("paths.results_dir"), "table3_imbalance_treatments")

    degenerate = table[table["Degenerate"]]
    if not degenerate.empty:
        print("\nCollapsed into trivial classifiers:",
              ", ".join(degenerate["Method"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
