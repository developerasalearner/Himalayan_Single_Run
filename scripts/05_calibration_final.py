#!/usr/bin/env python3
"""Stage 5 - Table IV and the final evaluated configuration.

Two things happen here.

First, Table IV isolates the calibration step: calibrated against
uncalibrated, under a threshold rule that maximises 0.6*recall +
0.4*precision on a grid over [0.10, 0.20].

Second, the configuration fixed in Section III-I is fitted and evaluated:
the hybrid subset, class-weighted XGBoost without resampling, sigmoid
calibration by three-fold cross-validation, and the chosen threshold.
Every final metric, the confusion matrix, both curves, the reliability
diagram and the importance ranking derive from that one prediction
vector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import build_parser, require_frozen_features, setup  # noqa: E402

import pandas as pd  # noqa: E402

from himalayan_risk.evaluate import (  # noqa: E402
    evaluate_final,
    extract_feature_importance,
    fit_final_model,
    run_calibration_comparison,
)
from himalayan_risk.figures import make_all_figures  # noqa: E402
from himalayan_risk.pipeline import build_splits, save_table  # noqa: E402


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    cfg, device = setup(args)

    X_train, X_test, y_train, y_test = build_splits(cfg, use_cache=not args.no_cache)

    feature_sets = require_frozen_features(cfg)

    selector = cfg.get_dotted("final.selector")
    features = [f for f in feature_sets[selector] if f in X_train.columns]
    X_tr, X_te = X_train[features], X_test[features]

    results_dir = cfg.get_path("paths.results_dir")

    # ---- Table IV -----------------------------------------------------
    table4 = run_calibration_comparison(
        X_tr, X_te, y_train, y_test,
        methods=cfg.get_dotted("calibration.compare_methods"),
        seed=cfg.get_dotted("seed"),
        device=device,
        calibration_method=cfg.get_dotted("calibration.method"),
        calibration_cv=cfg.get_dotted("calibration.cv"),
        threshold_grid=cfg.get_dotted("calibration.threshold_grid"),
        objective=cfg.get_dotted("calibration.objective"),
    )
    save_table(table4, results_dir, "table4_calibration_effect")

    # ---- Final evaluated configuration --------------------------------
    model = fit_final_model(
        X_tr, y_train,
        seed=cfg.get_dotted("seed"),
        device=device,
        calibrate=cfg.get_dotted("final.calibrate"),
        calibration_method=cfg.get_dotted("calibration.method"),
        calibration_cv=cfg.get_dotted("calibration.cv"),
    )

    threshold = float(cfg.get_dotted("final.threshold"))
    final = evaluate_final(model, X_te, y_test, threshold=threshold)

    save_table(pd.DataFrame([final["summary"]]), results_dir, "final_metrics")

    importance = extract_feature_importance(model, list(X_te.columns))
    save_table(importance, results_dir, "final_feature_importance")

    predictions = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_proba": final["proba"],
        "y_pred": final["y_pred"],
    })
    predictions.to_csv(results_dir / "final_predictions.csv", index=False)

    (results_dir / "final_summary.json").write_text(
        json.dumps(final["summary"], indent=2)
    )

    if not args.skip_figures:
        paths = make_all_figures(
            y_test, final["proba"], threshold, importance,
            cfg.get_path("paths.figures_dir"),
        )
        print("\nFigures:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
