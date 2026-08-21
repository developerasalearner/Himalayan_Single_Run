#!/usr/bin/env python3
"""Stage 6 - the leakage ablation.

Runs the final configuration twice: once with the termination reason code
excluded, once with it readmitted, changing nothing else. The gap is what
the exclusion rule costs in apparent performance.

The higher figures are not better performance. The termination reason
records why a trip ended, and one of its categories covers accidents
involving death or serious injury, so readmitting it hands the classifier
partial access to its own label. The point of reporting the gap is that a
shift of this size is easily mistaken for a methodological advance.
"""

from __future__ import annotations

import copy
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
)
from himalayan_risk.pipeline import build_splits, save_table  # noqa: E402


def run_variant(cfg, device, exclude_termreason: bool, use_cache: bool) -> dict:
    variant = copy.deepcopy(cfg)
    variant.set_dotted("leakage.exclude_termreason", exclude_termreason)

    X_train, X_test, y_train, y_test = build_splits(variant, use_cache=use_cache)

    feature_sets = require_frozen_features(cfg)
    selector = cfg.get_dotted("final.selector")
    source = "selected_features.json"

    if not exclude_termreason:
        # Readmitting termreason enlarges the screened column set from 47
        # to 48, so the search was re-run for the ablation and settled on
        # a different subset. Reusing the main subset plus one appended
        # column would answer a different question, so the ablation's own
        # subset is preferred when it is available.
        ablation_path = cfg.get_path("paths.artifacts_dir") / "selected_features_ablation.json"
        if ablation_path.exists():
            with open(ablation_path, "r", encoding="utf-8") as handle:
                candidate = json.load(handle).get("features", {})
            if selector in candidate and candidate[selector]:
                feature_sets = candidate
                source = "selected_features_ablation.json"
        else:
            source = "selected_features.json, termreason appended"

    features = [f for f in feature_sets[selector] if f in X_train.columns]

    if not exclude_termreason and "termreason" in X_train.columns:
        if "termreason" not in features:
            features = features + ["termreason"]

    print(f"  subset: {source} ({len(features)} features)")

    model = fit_final_model(
        X_train[features], y_train,
        seed=cfg.get_dotted("seed"),
        device=device,
        calibrate=cfg.get_dotted("final.calibrate"),
        calibration_method=cfg.get_dotted("calibration.method"),
        calibration_cv=cfg.get_dotted("calibration.cv"),
    )

    final = evaluate_final(
        model, X_test[features], y_test,
        threshold=float(cfg.get_dotted("final.threshold")),
    )
    importance = extract_feature_importance(model, features)

    return {
        "summary": final["summary"],
        "importance": importance,
        "features": features,
        "subset_source": source,
    }


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    cfg, device = setup(args)

    excluded = run_variant(cfg, device, True, not args.no_cache)
    readmitted = run_variant(cfg, device, False, not args.no_cache)

    rows = [
        {"Configuration": "termreason excluded (main)",
         "Subset": excluded["subset_source"], **excluded["summary"]},
        {"Configuration": "termreason readmitted (ablation)",
         "Subset": readmitted["subset_source"], **readmitted["summary"]},
    ]
    table = pd.DataFrame(rows)
    save_table(table, cfg.get_path("paths.results_dir"), "table5_leakage_ablation")

    delta = {
        "delta_roc_auc": round(
            readmitted["summary"]["ROC_AUC"] - excluded["summary"]["ROC_AUC"], 4
        ),
        "delta_average_precision": round(
            readmitted["summary"]["AveragePrecision"]
            - excluded["summary"]["AveragePrecision"], 4
        ),
        "delta_f1": round(readmitted["summary"]["F1"] - excluded["summary"]["F1"], 4),
    }

    importance = readmitted["importance"]
    if not importance.empty:
        rank = importance.index[importance["Feature"] == "termreason"]
        delta["termreason_importance_rank"] = int(rank[0]) + 1 if len(rank) else None

    (cfg.get_path("paths.results_dir") / "leakage_ablation_delta.json").write_text(
        json.dumps(delta, indent=2)
    )

    print("\nEffect of readmitting the termination reason code:")
    for key, value in delta.items():
        print(f"  {key}: {value}")
    print("\nThese are not improvements. See the docstring at the top of this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
