#!/usr/bin/env python3
"""Stage 3 - Table II: feature-selection strategies.

Runs the embedded stage and every metaheuristic, then scores each
resulting subset under the same classifier so differences are
attributable to the subset rather than to the learner.

The search is stochastic. Selected subsets are frozen to
artifacts/selected_features.json, and later runs reuse that file by
default so downstream stages reproduce exactly. Pass --research to force
a fresh search.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import build_parser, setup  # noqa: E402

from himalayan_risk.pipeline import build_splits, save_table  # noqa: E402
from himalayan_risk.evaluate import evaluate_feature_sets  # noqa: E402
from himalayan_risk.selection import (  # noqa: E402
    load_frozen_features,
    run_all_optimizers,
    run_embedded_selection,
    save_frozen_features,
)


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--research",
        action="store_true",
        help="re-run the search even if frozen feature sets exist",
    )
    args = parser.parse_args()
    cfg, device = setup(args)

    X_train, X_test, y_train, y_test = build_splits(cfg, use_cache=not args.no_cache)
    artifacts_dir = cfg.get_path("paths.artifacts_dir")

    reuse = cfg.get_dotted("selection.reuse_frozen_features") and not args.research

    feature_sets = None
    if reuse:
        try:
            feature_sets = load_frozen_features(artifacts_dir)
        except RuntimeError as error:
            # The shipped placeholder. Stages 4-6 stop on it, because
            # silently proceeding there would produce meaningless tables.
            # This stage is the one that populates it, so it searches.
            print(f"{error}\n\nNo subsets frozen yet - running the search now.\n")
            feature_sets = None

    if feature_sets is None:
        embedded, embedded_score = run_embedded_selection(
            X_train, y_train,
            seed=cfg.get_dotted("seed"),
            device=device,
            cv_folds=cfg.get_dotted("selection.embedded.cv_folds"),
            n_estimators=cfg.get_dotted("selection.embedded.n_estimators"),
            max_depth=cfg.get_dotted("selection.embedded.max_depth"),
            learning_rate=cfg.get_dotted("selection.embedded.learning_rate"),
        )

        feature_sets = {"Embedded": embedded}
        feature_sets.update(
            run_all_optimizers(
                X_train, y_train,
                optimizers=cfg.get_dotted("selection.optimizers"),
                seed=cfg.get_dotted("seed"),
                device=device,
                inner_val_size=cfg.get_dotted("selection.inner_val_size"),
                pop_size=cfg.get_dotted("selection.population_size"),
                iterations=cfg.get_dotted("selection.iterations"),
                size_penalty=cfg.get_dotted("selection.fitness.size_penalty"),
                surrogate=cfg.get_dotted("selection.surrogate"),
                fitness_weights=cfg.get_dotted("selection.fitness"),
            )
        )

        save_frozen_features(
            artifacts_dir,
            feature_sets,
            metadata={
                "seed": cfg.get_dotted("seed"),
                "termreason_excluded": cfg.get_dotted("leakage.exclude_termreason"),
                "screened_feature_count": int(X_train.shape[1]),
                "population_size": cfg.get_dotted("selection.population_size"),
                "iterations": cfg.get_dotted("selection.iterations"),
                "embedded_score": round(embedded_score, 5),
            },
        )

    table = evaluate_feature_sets(
        feature_sets, X_train, X_test, y_train, y_test,
        seed=cfg.get_dotted("seed"), device=device,
    )
    save_table(table, cfg.get_path("paths.results_dir"), "table2_feature_selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
