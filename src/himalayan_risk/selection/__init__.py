"""Feature-selection stage.

Exposes every optimiser under one registry and a runner that scores all
of them against the same inner validation split, then freezes the chosen
subsets to disk so downstream stages are reproducible even though the
search itself is stochastic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from ..modelling import make_evaluator
from .embedded import run_embedded_selection
from .hybrid import FOA_GWO_hybrid_feature_selection
from .metaheuristics import (
    EVO_feature_selection,
    FOA_feature_selection,
    GA_feature_selection,
    GWO_feature_selection,
    PSO_feature_selection,
)

LOGGER = logging.getLogger(__name__)

OPTIMIZERS = {
    "GA": GA_feature_selection,
    "PSO": PSO_feature_selection,
    "GWO": GWO_feature_selection,
    "EVO": EVO_feature_selection,
    "FOA": FOA_feature_selection,
    "FOA_GWO": FOA_GWO_hybrid_feature_selection,
}

FROZEN_FILENAME = "selected_features.json"

__all__ = [
    "OPTIMIZERS",
    "run_all_optimizers",
    "run_embedded_selection",
    "load_frozen_features",
    "save_frozen_features",
]


def load_frozen_features(artifacts_dir: Path) -> Dict[str, List[str]] | None:
    """Return previously frozen feature sets, or None if there are none.

    The shipped file is a placeholder with empty subsets. Silently
    treating it as "no features selected" would produce plausible-looking
    but meaningless tables downstream, so it raises instead.
    """
    path = Path(artifacts_dir) / FROZEN_FILENAME
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features", payload)

    if payload.get("_placeholder") or not any(features.values()):
        raise RuntimeError(
            f"{path} is still the shipped placeholder - no feature subsets have "
            "been frozen yet.\n\n"
            "Either paste the subsets from the manuscript run into its "
            "'features' block, or run:\n\n"
            "    python scripts/03_feature_selection.py --research\n\n"
            "See docs/REPRODUCIBILITY.md."
        )

    LOGGER.info("reusing frozen feature sets from %s", path)
    return features


def save_frozen_features(
    artifacts_dir: Path,
    features: Dict[str, List[str]],
    metadata: Dict | None = None,
) -> Path:
    """Write the selected subsets so a later run reproduces them exactly."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / FROZEN_FILENAME

    payload = {"metadata": metadata or {}, "features": features}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    LOGGER.info("froze %d feature sets to %s", len(features), path)
    return path


def run_all_optimizers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optimizers: List[str],
    seed: int = 42,
    device: str = "cpu",
    inner_val_size: float = 0.20,
    pop_size: int = 20,
    iterations: int = 30,
    size_penalty: float = 0.02,
    surrogate: Dict | None = None,
    fitness_weights: Dict | None = None,
) -> Dict[str, List[str]]:
    """Run each requested optimiser against a shared inner validation split.

    The split comes out of the training partition only. The test set is
    never seen during search, so nothing here can tune to it.
    """
    surrogate = surrogate or {}

    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_train,
        y_train,
        test_size=inner_val_size,
        stratify=y_train,
        random_state=seed,
    )

    evaluator = make_evaluator(
        X_inner_train,
        X_inner_val,
        y_inner_train,
        y_inner_val,
        device=device,
        seed=seed,
        n_estimators=surrogate.get("n_estimators", 50),
        max_depth=surrogate.get("max_depth", 6),
        learning_rate=surrogate.get("learning_rate", 0.1),
        weights=fitness_weights,
    )

    results: Dict[str, List[str]] = {}

    for name in optimizers:
        if name not in OPTIMIZERS:
            raise KeyError(f"unknown optimiser {name!r}; choose from {sorted(OPTIMIZERS)}")

        LOGGER.info("=== %s: searching over %d features ===", name, X_train.shape[1])

        kwargs = dict(max_iter=iterations, size_penalty=size_penalty)
        kwargs["pop_size" if name != "PSO" else "swarm_size"] = pop_size
        if name == "GA":
            kwargs = dict(
                pop_size=pop_size, generations=iterations, size_penalty=size_penalty
            )
        elif name == "GWO":
            kwargs = dict(
                pack_size=pop_size, max_iter=iterations, size_penalty=size_penalty
            )

        selected = OPTIMIZERS[name](X_train, y_train, evaluator, **kwargs)
        results[name] = list(selected)
        LOGGER.info("=== %s selected %d features ===", name, len(selected))

    return results
