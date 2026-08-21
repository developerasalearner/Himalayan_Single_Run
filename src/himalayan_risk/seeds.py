"""Determinism helpers.

The original notebook seeded scikit-learn and XGBoost but left the
`random` and `numpy.random` global streams unseeded, which made the
metaheuristic search irreproducible between runs. `set_seeds` closes
that gap; call it once at the top of every entry point.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

LOGGER = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Seed every global random stream the pipeline touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    LOGGER.info("global seeds set to %d", seed)


def resolve_device(requested: str = "auto") -> str:
    """Return 'cuda' or 'cpu'.

    'auto' probes XGBoost for a usable GPU and falls back silently.
    Tree ensembles this small give identical predictions either way;
    only wall-clock time differs.
    """
    requested = (requested or "auto").lower()
    if requested in {"cpu", "cuda"}:
        return requested
    if requested != "auto":
        raise ValueError(f"device must be auto, cpu or cuda; got {requested!r}")

    try:
        import numpy as _np
        import xgboost as xgb

        probe = xgb.XGBClassifier(
            n_estimators=1, tree_method="hist", device="cuda", verbosity=0
        )
        probe.fit(_np.zeros((4, 2)), _np.array([0, 1, 0, 1]))
        probe.predict(_np.zeros((2, 2)))
        LOGGER.info("GPU detected; using device=cuda")
        return "cuda"
    except Exception:  # noqa: BLE001 - any failure means no usable GPU
        LOGGER.info("no usable GPU; using device=cpu")
        return "cpu"
