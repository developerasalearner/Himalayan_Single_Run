"""Embedded selection stage: an L1 pre-filter followed by greedy
backward elimination scored by cross-validation.

Note on the manuscript: Equation 4 states the objective as AUC. The
implementation here, like the original notebook, scores each candidate
subset with the same recall-weighted composite used by the metaheuristic
stage (Equation 5 without the size penalty). See docs/PAPER_MAPPING.md.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..modelling import composite_score, scale_pos_weight

LOGGER = logging.getLogger(__name__)


def _make_l1_logistic(seed: int) -> LogisticRegression:
    """Build a pure-L1 logistic model across scikit-learn versions.

    scikit-learn 1.8 deprecated `penalty` in favour of `l1_ratio`; passing
    the old argument there silently mixes in an L2 term, which would
    change which features survive. This picks the right spelling.
    """
    common = dict(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    try:
        from sklearn import __version__ as sklearn_version

        major, minor = (int(part) for part in sklearn_version.split(".")[:2])
    except Exception:  # noqa: BLE001
        major, minor = 1, 0

    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


def l1_prefilter(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> List[str]:
    """Zero out features an L1-penalised logistic model finds uninformative."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train)

    model = _make_l1_logistic(seed)
    model.fit(scaled, y_train)

    coefficients = model.coef_.ravel()
    selected = list(X_train.columns[coefficients != 0])

    LOGGER.info("L1 kept %d of %d features", len(selected), X_train.shape[1])
    return selected


def score_subset(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    cv_folds: int = 5,
    seed: int = 42,
    device: str = "cpu",
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> float:
    """Mean cross-validated composite score for one feature subset."""
    subset = X[features]
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in splitter.split(subset, y):
        X_tr, X_val = subset.iloc[train_idx], subset.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight(y_tr),
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            device=device,
            verbosity=0,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        scores.append(composite_score(y_val, model.predict(X_val)))

    return float(np.mean(scores))


def greedy_backward_elimination(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    initial_features: List[str],
    **score_kwargs,
) -> List[str]:
    """Remove the least useful feature one at a time until nothing helps.

    Each pass tries deleting every surviving feature, keeps the deletion
    with the best score, and stops when no single deletion improves on the
    current subset.
    """
    current = list(initial_features)
    best = score_subset(X_train, y_train, current, **score_kwargs)
    LOGGER.info("initial score %.4f with %d features", best, len(current))

    improved = True
    while improved and len(current) > 1:
        improved = False
        trials = []

        for feature in current:
            trial = [f for f in current if f != feature]
            if not trial:
                continue
            trials.append((feature, score_subset(X_train, y_train, trial, **score_kwargs)))

        if not trials:
            break

        feature, score = max(trials, key=lambda pair: pair[1])
        if score > best:
            LOGGER.info("dropping %s -> %.4f", feature, score)
            current.remove(feature)
            best = score
            improved = True

    LOGGER.info("embedded stage settled on %d features (score %.4f)", len(current), best)
    return current


def run_embedded_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    device: str = "cpu",
    cv_folds: int = 5,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> Tuple[List[str], float]:
    """L1 pre-filter, then greedy backward elimination."""
    prefiltered = l1_prefilter(X_train, y_train, seed=seed)
    kwargs = dict(
        cv_folds=cv_folds,
        seed=seed,
        device=device,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    selected = greedy_backward_elimination(
        X_train[prefiltered], y_train, prefiltered, **kwargs
    )
    return selected, score_subset(X_train, y_train, selected, **kwargs)
