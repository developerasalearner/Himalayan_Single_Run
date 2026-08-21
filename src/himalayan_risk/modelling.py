"""Shared modelling pieces: the classifier factory, the fitness function
used during feature search, and the two threshold-selection protocols.

The two protocols differ deliberately, and conflating them is what makes
Tables III and IV look inconsistent at first glance:

* `best_threshold_f1`   - Table III. Grid [0.10, 0.30], maximise minority F1.
* `best_threshold_weighted` - Table IV. Grid [0.10, 0.20], maximise
  0.6*recall + 0.4*precision.

Both tables hold everything else fixed, so a row can appear in both with
different numbers purely because the operating point was chosen under a
different rule.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)


def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Majority-to-minority ratio, applied to positive instances."""
    y = np.asarray(y)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / positives if positives else 1.0


def make_xgb(
    y_train,
    device: str = "cpu",
    seed: int = 42,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    weighted: bool = True,
) -> XGBClassifier:
    """Build a class-weighted XGBoost classifier.

    Skew is absorbed by reweighting the loss rather than by resampling,
    which leaves the geometry of the minority class untouched.
    """
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        verbosity=0,
        random_state=seed,
    )
    if weighted:
        params["scale_pos_weight"] = scale_pos_weight(y_train)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if max_depth is not None:
        params["max_depth"] = max_depth
    if learning_rate is not None:
        params["learning_rate"] = learning_rate

    return XGBClassifier(**params)


def make_calibrated(base, method: str = "sigmoid", cv: int = 3) -> CalibratedClassifierCV:
    """Wrap a classifier in a cross-validated calibration map.

    A sigmoid map is fitted to the raw scores by three-fold CV, so the
    outputs can be read as probabilities rather than as rankings.
    """
    return CalibratedClassifierCV(base, method=method, cv=cv)


# ---------------------------------------------------------------------
# Fitness used by every feature-selection optimiser
# ---------------------------------------------------------------------

def composite_score(
    y_true,
    y_pred,
    w_recall: float = 0.5,
    w_f1: float = 0.3,
    w_precision: float = 0.2,
) -> float:
    """Minority-sensitive composite. Recall carries the largest weight,
    reflecting that a missed fatality costs more than a false alarm."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return w_recall * recall + w_f1 * f1 + w_precision * precision


def mask_fitness(
    mask: np.ndarray,
    evaluator: Callable[[np.ndarray], float],
    n_features: int,
    size_penalty: float = 0.02,
) -> float:
    """Equation 5: composite score minus a levy per retained feature."""
    if np.sum(mask) == 0:
        return 0.0
    return evaluator(mask) - size_penalty * (np.sum(mask) / n_features)


def make_evaluator(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    y_tr: pd.Series,
    y_val: pd.Series,
    device: str = "cpu",
    seed: int = 42,
    n_estimators: int = 50,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    weights: Dict[str, float] | None = None,
) -> Callable[[np.ndarray], float]:
    """Return a function scoring one binary feature mask.

    The surrogate is deliberately small and is fitted on an inner split of
    the training partition, so the test set stays untouched during search.
    """
    weights = weights or {}
    spw = scale_pos_weight(y_tr)

    def evaluator(mask: np.ndarray) -> float:
        if np.sum(mask) == 0:
            return 0.0
        columns = np.where(mask == 1)[0]

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            scale_pos_weight=spw,
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            max_bin=256,
            verbosity=0,
            random_state=seed,
        )
        model.fit(X_tr.iloc[:, columns], y_tr)
        predictions = model.predict(X_val.iloc[:, columns])

        return composite_score(
            y_val,
            predictions,
            w_recall=weights.get("w_recall", 0.5),
            w_f1=weights.get("w_f1", 0.3),
            w_precision=weights.get("w_precision", 0.2),
        )

    return evaluator


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def _grid(low: float, high: float, points: int) -> np.ndarray:
    return np.linspace(low, high, points)


def best_threshold_f1(
    y_true,
    y_proba,
    low: float = 0.10,
    high: float = 0.30,
    points: int = 41,
) -> Tuple[float, float, float, float]:
    """Table III protocol: maximise minority F1 over the grid.

    The grid avoids penalising methods whose scores concentrate elsewhere
    in the unit interval.

    Returns (threshold, f1, precision, recall).
    """
    best = (0.5, -1.0, 0.0, 0.0)
    for tau in _grid(low, high, points):
        predicted = (np.asarray(y_proba) >= tau).astype(int)
        f1 = f1_score(y_true, predicted, zero_division=0)
        if f1 > best[1]:
            best = (
                float(tau),
                float(f1),
                float(precision_score(y_true, predicted, zero_division=0)),
                float(recall_score(y_true, predicted, zero_division=0)),
            )
    return best


def best_threshold_weighted(
    y_true,
    y_proba,
    low: float = 0.10,
    high: float = 0.20,
    points: int = 41,
    w_recall: float = 0.6,
    w_precision: float = 0.4,
) -> Tuple[float, float, float, float]:
    """Table IV protocol: maximise 0.6*recall + 0.4*precision.

    Equivalent, up to an additive constant, to 0.6*recall - 0.4*(1 -
    precision), so it selects the same threshold as that form.

    Returns (threshold, objective, precision, recall).
    """
    best = (0.5, -1.0, 0.0, 0.0)
    for tau in _grid(low, high, points):
        predicted = (np.asarray(y_proba) >= tau).astype(int)
        precision = precision_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        objective = w_recall * recall + w_precision * precision
        if objective > best[1]:
            best = (float(tau), float(objective), float(precision), float(recall))
    return best
