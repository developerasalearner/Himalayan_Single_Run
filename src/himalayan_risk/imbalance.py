"""Imbalance treatment comparison (Table III).

Six options on the selected subset: SMOTE, ADASYN, SMOTE-ENN,
SMOTE-Tomek, random undersampling, and a weighted baseline that leaves
the distribution untouched.

Calibration is held fixed across all six - every treatment is fitted as a
sigmoid-calibrated, class-weighted classifier - so only the sampling
strategy varies. Each treatment then gets its own operating point,
chosen by maximising minority F1, which avoids penalising methods whose
scores concentrate in a different part of the unit interval.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score

from .modelling import best_threshold_f1, make_calibrated, make_xgb

LOGGER = logging.getLogger(__name__)

SAMPLERS = {
    "SMOTE": SMOTE,
    "ADASYN": ADASYN,
    "SMOTEENN": SMOTEENN,
    "SMOTETomek": SMOTETomek,
    "UnderSampling": RandomUnderSampler,
}


def apply_sampling(
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Resample the training partition only. 'Baseline' is a no-op."""
    if method not in SAMPLERS:
        return X, y
    sampler = SAMPLERS[method](random_state=seed)
    return sampler.fit_resample(X, y)


def run_imbalance_comparison(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    methods: List[str],
    seed: int = 42,
    device: str = "cpu",
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    threshold_grid: Dict | None = None,
) -> pd.DataFrame:
    """Fit every treatment, pick its own F1-optimal threshold, tabulate."""
    grid = threshold_grid or {"low": 0.10, "high": 0.30, "points": 41}
    rows = []

    for method in methods:
        LOGGER.info("=== imbalance treatment: %s ===", method)

        X_res, y_res = apply_sampling(method, X_train, y_train, seed=seed)
        LOGGER.info("training rows after resampling: %d", len(X_res))

        base = make_xgb(y_train, device=device, seed=seed)
        model = make_calibrated(base, method=calibration_method, cv=calibration_cv)
        model.fit(X_res, y_res)

        proba = model.predict_proba(X_test)[:, 1]
        tau, f1, precision, recall = best_threshold_f1(
            y_test, proba, grid["low"], grid["high"], grid["points"]
        )
        predicted = (proba >= tau).astype(int)

        LOGGER.info(
            "%s | tau %.3f | F1 %.4f | precision %.4f | recall %.4f",
            method, tau, f1, precision, recall,
        )

        rows.append({
            "Method": method,
            "Threshold": round(tau, 4),
            "F1": round(f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Accuracy": round(accuracy_score(y_test, predicted), 4),
            "TrainRows": len(X_res),
            "Features": X_train.shape[1],
        })

    table = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return table


def flag_degenerate(table: pd.DataFrame, base_rate: float, tol: float = 1e-3) -> pd.DataFrame:
    """Mark rows that have collapsed into a trivial classifier.

    A recall of exactly one paired with a precision equal to the test-set
    positive rate means the model is labelling everything positive.
    """
    out = table.copy()
    out["Degenerate"] = (
        (out["Recall"] >= 1.0 - tol) & (np.abs(out["Precision"] - base_rate) < 0.01)
    )
    return out
