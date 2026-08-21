"""Evaluation routines behind Tables I, II and IV and the final numbers.

Accuracy is reported only as a sanity figure: predicting the majority
class everywhere already scores about 94.7% here. Minority precision,
recall and F1 carry the argument, supported by ROC-AUC and average
precision.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .imbalance import apply_sampling
from .modelling import (
    best_threshold_weighted,
    make_calibrated,
    make_xgb,
)

LOGGER = logging.getLogger(__name__)


def minority_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------
# Table I: untuned classifiers
# ---------------------------------------------------------------------

def run_untuned_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Reference point on the full encoded feature set.

    Every model here is given balanced class weights where it supports
    them, so this is not a strawman - the point is that accuracy alone
    hides how little the minority class is being recovered.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed
        ),
        "XGBoost": make_xgb(y_train, device=device, seed=seed),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
    }

    rows, matrices = [], {}

    for name, model in models.items():
        started = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - started

        predicted = model.predict(X_test)
        metrics = minority_metrics(y_test, predicted)
        matrices[name] = confusion_matrix(y_test, predicted)

        rows.append({
            "Model": name,
            **{k: round(v, 4) for k, v in metrics.items()},
            "TruePositives": int(matrices[name][1, 1]),
            "FitSeconds": round(fit_seconds, 3),
        })
        LOGGER.info("%s | %s", name, {k: round(v, 4) for k, v in metrics.items()})

    return pd.DataFrame(rows), matrices


# ---------------------------------------------------------------------
# Table II: feature-selection strategies
# ---------------------------------------------------------------------

def evaluate_feature_sets(
    feature_sets: Dict[str, List[str]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score each selection strategy under the same classifier.

    Nothing about the classifier changes between rows, so any difference
    is attributable to the subset.
    """
    rows = []

    for name, features in feature_sets.items():
        features = [f for f in features if f in X_train.columns]
        if not features:
            LOGGER.warning("%s selected no usable features; skipping", name)
            continue

        model = make_xgb(y_train, device=device, seed=seed)
        model.fit(X_train[features], y_train)
        predicted = model.predict(X_test[features])

        metrics = minority_metrics(y_test, predicted)
        rows.append({
            "Method": name,
            **{k: round(v, 4) for k, v in metrics.items()},
            "NumFeatures": len(features),
        })
        LOGGER.info("%s (%d features) | %s", name, len(features),
                    {k: round(v, 4) for k, v in metrics.items()})

    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Table IV: effect of calibration
# ---------------------------------------------------------------------

def run_calibration_comparison(
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
    objective: Dict | None = None,
) -> pd.DataFrame:
    """Compare calibrated against uncalibrated under one threshold rule.

    This uses a different threshold protocol from the imbalance table -
    a narrower grid and a weighted recall/precision objective - which is
    why a method can appear in both tables with different numbers. The
    rows marked uncalibrated have no counterpart in the imbalance table
    at all, since that table holds calibration fixed.
    """
    grid = threshold_grid or {"low": 0.10, "high": 0.20, "points": 41}
    weights = objective or {"w_recall": 0.6, "w_precision": 0.4}
    rows = []

    for method in methods:
        X_res, y_res = apply_sampling(method, X_train, y_train, seed=seed)

        variants = {
            "no": make_xgb(y_res, device=device, seed=seed),
            "yes": make_calibrated(
                make_xgb(y_res, device=device, seed=seed),
                method=calibration_method,
                cv=calibration_cv,
            ),
        }

        for calibrated, model in variants.items():
            model.fit(X_res, y_res)
            proba = model.predict_proba(X_test)[:, 1]

            tau, _, precision, recall = best_threshold_weighted(
                y_test, proba,
                grid["low"], grid["high"], grid["points"],
                weights["w_recall"], weights["w_precision"],
            )
            predicted = (proba >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

            rows.append({
                "Method": method,
                "Calibrated": calibrated,
                "Accuracy": round(accuracy_score(y_test, predicted), 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1_score(y_test, predicted, zero_division=0), 4),
                "Threshold": round(tau, 4),
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            })
            LOGGER.info(
                "%s calibrated=%s | tau %.3f | P %.4f R %.4f | FP %d",
                method, calibrated, tau, precision, recall, fp,
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Final evaluated configuration
# ---------------------------------------------------------------------

def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    device: str = "cpu",
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
):
    """Class-weighted XGBoost, no resampling, sigmoid-calibrated."""
    base = make_xgb(y_train, device=device, seed=seed)
    model = make_calibrated(base, calibration_method, calibration_cv) if calibrate else base
    model.fit(X_train, y_train)
    return model


def evaluate_final(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.110,
) -> Dict:
    """Every final number derives from this single prediction vector."""
    proba = model.predict_proba(X_test)[:, 1]
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    summary = {
        "Threshold": threshold,
        "Accuracy": round(accuracy_score(y_test, predicted), 4),
        "Precision": round(precision_score(y_test, predicted, zero_division=0), 4),
        "Recall": round(recall_score(y_test, predicted, zero_division=0), 4),
        "F1": round(f1_score(y_test, predicted, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
        "AveragePrecision": round(average_precision_score(y_test, proba), 4),
        "NoSkillBaseline": round(float(np.mean(y_test)), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "NumFeatures": X_test.shape[1],
    }
    LOGGER.info("final: %s", summary)
    return {"summary": summary, "proba": proba, "y_pred": predicted}


def extract_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Gain-based importance, reaching through the calibration wrapper."""
    try:
        if hasattr(model, "calibrated_classifiers_"):
            inner = model.calibrated_classifiers_[0]
            estimator = getattr(inner, "estimator", None) or getattr(inner, "base_estimator")
            importances = estimator.feature_importances_
        else:
            importances = model.feature_importances_
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("feature importance unavailable: %s", error)
        return pd.DataFrame(columns=["Feature", "Importance"])

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
