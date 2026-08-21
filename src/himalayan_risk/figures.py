"""Figures.

Produces the three two-panel figures used in the manuscript, plus the
individual panels for anyone who wants them separately. Panels are
pre-combined into wide PNGs so they fit a two-column layout without
further editing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)

DPI = 300


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", path)
    return path


def figure_curves(y_test, proba, figures_dir: Path) -> Path:
    """Figure 2: ROC and precision-recall curves side by side.

    Under a positive rate near 5% the precision-recall curve is the more
    informative of the two, so average precision is annotated against the
    no-skill baseline.
    """
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    no_skill = float(np.mean(y_test))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="tab:orange", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, label=f"AP = {ap:.3f}")
    axes[1].axhline(no_skill, linestyle="--", color="grey", linewidth=1,
                    label=f"No skill = {no_skill:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, Path(figures_dir) / "fig2_roc_pr_curves.png")


def figure_confusion_and_reliability(
    y_test, proba, threshold: float, figures_dir: Path
) -> Path:
    """Figure 3: confusion matrix beside the reliability diagram.

    Predicted probabilities concentrate in the lower part of the unit
    interval, which limits how many calibration bins are populated - the
    diagram indicates calibration behaviour rather than settling the
    question of calibration quality.
    """
    predicted = (np.asarray(proba) >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predicted)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    n_bins = 10
    fraction, mean_predicted = calibration_curve(y_test, proba, n_bins=n_bins)
    axes[1].plot([0, 1], [0, 1], "--", label="Perfect")
    axes[1].plot(mean_predicted, fraction, marker="o", label="Model")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of Positives")
    axes[1].set_title("Calibration Curve")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    return _save(fig, Path(figures_dir) / "fig3_confusion_reliability.png")


def figure_threshold_and_importance(
    y_test,
    proba,
    threshold: float,
    importance: pd.DataFrame,
    figures_dir: Path,
    top_n: int = 15,
) -> Path:
    """Figure 4: threshold sensitivity beside ranked feature importance."""
    grid = np.linspace(0.05, 0.50, 50)
    precisions, recalls = [], []
    for tau in grid:
        predicted = (np.asarray(proba) >= tau).astype(int)
        precisions.append(precision_score(y_test, predicted, zero_division=0))
        recalls.append(recall_score(y_test, predicted, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(grid, precisions, label="Precision")
    axes[0].plot(grid, recalls, label="Recall")
    axes[0].axvline(threshold, linestyle="--", color="grey",
                    label=f"Chosen threshold = {threshold:.3f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold vs Precision/Recall")
    axes[0].legend(loc="upper right", fontsize=8)

    if not importance.empty:
        top = importance.head(top_n)
        sns.barplot(x="Importance", y="Feature", data=top, ax=axes[1], color="tab:blue")
        axes[1].set_title("Top Feature Importances")
    else:
        axes[1].text(0.5, 0.5, "importance unavailable", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    return _save(fig, Path(figures_dir) / "fig4_threshold_importance.png")


def figure_probability_distribution(y_test, proba, figures_dir: Path) -> Path:
    """Supplementary: score distributions for each class."""
    y_test = np.asarray(y_test)
    proba = np.asarray(proba)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(proba[y_test == 0], bins=30, alpha=0.55, label="Non-fatal")
    ax.hist(proba[y_test == 1], bins=30, alpha=0.55, label="Fatal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()

    fig.tight_layout()
    return _save(fig, Path(figures_dir) / "figS1_probability_distribution.png")


def make_all_figures(
    y_test,
    proba,
    threshold: float,
    importance: pd.DataFrame,
    figures_dir: Path,
) -> Dict[str, Path]:
    return {
        "fig2": figure_curves(y_test, proba, figures_dir),
        "fig3": figure_confusion_and_reliability(y_test, proba, threshold, figures_dir),
        "fig4": figure_threshold_and_importance(
            y_test, proba, threshold, importance, figures_dir
        ),
        "figS1": figure_probability_distribution(y_test, proba, figures_dir),
    }
