"""Cleaning, encoding, redundancy screening and the train/test split.

The exclusion rule is applied here and nowhere else, so that what the
model may see is decided in one readable place.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .features import CLIMATE_COLUMNS

LOGGER = logging.getLogger(__name__)

TARGET = "fatality"

# Identifiers, free text and high-cardinality strings. Dropped because
# they carry no transferable signal, not because they leak.
NON_PREDICTIVE_COLUMNS = [
    "expid", "peakid", "primid", "chksum",
    "route1", "route2", "route3", "route4",
    "leaders", "sponsor", "nation", "host",
    "countries", "approach", "agency",
    "campsites", "accidents", "achievment",
    "othersmts", "termnote",
    "ascent1", "ascent2", "ascent3", "ascent4",
    "bcdate", "smtdate", "termdate", "start_date", "end_date",
    "smttime", "smtdays", "summit_rate",
]

DURATION_COLUMNS = ["totdays", "exposure_days", "camp_density", "technical_difficulty"]


def drop_outcome_fields(
    frame: pd.DataFrame,
    exclude_fields: Sequence[str],
    exclude_termreason: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove fields that encode the outcome, then the non-predictive ones.

    `termreason` is separated out because Section V-E of the paper
    readmits it to measure what the exclusion rule costs. It records why
    a trip ended, and one of its categories covers accidents involving
    death or serious injury, so it partially encodes the label.
    """
    to_drop = [c for c in exclude_fields if c in frame.columns]

    if exclude_termreason and "termreason" in frame.columns:
        to_drop.append("termreason")
    elif not exclude_termreason:
        LOGGER.warning(
            "termreason RETAINED - this is the leakage ablation, not the "
            "paper's main configuration"
        )

    non_predictive = [c for c in NON_PREDICTIVE_COLUMNS if c in frame.columns]
    cleaned = frame.drop(columns=to_drop + non_predictive)

    LOGGER.info(
        "excluded %d outcome-encoding fields (%s) and %d non-predictive fields",
        len(to_drop), ", ".join(sorted(to_drop)) or "none", len(non_predictive),
    )
    return cleaned, to_drop


def add_missingness_indicators_and_impute(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag missing climate values, then median-fill.

    Missingness carries information: an absent weather slice usually means
    an undated or aborted expedition, so the indicator is retained as a
    feature. The median resists the long right tails in snowfall and wind.
    """
    out = frame.copy()

    for column in CLIMATE_COLUMNS:
        if column not in out.columns:
            continue
        out[f"{column}_missing"] = out[column].isna().astype(int)
        out[column] = out[column].fillna(out[column].median())

    for column in DURATION_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna(out[column].median())

    # Anything still missing in a numeric column would break the encoders
    # downstream. Fill it and say so, rather than failing silently.
    numeric = out.select_dtypes(include=[np.number]).columns
    residual = [c for c in numeric if out[c].isna().any()]
    if residual:
        LOGGER.info("median-filling residual gaps in: %s", ", ".join(residual))
        for column in residual:
            out[column] = out[column].fillna(out[column].median())

    return out


def label_encode(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode every non-numeric column, keeping the fitted encoders."""
    encoded = frame.copy()
    categorical = encoded.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns
    encoders: Dict[str, LabelEncoder] = {}

    for column in categorical:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder

    if len(categorical):
        LOGGER.info("label-encoded %d columns: %s", len(categorical), ", ".join(categorical))
    return encoded, encoders


def split_features_target(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[TARGET]), frame[TARGET]


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.30,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split once, with stratification, under a fixed seed."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target if stratify else None,
        random_state=seed,
    )
    LOGGER.info(
        "split: %d train (%d positive, %.3f%%) / %d test (%d positive, %.3f%%)",
        len(X_train), int(y_train.sum()), y_train.mean() * 100,
        len(X_test), int(y_test.sum()), y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Drop one member of every pair whose absolute correlation exceeds
    the threshold.

    Fitted on the training partition only, then applied to the test
    partition, so the test set contributes nothing to the decision.
    """
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [column for column in upper.columns if (upper[column] > threshold).any()]

    LOGGER.info(
        "redundancy screen: %d -> %d columns (dropped %s)",
        X_train.shape[1], X_train.shape[1] - len(dropped), ", ".join(dropped) or "none",
    )
    return X_train.drop(columns=dropped), X_test.drop(columns=dropped), dropped
