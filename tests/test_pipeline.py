"""Fast unit tests. No data download required.

Run with `pytest tests/` or `make test`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from himalayan_risk import preprocess  # noqa: E402
from himalayan_risk.config import load_config  # noqa: E402
from himalayan_risk.features import summarise_weather  # noqa: E402
from himalayan_risk.imbalance import flag_degenerate  # noqa: E402
from himalayan_risk.modelling import (  # noqa: E402
    best_threshold_f1,
    best_threshold_weighted,
    composite_score,
    mask_fitness,
    scale_pos_weight,
)
from himalayan_risk.seeds import set_seeds  # noqa: E402


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

def test_config_loads_and_overrides():
    cfg = load_config(overrides=["seed=7", "leakage.exclude_termreason=false"])
    assert cfg.get_dotted("seed") == 7
    assert cfg.get_dotted("leakage.exclude_termreason") is False


def test_config_default_excludes_termreason():
    """The default must be the leakage-free configuration."""
    cfg = load_config()
    assert cfg.get_dotted("leakage.exclude_termreason") is True


def test_config_missing_key_raises():
    cfg = load_config()
    with pytest.raises(KeyError):
        cfg.get_dotted("nonexistent.key")


# ---------------------------------------------------------------------
# The exclusion rule
# ---------------------------------------------------------------------

def _frame_with_outcome_fields() -> pd.DataFrame:
    return pd.DataFrame({
        "mdeaths": [0, 1, 0],
        "hdeaths": [0, 0, 1],
        "death_rate": [0.0, 0.2, 0.0],
        "severe_outcome": [0, 1, 0],
        "smtmembers": [3, 1, 5],
        "success1": ["Y", "N", "Y"],
        "success2": ["Y", "N", "Y"],
        "success3": ["Y", "N", "Y"],
        "success4": ["Y", "N", "Y"],
        "termreason": [1, 9, 2],
        "totmembers": [8, 4, 9],
        "temp_mean": [-5.0, -12.0, -3.0],
        "fatality": [0, 1, 0],
    })


EXCLUDE = ["mdeaths", "hdeaths", "death_rate", "severe_outcome",
           "smtmembers", "success1", "success2", "success3", "success4"]


def test_outcome_fields_are_removed():
    cleaned, dropped = preprocess.drop_outcome_fields(
        _frame_with_outcome_fields(), EXCLUDE, exclude_termreason=True
    )
    for field in EXCLUDE + ["termreason"]:
        assert field not in cleaned.columns, f"{field} survived the exclusion rule"
    assert "termreason" in dropped
    # The target and legitimate predictors must survive.
    assert "fatality" in cleaned.columns
    assert "totmembers" in cleaned.columns
    assert "temp_mean" in cleaned.columns


def test_termreason_retained_only_when_asked():
    cleaned, dropped = preprocess.drop_outcome_fields(
        _frame_with_outcome_fields(), EXCLUDE, exclude_termreason=False
    )
    assert "termreason" in cleaned.columns
    assert "termreason" not in dropped
    # Everything else still goes, even in the ablation.
    for field in EXCLUDE:
        assert field not in cleaned.columns


# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------

def test_missingness_indicator_precedes_imputation():
    frame = pd.DataFrame({
        "temp_mean": [1.0, np.nan, 3.0, np.nan],
        "wind_max": [10.0, 20.0, np.nan, 40.0],
    })
    out = preprocess.add_missingness_indicators_and_impute(frame)

    assert list(out["temp_mean_missing"]) == [0, 1, 0, 1]
    assert not out["temp_mean"].isna().any()
    # Median of the observed values, not of the filled column.
    assert out.loc[1, "temp_mean"] == pytest.approx(2.0)


def test_correlation_filter_drops_one_of_a_duplicated_pair():
    base = np.arange(50, dtype=float)
    X_train = pd.DataFrame({
        "a": base,
        "b": base * 2 + 1e-9,     # perfectly correlated with a
        "c": np.sin(base),
    })
    X_test = X_train.copy()

    reduced_train, reduced_test, dropped = preprocess.correlation_filter(
        X_train, X_test, threshold=0.90
    )
    assert len(dropped) == 1
    assert "c" in reduced_train.columns
    assert list(reduced_train.columns) == list(reduced_test.columns)


def test_split_is_stratified_and_repeatable():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 4)), columns=list("abcd"))
    y = pd.Series((rng.random(1000) < 0.053).astype(int))

    first = preprocess.stratified_split(X, y, test_size=0.3, seed=42)
    second = preprocess.stratified_split(X, y, test_size=0.3, seed=42)

    pd.testing.assert_frame_equal(first[0], second[0])
    assert abs(first[2].mean() - first[3].mean()) < 0.01


# ---------------------------------------------------------------------
# Climate features
# ---------------------------------------------------------------------

def test_summarise_weather_on_missing_slice():
    summary = summarise_weather(None)
    assert all(np.isnan(value) for value in summary.values())


def test_event_counters_use_configured_thresholds():
    frame = pd.DataFrame({
        "temperature_2m": [0.0, 6.0, 6.5, -10.0],   # one jump above +5
        "snowfall": [0.0, 12.0, 3.0, 40.0],          # two days above 10 mm
        "wind_speed_100m": [10.0, 80.0, 90.0, 5.0],  # two days above 70
        "surface_pressure": [400.0, 401.0, 402.0, 403.0],
        "weather_code": [1, 75, 80, 2],              # two codes at or above 70
    })
    summary = summarise_weather(frame)

    assert summary["heavy_snow_days"] == 2
    assert summary["extreme_wind_days"] == 2
    assert summary["rapid_warming_days"] == 1
    assert summary["storm_days"] == 2
    assert summary["snow_total"] == pytest.approx(55.0)


def test_summarise_weather_respects_custom_thresholds():
    frame = pd.DataFrame({
        "temperature_2m": [0.0, 3.0],
        "snowfall": [5.0, 5.0],
        "wind_speed_100m": [50.0, 50.0],
        "surface_pressure": [400.0, 400.0],
        "weather_code": [10, 10],
    })
    strict = summarise_weather(frame, heavy_snow_mm=10.0, extreme_wind_kmh=70.0)
    loose = summarise_weather(frame, heavy_snow_mm=1.0, extreme_wind_kmh=10.0)

    assert strict["heavy_snow_days"] == 0
    assert loose["heavy_snow_days"] == 2
    assert loose["extreme_wind_days"] == 2


# ---------------------------------------------------------------------
# Fitness and thresholds
# ---------------------------------------------------------------------

def test_composite_score_weights_recall_most():
    y_true = np.array([0, 0, 1, 1])
    high_recall = np.array([1, 1, 1, 1])      # recall 1.0, precision 0.5
    high_precision = np.array([0, 0, 1, 0])   # recall 0.5, precision 1.0

    assert composite_score(y_true, high_recall) > composite_score(y_true, high_precision)


def test_size_penalty_prefers_the_smaller_of_two_equal_masks():
    n = 10
    evaluator = lambda mask: 0.5  # noqa: E731 - constant, so only size differs

    small = np.zeros(n, dtype=int)
    small[:2] = 1
    large = np.ones(n, dtype=int)

    assert mask_fitness(small, evaluator, n, 0.02) > mask_fitness(large, evaluator, n, 0.02)


def test_empty_mask_scores_zero():
    assert mask_fitness(np.zeros(10, dtype=int), lambda m: 1.0, 10) == 0.0


def test_scale_pos_weight_matches_imbalance_ratio():
    y = np.array([0] * 95 + [1] * 5)
    assert scale_pos_weight(y) == pytest.approx(19.0)


def test_the_two_threshold_protocols_are_distinct():
    """Tables III and IV differ because these rules differ, not by error."""
    rng = np.random.default_rng(1)
    y_true = (rng.random(600) < 0.05).astype(int)
    proba = np.clip(rng.beta(2, 12, 600) + 0.15 * y_true, 0, 1)

    tau_f1, f1, _, _ = best_threshold_f1(y_true, proba)
    tau_weighted, _, _, _ = best_threshold_weighted(y_true, proba)

    assert 0.10 <= tau_f1 <= 0.30
    assert 0.10 <= tau_weighted <= 0.20   # the narrower grid
    assert 0.0 <= f1 <= 1.0


def test_threshold_search_respects_grid_bounds():
    y_true = np.array([0, 1] * 50)
    proba = np.linspace(0, 1, 100)
    tau, _, _, _ = best_threshold_f1(y_true, proba, low=0.4, high=0.6, points=5)
    assert 0.4 <= tau <= 0.6


# ---------------------------------------------------------------------
# Degenerate-classifier detection
# ---------------------------------------------------------------------

def test_trivial_classifier_is_flagged():
    table = pd.DataFrame([
        {"Method": "UnderSampling", "Recall": 1.0, "Precision": 0.0532, "F1": 0.1010},
        {"Method": "Baseline", "Recall": 0.2318, "Precision": 0.2734, "F1": 0.2509},
    ])
    flagged = flag_degenerate(table, base_rate=0.0532)

    assert bool(flagged.loc[0, "Degenerate"]) is True
    assert bool(flagged.loc[1, "Degenerate"]) is False


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------

def test_set_seeds_makes_draws_repeatable():
    import random

    set_seeds(42)
    first = (random.random(), np.random.rand(), np.random.randint(0, 2, 20).tolist())

    set_seeds(42)
    second = (random.random(), np.random.rand(), np.random.randint(0, 2, 20).tolist())

    assert first == second
