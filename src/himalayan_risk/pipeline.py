"""End-to-end assembly of the modelling table.

`build_dataset` runs everything from the two raw archives through to a
train/test split with the redundancy screen applied. It caches the
result, so the expensive weather-slicing pass happens once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from . import data, features, preprocess
from .config import Config

LOGGER = logging.getLogger(__name__)

SplitBundle = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def _cache_key(cfg: Config) -> str:
    """Cached tables are keyed by whether termreason was excluded."""
    return "excl_termreason" if cfg.get_dotted("leakage.exclude_termreason") else "incl_termreason"


def build_modelling_table(cfg: Config, use_cache: bool = True) -> pd.DataFrame:
    """Raw archives -> encoded, imputed, leakage-screened modelling table."""
    processed_dir = cfg.get_path("paths.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / f"modelling_table_{_cache_key(cfg)}.parquet"

    if use_cache and cache_path.exists():
        LOGGER.info("loading cached modelling table from %s", cache_path)
        return pd.read_parquet(cache_path)

    expeditions = data.load_expeditions(cfg.get_path("paths.expeditions_dir"))
    weather = data.load_weather(
        cfg.get_path("paths.weather_dir"),
        sort_by_date=cfg.get_dotted("data.sort_weather_by_date"),
    )

    kept = features.filter_to_peaks_with_weather(expeditions, weather)
    kept = features.add_expedition_intervals(kept)

    climate = features.build_climate_features(
        kept,
        weather,
        heavy_snow_mm=cfg.get_dotted("features.heavy_snow_mm"),
        extreme_wind_kmh=cfg.get_dotted("features.extreme_wind_kmh"),
        rapid_warming_delta_c=cfg.get_dotted("features.rapid_warming_delta_c"),
        storm_weather_code_min=cfg.get_dotted("features.storm_weather_code_min"),
    )

    full = pd.concat(
        [kept.reset_index(drop=True), climate.reset_index(drop=True)], axis=1
    )
    full = features.add_vulnerability_features(full)
    full = features.add_target(full)

    cleaned, excluded = preprocess.drop_outcome_fields(
        full,
        exclude_fields=cfg.get_dotted("leakage.exclude_fields"),
        exclude_termreason=cfg.get_dotted("leakage.exclude_termreason"),
    )
    cleaned = preprocess.add_missingness_indicators_and_impute(cleaned)
    encoded, _ = preprocess.label_encode(cleaned)

    encoded.to_parquet(cache_path, index=False)
    LOGGER.info("wrote modelling table (%d x %d) to %s",
                *encoded.shape, cache_path)

    manifest = {
        "rows": int(encoded.shape[0]),
        "columns": int(encoded.shape[1]),
        "positives": int(encoded[preprocess.TARGET].sum()),
        "positive_rate": float(encoded[preprocess.TARGET].mean()),
        "excluded_outcome_fields": excluded,
        "termreason_excluded": bool(cfg.get_dotted("leakage.exclude_termreason")),
    }
    (processed_dir / f"manifest_{_cache_key(cfg)}.json").write_text(
        json.dumps(manifest, indent=2)
    )

    return encoded


def build_splits(cfg: Config, use_cache: bool = True) -> SplitBundle:
    """Modelling table -> stratified split with the redundancy screen applied."""
    table = build_modelling_table(cfg, use_cache=use_cache)
    X, y = preprocess.split_features_target(table)

    X_train, X_test, y_train, y_test = preprocess.stratified_split(
        X,
        y,
        test_size=cfg.get_dotted("preprocess.test_size"),
        seed=cfg.get_dotted("seed"),
        stratify=cfg.get_dotted("preprocess.stratify"),
    )

    X_train, X_test, _ = preprocess.correlation_filter(
        X_train, X_test, threshold=cfg.get_dotted("preprocess.correlation_threshold")
    )

    return X_train, X_test, y_train, y_test


def save_table(frame: pd.DataFrame, results_dir: Path, name: str) -> Path:
    """Write a results table as CSV and echo it to the log."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    LOGGER.info("wrote %s\n%s", path, frame.to_string(index=False))
    return path
