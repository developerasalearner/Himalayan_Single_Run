"""Weather-aware feature construction.

Three stages, in order:

1. Temporal alignment - resolve each expedition's recorded interval and
   slice the matching peak's hourly weather series to it.
2. Climate descriptors - compress each slice into summary statistics and
   event counters that averages would otherwise hide.
3. Vulnerability descriptors - party size, exposure, camp density, route
   character and oxygen usage.

The weather statistics summarise the interval from base camp to
termination. That endpoint is unknown while a climb is under way, so
these are retrospective descriptors, not forecast inputs.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CLIMATE_COLUMNS = [
    "temp_mean", "temp_max", "temp_min", "temp_std",
    "snow_total", "snow_max", "heavy_snow_days",
    "wind_mean", "wind_max", "extreme_wind_days",
    "pressure_mean", "pressure_min",
    "rapid_warming_days", "storm_days",
]

_EMPTY_CLIMATE = {name: np.nan for name in CLIMATE_COLUMNS}


# ---------------------------------------------------------------------
# 1. Temporal alignment
# ---------------------------------------------------------------------

def resolve_interval(row: pd.Series) -> tuple:
    """Return (start, end) for one expedition.

    Start is the base-camp date. End is the termination date; where that
    is missing the summit date substitutes, and where both are missing
    the interval closes at base camp plus the recorded duration.
    """
    start = row["bcdate"]
    end = row["termdate"]

    if pd.isna(end) and not pd.isna(row.get("smtdate")):
        end = row["smtdate"]

    if pd.isna(end) and not pd.isna(start) and not pd.isna(row.get("totdays")):
        end = start + timedelta(days=int(row["totdays"]))

    return start, end


def add_expedition_intervals(expeditions: pd.DataFrame) -> pd.DataFrame:
    """Attach start_date and end_date columns."""
    frame = expeditions.copy()
    bounds = frame.apply(resolve_interval, axis=1, result_type="expand")
    frame["start_date"] = bounds[0]
    frame["end_date"] = bounds[1]

    unresolved = frame["end_date"].isna().mean() * 100
    LOGGER.info("interval unresolved for %.2f%% of records", unresolved)
    return frame


def filter_to_peaks_with_weather(
    expeditions: pd.DataFrame, weather_by_peak: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Keep only expeditions on peaks the weather archive covers."""
    codes = set(weather_by_peak.keys())
    kept = expeditions[expeditions["peakid"].isin(codes)].copy()
    LOGGER.info(
        "peak intersection retains %d of %d expeditions across %d peaks",
        len(kept), len(expeditions), len(codes & set(expeditions["peakid"])),
    )
    return kept


# ---------------------------------------------------------------------
# 2. Climate descriptors
# ---------------------------------------------------------------------

def summarise_weather(
    slice_df: Optional[pd.DataFrame],
    heavy_snow_mm: float = 10.0,
    extreme_wind_kmh: float = 70.0,
    rapid_warming_delta_c: float = 5.0,
    storm_weather_code_min: int = 70,
) -> Dict[str, float]:
    """Compress one expedition's weather slice into climate descriptors.

    Returns all-NaN when no usable interval could be reconstructed; those
    rows are flagged rather than deleted, since a missing interval is
    itself informative about how the expedition ended.
    """
    if slice_df is None or slice_df.empty:
        return dict(_EMPTY_CLIMATE)

    frame = slice_df.copy()
    temp_change = frame["temperature_2m"].diff()

    if "weather_code" in frame.columns:
        storm_days = int((frame["weather_code"] >= storm_weather_code_min).sum())
    else:
        storm_days = np.nan

    return {
        "temp_mean": frame["temperature_2m"].mean(),
        "temp_max": frame["temperature_2m"].max(),
        "temp_min": frame["temperature_2m"].min(),
        "temp_std": frame["temperature_2m"].std(),
        "snow_total": frame["snowfall"].sum(),
        "snow_max": frame["snowfall"].max(),
        "heavy_snow_days": int((frame["snowfall"] > heavy_snow_mm).sum()),
        "wind_mean": frame["wind_speed_100m"].mean(),
        "wind_max": frame["wind_speed_100m"].max(),
        "extreme_wind_days": int((frame["wind_speed_100m"] > extreme_wind_kmh).sum()),
        "pressure_mean": frame["surface_pressure"].mean(),
        "pressure_min": frame["surface_pressure"].min(),
        "rapid_warming_days": int((temp_change > rapid_warming_delta_c).sum()),
        "storm_days": storm_days,
    }


def build_climate_features(
    expeditions: pd.DataFrame,
    weather_by_peak: Dict[str, pd.DataFrame],
    **thresholds,
) -> pd.DataFrame:
    """Slice weather per expedition and summarise, preserving row order."""
    summaries = [None] * len(expeditions)
    positions = {idx: pos for pos, idx in enumerate(expeditions.index)}

    for peak, group in expeditions.groupby("peakid"):
        weather = weather_by_peak.get(peak)

        if weather is None:
            for idx in group.index:
                summaries[positions[idx]] = dict(_EMPTY_CLIMATE)
            continue

        dates = weather["date"].to_numpy()

        for idx, row in group.iterrows():
            start, end = row["start_date"], row["end_date"]

            if pd.isna(start) or pd.isna(end):
                summaries[positions[idx]] = dict(_EMPTY_CLIMATE)
                continue

            mask = (dates >= start) & (dates <= end)
            window = weather.loc[mask]
            summaries[positions[idx]] = summarise_weather(
                window if not window.empty else None, **thresholds
            )

    climate = pd.DataFrame(summaries, index=range(len(expeditions)))
    missing_share = climate["temp_mean"].isna().mean() * 100
    LOGGER.info("no usable weather interval for %.2f%% of records", missing_share)
    return climate


# ---------------------------------------------------------------------
# 3. Vulnerability descriptors and target
# ---------------------------------------------------------------------

def _yes_flag(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: 1 if str(v).strip().upper() == "Y" else 0)


def add_vulnerability_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Party, route and exposure descriptors."""
    out = frame.copy()

    out["team_size"] = out["totmembers"]
    safe_team = out["team_size"].replace(0, np.nan)
    safe_days = out["totdays"].replace(0, np.nan)

    # summit_rate and death_rate are computed here for the audit trail in
    # docs/PAPER_MAPPING.md, then dropped before modelling: both encode
    # the outcome. See drop_outcome_fields.
    out["summit_rate"] = out["smtmembers"] / safe_team
    out["death_rate"] = out["mdeaths"] / safe_team

    out["oxygen_used"] = _yes_flag(out["o2used"])
    out["oxygen_free_ascent"] = _yes_flag(out["o2none"])
    out["exposure_days"] = out["totdays"]
    out["elevation_risk"] = out["highpoint"]
    out["camp_density"] = out["camps"] / safe_days
    out["technical_difficulty"] = out["rope"] / safe_days
    out["is_standard_route"] = _yes_flag(out["stdrte"])
    out["is_commercial_route"] = _yes_flag(out["comrte"])

    season_dummies = pd.get_dummies(out["season"], prefix="season", dummy_na=False)
    out = pd.concat([out, season_dummies], axis=1)

    return out


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Binary target: one when a member death is recorded."""
    out = frame.copy()
    out["fatality"] = (out["mdeaths"] > 0).astype(int)
    out["severe_outcome"] = (
        out["death_rate"].notna() & (out["death_rate"] > 0.20)
    ).astype(int)

    positives = int(out["fatality"].sum())
    rate = positives / len(out) * 100
    ratio = (len(out) - positives) / max(positives, 1)
    LOGGER.info(
        "target fires on %d of %d expeditions (%.2f%%, imbalance %.1f:1)",
        positives, len(out), rate, ratio,
    )
    return out
