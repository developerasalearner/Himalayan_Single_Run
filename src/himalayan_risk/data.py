"""Loading the two source archives.

Expedition history comes from a public mirror of the Himalayan Database;
daily meteorological series come from a companion weather archive, one
parquet file per peak, named by peak identifier.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXPEDITIONS_SLUG = "siddharth0935/himalayan-expeditions"
WEATHER_SLUG = "bonesclarke26/historic-weather-data-for-himalayan-peaks"

DATE_COLUMNS = ["bcdate", "smtdate", "termdate"]


def download_from_kaggle(destination: Path) -> Dict[str, Path]:
    """Fetch both archives with kagglehub and copy them under `destination`.

    Requires Kaggle credentials (~/.kaggle/kaggle.json or the KAGGLE_USERNAME
    and KAGGLE_KEY environment variables). See docs/DATA.md for the manual
    alternative if you would rather download by hand.
    """
    import shutil

    import kagglehub

    destination.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Path] = {}

    for name, slug in (
        ("himalayan-expeditions", EXPEDITIONS_SLUG),
        ("historic-weather-data-for-himalayan-peaks", WEATHER_SLUG),
    ):
        LOGGER.info("downloading %s", slug)
        cached = Path(kagglehub.dataset_download(slug))
        target = destination / name
        if target.exists():
            LOGGER.info("%s already present, leaving it alone", target)
        else:
            shutil.copytree(cached, target)
        resolved[name] = target
        LOGGER.info("%s -> %s", slug, target)

    return resolved


def load_expeditions(expeditions_dir: Path) -> pd.DataFrame:
    """Read exped.csv and parse the three date columns to plain dates."""
    path = Path(expeditions_dir) / "exped.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/00_download_data.py` "
            "or see docs/DATA.md."
        )
    expeditions = pd.read_csv(path, low_memory=False)

    for column in DATE_COLUMNS:
        expeditions[column] = pd.to_datetime(
            expeditions[column], errors="coerce"
        ).dt.date

    LOGGER.info("loaded %d expedition records from %s", len(expeditions), path)
    return expeditions


def load_peaks(expeditions_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(expeditions_dir) / "peaks.csv", low_memory=False)


def load_weather(weather_dir: Path, sort_by_date: bool = True) -> Dict[str, pd.DataFrame]:
    """Read one parquet per peak, keyed by upper-case peak identifier.

    Sorting by date matters: the rapid-warming counter differences
    consecutive rows, so an unsorted frame would make that feature depend
    on file ordering rather than on the weather.
    """
    weather_dir = Path(weather_dir)
    files = sorted(glob.glob(str(weather_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no .parquet files under {weather_dir}. "
            "Run `python scripts/00_download_data.py` or see docs/DATA.md."
        )

    weather_by_peak: Dict[str, pd.DataFrame] = {}
    for file in files:
        peak_code = os.path.basename(file).replace(".parquet", "").upper()
        frame = pd.read_parquet(file)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        if sort_by_date:
            frame = frame.sort_values("date").reset_index(drop=True)
        weather_by_peak[peak_code] = frame

    LOGGER.info(
        "loaded weather for %d peaks: %s",
        len(weather_by_peak),
        ", ".join(sorted(weather_by_peak)),
    )
    return weather_by_peak
