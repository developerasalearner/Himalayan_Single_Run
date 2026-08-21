#!/usr/bin/env python3
"""Generate a small synthetic stand-in for both archives.

This exists so anyone can verify the code runs end to end before dealing
with Kaggle credentials, and so continuous integration has something to
execute. The numbers it produces are meaningless - the schema is real,
the signal is not. Never cite results from synthetic data.

    python scripts/make_synthetic_data.py --out data/synthetic
    python scripts/run_all.py \\
        --set paths.expeditions_dir=data/synthetic/himalayan-expeditions \\
        --set paths.weather_dir=data/synthetic/weather \\
        --set selection.population_size=6 --set selection.iterations=3
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PEAKS = ["EVER", "AMAD", "CHOY", "MANA", "DHA1", "ANN1", "LHOT", "MAKA", "KANG"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
TERM_REASONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def make_weather(rng: np.random.Generator, start: date, days: int) -> pd.DataFrame:
    index = np.arange(days)
    return pd.DataFrame({
        "date": [start + timedelta(days=int(i)) for i in index],
        "temperature_2m": rng.normal(-8, 9, days),
        "snowfall": np.abs(rng.gamma(1.4, 4.0, days)),
        "wind_speed_100m": np.abs(rng.normal(38, 22, days)),
        "surface_pressure": rng.normal(415, 22, days),
        "weather_code": rng.integers(0, 100, days),
    })


def make_expeditions(rng: np.random.Generator, n: int) -> pd.DataFrame:
    base = date(1990, 1, 1)
    bc_dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 12000, n)]
    totdays = rng.integers(5, 60, n)

    members = rng.integers(1, 25, n)
    # Fatalities are driven weakly by team size and duration, so there is
    # some learnable signal but nothing resembling the real archive.
    hazard = 0.02 + 0.0015 * members + 0.0009 * totdays
    mdeaths = rng.binomial(1, np.clip(hazard, 0, 0.5), n) * rng.integers(1, 3, n)

    frame = pd.DataFrame({
        "expid": [f"SYN{i:05d}" for i in range(n)],
        "peakid": rng.choice(PEAKS, n),
        "primid": rng.integers(0, 5, n),
        "chksum": rng.integers(0, 999999, n),
        "bcdate": bc_dates,
        "smtdate": [d + timedelta(days=int(t // 2)) for d, t in zip(bc_dates, totdays)],
        "termdate": [d + timedelta(days=int(t)) for d, t in zip(bc_dates, totdays)],
        "totdays": totdays,
        "smtdays": rng.integers(0, 30, n),
        "smttime": rng.integers(0, 2400, n),
        "totmembers": members,
        "smtmembers": rng.integers(0, 10, n),
        "mdeaths": mdeaths,
        "hdeaths": rng.binomial(1, 0.02, n),
        "camps": rng.integers(0, 8, n),
        "rope": rng.integers(0, 4000, n),
        "highpoint": rng.integers(5000, 8850, n),
        "season": rng.choice(SEASONS, n),
        "o2used": rng.choice(["Y", "N"], n),
        "o2none": rng.choice(["Y", "N"], n),
        "stdrte": rng.choice(["Y", "N"], n),
        "comrte": rng.choice(["Y", "N"], n),
        "claimed": rng.choice(["Y", "N"], n),
        "disputed": rng.choice(["Y", "N"], n),
        "primref": rng.integers(0, 400, n),
        "ski": rng.choice(["Y", "N"], n, p=[0.02, 0.98]),
        "parapente": rng.choice(["Y", "N"], n, p=[0.02, 0.98]),
        "success1": rng.choice(["Y", "N"], n),
        "success2": rng.choice(["Y", "N"], n),
        "success3": rng.choice(["Y", "N"], n),
        "success4": rng.choice(["Y", "N"], n),
        "nation": rng.choice(["Nepal", "USA", "Japan", "UK"], n),
        "host": rng.integers(0, 3, n),
        "leaders": "synthetic",
        "sponsor": "synthetic",
        "countries": "synthetic",
        "approach": "synthetic",
        "agency": "synthetic",
        "campsites": "synthetic",
        "accidents": "synthetic",
        "achievment": "synthetic",
        "othersmts": "synthetic",
        "termnote": "synthetic",
        "route1": "synthetic", "route2": "synthetic",
        "route3": "synthetic", "route4": "synthetic",
        "ascent1": "synthetic", "ascent2": "synthetic",
        "ascent3": "synthetic", "ascent4": "synthetic",
    })

    # termreason correlates with the outcome, mirroring the real field:
    # one of its categories covers accidents involving death or injury.
    frame["termreason"] = np.where(
        frame["mdeaths"] > 0,
        rng.choice([9, 10], n),
        rng.choice(TERM_REASONS[:8], n),
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--n-expeditions", type=int, default=2000)
    parser.add_argument("--weather-days", type=int, default=13000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    exped_dir = out / "himalayan-expeditions"
    weather_dir = out / "weather"
    exped_dir.mkdir(parents=True, exist_ok=True)
    weather_dir.mkdir(parents=True, exist_ok=True)

    expeditions = make_expeditions(rng, args.n_expeditions)
    expeditions.to_csv(exped_dir / "exped.csv", index=False)

    pd.DataFrame({
        "peakid": PEAKS,
        "pkname": [f"Synthetic {p}" for p in PEAKS],
        "pkname2": ["" for _ in PEAKS],
    }).to_csv(exped_dir / "peaks.csv", index=False)

    pd.DataFrame({"expid": expeditions["expid"], "membid": range(len(expeditions))}).to_csv(
        exped_dir / "members.csv", index=False
    )

    for peak in PEAKS:
        make_weather(rng, date(1990, 1, 1), args.weather_days).to_parquet(
            weather_dir / f"{peak}.parquet", index=False
        )

    positives = int((expeditions["mdeaths"] > 0).sum())
    print(f"Wrote {len(expeditions)} synthetic expeditions "
          f"({positives} positive, {positives / len(expeditions):.2%}) to {out}")
    print("Reminder: synthetic data. Schema is real, results are not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
