# Data

Two public archives. Neither is redistributed in this repository — you must
fetch them yourself, and the terms are the publishers', not ours.

---

## Sources

### 1. Himalayan expeditions

A public mirror of the Himalayan Database.

- <https://www.kaggle.com/datasets/siddharth0935/himalayan-expeditions>
- 11,425 expeditions recorded between 1905 and 2024.
- Files used: `exped.csv` (the modelling table), `peaks.csv` (peak identifiers).
  `members.csv` is downloaded but not used by the pipeline.

Fields relied on include peak identifiers, party size, camp and rope counts,
oxygen usage flags, route class, the three date columns (`bcdate`, `smtdate`,
`termdate`), and `mdeaths`, from which the target is derived.

### 2. Historic weather for Himalayan peaks

- <https://www.kaggle.com/datasets/bonesclarke26/historic-weather-data-for-himalayan-peaks>
- One parquet file per peak, named by peak identifier, e.g. `EVER.parquet`.
- **Hourly** series, not daily: 394,488 records per peak, covering
  1980-01-01 00:00 to 2024-12-31 23:00 with complete 24-hour days throughout.
- Columns used: `date`, `temperature_2m` (°C), `snowfall` (**cm**),
  `wind_speed_100m` (**km/h**), `surface_pressure` (hPa), `weather_code` (WMO).
  The archive also carries humidity, precipitation, rain, wind direction,
  evapotranspiration, vapour-pressure deficit, snow depth and radiation, none of
  which the pipeline uses.

Weather exists for **27 peaks**. Intersecting peak identifiers with the
expedition archive retains 9,457 expeditions.

The underlying source is the Open-Meteo Historical Weather API, released under
CC BY 4.0. These are model-based (reanalysis) estimates at each peak's grid
location, not on-mountain station observations — attribution is required when
redistributing anything derived from them.

---

## Automatic download

```bash
python scripts/00_download_data.py
```

Needs Kaggle credentials — either `~/.kaggle/kaggle.json` or the environment
variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Get a key from your Kaggle account page, under *API → Create New Token*.

---

## Manual download

If you would rather not install `kagglehub`, download both archives from the
links above and unpack them into this layout:

```
data/raw/
├── himalayan-expeditions/
│   ├── exped.csv
│   ├── peaks.csv
│   └── members.csv
└── historic-weather-data-for-himalayan-peaks/
    ├── AMAD.parquet
    ├── ANN1.parquet
    ├── ...
    └── (one parquet per peak, 27 in all)
```

Those are the paths already set in `config/default.yaml`. Anywhere else works
too — just point the config at it:

```bash
python scripts/run_all.py \
  --set paths.expeditions_dir=/somewhere/exped \
  --set paths.weather_dir=/somewhere/weather
```

---

## Temporal alignment

Getting the join right matters more than it might look.

**Interval resolution.** Each expedition's interval runs from the base-camp date
to the termination date. Where the termination date is missing the summit date
substitutes; where both are missing, the interval closes at the base-camp date
plus the recorded duration. No substitute is applied for a missing base-camp
date.

**What the 15.4% actually is.** 1,452 of the 9,457 records get no weather slice:

| Cause | Records |
|---|---|
| No base-camp date, so no interval start | 1,186 |
| No resolvable end date | 27 |
| Interval ends before the weather archive begins in 1980 | 239 |

Those rows are **flagged, not deleted** — a missing slice usually means an
undated, aborted or pre-1980 expedition, which is itself informative. Records
without a slice have a fatality rate of 7.7%, against 4.9% for the rest, which is
why the missingness indicator survives the redundancy screen and ranks in the
paper's importance plot.

**Time zones.** The `date` column is timezone-aware at UTC. The pipeline slices
by **UTC calendar date** and does not convert to Nepal time (UTC+05:45), which is
what the published run did. The effect is negligible at these interval lengths:
re-slicing on local dates correlates above 0.999 with the published features for
mean temperature, total snowfall and peak wind.

**Ordering.** Weather frames are sorted by date on load
(`data.sort_weather_by_date`). The rapid-warming counter differences consecutive
rows, so without sorting that feature would depend on file ordering rather than
on the weather. The published parquet files happen to arrive sorted already.

---

## What the counters count

Because the series is hourly, the four event counters tally **hours**, not days,
despite their `_days` names — the names are kept to match the published feature
lists. Two of them never fire on this data:

| Counter | Threshold | Slices with a non-zero value |
|---|---|---|
| `heavy_snow_days` | snowfall > 10 cm in one hour | 0 — the largest hourly value inside any slice is 7.2 cm |
| `extreme_wind_days` | 100-m wind > 70 km/h | 0 — the largest is 47.2 km/h |
| `rapid_warming_days` | temperature rise > 5 °C over the previous hour | 726 |
| `storm_days` | WMO weather code ≥ 70 (snowfall codes) | 7,673, but the column is dropped by the redundancy screen |

The first two are therefore constant-zero columns that survive screening (a
constant column has undefined correlation) and appear in some selected subsets.
They carry no information. See the *Known properties of the published run*
section of the top-level README.

---

## Derived target

Binary, set to one when a member death is recorded. It fires on 504 of 9,457
expeditions — a positive rate of 5.33%, an imbalance ratio near 17.8:1.

---

## Synthetic stand-in

For testing the code path without credentials:

```bash
python scripts/make_synthetic_data.py --out data/synthetic
```

This writes files matching the real schema, with a weak planted signal and a
`termreason` field deliberately correlated with the outcome so the leakage
ablation has something to detect.

**The schema is real. The results are not.** Do not cite anything produced from
synthetic data.
