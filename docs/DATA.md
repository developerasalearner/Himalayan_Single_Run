# Data

Two public archives. Neither is redistributed in this repository — you must
fetch them yourself, and the terms are the publishers', not ours.

---

## Sources

### 1. Himalayan expeditions

A public mirror of the Himalayan Database.

- <https://www.kaggle.com/datasets/siddharth0935/himalayan-expeditions>
- Around 11,425 expeditions recorded between 1905 and 2024.
- Files used: `exped.csv` (the modelling table), `peaks.csv` (peak identifiers).
  `members.csv` is downloaded but not used by the pipeline.

Fields relied on include peak identifiers, party size, camp and rope counts,
oxygen usage flags, route class, the three date columns (`bcdate`, `smtdate`,
`termdate`), and `mdeaths`, from which the target is derived.

### 2. Historic weather for Himalayan peaks

- <https://www.kaggle.com/datasets/bonesclarke26/historic-weather-data-for-himalayan-peaks>
- One parquet file per peak, named by peak identifier, e.g. `EVER.parquet`.
- Daily series: `date`, `temperature_2m`, `snowfall`, `wind_speed_100m`,
  `surface_pressure`, `weather_code`.

Weather exists for nine peaks. Intersecting peak identifiers with the expedition
archive retains roughly 9,457 expeditions.

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
    └── (one parquet per peak)
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
plus the recorded duration. For about 15.4% of records no usable interval can be
reconstructed. Those rows are **flagged, not deleted** — a missing interval
usually means an undated or aborted expedition, which is itself informative, so
the missingness indicator is retained as a feature.

**Time zones.** Timestamps are stored in UTC and converted before slicing. A
naive join displaces every interval by several hours.

**Ordering.** Weather frames are sorted by date on load
(`data.sort_weather_by_date`). The rapid-warming counter differences consecutive
rows, so without sorting that feature would depend on file ordering rather than
on the weather.

---

## Derived target

Binary, set to one when a member death is recorded. It fires on roughly 504 of
9,457 expeditions — a positive rate near 5.33%, an imbalance ratio around
17.8:1.

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
