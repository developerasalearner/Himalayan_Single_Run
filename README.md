# Weather-Aware Imbalanced Risk Prediction on Himalayan Expeditions

Reproduction code for *Hybrid Metaheuristic Feature Selection and Probabilistic
Calibration for Weather-Aware Imbalanced Risk Prediction: A Case Study on
Himalayan Expeditions*.

Fatal outcomes on Himalayan climbs are rare — a member death is recorded in
roughly one expedition in nineteen in the archive studied here. That rarity is
what makes the task awkward: a classifier trained on the raw distribution
minimises overall error by neglecting the minority class, reporting accuracy
above 0.94 while detecting almost nothing.

This repository contains the full retrospective, expedition-level pipeline:
weather-aware feature construction, a sequential FOA→GWO feature search, a
comparison of six imbalance treatments, post-hoc probability calibration, and a
leakage ablation quantifying what the exclusion rule costs.

---

## What this study is, and is not

**It is retrospective.** Weather statistics are aggregated over the recorded
expedition interval, from the base-camp date to the termination date. That
endpoint is unknown while a climb is under way, so these features describe
environmental exposure after the fact. Nothing here is a real-time forecast or a
deployable warning system.

**Performance is modest and reported honestly.** Average precision of roughly
0.155 against a no-skill floor of 0.053 suits triage, not decision authority.
Coverage spans 27 peaks, and 1,452 of the 9,457 records (15.4%) have no usable
weather slice: 1,186 lack a base-camp date, 27 have no resolvable end date, and
239 ended before the weather archive begins in 1980.

**Operating points come from the test partition.** Every threshold in the
paper — the per-treatment `τ*` of Table III, the thresholds of Table IV, and the
final `τ = 0.110` — was selected against test labels, so thresholded precision,
recall and F1 are optimistic. ROC-AUC and average precision are threshold-free
and unaffected. The pipeline reproduces that protocol as published; see
[Known properties of the published run](#known-properties-of-the-published-run).

---

## Quick start

```bash
git clone https://github.com/developerasalearner/Himalayan_Single_Run.git && cd Himalayan_Single_Run
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_download_data.py      # needs Kaggle credentials
python scripts/run_all.py
```

Results land in `results/`, figures in `figures/`, and a manifest recording the
seed, device, git commit and every package version in
`results/run_manifest.json`.

### No Kaggle credentials yet?

You can verify the code runs end to end against a synthetic stand-in first. The
schema is real; the signal is not. **Never cite numbers produced this way.**

```bash
make smoke
```

---

## Pipeline stages

| Stage | Script | Produces |
|---|---|---|
| 0 | `00_download_data.py` | `data/raw/` |
| 1 | `01_build_dataset.py` | modelling table, split summary, screened feature list |
| 2 | `02_baselines.py` | Table I — untuned classifiers |
| 3 | `03_feature_selection.py` | Table II — selection strategies; frozen subsets |
| 4 | `04_imbalance.py` | Table III — imbalance treatments |
| 5 | `05_calibration_final.py` | Table IV, final metrics, Figures 2–4 |
| 6 | `06_leakage_ablation.py` | the leakage ablation |

Run any stage on its own, or all of them with `scripts/run_all.py`. Full
table-and-figure mapping is in [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md).

Stage 3 runs an embedded reference (ℓ1 logistic screen plus greedy backward
elimination) and six population searches on the same 47 screened columns: a
genetic algorithm, binary PSO, Grey Wolf Optimization, the Energy Valley
Optimizer, the Forest Optimization Algorithm, and the sequential FOA→GWO hybrid.
The embedded subset is a comparison point, not a pre-filter for the searches.

---

## Data

Two public archives, neither redistributed here. See
[`docs/DATA.md`](docs/DATA.md) for sources, licences and the expected directory
layout.

The expedition archive lists 11,425 expeditions recorded between 1905 and 2024;
intersecting peak identifiers with the weather archive retains 9,457. The
weather archive holds **hourly** model-based series from the Open-Meteo
Historical Weather API (CC BY 4.0) for 27 peaks, complete from 1980 to 2024 and
time-stamped in UTC. Snowfall is in cm, 100-m wind speed in km/h. Slices are
taken by UTC calendar date; re-slicing on Nepal local dates (UTC+5:45) leaves
the features essentially unchanged (correlation above 0.999 for mean
temperature, total snowfall and peak wind).

---

## Configuration

Every number that can change a reported result lives in
[`config/default.yaml`](config/default.yaml). Override any key from the command
line:

```bash
python scripts/run_all.py --set seed=7
python scripts/run_all.py --set compute.device=cpu
python scripts/run_all.py --set final.threshold=0.12
```

---

## The exclusion rule

Expedition archives are compiled retrospectively, so several fields describe how
a trip concluded. Those correlate strongly with the outcome, and admitting them
yields results that look impressive while telling you little.

Excluded before any modelling: member and hired-staff death counts, the derived
death rate, the severe-outcome flag, and summit-member and success counts.

`termreason` is handled separately, because it is the subtle one. It records
*why* a trip ended, and one of its categories covers accidents involving death
or serious injury — so it partially encodes the label without being an outcome
count. It is excluded by default:

```yaml
leakage:
  exclude_termreason: true     # main results
```

Setting it to `false` readmits the field and reproduces the ablation. Doing so
moves ROC-AUC from 0.696 to 0.786 and average precision from 0.155 to 0.339, and
`termreason` then ranks first by gain, well clear of every climate and party
descriptor. **Those are not better results.** The classifier has simply been
handed partial access to its own label. The gap is reported precisely because a
shift of that size is easily mistaken for a methodological advance.

A related caveat is on record in the paper's limitations: fields such as
`claimed`, `disputed` and `primref` are archival rather than operational — they
record how the database later catalogued an expedition — and `claimed` tops the
importance ranking. None encodes a fatality, so the exclusion rule stands, but
an operational model should drop them and be re-scored.

---

## Known properties of the published run

The pipeline reproduces the run reported in the paper, which means it also
reproduces its quirks. These are documented rather than silently fixed, so the
code and the manuscript agree.

- **Test-partition thresholds.** All operating points are selected against test
  labels (see above). A validation-based protocol would be the correct fix and is
  listed as future work in the paper.
- **Six constant columns among the 47.** `heavy_snow_days` (> 10 cm in an hour)
  and `extreme_wind_days` (> 70 km/h) are never triggered — the largest values
  inside any slice are 7.2 cm and 47.2 km/h — and four derived indicators
  (`oxygen_used`, `oxygen_free_ascent`, `is_standard_route`,
  `is_commercial_route`) compare Boolean archive flags against the string `"Y"`
  and are therefore always zero. The raw flags they duplicate are retained and do
  carry the information. Reported subset sizes include these dead columns: the
  34-feature hybrid subset holds 30 informative ones.
- **Double class weighting in Stage 4.** The positive weight
  `w = N_maj / N_min ≈ 17.75` is computed from the original training
  distribution and applied unchanged after resampling, so every resampled
  treatment in Table III corrects the same skew twice. Stage 5 recomputes the
  weight after resampling instead (0.82 for SMOTE-ENN, 1.00 for SMOTE-Tomek), so
  Table III and Table IV rows are not the same models.
- **Unequal search budgets.** Each strategy uses a population of 20 over 30
  iterations, but FOA evaluates 1,220 masks and the FOA→GWO hybrid 1,820, against
  620 for GA, PSO, GWO and EVO.
- **Simplified GWO baseline.** The standalone GWO samples each bit from a sigmoid
  of the mean leader position, without the `a` schedule, and returns the best mask
  of its final iteration. The hybrid's GWO phase uses the canonical update.

---

## Reproducibility

The metaheuristic search is stochastic. Two things make results repeatable:

1. **Global seeding.** `seeds.set_seeds()` seeds `random` and `numpy.random`,
   not just the estimators. The original notebook left those streams unseeded, so
   the published subsets cannot be regenerated from the code — which is why they
   are frozen here.
2. **Frozen subsets.** Stage 3 writes the selected feature sets to
   `artifacts/selected_features.json`, and later runs reuse that file, so stages
   4–6 do not depend on search variance. Force a fresh search with `--research`.

The subsets shipped in `artifacts/selected_features.json` are the ones from the
manuscript run, read back out of the archived notebook's cell outputs.
`artifacts/selected_features_ablation.json` holds the separate 35-feature subset
behind Section V-E.

**On exact reproduction.** The manuscript's models were trained with
`device="cuda"`. Only the CPU-trained rows reproduce bit-for-bit from a clean
clone: Table I and the embedded row of Table II. Re-running the remaining rows on
CPU with the same code and the same frozen subsets shifts the numbers — the final
model gives ROC-AUC 0.706 and average precision 0.147 instead of 0.696 and 0.155,
and the hybrid's Table II F1 becomes 0.2415 instead of 0.1945. Set
`compute.device=cuda` to reproduce the published values, and treat the CPU gap as
further evidence that the small margins between methods in Table II are not
stable. The dataset build is fully deterministic: the record counts, the 70 → 47
redundancy screen and the dropped-column list reproduce exactly on any device.

Details on determinism, hardware, version sensitivity and expected tolerances
are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Repository layout

```
config/default.yaml            every tunable parameter
src/himalayan_risk/            the library
  ├── data.py                  archive loading
  ├── features.py              temporal alignment, climate, vulnerability
  ├── preprocess.py            exclusion rule, imputation, encoding, screening
  ├── modelling.py             classifier factory, fitness, threshold protocols
  ├── selection/               embedded stage + five optimisers + hybrid
  ├── imbalance.py             the six treatments
  ├── evaluate.py              Tables I, II, IV and final metrics
  ├── figures.py               Figures 2–4
  └── pipeline.py              end-to-end assembly
scripts/                       one script per stage, plus run_all
notebooks/                     the original exploratory notebook, archived
docs/                          data, reproducibility, paper mapping
tests/                         fast unit and smoke tests
```

---

## Original notebook

`notebooks/Himalayan_Expeditions_Opti_original.ipynb` is the exploratory
notebook the study grew out of, archived unmodified. It applies the `termreason`
exclusion and produced every number in the paper, but it is **not** the
reproduction path: it carries Kaggle-absolute paths, a hard-coded CUDA device and
unseeded random streams. The companion
`Himalayan_Expeditions_Opti_leakage_ablation.ipynb` is the same notebook with
`termreason` readmitted, behind Section V-E. See `notebooks/README.md` for the
full list of differences.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper rather than this
repository where the work itself is what you are referring to.

## Licence

Code is MIT-licensed (see [`LICENSE`](LICENSE)). The licence covers this
repository's code only — the source datasets carry their own terms.

## Acknowledgment

In accordance with IEEE policy on AI-generated content, the authors disclose
that Claude (Anthropic) assisted in drafting and editing the manuscript text and
its LaTeX formatting, and in refactoring this repository from the original
research notebook. All data processing, experiments, numerical results and
figures are the authors' own work, and the authors take full responsibility for
the content.
