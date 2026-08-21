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
Coverage spans nine peaks, and about 15% of records have no reconstructable
interval.

---

## Quick start

```bash
git clone <your-repo-url> && cd himalayan-expedition-risk

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
raises ROC-AUC and average precision substantially, and `termreason` then ranks
first by gain, well clear of every climate and party descriptor. **Those are not
better results.** The classifier has simply been handed partial access to its
own label. The gap is reported precisely because a shift of that size is easily
mistaken for a methodological advance.

A related caveat is on record in the paper's limitations: fields such as
`claimed`, `disputed` and `primref` are archival rather than operational — they
record how the database later catalogued an expedition — and `claimed` tops the
importance ranking. None encodes a fatality, so the exclusion rule stands, but
an operational model should drop them and be re-scored.

---

## Reproducibility

The metaheuristic search is stochastic. Two things make results repeatable:

1. **Global seeding.** `seeds.set_seeds()` seeds `random` and `numpy.random`,
   not just the estimators. The original notebook left those streams unseeded.
2. **Frozen subsets.** Stage 3 writes the selected feature sets to
   `artifacts/selected_features.json`, and later runs reuse that file, so
   stages 4–6 reproduce exactly regardless of search variance. Force a fresh
   search with `--research`.

The subsets shipped in `artifacts/selected_features.json` are the ones from the
manuscript run, read back out of the archived notebook's cell outputs — so a
clean clone reproduces Tables II, III and IV exactly rather than re-running the
search. `artifacts/selected_features_ablation.json` holds the separate
35-feature subset behind Section V-E.

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
notebook the study grew out of, archived unmodified. It is kept for provenance
and is **not** the reproduction path — it carries Kaggle-absolute paths, a
hard-coded CUDA device, unseeded random streams, and does not apply the
`termreason` exclusion. See `notebooks/README.md` for the full list of
differences.

---

## Data

Two public archives, neither redistributed here. See
[`docs/DATA.md`](docs/DATA.md) for sources, licences and the expected directory
layout.

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
