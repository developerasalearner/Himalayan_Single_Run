# Reproducibility

What is deterministic here, what is not, and how close you should expect to get.

---

## The short version

| Stage | Deterministic? | Why |
|---|---|---|
| 1. Dataset build | Yes | pure transformation of the archives |
| 1. Train/test split | Yes | fixed seed, stratified |
| 2. Baseline classifiers | Yes | seeded estimators |
| 3. Embedded selection | Yes | seeded L1 and cross-validation |
| 3. Metaheuristic search | Yes **given a seed** | see below |
| 4–6. Everything downstream | Yes **given the frozen subsets** | |

---

## Global seeding

The original notebook seeded scikit-learn and XGBoost but left Python's `random`
and NumPy's global stream unseeded. Every optimiser draws from those — initial
populations, crossover points, mutation flips, the sigmoid binarisation
threshold — so repeated runs produced different feature subsets, and therefore
different numbers in every table downstream.

`seeds.set_seeds(seed)` seeds `PYTHONHASHSEED`, `random` and `numpy.random`, and
is called at the top of every entry point. With a fixed seed and a fixed
environment, a full run is reproducible bit for bit.

Change the seed to check robustness:

```bash
python scripts/run_all.py --research --set seed=7
```

If a conclusion only survives at seed 42, it is not a conclusion.

---

## Frozen feature subsets

Stage 3 writes every selected subset to `artifacts/selected_features.json`:

```json
{
  "metadata": {
    "seed": 42,
    "termreason_excluded": true,
    "screened_feature_count": 47,
    "population_size": 20,
    "iterations": 30
  },
  "features": {
    "Embedded": ["..."],
    "GA": ["..."],
    "FOA_GWO": ["..."]
  }
}
```

Stages 4, 5 and 6 load this file rather than re-running the search, so they
reproduce exactly even if you are on different hardware or a slightly different
library version. Force a fresh search with `--research`, or set
`selection.reuse_frozen_features: false`.

### Provenance of the frozen subsets

`artifacts/selected_features.json` contains the subsets from the manuscript run.
They were not regenerated — they were read back out of the cell outputs of
`notebooks/Himalayan_Expeditions_Opti_original.ipynb`, so they are the same
subsets the published tables were computed from.

Sizes match Table II: Embedded 34, GA 17, PSO 20, GWO 29, EVO 2, FOA 18,
FOA-GWO 34.

`artifacts/selected_features_ablation.json` contains the 35-feature subset from
the leakage-ablation run. It is kept separate because readmitting `termreason`
enlarged the screened column set, so the search re-ran and returned a different
subset — the ablation is not the main subset with one column appended.

Both files are tracked in version control on purpose. They are what make a clean
clone reproduce the published numbers.

---

## Hardware and device

`compute.device` accepts `auto`, `cpu` or `cuda`. On `auto` the code probes
XGBoost for a usable GPU and falls back silently.

Tree ensembles this small produce identical predictions on either device; only
wall-clock time differs. The original notebook hard-coded `device="cuda"`, which
fails or warns on CPU-only machines — that is fixed here.

Rough timings for a full `run_all.py` on the real data:

| Stage | CPU (8 cores) | GPU |
|---|---|---|
| Dataset build | 2–5 min (first run; cached after) | same |
| Baselines | under 1 min | under 1 min |
| Feature search, all 6 optimisers | 40–90 min | 15–30 min |
| Imbalance + calibration + ablation | 5–15 min | 3–8 min |

The search dominates. If you only need the downstream tables, keep the frozen
subsets and skip `--research`.

---

## Library versions

`requirements.txt` pins the ranges the reported numbers come from. Two
version-sensitive spots are worth knowing about:

**scikit-learn 1.8 changed `LogisticRegression`.** The `penalty` argument was
deprecated in favour of `l1_ratio`, and passing `penalty="l1"` there silently
mixes in an L2 term — which changes which features survive the embedded stage.
`selection/embedded.py` detects the version and uses the right spelling, so
newer releases run correctly, but the pin is what the paper used.

**`CalibratedClassifierCV` renamed `base_estimator` to `estimator`** across
recent versions. `evaluate.extract_feature_importance` handles both.

The version actually used is recorded in `results/run_manifest.json` on every
run, alongside the git commit, seed and device.

---

## Expected tolerances

Reproducing on the same versions with the same seed: exact match.

Reproducing across library versions, or with a fresh feature search: expect
minority F1 to move by roughly ±0.02 and ROC-AUC by roughly ±0.01. That range is
comparable to the gap between several of the methods compared in Table II, which
is itself worth bearing in mind when reading that table — the paper claims only
that the hybrid performed best in this comparison, not that it is optimal in
general.

---

## Two failure modes the pipeline watches for

The composite objectives used here reward recall heavily, and can be satisfied
by classifiers that do nothing useful.

**EVO** tends to converge onto a tiny mask whose classifier labels nearly every
expedition fatal — high recall, precision at the base rate, accuracy collapsed.

**Random undersampling** does the same from the other direction: discarding most
of the majority class leaves too little to learn from, and the result is recall
of exactly 1.0 paired with precision equal to the test-set positive rate.

`imbalance.flag_degenerate` marks such rows automatically in Table III. An
explicit precision floor in the fitness function would close the gap; that is
noted as future work rather than done here.

---

## Verifying an install

```bash
make test         # fast unit tests, no data needed
make smoke        # full pipeline on synthetic data, a couple of minutes
```

`make smoke` exercises every stage including the leakage ablation. It proves the
code path works. It proves nothing about the science.
