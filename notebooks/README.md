# Reproducibility

What is deterministic here, what is not, and how close you should expect to get.

---

## The short version

| Stage | Deterministic? | Why |
|---|---|---|
| 1. Dataset build | Yes, on any device | pure transformation of the archives |
| 1. Train/test split | Yes | fixed seed, stratified |
| 2. Baseline classifiers | Yes | seeded estimators, CPU |
| 3. Embedded selection | Yes | seeded L1 and cross-validation, CPU |
| 3. Metaheuristic search | Yes **given a seed** | see below |
| 4–6. Everything downstream | Yes **given the frozen subsets and the same device** | see *Hardware and device* |

Reproducing the **published** numbers for stages 3–6 additionally requires
`compute.device=cuda`, because that is what the manuscript run used. CPU gives
close but not identical results; the gap is quantified below.

---

## Global seeding

The original notebook seeded scikit-learn and XGBoost but left Python's `random`
and NumPy's global stream unseeded. Every optimiser draws from those — initial
populations, crossover points, mutation flips, the sigmoid binarisation
threshold — so repeated runs produced different feature subsets, and therefore
different numbers in every table downstream. That is why the published subsets
are frozen in `artifacts/` rather than regenerated: they cannot be recovered by
re-running the search.

`seeds.set_seeds(seed)` seeds `PYTHONHASHSEED`, `random` and `numpy.random`, and
is called at the top of every entry point. With a fixed seed, a fixed device and a
fixed environment, a full run is reproducible bit for bit.

Change the seed to check robustness:

```bash
python scripts/run_all.py --research --set seed=7
```

If a conclusion only survives at seed 42, it is not a conclusion. The manuscript
is explicit that Table II comes from one unseeded run per strategy and therefore
establishes no ranking of the methods.

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

Stages 4, 5 and 6 load this file rather than re-running the search, so they do not
depend on search variance. Force a fresh search with `--research`, or set
`selection.reuse_frozen_features: false`.

### Provenance of the frozen subsets

`artifacts/selected_features.json` contains the subsets from the manuscript run.
They were not regenerated — they were read back out of the cell outputs of
`notebooks/Himalayan_Expeditions_Opti_original.ipynb`, so they are the same
subsets the published tables were computed from.

Sizes match Table II: Embedded 34, GA 17, PSO 20, GWO 29, EVO 2, FOA 18,
FOA–GWO 34. Six of the 47 screened columns are constant (see the top-level
README), so some subsets contain fewer informative features than their size
suggests: PSO 18, GWO 27, FOA 15, FOA–GWO 30.

`artifacts/selected_features_ablation.json` contains the 35-feature subset from
the leakage-ablation run. It is kept separate because readmitting `termreason`
enlarged the screened column set, so the search re-ran and returned a different
subset — the ablation is not the main subset with one column appended.

Both files are tracked in version control on purpose. They are what let a clean
clone skip the search entirely.

---

## Hardware and device

`compute.device` accepts `auto`, `cpu` or `cuda`. On `auto` the code probes
XGBoost for a usable GPU and falls back silently.

**The device changes the numbers.** An earlier version of this document claimed
tree ensembles this small predict identically on either device. That is wrong, and
it was measured: re-running the published pipeline on CPU with the same code and
the same frozen subsets gives

| Quantity | Published (CUDA) | CPU rerun |
|---|---|---|
| Final ROC-AUC | 0.696 | 0.706 |
| Final average precision | 0.155 | 0.147 |
| FOA–GWO F1, Table II | 0.1945 | 0.2415 |

GPU and CPU histogram construction bin features differently, which is enough to
move a model this weak on 151 test positives. Only the CPU-trained rows —
Table I and the embedded row of Table II — reproduce bit-for-bit on any machine.
The dataset build, the 70 → 47 redundancy screen, the dropped-column list and the
split are fully deterministic regardless of device.

Treat the CPU/GPU gap as further evidence for what the manuscript already says:
the margins between methods in Table II are smaller than the noise around them.

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
`selection/embedded.py` detects the version and uses the right spelling, so newer
releases run correctly, but the pin is what the paper used.

**`CalibratedClassifierCV` renamed `base_estimator` to `estimator`** across recent
versions. `evaluate.extract_feature_importance` handles both. Note that this
estimator is an ensemble: `cv=3` fits three models on two-thirds of the training
partition each and averages their calibrated scores, which is what Eq. 7 of the
revised manuscript describes. The importance plot is read from the first of the
three.

The version actually used is recorded in `results/run_manifest.json` on every
run, alongside the git commit, seed and device.

---

## Expected tolerances

Same versions, same seed, same device: exact match.

Across devices: see the table above — roughly ±0.01 on ROC-AUC and up to ±0.05 on
minority F1 for the weaker configurations.

Across library versions, or with a fresh feature search: expect minority F1 to
move by roughly ±0.02 and ROC-AUC by roughly ±0.01. That range is comparable to
the gap between several of the methods compared in Table II, which is itself worth
bearing in mind when reading that table — the paper claims only that the hybrid
performed best in this single run, not that it is optimal in general, and notes
that it also used the largest search budget (1,820 evaluations against 620 for
GA, PSO, GWO and EVO) while reaching a *lower* search fitness than FOA, GA and
PSO.

---

## Protocol choices that affect what the numbers mean

Reproducing a number is not the same as trusting it. Three protocol details of
the published run are reproduced here faithfully and documented rather than
silently fixed:

1. **Thresholds are selected on the test partition** — `τ*` in Table III, every
   `τ` in Table IV, and the final 0.110. Thresholded precision, recall and F1 are
   therefore optimistic; ROC-AUC and average precision are not affected.
2. **Stage 4 keeps the original class weight after resampling** (`w ≈ 17.75`), so
   every resampled treatment in Table III corrects the same skew twice. Stage 5
   recomputes it instead.
3. **Median imputation is fitted on all 9,457 records before the split.** The step
   uses no labels, but it is not confined to the training partition.

---

## Two failure modes the pipeline watches for

The composite objectives used here reward recall heavily, and can be satisfied by
classifiers that do nothing useful.

**EVO** (the Energy Valley Optimizer) converged onto a two-feature mask whose
classifier labels nearly every expedition fatal — recall 0.9669, precision at the
base rate, accuracy 0.0662. That mask also scored the **highest fitness of any
strategy**, 0.5409, which is roughly what an all-positive classifier earns under
Eq. 5. The objective's best region contains near-trivial solutions.

**Random undersampling** does the same from the other direction: discarding most
of the majority class leaves 706 training rows, and with positives still weighted
17.75-fold the result is recall of exactly 1.0 paired with precision equal to the
test-set positive rate.

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
