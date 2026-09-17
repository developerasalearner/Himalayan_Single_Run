# Paper ↔ code mapping

Where each claim, table, figure and equation in the manuscript comes from.

This document tracks the **revised manuscript** (the version with the fitness and
evaluation-budget columns in Table II and the threshold-protocol statement in
Section III-I). The discrepancies recorded in earlier revisions of this file have
all been resolved in the text; they are listed at the end for the record.

---

## Tables

| Paper | Script | Output |
|---|---|---|
| Table I — untuned classifiers | `02_baselines.py` | `results/table1_untuned_classifiers.csv` |
| Table II — selection strategies | `03_feature_selection.py` | `results/table2_feature_selection.csv` |
| Table III — imbalance treatments | `04_imbalance.py` | `results/table3_imbalance_treatments.csv` |
| Table IV — effect of calibration | `05_calibration_final.py` | `results/table4_calibration_effect.csv` |
| §V-E — leakage ablation | `06_leakage_ablation.py` | `results/table5_leakage_ablation.csv` |

## Figures

| Paper | Function | Output |
|---|---|---|
| Fig. 1 — pipeline diagram | drawn by hand, not generated | — |
| Fig. 2 — ROC and PR curves | `figures.figure_curves` | `figures/fig2_roc_pr_curves.png` |
| Fig. 3 — confusion matrix and reliability | `figures.figure_confusion_and_reliability` | `figures/fig3_confusion_reliability.png` |
| Fig. 4 — threshold sensitivity and importance | `figures.figure_threshold_and_importance` | `figures/fig4_threshold_importance.png` |
| — (supplementary) | `figures.figure_probability_distribution` | `figures/figS1_probability_distribution.png` |

Panels are pre-combined into wide two-panel PNGs so they drop into a two-column
layout without further editing.

The importance panel of Fig. 4 is read from the **first of the three
calibration-fold models** (`calibrated_classifiers_[0].estimator`), using
XGBoost's normalised gain, as the figure caption states.

## Equations

| Eq. | Where |
|---|---|
| 1 — missingness indicator | `preprocess.add_missingness_indicators_and_impute` |
| 2 — median imputation | `preprocess.add_missingness_indicators_and_impute` |
| 3 — Pearson redundancy screen | `preprocess.correlation_filter` |
| 4 — embedded-stage objective Φ₀ | `selection/embedded.py::score_subset` |
| 5 — mask fitness Φ | `modelling.mask_fitness`, `modelling.composite_score` |
| 6 — F1-optimal threshold | `modelling.best_threshold_f1` |
| 7 — averaged sigmoid calibration map | `modelling.make_calibrated` (scikit-learn's `CalibratedClassifierCV`, `cv=3`) |
| 8 — weighted threshold objective | `modelling.best_threshold_weighted` |

Equation 7 is written in the revised manuscript as the mean of three fold-wise
sigmoid maps, which is what `CalibratedClassifierCV(cv=3)` computes: three
XGBoost models each trained on two-thirds of the training partition, each with a
sigmoid fitted on its held-out third, averaged at prediction time.

## Algorithm 1

`selection/hybrid.py::FOA_GWO_hybrid_feature_selection`. The FOA phase flips one
bit per mask and truncation keeps the fittest P of the pooled 2P; the GWO phase
draws the population toward the α, β and δ leaders with `a` decaying linearly
from 2 to 0 as `a = 2 - 2(t-1)/T`.

## Section III-I — the final evaluated configuration

Fixed in `config/default.yaml` under `final:` and executed by
`evaluate.fit_final_model` / `evaluate.evaluate_final`: the FOA–GWO subset,
class-weighted XGBoost without resampling, sigmoid calibration by three-fold
cross-validation, threshold 0.110. Every final metric, the confusion matrix, both
curves, the reliability diagram and the importance ranking derive from that one
prediction vector — `results/final_predictions.csv`.

**Threshold provenance.** All operating points in the paper — `τ*` in Table III,
every `τ` in Table IV, and the final 0.110 — are selected against **test-partition
labels**. Section III-I of the revised manuscript says so explicitly, and
`modelling.best_threshold_f1` / `best_threshold_weighted` are called with the test
split to reproduce that protocol. ROC-AUC and average precision are threshold-free
and unaffected. A validation-based protocol is listed as future work.

---

## Why Tables III and IV differ

They are not inconsistent; they use different protocols, and the manuscript states
both. Spelled out:

|  | Table III | Table IV |
|---|---|---|
| Grid | 41 points on [0.10, 0.30] | 41 points on [0.10, 0.20] |
| Objective | maximise minority F1 (Eq. 6) | maximise 0.6·recall + 0.4·precision (Eq. 8) |
| Calibration | held fixed — every row is calibrated | varied — that is the point of the table |
| Positive weight | `w ≈ 17.75` from the **original** training distribution, kept after resampling | recomputed **after** resampling (0.82 for SMOTE-ENN, 1.00 for SMOTE-Tomek) |
| Implementation | `modelling.best_threshold_f1` | `modelling.best_threshold_weighted` |

So Table III's baseline row and Table IV's calibrated baseline row are the same
model at two different operating points, 0.105 against 0.1025. The resampled rows
are **not** the same models across the two tables, because the weight differs. The
rows marked uncalibrated in Table IV have no counterpart in Table III at all,
since that table holds calibration fixed.

The retained weight in Table III means every resampled row corrects the same skew
twice. The revised manuscript states this in Section III-G and Section V-C, and
scopes the conclusion accordingly: weighting alone outperformed resampling
*combined with the same weighting*.

---

## Verification against the manuscript run

Every figure below was read back out of the archived notebook's cell outputs and
compared with the manuscript. Dataset-level claims were additionally re-derived
from the public archives. Unless flagged, they agree to four decimal places.

### Dataset-level claims

| Claim in text | Check | Status |
|---|---|---|
| 11,425 expeditions, 1905–2024 | rebuilt from `exped.csv` | agrees |
| 9,457 retained after peak intersection | rebuilt | agrees |
| 27 peaks with weather | 27 parquet files | agrees |
| 1,452 records (15.4%) with no slice = 1,186 + 27 + 239 | rebuilt | agrees |
| 504 positives, 5.33%, ratio ≈ 17.8:1 | rebuilt | agrees |
| 70 → 47 columns after redundancy screen | rebuilt, dropped-column list identical | agrees |
| Six constant columns among the 47 | rebuilt | agrees |
| L1 zeroes 7 of the 47 survivors | kept 40 of 47 | agrees |
| Embedded stage stabilises at 34 features | 34 | agrees |
| 6,619 train / 2,838 test, 353 / 151 positives | rebuilt | agrees |
| Positive rate differs by 0.012 percentage points | 0.01248 pp | agrees |
| Slice maxima 7.2 cm and 47.2 km/h | rebuilt | agrees |
| Local-time re-slicing correlates above 0.999 | rebuilt | agrees |
| Records without a slice are 7.7% fatal vs 4.9% | rebuilt | agrees |

### Table I

All four rows agree exactly, and reproduce exactly on a clean rebuild (these
models were trained on CPU). Random forest recovers 3 of 151 and k-NN 2, as the
text states — confirmed by the confusion matrices in cell 61.

### Table II

All seven rows agree exactly, including every subset size: Embedded 34, GA 17,
PSO 20, GWO 29, EVO 2, FOA 18, FOA–GWO 34. The parenthetical sizes in the revised
table exclude the six constant columns: PSO 18, GWO 27, FOA 15, FOA–GWO 30; the
Embedded and GA subsets contain none.

The Φ column holds each optimiser's best inner-validation fitness as logged
during the search — GA 0.4224, PSO 0.4124, GWO 0.3712, EVO 0.5409, FOA 0.4255,
FOA–GWO 0.3994. The Evals column counts fitness evaluations: 620 for GA, PSO, GWO
and EVO (20 initial + 30 × 20), 1,220 for FOA (20 + 30 × 40), 1,820 for the hybrid
(20 + 30 × 60).

The hybrid's false-alarm counts quoted in Section V-B follow from the table: 32
true positives at precision 0.1798 is 178 flagged, so 146 false alarms; GA 35 at
0.1584 is 186; FOA 36 at 0.1452 is 212.

EVO is the Energy Valley Optimizer (Azizi et al., *Sci. Rep.* 2023), cited as [14]
in the revised manuscript. Its degenerate mask is `['traverse', 'primrte']`.

### Table III

All six rows agree exactly, including each row's selected threshold.
Undersampling leaves 706 training rows, as the text states.

### Table IV

All six rows agree with the notebook and now follow one threshold rule (Eq. 8).
The calibrated baseline row is the Eq. 8 result — τ = 0.1025, accuracy 0.9253,
precision 0.2707, recall 0.2384, F1 0.2535 — with false positives falling from
661 to 97 and true positives from 76 to 36. Earlier revisions of the manuscript
reported the final configuration at τ = 0.110 in this row; that is no longer the
case, and Section III-I keeps the final configuration separate.

### Final configuration

Accuracy 0.9281, precision 0.2696, recall 0.2053, F1 0.2331, confusion matrix
`[[2603, 84], [120, 31]]` — all agree. ROC-AUC 0.696 and average precision 0.155
appear only inside figure images in the notebook and were read from those; the
pipeline recomputes both and writes them to `results/final_summary.json`.

### Section V-E

The ablation run reports F1 0.3516 against the main run's 0.2331, a rise of
0.1185 — the text's 0.119. ROC-AUC 0.786, average precision 0.339, 35 features,
48 screened columns, `termreason` first by gain. Confusion matrix
`[[2538, 149], [87, 64]]`.

---

## Device sensitivity

The manuscript's models were trained with `device="cuda"`. Only the CPU-trained
rows reproduce bit-for-bit on a clean clone: Table I and the embedded row of
Table II. Re-running the rest on CPU with the same code and the same frozen
subsets shifts the numbers — the final model gives ROC-AUC 0.706 and average
precision 0.147 rather than 0.696 and 0.155, and the hybrid's Table II F1 becomes
0.2415 rather than 0.1945. Set `compute.device=cuda` to reproduce the published
values. The dataset build, the redundancy screen and the split are deterministic
on any device.

---

## Resolved discrepancies

These were open against earlier revisions of the manuscript and are now fixed in
the text. Kept for the record.

1. **Equation 4 stated the embedded objective as AUC.** The implementation scores
   each candidate subset with the recall-weighted composite without the size
   penalty. The revised manuscript writes Eq. 4 as Φ₀ = 0.5R + 0.3F₁ + 0.2P,
   matching `selection/embedded.py::score_subset`.
2. **A claim that the archived notebook does not exclude `termreason`.** It does:
   `drop_cols` in `Himalayan_Expeditions_Opti_original.ipynb` includes
   `termreason`, and that notebook produced the paper's main figures. The separate
   `..._leakage_ablation.ipynb` is the arm that readmits it. This repository makes
   exclusion the default and readmission a configuration flag, so one codebase
   produces both.
3. **Dead code in the original notebook.** Two cells drop the `_missing` indicator
   columns from `exped_clean`, but they run *after* `df_encoded` was copied, so the
   indicators survive into the model. This matches `temp_mean_missing` appearing in
   the importance ranking. The dead path is not carried over here.
4. **Table IV's calibrated baseline used a fixed threshold.** Now reported at the
   Eq. 8 threshold like every other row; see the Table IV note above.
5. **Table IV's SMOTE-Tomek uncalibrated threshold column.** The notebook reports
   0.110 and an earlier revision printed 0.120. The revised manuscript prints
   0.1100.
6. **"With no other change" in Section V-E.** The revised text reads "with the
   selection stage re-run on the resulting 48 screened columns and every other
   choice fixed."

---

## Claims in the text and where to check them

| Claim | Source |
|---|---|
| 11,425 expeditions, 1905–2024 | `data.load_expeditions` log line |
| 9,457 after peak intersection | `features.filter_to_peaks_with_weather` log line |
| 15.4% with no slice, split three ways | `features.build_climate_features` log lines |
| 504 positives, 5.33%, ratio ≈ 17.8:1 | `features.add_target` log line |
| 70 → 47 columns after redundancy screen | `preprocess.correlation_filter` log line |
| Six constant columns among the 47 | `preprocess.correlation_filter` log line |
| 6,619 train / 2,838 test, 353 / 151 positives | `results/00_split_summary.csv` |
| No-skill baseline 0.053 | `results/final_summary.json`, `NoSkillBaseline` |
| `claimed` leads importance by ~3× | `results/final_feature_importance.csv` |
| Undersampling degenerate: recall 1.0, precision = base rate | `results/table3_imbalance_treatments.csv`, `Degenerate` column |

Every one of these is logged at `INFO` during a run, so a full log is an audit
trail for the numbers in the text.
