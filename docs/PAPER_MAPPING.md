# Paper ↔ code mapping

Where each claim, table, figure and equation in the manuscript comes from.

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

## Equations

| Eq. | Where |
|---|---|
| 1 — missingness indicator | `preprocess.add_missingness_indicators_and_impute` |
| 2 — median imputation | `preprocess.add_missingness_indicators_and_impute` |
| 3 — Pearson redundancy screen | `preprocess.correlation_filter` |
| 4 — embedded-stage objective | `selection/embedded.py` — **see discrepancy 1 below** |
| 5 — mask fitness | `modelling.mask_fitness`, `modelling.composite_score` |
| 6 — F1-optimal threshold | `modelling.best_threshold_f1` |
| 7 — sigmoid calibration map | `modelling.make_calibrated` (scikit-learn's `CalibratedClassifierCV`) |
| 8 — weighted threshold objective | `modelling.best_threshold_weighted` |

## Algorithm 1

`selection/hybrid.py::FOA_GWO_hybrid_feature_selection`. The FOA phase supplies
local seeding; the GWO phase draws the population toward the α, β and δ leaders
with the control parameter `a` decaying linearly from 2 to 0.

## Section III-I — the final evaluated configuration

Fixed in `config/default.yaml` under `final:` and executed by
`evaluate.fit_final_model` / `evaluate.evaluate_final`: the FOA–GWO subset,
class-weighted XGBoost without resampling, sigmoid calibration by three-fold
cross-validation, threshold 0.110. Every final metric, the confusion matrix,
both curves, the reliability diagram and the importance ranking derive from that
one prediction vector — `results/final_predictions.csv`.

---

## Why Tables III and IV disagree

They are not inconsistent; they use different threshold-selection protocols, and
the manuscript says so in a footnote. Spelled out:

|  | Table III | Table IV |
|---|---|---|
| Grid | 41 points on [0.10, 0.30] | 41 points on [0.10, 0.20] |
| Objective | maximise minority F1 (Eq. 6) | maximise 0.6·recall + 0.4·precision (Eq. 8) |
| Calibration | held fixed — every row is calibrated | varied — that is the point of the table |
| Implementation | `modelling.best_threshold_f1` | `modelling.best_threshold_weighted` |

So Table III's baseline row and Table IV's calibrated baseline row describe the
**same model at different operating points** — 0.105 against 0.110. The rows
marked uncalibrated in Table IV have no counterpart in Table III at all, since
that table holds calibration fixed.


---

## Verification against the manuscript run

Every reported figure below was read back out of the archived notebook's cell
outputs and compared to the submitted manuscript. Unless flagged, they agree to
four decimal places.

### Dataset-level claims

| Claim in text | Notebook output | Status |
|---|---|---|
| 11,425 expeditions in the archive | 11,425 | agrees |
| 9,457 retained after peak intersection | 9,457 | agrees |
| 15.4% with no usable interval | 15.3537% | agrees (rounded) |
| 504 positives, 5.33%, ratio approx. 17.8:1 | 5.3294% | agrees |
| 70 to 47 columns after redundancy screen | 70 to 47 | agrees |
| L1 zeroes 7 of the 47 survivors | kept 40 of 47 | agrees |
| Embedded stage stabilises at 34 features | 34 | agrees |
| 6,619 train / 2,838 test, 353 / 151 positives | 2,838 test, 151 positive | agrees |

### Table I

All four rows agree exactly. Random forest recovers 3 of 151 and k-NN 2, as
stated in the text — confirmed by the confusion matrices in cell 61.

### Table II

All seven rows agree exactly, including every subset size: Embedded 34, GA 17,
PSO 20, GWO 29, EVO 2, FOA 18, FOA-GWO 34. EVO's degenerate mask is
`['traverse', 'primrte']`.

### Table III

All six rows agree exactly, including each row's selected threshold.
Undersampling leaves 706 training rows, as the text states.

### Table IV

| Row | Status |
|---|---|
| Baseline, uncalibrated | agrees |
| Baseline, calibrated | see discrepancy 4 |
| SMOTE-ENN, both | agrees |
| SMOTE-Tomek, calibrated | agrees |
| SMOTE-Tomek, uncalibrated | metrics agree; **threshold column wrong**, see discrepancy 5 |

### Final configuration

Accuracy 0.9281, precision 0.2696, recall 0.2053, F1 0.2331, confusion matrix
`[[2603, 84], [120, 31]]` — all agree. ROC-AUC 0.696 and average precision 0.155
appear only inside figure images in the notebook and could not be re-read
numerically; the pipeline recomputes both and writes them to
`results/final_summary.json`.

### Section V-E

The ablation run reports F1 0.3516 against the main run's 0.2331, a rise of
0.1185 — the text's 0.119. Confusion matrix `[[2538, 149], [87, 64]]`.

---

## Known discrepancies between manuscript and code

These are recorded rather than quietly patched. Both are worth resolving in the
manuscript before camera-ready.

### 1. Equation 4 states the embedded objective as AUC

Equation 4 in the paper writes the embedded stage as maximising AUC over feature
subsets. The implementation — both the original notebook and this refactor —
scores each candidate subset with the same recall-weighted composite used by the
metaheuristic stage (0.5·recall + 0.3·F1 + 0.2·precision), without the size
penalty, averaged over stratified cross-validation folds.

See `selection/embedded.py::score_subset`. **The equation should be corrected to
match the code, or the code changed to match the equation.** They currently
disagree.

### 2. The original notebook does not apply the `termreason` exclusion

The `drop_cols` list in the archived notebook removes `mdeaths`, `hdeaths`,
`death_rate`, `severe_outcome`, `smtmembers` and `success1`–`success4`, but not
`termreason`. Section III-A of the paper states that `termreason` is excluded,
and the headline figures (ROC-AUC ≈ 0.696, AP ≈ 0.155) are the leakage-free
ones.

Run as written, the notebook therefore reproduces the **ablation** row
(≈ 0.786 / 0.339), not the main results.

This repository resolves it by making exclusion the default and readmission a
configuration flag, so one codebase produces both — see
`preprocess.drop_outcome_fields` and `leakage.exclude_termreason`.

### 3. Dead code in the original notebook

Two cells drop the `_missing` indicator columns from `exped_clean`. They run
*after* `df_encoded` was copied from `exped_clean`, so they have no effect on the
modelling table — the indicators survive into the model. This is consistent with
`temp_mean_missing` appearing in the paper's importance ranking, and with the
paper's argument for retaining missingness indicators. The manuscript is right;
the notebook is merely misleading. The dead path is not carried over here.

---

## Claims in the text and where to check them

| Claim | Source |
|---|---|
| ~11,425 expeditions, 1905–2024 | `data.load_expeditions` log line |
| 9,457 after peak intersection | `features.filter_to_peaks_with_weather` log line |
| 15.4% with no usable interval | `features.build_climate_features` log line |
| 504 positives, 5.33%, ratio ≈ 17.8:1 | `features.add_target` log line |
| 70 → 47 columns after redundancy screen | `preprocess.correlation_filter` log line |
| 6,619 train / 2,838 test, 353 / 151 positives | `results/00_split_summary.csv` |
| No-skill baseline 0.053 | `results/final_summary.json`, `NoSkillBaseline` |
| `claimed` leads importance by ~3× | `results/final_feature_importance.csv` |
| Undersampling degenerate: recall 1.0, precision = base rate | `results/table3_imbalance_treatments.csv`, `Degenerate` column |

Every one of these is logged at `INFO` during a run, so a full log is an audit
trail for the numbers in the text.

### 4. Table IV's calibrated baseline uses a fixed threshold, not the search

The other five rows of Table IV report the threshold chosen by the search over
`[0.10, 0.20]`. For the calibrated baseline the search selects tau = 0.1025,
giving accuracy 0.9253, precision 0.2707, recall 0.2384, F1 0.2535.

The manuscript instead reports tau = 0.110 with accuracy 0.9281, precision
0.2696, recall 0.2053, F1 0.2331 — the final evaluated configuration of
Section III-I.

**This is disclosed**: the table footnote states that the calibrated baseline is
the final evaluated configuration at tau = 0.110, and the surrounding text quotes
the matching false-positive counts (661 down to 84). So the table is honest. It
is still worth being aware that one row of the table was fixed while its
neighbours were searched, since a careful reviewer will notice the tau column is
not produced by a single rule.

### 5. Table IV, SMOTE-Tomek uncalibrated row: the threshold column is wrong

The manuscript prints tau = 0.120 for this row. The notebook reports **0.110**.

All four metrics on that row (0.7290, 0.1079, 0.5629, 0.1810) agree exactly, so
only the threshold column is affected — most likely 0.120 was carried up from the
calibrated row directly beneath it.

**Fix in Table IV: change `0.120` to `0.110` on the SMOTE-Tomek uncalibrated
row.** Nothing else changes; no text depends on this value.

### 6. Section V-E: "with no other change" is slightly stronger than what happened

Readmitting `termreason` enlarged the screened column set from 47 to 48, so the
feature search ran again and returned 35 features rather than 34, with a partly
different composition. Every modelling choice was held fixed, and the exclusion
rule was the only thing deliberately altered — but the selected subset did change
as a downstream consequence.

Consider softening to something like: readmitting the field, with the selection
stage re-run on the enlarged column set and every other choice held fixed.
