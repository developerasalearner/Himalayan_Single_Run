# Notebooks

Two revisions are archived here, **unmodified, with their original cell outputs
intact**. They are the record of what was actually run, and are deliberately not
cleaned up. Neither is the reproduction path — use `scripts/run_all.py` for that.

---

## `Himalayan_Expeditions_Opti_original.ipynb`

The run behind the manuscript's main results. `termreason` is excluded.

Its cell outputs are the source for every frozen subset in
`artifacts/selected_features.json`, and for the verification table in
`docs/PAPER_MAPPING.md`.

## `Himalayan_Expeditions_Opti_leakage_ablation.ipynb`

An earlier revision, identical except that `termreason` is **not** dropped. This
is the run behind Section V-E.

The two differ in exactly two places:

```diff
- expedition_path = "/kaggle/input/himalayan-expeditions"
+ expedition_path = "/root/.cache/kagglehub/datasets/.../versions/1"

- 'smtmembers','success1','success2','success3','success4'
+ 'smtmembers','success1','success2','success3','success4', 'termreason'
```

Everything else — every optimiser, every threshold rule, every model — is
identical.

### Why the ablation is not simply "one more column"

Readmitting `termreason` enlarges the screened column set from 47 to 48, so the
feature search ran again and settled on **35** features rather than 34, with a
partly different composition. Section V-E's figures come from that subset, not
from the main subset with one column appended.

`scripts/06_leakage_ablation.py` reproduces this by loading
`artifacts/selected_features_ablation.json` for the readmitted arm. It falls back
to appending the column if that file is missing, and reports which it used.

The manuscript describes the ablation as readmitting the field "with no other
change". That is true of the exclusion rule and of every modelling choice, but
the selection stage did re-run. See `docs/PAPER_MAPPING.md` § *Known
discrepancies*.

---

## Differences from the packaged pipeline

| Notebook | Packaged pipeline | Why it matters |
|---|---|---|
| Paths hard-coded to a local kagglehub cache | paths in `config/default.yaml` | the notebook only runs on the machine that produced it |
| `device="cuda"` hard-coded in six places | `compute.device: auto`, with fallback | the notebook fails or warns on CPU-only machines |
| `random` and `numpy.random` never seeded | `seeds.set_seeds()` at every entry point | the search was non-deterministic, so every downstream table moved between runs |
| Leakage arm is a separate notebook file | one config flag, `leakage.exclude_termreason` | both arms from one codebase, no copy-paste drift |
| `_missing` columns dropped after `df_encoded` was copied | dead path removed | the drop never took effect; the indicators were always in the model |
| `find_best_threshold_precise` redefined three times with different bodies | two clearly named protocols in `modelling.py` | the redefinitions are what make Tables III and IV look inconsistent |
| Results printed to cell output | written to `results/*.csv` and JSON | nothing to transcribe by hand |
| No run manifest | `results/run_manifest.json` | seed, device, git commit and versions recorded per run |

The threshold point deserves emphasis. The notebook defines a function of the
same name three separate times: once maximising F1 over `[0.10, 0.30]`, twice
maximising `0.6·recall + 0.4·precision` over `[0.10, 0.20]`. Whichever cell ran
most recently determined what a later cell computed. In the package these are
`modelling.best_threshold_f1` and `modelling.best_threshold_weighted`, and each
table names the one it uses.
