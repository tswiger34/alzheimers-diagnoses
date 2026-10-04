# LTSA MSTAT Project

**Scientific question: Does longitudinal MRI history improve AD prognosis beyond
the information contained in the most recent MRI?**

The current pipeline trains dynamic continuous-survival models using the binned
MRI inputs from `core_image_set`, the canonical schedule in
`core_follow_up_buckets`, and clinical outcomes in `core_survival_landmarks`.

| Component | Baseline A (`a`) | Baseline B (`b`) | MRI history (`history`) |
|---|---|---|---|
| MRI information | Latest available image | Latest available image | Available image prefix, including latest |
| Explicit acquisition time | No | Actual months since baseline | Actual months for each image |
| Temporal transformer | No | No | Yes, missing slots masked |
| Image encoder | Same ResNet-101 center-slice encoder | Same | Same |
| Survival head/objective | Continuous neural hazard / conditional NLL | Same | Same |
| Optimization | AdamW, validation-driven scheduler | Same | Same |
| Augmentation / step-ahead loss | Disabled | Disabled | Disabled |

The encoder, image projection, and survival head have identical initial weights
for the same seed. Each arm is then trained separately. The existing patient
splits, preprocessing, sampled training landmarks, validation weighting,
checkpoint selection, and evaluation patients are shared within a comparison run.

## Prediction setting

At prediction month `t`, require `T > t`, then select images whose **actual**
acquisition time `s <= t`. A missing bin stays missing: at month 60, a patient with
images at 0, 6, 12, 24 and 132 months contributes the month-24 image to each baseline
and the prefix through month 24 to the history model. An image at 60.2 months is
not available at month 60, even when it occupies the nominal month-60 bin.

The clinical endpoint `T` is the first AD examination date, or the last examination
with a valid diagnosis if no AD is observed, relative to baseline MRI. Months use
`days * 12 / 365` without binning or rounding the outcome. Training MRI rows must
satisfy `s < T`; same-day and post-outcome images are excluded. The new clinical
table retains exclusion reasons for auditing. It does not overwrite the older
image-derived outcomes in `core_image_set` or in tensor packages.

One eligible MRI landmark per training patient is sampled each epoch. Validation
uses all eligible MRI landmarks with weight `1 / n_i`, giving each patient equal
total influence. Each variant trains one model across follow-up, rather than a
separate model per nominal visit. At an acquisition landmark the likelihood is
`integral_s^T lambda(u|z) du - event * log(lambda(T|z))`. At a later prediction
landmark, survival is conditioned through `t`, integrating from `t`, not from the
older MRI date. See the [protocol and run guide](docs/continuous_survival.md).

## Cohort preparation and exploration

The existing notebooks remain the split/cohort exploration entry points:

1. [`notebooks/data_splitting.ipynb`](notebooks/data_splitting.ipynb) validates the
   binned image cohort, checks tensor-file presence, and saves patient assignments
   to `_raw.train_test_split`. It archives cohort and assignment snapshots under
   `data/artifacts/data_splits/`. `WRITE_SPLITS=False` previews without writing.
2. [`notebooks/data_viz.ipynb`](notebooks/data_viz.ipynb) verifies that snapshot and
   displays image-bin coverage and cohort summaries. Its historical experiment
   section describes the legacy runs.

The continuous training pipeline reads the existing assignments and never changes
them. Its clinical exclusions produce a smaller prognosis cohort with different
event counts from the image-cohort notebook. Use each run's `manifest.json` and
`training_landmarks.csv` for the actual training cohort. Do not regenerate splits
between model arms. Tensor labels from prior preprocessing are ignored; every
selected MRI is located by image ID and checked before training.

## Run from a PowerShell terminal

These commands use the existing project environment and load database/image paths
from `.env`. Ensure Postgres is available. The clinical dbt model must already be
built; [the run guide](docs/continuous_survival.md#prepare-the-database) includes
the rebuild command if needed.

First validate cohort and tensor availability without starting a model:

```powershell
.venv\Scripts\python.exe -m mstat_project.ml.baseline --preflight-only
```

Run the requested two-epoch checks for both pretrained single-image baselines:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:POLARS_MAX_THREADS = '2'
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.baseline --epochs 2 --patience 2 --device cuda --cpu-threads 2 --bootstrap-samples 0
```

Removing `CUDA_VISIBLE_DEVICES=-1` is necessary if the earlier CPU-only test
commands were run in this same terminal. `--device cpu` is supported as an
alternative. `--bootstrap-samples 0` skips confidence intervals for this short
pipeline check. Two epochs are not a final scientific experiment.

For a full matched comparison, run all three arms in one invocation:

```powershell
.venv\Scripts\python.exe -u -m mstat_project.ml.compare --variants a b history
```

`baseline` defaults to A/B, `ltsa` defaults to history, and `compare` defaults to all
three. `mstat_project.ml.survival` is the common implementation and also defaults to
A/B. All support `--help`. The defaults are 50 epochs, patience 10, batch size 2,
learning rate `1e-4`, weight decay `1e-4`, gradient clipping 5, seed 42, and no data
loader worker processes. The shared ReduceLROnPlateau scheduler halves the learning
rate after its patience of 3; checkpoints minimize patient-averaged validation
NLL. Change the batch size for all arms together if GPU memory requires it.

## Results and comparison

Each invocation creates a unique directory under
`data/artifacts/model_checkpoints/continuous/<run-id>/` and prints its path.

- `manifest.json`: configuration, cohort/exclusion counts, source/split/schedule
  hashes, code hashes and tensor preflight results.
- `source.parquet`, `schedule.csv`, `patient_splits.csv`, `training_landmarks.csv`:
  frozen inputs and the exact eligible MRI-landmark table.
- `a/`, `b/`, `history/`: selected `best.pt`, most recent `last.pt`, epoch metrics,
  sampled MRI IDs per epoch, summaries and patient predictions.
- `results.json`: metrics for each arm; `comparisons.json`: paired C-index
  differences and, when enabled, 95% patient-bootstrap intervals.
- `status.json`, `training.log`: progress/failure state and training log. GPU runs
  report PyTorch allocated/reserved/peak memory every 100 training batches.

Run status, configuration, summaries and comparisons are also stored in
`_ml.continuous_survival_runs`. Detailed patient predictions and epochs are local
artifacts. Keep the run directory with the database entry.

Evaluation uses all canonical schedule targets. At each target it reports
conditional survival at 12, 24 and 60 future months, NLL, and Harrell C-index with
outcomes truncated at the corresponding horizon. Missing comparable pairs yield
`null`, not zero. Comparisons verify identical patients, outcomes and available
image histories, then resample patients jointly **within each landmark**. They
report B-minus-A, history-minus-A, and history-minus-B as available. Pointwise
intervals do not adjust for testing many landmarks/horizons. This is a ranking
evaluation, not a calibration or IPCW analysis.

The history-minus-B comparison most directly addresses added imaging history
beyond explicit acquisition time. Historical Cox/discrete scores do not form a
matched comparison with these continuous models. Select the primary landmark and
horizon before interpreting held-out test results.

## Legacy experiments

Historical Python APIs, checkpoints and `_ml.baseline_*` / `_ml.survival_*` tables
remain available. Their CLI now requires explicit `--legacy`, including legacy
checkpoint resume or initial-weight options:

```powershell
.venv\Scripts\python.exe -m mstat_project.ml.baseline --legacy --help
.venv\Scripts\python.exe -m mstat_project.ml.ltsa --legacy --help
.venv\Scripts\python.exe -m mstat_project.ml.compare --legacy --help
```

The continuous command creates a new run each time. Its `last.pt` preserves model,
optimizer and scheduler state for inspection, but automatic crash resume is not
implemented. It does not load legacy checkpoints.

## Verification

Run these yourself from the repository root. The new tests use synthetic tensors
and do not require Postgres or pretrained model downloads:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:POLARS_MAX_THREADS = '2'
$env:CUDA_VISIBLE_DEVICES = '-1'
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_continuous_survival.py tests/mstat_project/test_survival_comparison.py -x -v
```

The regression command, optional lint command, expected artifacts, and manual
monitoring commands are in the [run guide](docs/continuous_survival.md).

Repository-wide tests have known legacy failures outside this pipeline, including
`test_diagnoses_csv.py` importing removed APIs and older NIfTI helper tests.

The history approach is inspired by [the LTSA paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11329720/)
and [authors' implementation](https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis).
The current continuous hazard and MRI cohort are adaptations, not a reproduction
of the paper's discrete objective.
