# LTSA MSTAT Project

This project compares a longitudinal transformer survival model (LTSA) with a matched single-MRI Cox baseline for Alzheimer's disease prognosis.

## Fixed-landmark comparison

The comparison uses the existing patient-level train, validation, and test split. For each landmark `L` (0, 12, 24, and 36 months by default):

- A patient is eligible only when their event or censoring time is strictly greater than `L`.
- Only valid MRIs acquired on or before `L` are included.
- The Cox baseline receives the latest eligible MRI.
- LTSA receives the complete chronological MRI prefix.
- Follow-up is modeled from the landmark: `outcome time - L`.

Before either model starts, every configured landmark is checked for tensor availability, non-empty splits, observed events, and comparable survival pairs. Both models then use the same frozen cohort records.

The shared MRI encoder converts each volume to axial, coronal, and sagittal center slices and applies identical ResNet-101/ImageNet preprocessing. BatchNorm running statistics remain frozen during fine-tuning.

## Training

### Cohort preparation and exploration

Run these notebooks in order using the project Python environment:

1. [`notebooks/data_splitting.ipynb`](notebooks/data_splitting.ipynb) validates the
   baseline-relative bins from `_core.core_image_set` and `_core.core_follow_up_buckets`,
   keeps patients with tensor files, and assigns whole patients to 70% train / 15%
   validation / 15% test groups. It balances retained-image counts within baseline
   diagnosis and censoring strata with seed 42. `WRITE_SPLITS=True` updates
   `_raw.train_test_split`; set it to `False` for a preview.
2. [`notebooks/data_viz.ipynb`](notebooks/data_viz.ipynb) reads those saved assignments,
   verifies source hashes, and renders cohort summaries, scheduled-bin coverage,
   actual timing offsets, diagnoses, demographics, MRI metadata, and Table 1.
   It does not write split assignments. Historical experiment diagnostics are optional
   and explicitly separate from the refreshed cohort.

The split notebook archives prior assignments, selected cohort rows, the schedule,
and provenance under `data/artifacts/data_splits/`; keep these local artifacts to
reproduce or restore an assignment. The visualization notebook requires its
`latest.json` manifest. After a source change, rerun splitting before visualization.
Tensor-file eligibility checks existence only; selected image IDs and tensor contents
must still agree before training. Visit codes represent scheduled bins, while actual
acquisition times continue to determine prediction-time eligibility.

To execute and save both notebooks from the repository root:

```powershell
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/data_splitting.ipynb
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/data_viz.ipynb
```

### Model commands

Install the workspace and verify the command-line entry points:

```powershell
uv sync
uv run python -m mstat_project.ml.ltsa --help
uv run python -m mstat_project.ml.compare --help
```

Train one LTSA landmark:

```powershell
uv run python -m mstat_project.ml.ltsa --landmark-months 12
```

Train matched baseline/LTSA pairs at all default landmarks and compute a 1,000-sample paired patient-bootstrap confidence interval:

```powershell
uv run python -m mstat_project.ml.compare
```

Useful smoke-test options are `--epochs 1 --patience 1 --no-pretrained`. Real training defaults to AdamW with learning rate `1e-4`, weight decay `1e-4`, gradient clipping at `5.0`, batch size 2, 50 epochs, patience 10, and seed 42.

LTSA uses six-month discrete hazards, censoring NLL with `beta=0.15`, and next-visit feature MSE weighted by `1.0`. Its final time bin is reserved for durations beyond the maximum training support. Checkpoints are selected by validation C-index, with validation loss as the tie-breaker. LTSA risk is negative restricted mean survival time; the baseline uses the exact full-risk-set Cox partial likelihood.

The legacy baseline can either continue a checkpoint with `--resume-from` or start an independent run
at epoch 1 with `--initial-weights-from`. The independent mode loads only model parameters, so it uses
the new run's configuration and a fresh optimizer. These modes are mutually exclusive.

## Results

Checkpoints are written below `data/artifacts/model_checkpoints`. New comparison runs populate:

- `_ml.survival_runs`: configuration, cohort/event counts, checkpoint selection, test C-index, and paired confidence intervals.
- `_ml.survival_epoch_metrics`: total, survival, and auxiliary losses; C-indices; learning rate; and checkpoint path.
- `_ml.survival_run_patients`: exact split, outcome, selected MRI, and chronological image history.
- `_ml.survival_test_predictions`: scalar risk, outcome, image history, and LTSA survival curve.

The original `mstat_project.ml.baseline` command and `_ml.baseline_*` tables remain available as legacy history.
`_ml.baseline_runs` records the initial-weight kind, source run ID, and checkpoint path for new and resumed
legacy runs.

## Verification

Run the focused ML/package checks with:

```powershell
uv run pytest tests/ltsa tests/mstat_project/test_baseline.py tests/mstat_project/test_landmarks.py tests/mstat_project/test_ltsa_training.py tests/mstat_project/test_comparison_e2e.py -q
uv run ruff check packages/ltsa/src src/mstat_project/ml
uv run ty check
```

The LTSA architecture follows [Harnessing the Power of Longitudinal Medical Imaging for Eye Disease Prognosis Using Transformer-based Sequence Modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC11329720/) and the [authors' implementation](https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis).

## Known unrelated test drift

The repository-wide suite still contains tests outside this comparison scope that target removed or renamed APIs. `test_diagnoses_csv.py` imports the absent `mstat_project.data_models` package, and six `test_to_nifti.py` cases reference older NIfTI helper names. These failures are separate from the focused ML/LTSA suite above.
