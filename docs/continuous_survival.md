# Historical continuous MRI survival: version 1 protocol and run guide

This document describes the reference Tanh experiments. Its training commands and
defaults are historical. For the current stabilized implementation, test commands,
gates, cache workflow, and explicit held-out reporting, use [the version 2 guide](ltsa_v2.md).

The controlled question is whether prior MRI images improve AD prognosis beyond
the latest available MRI. Run A and B first to check the pipeline; use a shared
three-arm run for the final comparison. No run can establish a benefit from history
until the history model is evaluated on the same patients and prediction settings.

## Model contract

- **A:** latest MRI -> shared ResNet-101 -> shared projection -> continuous hazard.
  It receives no acquisition time in its embedding. The hazard query `u` is the
  survival-time variable present in every model; this is not a discrete output bin.
- **B:** the same MRI embedding plus a learned embedding of actual acquisition
  months `s / 12`. Nominal visit code and image age `t-s` are not extra features in
  this implementation. At a later landmark `t`, survival is conditioned through `t`.
- **History:** the same image encoder/projection and acquisition-time embedding
  for each available MRI, followed by a one-layer transformer (4 heads, width 128,
  feed-forward width 256, dropout 0). Select the latest observed token and pass it
  through exactly the same hazard-head class. Prefixes contain no future images;
  empty canonical slots are masked and never passed through the image encoder.

All arms use the same image standardization/resizing, orthogonal center slices,
ImageNet normalization, frozen BatchNorm running statistics and trainable encoder
weights. Augmentation is disabled for all arms. The optional step-ahead component
is disabled; the continuous history model currently implements survival only.

The positive monthly hazard is an MLP with softplus output. Thirty-two-point
Gauss-Legendre quadrature integrates it over continuous time. Quadrature points
are numerical integration locations and have no relationship to MRI bins.

Training minimizes `H(T|z)-H(s|z)-delta*log(lambda(T|z))`. At an evaluation landmark
`t`, require `T>t`, retain only MRIs with `s<=t`, and evaluate
`S(t+h|T>t,z)=exp(-integral_t^(t+h) lambda(u|z) du)`.

## Clinical eligibility and source scope

`core_survival_landmarks` joins the retained images in `core_image_set` to the
complete diagnosis-examination history in `ptid_dx_transforms`. The first AD
examination defines the event; otherwise the last valid diagnosis examination
defines censoring. No rounding or nominal bin target substitutes for these dates.
Clinical follow-up may extend beyond the last retained MRI or the image schedule.

The source preserves exclusion reasons. Exclusions include conflicting diagnoses
on the same clinical date, no clinical outcome, an outcome at/before baseline,
images at/after the endpoint, and images labeled AD. This rule uses dates and the
existing diagnosis label; it does not infer examination membership for an MRI
acquired on an earlier date. Source images are still limited to those retained by
`core_image_set`; this model does not reselect images from raw acquisitions.

Assignments come from `_raw.train_test_split`, never from cached tensor labels.
Unassigned source images are counted but excluded. A patient with no eligible
pre-outcome MRI is excluded from all arms. At a later landmark a patient whose
clinical follow-up has ended is also excluded from all arms.

The image-cohort notebooks and their earlier event counts describe a different
population/endpoint. Use the continuous run manifest for the prognosis cohort.
The initial audited source yielded 2,448 eligible patients and 7,222 MRI landmarks;
these are a historical check, not hard-coded acceptance thresholds.

## Prepare the database

Use the existing `.venv` in the repository root and start the existing Postgres
service. The new dbt model was already built successfully during initial setup.
If it is missing, or its upstream sources have intentionally changed, this command
loads `.env` and rebuilds only the clinical landmark model and its dbt test:

```powershell
.venv\Scripts\python.exe -c "from dotenv import load_dotenv; from dbt.cli.main import dbtRunner; load_dotenv('.env'); result = dbtRunner().invoke(['build', '--project-dir', 'mstat_project_dbt', '--profiles-dir', 'mstat_project_dbt', '--select', 'core_survival_landmarks']); raise SystemExit(0 if result.success else 1)"
```

This assumes `core_image_set`, `core_follow_up_buckets`, `ptid_dx_transforms`, and
patient splits are already present and current. It does not rerun the split
notebook. Keep all three model arms on one source/split snapshot.

## Verify changes before training

Run from a normal PowerShell terminal, outside Codex. The following tests exercise
the likelihood, missing bins, leakage boundaries, shared initialization, patient
sampling, checkpoint selection, artifacts, paired comparisons and CLI routing:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:POLARS_MAX_THREADS = '2'
$env:CUDA_VISIBLE_DEVICES = '-1'
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_continuous_survival.py tests/mstat_project/test_survival_comparison.py -x -v
```

The wrappers retain old Python APIs. Check those affected integrations separately:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_baseline.py tests/mstat_project/test_landmarks.py tests/mstat_project/test_ltsa_training.py tests/mstat_project/test_comparison_e2e.py -x -q
```

Optional static lint (no training):

```powershell
.venv\Scripts\python.exe -m ruff check src/mstat_project/ml
```

Then inspect source/tensor eligibility:

```powershell
.venv\Scripts\python.exe -m mstat_project.ml.baseline --preflight-only
```

A successful preflight leaves `status.json` containing `preflight_passed` and a
manifest with `tensor_patients_checked`. It checks package ownership, image IDs
and shapes for every eligible patient. Pixel finiteness is checked when an image
is loaded during training/evaluation. A missing image stops the run; it is never
silently replaced by another image or a different patient subset.

## Run the requested two-epoch checks

After the tests and preflight pass, enable the GPU and train A and B sequentially:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.baseline --epochs 2 --patience 2 --device cuda --cpu-threads 2 --bootstrap-samples 0
```

ImageNet weights are used by default and may download on first use. All eligible
training patients are included; each contributes one sampled MRI per epoch.
Validation uses every eligible MRI with equal total weight per patient. The two
arms use identical sampled patient/image lists, saved in `sampled_epoch_001.csv`
and `sampled_epoch_002.csv`. The selected checkpoint is restored before test
evaluation. Test results never choose the best epoch.

Expect a new run directory with:

| Artifact | Acceptance check |
|---|---|
| `status.json` | `completed` |
| `a/epochs.json`, `b/epochs.json` | Two epochs with finite train/validation NLL |
| `a/best.pt`, `b/best.pt` | Both exist; selected epoch is 1 or 2 |
| `a/test_predictions.json`, `b/test_predictions.json` | Same patient/landmark/outcome records and latest image |
| `comparisons.json` | B-minus-A point estimates; null intervals because bootstrap is disabled |
| `manifest.json` | Reviewed patient/event/exclusion counts and source hashes |

Later landmarks may legitimately contain no comparable event pairs. Their
C-indices are null. A passed two-epoch run demonstrates execution, not convergence
or a scientific result. Do not choose hyperparameters by inspecting these test
scores. Use validation data for model development.

## Monitoring and interruption

### Frozen-encoder control

To isolate encoder fine-tuning as a source of feature-scale growth, run a fresh
model A with the encoder frozen and diagnostics enabled:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.baseline --variants a --epochs 2 --patience 2 --device cuda --cpu-threads 2 --bootstrap-samples 0 --diagnostics --freeze-encoder
```

`--freeze-encoder` fixes encoder parameters and keeps the encoder in evaluation
mode, including BatchNorm running statistics. Frozen parameters are excluded from
the optimizer. The projection, hazard head, and applicable time/transformer
components remain trainable. Without the flag, the usual fine-tuning is unchanged.
The flag is saved in the manifest, checkpoints, and diagnostic reports.

This starts from the original ImageNet encoder by default, not a collapsed run's
checkpoint. Keep seed, preprocessing, batch size, and learning rate unchanged for
the control. Compare epoch-zero and post-training encoder feature summaries:
they should remain fixed. Then check whether projected features grow, Tanh
saturation develops, and hazard predictions still vary across the fixed panel.
If freezing prevents collapse, it implicates encoder fine-tuning under the current
settings; it does not establish clinical predictive performance. If collapse
persists with fixed encoder features, investigate the trainable projection/head.

### Optional fixed-panel diagnostics

Add `--diagnostics` to the continuous `baseline`, `ltsa`, `compare`, or `survival`
command. This is a boolean flag, disabled by default. It saves
`<variant>/diagnostics_epoch_000.json` before training and one
`diagnostics_epoch_NNN.json` after each completed epoch. These files do not select
checkpoints or change the training objective. Epoch zero makes it possible to
distinguish an initialization problem from collapse during optimization.

The panel contains one event patient and one censored patient from each of train
and validation, selected deterministically by seed. Missing strata are reported;
test patients are never substituted. Each patient's earliest eligible MRI is held
fixed across epochs and variants. History diagnostics therefore test a one-image
prefix, not longitudinal transformer behavior.

The trace uses the model already in memory, evaluation mode and inference mode,
processing one image at a time. It restores module modes and removes hooks after
each trace. It does not load another model/checkpoint or retain full images or
gradient graphs. It adds four MRI reads/forwards per trace when all strata exist,
plus a memory-mapped raw-image summary for each patient.

Each report contains per-patient magnitude statistics and across-patient variation
for raw volume, preprocessed volume, normalized three-channel input, encoder
features, projection, final head embedding, pre-Tanh values, Tanh outputs, monthly
hazard and survival. Raw/preprocessed images use 4x4x4 pooled sketches for
cross-patient comparisons; normalized slices use 8x8 per channel. Magnitude
statistics use full tensors. Features are compared coordinate by coordinate.
Image sketches cannot establish that full images are identical. No image files
are exported; use the image viewer separately for visual inspection.

Hazards are queried at a common anchor (the latest acquisition time in the panel)
and at anchor plus each configured horizon. Survival is conditioned at that same
anchor. This makes panel comparisons independent of differing evaluation times;
within-patient hazard ranges expose time sensitivity. These are descriptive
probes, not clinical evaluations or performance estimates. B/history embeddings
also contain acquisition time, so their across-patient variation is not purely
image sensitivity.

Inspect `panel_variation`, `first_near_constant_stage`, each patient's `stages`,
`hazard_per_month`, and `hazard_range_across_times`. Near-constant screening uses
coordinate range <= `1e-8 + 1e-6 * max(abs(value))` across the panel. Warnings also
flag nonfinite activations and more than 95% of Tanh values with absolute value
>= 0.999. These small-panel thresholds are diagnostic heuristics, not proof of a
cause. Forward traces do not measure parameter gradients.

The run's `training.log` records training progress and GPU allocated/reserved/peak
memory every 100 batches. This only measures PyTorch GPU memory, not CPU temperature
or all system memory. In a second terminal, optionally record GPU temperature,
power, utilization and memory without starting another workload:

```powershell
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv -l 2
```

Watch system RAM/commit in Task Manager. Run only one experiment process at a time.
`Ctrl+C` interrupts training; the last fully saved checkpoint remains. If a process
or the computer stops abruptly, `status.json` may remain `running` and the database
may show the same. Do not interpret this as a completed run. Automatic resume for
the continuous pipeline is not implemented; restarting the command creates a new
run, while preserving the interrupted run's files.

## Full controlled comparison

Use a single invocation to freeze the cohort across A, B and history:

```powershell
.venv\Scripts\python.exe -u -m mstat_project.ml.compare --variants a b history --epochs 50 --patience 10 --device cuda --bootstrap-samples 1000
```

If the history encoder requires a smaller batch on the 8-GB GPU, set
`--batch-size 1` for this entire comparison. The same hyperparameters, objective,
head, input preprocessing and patient sampling then apply to all arms. The
history model has extra transformer parameters; this comparison does not isolate
the effect of model capacity itself.

`comparisons.json` reports candidate-minus-reference differences for B vs A,
history vs A, and history vs B. Each landmark/horizon comparison uses the same
patients, identical endpoints, and identical latest available MRI. The bootstrap
resamples patients jointly, discards replicates without comparable pairs and
reports the number of usable replicates. No confidence interval is reported if
none are estimable. Results from separate invocations are not automatically
combined: rerun all three together for the controlled comparison.

The reported C-index is Harrell's ranking statistic after administrative censoring
at the horizon, not an IPCW estimate. Sparse late-landmark groups and heavy
censoring require caution; the intervals are pointwise rather than adjusted for
multiple comparisons. A positive history-minus-B estimate with uncertainty that
supports a benefit is evidence for the scientific question, not proof by itself.
Prespecify primary landmarks/horizons and consider additional seeds and calibration
assessment before a final research claim.

## Historical commands

`--legacy` on `baseline`, `ltsa`, or `compare` runs the previous Cox/discrete code.
Old Python `run_training`/`run_comparison` APIs remain historical. Continuous runs
use `mstat_project.ml.survival.run(SurvivalConfig(...))`. Historical checkpoint
resume options are only accepted with `--legacy`; their model states and objectives
are incompatible with the new neural hazard. The existing experiment-result
notebook/dashboard reads historical tables; inspect the new local result files
or `_ml.continuous_survival_runs` for this pipeline.
