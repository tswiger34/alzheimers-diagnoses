# Stabilized LTSA: tests and gated experiments

The primary comparison is LTSA-1L versus time-aware Baseline B at month 24,
predicting the next 60 months. Retain A as the image-only control. The full
landmark/horizon grid is secondary. [ltsa_protocol.json](ltsa_protocol.json) pins
reference run `611be867-cfda-40a2-8dc2-04fa5b2e5046`, its data fingerprints,
checkpoint hashes, baseline scores, and thresholds. No new experimental results
are claimed by this guide. The implementation agent has not run the tests.

## Run the automated tests first

From the repository root in PowerShell:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:POLARS_MAX_THREADS = '2'
$env:CUDA_VISIBLE_DEVICES = '-1'
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_continuous_survival.py tests/mstat_project/test_ltsa_stabilization.py tests/mstat_project/test_survival_artifacts.py tests/mstat_project/test_survival_comparison.py -x -v
```

These tests use small synthetic CPU tensors. They require no Postgres, MRI
files, pretrained downloads, or GPU. They cover the numerical survival contract,
cache equivalence/provenance, causal masks, baseline transfer, diagnostics,
step-ahead targets, partial fine-tuning, gates, and paired reporting.

After those pass, check preserved historical APIs and wrappers:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_baseline.py tests/mstat_project/test_landmarks.py tests/mstat_project/test_ltsa_training.py tests/mstat_project/test_comparison_e2e.py -x -q
```

Optional static lint:

```powershell
.venv\Scripts\python.exe -m ruff check src/mstat_project/ml/continuous_survival.py src/mstat_project/ml/survival.py src/mstat_project/ml/survival_embeddings.py src/mstat_project/ml/survival_protocol.py src/mstat_project/ml/survival_optimization.py src/mstat_project/ml/survival_diagnostics.py src/mstat_project/ml/survival_comparison.py src/mstat_project/ml/ltsa_experiments.py
```

Repository-wide collection has known unrelated historical failures, including
`test_diagnoses_csv.py`. Synthetic tests cannot establish real-data overfitting
or an improvement from longitudinal information.

## Model and data contract

- Freeze the shared ImageNet ResNet-101 orthogonal center-slice encoder. Cache
  raw 2,048-dimensional features, then train the 2,048-to-128 projection and
  LayerNorm. The encoder stays in evaluation mode.
- The hazard head applies input LayerNorm, a 64-unit linear layer, GELU, a final
  linear layer, and Softplus. Monthly units remain
  `(softplus(logit)+1e-7)/12`. Initialize the final bias from patient-weighted
  training events/exposure using the inverse output transformation.
- Model time inputs use `t/(t+180)`; integrals, clinical times, and reported
  durations stay in original months. Thirty-two Gauss-Legendre points integrate
  the continuous hazard. Acquisition time and hazard query time remain distinct.
- LTSA uses width 128, four heads, feed-forward width 256, one pre-norm layer,
  GELU, and dropout 0.1. Only observed visits enter causal attention. Temporary
  compaction avoids fully masked queries; outputs return to nominal slots.
- Copy B's projection, normalization, time embedding, and head. Predict from
  `latest_B + alpha * LayerNorm(temporal_latest)`, initializing trainable alpha
  to 0.01. Alpha zero preserves B's predictions. Use a fresh optimizer.
- At prediction month `t`, require `T>t`, select actual acquisitions `s<=t`, and
  retain only scans strictly before the clinical endpoint (`s<T`). Same-day and
  post-outcome MRIs remain excluded. Nominal bins never substitute for dates.
- Train with one sampled MRI landmark per patient per epoch. Validation uses
  all eligible landmarks with weight `1/n_i`. Checkpoint selection and scheduler
  use patient-averaged survival NLL, including during auxiliary-loss experiments.

The conditional likelihood integrates from acquisition month `s` to endpoint
`T`. At later prediction landmarks, survival integrates from `t`, not the older
MRI date. Clinical labels come from the frozen clinical source, not old tensor
labels. Splits, preprocessing, and eligibility are shared by all arms.

## Research stages — only after tests pass

Run one stage at a time and inspect its artifacts. These commands perform real
data processing or training and are separate from the automated checks.

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m mstat_project.ml.ltsa_experiments audit
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments cache-debug --device cuda
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments debug --device cuda
```

Default workspace: `data/artifacts/ltsa_v2/`; change it with `--workspace`.
`--device cpu` is supported. `--tensor-dir` can relocate the same MRI packages.
The runner reads the saved reference snapshot without querying or changing the
database. Audit records validation support for the primary endpoint: at least
30 patients, five events within the horizon, and a comparable pair. This check
does not constitute a power calculation.

The cache stores image/patient IDs, splits, actual scan months, nominal positions,
observed status, endpoints, event flags, and fingerprints for the encoder,
preprocessing, and source snapshots. It rejects stale, missing, nonfinite, or
mismatched entries. Cache export writes atomically and encodes each MRI once.

The debug cohort fixes 16 training patients (eight events/eight censored) and
their latest eligible prefixes, including histories with missing visits. Seed
42, learning rate `1e-3`, no weight decay, dropout zero, and at most 2,000 updates
are used. Each arm must reduce NLL by >=0.5 nats, reach training C-index >=0.95,
and retain risk variance >`1e-8`. History starts from debug B and its initial
survival deviation must be <=0.01. Add `--anomaly-detection` for debugging.
Debug C-index ranks 60-month cumulative hazards on full-follow-up comparable
pairs. This preserves the mathematical ordering of 60-month probabilities
without introducing floating-point ties from `1-exp(-H)`. The trace also stores
the probability C-index and the count of probabilities rounded to one. The
variation threshold continues to use probability variance. NLL and C-index
thresholds must be met at the same evaluated update.
LTSA is checked after every update so a short passing interval is not missed.
A/B retain checks at update 1, every 25 updates and the final update, preserving
the baseline checkpoint used for transfer. Detailed optimization diagnostics
remain on that 25-update schedule for all arms. Extra LTSA evaluations restore
the CPU random state used by DataLoader iterator seeds. Trace rows without
detailed instrumentation have an empty `optimization` list; they still contain
the post-update gate metrics. Traces flush every 25 updates and on success or a
caught training error.

The debug runner saves `<arm>_best_rank.pt` whenever NLL improves among evaluated
updates with C-index >=0.95. These diagnostic checkpoints supplement the final
`<arm>.pt`; they do not independently authorize a passed gate.

Failures return a nonzero exit status and save `debug/gate.json` plus available
traces. Do not edit a failed report into a passing one. Full stages require the
same code, protocol, cohort, and encoder as the passed gate.

If NLL improves while the debug C-index falls, inspect the saved checkpoints
before rerunning training:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_stabilization.py -k "debug_risk_inspection or debug_prediction_inspection or debug_inspection_preserves" -x -v
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_debug_inspection --device cuda
```

This command performs inference on the same cached training observations for
each available A/B/history checkpoint. It writes a new timestamped
`debug/inspection_*.json` report, preserving the checkpoints, traces, and gate.
It does not train, regenerate embeddings, or authorize full-cohort runs.
The report binds to checkpoint/cache/data fingerprints and records any code
difference from training. This standalone inspector does not change the
training implementation fingerprints.

Compare the original full-follow-up probability C-index with the cumulative
hazard C-index. Although `1-exp(-H)` is mathematically monotone, finite precision
can round different risks to one (or zero). The report counts comparable pairs
tied only by this conversion, and also evaluates the probability conversion in
float64. Cumulative hazard avoids the probability conversion entirely. A
separate horizon-truncated analysis censors follow-up at 60 months; it does not
replace the gate's full-follow-up calculation. Inspection never automatically
converts a failed run to a pass.

History's NLL target is 0.5 below its own transferred-B initialization, not the
cold-start baseline initialization. The report includes that required NLL and
the lowest trace NLL with a passing recorded C-index. Earlier trace rows are
metrics only in the original run; the inspector cannot reconstruct unsaved
intermediate weights.

Checkpoint and JSON publication use unique temporary files in the destination
directory, closing each file before atomic replacement. Access-denied and
sharing-lock errors are retried eight times with bounded backoff (about 3.6
seconds total). Persistent errors preserve the old destination and retain the
new temporary file, whose path is included in the error. Other I/O errors fail
immediately. No existing destination is deleted to force a write. These retries
address temporary access failures; they cannot repair persistent permissions
or a file held open indefinitely.

An interrupted debug attempt should be retained and restarted under a fresh
`--debug-run` name. Tiny-subset checkpoints do not include optimizer state and
cannot provide an exact training resume. The cache remains reusable after this
artifact-writing change. A saved temporary checkpoint does not constitute a
passed gate.

### Diagnosed debug failure and numerical correction

The original seed-42 attempt in `debug/` passed A/B but failed history after
2,000 updates. The user-run report
`debug/inspection_20261009T023503433965Z.json` reproduced history NLL 0.350424,
with 10 probabilities rounded to one and 35 of 66 comparable pairs tied by the
probability conversion. History C-index was 0.734848 using float32 probability,
0.787879 using float64 probability, and 0.803030 using cumulative hazard.
Truncating follow-up at 60 months left all 66 pairs and scores unchanged.
Thus rounding explains part of the reported decline, but the final checkpoint
still fails the 0.95 ranking requirement. It has not been promoted.

After tests pass, retry with the corrected ranking calculation in a new named
directory, reusing the same cache and unchanged training settings:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_stabilization.py -x -v
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments debug --debug-run debug_risk_v2 --device cuda
```

Existing debug directories cannot be overwritten. `--debug-run` accepts `debug`
or a name such as `debug_risk_v2`; each attempt has its own gate and checkpoints.
Pass that same option to subsequent experiment stages to select its gate, and
to `ltsa_debug_inspection` to inspect it. For example, only after this retry
passes, use `cache-full --debug-run debug_risk_v2 --device cuda` and
`baselines --debug-run debug_risk_v2 --device cuda`. The original failed gate
remains failed; its training code fingerprint differs after the metric fix.

The corrected attempt `debug_risk_v2_retry1` also failed the joint criteria.
Its best saved checkpoint with C-index >=0.95 was update 525: NLL 1.162091,
C-index 0.969697. History's own initialized NLL was 1.500522, requiring NLL
<=1.000522. At update 2,000, NLL was 0.350424 but cumulative-hazard C-index
was 0.803030. These observations establish a remaining ranking problem;
they do not establish that the temporal module is its cause.

### Matched Baseline B continuation control

The standalone control below resumes **weights only** from that attempt's B
checkpoint, then creates fresh AdamW state as used for history initialization.
It trains only B's projection, time encoding and head on the exact same fixed
training panel, with dropout zero, learning rate `1e-3`, no weight decay,
clipping 5 and 2,000 additional full-batch updates. It uses the existing frozen
embedding cache. The source code, protocol, checkpoint identities and panel
must match before the control starts.

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_stabilization.py -k b_continuation -x -v
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_debug_control --debug-run debug_dense_v3 --device cuda
```

Each invocation creates a fresh timestamped `b_continuation_*` directory
inside the selected debug attempt. It preserves the source gate and weights,
and writes a report, optimization trace, and diagnostic checkpoints for lowest
NLL, lowest NLL with passing C-index, and the last update. It also saves the
first checkpoint meeting both numerical criteria, if one exists, but continues
through the full budget for a matched trajectory. Trace rows include the source
history metrics at the same update where available. No gate is created or
promoted by this control, regardless of its outcome.

If B also loses ranking as its likelihood improves, investigate the shared
survival objective, hazard behavior and optimization before attributing this
failure to temporal attention. If continued B meets the joint criteria while
history does not, investigate the additional temporal parameters and their
optimization. This is a single-seed diagnostic, not evidence of validation or
test performance. Use a source attempt created with the current training code;
the command above selects the dense-check attempt described below.

The completed control for `debug_risk_v2_retry1` is recorded in
`b_continuation_20261009T034417383794Z/report.json` within that attempt. B met
both numerical criteria at update 750: NLL 0.957844 and cumulative-hazard
C-index 0.984848, against required NLL <=1.000539. Its final C-index was
0.818182. Thus the shared model can meet the joint gate, and it too loses
ranking with continued likelihood optimization. This does not resolve whether
LTSA had an unobserved passing interval between the original 25-update checks.

The completed every-update attempt `debug_dense_v3/gate.json` records a passed
gate for all three arms, with saved checkpoints:

| Arm | Update | Training NLL | C-index (cumulative hazard) |
|---|---:|---:|---:|
| A | 25 | 1.499457 | 0.984848 |
| B | 25 | 1.500539 | 0.969697 |
| LTSA | 1367 | 0.652803 | 0.969697 |

LTSA reduced NLL by 0.847718 from its transferred initialization, retained risk
variance 0.127892, and passed the initialization check (maximum survival
deviation 0.000173). Its passing update lies between the earlier 25-update
checks. The trace still shows substantial fluctuations, and eight probabilities
at the selected checkpoint round to one; the probability C-index is 0.825758.
This passes the defined tiny-subset gate, not a validation-performance or
full-cohort stability assessment. Earlier failed runs remain unchanged.

Use the passed `debug_dense_v3` gate for full-cohort stages. First generate the
full embedding cache; only after successful completion, train A/B:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments cache-full --debug-run debug_dense_v3 --device cuda
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments baselines --debug-run debug_dense_v3 --device cuda
```

Baseline defaults match the reference: seed 42, batch size 6, AdamW learning
rate/weight decay `1e-4`, clipping 5, at most 30 epochs, patience 4. Plateau
scheduler patience is 3 with factor 0.5. Promotion requires stable predictions
and validation NLL <=1.0590972830 for A and <=1.0589065401 for B, with numerical
tolerance `1e-6`. `seed_42/baseline_gate.json` binds promotion to exact checkpoint
files and identities. A failed promotion blocks history training.

After baseline promotion:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments history --debug-run debug_dense_v3 --device cuda
```

`seed_42/history-1l/initialization_check.json` records deviation from B before
training. Epoch zero remains selectable. Development stages write validation
predictions, never held-out test predictions.

## Optional extensions

Before depth, width, or step-ahead comparisons, run `baselines` and `history`
with `--seed 43`, then `--seed 44`. Each seed must pass its baseline gate and
have a stable one-layer control. The frozen cache and seed-42 debug gate are shared.
Include `--debug-run debug_dense_v3` for each of those runs.

```powershell
.venv\Scripts\python.exe -u -m mstat_project.ml.ltsa_experiments depth --debug-run debug_dense_v3 --device cuda
.venv\Scripts\python.exe -u -m mstat_project.ml.ltsa_experiments width --debug-run debug_dense_v3 --device cuda
.venv\Scripts\python.exe -u -m mstat_project.ml.ltsa_experiments step-ahead --debug-run debug_dense_v3 --device cuda
.venv\Scripts\python.exe -u -m mstat_project.ml.ltsa_experiments fine-tune --debug-run debug_dense_v3 --device cuda
```

- Depth compares two layers against one across the three seeds. Retain only if
  mean validation NLL improves and primary validation C-index does not decrease.
- Width requires a completed depth decision. Width 256 uses adapters preserving
  the 128-dimensional baseline/head interface and compares against the retained
  depth at width 128.
- Step-ahead predicts the next observed frozen embedding using actual elapsed
  time. Targets are detached; MSE averages over features, valid pairs, then
  patients with pairs. No subsequent visit means no auxiliary loss. Start at
  weight 0.01; retain on improved mean NLL and nondecreasing primary C-index.
- Fine-tuning is a seed-42 secondary experiment initialized from frozen LTSA-1L.
  Train only ResNet `layer4` at `1e-5`, versus `1e-4` for other trainable modules;
  keep BatchNorm statistics fixed. Use raw images; cached inputs are rejected.

Decisions record validation results, code/protocol fingerprints, and checkpoint
hashes. Rejected variants are valid experimental outcomes. Further depth or
encoder unfreezing is not automatic.

## Final held-out report

After every attempted optional experiment has a recorded validation decision:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_experiments report --debug-run debug_dense_v3 --device cuda
```

This explicit stage locks retained checkpoints before loading test prediction
batches. `final_report/` contains the locked manifest, per-model test predictions,
model identities and counts, `comparisons.json`, and `report.md`.

Comparisons include test NLL, individual C-indices and intervals, paired
differences, and MRI-count strata (1, 2, >=3, including the latest scan). Bootstrap
resampling draws whole patients jointly within each landmark/stratum: 1,000
samples, 95% percentile intervals. Repeated landmarks are never treated as
independent patients. Strata below 30 patients or five events receive descriptive
estimates only; nonestimable metrics stay null. Intervals are pointwise, and
Harrell C-index is horizon-truncated rather than IPCW-adjusted.

## Diagnostics and compatibility

Forward panels preserve multi-visit prefixes and compare risk at common times.
Reports include hidden preactivations, temporal scale, hazard and cumulative
hazard distributions, and risk variance. Optimizer reports separate projection,
time, attention, feed-forward, residual, survival-head, and optional encoder
gradient/update norms. They record minibatch events and both NLL components.

The ordinary `baseline`, `ltsa`, and `compare` entry points require
`--debug-gate` for full production training. History also requires
`--baseline-init` and `--baseline-gate`. `--protocol-path` selects the locked
protocol; live database fingerprints must match it. `--cache-path` selects
embeddings, and `--experiment-id` names a single variant's directory. Training
rejects `--evaluate-test` and directs users to the separate reporting stage.

The low-level Python trainer remains usable for synthetic tests; production
callers supply the gates. Preflight-only commands do not require a passed gate.
Incomplete experiment directories are rejected; automatic crash resume is not
implemented. Use a fresh workspace for intentional code/protocol changes and
retain old artifacts.

Version-1 continuous Tanh checkpoints cannot load into the new architecture.
Historical Cox/discrete Python APIs and `--legacy` commands are unchanged.
# Exploratory LTSA amendment

For the authorized validation-only LTSA experiment initialized from the stable
but unpromoted seed-42 B checkpoint, follow
[the exploratory run guide](ltsa_exploratory.md). This separate entry point
records the historical NLL prerequisite waiver and preserves the original
production gates. It reuses `debug_dense_v5` and the existing frozen cache.

