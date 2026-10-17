# Scientific LTSA figure package

The question is whether earlier MRI scans improve survival prediction beyond the
latest MRI and its acquisition time, at month 24 over the next 60 months. The
reporter produces five main and three supplementary figures. Validation editions
are exploratory; held-out test editions require a separate final analysis lock.

The supplied [manifest](ltsa_figures_manifest.json) pins completed checkpoint,
summary and prediction SHA256 hashes. It includes the seed-42 models and seed-43/44
baselines. The matched continuation/history experiments and latest-only controls
are pending. A pending row is not a failed experiment or a zero effect.

## Commands

Run these from the repository root. The assistant has not executed these tests or
project commands. Each evaluation/render output directory must be new.

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_figures.py -x -v
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_figures audit --manifest docs/ltsa_figures_manifest.json
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_figures evaluate --manifest docs/ltsa_figures_manifest.json --split validation --device cuda --output data/artifacts/ltsa_v2/figure_evaluation_validation_v1
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_figures render --manifest docs/ltsa_figures_manifest.json --split validation --evaluation data/artifacts/ltsa_v2/figure_evaluation_validation_v1/evaluation.json --output data/artifacts/ltsa_v2/figures_validation_v1
```

To preview the figures before inference, omit `--evaluation` and choose a separate
new output directory. The likelihood panels will explicitly say that checkpoint
evaluation is pending. Render never constructs a model, selects a checkpoint,
trains, or calls `torch.load`. Evaluation constructs only frozen cached models;
it never constructs an optimizer or modifies the original runs.

The evaluation command replays the selected checkpoints and verifies their saved
primary predictions. It evaluates all eligible validation observations to obtain
the full patient-balanced loss components, and the month-24 cohort to obtain
60-month components. Historical B is loaded through the existing explicit legacy
inspection adapter. Its full-validation benchmark uses the verified singleton
cache replay rather than pretending its original raw-batch NLL is identical.

## Output and figure contents

Each figure is exported as vector PDF and SVG, plus a 300-dpi PNG. `captions.md`
contains scientific captions and pending experiment labels. `provenance.json`
contains source hashes, checkpoint hashes through the pinned manifest, reporting
implementation hash, bootstrap settings and figure hashes.

| Figure | Contents |
| --- | --- |
| 1 | Prediction timeline, shared model components, cohort support by MRI count |
| 2 | Absolute C-index, original/training-controlled paired C-index, paired 60-month NLL |
| 3 | Full-validation selection curves, matched-seed and direct history effects |
| 4 | Paired discrimination and primary-window NLL by MRI count, including ≥2 |
| 5 | Censoring-aware grouped probability calibration |
| S1 | Secondary landmark/horizon grid, support counts and sparse/nonestimable flags |
| S2 | Full-follow-up/60-month likelihood components and shared-architecture investigation |
| S3 | Actual update norms, fixed-panel risk variance, paired risk scatter, warnings and parameters |

`source_*.csv` files contain aggregate chart evidence. The risk scatter source is
exported as aggregate two-dimensional bin counts; raw patient identifiers and
patient-level evaluation rows are not exported into the figure directory.
`evaluation.json` contains patient-level rows and remains a local analysis artifact.

Open all PNGs at their intended print size. Check forest labels/intervals, calibration
labels, sparse markers, heatmap annotations, and grayscale distinguishability. The
automated export test checks file creation and source preservation; it is not a
substitute for reviewing rendered scientific figures.

## Statistical definitions

- The primary C-index preserves the project's existing Harrell rule: event time
  strictly shorter than the comparison time, half credit for tied risk, outcomes
  administratively censored at 60 months after the landmark.
- For primary likelihood, entry is month 24, endpoint is `min(outcome, 84)`, and
  events count only when they occur by month 84, inclusive. Quadrature uses original
  months. Saved full-follow-up prediction `nll` is never substituted for this loss.
- Overall selection NLL averages eligible observations within patients, then
  weights patients equally. Repeated landmarks are not independent patients.
- Use 1,000 paired patient resamples, seed 42, and 95% percentile intervals.
  Bootstrap multiplicities implement the same draws and estimates as the existing
  paired comparison implementation. Sparse groups (<30 patients or <5 horizon
  events) have descriptive point estimates without inferential intervals.
- With all seeds 42/43/44 complete, cross-seed summaries average the paired seed
  effects and draw the same patient IDs jointly across seeds. They do not average
  interval endpoints or count three seeds as three patient cohorts.
- Calibration uses model-specific predicted-risk tertiles, merged at tied
  boundaries. Within each fixed group, bootstrap whole patients. Observed risk is
  `1 - KM(60)` with early censoring retained. If follow-up ends before 60 while
  nonzero survival remains, observed risk is null. Sparse groups receive no interval.
  Independent censoring within groups is assumed. This is a coarse diagnostic,
  not a fitted calibration correction. Methodological context:
  [Austin et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7497089/).
- No practical-effect margin is selected. An interval containing zero supports an
  inconclusive finding; it does not prove equivalence or that earlier scans have
  no meaningful value. Subgroup interval differences are not interaction tests.

## Extending the manifest

Paths are relative to `root`, which is relative to the manifest location. Every
completed experiment needs `role`, `seed`, `checkpoint`, `checkpoint_sha256`,
`summary`, `summary_sha256`, `predictions` and `prediction_sha256` (the last two
are dictionaries keyed by split). Summaries need selected validation NLL, best
epoch, horizons, and the normal model identity. For older summaries that omit
`horizons_months` (including continued B), provide `specification` and
`specification_sha256` in the manifest. The reader verifies the saved specification
and uses its `config.horizons`; it never assumes a horizon order from defaults.
When both sources are present they must agree. Historical B is the explicit
legacy exception and also requires `reference_replay`. If a completed investigation
has a report, provide `report`; all its referenced artifact hashes are verified.

Completed models must share cohort, encoder, preprocessing and quadrature identity,
prediction horizons, patient/landmark keys, outcomes and eligible histories. Direct
history controls retain full eligible-history metadata even when earlier inputs
are masked. Do not truncate prediction metadata to describe only model inputs.

An optional `cohort_audit` provides reconciled image exclusions for Figure 1;
patients and images are always labeled with their respective units. A failed
experiment can be listed with `status: "failed"`, `reason`, `report`, and
`report_sha256`. Its failure is recorded in captions/provenance, and it receives
no performance estimate. Do not label a completed but unpromoted baseline as a
failed training run merely because its historical promotion threshold was missed.

For the full-history/latest-only pair, both entries also need a
`control_specification` JSON with matching:

- `shared_initial_weights` (file path) and `shared_initial_weights_sha256`;
- `baseline_checkpoint_sha256`, matching the corresponding seed's selected B;
- `sample_files_sha256`: shared selected-patient/prefix artifact paths and hashes;
- `optimizer`, `scheduler`, `selection`, `architecture`, `training_settings`;
- `input_mode`: `full_history` for full-history and `latest_only` for the control.

These artifacts come from a separately implemented, separately trained matched
control experiment. An inference-only mask is not accepted as that experiment.
For explicit checkpoint evaluation, the trained control checkpoint must also
record `config.input_mode = "latest_only"`. The reporter removes this reporting
extension before constructing the current `SurvivalConfig`, then masks all but the
latest observed input while retaining the full eligibility metadata.

The renderer integrates completed results supplied through this interface; it does
not launch missing seeds or train the latest-only control. No production training
files or fingerprint rules are modified by this reporting implementation.

## Final held-out edition

Create a new manifest after selecting the retained models and recording any protocol
amendments. Add already-authorized `test` prediction paths and hashes. Set
`final_lock` to an explicit reviewed JSON containing:

```json
{
  "selection_locked": true,
  "protocol_sha256": "canonical protocol fingerprint from survival_embeddings.fingerprint",
  "analysis_sha256": "SHA256 of ltsa_figures.py",
  "checkpoints": {"b:42": "selected checkpoint SHA256", "history-1l:42": "selected checkpoint SHA256"},
  "gates": [
    {"kind": "debug", "path": "path/to/gate.json", "sha256": "gate SHA256"},
    {"kind": "baseline", "path": "path/to/baseline_gate.json", "sha256": "gate SHA256"}
  ]
}
```

The checkpoint mapping must exactly include every completed manifest experiment.
Both gate kinds must be passed, match current production implementation hashes,
and match the locked protocol. Validation rendering works with the existing
exploratory results; those failed production baseline gates are not converted into
passed test authorization. Final lock validation occurs before test outcomes are
opened. The reporting command does not replace the project's explicit final test
reporting stage or authorize test-based model selection.

Use `--split test` with the same command structure, new evaluation/figure output
directories, and the final manifest. Training/selection curves remain explicitly
labeled validation curves inside the final figure edition.
