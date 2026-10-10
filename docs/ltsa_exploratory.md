# Exploratory frozen LTSA-1L protocol amendment

After completion, the [full-cohort B continuation control](ltsa_b_continuation.md)
tests whether further shared-component training explains the observed NLL change.
The subsequent [two-layer depth screen](ltsa_depth_expansion.md) changes depth
only and compares against both completed controls.

The user authorized starting exploratory LTSA after the seed-42 architecture
investigation. The original production specification required stabilized A/B
to match historical validation NLL before any full-cohort LTSA training.
This amendment waives **only that NLL promotion prerequisite** for a separately
named, validation-only history experiment. It does not promote either baseline
or change the historical thresholds.

The question is whether temporal history improves the stabilized B model.
The initialization is the retained `seed_42/b/best.pt`, selected at epoch 4
with validation NLL 1.0751483213429989. The historical B score is
1.0589065400769206. Improvement over stabilized B alone does not establish
improvement over historical B.

## Commands

Run the focused synthetic integration tests first. The implementation agent
has not run tests or training:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_exploratory.py -x -v
```

Then start the experiment:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_exploratory --debug-run debug_dense_v5 --device cuda
```

No debug rerun, baseline retraining, cache export, or raw-image replay is
needed. The original production `ltsa_experiments history` command continues
to enforce the original promotion requirement.

## Controls and interpretation

- Frozen cached encoder; one temporal layer, width 128, four heads, FF width
  256, dropout 0.1, residual gate initialized to 0.01.
- Seed 42, batch size 6, learning rate and weight decay `1e-4`, clipping 5,
  maximum 30 epochs, patience 4; same optimizer, scheduler, patient sampling,
  conditional likelihood and patient-balanced validation as the baseline.
- Passed empirical debug gate and matching code, cohort, cache, encoder,
  preprocessing, checkpoint and protocol identities are still required.
  Both source baselines must pass the existing stability checks.
- The source B score is reproduced. With alpha zero, LTSA is compared with
  the actual loaded B model. With alpha 0.01, the maximum survival-probability
  difference must be at most 0.01 on the validation observations.
- The untrained LTSA checkpoint is saved separately and epoch zero remains
  eligible for selection. A fresh optimizer and scheduler are used.
- The shared trainer records early-epoch validation NLL, parameter counts,
  gradient/update norms, risk variation and activation diagnostics.
- Final validation comparisons include stabilized B, selected LTSA-1L and
  historical B evaluated on the same cached inputs and snapshots. The report
  leads with month 24 / next 60 months, includes the full grid and MRI-count
  strata, and uses the existing 1,000 patient-bootstrap resamples. Sparse
  strata remain descriptive. Full-follow-up validation NLL is reported
  separately from the horizon-based discrimination metrics.

The historical B cached score can differ slightly from its saved raw-image
score because of the verified encoder batch-size effect; both are labeled.
Any improvement is exploratory validation evidence after model investigation,
not a confirmatory test result. Negative or unstable results are retained and
reported. Optional depth/width, auxiliary losses, encoder fine-tuning, and
held-out test evaluation are outside this amendment.

## Artifacts

Default directory:
`data/artifacts/ltsa_v2/investigations/exploratory_history_1l_seed_42/`.

- `specification.json` records the amendment, failed production gate status,
  exact source files, passed debug gate, configuration and code fingerprints.
- `initialization_check.json`, `epoch_zero.pt`, and
  `epoch_zero_predictions.json` preserve the starting model and checks.
- `history-1l/` contains best/last checkpoints, epoch traces, diagnostics,
  parameter counts, validation predictions and summary. Checkpoints and
  summary are explicitly labeled exploratory and not production-promoted.
- `b_validation_predictions.json` and `legacy_b_validation_predictions.json`
  provide the paired controls.
- `paired_validation.json` contains the complete comparisons; `report.json`
  records primary results, NLL differences, stability status and artifact hashes.

Production modules are unchanged. This entry point validates the debug and
source checks itself, then uses the existing low-level trainer with its
production-gate invocation disabled. The real passed debug gate is pinned in
the amendment; no passing baseline gate is fabricated or written.

A completed matching run is reused after artifact hashes are checked.
Interrupted runs are preserved and rejected; use a new name such as
`--run-name exploratory_history_1l_seed_42_retry1` for a fresh attempt.
