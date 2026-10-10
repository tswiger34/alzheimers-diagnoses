# Full-cohort Baseline B continuation control

The exploratory LTSA run selected epoch 3 with validation NLL 1.0701266075,
versus 1.0751483213 for its stabilized B initialization. Its primary validation
C-index difference was -0.0043437, with a 95% paired bootstrap interval
[-0.0388237, 0.0326719]. This does not establish benefit from MRI history:
LTSA also continued training the shared projection, acquisition-time embedding,
and survival head. The B continuation control measures that alternative.

## Commands

The implementation agent has not run these tests or experiments. First run:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_b_continuation.py -x -v
```

Then run:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_b_continuation --debug-run debug_dense_v5 --device cuda
```

This reuses the completed `exploratory_history_1l_seed_42` run. No baseline,
LTSA, debug, cache, or raw-image replay rerun is required. The original training
implementation and gates are unchanged.

## Matched design

The model is exactly Baseline B: latest observed embedding plus acquisition-time
encoding, with no temporal module. Its initial weights are loaded strictly from
the same selected B checkpoint used by LTSA. The encoder remains frozen and is
never invoked. AdamW and ReduceLROnPlateau start fresh; saved optimizer state is
not loaded. The original B checkpoint is retained as epoch zero and is eligible
for selection.

Seed, patient sampling, loss, patient-balanced validation, batch size, learning
rate, weight decay, clipping, scheduler settings, patience and maximum epochs
match LTSA. Sampled patient/image CSVs are checked against LTSA for every
overlapping epoch. Each model's scheduler and early stopping respond to its own
validation loss, so selected epochs and eventual training lengths may differ.
This is the same selection procedure, not forced equality of training duration.

Before training, the runner checks the passed debug gate, stable A/B source
evidence, completed stable LTSA run, data and cache identity, configuration, and
artifact hashes. These remain exploratory validation experiments; the historical
NLL promotion prerequisite waiver is inherited from the LTSA amendment. There
is no held-out test evaluation or production promotion.

The trainer records the same optimization and fixed-panel diagnostics. It
reports NLL at matching epochs, independently selected NLLs, and paired
validation comparisons among original B, continued B and LTSA. The primary
comparison is **LTSA minus continued B at month 24 / next 60 months**, with the
full grid and MRI-count strata retained. Bootstrap intervals use 1,000 paired
patient resamples; sparse strata remain descriptive.

Negative `history_minus_continuation_nll` favors LTSA's overall conditional
likelihood. Positive primary C-index difference favors LTSA's discrimination.
Continued B matching or exceeding LTSA weakens the interpretation that history
caused the earlier NLL improvement. A favorable LTSA result remains single-seed,
post-investigation validation evidence, not proof of generalizable history benefit.

## Artifacts

Output: `data/artifacts/ltsa_v2/investigations/exploratory_b_continuation_seed_42/`.

- `specification.json` pins the parent run, initialization, configuration and code.
- `b-continuation/` contains initial/best/last checkpoints, epoch traces,
  sampled observations, parameter counts, diagnostics and validation predictions.
- `paired_validation.json` contains discrimination comparisons and strata.
- `report.json` contains selected NLL differences, common-epoch trajectories,
  stability status, primary paired results and artifact hashes.

Completed matching runs are reused after verification. Incomplete runs are
preserved; choose a fresh `--run-name exploratory_b_continuation_seed_42_retry1`
after an interruption. `--history-run` selects a different completed exploratory
LTSA source, subject to the same identity and configuration checks.
