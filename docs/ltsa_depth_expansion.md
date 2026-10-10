# Exploratory two-layer LTSA expansion

This experiment tests depth only: two temporal layers instead of one, retaining
width 128, four heads, feed-forward width 256, dropout 0.1, causal masking and
the normalized residual correction. It starts from the same stabilized B
checkpoint as the completed one-layer run, with a fresh optimizer and scheduler.
It does not resume one-layer trained weights. All sampling, loss, optimization,
early stopping and validation settings remain fixed.

Run the focused tests first; the implementation agent has not run them:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_depth_expansion.py -x -v
```

Then run the seed-42 experiment:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_depth_expansion --debug-run debug_dense_v5 --device cuda
```

Defaults reuse `exploratory_history_1l_seed_42` and
`exploratory_b_continuation_seed_42_retry1`. Neither needs rerunning; neither do
debugging or cache generation. Use `--history-run` or `--control-run` only if
the corresponding completed run has a different name.

The entry point checks the passed debug gate, stable source baselines, completed
stable one-layer run and continuation control, matching configurations, and
artifact fingerprints. Initialization must satisfy the existing 0.01 maximum
survival-probability difference from alpha-zero B on validation observations.
Epoch zero is saved separately and remains eligible for checkpoint selection.
The shared trainer produces gradient/update, activation, risk-variation and
parameter-count diagnostics. Sampled patients and MRIs are verified against
the one-layer run for overlapping epochs.

Results go to
`data/artifacts/ltsa_v2/investigations/exploratory_history_2l_seed_42/`.
`report.json` records selected NLL differences versus one layer and continued B,
stability, parameter counts and the primary paired depth comparison at month 24
/ next 60 months. `paired_validation.json` includes the full grid and MRI-count
strata, with 1,000 patient-bootstrap resamples where support permits.

This is a user-authorized **exploratory seed-42 depth screen**, following the
earlier baseline-threshold amendment. It does not fulfill the original issue-12
multi-seed promotion requirement. A lower validation NLL with no decrease in
primary C-index and passed stability checks yields
`candidate_for_multiseed_validation`; otherwise the report says
`no_depth_benefit_established`. Neither status promotes a production model.
Matched baseline initializations and one-/two-layer runs for seeds 43 and 44
would still be needed before a multi-seed depth decision. Width expansion,
auxiliary loss and encoder fine-tuning are not bundled into this depth test.
Held-out test evaluation remains deferred.

Existing runs are preserved. Completed matching output is reused after hash
verification. For an interrupted experiment, use a fresh name such as
`--run-name exploratory_history_2l_seed_42_retry1`.
