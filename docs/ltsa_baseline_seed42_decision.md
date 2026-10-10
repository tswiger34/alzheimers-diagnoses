# Seed-42 stabilized baselines: promotion rejected

The user-run production baseline experiment completed for A and B. Neither
selected checkpoint meets the locked validation NLL requirement. Full-cohort
LTSA training remains blocked. This is an experimental outcome, not an
unhandled training failure.

## Evidence

Source workspace: `data/artifacts/ltsa_v2/`.
Debug gate: `debug_dense_v5/gate.json` (passed).
Baseline decision: `seed_42/baseline_gate.json` (failed).
Per-arm records: `seed_42/{a,b}/summary.json`, `epochs.json`,
`diagnostics_epoch_*.json`, `optimization_epoch_*.json`, and `best.pt`.

| Arm | Selected epoch | Validation NLL | Reference limit | Excess NLL | Implemented stability checks |
|---|---:|---:|---:|---:|---|
| A | 4 | 1.0760007931 | 1.0590972830 | 0.0169035100 | Passed |
| B | 4 | 1.0751483213 | 1.0589065401 | 0.0162417813 | Passed |

The permitted numerical comparison tolerance is `1e-6`; neither difference
can be explained by that tolerance. Both runs stopped after epoch 8 with four
successive non-improving validation epochs. Saved checkpoints and the cache
remain available. The existing-run check will reuse these completed runs;
repeating the same baseline command is not a new investigation.

Checkpoint SHA-256 values recorded by the gate:

- A: `e56665591edee279cf3d23cbb8294bcc5e09c0d6d98e98b43250281ec87c655d`
- B: `29e0a43583de6759694a004525aed717634df94baf03057e87c6aaca13d6e201`

The gate records architecture version 2, time transform `t/(t+180)`, frozen
encoder/cache identity, preprocessing fingerprint, cohort fingerprint, protocol
fingerprint, and implementation hashes. No held-out test score was used for
this decision.

## Interpretation and tested scheduling hypothesis

Between epochs 4 and 8, A's sampled training NLL decreased from 1.068614 to
1.039641 while validation NLL rose from 1.076001 to 1.086738. B showed the same
pattern: sampled training NLL decreased from 1.068003 to 1.039313 while
validation NLL rose from 1.075148 to 1.086715. The sampled training observations
change each epoch, so these are not fixed-panel learning curves. The repeated
validation deterioration nevertheless indicates a generalization problem to
investigate. Passing finite/variation diagnostics does not establish that the
new architecture matches the reference's validation likelihood.

There is also a concrete scheduling limitation. The trainer uses plateau
scheduler patience 3, reduction factor 0.5, and early-stopping patience 4.
After the epoch-4 optimum, the scheduler's first reduction is scheduled after
the fourth bad epoch (epoch 8), immediately before early stopping. All eight
epochs therefore trained at `1e-4`; no epoch used the reduced rate. This follows
the existing settings and does not establish that a longer run would improve
the selected score.

The follow-up experiment isolated this scheduling effect: a separately
identified, validation-only A/B investigation with early-stopping patience 8,
keeping scheduler patience 3, the 30-epoch maximum, seed, initialization, cache,
sampling, and all other training settings fixed. It would allow reduced-rate
epochs to run. This is an exploratory protocol variation, not a change to the
locked production specification.

## Running the requested investigation

The separate `ltsa_baseline_investigation` command implements the patience-8
comparison. It reconstructs each source configuration and changes only early
stopping patience scientifically. Both arms start fresh from the same seed;
they do not resume the trained weights. It verifies the source checkpoint,
cache, cohort, protocol, and passed debug gate before training. Production
training modules are unchanged, so `debug_dense_v5` remains reusable.

First run the focused synthetic regression, which has not been run by the
implementation agent:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_stabilization.py -k patience_investigation -x -v
```

Then run:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_baseline_investigation --debug-run debug_dense_v5 --device cuda
```

Results go to `data/artifacts/ltsa_v2/investigations/patience_8_seed_42/`.
The specification records the single setting change and source fingerprints.
The report compares validation NLL with the patience-4 controls and reference
limits, alongside the existing stability checks. Original baseline directories
and their failed gate are preserved; candidate results are never automatically
promoted.

## Patience-8 result: reject the extension

The completed user-run report is
`data/artifacts/ltsa_v2/investigations/patience_8_seed_42/report.json`.
Both arms ran through epoch 12 and still selected epoch 4. Selected validation
NLL was exactly unchanged: A 1.0760007931 and B 1.0751483213. Both passed the
implemented stability checks but failed their reference NLL limits.

The per-arm `epochs.json` files confirm that epochs 9 through 12 trained at
`5e-5`, so this experiment did exercise the scheduler reduction. The best NLL
among those additional epochs was 1.0847339010 for A and 1.0834644753 for B,
both at epoch 9 and both worse than the original epoch-4 selections. Epoch-12
validation NLL was 1.0887020134 for A and 1.0916555197 for B.

Decision: retain patience 4 in the locked specification; the tested patience-8
extension provides no validation benefit. The scheduler/early-stopping
interaction alone did not explain or resolve the reference-score gap in this
seed-42 experiment. It does not rule out other optimization effects. Further
investigation should isolate the shared architecture changes with matched
ablations and examine validation loss components, rather than infer a cause
from this negative scheduling result. No additional patience extension is
selected, and full-cohort LTSA remains blocked.

The next investigation is implemented in the separate
[shared architecture ablation guide](ltsa_architecture_ablation.md). It first
decomposes the saved validation losses, then tests matched single-factor
reversals and joint controls without changing the production specification.
Architecture attribution remains pending those results.

Investigation-selected checkpoint SHA-256 values:

- A: `5dd8fffa363b034227815ee9d078935ea7bd02b5c35affa78764affabf949428`
- B: `de065976cc5e96319a7068b77dab4ecba512ef64b20f8df7235330cc3e8fa511`

The failed run is preserved. No threshold has been relaxed, no checkpoint has
been promoted, and no implementation, optimizer setting, or provenance gate
was changed in the locked production specification as part of this review.
