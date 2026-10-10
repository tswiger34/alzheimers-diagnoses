# Shared architecture isolation and validation loss decomposition

The patience-8 investigation did not identify the source of the reference NLL
gap. This separate investigation first decomposes the saved reference and
stabilized checkpoints' validation losses, then trains matched architecture
controls. No empirical architecture results are available until these commands
are run. Production settings and promotion thresholds remain unchanged.

## Commands

From the repository root, run the new synthetic tests first (the implementation
agent has not executed tests or training):

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_architecture_ablation.py -x -v
```

Inspect saved A/B checkpoints without training:

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_architecture_ablation inspect --debug-run debug_dense_v5 --device cuda
```

This must report `reproduced: true`. It evaluates the historical checkpoint
architecture explicitly, verifies its encoder against the embedding cache,
and checks that both historical and stabilized validation NLLs reproduce
within `1e-5`. This diagnostic tolerance does not change the production
promotion tolerance of `1e-6`.

If only historical checkpoints fail reproduction, do not start training or
increase the tolerance. A separate raw-image diagnostic compares the original
validation batch size, singleton inference used by cache export, and cached
features on exactly the same validation observations:

```powershell
.venv\Scripts\python.exe -X faulthandler -m pytest tests/mstat_project/test_ltsa_reference_replay.py -x -v
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_reference_replay --debug-run debug_dense_v5 --device cuda
```

This requires the original MRI tensor drive and performs encoder inference,
without training, cache replacement, or changing the reproduction gate. It
preserves the current study and writes a new `reference_replay_<timestamp>/`
directory inside it. The report includes feature discrepancies, each path's
NLL under the original weighted reduction, and component differences. If raw
batches reproduce the historical score while singleton/cache paths agree with
one another but differ from raw batches, that supports an encoder execution
batching effect. If raw replay also fails, historical runtime/data/preprocessing
differences remain unresolved. The diagnostic by itself does not bypass the
reproduction gate.

### Verified replay result and cached-reference comparison

The user-run replay `reference_replay_20261010T001531716004Z/report.json`
reproduced both historical weighted validation NLLs exactly with raw batches
of six. Singleton inference matched cached features exactly. Batched versus
singleton features differed by up to 0.01283163 (mean per-image RMS 0.00034152).
The cached historical NLLs were A 1.0590723280129364 and B 1.0588779447988808,
respectively 0.0000249550 and 0.0000285953 below their historical raw scores.
This isolates the reproduction mismatch to encoder execution batch size in
this replay; it does not explain the much larger stabilized-model NLL gap or
identify which architecture change caused that gap.

The runner can pin this replay as evidence in a new study. It verifies the
original study fingerprint, scientific inputs, replay implementation, runtime,
raw score reproduction, and exact singleton/cache agreement. Cached reference
scores then become the inspection targets for this diagnostic comparison;
historical scores remain recorded and production promotion limits do not change.
The fresh stabilized training control must still reproduce before ablations.

After rerunning `test_ltsa_architecture_ablation.py`, use:

```powershell
$replay = 'data\artifacts\ltsa_v2\investigations\architecture_seed_42\reference_replay_20261010T001531716004Z\report.json'
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_architecture_ablation train --debug-run debug_dense_v5 --device cuda --study architecture_seed_42_replay --reference-replay $replay
```

The `train` stage first reruns saved-checkpoint inspection automatically. The
same `--study` and `--reference-replay` flags are required for later invocations
of this study. Original inspection and replay artifacts remain untouched.

For a study whose original cached inspection passes without replay evidence,
the basic architecture-comparison command is below. For the observed seed-42
batching discrepancy, use the replay-backed command above instead.

```powershell
.venv\Scripts\python.exe -u -X faulthandler -m mstat_project.ml.ltsa_architecture_ablation train --debug-run debug_dense_v5 --device cuda
```

The default is 19 cached training runs: nine for A and ten for B, including
fresh stabilized controls. To investigate B first, append `--arms b`.
To limit candidates, append e.g. `--experiments no_head_norm tanh`; the fresh
stabilized control is still required. Completed matching runs are reused when
subsequent commands expand the study. No cache generation or debug retraining
is needed for these additions.

## Controlled changes

Every single-factor candidate reverses one stabilized setting while retaining
the others. The legacy time transform is the actual historical `t/12`, not the
todo's suggested `t/180`.

| Experiment | Change relative to stabilized control |
|---|---|
| `stabilized` | None; must reproduce the saved stabilized selected NLL |
| `no_projection_norm` | Remove projection LayerNorm |
| `no_head_norm` | Remove head input LayerNorm |
| `tanh` | Replace head GELU with Tanh |
| `legacy_hazard_time` | Query the head with time divided by 12 |
| `legacy_acquisition_time` | Encode acquisition time divided by 12 (B only) |
| `default_output_weights` | Restore default final-linear weights |
| `default_output_bias` | Restore default final-linear bias |
| `legacy_initialization` | Restore both final weights and bias; joint control |
| `legacy_all` | Revert all seven settings; joint historical-architecture control |

Shared projection, time embedding, and hidden-layer initial weights match
across candidates. Random draws are held constant even when final-layer
initialization differs. Each run starts fresh with seed 42, the same frozen
encoder cache, patient cohort, sampling, conditional likelihood, quadrature,
optimizer, scheduler, batch size, 30-epoch cap and patience 4. Ordered sampled
observations are fingerprinted each epoch. Additional validation diagnostics
preserve the training random-number stream. Selection uses the existing
patient-balanced validation NLL.

A fresh stabilized control must reproduce the saved selected NLL before its
arm's other candidates train. The historical raw-encoder construction consumed
random numbers differently from cached model construction. Consequently,
`legacy_all` shares initialization with the new cached controls; its fresh
training trajectory need not reproduce the historical run. Saved historical
checkpoint evaluation, however, must reproduce the historical validation NLL.

## Reading the loss breakdown

For each observation with entry time `s`, outcome time `T`, and event `d`:

```text
NLL = integral(s, T, monthly hazard(t) dt) - d * log(monthly hazard(T))
    = cumulative_hazard + event_log_hazard
```

Integration remains in months. The event-log term is zero for censored
observations and can be negative for events; do not interpret it as a
nonnegative penalty. A smaller cumulative-hazard term can trade off against a
worse event-log term.

Reports average observations within each patient, then patients equally. They
include both terms, their sum, endpoint hazards, patient-level paired
differences, and the largest patient-level NLL increases. Event/censored
groups have conditional means and contributions to the overall mean.
Entry-time ranges `[0,24)`, `[24,60)`, and `[60,infinity)` use contributions
with the original patient weights, so repeated prefixes are never treated as
independent patients. Contributions sum back to the overall terms.

The saved-checkpoint report's `stabilized_minus_reference` explains which loss
term accounts numerically for the observed gap. Candidate reports show
differences from both saved controls. Negative delta NLL favors the candidate;
delta H plus delta event-log reconstructs delta NLL. The reports also retain
validation C-index based on cumulative hazard at month 24 / horizon 60,
comparable pairs, risk variation, and probability-rounding counts.

Single-factor reversals estimate effects conditional on the other stabilized
settings. They do not uniquely attribute causation when changes interact.
The joint initialization and all-legacy controls help reveal such interactions;
unexpected combinations warrant targeted follow-up. Seed-42 screening alone
does not establish a reproducible improvement across seeds. Held-out test
outcomes are not evaluated and no candidate is automatically promoted.

## Artifacts and recovery

Default root: `data/artifacts/ltsa_v2/investigations/architecture_seed_42/`.

- `specification.json`: pinned checkpoint, cache, cohort, protocol, code and
  debug-gate fingerprints, plus the complete experiment catalog.
- `saved_checkpoint_inspection/report.json`: saved-NLL reproduction and loss
  differences; adjacent per-arm JSON files retain observation/patient details.
- `{a,b}/{experiment}/`: initial/best/last checkpoints, epoch traces,
  optimizer diagnostics, parameter counts, selected loss components, paired
  differences and summary. Checkpoints use a separate diagnostic format.
- `report.md` and `report.json`: completed candidates' selected NLL and paired
  component differences. Refresh with the `report` stage and the same flags.

Existing production baselines and their failed promotion gate are preserved.
Completed candidates are reused only with matching identities and artifact
hashes. An interrupted candidate is preserved and rejected rather than silently
restarted or overwritten. Use a new name, e.g. `--study architecture_seed_42_v2`,
for a new study after interruption or a code/provenance change.
