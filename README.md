# LTSA MSTAT Project

## Single-image survival baseline

The baseline selects one MRI per patient: the final eligible MRI strictly before AD diagnosis for an event
patient, or the final MRI for a censored patient. It trains an ImageNet-pretrained ResNet-101 on the MRI's three
orthogonal center slices using the Cox partial likelihood, then evaluates the validation-selected checkpoint with
Harrell's C-index on the patient-level test split.

```powershell
uv run python -m mstat_project.ml.baseline --epochs 10 --batch-size 2 --device auto
```

Checkpoints are written to `data/artifacts/model_checkpoints/baseline/<run_id>/epoch_NNN.pt`. The command creates
and populates these Postgres tables:

- `_ml.baseline_runs`: configuration, cohort counts, best epoch, and test C-index
- `_ml.baseline_epoch_metrics`: training/validation metrics and checkpoint path for every epoch
- `_ml.baseline_run_patients`: exact image, outcome, and split used for every patient
- `_ml.baseline_test_predictions`: patient-level test risk scores

For example, inspect completed experiments with:

```sql
SELECT run_id, completed_at, best_epoch, best_validation_loss, test_c_index
FROM _ml.baseline_runs
WHERE status = 'completed'
ORDER BY completed_at DESC;
```

## Methodology

We approach disease prognosis through the lens of survival analysis, which aims to model a “time-to-event” outcome from potentially time-varying input features. We adopted a discrete-time survival model, given that imaging measurements were either acquired at intervals as short as 6 months (AREDS) or 1 year (OHTS), and we assumed uninformative right-censoring. The collection of longitudinal images for eye i can be written

where Ji is the number of longitudinal images acquired for eye i, ti, j is the time (in years) of the jth image measurement for eye i, and [Math Processing Error]
 is the fundus image (of height H and width W) acquired at time ti, j. Similar to the formulation of DynamicDeepHit33, we distinguish between discrete time steps j and actual elapsed times t since images are acquired at irregular intervals and the number of images per eye, Ji, is variable. In other words, Xi(t) represents the collection of longitudinal images of eye i acquired up until time t; for shorthand, we use Xi to denote the full available sequence of longitudinal images for eye i (i.e., [Math Processing Error]
). For each Xi, we also have a time to event τi, which is either the time at which the event occurred (e.g., eye developed late AMD – denoted ci = 0) or the censoring time (e.g., the patient was lost to follow-up or the study ended – denoted ci = 1).

The goal of deep survival analysis in longitudinal imaging is to approximate a function that links the time to event to our time-varying image measurements. A typical way to reason about the time to event is through the hazard function.

The conditional probability that eye i develops the disease at a discrete time step j, based on longitudinal measurements Xi, given that the true event time step is greater than or equal to j.

The probability that the eye i does not develop the disease (“survives”) past the time step j.

Specifically, in this study, we train a neural network f(∙) to directly map from a longitudinal imaging sequence to the discrete hazard distribution

Where Jmax is the total number of discrete time steps, typically chosen based on properties of the dataset, time-to-event task, and computational constraints. While hazards are computed for all time steps, including those that have already occurred before the time step t, our primary interest lies in the future hazards. As explained below, we mask out prior time steps to properly optimize and evaluate models. For models trained on AREDS data—captured in 6-month intervals with a maximum observed follow-up time of 13 years—we have set. For models trained on OHTS data—acquired in 1-year intervals with a maximum follow-up of 14 years—we have set
