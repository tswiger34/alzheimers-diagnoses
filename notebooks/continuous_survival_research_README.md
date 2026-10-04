# Continuous-survival research figures

Open these notebooks using the project's existing `.venv` Python kernel and run
all cells in order. They are intentionally delivered without executed outputs.

- `continuous_survival_research_figures.ipynb`: six slide-oriented figures covering
  the scientific comparison, risk-set/event support, training/selection,
  discrimination, paired incremental discrimination, and available MRI history.
- `continuous_survival_research_appendix.ipynb`: three supporting figures covering
  full event/censoring support, the full A/B difference grid, and baseline prediction
  agreement, plus complete numerical tables.

Both notebooks run independently. They use NumPy, pandas, Matplotlib, and IPython,
already present in the project's runtime/development dependency lists. Select the
project environment when choosing the kernel. No package installation is performed.

## Sources and exports

The default source is the completed run
`611be867-cfda-40a2-8dc2-04fa5b2e5046`, with frozen image encoder and batch size 6.
Only saved JSON artifacts are read. There is no Torch import, MRI loading,
checkpoint loading, database querying, or model inference.

Exports are written when the notebook runs to:

```text
data/artifacts/research_figures/611be867-cfda-40a2-8dc2-04fa5b2e5046/
```

Each figure has a 300-dpi PNG, vector PDF, and SVG. Summary CSVs retain exact values,
event counts, comparable-pair counts, usable bootstrap replicates, and intervals.
`figure_provenance.json` records source hashes, protocol settings, plotting scope,
library versions, and bootstrap settings. Patient identifiers are not exported.

## Main configuration

The setup cell exposes `RUN_ID`, `EXPORT_DIR`, `DISPLAY_LANDMARKS`,
`FOCUS_LANDMARKS`, `MIN_EVENTS_FLAG`, `BOOTSTRAP_SAMPLES`, `BOOTSTRAP_SEED`, and
`EXPORT_DPI`.

The default 1,000 bootstrap samples compute new pointwise 95% percentile intervals
from saved predictions. These intervals were not present in the training run.
Resampling is paired by patient at each landmark. Setting `BOOTSTRAP_SAMPLES = 0`
produces point-estimate-only figures. Comparisons first reconcile with every saved
C-index and difference; pairing or numerical disagreements stop execution.

Bootstrap intervals condition on the trained models and split, omit training-seed
uncertainty, and are not adjusted for multiple comparisons. Sparse strata may
produce many unusable replicates. These counts remain in the exported tables.

## Research scope

A uses the latest available MRI; B adds its acquisition time. Neither incorporates
longitudinal MRI history. Current results characterize baselines and cannot answer
whether history adds prognostic information. A matched three-arm run is needed.
When `RUN_ID` points to such a run, the main paired-difference figure switches to
history minus B, and the discrimination/training figures include history.

The selected landmarks are exploratory display choices, not a prespecified
confirmatory analysis. C-indices are horizon-truncated Harrell estimates, not IPCW.
No calibration claim is made. Risk-set decline includes AD events and censoring;
it does not imply that patients lack an MRI at the scheduled visit.

## Feedback and validation

Notebook schema and Python syntax are checked at creation. Cells and figure
rendering have not been executed or visually checked, as requested. After running,
inspect every export at slide size for readability, clipping, and overlapping
labels. Provide notebook errors or figure feedback before using the graphics in
a presentation.

The authoring script `analysis/create_survival_research_notebooks.py` can regenerate
the clean notebooks. It overwrites their cells and clears outputs; keep your
executed copies if you want to preserve results or edits.
