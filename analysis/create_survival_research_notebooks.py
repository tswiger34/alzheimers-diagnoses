"""Author unexecuted research notebooks; validate structure and Python syntax only."""

import ast
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


SETUP = r'''
from pathlib import Path
import hashlib
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from IPython.display import display

# Find the repository from either its root or the notebooks directory.
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "src/mstat_project/ml/survival.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook from the repository root or notebooks directory.")
RUN_ID = "611be867-cfda-40a2-8dc2-04fa5b2e5046"
RUN_DIR = ROOT / "data/artifacts/model_checkpoints/continuous" / RUN_ID
EXPORT_DIR = ROOT / "data/artifacts/research_figures" / RUN_ID

# These are exploratory display choices, not retrospectively prespecified endpoints.
DISPLAY_LANDMARKS = (0., 6., 12., 24., 36., 48., 60.)
FOCUS_LANDMARKS = (0., 12., 24., 36.)
MIN_EVENTS_FLAG = 5  # Transparency flag; not a statistical reliability guarantee.
BOOTSTRAP_SAMPLES = 1000  # Set to 0 for a quick point-estimate-only pass.
BOOTSTRAP_SEED = 42
EXPORT_DPI = 300

COLORS = {"a": "#0072B2", "b": "#D55E00", "history": "#6F4C9B"}
LABELS = {"a": "A · latest MRI", "b": "B · latest MRI + acquisition time",
          "history": "History · longitudinal MRI"}
SHORT_LABELS = {"a": "A", "b": "B", "history": "History"}
MARKERS = {"a": "o", "b": "s", "history": "^"}
LINESTYLES = {"a": "-", "b": "--", "history": "-."}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 16,
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.figsize": (12.8, 7.2), "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#56616B", "text.color": "#18232D",
    "axes.labelcolor": "#18232D", "grid.color": "#DCE1E5", "grid.linewidth": .6,
    "lines.linewidth": 2, "lines.markersize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)

def finish_figure(fig, name, title, note, *, legend=False):
    """Export vector PDF/SVG plus a high-resolution PNG, with source context."""
    fig.suptitle(title, x=.07, y=.97, ha="left", fontsize=20, fontweight="bold")
    fig.subplots_adjust(left=.09, right=.96, top=.79 if legend else .85, bottom=.20, wspace=.35)
    fig.text(.07, .045, textwrap.fill(note, width=145), fontsize=10, color="#52616D", va="bottom")
    fig.text(.96, .014, f"Run {RUN_ID[:8]} · held-out test cohort unless labeled otherwise",
             fontsize=9, ha="right", color="#52616D")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(EXPORT_DIR / f"{name}.{extension}", dpi=EXPORT_DPI, facecolor="white")
    plt.show()
    plt.close(fig)

def model_legend(fig, variants):
    handles = [Line2D([], [], color=COLORS[v], marker=MARKERS[v], linestyle=LINESTYLES[v],
                      label=LABELS[v]) for v in variants]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065, .91), frameon=False,
               ncol=2 if len(variants) == 2 else 3, fontsize=11)

print(f"Sources: {RUN_DIR}\nExports: {EXPORT_DIR}")
print("No Torch, model checkpoint, MRI tensor, database, or GPU access is used.")
'''

LOAD = r'''
manifest = read_json(RUN_DIR / "manifest.json")
status = read_json(RUN_DIR / "status.json")
if status.get("status") != "completed":
    raise ValueError("Choose a completed run so all model arms use a finalized snapshot.")
variants = [v for v in ("a", "b", "history") if (RUN_DIR / v / "summary.json").exists()]
if not {"a", "b"}.issubset(variants):
    raise ValueError("This presentation expects both latest-MRI baselines A and B.")
horizons = tuple(float(h) for h in manifest["config"]["horizons"])
summaries = {v: read_json(RUN_DIR / v / "summary.json") for v in variants}
epochs = {v: pd.DataFrame(read_json(RUN_DIR / v / "epochs.json")) for v in variants}
raw_predictions = {v: read_json(RUN_DIR / v / "test_predictions.json") for v in variants}
saved_comparisons = pd.DataFrame(read_json(RUN_DIR / "comparisons.json"))

indexed = {}
for variant, rows in raw_predictions.items():
    keyed = {}
    for row in rows:
        key = (float(row["landmark_months"]), row["ptid"])
        if key in keyed:
            raise ValueError(f"Duplicate patient/landmark record in {variant}.")
        curve = np.asarray(row["survival"], dtype=float)
        assert row["split"] == "test" and row["outcome_months"] > key[0]
        assert len(curve) == len(horizons) and np.isfinite(curve).all()
        assert ((curve >= 0) & (curve <= 1)).all() and (np.diff(curve) <= 1e-7).all()
        assert row["acquisition_months"] <= key[0]
        assert len(row["available_history_ids"]) >= 1
        keyed[key] = row
    indexed[variant] = keyed

reference = indexed["a"]
pairing_fields = ("split", "outcome_months", "event", "acquisition_months", "available_history_ids", "visit_nums")
for variant in variants:
    assert indexed[variant].keys() == reference.keys(), f"Unmatched cohort: {variant}"
    for key, left in reference.items():
        right = indexed[variant][key]
        assert all(left[field] == right[field] for field in pairing_fields), f"Unmatched endpoint/history: {variant}"
        if variant in ("a", "b"):
            assert right["image_ids"] == left["image_ids"], "A/B must receive the same latest image."

landmarks = sorted({key[0] for key in reference})
groups = {}
for landmark in landmarks:
    keys = sorted(key for key in reference if key[0] == landmark)
    rows = [reference[key] for key in keys]
    groups[landmark] = {
        "times": np.array([r["outcome_months"] - landmark for r in rows]),
        "events": np.array([r["event"] for r in rows], dtype=bool),
        "risks": {v: 1 - np.array([indexed[v][key]["survival"] for key in keys]) for v in variants},
        "history_count": np.array([len(r["available_history_ids"]) for r in rows]),
        "image_age": np.array([landmark - r["acquisition_months"] for r in rows]),
    }
configuration = manifest["config"]
display(pd.DataFrame([
    {"Model": LABELS[v], "Selected epoch": summaries[v]["best_epoch"],
     "Validation NLL": summaries[v]["validation_nll"], "Epochs completed": len(epochs[v])}
    for v in variants
]).round(4))
print(f"Patients by split: { {s: x['patients'] for s, x in manifest['audit']['splits'].items()} }")
print(f"Frozen encoder: {configuration.get('freeze_encoder', False)}; batch size: {configuration['batch_size']}")
print("History arm available:", "history" in variants)
'''

METRICS = r'''
def comparable_pairs(times, events, horizon):
    """Match the repository's horizon-truncated Harrell C-index exactly."""
    observed = np.minimum(times, horizon)
    observed_events = events & (times <= horizon)
    earlier, later = np.where(observed_events[:, None] & (observed[:, None] < observed[None, :]))
    return earlier, later, observed_events

def pair_scores(risk, earlier, later):
    difference = risk[earlier] - risk[later]
    return (difference > 0).astype(float) + .5 * (difference == 0)

metric_rows, support_rows = [], []
for landmark, group in groups.items():
    for j, horizon in enumerate(horizons):
        earlier, later, observed_events = comparable_pairs(group["times"], group["events"], horizon)
        support = {"landmark": landmark, "horizon": horizon, "patients": len(group["times"]),
                   "events": int(observed_events.sum()), "comparable_pairs": len(earlier),
                   "censored_before_horizon": int((~group["events"] & (group["times"] < horizon)).sum())}
        support_rows.append(support)
        for variant in variants:
            score = pair_scores(group["risks"][variant][:, j], earlier, later)
            c_index = float(score.mean()) if len(score) else np.nan
            saved = summaries[variant]["test_by_landmark"][str(landmark)]
            saved_c = saved["c_index"][j]
            assert (saved_c is None and np.isnan(c_index)) or np.isclose(c_index, saved_c, atol=1e-12)
            assert saved["patients"] == support["patients"]
            prediction_nll = np.mean([p["nll"] for p in raw_predictions[variant]
                                      if p["landmark_months"] == landmark])
            assert np.isclose(prediction_nll, saved["nll"], atol=1e-10)
            metric_rows.append({**support, "variant": variant, "c_index": c_index,
                                "test_nll": saved["nll"]})
metrics = pd.DataFrame(metric_rows)
support = pd.DataFrame(support_rows)
pairings = [("a", "b")]
if "history" in variants:
    pairings.extend([("a", "history"), ("b", "history")])

delta_rows = []
for left, right in pairings:
    paired = metrics[metrics.variant == left].merge(metrics[metrics.variant == right],
                                                  on=["landmark", "horizon"], suffixes=("_left", "_right"))
    for _, row in paired.iterrows():
        delta = row.c_index_right - row.c_index_left
        saved = saved_comparisons[(saved_comparisons.reference == left) &
                                 (saved_comparisons.candidate == right) &
                                 (saved_comparisons.landmark_months == row.landmark) &
                                 (saved_comparisons.horizon_months == row.horizon)]
        assert len(saved) == 1
        saved_delta = saved.iloc[0]["difference"]
        assert (pd.isna(saved_delta) and np.isnan(delta)) or np.isclose(delta, saved_delta, atol=1e-12)
        assert int(saved.iloc[0].comparable_pairs) == int(row.comparable_pairs_left)
        delta_rows.append({"reference": left, "candidate": right, "landmark": row.landmark,
                           "horizon": row.horizon, "difference": delta})
deltas = pd.DataFrame(delta_rows)
for frame in (metrics, deltas):
    frame["ci_low"] = np.nan
    frame["ci_high"] = np.nan
    frame["bootstrap_used"] = 0
print("All saved C-indices, paired differences, patient counts, comparable-pair counts, and mean test NLLs reconciled.")
'''

BOOTSTRAP = r'''
# Patient bootstrap uses identical resampling weights for all models at a given
# landmark. Pair weights represent repeated patients exactly; pairs remain strict
# time comparisons. Never resample comparable pairs themselves or pool landmarks.
if BOOTSTRAP_SAMPLES:
    for landmark, group in groups.items():
        n = len(group["times"])
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, int(round(landmark * 1000))]))
        samples = {(v, j): [] for v in variants for j in range(len(horizons))}
        for offset in range(0, BOOTSTRAP_SAMPLES, 64):
            batch_size = min(64, BOOTSTRAP_SAMPLES - offset)
            weights = rng.multinomial(n, np.full(n, 1 / n), size=batch_size)
            for j, horizon in enumerate(horizons):
                earlier, later, _ = comparable_pairs(group["times"], group["events"], horizon)
                if not len(earlier):
                    continue
                pair_weights = weights[:, earlier] * weights[:, later]
                denominator = pair_weights.sum(axis=1)
                usable = denominator > 0
                for variant in variants:
                    scores = pair_scores(group["risks"][variant][:, j], earlier, later)
                    estimates = (pair_weights[usable] @ scores) / denominator[usable]
                    samples[variant, j].extend(estimates.tolist())
        for j, horizon in enumerate(horizons):
            for variant in variants:
                values = np.asarray(samples[variant, j])
                mask = (metrics.landmark == landmark) & (metrics.horizon == horizon) & (metrics.variant == variant)
                if len(values):
                    metrics.loc[mask, ["ci_low", "ci_high"]] = np.quantile(values, [.025, .975])
                    metrics.loc[mask, "bootstrap_used"] = len(values)
            for left, right in pairings:
                values = np.asarray(samples[right, j]) - np.asarray(samples[left, j])
                mask = ((deltas.landmark == landmark) & (deltas.horizon == horizon) &
                        (deltas.reference == left) & (deltas.candidate == right))
                if len(values):
                    deltas.loc[mask, ["ci_low", "ci_high"]] = np.quantile(values, [.025, .975])
                    deltas.loc[mask, "bootstrap_used"] = len(values)
    print("Computed pointwise 95% percentile intervals from paired patient resampling.")
else:
    print("Point estimates only: confidence intervals are omitted.")

INTERVAL_NOTE = (f"Pointwise 95% patient-bootstrap percentile intervals; {BOOTSTRAP_SAMPLES:,} requested replicates. "
                 "Replicates without comparable pairs are excluded; no multiplicity adjustment."
                 if BOOTSTRAP_SAMPLES else "Point estimates only; uncertainty intervals were not computed.")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
metrics.to_csv(EXPORT_DIR / "model_metrics.csv", index=False)
support.to_csv(EXPORT_DIR / "evaluation_support.csv", index=False)
deltas.to_csv(EXPORT_DIR / "paired_c_index_differences.csv", index=False)
sources = [RUN_DIR / "manifest.json", RUN_DIR / "comparisons.json"]
sources += [RUN_DIR / v / name for v in variants
            for name in ("summary.json", "epochs.json", "test_predictions.json")]
provenance = {"run_id": RUN_ID, "source_snapshot": manifest["hashes"], "config": configuration,
              "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
              "metric": "Harrell C-index after administrative censoring at each horizon; not IPCW",
              "intervals": INTERVAL_NOTE, "display_landmarks": DISPLAY_LANDMARKS,
              "focus_landmarks": FOCUS_LANDMARKS,
              "library_versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": mpl.__version__},
              "scope": "A/B latest-MRI comparison; history benefit requires a matched history arm."}
(EXPORT_DIR / "figure_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
'''

DESIGN = r'''
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots()
ax.set_axis_off()
rows = [("A", "Latest available MRI", "Image embedding", "Continuous hazard", COLORS["a"]),
        ("B", "Latest available MRI", "+ acquisition time", "Continuous hazard", COLORS["b"]),
        ("History", "All available prior MRIs", "Time-aware transformer", "Continuous hazard", COLORS["history"])]
for i, (name, input_text, middle, output_text, color) in enumerate(rows):
    y = .77 - i * .30
    ax.text(.005, y, name, va="center", fontsize=16, fontweight="bold", color=color, transform=ax.transAxes)
    for x, width, label in [(.13, .25, input_text), (.44, .26, middle), (.76, .23, output_text)]:
        pending = name == "History" and "history" not in variants
        ax.add_patch(FancyBboxPatch((x, y - .075), width, .15, boxstyle="round,pad=.008",
                                   edgecolor=color, facecolor="white", linewidth=1.5,
                                   linestyle="--" if pending else "-", transform=ax.transAxes))
        ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=12, transform=ax.transAxes)
    for start, end in [(.39, .43), (.71, .75)]:
        ax.annotate("", xy=(end, y), xytext=(start, y), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#52616D", "lw": 1.5})
if "history" not in variants:
    ax.text(.13, -.03, "History arm: required next comparison; no history results in this run",
            transform=ax.transAxes, fontsize=12, color=COLORS["history"])
finish_figure(fig, "01_research_design", "Does MRI history add prognostic information?",
              "Current A/B results establish latest-MRI baselines. The primary contrast is history minus B; "
              "history minus A is complementary. All comparisons require identical patients, endpoints, and latest available MRI.")
'''

COHORT = r'''
fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
cohort_rows = [{"landmark": t, "patients": len(g["times"]), "events": int(g["events"].sum())}
               for t, g in groups.items()]
cohort = pd.DataFrame(cohort_rows)
axes[0].plot(cohort.landmark, cohort.patients, color=COLORS["a"], marker="o")
axes[0].set(xlabel="Prediction landmark (months)", ylabel="Patients still at risk",
            title="Clinical risk set", ylim=(0, cohort.patients.max() * 1.1), xlim=(0, max(landmarks)))
axes[0].set_xticks([0, 36, 72, 108, 144, 180])
axes[0].grid(axis="y")
for t in (0., 36., 180.):
    if t in groups:
        count = len(groups[t]["times"])
        axes[0].annotate(str(count), (t, count), xytext=(0, 10), textcoords="offset points", ha="center")
counts = support[support.landmark.isin(DISPLAY_LANDMARKS)].pivot(index="landmark", columns="horizon", values="events")
axes[1].imshow(counts.to_numpy(), cmap="Blues", vmin=0, vmax=max(1, counts.to_numpy().max()), aspect="auto")
axes[1].set_xticks(range(len(horizons)), [f"{h:g}" for h in horizons])
axes[1].set_yticks(range(len(counts)), [f"{t:g}" for t in counts.index])
axes[1].set(xlabel="Prediction horizon (months ahead)", ylabel="Prediction landmark (months)",
            title="Observed AD events within horizon")
for i in range(len(counts)):
    for j in range(len(horizons)):
        count = int(counts.iloc[i, j])
        axes[1].text(j, i, str(count), ha="center", va="center", fontsize=14,
                     color="white" if count > counts.to_numpy().max() * .55 else "#18232D")
        if count < MIN_EVENTS_FLAG:
            axes[1].add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch="///", edgecolor="#697680", linewidth=0))
finish_figure(fig, "02_cohort_and_event_support", "Interpret model performance alongside event support",
              f"Risk-set decline combines AD events and censoring; it is not a count of missing MRIs. "
              f"An earlier MRI may be reused. Hatching flags fewer than {MIN_EVENTS_FLAG} events; denominators differ by landmark.")
'''

TRAINING = r'''
fig, axes = plt.subplots(1, 2)
for variant in variants:
    history = epochs[variant]
    for ax, column in zip(axes, ("train_nll", "validation_nll"), strict=True):
        ax.plot(history.epoch, history[column], color=COLORS[variant], marker=MARKERS[variant],
                linestyle=LINESTYLES[variant], label=LABELS[variant])
    best = summaries[variant]["best_epoch"]
    row = history.loc[history.epoch == best].iloc[0]
    axes[1].scatter(best, row.validation_nll, s=160, facecolors="none", edgecolors=COLORS[variant], linewidth=1.5)
    axes[1].annotate(f"{SHORT_LABELS[variant]}: epoch {best}", (best, row.validation_nll),
                     xytext=(6, 10 if variant == "a" else -20), textcoords="offset points", fontsize=11)
for ax, title in zip(axes, ("Training objective", "Checkpoint selection on validation"), strict=True):
    ax.set(xlabel="Epoch", ylabel="Conditional NLL (lower is better)", title=title)
    ax.grid(axis="y")
    ax.set_xlim(.5, max(len(e) for e in epochs.values()) + .5)
model_legend(fig, variants)
finish_figure(fig, "03_training_and_selection", "Training and validation trajectories", 
              "Training uses one sampled eligible MRI per patient per epoch. Validation averages MRI losses within patients. "
              "Open rings mark selected epochs; test scores do not select checkpoints. Training/validation objectives have different observation weighting.",
              legend=True)
'''

DISCRIMINATION = r'''
fig, axes = plt.subplots(1, len(horizons), squeeze=False)
for ax, horizon in zip(axes[0], horizons, strict=True):
    for variant in variants:
        rows = metrics[(metrics.variant == variant) & (metrics.horizon == horizon) &
                       metrics.landmark.isin(DISPLAY_LANDMARKS)].sort_values("landmark")
        ax.plot(rows.landmark, rows.c_index, color=COLORS[variant], linestyle=LINESTYLES[variant], alpha=.75)
        if BOOTSTRAP_SAMPLES:
            ax.fill_between(rows.landmark.to_numpy(), rows.ci_low.to_numpy(), rows.ci_high.to_numpy(),
                            color=COLORS[variant], alpha=.10, linewidth=0)
        for _, row in rows.iterrows():
            ax.plot(row.landmark, row.c_index, marker=MARKERS[variant], linestyle="none",
                    color=COLORS[variant], markerfacecolor="white" if row.events < MIN_EVENTS_FLAG else COLORS[variant])
    ax.axhline(.5, color="#52616D", linestyle=":", linewidth=1)
    ax.set(title=f"Next {horizon:g} months", xlabel="Prediction landmark (months)", ylim=(0, 1))
    ax.set_xticks([t for t in (0, 12, 24, 36, 48, 60) if t in DISPLAY_LANDMARKS])
    ax.grid(axis="y")
axes[0, 0].set_ylabel("Horizon-truncated Harrell C-index")
model_legend(fig, variants)
finish_figure(fig, "04_discrimination_by_landmark", "Discrimination of the latest-MRI baselines",
              f"{INTERVAL_NOTE} Hollow markers flag fewer than {MIN_EVENTS_FLAG} events. "
              "The 0.5 reference denotes chance ranking. This metric is not IPCW and does not assess calibration.", legend=True)
'''

DELTAS = r'''
# If a matched history arm is added, use the primary history-minus-B contrast.
left, right = ("b", "history") if "history" in variants else ("a", "b")
contrast_label = f"{SHORT_LABELS[right]} − {SHORT_LABELS[left]}"
fig, axes = plt.subplots(1, len(horizons), squeeze=False)
contrast = deltas[(deltas.reference == left) & (deltas.candidate == right)]
visible = contrast[contrast.landmark.isin(FOCUS_LANDMARKS)]
extent_values = np.concatenate([visible.difference.dropna(), visible.ci_low.dropna(), visible.ci_high.dropna()])
limit = max(.03, np.max(np.abs(extent_values)) * 1.25) if len(extent_values) else .1
for ax, horizon in zip(axes[0], horizons, strict=True):
    rows = visible[visible.horizon == horizon].set_index("landmark")
    for y, landmark in enumerate(FOCUS_LANDMARKS):
        if landmark not in rows.index:
            continue
        row = rows.loc[landmark]
        if pd.isna(row.difference):
            ax.text(0, y, "Not estimable", ha="center", fontsize=10)
            continue
        if pd.notna(row.ci_low):
            ax.hlines(y, row.ci_low, row.ci_high, color=COLORS[right], linewidth=1.8)
            ax.vlines([row.ci_low, row.ci_high], y-.07, y+.07, color=COLORS[right], linewidth=1)
        ax.plot(row.difference, y, marker=MARKERS[right], color=COLORS[right], linestyle="none")
        ax.annotate(f"{row.difference:+.3f}", (row.difference, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.axvline(0, color="#52616D", linestyle=":", linewidth=1)
    ax.set(title=f"Next {horizon:g} months", xlim=(-limit, limit), ylim=(len(FOCUS_LANDMARKS)-.5, -.6),
           xlabel=f"Δ C-index ({contrast_label})")
    ax.set_yticks(range(len(FOCUS_LANDMARKS)), [f"{t:g}" for t in FOCUS_LANDMARKS])
    ax.grid(axis="x")
axes[0, 0].set_ylabel("Prediction landmark (months)")
finish_figure(fig, "05_paired_discrimination_difference", f"Incremental discrimination: {contrast_label}",
              f"{INTERVAL_NOTE} Positive values favor {SHORT_LABELS[right]}. Focused delta axis; zero is shown. "
              "A/B tests acquisition-time information, not longitudinal MRI history. Displayed endpoints are exploratory."
              if right == "b" else f"{INTERVAL_NOTE} Positive values favor history; endpoints are exploratory.")
'''

BASELINE = r'''
display_table = metrics[(metrics.landmark == 0)].copy()
display_table["Model"] = display_table.variant.map(LABELS)
display_table["Horizon (months)"] = display_table.horizon.astype(int)
display_table = display_table.rename(columns={"c_index": "C-index", "ci_low": "95% CI low",
    "ci_high": "95% CI high", "events": "AD events within horizon", "patients": "Patients",
    "bootstrap_used": "Usable bootstrap replicates"})
display(display_table[["Model", "Horizon (months)", "C-index", "95% CI low", "95% CI high",
                       "Patients", "AD events within horizon", "Usable bootstrap replicates"]].round(3))
print("Use these estimates with their event counts and intervals; absence of a detected A/B difference does not establish equivalence.")
'''

HISTORY_AVAILABILITY = r'''
fig, axes = plt.subplots(1, 2)
selected = [t for t in DISPLAY_LANDMARKS if t in groups]
fractions = np.array([[np.mean(groups[t]["history_count"] == 1),
                       np.mean(groups[t]["history_count"] == 2),
                       np.mean(groups[t]["history_count"] >= 3)] for t in selected])
bottom = np.zeros(len(selected))
for j, (label, color, hatch) in enumerate([("1 MRI", "#C9D1D8", ""), ("2 MRIs", COLORS["a"], "//"),
                                         ("3+ MRIs", COLORS["history"], "xx")]):
    axes[0].bar(range(len(selected)), fractions[:, j] * 100, bottom=bottom, label=label,
                color=color, edgecolor="white", hatch=hatch, linewidth=.7)
    bottom += fractions[:, j] * 100
axes[0].set_xticks(range(len(selected)), [f"{t:g}" for t in selected])
axes[0].set(title="Available MRI history", xlabel="Prediction landmark (months)",
            ylabel="Patients within landmark (%)", ylim=(0, 100))
axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=3, frameon=False, fontsize=10)
boxes = axes[1].boxplot([groups[t]["image_age"] for t in selected], positions=np.arange(len(selected)),
                        widths=.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "#18232D", "linewidth": 1.5})
for box in boxes["boxes"]:
    box.set_facecolor("#D9EAF4")
axes[1].set_xticks(range(len(selected)), [f"{t:g}" for t in selected])
axes[1].set(title="Age of latest available MRI", xlabel="Prediction landmark (months)", ylabel="Months since latest MRI")
axes[1].grid(axis="y")
finish_figure(fig, "06_history_availability_and_image_age", "Where can longitudinal history add information?",
              "Histories count retained eligible MRIs available by each landmark, not all raw acquisitions. "
              "Each landmark has a different at-risk cohort. Boxes show median and IQR; whiskers extend to 1.5×IQR; outliers omitted. "
              "Availability describes an opportunity for a history model, not evidence of benefit.")
'''

FULL_SUPPORT = r'''
fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
counts = support.pivot(index="landmark", columns="horizon", values="events")
unobserved = support.pivot(index="landmark", columns="horizon", values="censored_before_horizon")
for ax, frame, title, palette in [(axes[0], counts, "AD events within horizon", "Blues"),
                                  (axes[1], unobserved, "Censored before horizon", "Greys")]:
    ax.imshow(frame.to_numpy(), cmap=palette, vmin=0, vmax=max(1, frame.to_numpy().max()), aspect="auto")
    ax.set_xticks(range(len(horizons)), [f"{h:g}" for h in horizons])
    ax.set_yticks(range(len(frame)), [f"{t:g}" for t in frame.index])
    ax.set(title=title, xlabel="Prediction horizon (months ahead)")
    for i in range(len(frame)):
        for j in range(len(horizons)):
            value = int(frame.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10,
                    color="white" if value > frame.to_numpy().max() * .55 else "#18232D")
axes[0].set_ylabel("Prediction landmark (months)")
fig.set_size_inches(12.8, 9)
finish_figure(fig, "appendix_01_full_event_and_censoring_support", "Follow-up support across all evaluated landmarks",
              "Event counts refer to the stated future horizon. Censored patients may have unknown status by that horizon; "
              "they are not counted as event-free outcomes. These two counts do not sum to the landmark cohort size.")
'''

FULL_DELTA = r'''
frame = deltas[(deltas.reference == "a") & (deltas.candidate == "b")].pivot(
    index="landmark", columns="horizon", values="difference")
events_grid = support.pivot(index="landmark", columns="horizon", values="events").reindex(frame.index)
finite = frame.to_numpy()[np.isfinite(frame.to_numpy())]
limit = max(.03, np.max(np.abs(finite))) if len(finite) else .1
fig, ax = plt.subplots(figsize=(10, 9))
cmap = plt.get_cmap("PuOr").copy()
cmap.set_bad("#E5E8EB")
image = ax.imshow(np.ma.masked_invalid(frame.to_numpy()), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(len(horizons)), [f"{h:g}" for h in horizons])
ax.set_yticks(range(len(frame)), [f"{t:g}" for t in frame.index])
ax.set(xlabel="Prediction horizon (months ahead)", ylabel="Prediction landmark (months)")
for i in range(len(frame)):
    for j in range(len(horizons)):
        value = frame.iloc[i, j]
        ax.text(j, i, "NE" if pd.isna(value) else f"{value:+.3f}", ha="center", va="center", fontsize=10,
                color="white" if pd.notna(value) and abs(value) > limit*.65 else "#18232D")
        if events_grid.iloc[i, j] < MIN_EVENTS_FLAG and pd.notna(value):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch="///", edgecolor="#697680", linewidth=0))
fig.colorbar(image, ax=ax, fraction=.055, pad=.035, label="Δ C-index (B − A)")
finish_figure(fig, "appendix_02_full_paired_difference_grid", "A/B differences across the complete evaluation grid",
              f"Point estimates; full grid is exploratory and not multiplicity-adjusted. NE = no comparable pairs, not zero difference. "
              f"Hatching flags fewer than {MIN_EVENTS_FLAG} events. Positive values favor B. See paired interval table for uncertainty.")
'''

AGREEMENT = r'''
fig, axes = plt.subplots(1, len(horizons), squeeze=False)
baseline = groups[0.]
agreement_rows = []
for j, (ax, horizon) in enumerate(zip(axes[0], horizons, strict=True)):
    a_risk, b_risk = baseline["risks"]["a"][:, j], baseline["risks"]["b"][:, j]
    ax.scatter(a_risk * 100, b_risk * 100, s=18, color=COLORS["a"], alpha=.50, edgecolors="none")
    ax.plot([0, 100], [0, 100], linestyle=":", color="#52616D", linewidth=1)
    ax.set(title=f"Next {horizon:g} months", xlabel="A predicted AD risk (%)",
           xlim=(0, 100), ylim=(0, 100), aspect="equal")
    agreement_rows.append({"horizon": horizon, "patients": len(a_risk),
                           "mean_absolute_risk_difference_percentage_points": float(np.abs(a_risk-b_risk).mean()*100),
                           "median_absolute_risk_difference_percentage_points": float(np.median(np.abs(a_risk-b_risk))*100),
                           "A_risk_sd_percentage_points": float(a_risk.std()*100),
                           "B_risk_sd_percentage_points": float(b_risk.std()*100)})
axes[0, 0].set_ylabel("B predicted AD risk (%)")
finish_figure(fig, "appendix_03_baseline_prediction_agreement", "Do A and B assign similar baseline risks?",
              "Each point is one held-out patient at baseline. Diagonal denotes equal predicted risk. "
              "These are model probabilities; agreement and prediction spread do not demonstrate calibration or history benefit.")
agreement = pd.DataFrame(agreement_rows)
agreement.to_csv(EXPORT_DIR / "baseline_prediction_agreement.csv", index=False)
display(agreement.round(3))
'''

FINAL = r'''
print("Figure files:")
for path in sorted(EXPORT_DIR.glob("*.png")):
    print(" ", path.name)
print("\nTables: model_metrics.csv, evaluation_support.csv, paired_c_index_differences.csv")
print("Source hashes, settings, and scope: figure_provenance.json")
print("\nPresentation checklist:")
print("1. Inspect every PNG/PDF at slide size for clipping, readable labels, and confidence interval overlap.")
print("2. Include event counts and interval qualifications when presenting C-indices.")
print("3. Treat all A/B results as baseline characterization; reserve the history claim for a matched history arm.")
print("4. Choose primary landmarks/horizons before a future confirmatory history comparison.")
'''


def shared_cells():
    return [
        markdown("""
            ## Context & methods

            A uses the latest available MRI. B adds its actual acquisition time.
            Both use the same source snapshot and held-out patients. This run used a
            frozen ImageNet encoder; checkpoint selection used patient-mean validation
            NLL. Survival is conditional on remaining at risk at the prediction landmark.

            **Key assumptions:** independent patients; identical clinical endpoints;
            horizon-truncated Harrell concordance with half credit for exact risk ties.
            Censoring affects comparable pairs. This is not IPCW concordance and does
            not establish calibration. Sparse late landmarks are exploratory.

            The main scientific estimand is **history − B**: improvement beyond latest
            MRI plus acquisition time. **History − A** is complementary. A/B alone cannot
            answer whether longitudinal MRI history improves prognosis. A future history
            comparison must use a single matched three-arm run; do not combine separately
            trained arms without checking all protocol and source alignment.

            These notebooks are intentionally unexecuted. Run all cells in the project
            `.venv` kernel (NumPy, pandas, Matplotlib, IPython). No new dependencies,
            checkpoints, image tensors, database queries, or model inference are needed.
            Outputs are generated only when you run the notebook.
        """),
        markdown("### 1. Configure the run, display choices, and export style"), code(SETUP),
        markdown("""
            ## Data

            Fixed source: `data/artifacts/model_checkpoints/continuous/611be867-cfda-40a2-8dc2-04fa5b2e5046`.
            Read `manifest.json`, `comparisons.json`, and each arm's `summary.json`,
            `epochs.json`, and `test_predictions.json`. Protocol definitions live in
            `docs/continuous_survival.md`; the source metric implementation is
            `src/mstat_project/ml/survival_comparison.py`.

            ### 2. Load saved artifacts and verify patient/endpoint pairing
        """), code(LOAD),
        markdown("### 3. Recompute concordance and reconcile it with every saved estimate"), code(METRICS),
        markdown("""
            ### 4. Uncertainty and reproducible exports

            The training run disabled bootstrap. The cell below can compute new
            **pointwise 95% percentile intervals** from saved predictions, without
            retraining. Patients are resampled jointly across arms at each landmark;
            repeated observations from different landmarks are never pooled as
            independent samples. Zero-comparable-pair replicates are excluded and
            their usable counts are exported. Intervals are conditional on fitted
            models and this split; they omit training/seed uncertainty and multiple
            comparison adjustment. Failure to detect an A/B difference is not an
            equivalence test.

            Set `BOOTSTRAP_SAMPLES = 0` for point estimates only. Input files are
            untouched; exports go into a separate run-specific research-figure directory.
        """), code(BOOTSTRAP),
    ]


def main():
    presentation = [markdown("""
        # MRI-based AD prognosis — latest-MRI baselines

        **Research question:** Does longitudinal MRI history improve AD prognosis
        beyond the information contained in the most recent MRI?

        This notebook prepares the completed A/B comparison for research slides and
        establishes the baselines for a subsequent matched history comparison.
        Every plot exports PNG (300 dpi), vector PDF, and editable SVG when run.

        ## Reading guide

        1. Scientific comparison and current evidence scope.
        2. Cohort and event support.
        3. Training and validation selection.
        4. Discrimination across landmarks and paired incremental differences.
        5. Available MRI history and age of the latest scan.

        **Execution status:** authored and syntax-checked, not executed or visually
        inspected. Interpretation should follow your rendered outputs.
    """)]
    presentation += shared_cells()
    presentation += [
        markdown("## Results\n\n### 5. Research comparison diagram"), code(DESIGN),
        markdown("### 6. Risk-set size and horizon-specific event support"), code(COHORT),
        markdown("### 7. Training behavior and validation-selected checkpoints"), code(TRAINING),
        markdown("### 8. Discrimination of the matched models"), code(DISCRIMINATION),
        markdown("""
            ### 9. Paired incremental discrimination

            The difference plot is the central comparison: joint patient resampling
            preserves the dependence between A and B scores. When a matched history
            arm exists in the selected run, this plot switches to **history − B**.
            Current display landmarks are exploratory, not prespecified study endpoints.
        """), code(DELTAS),
        markdown("### 10. Baseline estimates for exact-value slide annotations"), code(BASELINE),
        markdown("### 11. Longitudinal information available to a future history model"), code(HISTORY_AVAILABILITY),
        markdown("""
            ## Takeaways to write after execution

            - Describe A/B differences using the paired intervals and event support.
            - Inspect sparse horizons before discussing extreme C-indices.
            - Retain the distinction between discrimination and calibrated risk.
            - No A/B result establishes the benefit of longitudinal history; the
              decisive comparison requires a matched history arm and validation-based
              development, with primary endpoints chosen before confirmatory evaluation.
            - Freezing, batch size, and training duration differ from earlier collapsed
              runs. A historical before/after plot would not isolate freezing causally.
        """), code(FINAL),
    ]
    appendix = [markdown("""
        # MRI-based AD prognosis — evidence and reporting appendix

        Companion to `continuous_survival_research_figures.ipynb`.
        Full event/censoring support, all landmark/horizon A/B differences, baseline
        prediction agreement, and complete numerical reporting tables.

        This appendix runs independently. It reads saved JSON only, exports static
        PNG/PDF/SVG figures, and preserves non-estimable cells as missing.
        **Execution status:** authored and syntax-checked; not executed or visually inspected.
    """)]
    appendix += shared_cells()
    appendix += [
        markdown("## Results\n\n### 5. Event and censoring counts across the full follow-up grid"), code(FULL_SUPPORT),
        markdown("### 6. All paired A/B differences, including sparse and non-estimable cells"), code(FULL_DELTA),
        markdown("### 7. Baseline risk agreement and prediction spread"), code(AGREEMENT),
        markdown("### 8. Complete reporting tables — retain these alongside research figures"),
        code('''
            complete = metrics.merge(deltas[(deltas.reference == "a") & (deltas.candidate == "b")]
                                     [["landmark", "horizon", "difference"]], on=["landmark", "horizon"])
            complete = complete.sort_values(["landmark", "horizon", "variant"])
            display(complete.round(4))
            complete.to_csv(EXPORT_DIR / "complete_reporting_table.csv", index=False)
            display(deltas.round(4))
        '''),
        markdown("""
            ## Takeaways to write after execution

            The risk set changes by landmark, and later outcomes are often supported
            by very few events. Missing C-indices indicate no comparable event pairs.
            Extreme sparse estimates should not be promoted as robust findings.
            Risk agreement does not demonstrate accuracy, calibration, equivalence,
            or longitudinal history benefit. No patient identifiers are exported in
            the figures or aggregate reporting tables.
        """), code(FINAL),
    ]
    for filename, cells in [("continuous_survival_research_figures.ipynb", presentation),
                             ("continuous_survival_research_appendix.ipynb", appendix)]:
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata = {
            "kernelspec": {"display_name": "Python (project .venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "file_extension": ".py"},
            "research_artifact": {"run_id": "611be867-cfda-40a2-8dc2-04fa5b2e5046",
                                  "execution_status": "unexecuted_at_user_request"},
        }
        nbf.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.execution_count is None and cell.outputs == []
        target = ROOT / "notebooks" / filename
        nbf.write(notebook, target)
        print(f"Created {target.name}: {len(cells)} cells; schema and syntax valid; unexecuted.")


if __name__ == "__main__":
    main()
