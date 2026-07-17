# Longitudinal Transformer for Survival Analysis

`ltsa` provides a causal longitudinal image transformer, discrete survival losses, and a canonical Cox partial-likelihood loss. It generalizes the model proposed in [Harnessing the Power of Longitudinal Medical Imaging for Eye Disease Prognosis Using Transformer-based Sequence Modeling](https://www.nature.com/articles/s41746-024-01207-4).

## Usage

```python
import torch
from ltsa import LTSA, NLLSurvLoss, ResNetEncoder

encoder = ResNetEncoder(weights=None)
model = LTSA(
    encoder,
    n_heads=8,
    dropout=0.1,
    n_layers=1,
    max_sequence_length=4,
    max_time_index=36,
    n_time_bins=8,
)

images = torch.randn(2, 4, 3, 224, 224)
sequence_lengths = torch.tensor([4, 2])
relative_times = torch.tensor(
    [[0.0, 6.0, 18.0, 36.0], [0.0, 12.0, 0.0, 0.0]]
)
outputs = model(
    images,
    sequence_lengths=sequence_lengths,
    relative_times=relative_times,
)

last_visit = sequence_lengths - 1
hazards = outputs.hazards[torch.arange(2), last_visit]
survival = outputs.surv[torch.arange(2), last_visit]
loss = NLLSurvLoss(beta=0.15)(
    hazards,
    survival,
    labels=torch.tensor([2, 5]),
    censorship=torch.tensor([0, 1]),
)
loss.backward()
```

Inputs are shaped `[batch, visits, ...]`. `sequence_lengths` identifies valid visits and `relative_times` contains continuous elapsed times. `LTSAOutputs` includes boolean valid/next-visit masks and per-layer, per-head attention maps.

For proportional hazards models, use `CoxPHSurvLoss()(risk, time, event)` or `cox_ph_loss(risk, time, event)`. Tied event times use Breslow handling.

## Citation

Holste, Gregory, et al. "Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling." *npj Digital Medicine* 7.1 (2024): 216.
