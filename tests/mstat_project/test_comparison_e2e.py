from pathlib import Path
from typing import Any, ClassVar

import torch

import mstat_project.ml.compare as compare_module
import mstat_project.ml.landmark_baseline as baseline_module
import mstat_project.ml.ltsa as ltsa_module
from mstat_project.ml.compare import ComparisonConfig, run_comparison
from mstat_project.ml.landmarks import LandmarkPatientRecord


class TinyImageEncoder(torch.nn.Module):
    n_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(1, self.n_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.flatten(1)[:, :1])


class RecordingResultStore:
    calls: ClassVar[list[tuple[str, tuple[Any, ...], dict[str, Any]]]] = []

    def __init__(self, _engine: object) -> None:
        pass

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def ensure_schema(self) -> None:
        self._record("ensure_schema")

    def start_run(self, *args: Any, **kwargs: Any) -> None:
        self._record("start_run", *args, **kwargs)

    def record_epoch(self, *args: Any, **kwargs: Any) -> None:
        self._record("record_epoch", *args, **kwargs)

    def record_predictions(self, *args: Any, **kwargs: Any) -> None:
        self._record("record_predictions", *args, **kwargs)

    def complete_run(self, *args: Any, **kwargs: Any) -> None:
        self._record("complete_run", *args, **kwargs)

    def record_comparison(self, *args: Any, **kwargs: Any) -> None:
        self._record("record_comparison", *args, **kwargs)

    def fail_run(self, *args: Any, **kwargs: Any) -> None:
        self._record("fail_run", *args, **kwargs)


def _records(tmp_path: Path) -> list[LandmarkPatientRecord]:
    records: list[LandmarkPatientRecord] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for patient_index, (duration, event) in enumerate(((6.0, True), (12.0, False))):
            ptid = f"P{split_index}{patient_index}"
            image_ids = (f"{ptid}-0", f"{ptid}-1")
            tensor_path = tmp_path / f"{ptid}.pt"
            generator = torch.Generator().manual_seed(split_index * 10 + patient_index)
            torch.save(
                {
                    "ptid": ptid,
                    "img_ids": list(image_ids),
                    "images": torch.randn(2, 1, 16, 16, 16, generator=generator),
                },
                tensor_path,
            )
            records.append(
                LandmarkPatientRecord(
                    ptid=ptid,
                    split=split,
                    landmark_months=12.0,
                    observed_time_months=duration,
                    event_observed=event,
                    image_ids=image_ids,
                    relative_times=(0.0, 12.0),
                    tensor_path=tensor_path,
                )
            )
    return records


def test_tiny_paired_run_trains_checkpoints_and_persists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = _records(tmp_path)
    RecordingResultStore.calls.clear()

    def make_encoder(*, weights=None) -> TinyImageEncoder:
        return TinyImageEncoder()

    monkeypatch.setattr(ltsa_module, "OrthogonalSliceResNet101Encoder", make_encoder)
    monkeypatch.setattr(baseline_module, "OrthogonalSliceResNet101Encoder", make_encoder)
    monkeypatch.setattr(ltsa_module, "SurvivalResultStore", RecordingResultStore)
    monkeypatch.setattr(baseline_module, "SurvivalResultStore", RecordingResultStore)
    monkeypatch.setattr(compare_module, "SurvivalResultStore", RecordingResultStore)
    monkeypatch.setattr(
        compare_module,
        "preflight_comparison_cohorts",
        lambda _config, _engine: {12.0: records},
    )

    results = run_comparison(
        ComparisonConfig(
            landmarks_months=(12.0,),
            epochs=2,
            patience=1,
            batch_size=2,
            bootstrap_samples=50,
            n_heads=2,
            pretrained=False,
            tensor_dir=tmp_path,
            checkpoint_root=tmp_path / "checkpoints",
            spatial_size=(16, 16, 16),
        ),
        engine=object(),  # type: ignore[arg-type]
    )

    assert len(results) == 1
    assert results[0].baseline.predictions.ptids == results[0].ltsa.predictions.ptids
    assert torch.isfinite(torch.tensor(results[0].comparison.difference))
    assert len(list((tmp_path / "checkpoints").rglob("epoch_*.pt"))) >= 2

    call_names = [name for name, _, _ in RecordingResultStore.calls]
    assert call_names.count("start_run") == 2
    assert call_names.count("record_predictions") == 2
    assert call_names.count("complete_run") == 2
    assert call_names.count("record_comparison") == 1
    assert "fail_run" not in call_names
