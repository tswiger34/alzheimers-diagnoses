"""Reference replay diagnostics on synthetic images; no training or downloads."""

import copy
from dataclasses import replace

import pytest
import torch

from test_ltsa_stabilization import cached  # noqa: F401
from mstat_project.ml.ltsa_architecture_ablation import AblationModel, EXPERIMENTS, validation_components
from mstat_project.ml.ltsa_reference_replay import replay_batches
from mstat_project.ml.survival import SurvivalConfig, _loader, validation_loss


def loaders(cached, split="validation"):
    config = SurvivalConfig(cache_path=cached.path, spatial_size=(16, 16, 16),
                            device="cpu", batch_size=3, num_workers=0)
    records = [r for r in cached.records if r.split == split]
    return (_loader(records, replace(config, cache_path=None), "a", 1),
            _loader(records, config, "a", 1))


def test_replay_matches_cache_preserves_encoder_and_loss(cached):
    cached.encoder.train()
    flags = [module.training for module in cached.encoder.modules()]
    weights = copy.deepcopy(cached.encoder.state_dict())
    batches, report = replay_batches(cached.encoder, *loaders(cached), "cpu")
    assert report["observations"] == sum(r.split == "validation" for r in cached.records)
    assert [module.training for module in cached.encoder.modules()] == flags
    for name, value in weights.items():
        torch.testing.assert_close(cached.encoder.state_dict()[name], value, rtol=0, atol=0)
    for comparison in report["summary"].values():
        assert comparison["max_abs"] < 1e-5
    model = AblationModel(cached.manifest, "b", EXPERIMENTS["legacy_all"], .003)
    scores = [validation_loss(model, batches[name], "cpu") for name in batches]
    assert max(scores) - min(scores) < 1e-5
    components = validation_components(model, batches["raw_batch"], "cpu")
    assert components["overall"]["nll"] == pytest.approx(scores[0], abs=1e-6)


def test_replay_rejects_test_and_mismatched_metadata(cached):
    with pytest.raises(ValueError, match="identical validation"):
        replay_batches(cached.encoder, *loaders(cached, "test"), "cpu")
    raw, cache = loaders(cached)
    raw_batch, cache_batch = next(iter(raw)), next(iter(cache))
    cache_batch["times"] += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        replay_batches(cached.encoder, [raw_batch], [cache_batch], "cpu")


def test_replay_detects_feature_discrepancy_without_overwriting_cache(cached):
    raw, cache = loaders(cached)
    raw_batch, cache_batch = next(iter(raw)), next(iter(cache))
    cache_batch["embeddings"] = cache_batch["embeddings"] + .1
    batches, report = replay_batches(cached.encoder, [raw_batch], [cache_batch], "cpu")
    assert report["summary"]["singleton_minus_cache"]["max_abs"] == pytest.approx(.1, abs=1e-6)
    assert report["summary"]["batch_minus_singleton"]["max_abs"] < 1e-5
    torch.testing.assert_close(batches["cached"][0]["embeddings"], cache_batch["embeddings"])
