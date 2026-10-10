"""Atomic artifact publication under transient and persistent file access errors."""

import errno
import json
from pathlib import Path

import pytest
import torch

from mstat_project.ml import survival


@pytest.fixture(params=["checkpoint", "json"])
def artifact_writer(request):
    if request.param == "checkpoint":
        return survival._save_checkpoint, lambda path: torch.load(path, weights_only=True), ".pt"
    return survival.write_json, lambda path: json.loads(path.read_text(encoding="utf-8")), ".json"


def test_artifact_retries_lock_without_rewriting_or_removing_destination(tmp_path, monkeypatch, artifact_writer):
    write, read, suffix = artifact_writer
    path = tmp_path / f"artifact{suffix}"
    write(path, {"value": "old"})
    previous = path.read_bytes()
    original = Path.replace
    attempts, sleeps = [], []

    def replace(temporary, destination):
        attempts.append(temporary)
        assert destination == path and path.read_bytes() == previous
        assert read(temporary) == {"value": "new"}
        if len(attempts) < 3:
            raise PermissionError(errno.EACCES, "simulated file lock")
        return original(temporary, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(survival.time, "sleep", sleeps.append)
    write(path, {"value": "new"})
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert sleeps == [.05, .1]
    assert read(path) == {"value": "new"}
    assert not attempts[0].exists()


def test_artifact_persistent_lock_keeps_both_versions_and_stale_temp(tmp_path, monkeypatch, artifact_writer):
    write, read, suffix = artifact_writer
    path = tmp_path / f"artifact{suffix}"
    write(path, {"value": "old"})
    stale = path.with_suffix(".tmp")
    stale.write_bytes(b"earlier failed save")
    attempts, sleeps = [], []

    def replace(temporary, destination):
        attempts.append(temporary)
        raise PermissionError(errno.EACCES, "persistent access denial")

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(survival.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="after 8 attempts.*retained at") as exc:
        write(path, {"value": "new"})
    assert isinstance(exc.value.__cause__, PermissionError)
    assert len(attempts) == 8 and len(sleeps) == 7 and sum(sleeps) < 4.
    assert read(path) == {"value": "old"}
    assert len(set(attempts)) == 1 and read(attempts[0]) == {"value": "new"}
    assert stale.read_bytes() == b"earlier failed save"


def test_artifact_does_not_retry_unrelated_io_error(tmp_path, monkeypatch, artifact_writer):
    write, read, suffix = artifact_writer
    path = tmp_path / f"artifact{suffix}"
    attempts, sleeps = [], []

    def replace(temporary, destination):
        attempts.append(temporary)
        raise OSError(errno.EIO, "unrelated I/O error")

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(survival.time, "sleep", sleeps.append)
    with pytest.raises(OSError, match="unrelated I/O error"):
        write(path, {"value": "new"})
    assert len(attempts) == 1 and not sleeps
    assert not path.exists() and read(attempts[0]) == {"value": "new"}


def test_artifact_uses_distinct_closed_temporary_files(tmp_path, monkeypatch, artifact_writer):
    write, read, suffix = artifact_writer
    path = tmp_path / f"artifact{suffix}"
    original_replace, original_save = Path.replace, torch.save
    paths, streams = [], []

    def save(payload, stream):
        streams.append(stream)
        original_save(payload, stream)

    def replace(temporary, destination):
        assert all(stream.closed for stream in streams)
        paths.append(temporary)
        return original_replace(temporary, destination)

    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(Path, "replace", replace)
    write(path, {"value": 1})
    write(path, {"value": 2})
    assert len(set(paths)) == 2 and all(p.parent == path.parent for p in paths)
    assert all(not p.exists() for p in paths)
    assert read(path) == {"value": 2}
