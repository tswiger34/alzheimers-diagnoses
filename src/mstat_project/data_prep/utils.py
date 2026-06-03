import os
from pathlib import Path

IMAGES_PATH = Path(os.getenv("IMAGES_PATH", "data/images"))


def get_cohort_name(cohort: int) -> str:
    """Return the canonical cohort folder/archive stem."""

    return f"cohort_{str(cohort).zfill(2)}"


def resolve_max_workers(max_workers: int | None = None) -> int:
    """Resolve the configured worker count."""

    if max_workers is None:
        raw_max_workers = os.getenv("NIFTI_WORKERS", str(16))
        try:
            max_workers = int(raw_max_workers)
        except ValueError as exc:
            msg = f"{'NIFTI_WORKERS'} must be an integer, got {raw_max_workers!r}"
            raise ValueError(msg) from exc

        max_workers = min(max_workers, os.cpu_count() or 1)

    if max_workers < 1:
        msg = f"max_workers must be at least 1, got {max_workers}"
        raise ValueError(msg)

    return max_workers
