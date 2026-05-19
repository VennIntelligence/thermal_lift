"""Small internal utilities shared across TCForge modules."""

from __future__ import annotations


def resolve_workers(workers: int | None = None, n_jobs: int | None = None) -> int:
    """Resolve the shared workers/n_jobs convention to a positive integer."""

    value = workers if workers is not None else n_jobs
    if value is None:
        return 1
    value = int(value)
    if value < 1:
        raise ValueError("workers/n_jobs must be >= 1")
    return value
