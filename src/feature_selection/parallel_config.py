"""Parallel execution configuration for feature selection.

This module provides centralized configuration for parallelization in the
feature selection pipeline.

Backend Selection:
- Threading backend: Fast startup, shared memory, but limited by Python GIL.
  Use for I/O-bound or NumPy-heavy workloads where native code releases GIL.
- Loky backend: Process-based parallelism that bypasses GIL. Use for CPU-bound
  workloads like LightGBM training where threads cannot run in parallel.

For move evaluation in group selection, LightGBM training is CPU-bound and
the GIL prevents threads from running in parallel. Use loky backend for
true parallel execution of candidate moves.

Usage:
    from src.feature_selection.parallel_config import (
        get_parallel_config, get_joblib_kwargs, get_loky_kwargs
    )

    # For threading (shared memory, GIL-limited):
    Parallel(**get_joblib_kwargs(n_jobs))(delayed(func)(...) for ...)

    # For process parallelism (bypasses GIL, requires pickling):
    Parallel(**get_loky_kwargs(n_jobs))(delayed(func)(...) for ...)
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes:
        candidate_jobs: Number of parallel jobs for candidate evaluation (threading).
        lgb_threads: Number of threads for LightGBM internally.
        total_cores: Total CPU cores available.
        fold_jobs: Number of jobs for CV fold evaluation (should be 1 when threading candidates).
    """
    candidate_jobs: int
    lgb_threads: int
    total_cores: int
    fold_jobs: int = 1  # Always 1 when using threading for candidates

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'candidate_jobs': self.candidate_jobs,
            'lgb_threads': self.lgb_threads,
            'total_cores': self.total_cores,
            'fold_jobs': self.fold_jobs,
        }


def get_parallel_config(
    candidate_jobs: Optional[int] = None,
    lgb_threads: Optional[int] = None,
) -> ParallelConfig:
    """Compute parallel configuration with safe defaults.

    The default strategy:
    - candidate_jobs = min(8, TOTAL_CORES // 2)
    - lgb_threads = max(1, TOTAL_CORES // candidate_jobs)

    This avoids oversubscription while allowing reasonable parallelism.

    Args:
        candidate_jobs: Override for number of candidate parallel jobs.
            If None, computed from CPU count.
        lgb_threads: Override for LightGBM thread count.
            If None, computed from remaining cores.

    Returns:
        ParallelConfig with computed or overridden values.
    """
    total_cores = os.cpu_count() or 4

    # Compute candidate_jobs if not specified
    if candidate_jobs is None:
        # Conservative default: at most 8, at most half of cores
        candidate_jobs = min(8, max(1, total_cores // 2))
    else:
        candidate_jobs = max(1, candidate_jobs)

    # Compute lgb_threads if not specified
    if lgb_threads is None:
        # Give LightGBM the remaining cores
        lgb_threads = max(1, total_cores // candidate_jobs)
    else:
        lgb_threads = max(1, lgb_threads)

    return ParallelConfig(
        candidate_jobs=candidate_jobs,
        lgb_threads=lgb_threads,
        total_cores=total_cores,
        fold_jobs=1,  # Always 1 for thread-based candidate parallelism
    )


def print_parallel_config(config: ParallelConfig) -> None:
    """Print parallel configuration for visibility.

    Args:
        config: ParallelConfig to display.
    """
    print(f"Parallel Configuration:")
    print(f"  Total cores: {config.total_cores}")
    print(f"  Candidate jobs (threading): {config.candidate_jobs}")
    print(f"  LightGBM threads: {config.lgb_threads}")
    print(f"  CV fold jobs: {config.fold_jobs}")
    print(f"  Theoretical max threads: {config.candidate_jobs * config.lgb_threads}")
    print()


# Joblib configuration constants
JOBLIB_BACKEND = 'threading'
JOBLIB_PREFER = 'threads'
JOBLIB_MAX_NBYTES = None  # Disable memmapping


def get_joblib_kwargs(n_jobs: int) -> dict:
    """Get standard joblib Parallel kwargs for threading backend.

    Threading is fast to start but limited by Python GIL for CPU-bound work.
    Use for I/O-bound or NumPy-heavy workloads.

    Args:
        n_jobs: Number of parallel jobs.

    Returns:
        Dict of kwargs for joblib.Parallel.
    """
    return {
        'n_jobs': n_jobs,
        'backend': JOBLIB_BACKEND,
        'prefer': JOBLIB_PREFER,
        'max_nbytes': JOBLIB_MAX_NBYTES,
        'verbose': 0,
    }


# Loky configuration constants
LOKY_BACKEND = 'loky'
LOKY_PREFER = 'processes'


def get_loky_kwargs(n_jobs: int) -> dict:
    """Get joblib Parallel kwargs for loky (process-based) backend.

    Loky spawns separate processes that bypass the Python GIL, enabling true
    parallel execution of CPU-bound workloads like LightGBM training.

    Trade-offs vs threading:
    - Pro: True parallelism for CPU-bound code (GIL bypass)
    - Con: Slower startup (process spawn)
    - Con: Objects must be picklable (can't share live objects)
    - Con: In-memory caches are not shared across workers

    Use this for move evaluation in group selection where LightGBM training
    dominates runtime and threads would serialize execution.

    Args:
        n_jobs: Number of parallel jobs.

    Returns:
        Dict of kwargs for joblib.Parallel.
    """
    return {
        'n_jobs': n_jobs,
        'backend': LOKY_BACKEND,
        'prefer': LOKY_PREFER,
        'verbose': 0,
    }


def configure_model_for_threading(model_config, lgb_threads: Optional[int] = None) -> None:
    """Configure a ModelConfig for threaded candidate evaluation.

    Sets num_threads on the model config to the recommended value for
    use with threaded candidate parallelism.

    Args:
        model_config: ModelConfig to update (modified in place).
        lgb_threads: Number of threads for LightGBM. If None, uses
            default from get_parallel_config().
    """
    if lgb_threads is None:
        parallel_config = get_parallel_config()
        lgb_threads = parallel_config.lgb_threads

    model_config.num_threads = lgb_threads
