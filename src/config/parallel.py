"""
Unified parallel processing configuration for the feature pipeline.

This module provides a single configuration class that controls parallelization
across all pipeline stages, ensuring consistent behavior and easy tuning.
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """
    Global parallelization configuration for the feature pipeline.

    This configuration flows through all pipeline stages to ensure consistent
    parallel processing behavior. All stages that can be parallelized by symbol
    will use these settings.

    Attributes:
        n_jobs: Number of parallel workers.
            -1 = all available cores
            1 = sequential processing (useful for debugging)
            N = use N workers
        batch_size: Number of symbols to process per batch.
            'auto' = let joblib determine optimal size (recommended for large core counts)
            Larger batches reduce IPC overhead but increase memory usage.
        backend: joblib backend to use.
            'loky' = robust process-based (default, recommended)
            'multiprocessing' = standard multiprocessing
            'threading' = thread-based (for I/O bound tasks)
        verbose: Verbosity level for progress reporting.
            0 = silent
            1 = progress bar
            >1 = detailed progress
        prefer: joblib prefer setting.
            'processes' = prefer process-based parallelism
            'threads' = prefer thread-based parallelism
            None = let joblib decide

    Example:
        >>> config = ParallelConfig(n_jobs=8, batch_size='auto')
        >>> # Pass to pipeline
        >>> run_pipeline_v2(data, parallel_config=config)

        >>> # Or use defaults with all cores
        >>> config = ParallelConfig.default()

        >>> # For debugging, use sequential
        >>> config = ParallelConfig.sequential()
    """
    n_jobs: int = -1
    batch_size: str = 'auto'  # Changed from int to str for 'auto' support
    backend: str = 'loky'
    verbose: int = 0
    prefer: Optional[str] = 'processes'

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0. Use -1 for all cores or 1 for sequential.")
        # batch_size can be 'auto' (str) or an int >= 1
        if isinstance(self.batch_size, int) and self.batch_size < 1:
            raise ValueError("batch_size must be at least 1 or 'auto'")
        if isinstance(self.batch_size, str) and self.batch_size != 'auto':
            raise ValueError("batch_size must be an integer or 'auto'")
        if self.backend not in ('loky', 'multiprocessing', 'threading'):
            raise ValueError(f"Unknown backend: {self.backend}")

    @classmethod
    def default(cls) -> 'ParallelConfig':
        """Create default configuration using all available cores."""
        return cls(
            n_jobs=-1,
            batch_size='auto',
            backend='loky',
            verbose=0,
            prefer='processes'
        )

    @classmethod
    def sequential(cls) -> 'ParallelConfig':
        """Create configuration for sequential processing (useful for debugging)."""
        return cls(
            n_jobs=1,
            batch_size='auto',
            backend='loky',
            verbose=0,
            prefer=None
        )

    @classmethod
    def from_env(cls) -> 'ParallelConfig':
        """
        Create configuration from environment variables.

        Environment variables:
            PIPELINE_N_JOBS: Number of parallel jobs (default: -1)
            PIPELINE_BATCH_SIZE: Batch size (default: 'auto', or an integer)
            PIPELINE_BACKEND: joblib backend (default: loky)
            PIPELINE_VERBOSE: Verbosity level (default: 0)
        """
        batch_size_str = os.getenv('PIPELINE_BATCH_SIZE', 'auto')
        batch_size = 'auto' if batch_size_str == 'auto' else int(batch_size_str)
        return cls(
            n_jobs=int(os.getenv('PIPELINE_N_JOBS', '-1')),
            batch_size=batch_size,
            backend=os.getenv('PIPELINE_BACKEND', 'loky'),
            verbose=int(os.getenv('PIPELINE_VERBOSE', '0')),
        )

    @property
    def effective_n_jobs(self) -> int:
        """Get the effective number of jobs (resolving -1 to actual core count)."""
        if self.n_jobs == -1:
            import multiprocessing
            return multiprocessing.cpu_count()
        return self.n_jobs

    @property
    def is_parallel(self) -> bool:
        """Check if configuration enables parallel processing."""
        return self.n_jobs != 1

    def for_stage(self, stage_name: str, override_batch_size: Optional[int] = None) -> 'ParallelConfig':
        """
        Create a potentially modified config for a specific stage.

        Some stages may benefit from different batch sizes. This method
        allows stage-specific overrides while keeping other settings.

        Args:
            stage_name: Name of the pipeline stage (for logging)
            override_batch_size: Optional batch size override for this stage

        Returns:
            ParallelConfig instance (possibly modified)
        """
        if override_batch_size is not None and override_batch_size != self.batch_size:
            logger.debug(f"Stage '{stage_name}' using batch_size={override_batch_size} (override)")
            return ParallelConfig(
                n_jobs=self.n_jobs,
                batch_size=override_batch_size,
                backend=self.backend,
                verbose=self.verbose,
                prefer=self.prefer
            )
        return self

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        return (
            f"ParallelConfig(n_jobs={self.n_jobs} [{self.effective_n_jobs} cores], "
            f"batch_size={self.batch_size}, backend='{self.backend}')"
        )

    def __repr__(self) -> str:
        return (
            f"ParallelConfig(n_jobs={self.n_jobs}, batch_size={self.batch_size}, "
            f"backend='{self.backend}', verbose={self.verbose}, prefer={self.prefer!r})"
        )
