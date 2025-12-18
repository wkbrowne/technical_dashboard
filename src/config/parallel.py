"""
Unified parallel processing configuration for the feature pipeline.

This module provides a single configuration class that controls parallelization
across all pipeline stages, ensuring consistent behavior and easy tuning.

Worker Lifecycle:
- Worker lifecycle (spawning, pooling, shutdown) is managed entirely by joblib
- We do NOT explicitly manage worker processes
- See Section 4 of FEATURE_PIPELINE_ARCHITECTURE.md

Memory Management:
- Bounded batch scheduling to prevent memory explosion
- gc.collect() between batches for memory cleanup

Diagnostics:
- Progress logging with task counts, active workers, and ETA
- Optional memory usage reporting (requires psutil)
- Debug mode for single-threaded execution
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Callable, Any, Generator
import os
import gc
import time
import logging

logger = logging.getLogger(__name__)

# Optional psutil for memory reporting
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_memory_mb() -> float:
    """Get current process memory usage in MB (or 0 if psutil not available)."""
    if _HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available / 1024**3
    return 0.0


# Default maximum workers to prevent memory explosion
# 32 workers is a good balance for 45GB RAM systems
DEFAULT_MAX_WORKERS = 32

# Default stocks per worker - controls parallelism granularity
# 200 stocks/worker means 3000 stocks = 15 workers, 200 stocks = 1 worker
DEFAULT_STOCKS_PER_WORKER = 200


def calculate_workers_from_items(
    n_items: int,
    items_per_worker: int = DEFAULT_STOCKS_PER_WORKER,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_workers: int = 1
) -> int:
    """
    Calculate number of workers based on item count and items-per-worker setting.

    This provides predictable parallelism:
    - 3000 items with 100/worker = 30 workers
    - 100 items with 100/worker = 1 worker
    - Workers capped at max_workers and available cores

    Args:
        n_items: Number of items to process (e.g., stocks)
        items_per_worker: Target items per worker (default 100)
        max_workers: Maximum workers regardless of item count
        min_workers: Minimum workers to use

    Returns:
        Number of workers to use
    """
    import multiprocessing

    if n_items <= 0:
        return min_workers

    # Calculate workers needed based on items_per_worker
    workers_needed = max(1, (n_items + items_per_worker - 1) // items_per_worker)

    # Cap at max_workers and available cores
    available_cores = multiprocessing.cpu_count()
    workers = min(workers_needed, max_workers, available_cores)
    workers = max(workers, min_workers)

    logger.debug(
        f"calculate_workers_from_items: {n_items} items / {items_per_worker} per worker "
        f"= {workers_needed} needed, capped to {workers} (max={max_workers}, cores={available_cores})"
    )

    return workers


def get_safe_n_jobs(n_jobs: int = -1, max_workers: int = DEFAULT_MAX_WORKERS) -> int:
    """
    Get a safe n_jobs value, capped at max_workers to prevent memory explosion.

    Args:
        n_jobs: Requested n_jobs (-1 for all cores)
        max_workers: Maximum workers to allow (default 32)

    Returns:
        Safe n_jobs value
    """
    import multiprocessing

    if n_jobs == -1:
        effective = multiprocessing.cpu_count()
    elif n_jobs < 1:
        effective = 1
    else:
        effective = n_jobs

    safe_jobs = min(effective, max_workers)
    if effective != safe_jobs:
        logger.debug(f"Capping n_jobs from {effective} to {safe_jobs} (max_workers={max_workers})")

    return safe_jobs


def get_memory_safe_workers(
    n_items: int,
    memory_per_item_mb: float = 30.0,
    memory_headroom_gb: float = 8.0,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_workers: int = 4
) -> int:
    """
    Calculate safe number of workers based on available memory.

    For a dataset of n_items, estimates memory needed per worker (which gets
    a copy of its chunk during serialization) and limits workers to avoid OOM.

    Args:
        n_items: Number of items to process (e.g., symbols)
        memory_per_item_mb: Estimated memory per item in MB (default 30MB for stock data)
        memory_headroom_gb: Reserve this much memory for system (default 8GB)
        max_workers: Maximum workers regardless of memory (default 32)
        min_workers: Minimum workers to use (default 4)

    Returns:
        Recommended number of workers
    """
    if not _HAS_PSUTIL:
        logger.debug("psutil not available, using default max_workers")
        return max_workers

    available_gb = get_available_memory_gb()
    usable_gb = max(0, available_gb - memory_headroom_gb)

    # Each worker gets n_items/n_workers items, plus serialization overhead (~1.5x)
    # Memory per worker = (n_items / n_workers) * memory_per_item_mb * 1.5 / 1024
    # Solving for n_workers: n_workers = (n_items * memory_per_item_mb * 1.5) / (usable_gb * 1024)

    if usable_gb <= 0:
        logger.warning(f"Low memory: only {available_gb:.1f}GB available, using minimum {min_workers} workers")
        return min_workers

    total_data_gb = (n_items * memory_per_item_mb * 1.5) / 1024

    # Safe workers = usable memory / memory per worker
    # Each worker holds ~total_data_gb / n_workers data, so n_workers = total_data_gb / (usable_gb / parallelism_factor)
    # With parallelism_factor accounting for concurrent workers

    # Simpler: cap at 32 workers for memory safety, scale down if needed
    if total_data_gb > usable_gb:
        # Need to reduce parallelism
        safe_workers = max(min_workers, int(usable_gb / (total_data_gb / max_workers)))
    else:
        safe_workers = max_workers

    safe_workers = max(min_workers, min(max_workers, safe_workers))

    logger.debug(
        f"Memory-safe workers: {safe_workers} "
        f"(available={available_gb:.1f}GB, usable={usable_gb:.1f}GB, "
        f"est_data={total_data_gb:.1f}GB, n_items={n_items})"
    )

    return safe_workers


@contextmanager
def parallel_stage(
    stage_name: str,
    n_workers: int = -1,
) -> Generator[None, None, None]:
    """
    Context manager for a parallel processing stage.

    Note: Worker lifecycle is managed entirely by joblib - we do not explicitly
    manage worker processes. This context manager just provides timing/memory logging.

    Args:
        stage_name: Name of the stage (for logging)
        n_workers: Number of workers (-1 for all cores), for logging only

    Example:
        with parallel_stage("Feature Computation", n_workers=8):
            results = Parallel(n_jobs=8)(...)(delayed(func)(x) for x in items)
    """
    start_time = time.time()
    start_mem = get_memory_mb()

    logger.info(f"[{stage_name}] Starting (workers={n_workers}, mem={start_mem:.0f}MB)")

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        end_mem = get_memory_mb()
        mem_delta = end_mem - start_mem
        logger.info(
            f"[{stage_name}] Completed in {elapsed:.1f}s "
            f"(mem: {start_mem:.0f}MB -> {end_mem:.0f}MB, delta={mem_delta:+.0f}MB)"
        )


def chunked_parallel(
    items: List[Any],
    func: Callable,
    chunk_size: int = 100,
    n_jobs: int = -1,
    max_pending: int = None,
    stage_name: str = "Processing",
    verbose: bool = True
) -> List[Any]:
    """
    Execute parallel computation with bounded batching and progress reporting.

    This function implements backpressure by processing chunks sequentially
    in groups, preventing memory explosion from having too many pending futures.

    Args:
        items: List of items to process
        func: Function to apply to each chunk (receives List[items], returns List[results])
        chunk_size: Number of items per chunk
        n_jobs: Number of parallel workers
        max_pending: Maximum number of chunks to process at once (None = n_jobs * 2)
        stage_name: Name for logging
        verbose: Whether to log progress

    Returns:
        Flattened list of results
    """
    from joblib import Parallel, delayed
    import multiprocessing

    n_items = len(items)
    if n_items == 0:
        return []

    # Resolve effective workers
    effective_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    # Create chunks
    chunks = [items[i:i + chunk_size] for i in range(0, n_items, chunk_size)]
    n_chunks = len(chunks)

    # Set max pending based on workers
    if max_pending is None:
        max_pending = effective_jobs * 2

    if verbose:
        logger.info(
            f"[{stage_name}] {n_items} items in {n_chunks} chunks "
            f"({chunk_size}/chunk, {effective_jobs} workers, max_pending={max_pending})"
        )

    start_time = time.time()
    all_results = []
    chunks_done = 0

    # Process chunks in bounded groups
    with parallel_stage(stage_name, n_workers=effective_jobs):
        for batch_start in range(0, n_chunks, max_pending):
            batch_end = min(batch_start + max_pending, n_chunks)
            batch_chunks = chunks[batch_start:batch_end]

            # Process this batch
            batch_results = Parallel(
                n_jobs=effective_jobs,
                backend='loky',
                verbose=0
            )(
                delayed(func)(chunk) for chunk in batch_chunks
            )

            # Flatten results
            for result in batch_results:
                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)

            chunks_done += len(batch_chunks)

            # Progress report
            if verbose:
                elapsed = time.time() - start_time
                items_done = chunks_done * chunk_size
                rate = items_done / elapsed if elapsed > 0 else 0
                remaining = n_items - items_done
                eta = remaining / rate if rate > 0 else 0

                mem_mb = get_memory_mb()
                logger.info(
                    f"[{stage_name}] Progress: {chunks_done}/{n_chunks} chunks "
                    f"({items_done}/{n_items} items, {rate:.0f}/s, ETA {eta:.0f}s, mem={mem_mb:.0f}MB)"
                )

            # Memory cleanup between batches
            gc.collect()

    if verbose:
        total_time = time.time() - start_time
        logger.info(f"[{stage_name}] Complete: {len(all_results)} results in {total_time:.1f}s")

    return all_results


def run_sequential_or_parallel(
    items: List[Any],
    func: Callable,
    n_jobs: int = -1,
    chunk_size: int = 100,
    min_parallel_items: int = 10,
    stage_name: str = "Processing"
) -> List[Any]:
    """
    Run computation sequentially or in parallel based on item count.

    Automatically falls back to sequential for small datasets or when
    n_jobs=1 (debug mode).

    Args:
        items: Items to process
        func: Function to apply to each chunk
        n_jobs: Number of jobs (1 for sequential/debug mode)
        chunk_size: Items per chunk for parallel mode
        min_parallel_items: Minimum items to use parallelism
        stage_name: Name for logging

    Returns:
        Results list
    """
    n_items = len(items)

    if n_items == 0:
        return []

    # Sequential mode
    if n_jobs == 1 or n_items < min_parallel_items:
        logger.info(f"[{stage_name}] Running sequentially ({n_items} items)")

        start_time = time.time()
        # Process all items as one chunk
        results = func(items)
        elapsed = time.time() - start_time

        logger.info(f"[{stage_name}] Sequential complete in {elapsed:.1f}s")
        return results if isinstance(results, list) else [results]

    # Parallel mode
    return chunked_parallel(
        items, func,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        stage_name=stage_name,
        verbose=True
    )


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
        debug_mode: If True, forces sequential execution for debugging.
            Overrides n_jobs setting to run single-threaded.
        report_memory: If True, logs memory usage during processing.
            Requires psutil to be installed.

    Example:
        >>> config = ParallelConfig(n_jobs=8, batch_size='auto')
        >>> # Pass to pipeline
        >>> run_pipeline_v2(data, parallel_config=config)

        >>> # Or use defaults with all cores
        >>> config = ParallelConfig.default()

        >>> # For debugging, use sequential
        >>> config = ParallelConfig.sequential()

        >>> # Debug mode for isolating issues
        >>> config = ParallelConfig(debug_mode=True)

        >>> # Control parallelism by stocks per worker
        >>> config = ParallelConfig(stocks_per_worker=100)  # 3000 stocks = 30 workers
    """
    n_jobs: int = -1
    batch_size: str = 'auto'  # Changed from int to str for 'auto' support
    backend: str = 'loky'
    verbose: int = 0
    prefer: Optional[str] = 'processes'
    debug_mode: bool = False
    report_memory: bool = True
    stocks_per_worker: int = DEFAULT_STOCKS_PER_WORKER  # Stocks per parallel worker
    max_workers: int = DEFAULT_MAX_WORKERS  # Maximum workers regardless of cores

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

    def get_workers_for_items(self, n_items: int) -> int:
        """
        Calculate number of workers based on item count and stocks_per_worker.

        This provides predictable parallelism:
        - 3000 items with 100/worker = 30 workers
        - 100 items with 100/worker = 1 worker
        - Capped at max_workers and available cores

        Args:
            n_items: Number of items to process (e.g., stocks)

        Returns:
            Number of workers to use
        """
        return calculate_workers_from_items(
            n_items=n_items,
            items_per_worker=self.stocks_per_worker,
            max_workers=self.max_workers,
            min_workers=1
        )

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
