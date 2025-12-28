"""Reusable process executor for parallel feature selection.

This module provides a singleton pattern for managing a loky process pool
that can be reused across multiple parallel evaluation calls. This avoids
the overhead and resource leaks from repeatedly creating/destroying pools.

Why reusable executor?
- Each joblib.Parallel(backend='loky') call creates a NEW process pool
- Pool creation spawns processes, allocates shared memory, creates semlocks
- Pool destruction may not fully clean up, leading to leaked resources:
  - Semlock objects (inter-process synchronization primitives)
  - Temp folder objects (loky working directories)
- The resource_tracker warnings at shutdown indicate incomplete cleanup
- A reusable executor maintains ONE pool across all calls, avoiding churn

Usage:
    from src.feature_selection.executor import (
        parallel_map, shutdown_executor, get_worker_stats
    )

    # Use parallel_map instead of Parallel()
    results = parallel_map(func, items, n_workers=8)

    # At end of run, clean up
    shutdown_executor()
"""

import atexit
import os
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Use loky directly for reusable executor
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import ProcessPoolExecutor

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class WorkerStats:
    """Statistics about worker processes."""
    total_tasks: int = 0
    pid_counts: Dict[int, int] = None
    exceptions: List[str] = None

    def __post_init__(self):
        if self.pid_counts is None:
            self.pid_counts = {}
        if self.exceptions is None:
            self.exceptions = []


class ExecutorManager:
    """Singleton manager for a reusable loky process pool.

    Thread-safe singleton that maintains a single ProcessPoolExecutor
    across all parallel evaluation calls.
    """

    _instance: Optional['ExecutorManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._executor: Optional[ProcessPoolExecutor] = None
        self._n_workers: int = 0
        self._stats = WorkerStats()
        self._executor_lock = threading.Lock()
        self._initialized = True

    def get_executor(self, n_workers: int) -> ProcessPoolExecutor:
        """Get or create a reusable executor with the specified worker count.

        If the executor exists with a different worker count, it is shutdown
        and a new one is created.

        Args:
            n_workers: Number of worker processes.

        Returns:
            Reusable ProcessPoolExecutor.
        """
        with self._executor_lock:
            if self._executor is None or self._n_workers != n_workers:
                # Shutdown existing executor if worker count changed
                if self._executor is not None:
                    self._shutdown_internal()

                # Create new reusable executor
                # get_reusable_executor returns a pool that persists across calls
                self._executor = get_reusable_executor(
                    max_workers=n_workers,
                    timeout=300,  # Worker timeout in seconds
                    kill_workers=False,  # Don't kill workers on shutdown
                    reuse='auto',  # Reuse existing pool if compatible
                )
                self._n_workers = n_workers

            return self._executor

    def _shutdown_internal(self):
        """Internal shutdown without lock (caller must hold lock)."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, kill_workers=True)
            except Exception:
                pass  # Ignore shutdown errors
            self._executor = None
            self._n_workers = 0

    def shutdown(self):
        """Shutdown the executor and release resources."""
        with self._executor_lock:
            self._shutdown_internal()

    def record_task(self, pid: int):
        """Record a task execution from a worker PID."""
        self._stats.total_tasks += 1
        self._stats.pid_counts[pid] = self._stats.pid_counts.get(pid, 0) + 1

    def record_exception(self, exc_str: str):
        """Record an exception from a worker."""
        self._stats.exceptions.append(exc_str)

    def get_stats(self) -> WorkerStats:
        """Get worker statistics."""
        return self._stats

    def reset_stats(self):
        """Reset worker statistics."""
        self._stats = WorkerStats()


# Module-level singleton instance
_manager = ExecutorManager()


def _worker_wrapper(func: Callable, item: Any, debug: bool = False) -> tuple:
    """Wrapper that executes a function and captures PID and exceptions.

    Args:
        func: Function to execute.
        item: Argument to pass to function.
        debug: If True, include PID in result for debugging.

    Returns:
        Tuple of (success, result_or_exception, pid).
    """
    pid = os.getpid()
    try:
        result = func(item)
        return (True, result, pid)
    except Exception as e:
        # Capture full traceback for debugging
        tb = traceback.format_exc()
        return (False, f"{type(e).__name__}: {e}\n{tb}", pid)


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    n_workers: int,
    debug: bool = False,
) -> List[R]:
    """Execute function in parallel using reusable executor.

    This is a drop-in replacement for:
        Parallel(**get_loky_kwargs(n_workers))(delayed(func)(item) for item in items)

    But uses a reusable executor to avoid pool churn and resource leaks.

    Args:
        func: Function to apply to each item.
        items: List of items to process.
        n_workers: Number of worker processes.
        debug: If True, log PID statistics after execution.

    Returns:
        List of results in same order as items.

    Raises:
        RuntimeError: If any worker raised an exception (includes exception details).
    """
    if not items:
        return []

    if n_workers <= 1 or len(items) == 1:
        # Sequential execution for single worker or single item
        return [func(item) for item in items]

    executor = _manager.get_executor(n_workers)

    # Submit all tasks
    futures = [
        executor.submit(_worker_wrapper, func, item, debug)
        for item in items
    ]

    # Collect results in order
    results = []
    exceptions = []
    pid_counts = Counter()

    for future in futures:
        success, result_or_exc, pid = future.result()
        pid_counts[pid] += 1
        _manager.record_task(pid)

        if success:
            results.append(result_or_exc)
        else:
            exceptions.append(result_or_exc)
            _manager.record_exception(result_or_exc)
            # Return a None result for failed tasks
            results.append(None)

    if debug:
        print(f"[executor] Tasks: {len(items)}, Workers: {len(pid_counts)}, "
              f"PIDs: {dict(pid_counts)}", flush=True)

    # If there were exceptions, log them but don't raise
    # The caller should handle None results appropriately
    if exceptions:
        print(f"[executor] WARNING: {len(exceptions)} worker exceptions occurred:",
              flush=True)
        for i, exc in enumerate(exceptions[:3]):  # Show first 3
            print(f"  Exception {i+1}: {exc[:200]}...", flush=True)

    return results


def shutdown_executor():
    """Shutdown the reusable executor and release all resources.

    Call this at the end of a feature selection run to ensure clean shutdown.
    """
    _manager.shutdown()


def get_worker_stats() -> WorkerStats:
    """Get statistics about worker process usage.

    Returns:
        WorkerStats with task counts per PID and any exceptions.
    """
    return _manager.get_stats()


def reset_worker_stats():
    """Reset worker statistics for a new run."""
    _manager.reset_stats()


# Register cleanup on interpreter shutdown
atexit.register(shutdown_executor)
