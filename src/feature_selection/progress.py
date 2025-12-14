"""Progress tracking and ETA estimation.

This module provides progress reporting functionality for long-running
feature selection operations, including console logging and callbacks.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional
import sys


@dataclass
class PhaseProgress:
    """Progress state for a single phase of the search.

    Attributes:
        phase_name: Name of the current phase.
        completed_evals: Number of evaluations completed.
        total_evals: Total expected evaluations (may be estimate).
        total_evals_is_estimate: Whether total_evals is an estimate.
        best_metric: Best metric value found so far.
        best_subset_size: Size of the best subset found.
        start_time: Phase start timestamp.
        eval_times: Recent evaluation durations for ETA calculation.
    """
    phase_name: str
    completed_evals: int = 0
    total_evals: int = 0
    total_evals_is_estimate: bool = False
    best_metric: float = 0.0
    best_subset_size: int = 0
    start_time: float = field(default_factory=time.time)
    eval_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def elapsed_seconds(self) -> float:
        """Get elapsed time since phase start."""
        return time.time() - self.start_time

    def avg_eval_time(self) -> float:
        """Get average evaluation time from recent evaluations."""
        if not self.eval_times:
            return 0.0
        return sum(self.eval_times) / len(self.eval_times)

    def eta_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if not self.eval_times or self.total_evals <= 0:
            return None

        remaining = self.total_evals - self.completed_evals
        if remaining <= 0:
            return 0.0

        return remaining * self.avg_eval_time()

    def progress_pct(self) -> float:
        """Get progress as percentage."""
        if self.total_evals <= 0:
            return 0.0
        return min(100.0, 100.0 * self.completed_evals / self.total_evals)


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable form.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "2h 15m" or "45s".
    """
    if seconds is None:
        return "unknown"

    if seconds < 0:
        return "N/A"

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


class ProgressTracker:
    """Tracks and reports progress for feature selection.

    Handles progress updates, ETA estimation, and reporting via
    console output and/or callbacks.

    Attributes:
        enable_console: Whether to print to console.
        update_freq_evals: Update after every N evaluations.
        update_freq_seconds: Update at least every N seconds.
        callback: Optional callback function.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        enable_console: bool = True,
        update_freq_evals: int = 10,
        update_freq_seconds: float = 30.0,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: int = 1
    ):
        """Initialize the progress tracker.

        Args:
            enable_console: Whether to print progress to console.
            update_freq_evals: Print every N evaluations.
            update_freq_seconds: Print at least every N seconds.
            callback: Optional callback receiving progress dict.
            verbose: Verbosity (0=silent, 1=phases, 2=detailed).
        """
        self.enable_console = enable_console
        self.update_freq_evals = update_freq_evals
        self.update_freq_seconds = update_freq_seconds
        self.callback = callback
        self.verbose = verbose

        self._current_phase: Optional[PhaseProgress] = None
        self._last_update_time: float = 0
        self._last_update_eval: int = 0
        self._global_start_time: float = time.time()
        self._phase_history: List[Dict[str, Any]] = []

    def start_phase(
        self,
        phase_name: str,
        total_evals: int = 0,
        is_estimate: bool = False
    ):
        """Start a new phase of the search.

        Args:
            phase_name: Name of the phase (e.g., "forward_selection").
            total_evals: Expected number of evaluations.
            is_estimate: Whether total_evals is an estimate.
        """
        # Save previous phase info
        if self._current_phase is not None:
            self._phase_history.append({
                'phase': self._current_phase.phase_name,
                'duration': self._current_phase.elapsed_seconds(),
                'evals': self._current_phase.completed_evals,
                'best_metric': self._current_phase.best_metric
            })

        self._current_phase = PhaseProgress(
            phase_name=phase_name,
            total_evals=total_evals,
            total_evals_is_estimate=is_estimate
        )
        self._last_update_time = time.time()
        self._last_update_eval = 0

        if self.verbose >= 1 and self.enable_console:
            total_str = f"~{total_evals}" if is_estimate else str(total_evals)
            print(f"\n[{phase_name}] Starting ({total_str} evaluations expected)")
            sys.stdout.flush()

    def update(
        self,
        completed: int,
        best_metric: float,
        best_subset_size: int,
        eval_time: Optional[float] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """Update progress.

        Args:
            completed: Number of evaluations completed.
            best_metric: Current best metric value.
            best_subset_size: Size of current best subset.
            eval_time: Duration of the last evaluation.
            extra_info: Additional info to include in callback.
        """
        if self._current_phase is None:
            return

        self._current_phase.completed_evals = completed
        self._current_phase.best_metric = best_metric
        self._current_phase.best_subset_size = best_subset_size

        if eval_time is not None:
            self._current_phase.eval_times.append(eval_time)

        # Check if we should report
        should_report = False
        evals_since_update = completed - self._last_update_eval
        time_since_update = time.time() - self._last_update_time

        if evals_since_update >= self.update_freq_evals:
            should_report = True
        elif time_since_update >= self.update_freq_seconds:
            should_report = True

        if should_report:
            self._report(extra_info)

    def _report(self, extra_info: Optional[Dict[str, Any]] = None):
        """Generate and output a progress report."""
        if self._current_phase is None:
            return

        phase = self._current_phase
        self._last_update_time = time.time()
        self._last_update_eval = phase.completed_evals

        # Build progress info dict
        info = {
            'phase': phase.phase_name,
            'completed_evals': phase.completed_evals,
            'total_evals': phase.total_evals,
            'total_evals_is_estimate': phase.total_evals_is_estimate,
            'progress_pct': phase.progress_pct(),
            'best_metric': phase.best_metric,
            'best_subset_size': phase.best_subset_size,
            'elapsed_seconds': phase.elapsed_seconds(),
            'eta_seconds': phase.eta_seconds(),
            'avg_eval_time': phase.avg_eval_time(),
            'global_elapsed': time.time() - self._global_start_time
        }

        if extra_info:
            info.update(extra_info)

        # Console output
        if self.enable_console and self.verbose >= 1:
            self._print_progress(info)

        # Callback
        if self.callback is not None:
            try:
                self.callback(info)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"  [Warning] Callback error: {e}")

    def _print_progress(self, info: Dict[str, Any]):
        """Print progress to console.

        Args:
            info: Progress information dict.
        """
        phase = info['phase']
        completed = info['completed_evals']
        total = info['total_evals']
        metric = info['best_metric']
        eta = info.get('eta_seconds')
        subset_size = info['best_subset_size']

        # Format total (with ~ if estimate)
        total_str = f"~{total}" if info.get('total_evals_is_estimate') else str(total)

        # Build status line
        eta_str = format_duration(eta)
        line = (
            f"[{phase}] eval {completed}/{total_str} | "
            f"best={metric:.4f} (n={subset_size}) | "
            f"ETA {eta_str}"
        )

        print(f"\r{line}", end='')
        sys.stdout.flush()

    def end_phase(self, final_metric: Optional[float] = None):
        """End the current phase.

        Args:
            final_metric: Final metric value for the phase.
        """
        if self._current_phase is None:
            return

        if final_metric is not None:
            self._current_phase.best_metric = final_metric

        if self.enable_console and self.verbose >= 1:
            phase = self._current_phase
            duration = format_duration(phase.elapsed_seconds())
            print(f"\n[{phase.phase_name}] Complete: {phase.completed_evals} evals, "
                  f"best={phase.best_metric:.4f}, time={duration}")
            sys.stdout.flush()

    def finish(self):
        """Finish all tracking and print summary."""
        self.end_phase()

        if self.enable_console and self.verbose >= 1:
            total_time = time.time() - self._global_start_time
            print(f"\n{'='*60}")
            print(f"Feature selection complete. Total time: {format_duration(total_time)}")

            if self._phase_history:
                print("\nPhase summary:")
                for ph in self._phase_history:
                    print(f"  {ph['phase']}: {ph['evals']} evals, "
                          f"best={ph['best_metric']:.4f}, "
                          f"time={format_duration(ph['duration'])}")

            print('='*60)
            sys.stdout.flush()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all phases.

        Returns:
            Dict with phase summaries and global stats.
        """
        return {
            'total_time': time.time() - self._global_start_time,
            'phases': self._phase_history.copy(),
            'current_phase': self._current_phase.phase_name if self._current_phase else None
        }


class NullProgressTracker(ProgressTracker):
    """A no-op progress tracker for when progress reporting is disabled."""

    def __init__(self):
        super().__init__(enable_console=False, callback=None, verbose=0)

    def start_phase(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def end_phase(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def create_progress_tracker(
    enable_console: bool = True,
    update_freq_evals: int = 10,
    update_freq_seconds: float = 30.0,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose: int = 1
) -> ProgressTracker:
    """Factory function to create a progress tracker.

    Args:
        enable_console: Whether to print to console.
        update_freq_evals: Update frequency by evaluation count.
        update_freq_seconds: Update frequency by time.
        callback: Optional callback function.
        verbose: Verbosity level.

    Returns:
        ProgressTracker instance (or NullProgressTracker if all disabled).
    """
    if not enable_console and callback is None:
        return NullProgressTracker()

    return ProgressTracker(
        enable_console=enable_console,
        update_freq_evals=update_freq_evals,
        update_freq_seconds=update_freq_seconds,
        callback=callback,
        verbose=verbose
    )
