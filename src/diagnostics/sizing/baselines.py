"""Baseline computation and management for drift metrics.

Baselines are computed from historical OOS predictions and stored for
comparison against current predictions. Drift is measured using PSI and KS.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd


@dataclass
class DistributionBaseline:
    """Stored baseline distribution for a model.

    Attributes:
        model_key: Model identifier (e.g., 'long_normal').
        metric: What was measured (e.g., 'probability', 'edge').
        n_samples: Number of samples used to compute baseline.
        date_range: Start and end dates of baseline period.
        computed_at: When baseline was computed.
        histogram_bins: Bin edges for histogram.
        histogram_counts: Counts per bin (normalized to sum to 1).
        percentiles: Key percentiles (p5, p10, p25, p50, p75, p90, p95).
        mean: Mean of baseline distribution.
        std: Standard deviation.
        min_val: Minimum value.
        max_val: Maximum value.
    """
    model_key: str
    metric: str
    n_samples: int
    date_range: Dict[str, str]
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[float] = field(default_factory=list)
    percentiles: Dict[str, float] = field(default_factory=dict)
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistributionBaseline":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DistributionBaseline":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_psi(
    baseline_counts: np.ndarray,
    current_counts: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute Population Stability Index (PSI).

    PSI measures shift between two distributions:
    - PSI < 0.10: No significant shift
    - 0.10 <= PSI < 0.25: Moderate shift
    - PSI >= 0.25: Significant shift

    Args:
        baseline_counts: Normalized histogram counts for baseline.
        current_counts: Normalized histogram counts for current.
        epsilon: Small value to avoid log(0).

    Returns:
        PSI value.
    """
    # Ensure both are normalized
    baseline = np.array(baseline_counts) + epsilon
    current = np.array(current_counts) + epsilon

    baseline = baseline / baseline.sum()
    current = current / current.sum()

    # PSI formula: sum((current - baseline) * log(current / baseline))
    psi = np.sum((current - baseline) * np.log(current / baseline))

    return float(psi)


def compute_ks_statistic(
    baseline_values: np.ndarray,
    current_values: np.ndarray,
) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic and p-value.

    KS statistic measures the maximum difference between CDFs.

    Args:
        baseline_values: Baseline sample values.
        current_values: Current sample values.

    Returns:
        Tuple of (ks_statistic, p_value).
    """
    from scipy import stats

    if len(baseline_values) == 0 or len(current_values) == 0:
        return 0.0, 1.0

    ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)
    return float(ks_stat), float(p_value)


def compute_baseline_from_predictions(
    predictions: pd.DataFrame,
    model_key: str,
    metric: str = "probability",
    n_bins: int = 20,
) -> DistributionBaseline:
    """Compute baseline distribution from historical predictions.

    Args:
        predictions: DataFrame with prediction columns.
        model_key: Model to compute baseline for.
        metric: 'probability' or 'edge'.
        n_bins: Number of histogram bins.

    Returns:
        DistributionBaseline object.
    """
    # Determine column name
    if metric == "probability":
        col = f"p_{model_key}"
    elif metric == "edge":
        col = f"edge_{model_key}"
    else:
        col = metric

    if col not in predictions.columns:
        raise ValueError(f"Column {col} not found in predictions")

    values = predictions[col].dropna().values

    if len(values) == 0:
        raise ValueError(f"No valid values for {col}")

    # Compute histogram
    if metric == "probability":
        bins = np.linspace(0, 1, n_bins + 1)
    else:
        # For edge, use data-driven bins
        min_v, max_v = values.min(), values.max()
        bins = np.linspace(min_v, max_v, n_bins + 1)

    counts, _ = np.histogram(values, bins=bins)
    counts_normalized = counts / counts.sum()

    # Compute percentiles
    percentiles = {
        'p5': float(np.percentile(values, 5)),
        'p10': float(np.percentile(values, 10)),
        'p25': float(np.percentile(values, 25)),
        'p50': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)),
    }

    # Get date range
    date_col = 'week_monday' if 'week_monday' in predictions.columns else 'date'
    if date_col in predictions.columns:
        dates = pd.to_datetime(predictions[date_col])
        date_range = {
            'start': str(dates.min().date()),
            'end': str(dates.max().date()),
        }
    else:
        date_range = {'start': 'unknown', 'end': 'unknown'}

    return DistributionBaseline(
        model_key=model_key,
        metric=metric,
        n_samples=len(values),
        date_range=date_range,
        histogram_bins=bins.tolist(),
        histogram_counts=counts_normalized.tolist(),
        percentiles=percentiles,
        mean=float(values.mean()),
        std=float(values.std()),
        min_val=float(values.min()),
        max_val=float(values.max()),
    )


class BaselineManager:
    """Manages baseline storage and comparison.

    Baselines are stored at:
        artifacts/diagnostics/baselines/<model_key>_<metric>_baseline.json

    Usage:
        manager = BaselineManager("artifacts/diagnostics/baselines")

        # Compute and save baselines
        manager.compute_baselines(predictions, models=['long_normal'])

        # Compare current to baseline
        psi = manager.compute_psi('long_normal', 'probability', current_values)
    """

    def __init__(
        self,
        baseline_dir: str = "artifacts/diagnostics/baselines",
    ):
        self.baseline_dir = Path(baseline_dir)
        self._cache: Dict[str, DistributionBaseline] = {}

    def _get_path(self, model_key: str, metric: str) -> Path:
        """Get path for a baseline file."""
        return self.baseline_dir / f"{model_key}_{metric}_baseline.json"

    def has_baseline(self, model_key: str, metric: str = "probability") -> bool:
        """Check if baseline exists."""
        return self._get_path(model_key, metric).exists()

    def load_baseline(
        self,
        model_key: str,
        metric: str = "probability",
    ) -> Optional[DistributionBaseline]:
        """Load baseline from disk.

        Args:
            model_key: Model identifier.
            metric: 'probability' or 'edge'.

        Returns:
            DistributionBaseline or None if not found.
        """
        cache_key = f"{model_key}_{metric}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_path(model_key, metric)
        if not path.exists():
            return None

        baseline = DistributionBaseline.from_json(str(path))
        self._cache[cache_key] = baseline
        return baseline

    def save_baseline(
        self,
        baseline: DistributionBaseline,
    ) -> None:
        """Save baseline to disk.

        Args:
            baseline: DistributionBaseline to save.
        """
        path = self._get_path(baseline.model_key, baseline.metric)
        baseline.to_json(str(path))

        cache_key = f"{baseline.model_key}_{baseline.metric}"
        self._cache[cache_key] = baseline

    def compute_baselines(
        self,
        predictions: pd.DataFrame,
        models: Optional[List[str]] = None,
        metrics: List[str] = ["probability", "edge"],
        overwrite: bool = False,
    ) -> Dict[str, DistributionBaseline]:
        """Compute baselines for multiple models/metrics.

        Args:
            predictions: DataFrame with prediction columns.
            models: List of model keys (default: auto-detect).
            metrics: List of metrics to compute.
            overwrite: Whether to overwrite existing baselines.

        Returns:
            Dict of computed baselines keyed by f"{model}_{metric}".
        """
        # Auto-detect models from columns
        if models is None:
            prob_cols = [c for c in predictions.columns if c.startswith('p_')]
            models = [c[2:] for c in prob_cols]

        results = {}
        for model in models:
            for metric in metrics:
                key = f"{model}_{metric}"

                # Skip if exists and not overwriting
                if not overwrite and self.has_baseline(model, metric):
                    print(f"  Skipping {key} (exists)")
                    continue

                try:
                    baseline = compute_baseline_from_predictions(
                        predictions, model, metric
                    )
                    self.save_baseline(baseline)
                    results[key] = baseline
                    print(f"  Computed {key}: n={baseline.n_samples}")
                except Exception as e:
                    print(f"  Warning: Could not compute {key}: {e}")

        return results

    def compute_psi_vs_baseline(
        self,
        model_key: str,
        metric: str,
        current_values: np.ndarray,
    ) -> Optional[float]:
        """Compute PSI of current values vs baseline.

        Args:
            model_key: Model identifier.
            metric: 'probability' or 'edge'.
            current_values: Current sample values.

        Returns:
            PSI value or None if no baseline.
        """
        baseline = self.load_baseline(model_key, metric)
        if baseline is None:
            return None

        # Compute histogram with same bins
        bins = np.array(baseline.histogram_bins)
        counts, _ = np.histogram(current_values, bins=bins)
        counts_normalized = counts / (counts.sum() + 1e-10)

        return compute_psi(
            np.array(baseline.histogram_counts),
            counts_normalized,
        )

    def compute_ks_vs_baseline(
        self,
        model_key: str,
        metric: str,
        current_values: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Compute KS statistic of current values vs baseline.

        Note: This requires storing raw values or using approximation.
        For simplicity, we approximate by generating samples from histogram.

        Args:
            model_key: Model identifier.
            metric: 'probability' or 'edge'.
            current_values: Current sample values.

        Returns:
            Tuple of (ks_stat, p_value) or None if no baseline.
        """
        baseline = self.load_baseline(model_key, metric)
        if baseline is None:
            return None

        # Approximate baseline distribution by generating samples from histogram
        bins = np.array(baseline.histogram_bins)
        counts = np.array(baseline.histogram_counts)

        # Generate synthetic samples from histogram
        n_samples = min(len(current_values), 10000)
        bin_indices = np.random.choice(
            len(counts),
            size=n_samples,
            p=counts / counts.sum(),
        )
        # Sample uniformly within each bin
        baseline_samples = np.array([
            np.random.uniform(bins[i], bins[i + 1])
            for i in bin_indices
        ])

        return compute_ks_statistic(baseline_samples, current_values)

    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get summary of all available baselines.

        Returns:
            Dict with baseline metadata.
        """
        summary = {'baselines': []}

        if not self.baseline_dir.exists():
            return summary

        for path in self.baseline_dir.glob("*_baseline.json"):
            try:
                baseline = DistributionBaseline.from_json(str(path))
                summary['baselines'].append({
                    'model_key': baseline.model_key,
                    'metric': baseline.metric,
                    'n_samples': baseline.n_samples,
                    'date_range': baseline.date_range,
                    'computed_at': baseline.computed_at,
                })
            except Exception as e:
                summary['baselines'].append({
                    'path': str(path),
                    'error': str(e),
                })

        return summary
