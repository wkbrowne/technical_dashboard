"""
Multi-model feature selection support.

This module provides utilities for running feature selection across multiple
model targets (LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC)
in a single run, with shared CV splits and feature universe.

Key components:
- FeatureSelectionResult: Dataclass holding per-model selection results
- run_single_model_selection: Reusable function for single-model selection
- compute_overlap_analysis: Jaccard/intersection analysis across models
- compute_run_signature: Stable hash for reproducibility tracking
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Import ModelKey
try:
    from ..config.model_keys import ModelKey
except ImportError:
    from src.config.model_keys import ModelKey

from .config import (
    CVConfig,
    CVScheme,
    MetricConfig,
    MetricType,
    ModelConfig,
    ModelType,
    SearchConfig,
    TaskType,
)
from .pipeline import LooseTightConfig, LooseTightPipeline


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class FeatureSelectionResult:
    """Result of feature selection for a single model.

    Attributes:
        model_key: The model key (e.g., LONG_NORMAL)
        selected_features: List of selected feature names in order selected
        n_features: Number of selected features
        cv_auc_mean: Mean AUC across CV folds
        cv_auc_std: Std dev of AUC across CV folds
        fold_metrics: Per-fold AUC values
        secondary_metrics: Dict of (metric_name -> (mean, std))
        best_stage: Pipeline stage that produced best result
        algorithm: Feature selection algorithm used
        algorithm_params: Algorithm configuration parameters
        cv_config_hash: Hash of CV configuration for reproducibility
        date_range: (start_date, end_date) of data used
        universe_hash: Hash of feature universe
        timestamp: When selection was run (ISO format)
        run_signature: Combined hash for reproducibility
        elapsed_seconds: Time taken for selection
    """
    model_key: str
    selected_features: List[str]
    n_features: int
    cv_auc_mean: float
    cv_auc_std: float
    fold_metrics: List[float] = field(default_factory=list)
    secondary_metrics: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    best_stage: str = ""
    algorithm: str = "loose_tight_pipeline"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    cv_config_hash: str = ""
    date_range: Tuple[str, str] = ("", "")
    universe_hash: str = ""
    timestamp: str = ""
    run_signature: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Ensure all values are JSON serializable
        d['secondary_metrics'] = {
            k: {"mean": float(v[0]), "std": float(v[1])}
            for k, v in self.secondary_metrics.items()
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSelectionResult":
        """Create from dict."""
        # Convert secondary_metrics back to tuple format
        if 'secondary_metrics' in d and d['secondary_metrics']:
            d['secondary_metrics'] = {
                k: (v['mean'], v['std'])
                for k, v in d['secondary_metrics'].items()
            }
        # Convert date_range to tuple
        if 'date_range' in d and isinstance(d['date_range'], list):
            d['date_range'] = tuple(d['date_range'])
        return cls(**d)


@dataclass
class MultiModelSelectionSummary:
    """Summary of feature selection across all models.

    Attributes:
        results: Dict mapping model_key -> FeatureSelectionResult
        core_features: Features selected by ALL models (intersection)
        head_features: Dict mapping model_key -> features unique to that model
        overlap_matrix: Jaccard similarity between model pairs
        intersection_matrix: Intersection size between model pairs
        union_size: Total unique features across all models
        top_shared_features: Features shared by most models (with counts)
        run_signature: Combined signature for all models
        timestamp: When summary was generated
    """
    results: Dict[str, FeatureSelectionResult] = field(default_factory=dict)
    core_features: List[str] = field(default_factory=list)
    head_features: Dict[str, List[str]] = field(default_factory=dict)
    overlap_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    intersection_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    union_size: int = 0
    top_shared_features: List[Tuple[str, int]] = field(default_factory=list)
    run_signature: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "core_features": self.core_features,
            "head_features": self.head_features,
            "overlap_matrix": self.overlap_matrix,
            "intersection_matrix": self.intersection_matrix,
            "union_size": self.union_size,
            "top_shared_features": [
                {"feature": f, "count": c} for f, c in self.top_shared_features
            ],
            "run_signature": self.run_signature,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Signature Computation
# =============================================================================

def compute_cv_config_hash(cv_config: CVConfig) -> str:
    """Compute stable hash for CV configuration."""
    config_str = (
        f"n_splits={cv_config.n_splits},"
        f"scheme={cv_config.scheme.value},"
        f"gap={cv_config.gap},"
        f"purge_window={cv_config.purge_window},"
        f"min_train_samples={cv_config.min_train_samples}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def compute_universe_hash(feature_names: List[str]) -> str:
    """Compute stable hash for feature universe."""
    sorted_features = sorted(feature_names)
    feature_str = ",".join(sorted_features)
    return hashlib.md5(feature_str.encode()).hexdigest()[:12]


def compute_run_signature(
    universe_hash: str,
    cv_config_hash: str,
    date_range: Tuple[str, str],
    algorithm_params: Dict[str, Any],
    model_key: str,
) -> str:
    """Compute stable run signature for reproducibility tracking.

    The signature combines:
    - Feature universe hash
    - CV configuration hash
    - Date range of data
    - Algorithm parameters
    - Model key

    Returns:
        12-character hex hash
    """
    # Sort algorithm params for stability
    params_str = json.dumps(algorithm_params, sort_keys=True)

    signature_str = (
        f"universe={universe_hash},"
        f"cv={cv_config_hash},"
        f"dates={date_range[0]}_{date_range[1]},"
        f"params={params_str},"
        f"model={model_key}"
    )
    return hashlib.md5(signature_str.encode()).hexdigest()[:12]


# =============================================================================
# Core Selection Function
# =============================================================================

def run_single_model_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: ModelKey,
    features: List[str],
    sample_weight: Optional[pd.Series] = None,
    cv_config: Optional[CVConfig] = None,
    model_config: Optional[ModelConfig] = None,
    metric_config: Optional[MetricConfig] = None,
    pipeline_config: Optional[LooseTightConfig] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
) -> FeatureSelectionResult:
    """Run feature selection for a single model.

    This is the core reusable function that runs the Loose-Tight pipeline
    for a single model target.

    Args:
        X: Feature DataFrame with all candidate features
        y: Target Series (binary labels for this model)
        model_key: Model key (LONG_NORMAL, etc.)
        features: List of candidate feature names to consider
        sample_weight: Optional sample weights (e.g., overlap inverse)
        cv_config: Cross-validation configuration
        model_config: LightGBM/XGBoost configuration
        metric_config: Metrics configuration
        pipeline_config: Loose-Tight pipeline configuration
        verbose: Whether to print progress
        checkpoint_path: Custom checkpoint path

    Returns:
        FeatureSelectionResult with selected features and metrics
    """
    start_time = time.time()

    # Use defaults if not provided
    if cv_config is None:
        cv_config = CVConfig(
            n_splits=5,
            scheme=CVScheme.EXPANDING,
            gap=20,
            purge_window=0,
            min_train_samples=1000,
        )

    if model_config is None:
        model_config = ModelConfig(
            model_type=ModelType.LIGHTGBM,
            task_type=TaskType.CLASSIFICATION,
            params={
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            },
            num_threads=1,
            early_stopping_rounds=50,
            num_boost_round=300,
        )

    if metric_config is None:
        metric_config = MetricConfig(
            primary_metric=MetricType.AUC,
            secondary_metrics=[MetricType.AUPR, MetricType.LOG_LOSS],
            tail_quantile=0.1,
        )

    if pipeline_config is None:
        # Auto-detect CPU count for parallelism
        import os
        n_jobs = os.cpu_count() or 8
        pipeline_config = LooseTightConfig(
            run_base_elimination=False,
            epsilon_add_loose=0.0002,
            min_fold_improvement_ratio_loose=0.6,
            max_features_loose=80,
            epsilon_remove_strict=0.0,
            run_interactions=True,
            max_interactions=8,
            epsilon_add_interaction=0.001,
            epsilon_swap=0.0005,
            max_swap_iterations=50,
            n_jobs=n_jobs,
        )

    search_config = SearchConfig(n_jobs=1, random_state=42)

    # Filter X to valid features
    valid_features = [f for f in features if f in X.columns]
    X_subset = X[valid_features].copy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature Selection: {model_key.value.upper()}")
        print(f"{'='*60}")
        print(f"Candidate features: {len(valid_features)}")
        print(f"Samples: {len(X_subset)}")

    # Create and run pipeline
    pipeline = LooseTightPipeline(
        model_config=model_config,
        cv_config=cv_config,
        search_config=search_config,
        metric_config=metric_config,
        pipeline_config=pipeline_config,
        checkpoint_path=checkpoint_path,
    )

    # Align sample weights with X for pipeline
    # sample_weight must have same index as X for boolean masking in evaluation
    sample_weight_aligned = None
    if sample_weight is not None:
        if len(sample_weight) == len(X_subset):
            # Already aligned in length - set index to match X's index
            sample_weight_aligned = sample_weight.copy()
            sample_weight_aligned.index = X_subset.index
        else:
            # Length mismatch - try to align or create uniform weights
            try:
                sample_weight_aligned = sample_weight.loc[X_subset.index]
            except (KeyError, TypeError):
                # Fallback: create uniform weights with correct index
                # This handles cases where sample_weight has different length/index
                sample_weight_aligned = pd.Series(
                    np.ones(len(X_subset)),
                    index=X_subset.index
                )

    # Run pipeline
    pipeline.run(X_subset, y, verbose=verbose, sample_weight=sample_weight_aligned)

    elapsed = time.time() - start_time

    # Extract results
    best_features = pipeline.get_best_features()
    best_metric = pipeline.get_best_metric()

    # Compute hashes for reproducibility
    cv_hash = compute_cv_config_hash(cv_config)
    universe_hash = compute_universe_hash(valid_features)

    # Get date range
    date_min = str(X.index.min().date()) if hasattr(X.index.min(), 'date') else str(X.index.min())
    date_max = str(X.index.max().date()) if hasattr(X.index.max(), 'date') else str(X.index.max())
    date_range = (date_min, date_max)

    # Extract algorithm params
    algorithm_params = {
        'epsilon_add_loose': pipeline_config.epsilon_add_loose,
        'epsilon_remove_strict': pipeline_config.epsilon_remove_strict,
        'epsilon_add_interaction': pipeline_config.epsilon_add_interaction,
        'epsilon_swap': pipeline_config.epsilon_swap,
        'max_features_loose': pipeline_config.max_features_loose,
        'max_interactions': pipeline_config.max_interactions,
        'run_interactions': pipeline_config.run_interactions,
    }

    run_sig = compute_run_signature(
        universe_hash, cv_hash, date_range, algorithm_params, model_key.value
    )

    # Extract secondary metrics from best snapshot
    secondary_metrics = {}
    if pipeline.best_snapshot and pipeline.best_snapshot.fold_metrics:
        # Get extended metrics by re-evaluating (already done in pipeline)
        secondary_metrics['auc'] = (best_metric[0], best_metric[1])

    result = FeatureSelectionResult(
        model_key=model_key.value,
        selected_features=best_features,
        n_features=len(best_features),
        cv_auc_mean=best_metric[0],
        cv_auc_std=best_metric[1],
        fold_metrics=pipeline.best_snapshot.fold_metrics if pipeline.best_snapshot else [],
        secondary_metrics=secondary_metrics,
        best_stage=pipeline.best_snapshot.stage if pipeline.best_snapshot else "",
        algorithm="loose_tight_pipeline",
        algorithm_params=algorithm_params,
        cv_config_hash=cv_hash,
        date_range=date_range,
        universe_hash=universe_hash,
        timestamp=datetime.utcnow().isoformat() + "Z",
        run_signature=run_sig,
        elapsed_seconds=elapsed,
    )

    if verbose:
        print(f"\nResult: {len(best_features)} features, AUC={best_metric[0]:.4f}±{best_metric[1]:.4f}")
        print(f"Elapsed: {elapsed/60:.1f} minutes")

    return result


# =============================================================================
# Overlap Analysis
# =============================================================================

def compute_overlap_analysis(
    results: Dict[str, FeatureSelectionResult],
) -> MultiModelSelectionSummary:
    """Compute overlap analysis across multiple model results.

    Args:
        results: Dict mapping model_key -> FeatureSelectionResult

    Returns:
        MultiModelSelectionSummary with overlap analysis
    """
    model_keys = list(results.keys())

    # Compute feature sets per model
    feature_sets = {
        mk: set(results[mk].selected_features)
        for mk in model_keys
    }

    # Core features = intersection of all
    core_features = set.intersection(*feature_sets.values()) if feature_sets else set()

    # Head features = features unique to each model (in that model but not in core)
    head_features = {
        mk: sorted(feature_sets[mk] - core_features)
        for mk in model_keys
    }

    # Union size
    union_set = set.union(*feature_sets.values()) if feature_sets else set()
    union_size = len(union_set)

    # Overlap matrix (Jaccard similarity)
    overlap_matrix: Dict[str, Dict[str, float]] = {}
    intersection_matrix: Dict[str, Dict[str, int]] = {}

    for mk1 in model_keys:
        overlap_matrix[mk1] = {}
        intersection_matrix[mk1] = {}
        for mk2 in model_keys:
            intersection = len(feature_sets[mk1] & feature_sets[mk2])
            union = len(feature_sets[mk1] | feature_sets[mk2])
            jaccard = intersection / union if union > 0 else 0.0
            overlap_matrix[mk1][mk2] = round(jaccard, 4)
            intersection_matrix[mk1][mk2] = intersection

    # Top shared features (count how many models use each feature)
    feature_counts: Dict[str, int] = {}
    for mk in model_keys:
        for feat in feature_sets[mk]:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    # Sort by count descending
    top_shared = sorted(feature_counts.items(), key=lambda x: (-x[1], x[0]))

    # Compute combined run signature
    combined_sig = hashlib.md5(
        ",".join(sorted(r.run_signature for r in results.values())).encode()
    ).hexdigest()[:12]

    return MultiModelSelectionSummary(
        results=results,
        core_features=sorted(core_features),
        head_features=head_features,
        overlap_matrix=overlap_matrix,
        intersection_matrix=intersection_matrix,
        union_size=union_size,
        top_shared_features=top_shared[:20],  # Top 20
        run_signature=combined_sig,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# =============================================================================
# Artifact Writers
# =============================================================================

def write_per_model_artifacts(
    result: FeatureSelectionResult,
    output_dir: Path,
) -> None:
    """Write per-model artifacts (JSON + TXT).

    Args:
        result: Feature selection result for one model
        output_dir: Directory for this model's artifacts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON with full metadata
    json_path = output_dir / "selected_features.json"
    with open(json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    # Write TXT for convenience (feature list only)
    txt_path = output_dir / "selected_features.txt"
    with open(txt_path, 'w') as f:
        f.write(f"# Feature selection for {result.model_key}\n")
        f.write(f"# Stage: {result.best_stage}\n")
        f.write(f"# AUC: {result.cv_auc_mean:.4f} +/- {result.cv_auc_std:.4f}\n")
        f.write(f"# Run signature: {result.run_signature}\n")
        f.write(f"# Timestamp: {result.timestamp}\n\n")
        for feat in sorted(result.selected_features):
            f.write(f"{feat}\n")


def write_global_summary(
    summary: MultiModelSelectionSummary,
    output_dir: Path,
) -> None:
    """Write global summary JSON.

    Args:
        summary: Multi-model selection summary
        output_dir: Base output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)


# =============================================================================
# Validation
# =============================================================================

def validate_selected_features(
    selected_features: List[str],
    available_features: Set[str],
    base_features_registry: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """Validate selected features against available and registered features.

    Args:
        selected_features: List of selected feature names
        available_features: Set of features available in the data
        base_features_registry: Optional set of registered feature names

    Returns:
        Dict with 'valid', 'missing_from_data', 'not_in_registry' lists
    """
    selected_set = set(selected_features)

    valid = sorted(selected_set & available_features)
    missing_from_data = sorted(selected_set - available_features)

    not_in_registry = []
    if base_features_registry:
        not_in_registry = sorted(selected_set - base_features_registry)

    return {
        'valid': valid,
        'missing_from_data': missing_from_data,
        'not_in_registry': not_in_registry,
    }


# =============================================================================
# Auto-Update base_features.py
# =============================================================================

def generate_base_features_update(
    summary: MultiModelSelectionSummary,
    include_timestamp: bool = True,
) -> str:
    """Generate Python code to update CORE_FEATURES and HEAD_FEATURES.

    This generates the new content for the feature registry based on
    multi-model selection results.

    Args:
        summary: Multi-model selection summary with core/head features
        include_timestamp: Whether to include timestamp in comments

    Returns:
        Python code string for the updated feature definitions
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Categorize core features by domain (rough heuristic based on naming)
    def categorize_feature(feat: str) -> str:
        feat_lower = feat.lower()
        if any(x in feat_lower for x in ['alpha', 'beta', 'rel_strength', 'xsec']):
            return "RELATIVE PERFORMANCE / ALPHA"
        elif any(x in feat_lower for x in ['fred_', 'vix', 'copper', 'gold', 'qqq_spy', 'equity_bond', 'credit', 'cyclical']):
            return "MACRO / INTERMARKET"
        elif any(x in feat_lower for x in ['trend', 'slope', 'macd']):
            return "TREND STRENGTH"
        elif any(x in feat_lower for x in ['pct_dist', 'pos_in', 'days_since', 'relative_dist']):
            return "PRICE POSITION / MEAN REVERSION"
        elif any(x in feat_lower for x in ['vol_regime', 'atr', 'bb_width', 'rv_z', 'squeeze']):
            return "VOLATILITY / REGIME"
        elif any(x in feat_lower for x in ['breadth', 'mcclellan', 'ad_line']):
            return "SECTOR BREADTH"
        elif any(x in feat_lower for x in ['shadow', 'vwap', 'volshock', 'volume', 'obv', 'pv_div']):
            return "VOLUME / LIQUIDITY"
        elif any(x in feat_lower for x in ['rsi', 'chop', 'adx', 'di_', 'gap']):
            return "MOMENTUM / TREND QUALITY"
        elif any(x in feat_lower for x in ['drawdown', 'recovery']):
            return "DRAWDOWN / RECOVERY"
        elif any(x in feat_lower for x in ['overnight', 'range_', 'breakout']):
            return "RANGE / BREAKOUT"
        else:
            return "OTHER"

    # Group core features by category
    core_by_category: Dict[str, List[str]] = {}
    for feat in sorted(summary.core_features):
        cat = categorize_feature(feat)
        if cat not in core_by_category:
            core_by_category[cat] = []
        core_by_category[cat].append(feat)

    # Generate CORE_FEATURES code
    lines = []
    lines.append("# =============================================================================")
    lines.append(f"# CORE FEATURES - Shared backbone across all models ({len(summary.core_features)} features)")
    lines.append("# =============================================================================")
    if include_timestamp:
        lines.append(f"# Auto-generated from multi-model selection: {timestamp}")
        lines.append(f"# Run signature: {summary.run_signature}")
    lines.append("# These features represent the intersection of all 4 model selections.")
    lines.append("")
    lines.append("CORE_FEATURES: List[str] = [")

    # Category order for nice grouping
    category_order = [
        "RELATIVE PERFORMANCE / ALPHA",
        "MACRO / INTERMARKET",
        "TREND STRENGTH",
        "PRICE POSITION / MEAN REVERSION",
        "VOLATILITY / REGIME",
        "SECTOR BREADTH",
        "VOLUME / LIQUIDITY",
        "MOMENTUM / TREND QUALITY",
        "DRAWDOWN / RECOVERY",
        "RANGE / BREAKOUT",
        "OTHER",
    ]

    for cat in category_order:
        if cat in core_by_category and core_by_category[cat]:
            features = core_by_category[cat]
            lines.append(f"    # === {cat} ({len(features)} features) ===")
            for feat in features:
                lines.append(f'    "{feat}",')
            lines.append("")

    lines.append("]")
    lines.append("")

    # Generate HEAD_FEATURES code
    lines.append("")
    lines.append("# =============================================================================")
    lines.append("# HEAD FEATURES - Model-specific additive features")
    lines.append("# =============================================================================")
    if include_timestamp:
        lines.append(f"# Auto-generated from multi-model selection: {timestamp}")
    lines.append("# These are features selected by specific models but not in CORE.")
    lines.append("")
    lines.append("HEAD_FEATURES: Dict[ModelKey, List[str]] = {")

    model_descriptions = {
        "long_normal": "LONG_NORMAL: Standard long momentum (1.5 ATR style)",
        "long_parabolic": "LONG_PARABOLIC: Extended momentum / trend persistence",
        "short_normal": "SHORT_NORMAL: Breakdown / fragility / liquidity stress",
        "short_parabolic": "SHORT_PARABOLIC: Panic / regime shift / vol-of-vol",
    }

    for model_key in ["long_normal", "long_parabolic", "short_normal", "short_parabolic"]:
        head_feats = summary.head_features.get(model_key, [])
        desc = model_descriptions.get(model_key, model_key)
        model_enum = f"ModelKey.{model_key.upper()}"

        lines.append(f"    # {desc}")
        lines.append(f"    {model_enum}: [")
        for feat in sorted(head_feats):
            lines.append(f'        "{feat}",')
        lines.append("    ],")
        lines.append("")

    lines.append("}")

    return "\n".join(lines)


def update_base_features_file(
    summary: MultiModelSelectionSummary,
    base_features_path: Optional[Path] = None,
    backup: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Update base_features.py with new CORE_FEATURES and HEAD_FEATURES.

    This replaces the CORE_FEATURES and HEAD_FEATURES definitions in the
    base_features.py file based on multi-model selection results.

    Args:
        summary: Multi-model selection summary with core/head features
        base_features_path: Path to base_features.py (default: auto-detect)
        backup: Whether to create a backup before modifying
        dry_run: If True, don't actually modify the file
        verbose: Whether to print progress

    Returns:
        Dict with 'success', 'backup_path', 'changes' info
    """
    import re
    import shutil

    # Default path
    if base_features_path is None:
        base_features_path = Path(__file__).parent / "base_features.py"

    if not base_features_path.exists():
        raise FileNotFoundError(f"base_features.py not found at {base_features_path}")

    # Read current content
    with open(base_features_path, 'r') as f:
        content = f.read()

    # Create backup
    backup_path = None
    if backup and not dry_run:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = base_features_path.with_suffix(f".py.backup_{timestamp}")
        shutil.copy(base_features_path, backup_path)
        if verbose:
            print(f"Created backup: {backup_path}")

    # Generate new feature definitions
    new_definitions = generate_base_features_update(summary)

    # Find and replace CORE_FEATURES block
    # Pattern matches from "# ===...CORE FEATURES" to end of CORE_FEATURES list
    core_pattern = r'(# =+\n# CORE FEATURES.*?^CORE_FEATURES: List\[str\] = \[.*?^\])'
    core_match = re.search(core_pattern, content, re.MULTILINE | re.DOTALL)

    if not core_match:
        # Try simpler pattern
        core_pattern = r'(CORE_FEATURES: List\[str\] = \[.*?^\])'
        core_match = re.search(core_pattern, content, re.MULTILINE | re.DOTALL)

    # Find and replace HEAD_FEATURES block
    head_pattern = r'(# =+\n# HEAD FEATURES.*?^HEAD_FEATURES: Dict\[ModelKey, List\[str\]\] = \{.*?^\})'
    head_match = re.search(head_pattern, content, re.MULTILINE | re.DOTALL)

    if not head_match:
        # Try simpler pattern
        head_pattern = r'(HEAD_FEATURES: Dict\[ModelKey, List\[str\]\] = \{.*?^\})'
        head_match = re.search(head_pattern, content, re.MULTILINE | re.DOTALL)

    # Extract the new definitions parts
    new_core_section = new_definitions.split("# =============================================================================\n# HEAD FEATURES")[0]
    new_head_section = "# =============================================================================\n# HEAD FEATURES" + new_definitions.split("# =============================================================================\n# HEAD FEATURES")[1]

    changes = {
        'core_features_old_count': None,
        'core_features_new_count': len(summary.core_features),
        'head_features_updated': list(summary.head_features.keys()),
    }

    new_content = content

    if core_match:
        # Count old features
        old_core = core_match.group(1)
        old_count = old_core.count('",')
        changes['core_features_old_count'] = old_count

        new_content = new_content[:core_match.start()] + new_core_section.strip() + new_content[core_match.end():]

        if verbose:
            print(f"CORE_FEATURES: {old_count} -> {len(summary.core_features)} features")
    else:
        if verbose:
            print("Warning: Could not find CORE_FEATURES block to replace")

    # Re-find HEAD_FEATURES in potentially modified content
    head_match = re.search(head_pattern, new_content, re.MULTILINE | re.DOTALL)
    if head_match:
        new_content = new_content[:head_match.start()] + new_head_section.strip() + new_content[head_match.end():]

        if verbose:
            for mk, feats in summary.head_features.items():
                print(f"HEAD_FEATURES[{mk}]: {len(feats)} features")
    else:
        if verbose:
            print("Warning: Could not find HEAD_FEATURES block to replace")

    # Write new content
    if not dry_run:
        with open(base_features_path, 'w') as f:
            f.write(new_content)
        if verbose:
            print(f"Updated {base_features_path}")

    return {
        'success': True,
        'backup_path': str(backup_path) if backup_path else None,
        'changes': changes,
        'dry_run': dry_run,
    }


def compute_feature_diff(
    old_core: List[str],
    old_head: Dict[str, List[str]],
    new_summary: MultiModelSelectionSummary,
) -> Dict[str, Any]:
    """Compute the diff between old and new feature sets.

    Args:
        old_core: Previous CORE_FEATURES list
        old_head: Previous HEAD_FEATURES dict
        new_summary: New multi-model selection summary

    Returns:
        Dict with added/removed features for core and each model
    """
    old_core_set = set(old_core)
    new_core_set = set(new_summary.core_features)

    diff = {
        'core': {
            'added': sorted(new_core_set - old_core_set),
            'removed': sorted(old_core_set - new_core_set),
            'unchanged': sorted(old_core_set & new_core_set),
        },
        'head': {},
    }

    for model_key in ['long_normal', 'long_parabolic', 'short_normal', 'short_parabolic']:
        old_set = set(old_head.get(model_key, []))
        new_set = set(new_summary.head_features.get(model_key, []))

        diff['head'][model_key] = {
            'added': sorted(new_set - old_set),
            'removed': sorted(old_set - new_set),
            'unchanged': sorted(old_set & new_set),
        }

    return diff


def print_feature_diff(diff: Dict[str, Any]) -> None:
    """Print a formatted feature diff."""
    print("\n" + "=" * 60)
    print("FEATURE REGISTRY UPDATE DIFF")
    print("=" * 60)

    # Core features
    print("\nCORE_FEATURES:")
    print(f"  Added ({len(diff['core']['added'])}):   {', '.join(diff['core']['added'][:5])}" +
          (f"... +{len(diff['core']['added'])-5} more" if len(diff['core']['added']) > 5 else ""))
    print(f"  Removed ({len(diff['core']['removed'])}): {', '.join(diff['core']['removed'][:5])}" +
          (f"... +{len(diff['core']['removed'])-5} more" if len(diff['core']['removed']) > 5 else ""))
    print(f"  Unchanged: {len(diff['core']['unchanged'])}")

    # Head features per model
    for model_key in ['long_normal', 'long_parabolic', 'short_normal', 'short_parabolic']:
        head_diff = diff['head'][model_key]
        print(f"\nHEAD_FEATURES[{model_key}]:")
        print(f"  Added ({len(head_diff['added'])}):   {', '.join(head_diff['added'][:3])}" +
              (f"..." if len(head_diff['added']) > 3 else ""))
        print(f"  Removed ({len(head_diff['removed'])}): {', '.join(head_diff['removed'][:3])}" +
              (f"..." if len(head_diff['removed']) > 3 else ""))

    print("\n" + "=" * 60)
