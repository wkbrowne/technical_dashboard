"""
HPO (Hyperparameter Optimization) Configuration.

Defines settings for:
- Future-biased pruning (evaluate on recent folds)
- Multi-round search space refinement
- CV score weighting
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class RefineMethod(Enum):
    """Method for refining search space between rounds."""
    QUANTILE = "quantile"  # Simple: shrink ranges using elite quantiles
    IMPORTANCE = "importance"  # Advanced: use parameter importance + quantiles


class PruneMetric(Enum):
    """Metric for pruning decisions on prune folds."""
    MEAN = "mean"  # Average AUC across prune folds
    MIN = "min"  # Minimum AUC across prune folds


class CVScoreMethod(Enum):
    """Method for aggregating CV fold scores into final objective."""
    MEAN = "mean"  # Simple mean across all folds
    WEIGHTED = "weighted"  # Weight later folds more heavily


@dataclass
class PruningConfig:
    """
    Configuration for trial pruning.

    Future-biased pruning evaluates trials on recent folds (default: 3,4)
    to avoid optimizing for older, potentially stale regimes.
    """
    # Which folds to evaluate for pruning decisions (0-indexed, chronological)
    # Default [3, 4] = last two folds of a 5-fold CV
    prune_folds: List[int] = field(default_factory=lambda: [3, 4])

    # How to aggregate prune fold AUCs
    prune_metric: PruneMetric = PruneMetric.MEAN

    # Prune if prune_metric < best_prune_metric - prune_margin
    # (relative pruning after min_completed_trials)
    prune_margin: float = 0.002

    # Minimum trials before relative pruning kicks in
    min_completed_trials: int = 30

    # Whether to evaluate all folds for surviving trials (default: True)
    eval_all_folds: bool = True

    # Absolute thresholds (applied regardless of relative pruning)
    min_auc: float = 0.54
    max_brier: float = 0.26
    max_cv_coef: float = 0.20


@dataclass
class CVScoreConfig:
    """
    Configuration for aggregating CV scores.

    Weighted scoring favors later (more recent) folds.
    """
    # Scoring method
    method: CVScoreMethod = CVScoreMethod.MEAN

    # Weights for each fold (index 0 = oldest fold)
    # Used only when method=WEIGHTED
    # Default: [0.5, 0.75, 1.0, 1.25, 1.5] for 5 folds
    fold_weights: Optional[List[float]] = None

    def get_weights(self, n_folds: int) -> List[float]:
        """Get fold weights, generating defaults if not specified."""
        if self.fold_weights is not None:
            if len(self.fold_weights) != n_folds:
                raise ValueError(
                    f"fold_weights length ({len(self.fold_weights)}) != n_folds ({n_folds})"
                )
            return self.fold_weights

        # Generate default weights favoring later folds
        # For 5 folds: [0.5, 0.75, 1.0, 1.25, 1.5]
        base = 0.5
        step = 1.0 / (n_folds - 1) if n_folds > 1 else 0
        return [base + i * step for i in range(n_folds)]


@dataclass
class RefinementConfig:
    """
    Configuration for multi-round search space refinement.

    Simple default mode uses quantile-based refinement only.
    Advanced mode adds parameter importance analysis.
    """
    # Number of HPO rounds (each round narrows search space)
    rounds: int = 2

    # Trials per round
    trials_per_round: int = 200

    # Refinement method
    refine_method: RefineMethod = RefineMethod.QUANTILE

    # Fraction of top trials to consider "elite" for refinement
    elite_frac: float = 0.15

    # Quantiles for numeric parameter bounds [low, high]
    refine_quantiles: tuple = (0.1, 0.9)

    # Padding factor to avoid premature collapse
    # New range = quantile_range * (1 + 2*padding)
    refine_padding: float = 0.10

    # Minimum range fraction relative to original bounds
    # Prevents shrinking too aggressively
    min_range_frac: float = 0.20

    # --- Importance-based settings (only used when refine_method=IMPORTANCE) ---

    # Freeze parameters with importance below this threshold
    freeze_low_importance: bool = True
    importance_threshold: float = 0.02

    # Whether to allow dropping categorical options to singletons
    allow_singleton_cats: bool = False

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class HPOConfig:
    """
    Complete HPO configuration combining all settings.
    """
    # Pruning settings
    pruning: PruningConfig = field(default_factory=PruningConfig)

    # CV score aggregation
    cv_score: CVScoreConfig = field(default_factory=CVScoreConfig)

    # Multi-round refinement
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    # CV settings
    n_folds: int = 5
    gap: int = 20  # Embargo gap in days

    # Objective settings
    objective_mode: str = "composite"  # "composite" or "auc"
    variance_penalty: float = 1.0  # For auc mode only

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pruning': asdict(self.pruning),
            'cv_score': asdict(self.cv_score),
            'refinement': asdict(self.refinement),
            'n_folds': self.n_folds,
            'gap': self.gap,
            'objective_mode': self.objective_mode,
            'variance_penalty': self.variance_penalty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HPOConfig":
        """Create from dictionary."""
        pruning = PruningConfig(**d.get('pruning', {}))
        cv_score = CVScoreConfig(**d.get('cv_score', {}))
        refinement = RefinementConfig(**d.get('refinement', {}))
        return cls(
            pruning=pruning,
            cv_score=cv_score,
            refinement=refinement,
            n_folds=d.get('n_folds', 5),
            gap=d.get('gap', 20),
            objective_mode=d.get('objective_mode', 'composite'),
            variance_penalty=d.get('variance_penalty', 1.0),
        )


# Default configurations for different use cases

def get_default_hpo_config() -> HPOConfig:
    """Default HPO config with future-biased pruning and 2-round quantile refinement."""
    return HPOConfig()


def get_legacy_hpo_config() -> HPOConfig:
    """
    Legacy HPO config matching old behavior.

    - Prunes on fold 0 only
    - No multi-round refinement
    - Mean CV scoring
    """
    return HPOConfig(
        pruning=PruningConfig(
            prune_folds=[0],  # Old behavior: prune on first fold
            prune_margin=0.0,  # No relative pruning
            min_completed_trials=0,
        ),
        cv_score=CVScoreConfig(
            method=CVScoreMethod.MEAN,
        ),
        refinement=RefinementConfig(
            rounds=1,  # Single round = no refinement
            trials_per_round=200,
        ),
    )


def get_aggressive_hpo_config() -> HPOConfig:
    """
    Aggressive HPO config for thorough exploration.

    - 3 rounds with importance-based refinement
    - 150 trials per round
    - Weighted CV scoring
    """
    return HPOConfig(
        pruning=PruningConfig(
            prune_folds=[3, 4],
            prune_metric=PruneMetric.MEAN,
            prune_margin=0.003,
            min_completed_trials=50,
        ),
        cv_score=CVScoreConfig(
            method=CVScoreMethod.WEIGHTED,
        ),
        refinement=RefinementConfig(
            rounds=3,
            trials_per_round=150,
            refine_method=RefineMethod.IMPORTANCE,
            elite_frac=0.15,
        ),
    )


# Suggested budget recommendations (for documentation)
BUDGET_RECOMMENDATIONS = {
    'lgbm_per_model': {
        'description': 'LightGBM HPO per model',
        'rounds': 3,
        'trials_per_round': 150,
        'total_trials': 450,
    },
    'lgbm_quick': {
        'description': 'Quick LightGBM HPO',
        'rounds': 2,
        'trials_per_round': 100,
        'total_trials': 200,
    },
    'sizing_hpo': {
        'description': 'Position sizing HPO (faster objective)',
        'rounds': 3,
        'trials_per_round': 250,
        'total_trials': 750,
    },
}
