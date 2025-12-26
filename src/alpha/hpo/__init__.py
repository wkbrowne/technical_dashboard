"""
Hyperparameter Optimization (HPO) module.

Provides multi-round HPO with:
- Future-biased pruning on recent CV folds
- Quantile-based search space refinement
- Optional importance-based parameter freezing
"""

from .refinement import (
    SearchSpace,
    SearchSpaceParam,
    refine_search_space,
    compute_elite_quantiles,
)
from .pruning import (
    FoldEvaluator,
    PruneFoldResult,
    should_prune_trial,
)
from .artifacts import (
    HPOArtifactWriter,
    HPORoundResult,
)

__all__ = [
    'SearchSpace',
    'SearchSpaceParam',
    'refine_search_space',
    'compute_elite_quantiles',
    'FoldEvaluator',
    'PruneFoldResult',
    'should_prune_trial',
    'HPOArtifactWriter',
    'HPORoundResult',
]
