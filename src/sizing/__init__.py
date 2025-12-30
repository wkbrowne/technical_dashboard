"""Multi-model position sizing module.

This module implements a multi-model sizing engine that supports:
- Four separate models: LONG_NORMAL, LONG_PARABOLIC, SHORT_NORMAL, SHORT_PARABOLIC
- Combining policies: mode_priority (default) or blend
- Regime gating overlay for risk control
- TPE optimization of sizing and gating parameters

Key components:
- config.py: Configuration dataclasses and schema
- multi_model.py: Multi-model sizing engine
- regime_gating.py: Regime-based exposure gating
- predictions.py: Multi-model prediction data interface
"""

from .config import (
    ModelType,
    CombinePolicy,
    MultiModelSizingConfig,
    RegimeGatingConfig,
    load_multi_model_config,
    save_multi_model_config,
)
from .multi_model import (
    MultiModelSizingEngine,
    compute_edge_scores,
    combine_model_weights_with_justification,
    compute_model_weights_with_justification,
    create_decision_log,
    get_justification_columns,
    JUSTIFY_PREFIX,
)
from .regime_gating import (
    RegimeMultipliers,
    GatingJustification,
    apply_regime_gating,
    apply_regime_gating_vectorized,
    compute_regime_multipliers,
    suggest_gating_params,
    suggest_short_selectivity_params,
    create_gating_config_from_params,
    get_gating_diagnostics,
)
from .predictions import (
    load_multi_model_predictions,
    validate_predictions_format,
    convert_long_to_wide_predictions,
)

__all__ = [
    # Config
    "ModelType",
    "CombinePolicy",
    "MultiModelSizingConfig",
    "RegimeGatingConfig",
    "load_multi_model_config",
    "save_multi_model_config",
    # Multi-model engine
    "MultiModelSizingEngine",
    "compute_edge_scores",
    "combine_model_weights_with_justification",
    "compute_model_weights_with_justification",
    "create_decision_log",
    "get_justification_columns",
    "JUSTIFY_PREFIX",
    # Regime gating
    "RegimeMultipliers",
    "GatingJustification",
    "apply_regime_gating",
    "apply_regime_gating_vectorized",
    "compute_regime_multipliers",
    "suggest_gating_params",
    "suggest_short_selectivity_params",
    "create_gating_config_from_params",
    "get_gating_diagnostics",
    # Predictions
    "load_multi_model_predictions",
    "validate_predictions_format",
    "convert_long_to_wide_predictions",
]
