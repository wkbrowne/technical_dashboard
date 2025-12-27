"""Interaction feature production for model training.

This module computes interaction features that are registered in HEAD_FEATURES
and CORE_FEATURES but need to be generated from base features.

It bridges the gap between:
1. Feature selection (which discovers and validates interaction features)
2. Feature production (which computes all features including interactions)
3. Model training (which expects all features to be present in DataFrame)

Interaction features are identified by naming conventions:
- PRODUCT: feat_a_x_feat_b (multiplicative)
- GATED: feat_a_gated_feat_b (sign-gated)
- RATIO: feat_a_div_feat_b (scale-invariant ratio)
- THRESHOLD: feat_a_AND_feat_b_high (binary joint threshold)

Usage:
    from src.features.interaction_production import (
        compute_registered_interactions,
        parse_interaction_name,
        get_registered_interaction_features,
    )

    # Get list of interaction features from model registry
    interactions = get_registered_interaction_features(ModelKey.LONG_NORMAL)

    # Compute interactions and add to DataFrame
    df = compute_registered_interactions(df, interactions)
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# INTERACTION PARSING
# =============================================================================

# Patterns for parsing interaction feature names
# Order matters: check the unique/unambiguous patterns first
#
# NOTE: We use specific separators that are designed to be unambiguous:
# - "_x_" for product (e.g., feat_a_x_feat_b) - most common
# - "_gated_" for gated (e.g., feat_a_gated_feat_b) - signal gating
# - "_ratio_" for ratio (e.g., feat_a_ratio_feat_b) - scale-invariant
# - "_AND_..._high" for threshold (e.g., feat_a_AND_feat_b_high) - joint threshold
#
# TEMPLATE-BASED INTERACTIONS (new):
# - ix__{template}__{base}__gated__{gate} - template-based gated
# - ix__{template}__{feat_a}__x__{feat_b} - template-based product
#
# IMPORTANT: Check _x_ first because base features may contain "_ratio_" in their name
# (e.g., gold_spy_ratio_zscore is a base feature, not an interaction)
#
# Pattern matching strategy:
# - For _x_, _gated_, _AND_: These are very specific and unlikely to appear in base feature names
# - For _ratio_: Only match if the feature name doesn't contain other interaction markers
# - Template-based patterns start with "ix__" prefix

# Template-based interaction patterns (new format)
TEMPLATE_PATTERNS = [
    # Template gated: ix__{template}__{base}__gated__{gate}
    (r'^ix__([^_]+(?:_[^_]+)*)__(.+)__gated__(.+)$', 'template_gated'),
    # Template signed gate: ix__{template}__{base}__signed__{gate}
    (r'^ix__([^_]+(?:_[^_]+)*)__(.+)__signed__(.+)$', 'template_signed_gate'),
    # Template product: ix__{template}__{feat_a}__x__{feat_b}
    (r'^ix__([^_]+(?:_[^_]+)*)__(.+)__x__(.+)$', 'template_product'),
]

# Legacy interaction patterns
INTERACTION_PATTERNS = [
    # PRODUCT: feat_a_x_feat_b - check first as it's the most common
    # and other interaction types shouldn't contain _x_
    (r'^(.+)_x_(.+)$', 'product'),
    # THRESHOLD: feat_a_AND_feat_b_high (specific suffix)
    (r'^(.+)_AND_(.+)_high$', 'threshold'),
    # GATED: feat_a_gated_feat_b
    (r'^(.+)_gated_(.+)$', 'gated'),
    # RATIO: feat_a_ratio_feat_b - only if no other interaction markers present
    # Note: This is checked last because base features like "gold_spy_ratio_zscore"
    # contain "_ratio_" but are not interactions
    (r'^(.+)_ratio_(.+)$', 'ratio'),
]

# Known base feature name patterns that contain "_ratio_" but are NOT interactions
# These are excluded from ratio pattern matching
RATIO_FALSE_POSITIVES = {
    'gold_spy_ratio_zscore',
    'copper_gold_ratio',
    'cyclical_defensive_ratio',
    'financials_utilities_ratio',
    'tech_spy_ratio',
    'gap_atr_ratio',
    'gap_atr_ratio_raw',
    'overnight_ratio',
}


def parse_template_interaction_name(
    feature_name: str
) -> Optional[Tuple[str, str, str, str]]:
    """Parse a template-based interaction feature name.

    Template interactions follow the format:
    - ix__{template}__{base_feat}__gated__{gate_feat}
    - ix__{template}__{base_feat}__signed__{gate_feat}
    - ix__{template}__{feat_a}__x__{feat_b}

    Args:
        feature_name: The interaction feature name

    Returns:
        Tuple of (template_name, feature_a, feature_b, interaction_type) if parseable,
        None if not a template-based interaction.

    Examples:
        >>> parse_template_interaction_name('ix__momentum_x_vol_gate__rsi_14__gated__vol_regime_ema10')
        ('momentum_x_vol_gate', 'rsi_14', 'vol_regime_ema10', 'template_gated')
    """
    if not feature_name.startswith('ix__'):
        return None

    for pattern, interaction_type in TEMPLATE_PATTERNS:
        match = re.match(pattern, feature_name)
        if match:
            template_name = match.group(1)
            feat_a = match.group(2)
            feat_b = match.group(3)
            return (template_name, feat_a, feat_b, interaction_type)

    return None


def parse_interaction_name(
    feature_name: str
) -> Optional[Tuple[str, str, str]]:
    """Parse an interaction feature name to extract components.

    Identifies the two base features and interaction type from the naming
    convention used by the feature selection system. Handles both legacy
    and template-based interaction patterns.

    Args:
        feature_name: The interaction feature name (e.g., 'rsi_14_x_vol_regime_ema10')

    Returns:
        Tuple of (feature_a, feature_b, interaction_type) if parseable,
        None if not an interaction feature.

    Examples:
        >>> parse_interaction_name('gold_spy_ratio_zscore_x_w_rsp_spy_cumret_12')
        ('gold_spy_ratio_zscore', 'w_rsp_spy_cumret_12', 'product')

        >>> parse_interaction_name('momentum_gated_vol_regime')
        ('momentum', 'vol_regime', 'gated')

        >>> parse_interaction_name('ix__momentum_x_vol_gate__rsi_14__gated__vol_regime_ema10')
        ('rsi_14', 'vol_regime_ema10', 'gated')

        >>> parse_interaction_name('rsi_14')  # Not an interaction
        None

        >>> parse_interaction_name('gold_spy_ratio_zscore')  # Base feature, not interaction
        None
    """
    # Check if this is a known base feature that contains pattern-like substrings
    if feature_name in RATIO_FALSE_POSITIVES:
        return None

    # First check template-based patterns (new format with ix__ prefix)
    template_parsed = parse_template_interaction_name(feature_name)
    if template_parsed is not None:
        template_name, feat_a, feat_b, interaction_type = template_parsed
        # Map template interaction types to base types
        type_map = {
            'template_gated': 'gated',
            'template_signed_gate': 'signed_gate',
            'template_product': 'product',
        }
        return (feat_a, feat_b, type_map.get(interaction_type, 'product'))

    # Legacy patterns
    for pattern, interaction_type in INTERACTION_PATTERNS:
        match = re.match(pattern, feature_name)
        if match:
            feat_a = match.group(1)
            feat_b = match.group(2)

            # For ratio interactions, validate that the parts aren't false positives
            if interaction_type == 'ratio':
                # If the parsed feat_a is a known false positive prefix, skip
                if feat_a in RATIO_FALSE_POSITIVES:
                    continue
                # If the full name minus suffix matches a false positive, skip
                if any(feature_name.startswith(fp) for fp in RATIO_FALSE_POSITIVES):
                    # Check if this looks like a valid interaction
                    # Valid: gold_spy_ratio_zscore_x_other (product of false positive)
                    # Invalid: gold_spy_ratio_zscore alone
                    pass  # Continue to match, this is valid

            return (feat_a, feat_b, interaction_type)

    return None


def is_interaction_feature(feature_name: str) -> bool:
    """Check if a feature name represents an interaction feature.

    Args:
        feature_name: The feature name to check.

    Returns:
        True if the feature follows interaction naming conventions.
    """
    return parse_interaction_name(feature_name) is not None


def get_interaction_base_features(feature_name: str) -> Set[str]:
    """Get the set of base features required for an interaction.

    Args:
        feature_name: The interaction feature name.

    Returns:
        Set of base feature names needed, or empty set if not an interaction.
    """
    parsed = parse_interaction_name(feature_name)
    if parsed:
        return {parsed[0], parsed[1]}
    return set()


# =============================================================================
# INTERACTION COMPUTATION
# =============================================================================

def compute_interaction(
    df: pd.DataFrame,
    feat_a: str,
    feat_b: str,
    interaction_type: str = 'product',
    invert_gate: bool = False
) -> pd.Series:
    """Compute a single interaction feature.

    Supports five interaction types based on economic intuition:
    - PRODUCT: f1 * f2 - multiplicative amplification
    - GATED: f1 * I(f2 > median) - signal gated by binary condition
    - SIGNED_GATE: f1 * sign(f2) - signal gated by direction
    - RATIO: f1 / (|f2| + epsilon) - scale-invariant comparison
    - THRESHOLD: I(f1 > median) * I(f2 > median) - non-linear regime switching

    Args:
        df: DataFrame containing the base features.
        feat_a: First feature name (base signal).
        feat_b: Second feature name (gate/modifier).
        interaction_type: Type of interaction.
        invert_gate: If True, invert the gate logic (for stress indicators).

    Returns:
        Series containing the computed interaction values.

    Raises:
        KeyError: If base features are not in DataFrame.
        ValueError: If unknown interaction type.
    """
    if feat_a not in df.columns:
        raise KeyError(f"Feature '{feat_a}' not found in DataFrame")
    if feat_b not in df.columns:
        raise KeyError(f"Feature '{feat_b}' not found in DataFrame")

    a = df[feat_a]
    b = df[feat_b]

    if interaction_type == 'product':
        # Simple product interaction
        # Use case: Confirmation patterns where both signals reinforce
        values = a * b

    elif interaction_type == 'gated':
        # Binary gated interaction - feature a is active only when b > median
        # Use case: Regime conditioning where b determines if a is reliable
        b_median = b.median()
        if invert_gate:
            # Invert: gate is ON when b < median (e.g., low stress = reliable)
            b_indicator = np.where(b < b_median, 1.0, 0.0)
        else:
            # Normal: gate is ON when b > median
            b_indicator = np.where(b > b_median, 1.0, 0.0)
        values = a * b_indicator

    elif interaction_type == 'signed_gate':
        # Signed gated interaction - feature a is multiplied by sign of b
        # Use case: Direction matters (bullish vs bearish regime)
        if invert_gate:
            values = a * (-np.sign(b))
        else:
            values = a * np.sign(b)

    elif interaction_type == 'threshold':
        # Binary threshold interaction (both above median = 1, else 0)
        # Use case: Non-linear regime switching - only fire when both conditions met
        thresh_a = a.median()
        thresh_b = b.median()
        values = ((a > thresh_a) & (b > thresh_b)).astype(np.float32)

    elif interaction_type == 'ratio':
        # Ratio interaction (with protection against division by zero)
        # Use case: Scale-invariant comparisons - how big is a relative to b?
        values = a / (b.abs() + 1e-8)
        # Clip extreme values to avoid numerical issues
        values = values.clip(-100, 100)

    else:
        raise ValueError(
            f"Unknown interaction type: {interaction_type}. "
            f"Valid types: product, gated, signed_gate, threshold, ratio"
        )

    return values.astype(np.float32)


def compute_registered_interactions(
    df: pd.DataFrame,
    interaction_features: List[str],
    skip_missing: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """Compute interaction features and add them to the DataFrame.

    Parses each interaction feature name to extract base features and
    interaction type, then computes and adds the interaction column.

    Args:
        df: DataFrame containing base features.
        interaction_features: List of interaction feature names to compute.
        skip_missing: If True, skip interactions where base features are missing.
                      If False, raise KeyError for missing features.
        inplace: If True, modify df in place. If False, return a copy.

    Returns:
        DataFrame with interaction features added.

    Example:
        >>> interactions = ['gold_spy_ratio_zscore_x_w_rsp_spy_cumret_12']
        >>> df = compute_registered_interactions(df, interactions)
    """
    if not inplace:
        df = df.copy()

    computed = []
    skipped = []
    failed = []

    for feat_name in interaction_features:
        # Skip if already computed
        if feat_name in df.columns:
            continue

        parsed = parse_interaction_name(feat_name)
        if parsed is None:
            logger.debug(f"Not an interaction feature: {feat_name}")
            continue

        feat_a, feat_b, interaction_type = parsed

        # Check if base features exist
        if feat_a not in df.columns or feat_b not in df.columns:
            if skip_missing:
                skipped.append({
                    'interaction': feat_name,
                    'missing': [f for f in [feat_a, feat_b] if f not in df.columns]
                })
                continue
            else:
                raise KeyError(
                    f"Cannot compute {feat_name}: missing base features "
                    f"(need {feat_a}, {feat_b})"
                )

        try:
            values = compute_interaction(df, feat_a, feat_b, interaction_type)
            df[feat_name] = values
            computed.append(feat_name)
        except Exception as e:
            logger.warning(f"Failed to compute {feat_name}: {e}")
            failed.append({'interaction': feat_name, 'error': str(e)})

    if computed:
        logger.info(f"Computed {len(computed)} interaction features")
    if skipped:
        logger.debug(f"Skipped {len(skipped)} interactions (missing base features)")
    if failed:
        logger.warning(f"Failed to compute {len(failed)} interactions")

    return df


# =============================================================================
# REGISTRY INTEGRATION
# =============================================================================

def get_registered_interaction_features(
    model_key=None,
    include_core: bool = True,
    include_head: bool = True
) -> List[str]:
    """Get interaction features registered in the feature registry.

    Scans CORE_FEATURES and HEAD_FEATURES to find features that follow
    interaction naming conventions.

    Args:
        model_key: Optional ModelKey to get HEAD_FEATURES for specific model.
                   If None, checks all HEAD_FEATURES.
        include_core: Whether to check CORE_FEATURES.
        include_head: Whether to check HEAD_FEATURES.

    Returns:
        List of interaction feature names found in the registry.
    """
    # Import here to avoid circular imports
    from src.feature_selection.base_features import CORE_FEATURES, HEAD_FEATURES

    interactions = []

    if include_core:
        for feat in CORE_FEATURES:
            if is_interaction_feature(feat):
                interactions.append(feat)

    if include_head:
        if model_key is not None:
            head_features = HEAD_FEATURES.get(model_key, [])
            for feat in head_features:
                if is_interaction_feature(feat) and feat not in interactions:
                    interactions.append(feat)
        else:
            # Check all HEAD_FEATURES
            for head_list in HEAD_FEATURES.values():
                for feat in head_list:
                    if is_interaction_feature(feat) and feat not in interactions:
                        interactions.append(feat)

    return interactions


def get_all_interaction_base_features(
    model_key=None
) -> Set[str]:
    """Get all base features required to compute registered interactions.

    Useful for ensuring the feature pipeline computes the necessary
    base features before interaction computation.

    Args:
        model_key: Optional ModelKey for model-specific interactions.

    Returns:
        Set of base feature names required for all interactions.
    """
    interactions = get_registered_interaction_features(model_key)

    base_features = set()
    for interaction in interactions:
        base_features.update(get_interaction_base_features(interaction))

    return base_features


def validate_interaction_dependencies(
    df: pd.DataFrame,
    model_key=None
) -> Dict[str, any]:
    """Validate that all base features for registered interactions are present.

    Args:
        df: DataFrame to check.
        model_key: Optional ModelKey for model-specific validation.

    Returns:
        Dict with validation results:
        - 'valid': List of interactions that can be computed
        - 'missing_deps': Dict of interaction -> missing base features
        - 'all_present': Bool indicating if all dependencies are met
    """
    interactions = get_registered_interaction_features(model_key)

    valid = []
    missing_deps = {}

    for interaction in interactions:
        required = get_interaction_base_features(interaction)
        missing = required - set(df.columns)

        if missing:
            missing_deps[interaction] = list(missing)
        else:
            valid.append(interaction)

    return {
        'valid': valid,
        'missing_deps': missing_deps,
        'all_present': len(missing_deps) == 0,
    }


# =============================================================================
# FEATURE PRODUCTION INTEGRATION
# =============================================================================

def add_interactions_to_features(
    df: pd.DataFrame,
    model_key=None,
    verbose: bool = True
) -> pd.DataFrame:
    """Main entry point for adding interaction features to a DataFrame.

    This function is designed to be called from:
    1. Feature pipeline (orchestrator.py) after base features are computed
    2. Training scripts before model training
    3. Inference pipelines before prediction

    It finds all registered interaction features for the given model,
    validates that base features exist, and computes the interactions.

    Args:
        df: DataFrame containing base features.
        model_key: Optional ModelKey for model-specific interactions.
                   If None, computes interactions for all models.
        verbose: Whether to log progress.

    Returns:
        DataFrame with interaction features added.
    """
    # Get registered interactions
    interactions = get_registered_interaction_features(model_key)

    if not interactions:
        if verbose:
            logger.info("No interaction features registered")
        return df

    if verbose:
        logger.info(f"Computing {len(interactions)} registered interaction features")

    # Validate dependencies
    validation = validate_interaction_dependencies(df, model_key)

    if validation['missing_deps']:
        logger.warning(
            f"Cannot compute {len(validation['missing_deps'])} interactions "
            f"due to missing base features"
        )
        for interaction, missing in list(validation['missing_deps'].items())[:3]:
            logger.debug(f"  {interaction}: missing {missing}")

    # Compute interactions
    df = compute_registered_interactions(
        df,
        validation['valid'],
        skip_missing=True,
        inplace=False
    )

    if verbose:
        computed = [f for f in validation['valid'] if f in df.columns]
        logger.info(f"Successfully computed {len(computed)} interaction features")

    return df


def ensure_interactions_for_training(
    X: pd.DataFrame,
    selected_features: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """Ensure all selected interaction features are computed before training.

    This is called by training scripts to ensure interaction features
    that are part of the model's feature set are available.

    Args:
        X: Feature DataFrame.
        selected_features: List of features the model will use.
        verbose: Whether to log progress.

    Returns:
        DataFrame with any missing interaction features computed.
    """
    # Find interactions in selected features that are missing from X
    missing_interactions = []

    for feat in selected_features:
        if feat not in X.columns and is_interaction_feature(feat):
            missing_interactions.append(feat)

    if not missing_interactions:
        if verbose:
            logger.debug("All selected interaction features already present")
        return X

    if verbose:
        logger.info(f"Computing {len(missing_interactions)} missing interaction features")

    # Compute missing interactions
    X = compute_registered_interactions(
        X,
        missing_interactions,
        skip_missing=True,
        inplace=False
    )

    # Report any that still couldn't be computed
    still_missing = [f for f in missing_interactions if f not in X.columns]
    if still_missing:
        logger.warning(
            f"{len(still_missing)} interaction features could not be computed "
            f"(missing base features): {still_missing}"
        )

    return X


# =============================================================================
# TEMPLATE-BASED INTERACTION COMPUTATION
# =============================================================================

def compute_template_interactions(
    df: pd.DataFrame,
    template_name: str,
    template: Dict[str, any],
    skip_missing: bool = True,
    inplace: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute all interaction features for a given template.

    Args:
        df: DataFrame containing base features.
        template_name: Name of the template.
        template: Template definition dict with keys:
            - type: "gate", "signed_gate", or "product"
            - base_features: list of base feature names
            - gate_features: list of gate/modifier feature names
            - invert_gate: optional bool (default False)
        skip_missing: If True, skip interactions with missing base features.
        inplace: If True, modify df in place.

    Returns:
        Tuple of (DataFrame with interactions added, list of computed feature names)
    """
    if not inplace:
        df = df.copy()

    interaction_type = template["type"]
    base_features = template["base_features"]
    gate_features = template["gate_features"]
    invert_gate = template.get("invert_gate", False)

    # Map template types to compute types
    type_map = {
        "gate": "gated",
        "signed_gate": "signed_gate",
        "product": "product",
    }
    compute_type = type_map.get(interaction_type, "product")

    computed = []
    skipped = []

    for base_feat in base_features:
        for gate_feat in gate_features:
            # Determine feature name based on type
            if interaction_type in ("gate", "signed_gate"):
                feat_name = f"ix__{template_name}__{base_feat}__gated__{gate_feat}"
            else:  # product
                feat_name = f"ix__{template_name}__{base_feat}__x__{gate_feat}"

            # Skip if already computed
            if feat_name in df.columns:
                computed.append(feat_name)
                continue

            # Check if base features exist
            if base_feat not in df.columns or gate_feat not in df.columns:
                if skip_missing:
                    skipped.append({
                        'interaction': feat_name,
                        'missing': [f for f in [base_feat, gate_feat]
                                   if f not in df.columns]
                    })
                    continue
                else:
                    raise KeyError(
                        f"Cannot compute {feat_name}: missing base features "
                        f"(need {base_feat}, {gate_feat})"
                    )

            try:
                values = compute_interaction(
                    df, base_feat, gate_feat, compute_type, invert_gate
                )
                df[feat_name] = values
                computed.append(feat_name)
            except Exception as e:
                logger.warning(f"Failed to compute {feat_name}: {e}")

    if computed:
        logger.debug(f"Computed {len(computed)} features for template {template_name}")
    if skipped:
        logger.debug(f"Skipped {len(skipped)} features (missing base features)")

    return df, computed


def compute_all_template_interactions(
    df: pd.DataFrame,
    selected_groups: Optional[Set[str]] = None,
    skip_missing: bool = True,
    inplace: bool = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Compute interactions for all eligible templates.

    Args:
        df: DataFrame containing base features.
        selected_groups: Optional set of selected group names.
            If provided, only compute interactions where both parent groups
            are in the selected set.
        skip_missing: If True, skip interactions with missing base features.
        inplace: If True, modify df in place.

    Returns:
        Tuple of (DataFrame with interactions, dict mapping template -> features)
    """
    # Import templates
    from src.feature_selection.base_features import (
        INTERACTION_TEMPLATES,
        get_eligible_templates
    )

    if not inplace:
        df = df.copy()

    # Determine eligible templates
    if selected_groups is not None:
        eligible = get_eligible_templates(selected_groups)
    else:
        eligible = list(INTERACTION_TEMPLATES.keys())

    computed_by_template = {}

    for template_name in eligible:
        template = INTERACTION_TEMPLATES[template_name]
        df, computed = compute_template_interactions(
            df, template_name, template, skip_missing=skip_missing, inplace=True
        )
        if computed:
            computed_by_template[template_name] = computed

    total_computed = sum(len(v) for v in computed_by_template.values())
    if total_computed > 0:
        logger.info(
            f"Computed {total_computed} interaction features "
            f"from {len(computed_by_template)} templates"
        )

    return df, computed_by_template


def get_template_interaction_info(feature_name: str) -> Optional[Dict[str, any]]:
    """Get template information for a template-based interaction feature.

    Args:
        feature_name: The interaction feature name.

    Returns:
        Dict with template info or None if not a template interaction.
    """
    parsed = parse_template_interaction_name(feature_name)
    if parsed is None:
        return None

    template_name, feat_a, feat_b, interaction_type = parsed

    # Try to look up the template
    try:
        from src.feature_selection.base_features import INTERACTION_TEMPLATES
        template = INTERACTION_TEMPLATES.get(template_name)
    except ImportError:
        template = None

    return {
        'template_name': template_name,
        'base_feature': feat_a,
        'gate_feature': feat_b,
        'interaction_type': interaction_type,
        'template': template,
    }
