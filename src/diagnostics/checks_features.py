"""
Feature sanity and story consistency checks.

Checks:
- Feature family coverage
- Feature importance analysis
- Model-story consistency (e.g., parabolic models should emphasize volatility)
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
import json
import numpy as np

from .core import (
    DiagnosticFlag, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
    FEATURE_FAMILIES, EXPECTED_STORY_EMPHASIS,
    classify_feature_family,
)


def analyze_feature_families(
    feature_names: List[str],
) -> Dict[str, List[str]]:
    """
    Group features by their family.

    Args:
        feature_names: List of feature names

    Returns:
        Dictionary mapping family to list of features
    """
    families = {family: [] for family in FEATURE_FAMILIES}
    families['unclassified'] = []

    for feat in feature_names:
        family = classify_feature_family(feat)
        if family:
            families[family].append(feat)
        else:
            families['unclassified'].append(feat)

    return families


def check_family_coverage(
    feature_names: List[str],
    model_key: str,
) -> List[DiagnosticFlag]:
    """
    Check that important feature families are represented.

    Args:
        feature_names: List of feature names
        model_key: Model key string

    Returns:
        List of diagnostic flags
    """
    flags = []

    # Analyze family coverage
    families = analyze_feature_families(feature_names)

    # Check for missing critical families
    critical_families = ['alpha_relative_strength', 'volatility_regime', 'trend']

    for family in critical_families:
        if len(families.get(family, [])) == 0:
            flags.append(DiagnosticFlag(
                severity=Severity.WARN,
                check_name='missing_feature_family',
                symptom=f"No features from '{family}' family",
                why_it_matters=f"'{family}' features are typically important for trading models",
                suggested_fix=f"Consider adding features from the {family} category",
                evidence={
                    'model_key': model_key,
                    'missing_family': family,
                    'available_families': {k: len(v) for k, v in families.items() if v},
                },
            ))

    # Check for very unbalanced coverage
    family_counts = {k: len(v) for k, v in families.items() if k != 'unclassified' and v}

    if family_counts:
        max_count = max(family_counts.values())
        min_count = min(family_counts.values())

        if max_count > 10 * min_count and min_count > 0:
            max_family = max(family_counts, key=family_counts.get)
            min_family = min(family_counts, key=family_counts.get)

            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='unbalanced_family_coverage',
                symptom=f"'{max_family}' has {max_count} features vs '{min_family}' with {min_count}",
                why_it_matters="Highly unbalanced coverage may bias model toward certain signal types",
                suggested_fix="Consider if feature distribution matches intended model focus",
                evidence={
                    'family_counts': family_counts,
                },
            ))

    return flags


def check_story_consistency(
    feature_names: List[str],
    feature_importance: Optional[Dict[str, float]],
    model_key: str,
) -> List[DiagnosticFlag]:
    """
    Check if important features match expected "story" for this model type.

    For example:
    - Parabolic models should emphasize volatility/drawdown features
    - Long models should emphasize alpha/relative strength
    - Short models should emphasize breadth/fragility features

    Args:
        feature_names: List of feature names
        feature_importance: Feature importance dictionary (optional)
        model_key: Model key string

    Returns:
        List of diagnostic flags
    """
    flags = []

    expected_families = EXPECTED_STORY_EMPHASIS.get(model_key, [])
    if not expected_families:
        return flags

    # Analyze family coverage
    families = analyze_feature_families(feature_names)

    # If we have importance, check top 10 features
    if feature_importance and len(feature_importance) > 0:
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])
        top_10 = sorted_features[:10]

        # Classify top 10 features
        top_10_families = {}
        for feat, imp in top_10:
            family = classify_feature_family(feat)
            if family:
                if family not in top_10_families:
                    top_10_families[family] = []
                top_10_families[family].append((feat, imp))

        # Check if expected families are represented in top 10
        missing_expected = []
        for expected_family in expected_families[:2]:  # Check top 2 expected
            if expected_family not in top_10_families:
                missing_expected.append(expected_family)

        if missing_expected:
            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='story_mismatch',
                symptom=f"Expected families {missing_expected} not in top 10 by importance",
                why_it_matters=f"For {model_key}, expected emphasis on {expected_families[:2]}",
                suggested_fix="Review if model is learning expected patterns; may be fine if other signals work",
                evidence={
                    'model_key': model_key,
                    'expected_families': expected_families,
                    'top_10_families': {k: len(v) for k, v in top_10_families.items()},
                    'top_10_features': [(f, round(i, 4)) for f, i in top_10],
                },
            ))

    return flags


def check_importance_concentration(
    feature_importance: Optional[Dict[str, float]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for overly concentrated feature importance.

    Args:
        feature_importance: Feature importance dictionary
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if feature_importance is None or len(feature_importance) == 0:
        return flags

    # Normalize importance
    total_importance = sum(feature_importance.values())
    if total_importance == 0:
        return flags

    norm_importance = {k: v / total_importance for k, v in feature_importance.items()}

    # Sort and get top 5
    sorted_importance = sorted(norm_importance.items(), key=lambda x: -x[1])
    top_5 = sorted_importance[:5]
    top_5_share = sum(imp for _, imp in top_5)

    if top_5_share > thresholds.top_5_importance_concentration_warn:
        flags.append(DiagnosticFlag(
            severity=Severity.INFO,
            check_name='importance_concentration',
            symptom=f"Top 5 features account for {top_5_share*100:.1f}% of total importance",
            why_it_matters="High concentration means model relies heavily on few features",
            suggested_fix="Consider if reliance on few features is intentional; may increase fragility",
            evidence={
                'top_5_share': round(top_5_share * 100, 1),
                'top_5_features': [(f, round(i * 100, 2)) for f, i in top_5],
                'threshold': thresholds.top_5_importance_concentration_warn * 100,
            },
        ))

    return flags


def load_feature_importance(
    model_key: str,
    models_dir: Path = Path('artifacts/models'),
) -> Optional[Dict[str, float]]:
    """
    Load feature importance from trained model.

    Args:
        model_key: Model key string
        models_dir: Models directory

    Returns:
        Feature importance dictionary or None
    """
    importance_file = models_dir / model_key / 'feature_importance.csv'

    if not importance_file.exists():
        return None

    try:
        import pandas as pd
        df = pd.read_csv(importance_file)

        # Expect columns like 'feature', 'importance' or 'feature', 'gain'
        if 'feature' in df.columns:
            if 'importance' in df.columns:
                return dict(zip(df['feature'], df['importance']))
            elif 'gain' in df.columns:
                return dict(zip(df['feature'], df['gain']))
    except Exception:
        pass

    return None


def compute_feature_summary(
    feature_names: List[str],
    feature_importance: Optional[Dict[str, float]],
    model_key: str,
) -> Dict[str, Any]:
    """
    Compute summary of feature usage and importance.

    Args:
        feature_names: List of feature names
        feature_importance: Feature importance dictionary
        model_key: Model key string

    Returns:
        Summary dictionary
    """
    families = analyze_feature_families(feature_names)

    summary = {
        'n_features': len(feature_names),
        'family_breakdown': {k: len(v) for k, v in families.items() if v},
        'unclassified_features': families.get('unclassified', []),
        'expected_emphasis': EXPECTED_STORY_EMPHASIS.get(model_key, []),
    }

    if feature_importance:
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            norm_importance = {k: v / total_importance for k, v in feature_importance.items()}
            sorted_importance = sorted(norm_importance.items(), key=lambda x: -x[1])

            summary['importance_available'] = True
            summary['top_10_features'] = [
                {'feature': f, 'importance_pct': round(i * 100, 2)}
                for f, i in sorted_importance[:10]
            ]

            # Importance by family
            family_importance = {}
            for feat, imp in norm_importance.items():
                family = classify_feature_family(feat)
                if family:
                    family_importance[family] = family_importance.get(family, 0) + imp

            summary['family_importance'] = {
                k: round(v * 100, 2)
                for k, v in sorted(family_importance.items(), key=lambda x: -x[1])
            }
    else:
        summary['importance_available'] = False

    return summary


def run_feature_checks(
    feature_names: List[str],
    model_key: str,
    models_dir: Path = Path('artifacts/models'),
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run all feature sanity checks.

    Args:
        feature_names: List of feature names used by the model
        model_key: Model key string
        models_dir: Models directory for loading importance
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    # Load feature importance if available
    feature_importance = load_feature_importance(model_key, models_dir)

    # Family coverage
    flags.extend(check_family_coverage(feature_names, model_key))

    # Story consistency
    flags.extend(check_story_consistency(feature_names, feature_importance, model_key))

    # Importance concentration
    flags.extend(check_importance_concentration(feature_importance, thresholds))

    # Summary
    summary = compute_feature_summary(feature_names, feature_importance, model_key)

    return flags, summary
