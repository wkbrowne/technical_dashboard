"""
Feature Registry for ML Pipeline Reproducibility.

This module provides a minimal, audit-friendly feature registry that:
1. Records which features were selected per group (K-of-N selections)
2. Resolves to a concrete, ordered feature list for downstream stages
3. Persists to artifacts/<model_name>/features.json with stable signature hash
4. Provides a simple API for feature selection, hyperopt, and training scripts

Schema Version 1 Format:
{
  "schema_version": 1,
  "model": "long_normal",
  "created_at": "<iso timestamp>",
  "data_signature": "<optional: dataset window/version>",
  "groups": { ... },  # Optional: full group definitions
  "selection": {
    "baseline_k_of_n": {
      "group_name": {"k": 4, "chosen": ["f1", "f2", ...]},
      ...
    },
    "include_groups": ["group_full_inclusion", ...],
    "include_features": ["singleton_feature", ...],
    "exclude_features": ["excluded_feature", ...]
  },
  "resolved_features": ["f1", "f2", ...],  # Final deterministic list
  "feature_signature": "sha256:<hex>"
}

Usage:
    from src.features.registry import (
        build_registry_from_selection,
        resolve_features,
        compute_feature_signature,
        load_registry,
        save_registry,
    )

    # After feature selection
    registry = build_registry_from_selection(
        model_name="long_normal",
        groups=candidate_groups,
        selection_dict=selection_result,
    )
    save_registry(registry, "artifacts/long_normal/features.json")

    # In hyperopt/training
    registry = load_registry("artifacts/long_normal/features.json")
    features = registry["resolved_features"]
    signature = registry["feature_signature"]
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Current schema version
SCHEMA_VERSION = 1


def compute_feature_signature(resolved_features: List[str]) -> str:
    """
    Compute a stable hash signature for a feature list.

    The signature is deterministic for the same feature list (order-sensitive).

    Args:
        resolved_features: Ordered list of feature names

    Returns:
        Signature string in format "sha256:<hexdigest>"
    """
    # Join features with newlines and encode
    content = "\n".join(resolved_features).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def resolve_features(registry: Dict[str, Any]) -> List[str]:
    """
    Resolve a registry to a concrete, ordered feature list.

    Resolution rules (deterministic ordering):
    1. Start with baseline_k_of_n chosen lists (groups sorted by name)
    2. Add include_groups: expand to all features (group sort, then feature sort)
    3. Add include_features (sorted)
    4. Remove exclude_features
    5. Deduplicate while preserving first occurrence order

    For baseline_k_of_n chosen lists, the order is preserved as recorded
    (the selection algorithm's order may encode importance).

    Args:
        registry: Registry dict with "selection" key

    Returns:
        Deterministically ordered list of feature names

    Raises:
        ValueError: If chosen features don't exist in groups (when groups provided)
    """
    selection = registry.get("selection", {})
    groups = registry.get("groups", {})

    seen: Set[str] = set()
    result: List[str] = []

    def add_feature(feat: str) -> None:
        """Add feature if not seen."""
        if feat not in seen:
            seen.add(feat)
            result.append(feat)

    # 1. Process baseline_k_of_n (sorted by group name for determinism)
    baseline_k_of_n = selection.get("baseline_k_of_n", {})
    for group_name in sorted(baseline_k_of_n.keys()):
        group_selection = baseline_k_of_n[group_name]
        chosen = group_selection.get("chosen", [])

        # Validate chosen features exist in group (if groups provided)
        if groups and group_name in groups:
            group_features = set(groups[group_name].get("features", []))
            for feat in chosen:
                if feat not in group_features:
                    raise ValueError(
                        f"Feature '{feat}' in chosen for group '{group_name}' "
                        f"not found in group features"
                    )

        # Add in the order specified by chosen (preserves selection order)
        for feat in chosen:
            add_feature(feat)

    # 2. Process include_groups (sorted by group name)
    include_groups = selection.get("include_groups", [])
    for group_name in sorted(include_groups):
        if groups and group_name in groups:
            # Sort features within group for determinism
            group_features = sorted(groups[group_name].get("features", []))
            for feat in group_features:
                add_feature(feat)

    # 3. Add include_features (sorted for determinism)
    include_features = selection.get("include_features", [])
    for feat in sorted(include_features):
        add_feature(feat)

    # 4. Remove exclude_features
    exclude_features = set(selection.get("exclude_features", []))
    if exclude_features:
        result = [f for f in result if f not in exclude_features]

    return result


def build_registry_from_selection(
    model_name: str,
    selected_groups: Dict[str, List[str]],
    groups: Optional[Dict[str, Dict[str, Any]]] = None,
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    data_signature: Optional[str] = None,
    selection_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a feature registry from group selection results.

    This is the primary entry point after feature selection completes.

    Args:
        model_name: Model identifier (e.g., "long_normal")
        selected_groups: Dict mapping group_name -> list of chosen features
                         This is the K-of-N result from group selection
        groups: Optional full group definitions (for validation)
        include_features: Additional singleton features to include
        exclude_features: Features to exclude from final set
        data_signature: Optional dataset version/window identifier
        selection_metadata: Optional dict with selection metrics, timestamps, etc.

    Returns:
        Complete registry dict ready for saving
    """
    # Build baseline_k_of_n from selected_groups
    baseline_k_of_n = {}
    for group_name, chosen_features in selected_groups.items():
        baseline_k_of_n[group_name] = {
            "k": len(chosen_features),
            "chosen": list(chosen_features),  # Preserve order
        }

    # Build selection dict
    selection: Dict[str, Any] = {
        "baseline_k_of_n": baseline_k_of_n,
    }

    if include_features:
        selection["include_features"] = sorted(include_features)
    if exclude_features:
        selection["exclude_features"] = sorted(exclude_features)

    # Build initial registry
    registry: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
    }

    if data_signature:
        registry["data_signature"] = data_signature

    # Include groups if provided (optional, can be large)
    if groups:
        registry["groups"] = {
            name: {"features": list(features)}
            if isinstance(features, list) else dict(features)
            for name, features in groups.items()
        }

    # Add selection metadata if provided
    if selection_metadata:
        registry["selection_metadata"] = selection_metadata

    # Resolve features and compute signature
    resolved = resolve_features(registry)
    registry["resolved_features"] = resolved
    registry["feature_signature"] = compute_feature_signature(resolved)

    return registry


def load_registry(path: str | Path) -> Dict[str, Any]:
    """
    Load a feature registry from disk.

    Args:
        path: Path to features.json file

    Returns:
        Registry dict

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema version is unsupported
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")

    with open(path, "r") as f:
        registry = json.load(f)

    # Validate schema version
    schema_version = registry.get("schema_version", 0)
    if schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"Registry schema version {schema_version} is newer than "
            f"supported version {SCHEMA_VERSION}. Please update your code."
        )

    return registry


def save_registry(registry: Dict[str, Any], path: str | Path) -> None:
    """
    Save a feature registry to disk.

    Creates parent directories if needed.

    Args:
        registry: Registry dict to save
        path: Output path for features.json
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def get_registry_summary(registry: Dict[str, Any]) -> str:
    """
    Get a one-line summary of a registry.

    Args:
        registry: Registry dict

    Returns:
        Summary string like "long_normal: 23 features, sha256:abc123..."
    """
    model = registry.get("model", "unknown")
    n_features = len(registry.get("resolved_features", []))
    signature = registry.get("feature_signature", "")
    # Truncate signature for display
    sig_short = signature[:20] + "..." if len(signature) > 20 else signature

    return f"{model}: {n_features} features, {sig_short}"


def validate_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a registry for consistency.

    Checks:
    - Required fields present
    - resolved_features matches re-resolution
    - feature_signature matches re-computation

    Args:
        registry: Registry dict to validate

    Returns:
        Dict with "valid" bool and "issues" list
    """
    issues = []

    # Check required fields
    required = ["schema_version", "model", "selection", "resolved_features", "feature_signature"]
    for field in required:
        if field not in registry:
            issues.append(f"Missing required field: {field}")

    if issues:
        return {"valid": False, "issues": issues}

    # Check resolution consistency
    try:
        re_resolved = resolve_features(registry)
        if re_resolved != registry["resolved_features"]:
            issues.append(
                f"resolved_features mismatch: "
                f"stored has {len(registry['resolved_features'])} features, "
                f"re-resolution has {len(re_resolved)}"
            )
    except Exception as e:
        issues.append(f"Resolution error: {e}")

    # Check signature consistency
    stored_sig = registry.get("feature_signature", "")
    computed_sig = compute_feature_signature(registry.get("resolved_features", []))
    if stored_sig != computed_sig:
        issues.append(
            f"feature_signature mismatch: stored={stored_sig[:30]}..., "
            f"computed={computed_sig[:30]}..."
        )

    return {"valid": len(issues) == 0, "issues": issues}


def get_registry_path(model_name: str, base_dir: str = "artifacts") -> Path:
    """
    Get the standard path for a model's feature registry.

    Args:
        model_name: Model identifier (e.g., "long_normal")
        base_dir: Base artifacts directory

    Returns:
        Path like artifacts/long_normal/features.json
    """
    return Path(base_dir) / model_name / "features.json"


def registry_exists(model_name: str, base_dir: str = "artifacts") -> bool:
    """
    Check if a registry exists for a model.

    Args:
        model_name: Model identifier
        base_dir: Base artifacts directory

    Returns:
        True if registry file exists
    """
    return get_registry_path(model_name, base_dir).exists()


def compare_registries(
    registry_a: Dict[str, Any],
    registry_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two registries and report differences.

    Args:
        registry_a: First registry
        registry_b: Second registry

    Returns:
        Dict with comparison results
    """
    features_a = set(registry_a.get("resolved_features", []))
    features_b = set(registry_b.get("resolved_features", []))

    return {
        "signature_match": (
            registry_a.get("feature_signature") == registry_b.get("feature_signature")
        ),
        "feature_count_a": len(features_a),
        "feature_count_b": len(features_b),
        "only_in_a": sorted(features_a - features_b),
        "only_in_b": sorted(features_b - features_a),
        "common": sorted(features_a & features_b),
        "jaccard_similarity": (
            len(features_a & features_b) / len(features_a | features_b)
            if features_a | features_b else 1.0
        ),
    }
