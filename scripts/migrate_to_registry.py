#!/usr/bin/env python
"""
Migrate existing group_selection results to feature registry format.

This script converts legacy group_selection_*.json files to the new
feature registry format at artifacts/<model_name>/features.json.

Usage:
    python scripts/migrate_to_registry.py
    python scripts/migrate_to_registry.py --model long_normal
    python scripts/migrate_to_registry.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.registry import (
    build_registry_from_selection,
    save_registry,
    get_registry_summary,
    get_registry_path,
)


def migrate_model(model_name: str, dry_run: bool = False) -> bool:
    """
    Migrate a single model's group selection to registry format.

    Args:
        model_name: Model identifier (e.g., 'long_normal')
        dry_run: If True, only print what would be done

    Returns:
        True if migration successful, False otherwise
    """
    legacy_path = Path(f"artifacts/group_selection/group_selection_{model_name}.json")
    registry_path = get_registry_path(model_name)

    if not legacy_path.exists():
        print(f"  SKIP: {legacy_path} not found")
        return False

    print(f"  Loading: {legacy_path}")
    with open(legacy_path) as f:
        legacy = json.load(f)

    # Build selection metadata from legacy format
    selection_metadata = {
        "final_metric": legacy.get("final_metric"),
        "baseline_metric": legacy.get("baseline_metric"),
        "improvement": legacy.get("improvement"),
        "n_groups": legacy.get("n_groups"),
        "total_groups_evaluated": legacy.get("total_groups_evaluated"),
        "total_time_seconds": legacy.get("total_time_seconds"),
    }

    # Build registry
    registry = build_registry_from_selection(
        model_name=model_name,
        selected_groups=legacy["selected_groups"],
        selection_metadata=selection_metadata,
    )

    summary = get_registry_summary(registry)
    print(f"  Built: {summary}")

    if dry_run:
        print(f"  DRY RUN: Would save to {registry_path}")
    else:
        save_registry(registry, registry_path)
        print(f"  Saved: {registry_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate group selection to registry format")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to migrate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without saving",
    )
    args = parser.parse_args()

    models = ["long_normal", "long_parabolic", "short_normal", "short_parabolic"]

    if args.model:
        if args.model not in models:
            print(f"ERROR: Unknown model '{args.model}'")
            print(f"Valid models: {', '.join(models)}")
            return
        models = [args.model]

    print("Migrating group selection results to feature registry format")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN MODE - no files will be written\n")

    success_count = 0
    for model in models:
        print(f"\n{model.upper()}")
        if migrate_model(model, dry_run=args.dry_run):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Migrated {success_count}/{len(models)} models")

    if not args.dry_run and success_count > 0:
        print("\nNext steps:")
        print("  - Run training: python run_training.py --all-models")
        print("  - Verify features: python -c \"from src.features.registry import load_registry, get_registry_path; print(load_registry(get_registry_path('long_normal'))['resolved_features'][:5])\"")


if __name__ == "__main__":
    main()
