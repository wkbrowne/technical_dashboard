#!/usr/bin/env python
"""
Recompute interaction features after feature selection.

This script updates the features_complete.parquet file with any newly registered
interaction features from HEAD_FEATURES/CORE_FEATURES. It's designed to be run
after feature selection discovers new valuable interaction features.

Usage:
    # Compute all registered interactions that are missing
    python scripts/recompute_interactions.py

    # Compute specific interactions
    python scripts/recompute_interactions.py --interactions "feat_a_x_feat_b" "feat_c_gated_feat_d"

    # Preview what would be computed (dry run)
    python scripts/recompute_interactions.py --dry-run

    # Specify custom input/output paths
    python scripts/recompute_interactions.py --input artifacts/features_complete.parquet

Examples:
    After running feature selection and updating base_features.py with new
    interaction features:

    1. Run feature selection:
       python run_feature_selection.py --interactions-only

    2. Update base_features.py with selected interactions
       (manually or via the selection output)

    3. Recompute interactions in feature file:
       python scripts/recompute_interactions.py

    4. Train models with updated features:
       python run_training.py --all-models
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.interaction_production import (
    compute_registered_interactions,
    get_registered_interaction_features,
    parse_interaction_name,
    validate_interaction_dependencies,
    is_interaction_feature,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recompute_interactions(
    input_path: Path,
    output_path: Optional[Path] = None,
    interactions: Optional[List[str]] = None,
    dry_run: bool = False,
    backup: bool = True,
) -> dict:
    """
    Recompute interaction features in the feature file.

    Args:
        input_path: Path to features_complete.parquet
        output_path: Path to save updated features (default: overwrite input)
        interactions: List of specific interactions to compute (default: all registered)
        dry_run: If True, only report what would be computed
        backup: If True, create a backup before overwriting

    Returns:
        Dict with summary statistics
    """
    if output_path is None:
        output_path = input_path

    print("=" * 70)
    print("INTERACTION FEATURE RECOMPUTATION")
    print("=" * 70)

    # Load feature data
    print(f"\nLoading features from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Get interactions to compute
    if interactions:
        # Use specified interactions
        to_compute = interactions
        print(f"\nUsing {len(to_compute)} specified interactions")
    else:
        # Get all registered interactions
        to_compute = get_registered_interaction_features()
        print(f"\nFound {len(to_compute)} registered interaction features")

    if not to_compute:
        print("\nNo interaction features to compute")
        return {"computed": 0, "skipped": 0, "failed": 0}

    # Check which are missing
    missing = [f for f in to_compute if f not in df.columns]
    present = [f for f in to_compute if f in df.columns]

    print(f"\n  Already present: {len(present)}")
    print(f"  Missing (to compute): {len(missing)}")

    if not missing:
        print("\nAll interaction features already present")
        return {"computed": 0, "skipped": len(present), "failed": 0}

    # Validate dependencies
    print("\nValidating base feature dependencies...")
    validation = validate_interaction_dependencies(df)

    can_compute = [f for f in missing if f in validation['valid']]
    cannot_compute = [f for f in missing if f in validation['missing_deps']]

    print(f"  Can compute: {len(can_compute)}")
    print(f"  Cannot compute (missing deps): {len(cannot_compute)}")

    if cannot_compute:
        print("\n  Missing dependencies for:")
        for feat in cannot_compute[:5]:
            deps = validation['missing_deps'].get(feat, [])
            print(f"    {feat}: missing {deps}")
        if len(cannot_compute) > 5:
            print(f"    ... and {len(cannot_compute) - 5} more")

    if not can_compute:
        print("\nNo interaction features can be computed (missing base features)")
        return {"computed": 0, "skipped": len(present), "failed": len(cannot_compute)}

    if dry_run:
        print(f"\n[DRY RUN] Would compute {len(can_compute)} interaction features:")
        for feat in can_compute:
            parsed = parse_interaction_name(feat)
            if parsed:
                print(f"    {feat}")
                print(f"      = {parsed[0]} {parsed[2]} {parsed[1]}")
        return {"computed": 0, "skipped": len(present), "failed": 0, "would_compute": len(can_compute)}

    # Compute interactions
    print(f"\nComputing {len(can_compute)} interaction features...")
    df = compute_registered_interactions(
        df,
        can_compute,
        skip_missing=True,
        inplace=False
    )

    # Verify computation
    computed = [f for f in can_compute if f in df.columns]
    failed = [f for f in can_compute if f not in df.columns]

    print(f"\n  Successfully computed: {len(computed)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for f in failed[:3]:
            print(f"    - {f}")

    # Create backup if requested
    if backup and output_path == input_path and computed:
        backup_path = input_path.with_suffix('.parquet.bak')
        print(f"\nCreating backup: {backup_path}")
        # Read original and save as backup
        original = pd.read_parquet(input_path)
        original.to_parquet(backup_path, index=False)
        del original
        gc.collect()

    # Save updated features
    if computed:
        print(f"\nSaving updated features to: {output_path}")
        print(f"  New shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

        df.to_parquet(output_path, index=False)
        print(f"  Saved successfully")

        # Also update features_filtered.parquet if it exists
        filtered_path = output_path.parent / 'features_filtered.parquet'
        if filtered_path.exists():
            print(f"\nUpdating filtered features: {filtered_path}")
            from src.feature_selection.base_features import filter_output_columns
            df_filtered = filter_output_columns(df, keep_all=False, exclude_retired=False)
            df_filtered.to_parquet(filtered_path, index=False)
            print(f"  Filtered shape: {df_filtered.shape[0]:,} rows x {df_filtered.shape[1]} columns")
            del df_filtered

    print("\n" + "=" * 70)
    print("RECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Already present: {len(present)}")
    print(f"  Newly computed: {len(computed)}")
    print(f"  Could not compute: {len(cannot_compute) + len(failed)}")

    return {
        "computed": len(computed),
        "skipped": len(present),
        "failed": len(cannot_compute) + len(failed),
        "computed_features": computed,
    }


def list_registered_interactions() -> None:
    """List all registered interaction features and their components."""
    interactions = get_registered_interaction_features()

    print("=" * 70)
    print("REGISTERED INTERACTION FEATURES")
    print("=" * 70)

    if not interactions:
        print("\nNo interaction features registered in HEAD_FEATURES/CORE_FEATURES")
        return

    print(f"\nFound {len(interactions)} interaction features:\n")

    for feat in sorted(interactions):
        parsed = parse_interaction_name(feat)
        if parsed:
            feat_a, feat_b, itype = parsed
            print(f"  {feat}")
            print(f"    Type: {itype}")
            print(f"    Components: {feat_a}, {feat_b}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Recompute interaction features in feature file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='artifacts/features_complete.parquet',
        help='Input features file (default: artifacts/features_complete.parquet)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output features file (default: overwrite input)'
    )

    parser.add_argument(
        '--interactions',
        nargs='+',
        default=None,
        help='Specific interaction features to compute (default: all registered)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be computed without actually computing'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup before overwriting'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all registered interaction features and exit'
    )

    args = parser.parse_args()

    if args.list:
        list_registered_interactions()
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    result = recompute_interactions(
        input_path=input_path,
        output_path=output_path,
        interactions=args.interactions,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )

    if result['failed'] > 0 and result['computed'] == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
