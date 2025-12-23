"""Cross-validation with purging and embargo for overlapping labels."""

from .purged_walkforward import (
    PurgedWalkForwardCV,
    WeeklySignalCV,
    validate_no_leakage,
    get_fold_info,
)
from .embargo import (
    compute_embargo_mask,
    compute_purge_mask,
    check_label_overlap,
)

__all__ = [
    "PurgedWalkForwardCV",
    "WeeklySignalCV",
    "validate_no_leakage",
    "get_fold_info",
    "compute_embargo_mask",
    "compute_purge_mask",
    "check_label_overlap",
]
