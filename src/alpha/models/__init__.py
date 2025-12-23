"""Model training, calibration, and artifact management."""

from .train_lgbm import (
    train_lgbm_fold,
    train_with_cv,
    TrainingResult,
)
from .calibration import (
    CalibratedClassifier,
    calibrate_probabilities,
    evaluate_calibration,
)
from .artifacts import (
    save_model_artifact,
    load_model_artifact,
    ModelArtifact,
)

__all__ = [
    "train_lgbm_fold",
    "train_with_cv",
    "TrainingResult",
    "CalibratedClassifier",
    "calibrate_probabilities",
    "evaluate_calibration",
    "save_model_artifact",
    "load_model_artifact",
    "ModelArtifact",
]
