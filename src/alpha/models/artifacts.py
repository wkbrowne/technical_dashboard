"""Model artifact management.

Save and load trained models, calibrators, and associated metadata.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pickle
import numpy as np
import pandas as pd


@dataclass
class ModelArtifact:
    """Container for model artifacts.

    Attributes:
        model: Trained model object.
        calibrator: Fitted calibrator (optional).
        feature_names: List of feature names.
        metadata: Dict of additional metadata.
        created_at: Creation timestamp.
        version: Artifact version.
    """
    model: Any
    calibrator: Any = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"


def save_model_artifact(
    artifact: ModelArtifact,
    path: Union[str, Path],
    save_metadata: bool = True,
) -> Path:
    """Save model artifact to disk.

    Args:
        artifact: ModelArtifact to save.
        path: Base path (without extension).
        save_metadata: Whether to save metadata JSON.

    Returns:
        Path to saved artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = path.with_suffix(".pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": artifact.model,
            "calibrator": artifact.calibrator,
            "feature_names": artifact.feature_names,
        }, f)

    # Save metadata
    if save_metadata:
        meta_path = path.with_suffix(".json")
        metadata = {
            "feature_names": artifact.feature_names,
            "metadata": artifact.metadata,
            "created_at": artifact.created_at,
            "version": artifact.version,
        }

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        metadata = json.loads(
            json.dumps(metadata, default=convert_numpy)
        )

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return model_path


def load_model_artifact(
    path: Union[str, Path],
) -> ModelArtifact:
    """Load model artifact from disk.

    Args:
        path: Path to artifact (with or without extension).

    Returns:
        Loaded ModelArtifact.
    """
    path = Path(path)

    # Handle path with or without extension
    if path.suffix == ".pkl":
        model_path = path
        meta_path = path.with_suffix(".json")
    elif path.suffix == ".json":
        model_path = path.with_suffix(".pkl")
        meta_path = path
    else:
        model_path = path.with_suffix(".pkl")
        meta_path = path.with_suffix(".json")

    # Load model
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Load metadata
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return ModelArtifact(
        model=data["model"],
        calibrator=data.get("calibrator"),
        feature_names=data.get("feature_names", metadata.get("feature_names", [])),
        metadata=metadata.get("metadata", {}),
        created_at=metadata.get("created_at", ""),
        version=metadata.get("version", "1.0.0"),
    )


def save_cv_results(
    training_results: list,
    output_dir: Union[str, Path],
    prefix: str = "cv",
) -> Dict[str, Path]:
    """Save cross-validation results.

    Args:
        training_results: List of TrainingResult objects.
        output_dir: Output directory.
        prefix: Filename prefix.

    Returns:
        Dict of artifact type -> path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # Save each fold
    for result in training_results:
        fold_path = output_dir / f"{prefix}_fold{result.fold_idx}"

        artifact = ModelArtifact(
            model=result.model,
            calibrator=result.calibrator,
            feature_names=list(result.feature_importance.keys()),
            metadata={
                "fold_idx": result.fold_idx,
                "train_metrics": result.train_metrics,
                "val_metrics": result.val_metrics,
            },
        )
        saved_paths[f"fold_{result.fold_idx}"] = save_model_artifact(artifact, fold_path)

    # Save summary
    summary = {
        "n_folds": len(training_results),
        "metrics": {
            "val_auc_mean": np.mean([r.val_metrics["auc"] for r in training_results]),
            "val_auc_std": np.std([r.val_metrics["auc"] for r in training_results]),
            "val_aupr_mean": np.mean([r.val_metrics["aupr"] for r in training_results]),
        },
        "fold_metrics": [r.val_metrics for r in training_results],
    }

    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    saved_paths["summary"] = summary_path

    # Save feature importance
    importance_dfs = []
    for result in training_results:
        imp_df = pd.DataFrame({
            "feature": list(result.feature_importance.keys()),
            "importance": list(result.feature_importance.values()),
            "fold": result.fold_idx,
        })
        importance_dfs.append(imp_df)

    if importance_dfs:
        importance_df = pd.concat(importance_dfs)
        importance_agg = importance_df.groupby("feature")["importance"].agg(["mean", "std"])
        importance_agg = importance_agg.sort_values("mean", ascending=False)

        importance_path = output_dir / f"{prefix}_feature_importance.csv"
        importance_agg.to_csv(importance_path)
        saved_paths["feature_importance"] = importance_path

    return saved_paths


def save_sizing_params(
    params: Dict[str, Any],
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save optimized sizing parameters.

    Args:
        params: Sizing parameters dict.
        path: Output path.
        metadata: Optional metadata.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "params": params,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
    }

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    data = json.loads(json.dumps(data, default=convert_numpy))

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_sizing_params(path: Union[str, Path]) -> Dict[str, Any]:
    """Load sizing parameters.

    Args:
        path: Path to sizing params JSON.

    Returns:
        Sizing parameters dict.
    """
    with open(path) as f:
        data = json.load(f)
    return data["params"]
