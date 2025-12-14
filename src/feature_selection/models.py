"""Model wrappers for gradient boosting models.

This module provides a unified interface for LightGBM and XGBoost models,
handling training, prediction, and feature importance extraction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import gc

from .config import ModelConfig, ModelType, TaskType


class GBMWrapper:
    """Unified wrapper for gradient boosting models.

    Provides a consistent interface for LightGBM and XGBoost, handling
    training with early stopping, prediction, and importance extraction.

    Attributes:
        config: ModelConfig with model settings.
        model: The trained model (None before training).
        feature_names: List of feature names used for training.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the wrapper.

        Args:
            config: ModelConfig specifying model type and parameters.
        """
        self.config = config
        self.model = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'GBMWrapper':
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for early stopping).
            y_val: Validation labels.
            feature_names: List of feature names.

        Returns:
            self for chaining.
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]

        # Convert to numpy if needed
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

        if self.config.model_type == ModelType.LIGHTGBM:
            self._train_lightgbm(X_train_arr, y_train_arr, X_val, y_val)
        else:
            self._train_xgboost(X_train_arr, y_train_arr, X_val, y_val)

        return self

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]],
        y_val: Optional[Union[pd.Series, np.ndarray]]
    ):
        """Train LightGBM model."""
        import lightgbm as lgb

        params = self.config.get_model_params()

        # Create datasets
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=self.feature_names,
            free_raw_data=True
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            val_data = lgb.Dataset(
                X_val_arr, label=y_val_arr,
                reference=train_data,
                free_raw_data=True
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Training callbacks
        callbacks = [lgb.log_evaluation(period=0)]  # Suppress output

        if self.config.early_stopping_rounds and X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            )

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]],
        y_val: Optional[Union[pd.Series, np.ndarray]]
    ):
        """Train XGBoost model."""
        import xgboost as xgb

        params = self.config.get_model_params()

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            dval = xgb.DMatrix(X_val_arr, label=y_val_arr, feature_names=self.feature_names)
            evals.append((dval, 'valid'))

        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds if X_val is not None else None,
            verbose_eval=False
        )

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions (probabilities for classification, values for regression).
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if self.config.model_type == ModelType.LIGHTGBM:
            return self.model.predict(X_arr)
        else:
            import xgboost as xgb
            dtest = xgb.DMatrix(X_arr, feature_names=self.feature_names)
            return self.model.predict(dtest)

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'split', or 'weight').

        Returns:
            Dict mapping feature name to importance score.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.config.model_type == ModelType.LIGHTGBM:
            importance = self.model.feature_importance(importance_type=importance_type)
            return dict(zip(self.feature_names, importance))
        else:
            # XGBoost importance types: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
            xgb_type = 'gain' if importance_type == 'gain' else 'weight'
            importance = self.model.get_score(importance_type=xgb_type)
            # Fill missing features with 0
            return {f: importance.get(f, 0) for f in self.feature_names}

    def get_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False
    ) -> np.ndarray:
        """Compute SHAP values for the given data.

        Args:
            X: Feature matrix.
            check_additivity: Whether to verify SHAP additivity (slow).

        Returns:
            Array of SHAP values, shape (n_samples, n_features).
        """
        import shap

        if self.model is None:
            raise ValueError("Model not trained yet")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if self.config.model_type == ModelType.LIGHTGBM:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_arr, check_additivity=check_additivity)
            # For binary classification, shap_values is a list [neg_class, pos_class]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
        else:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_arr, check_additivity=check_additivity)

        return shap_values

    def get_shap_interaction_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: int = 1000
    ) -> np.ndarray:
        """Compute SHAP interaction values.

        Args:
            X: Feature matrix.
            max_samples: Maximum samples to use (interaction values are expensive).

        Returns:
            Array of interaction values, shape (n_samples, n_features, n_features).
        """
        import shap

        if self.model is None:
            raise ValueError("Model not trained yet")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # Subsample if needed
        if len(X_arr) > max_samples:
            idx = np.random.choice(len(X_arr), max_samples, replace=False)
            X_arr = X_arr[idx]

        explainer = shap.TreeExplainer(self.model)
        interaction_values = explainer.shap_interaction_values(X_arr)

        # For binary classification
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]

        return interaction_values

    def cleanup(self):
        """Free model memory."""
        self.model = None
        gc.collect()


def create_model(config: ModelConfig) -> GBMWrapper:
    """Factory function to create a model wrapper.

    Args:
        config: Model configuration.

    Returns:
        GBMWrapper instance.
    """
    return GBMWrapper(config)


def train_and_predict(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    config: ModelConfig,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, GBMWrapper]:
    """Train model and generate predictions in one call.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        config: Model configuration.
        feature_names: Optional feature names.

    Returns:
        Tuple of (predictions, trained_model).
    """
    model = GBMWrapper(config)

    # Use a portion of training data for early stopping if enabled
    if config.early_stopping_rounds:
        n_train = len(X_train)
        split_idx = int(n_train * 0.85)

        if isinstance(X_train, pd.DataFrame):
            X_tr = X_train.iloc[:split_idx]
            X_val = X_train.iloc[split_idx:]
        else:
            X_tr = X_train[:split_idx]
            X_val = X_train[split_idx:]

        if isinstance(y_train, pd.Series):
            y_tr = y_train.iloc[:split_idx]
            y_val = y_train.iloc[split_idx:]
        else:
            y_tr = y_train[:split_idx]
            y_val = y_train[split_idx:]

        model.train(X_tr, y_tr, X_val, y_val, feature_names)
    else:
        model.train(X_train, y_train, feature_names=feature_names)

    predictions = model.predict(X_test)

    return predictions, model
