"""Probability calibration for LightGBM classifiers.

Calibration is performed per-fold to prevent leakage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings


@dataclass
class CalibrationResult:
    """Result of probability calibration."""
    method: str
    prob_true: np.ndarray
    prob_pred: np.ndarray
    n_bins: int
    brier_before: float
    brier_after: float
    ece_before: float  # Expected Calibration Error
    ece_after: float


class CalibratedClassifier:
    """Wrapper that calibrates model probabilities.

    Calibration is fit on a held-out calibration set (or training data
    with care) and applied to transform raw probabilities.

    Supports:
        - isotonic: Isotonic regression (non-parametric)
        - platt: Platt scaling (logistic regression)
        - beta: Beta calibration (not implemented yet)
    """

    def __init__(
        self,
        method: str = "isotonic",
        n_bins: int = 10,
    ):
        """Initialize calibrator.

        Args:
            method: Calibration method ('isotonic' or 'platt').
            n_bins: Number of bins for calibration curve.
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrator_ = None
        self._fitted = False

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> "CalibratedClassifier":
        """Fit calibrator on held-out data.

        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.

        Returns:
            Fitted calibrator.
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()

        # Clip probabilities to valid range
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(
                y_min=0, y_max=1, out_of_bounds="clip"
            )
            self.calibrator_.fit(y_prob, y_true)

        elif self.method == "platt":
            # Platt scaling: fit logistic regression on log-odds
            log_odds = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
            self.calibrator_ = LogisticRegression(C=1e10, solver="lbfgs")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.calibrator_.fit(log_odds, y_true)

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self._fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities.

        Args:
            y_prob: Raw probabilities.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator must be fit first")

        y_prob = np.asarray(y_prob).ravel()
        y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

        if self.method == "isotonic":
            calibrated = self.calibrator_.predict(y_prob)

        elif self.method == "platt":
            log_odds = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
            calibrated = self.calibrator_.predict_proba(log_odds)[:, 1]

        return np.clip(calibrated, 0, 1)

    def fit_transform(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(y_true, y_prob).transform(y_prob)


def calibrate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "isotonic",
) -> Tuple[np.ndarray, CalibratedClassifier]:
    """Convenience function to calibrate probabilities.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        method: Calibration method.

    Returns:
        Tuple of (calibrated_probabilities, fitted_calibrator).
    """
    calibrator = CalibratedClassifier(method=method)
    calibrated = calibrator.fit_transform(y_true, y_prob)
    return calibrated, calibrator


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Evaluate probability calibration.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with calibration metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            mce = max(mce, abs(bin_accuracy - bin_confidence))

    # Log loss
    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    log_loss = -np.mean(
        y_true * np.log(y_prob_clipped) +
        (1 - y_true) * np.log(1 - y_prob_clipped)
    )

    return {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "log_loss": log_loss,
        "mean_predicted": y_prob.mean(),
        "mean_actual": y_true.mean(),
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    ax=None,
):
    """Plot calibration curve.

    Args:
        y_true: True labels.
        y_prob_raw: Raw predicted probabilities.
        y_prob_calibrated: Calibrated probabilities (optional).
        n_bins: Number of bins.
        ax: Matplotlib axis (optional).

    Returns:
        Matplotlib axis.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Raw probabilities
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob_raw, n_bins=n_bins, strategy="uniform"
    )
    ax.plot(prob_pred, prob_true, "s-", label="Raw", color="blue")

    # Calibrated probabilities
    if y_prob_calibrated is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true, y_prob_calibrated, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(prob_pred_cal, prob_true_cal, "o-", label="Calibrated", color="green")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax
