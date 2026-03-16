"""XGBoost classifier with Platt scaling for calibrated probability output."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MispricingModel:
    """
    Calibrated earnings surprise classifier.

    Trains XGBoost on transcript + macro features, applies Platt scaling
    to produce well-calibrated probabilities, and compares against
    market contract prices to flag mispricings.
    """

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        self.base_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        self.calibrated_model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model with Platt scaling calibration.

        Args:
            X: Feature matrix (transcript + macro features)
            y: Binary target (1 = earnings beat, 0 = miss/inline)
        """
        # Use time-series aware cross-validation
        cv = TimeSeriesSplit(n_splits=5)

        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method="sigmoid",  # Platt scaling
            cv=cv,
        )
        self.calibrated_model.fit(X, y)

        logger.info(
            f"Model trained on {len(X)} samples with "
            f"{X.shape[1]} features"
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated probability of earnings beat.

        Args:
            X: Feature matrix

        Returns:
            Array of calibrated probabilities (P(beat))
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.calibrated_model.predict_proba(X)[:, 1]

    def flag_mispricings(
        self, X: pd.DataFrame, market_probs: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare model probabilities to market prices and flag divergences.

        Args:
            X: Feature matrix for scoring
            market_probs: Array of market contract prices (0-1)

        Returns:
            DataFrame with model_prob, market_prob, divergence, signal
        """
        model_probs = self.predict_proba(X)
        divergence = model_probs - market_probs

        signals = pd.DataFrame(
            {
                "model_prob": model_probs,
                "market_prob": market_probs,
                "divergence": divergence,
                "abs_divergence": np.abs(divergence),
                "signal": np.where(
                    np.abs(divergence) > self.threshold,
                    np.where(divergence > 0, "underpriced", "overpriced"),
                    "no_signal",
                ),
            }
        )

        n_signals = (signals["signal"] != "no_signal").sum()
        logger.info(
            f"Flagged {n_signals} mispricings out of "
            f"{len(signals)} contracts (threshold={self.threshold})"
        )
        return signals

    def save(self, path: str = "models/mispricing_model.pkl"):
        """Serialize trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.calibrated_model, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = "models/mispricing_model.pkl"):
        """Load trained model from disk."""
        with open(path, "rb") as f:
            self.calibrated_model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
