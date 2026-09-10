"""Isolation Forest anomaly scoring, refitted from the cleaned portfolio.

Notebook 08 fitted an Isolation Forest and a Local Outlier Factor model but
persisted only the resulting watchlist CSV, so there is no estimator to load.
This module rebuilds the Isolation Forest half of that ensemble using the
notebook's exact recipe:

1. keep loans with a positive original principal,
2. ``log1p`` the six monetary columns,
3. ``RobustScaler`` the nine-column matrix,
4. ``IsolationForest(contamination=0.03, random_state=42)``.

LOF is deliberately not reproduced: ``novelty=False`` means it cannot score
unseen rows, which is precisely what the live prediction form needs. Scores are
therefore Isolation Forest only and will not equal the ``combined_score`` column
in the watchlist, which averages both detectors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def build_anomaly_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Assemble and log-transform the anomaly feature matrix.

    Args:
        frame: Loan-level rows containing ``config.ANOMALY_FEATURES``.

    Returns:
        A frame with the canonical anomaly feature columns, log-transformed where
        required. Missing columns are filled with zeros.
    """
    matrix = pd.DataFrame(index=frame.index)
    for column in config.ANOMALY_FEATURES:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
        elif column == "interest_rate" and "Interest Rate" in frame.columns:
            values = pd.to_numeric(frame["Interest Rate"], errors="coerce")
        else:
            logger.warning("Anomaly feature %s missing; filling with zeros", column)
            values = pd.Series(np.nan, index=frame.index)

        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if column in config.ANOMALY_LOG_FEATURES:
            values = np.log1p(values.clip(lower=0.0))
        matrix[column] = values
    return matrix


class AnomalyScorer:
    """Fits and applies an Isolation Forest over the loan portfolio."""

    def __init__(self, model: Any, scaler: Any) -> None:
        """Initialise the scorer.

        Args:
            model: Fitted ``IsolationForest``.
            scaler: Fitted ``RobustScaler`` matching ``model``.
        """
        self._model = model
        self._scaler = scaler

    @classmethod
    def fit(cls, loans: pd.DataFrame) -> "AnomalyScorer | None":
        """Fit a scorer against the cleaned portfolio.

        Args:
            loans: Loan-level portfolio frame.

        Returns:
            A fitted scorer, or ``None`` when the input is unusable or scikit-learn
            is unavailable.
        """
        if loans.empty:
            logger.warning("Cannot fit anomaly scorer: empty portfolio")
            return None

        principal = pd.to_numeric(
            loans.get("Original Principal Amount (US$)", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        eligible = loans.loc[principal > 0]
        if len(eligible) < 50:
            logger.warning("Cannot fit anomaly scorer: only %d eligible loans", len(eligible))
            return None

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import RobustScaler

            matrix = build_anomaly_matrix(eligible)
            scaler = RobustScaler()
            scaled = scaler.fit_transform(matrix)
            model = IsolationForest(
                contamination=config.ANOMALY_CONTAMINATION,
                random_state=config.ANOMALY_RANDOM_STATE,
            )
            model.fit(scaled)
        except (ImportError, ValueError, MemoryError):
            logger.exception("Anomaly scorer fit failed")
            return None

        logger.info("Fitted anomaly scorer on %d loans", len(eligible))
        return cls(model, scaler)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for loan rows.

        Args:
            frame: Loan-level rows containing ``config.ANOMALY_FEATURES``.

        Returns:
            Higher-is-more-anomalous scores, or an array of ``nan`` on failure.
        """
        if frame.empty:
            return np.array([], dtype=float)
        try:
            scaled = self._scaler.transform(build_anomaly_matrix(frame))
            return -self._model.score_samples(scaled)
        except (ValueError, AttributeError):
            logger.exception("Anomaly scoring failed")
            return np.full(len(frame), np.nan)

    def is_anomaly(self, frame: pd.DataFrame) -> np.ndarray:
        """Flag rows the Isolation Forest considers outliers.

        Args:
            frame: Loan-level rows containing ``config.ANOMALY_FEATURES``.

        Returns:
            A boolean array, all ``False`` on failure.
        """
        if frame.empty:
            return np.array([], dtype=bool)
        try:
            scaled = self._scaler.transform(build_anomaly_matrix(frame))
            return self._model.predict(scaled) == -1
        except (ValueError, AttributeError):
            logger.exception("Anomaly prediction failed")
            return np.zeros(len(frame), dtype=bool)

    def score_one(self, **kwargs: float) -> dict[str, Any] | None:
        """Score a single synthetic loan described by keyword values.

        Args:
            **kwargs: Any subset of ``config.ANOMALY_FEATURES``. Absent features
                default to zero.

        Returns:
            A mapping with ``anomaly_score`` and ``is_anomaly``, or ``None`` when
            scoring failed.
        """
        row = {name: float(kwargs.get(name, 0.0)) for name in config.ANOMALY_FEATURES}
        frame = pd.DataFrame([row])
        scores = self.score(frame)
        if not scores.size or not np.isfinite(scores[0]):
            return None
        return {
            "anomaly_score": float(scores[0]),
            "is_anomaly": bool(self.is_anomaly(frame)[0]),
        }

    def threshold(self, loans: pd.DataFrame, quantile: float = 0.97) -> float:
        """Compute a reference score cut-off from the portfolio distribution.

        Useful for positioning a single prediction against the population.

        Args:
            loans: Loan-level portfolio frame.
            quantile: Quantile of the score distribution to return.

        Returns:
            The score at ``quantile``, or ``nan`` when it cannot be computed.
        """
        scores = self.score(loans)
        finite = scores[np.isfinite(scores)]
        return float(np.quantile(finite, quantile)) if finite.size else float("nan")
