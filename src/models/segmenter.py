"""Country segmentation inference using the saved K-Means artifacts.

Notebook 06 fitted its ``StandardScaler`` on a *partially* log-transformed
feature matrix: the six monetary and count columns listed in
``config.SEGMENT_LOG_FEATURES`` were passed through ``log1p`` while the five
ratio and age columns were left raw. Reproducing that split exactly recovers the
stored cluster assignment for all 148 countries; omitting it silently produces
different clusters, because the scaler's centring statistics are on the log
scale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def build_segment_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook 06's log transform to a country feature frame.

    Args:
        frame: Country-level features containing ``config.SEGMENT_FEATURES``.

    Returns:
        A frame with the same columns in canonical order, log-transformed where
        required. Missing columns are filled with zeros.
    """
    matrix = pd.DataFrame(index=frame.index)
    for column in config.SEGMENT_FEATURES:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        else:
            logger.warning("Segment feature %s missing; filling with zeros", column)
            values = pd.Series(0.0, index=frame.index)
        if column in config.SEGMENT_LOG_FEATURES:
            values = np.log1p(values.clip(lower=0.0))
        matrix[column] = values
    return matrix


class CountrySegmenter:
    """Assigns countries to the trained K-Means segments."""

    def __init__(self, kmeans: Any, scaler: Any, label_names: dict[int, str] | None = None) -> None:
        """Initialise the segmenter.

        Args:
            kmeans: Fitted ``KMeans`` estimator.
            scaler: Fitted ``StandardScaler`` matching ``kmeans``.
            label_names: Optional mapping of cluster id to human-readable name.
        """
        self._kmeans = kmeans
        self._scaler = scaler
        self._label_names = dict(label_names or {})

    @classmethod
    def load(
        cls,
        kmeans_path: Path | None = None,
        scaler_path: Path | None = None,
        segments: pd.DataFrame | None = None,
    ) -> "CountrySegmenter | None":
        """Load the K-Means and scaler artifacts from disk.

        Args:
            kmeans_path: Override for the model location. Defaults to
                ``config.COUNTRY_KMEANS_PATH``.
            scaler_path: Override for the scaler location. Defaults to
                ``config.COUNTRY_SCALER_PATH``.
            segments: Optional segments frame used to recover the cluster-id to
                segment-name mapping.

        Returns:
            A ready segmenter, or ``None`` when either artifact is unavailable.
        """
        kmeans_path = kmeans_path or config.COUNTRY_KMEANS_PATH
        scaler_path = scaler_path or config.COUNTRY_SCALER_PATH

        if not kmeans_path.exists() or not scaler_path.exists():
            logger.warning("Country segmentation artifacts missing")
            return None

        try:
            import joblib

            kmeans = joblib.load(kmeans_path)
            scaler = joblib.load(scaler_path)
        except Exception:  # noqa: BLE001
            # Broad by design; see RepaymentPredictor.load for the rationale.
            logger.exception("Could not deserialise the country segmentation models")
            return None

        return cls(kmeans, scaler, segment_name_map(segments) if segments is not None else None)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Assign cluster ids to country feature rows.

        Args:
            frame: Country-level features containing ``config.SEGMENT_FEATURES``.

        Returns:
            An integer array of cluster ids, or an array of ``-1`` when
            prediction fails.
        """
        if frame.empty:
            return np.array([], dtype=int)

        matrix = build_segment_matrix(frame)
        try:
            scaled = self._scaler.transform(matrix)
            return self._kmeans.predict(scaled).astype(int)
        except (ValueError, AttributeError):
            logger.exception("Country segment prediction failed")
            return np.full(len(frame), -1, dtype=int)

    def label(self, cluster_id: int) -> str:
        """Resolve a cluster id to its segment name.

        Args:
            cluster_id: The numeric cluster id.

        Returns:
            The segment name, or ``"Cluster {id}"`` when no name is known.
        """
        return self._label_names.get(int(cluster_id), f"Cluster {cluster_id}")

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Append cluster ids and segment names to a country frame.

        Args:
            frame: Country-level features containing ``config.SEGMENT_FEATURES``.

        Returns:
            A copy of ``frame`` with ``cluster`` and ``segment_name`` columns.
        """
        assigned = frame.copy()
        clusters = self.predict(assigned)
        assigned["cluster"] = clusters
        assigned["segment_name"] = [self.label(value) for value in clusters]
        return assigned


def segment_name_map(segments: pd.DataFrame) -> dict[int, str]:
    """Recover the cluster-id to segment-name mapping from the segments export.

    Args:
        segments: The ``country_segments.csv`` contents.

    Returns:
        A mapping of cluster id to segment name, empty when the columns are
        absent.
    """
    if segments.empty or not {"cluster", "segment_name"}.issubset(segments.columns):
        return {}
    pairs = segments[["cluster", "segment_name"]].dropna().drop_duplicates()
    return {int(row.cluster): str(row.segment_name) for row in pairs.itertuples()}


def country_features_from_loans(loans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate loan-level rows into the country feature set used by K-Means.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        One row per country with ``config.SEGMENT_FEATURES`` columns, or an empty
        frame when the input lacks a country column.
    """
    if loans.empty or "Country / Economy" not in loans.columns:
        return pd.DataFrame()

    data = loans.copy()
    for column in ("is_cancelled", "is_active"):
        if column not in data.columns:
            data[column] = False

    grouped = data.groupby("Country / Economy", as_index=False).agg(
        total_commitments=("Original Principal Amount (US$)", "sum"),
        total_disbursed=("Disbursed Amount (US$)", "sum"),
        total_repaid=("Repaid to IBRD (US$)", "sum"),
        total_outstanding=("Due to IBRD (US$)", "sum"),
        num_loans=("Loan Number", "count"),
        avg_loan_size=("Original Principal Amount (US$)", "mean"),
        avg_repayment_ratio=("repayment_ratio", "mean"),
        avg_risk_score=("risk_score", "mean"),
        avg_loan_age=("loan_age_years", "mean"),
        cancellation_rate=("is_cancelled", "mean"),
        active_loans_count=("is_active", "sum"),
    )
    grouped["cancellation_rate"] = grouped["cancellation_rate"] * 100.0
    return grouped
