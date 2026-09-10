"""Aggregations over the K-Means country segmentation output."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROFILE_FEATURES: tuple[str, ...] = (
    "avg_repayment_ratio",
    "avg_risk_score",
    "avg_loan_age",
    "cancellation_rate",
)


def segment_overview(segments: pd.DataFrame) -> pd.DataFrame:
    """Summarise each segment's size and aggregate exposure.

    Args:
        segments: Contents of ``country_segments.csv``.

    Returns:
        A frame of ``segment_name``, ``countries``, ``total_commitments``,
        ``total_outstanding``, ``num_loans``, ``avg_repayment_ratio`` and
        ``avg_risk_score``, ordered by commitments descending.
    """
    if segments.empty or "segment_name" not in segments.columns:
        return pd.DataFrame(
            columns=[
                "segment_name",
                "countries",
                "total_commitments",
                "total_outstanding",
                "num_loans",
                "avg_repayment_ratio",
                "avg_risk_score",
            ]
        )

    return (
        segments.groupby("segment_name", as_index=False)
        .agg(
            countries=("segment_name", "size"),
            total_commitments=("total_commitments", "sum"),
            total_outstanding=("total_outstanding", "sum"),
            num_loans=("num_loans", "sum"),
            avg_repayment_ratio=("avg_repayment_ratio", "mean"),
            avg_risk_score=("avg_risk_score", "mean"),
        )
        .sort_values("total_commitments", ascending=False)
        .reset_index(drop=True)
    )


def segment_profiles(segments: pd.DataFrame) -> pd.DataFrame:
    """Compute normalised segment centroids for a radar comparison.

    Each feature is min-max scaled across segments so the axes share a 0-1 range.

    Args:
        segments: Contents of ``country_segments.csv``.

    Returns:
        A long frame of ``segment_name``, ``feature``, ``value`` and
        ``normalised``.
    """
    available = [column for column in _PROFILE_FEATURES if column in segments.columns]
    if segments.empty or "segment_name" not in segments.columns or not available:
        return pd.DataFrame(columns=["segment_name", "feature", "value", "normalised"])

    centroids = segments.groupby("segment_name", as_index=False)[available].mean()
    long = centroids.melt(id_vars="segment_name", var_name="feature", value_name="value")

    span = long.groupby("feature")["value"].transform(lambda values: values.max() - values.min())
    floor = long.groupby("feature")["value"].transform("min")
    long["normalised"] = ((long["value"] - floor) / span.replace(0.0, np.nan)).fillna(0.5).astype(float)
    return long.reset_index(drop=True)


def countries_in_segment(segments: pd.DataFrame, segment_name: str, top_n: int = 25) -> pd.DataFrame:
    """List the largest countries within a segment.

    Args:
        segments: Contents of ``country_segments.csv``.
        segment_name: Segment to filter to.
        top_n: Number of countries to return.

    Returns:
        A frame of country-level columns ordered by commitments descending.
    """
    if segments.empty or "segment_name" not in segments.columns:
        return pd.DataFrame()

    columns = [
        column
        for column in (
            "Country / Economy",
            "segment_name",
            "num_loans",
            "total_commitments",
            "total_outstanding",
            "avg_repayment_ratio",
            "avg_risk_score",
            "cancellation_rate",
        )
        if column in segments.columns
    ]
    subset = segments.loc[segments["segment_name"] == segment_name, columns]
    if "total_commitments" in subset.columns:
        subset = subset.sort_values("total_commitments", ascending=False)
    return subset.head(top_n).reset_index(drop=True)


def segment_scatter_data(segments: pd.DataFrame) -> pd.DataFrame:
    """Prepare a repayment against risk scatter of every country.

    Args:
        segments: Contents of ``country_segments.csv``.

    Returns:
        A frame of ``Country / Economy``, ``segment_name``,
        ``avg_repayment_ratio``, ``avg_risk_score``, ``total_commitments`` and
        ``num_loans``.
    """
    required = {"avg_repayment_ratio", "avg_risk_score", "segment_name"}
    if segments.empty or not required.issubset(segments.columns):
        return pd.DataFrame(
            columns=[
                "Country / Economy",
                "segment_name",
                "avg_repayment_ratio",
                "avg_risk_score",
                "total_commitments",
                "num_loans",
            ]
        )

    columns = [
        column
        for column in (
            "Country / Economy",
            "segment_name",
            "avg_repayment_ratio",
            "avg_risk_score",
            "total_commitments",
            "num_loans",
        )
        if column in segments.columns
    ]
    return segments[columns].dropna(subset=["avg_repayment_ratio", "avg_risk_score"]).reset_index(drop=True)


def segment_names(segments: pd.DataFrame) -> list[str]:
    """List the distinct segment names present.

    Args:
        segments: Contents of ``country_segments.csv``.

    Returns:
        Sorted segment names, empty when unavailable.
    """
    if segments.empty or "segment_name" not in segments.columns:
        return []
    return sorted(segments["segment_name"].dropna().astype(str).unique().tolist())
