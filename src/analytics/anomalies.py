"""Aggregations over the anomaly watchlist produced by notebook 08."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def watchlist_kpis(watchlist: pd.DataFrame) -> dict[str, float | int]:
    """Compute headline metrics for the anomaly watchlist.

    Args:
        watchlist: Contents of ``anomaly_watchlist.csv``.

    Returns:
        A mapping with the flagged-loan count, exposure at risk, mean combined
        score and number of distinct countries.
    """
    if watchlist.empty:
        return {"flagged_loans": 0, "exposure": 0.0, "mean_score": 0.0, "countries": 0}

    exposure = (
        float(pd.to_numeric(watchlist["Due to IBRD (US$)"], errors="coerce").fillna(0.0).sum())
        if "Due to IBRD (US$)" in watchlist.columns
        else 0.0
    )
    mean_score = (
        float(pd.to_numeric(watchlist["combined_score"], errors="coerce").mean())
        if "combined_score" in watchlist.columns
        else 0.0
    )
    return {
        "flagged_loans": int(len(watchlist)),
        "exposure": exposure,
        "mean_score": mean_score,
        "countries": (
            int(watchlist["Country / Economy"].nunique())
            if "Country / Economy" in watchlist.columns
            else 0
        ),
    }


def reason_breakdown(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Count flagged loans by the driving feature.

    Args:
        watchlist: Contents of ``anomaly_watchlist.csv``.

    Returns:
        A frame of ``reason``, ``count`` and ``exposure``, ordered by count
        descending.
    """
    if watchlist.empty or "reason" not in watchlist.columns:
        return pd.DataFrame(columns=["reason", "count", "exposure"])

    data = watchlist.copy()
    data["reason"] = data["reason"].fillna("Unspecified").replace("", "Unspecified")
    aggregations: dict[str, tuple[str, str]] = {"count": ("reason", "size")}
    if "Due to IBRD (US$)" in data.columns:
        aggregations["exposure"] = ("Due to IBRD (US$)", "sum")

    summary = data.groupby("reason", as_index=False).agg(**aggregations)
    if "exposure" not in summary.columns:
        summary["exposure"] = 0.0
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


def region_breakdown(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Count flagged loans by region.

    Args:
        watchlist: Contents of ``anomaly_watchlist.csv``.

    Returns:
        A frame of ``Region``, ``count`` and ``exposure``, ordered by count
        descending.
    """
    if watchlist.empty or "Region" not in watchlist.columns:
        return pd.DataFrame(columns=["Region", "count", "exposure"])

    aggregations: dict[str, tuple[str, str]] = {"count": ("Region", "size")}
    if "Due to IBRD (US$)" in watchlist.columns:
        aggregations["exposure"] = ("Due to IBRD (US$)", "sum")

    summary = watchlist.groupby("Region", as_index=False).agg(**aggregations)
    if "exposure" not in summary.columns:
        summary["exposure"] = 0.0
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


def top_anomalies(watchlist: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Return the highest-scoring flagged loans.

    Args:
        watchlist: Contents of ``anomaly_watchlist.csv``.
        top_n: Number of rows to return.

    Returns:
        A frame of display columns ordered by combined score descending.
    """
    if watchlist.empty:
        return pd.DataFrame()

    columns = [
        column
        for column in (
            "Loan Number",
            "Country / Economy",
            "Region",
            "Loan Status",
            "Original Principal Amount (US$)",
            "Disbursed Amount (US$)",
            "Due to IBRD (US$)",
            "repayment_ratio",
            "interest_rate",
            "reason",
            "combined_score",
        )
        if column in watchlist.columns
    ]
    data = watchlist[columns].copy()
    if "combined_score" in data.columns:
        data = data.sort_values("combined_score", ascending=False)
    return data.head(top_n).reset_index(drop=True)


def score_distribution(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Extract the combined-score series for a histogram.

    Args:
        watchlist: Contents of ``anomaly_watchlist.csv``.

    Returns:
        A frame with a single ``combined_score`` column, empty when unavailable.
    """
    if watchlist.empty or "combined_score" not in watchlist.columns:
        return pd.DataFrame(columns=["combined_score"])

    scores = pd.to_numeric(watchlist["combined_score"], errors="coerce").dropna()
    return pd.DataFrame({"combined_score": scores.to_numpy()})
