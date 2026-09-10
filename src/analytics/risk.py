"""Risk scoring aggregations over the loan portfolio."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_BINS: tuple[float, ...] = (-0.001, 0.25, 0.5, 0.75, 1.0)
RISK_LABELS: tuple[str, ...] = ("Low", "Moderate", "Elevated", "High")


def add_risk_bands(loans: pd.DataFrame) -> pd.DataFrame:
    """Append a discretised risk band to each loan.

    Args:
        loans: Loan-level portfolio frame containing ``risk_score``.

    Returns:
        A copy of ``loans`` with a ``risk_band`` column. Returns the input
        unchanged when ``risk_score`` is absent.
    """
    if loans.empty or "risk_score" not in loans.columns:
        return loans.copy()

    data = loans.copy()
    scores = pd.to_numeric(data["risk_score"], errors="coerce").clip(0.0, 1.0)
    data["risk_band"] = pd.cut(scores, bins=list(RISK_BINS), labels=list(RISK_LABELS))
    return data


def risk_distribution(loans: pd.DataFrame) -> pd.DataFrame:
    """Count loans and outstanding exposure per risk band.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``risk_band``, ``count`` and ``outstanding`` in band order.
    """
    banded = add_risk_bands(loans)
    if banded.empty or "risk_band" not in banded.columns:
        return pd.DataFrame(columns=["risk_band", "count", "outstanding"])

    summary = (
        banded.groupby("risk_band", as_index=False, observed=False)
        .agg(count=("risk_band", "size"), outstanding=("Due to IBRD (US$)", "sum"))
    )
    summary["risk_band"] = summary["risk_band"].astype(str)
    return summary.reset_index(drop=True)


def risk_by_dimension(loans: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Average risk and exposure grouped by an arbitrary column.

    Args:
        loans: Loan-level portfolio frame.
        dimension: Column to group by, such as ``"Region"`` or ``"age_category"``.

    Returns:
        A frame of the dimension plus ``loans``, ``avg_risk`` and ``outstanding``,
        ordered by average risk descending.
    """
    if loans.empty or dimension not in loans.columns or "risk_score" not in loans.columns:
        return pd.DataFrame(columns=[dimension, "loans", "avg_risk", "outstanding"])

    return (
        loans.groupby(dimension, as_index=False)
        .agg(
            loans=(dimension, "size"),
            avg_risk=("risk_score", "mean"),
            outstanding=("Due to IBRD (US$)", "sum"),
        )
        .sort_values("avg_risk", ascending=False)
        .reset_index(drop=True)
    )


def high_risk_summary(high_risk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the high-risk extract by region.

    Args:
        high_risk: Contents of ``high_risk_loans.csv``.

    Returns:
        A frame of ``Region``, ``count``, ``outstanding`` and ``avg_risk``,
        ordered by outstanding descending.
    """
    if high_risk.empty or "Region" not in high_risk.columns:
        return pd.DataFrame(columns=["Region", "count", "outstanding", "avg_risk"])

    return (
        high_risk.groupby("Region", as_index=False)
        .agg(
            count=("Region", "size"),
            outstanding=("Due to IBRD (US$)", "sum"),
            avg_risk=("risk_score", "mean"),
        )
        .sort_values("outstanding", ascending=False)
        .reset_index(drop=True)
    )


def top_exposures(loans: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Rank individual loans by outstanding exposure.

    Args:
        loans: Loan-level portfolio frame.
        top_n: Number of loans to return.

    Returns:
        A frame of identifying columns plus exposure and risk, ordered by
        outstanding descending.
    """
    if loans.empty or "Due to IBRD (US$)" not in loans.columns:
        return pd.DataFrame()

    columns = [
        column
        for column in (
            "Loan Number",
            "Country / Economy",
            "Region",
            "Loan Type",
            "Loan Status",
            "Original Principal Amount (US$)",
            "Due to IBRD (US$)",
            "repayment_ratio",
            "risk_score",
        )
        if column in loans.columns
    ]
    return (
        loans.nlargest(top_n, "Due to IBRD (US$)")[columns]
        .reset_index(drop=True)
    )


def risk_vs_age(loans: pd.DataFrame) -> pd.DataFrame:
    """Bin loans by age and report the risk profile of each bin.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``age_category``, ``loans``, ``avg_risk`` and ``outstanding``.
    """
    if loans.empty or "age_category" not in loans.columns:
        return pd.DataFrame(columns=["age_category", "loans", "avg_risk", "outstanding"])

    order = ["0-5", "5-10", "10-20", "20-30", "30-50", "50+"]
    summary = loans.groupby("age_category", as_index=False).agg(
        loans=("age_category", "size"),
        avg_risk=("risk_score", "mean"),
        outstanding=("Due to IBRD (US$)", "sum"),
    )
    summary["_order"] = summary["age_category"].apply(
        lambda value: order.index(value) if value in order else len(order)
    )
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def interest_rate_vs_risk(loans: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
    """Sample loans for an interest-rate against risk scatter.

    Args:
        loans: Loan-level portfolio frame.
        sample_size: Maximum number of points to return.

    Returns:
        A frame of ``Interest Rate``, ``risk_score``, ``Region`` and
        ``Original Principal Amount (US$)``.
    """
    required = {"Interest Rate", "risk_score"}
    if loans.empty or not required.issubset(loans.columns):
        return pd.DataFrame(columns=["Interest Rate", "risk_score", "Region", "Original Principal Amount (US$)"])

    columns = [
        column
        for column in ("Interest Rate", "risk_score", "Region", "Original Principal Amount (US$)")
        if column in loans.columns
    ]
    data = loans[columns].copy()
    data["Interest Rate"] = pd.to_numeric(data["Interest Rate"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Interest Rate", "risk_score"])

    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)
    return data.reset_index(drop=True)
