"""Regional and country-level portfolio aggregations."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def regional_summary(loans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the portfolio by region.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``Region``, ``loans``, ``commitments``, ``disbursed``,
        ``outstanding``, ``avg_risk`` and ``repayment_rate``, ordered by
        commitments descending.
    """
    if loans.empty or "Region" not in loans.columns:
        return pd.DataFrame(
            columns=["Region", "loans", "commitments", "disbursed", "outstanding", "avg_risk", "repayment_rate"]
        )

    summary = loans.groupby("Region", as_index=False).agg(
        loans=("Region", "size"),
        commitments=("Original Principal Amount (US$)", "sum"),
        disbursed=("Disbursed Amount (US$)", "sum"),
        repaid=("Repaid to IBRD (US$)", "sum"),
        outstanding=("Due to IBRD (US$)", "sum"),
        avg_risk=("risk_score", "mean"),
    )
    summary["repayment_rate"] = (
        summary["repaid"].div(summary["disbursed"].replace(0.0, np.nan)) * 100.0
    ).fillna(0.0)
    return summary.sort_values("commitments", ascending=False).reset_index(drop=True)


def country_summary(loans: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Aggregate the portfolio by country.

    Args:
        loans: Loan-level portfolio frame.
        top_n: When given, keep only the ``top_n`` countries by commitments.

    Returns:
        A frame of ``Country / Economy``, ``Region``, ``loans``, ``commitments``,
        ``outstanding``, ``avg_risk`` and ``repayment_rate``.
    """
    if loans.empty or "Country / Economy" not in loans.columns:
        return pd.DataFrame(
            columns=["Country / Economy", "Region", "loans", "commitments", "outstanding", "avg_risk", "repayment_rate"]
        )

    data = loans.copy()
    if "Region" not in data.columns:
        data["Region"] = "Unknown"

    summary = data.groupby("Country / Economy", as_index=False).agg(
        Region=("Region", "first"),
        loans=("Country / Economy", "size"),
        commitments=("Original Principal Amount (US$)", "sum"),
        disbursed=("Disbursed Amount (US$)", "sum"),
        repaid=("Repaid to IBRD (US$)", "sum"),
        outstanding=("Due to IBRD (US$)", "sum"),
        avg_risk=("risk_score", "mean"),
    )
    summary["repayment_rate"] = (
        summary["repaid"].div(summary["disbursed"].replace(0.0, np.nan)) * 100.0
    ).fillna(0.0)
    summary = summary.sort_values("commitments", ascending=False).reset_index(drop=True)
    return summary.head(top_n) if top_n else summary


def region_country_treemap_data(loans: pd.DataFrame, top_n: int = 60) -> pd.DataFrame:
    """Build a region/country hierarchy for a treemap.

    Args:
        loans: Loan-level portfolio frame.
        top_n: Maximum number of countries to include.

    Returns:
        A frame of ``Region``, ``Country / Economy``, ``commitments`` and
        ``avg_risk``.
    """
    summary = country_summary(loans, top_n=top_n)
    if summary.empty:
        return pd.DataFrame(columns=["Region", "Country / Economy", "commitments", "avg_risk"])
    return summary[["Region", "Country / Economy", "commitments", "avg_risk"]].copy()


def regional_yearly_trend(loans: pd.DataFrame) -> pd.DataFrame:
    """Track commitments by region and approval year.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``approval_year``, ``Region`` and ``commitments``.
    """
    required = {"approval_year", "Region"}
    if loans.empty or not required.issubset(loans.columns):
        return pd.DataFrame(columns=["approval_year", "Region", "commitments"])

    data = loans.copy()
    data["approval_year"] = pd.to_numeric(data["approval_year"], errors="coerce")
    data = data.dropna(subset=["approval_year"])

    summary = (
        data.groupby(["approval_year", "Region"], as_index=False)
        .agg(commitments=("Original Principal Amount (US$)", "sum"))
        .sort_values("approval_year")
    )
    summary["approval_year"] = summary["approval_year"].astype(int)
    return summary.reset_index(drop=True)
