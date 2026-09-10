"""Portfolio-level key performance indicators.

Every figure is computed live from the loan-level frame. The precomputed
``data/features/portfolio_summary.csv`` is deliberately ignored: it is stale
output from ``run_pipeline.py`` running against the 1,200-row synthetic dataset
and disagrees with the real 9,518-loan portfolio.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_sum(frame: pd.DataFrame, column: str) -> float:
    """Sum a column, tolerating absence and non-numeric content.

    Args:
        frame: Source frame.
        column: Column to sum.

    Returns:
        The column total, or ``0.0`` when the column is missing.
    """
    if column not in frame.columns:
        logger.warning("Column %s missing; treating total as zero", column)
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def portfolio_kpis(loans: pd.DataFrame) -> dict[str, Any]:
    """Compute headline portfolio metrics.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A mapping of metric name to value. Returns zeroed metrics when the input
        is empty.
    """
    if loans.empty:
        return {
            "num_loans": 0,
            "num_countries": 0,
            "num_regions": 0,
            "total_commitments": 0.0,
            "total_disbursed": 0.0,
            "total_repaid": 0.0,
            "total_outstanding": 0.0,
            "total_undisbursed": 0.0,
            "total_cancelled": 0.0,
            "avg_loan_size": 0.0,
            "median_loan_size": 0.0,
            "repayment_rate": 0.0,
            "cancellation_rate": 0.0,
            "active_loans": 0,
            "avg_risk_score": 0.0,
        }

    data = loans.copy()
    commitments = _safe_sum(data, "Original Principal Amount (US$)")
    disbursed = _safe_sum(data, "Disbursed Amount (US$)")
    repaid = _safe_sum(data, "Repaid to IBRD (US$)")
    cancelled = _safe_sum(data, "Cancelled Amount (US$)")
    principal = pd.to_numeric(
        data.get("Original Principal Amount (US$)", pd.Series(dtype=float)), errors="coerce"
    )

    return {
        "num_loans": int(len(data)),
        "num_countries": int(data["Country / Economy"].nunique()) if "Country / Economy" in data else 0,
        "num_regions": int(data["Region"].nunique()) if "Region" in data else 0,
        "total_commitments": commitments,
        "total_disbursed": disbursed,
        "total_repaid": repaid,
        "total_outstanding": _safe_sum(data, "Due to IBRD (US$)"),
        "total_undisbursed": _safe_sum(data, "Undisbursed Amount (US$)"),
        "total_cancelled": cancelled,
        "avg_loan_size": float(principal.mean()) if not principal.empty else 0.0,
        "median_loan_size": float(principal.median()) if not principal.empty else 0.0,
        "repayment_rate": (repaid / disbursed * 100.0) if disbursed else 0.0,
        "cancellation_rate": (cancelled / commitments * 100.0) if commitments else 0.0,
        "active_loans": int(data["is_active"].sum()) if "is_active" in data else 0,
        "avg_risk_score": (
            float(pd.to_numeric(data["risk_score"], errors="coerce").mean())
            if "risk_score" in data
            else 0.0
        ),
    }


def status_breakdown(loans: pd.DataFrame) -> pd.DataFrame:
    """Summarise the portfolio by loan status.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``Loan Status``, ``count``, ``total_amount`` and
        ``pct_portfolio``, ordered by count descending.
    """
    if loans.empty or "Loan Status" not in loans.columns:
        return pd.DataFrame(columns=["Loan Status", "count", "total_amount", "pct_portfolio"])

    summary = (
        loans.groupby("Loan Status", as_index=False)
        .agg(count=("Loan Status", "size"), total_amount=("Original Principal Amount (US$)", "sum"))
        .sort_values("count", ascending=False)
    )
    total = summary["count"].sum()
    summary["pct_portfolio"] = summary["count"] / total * 100.0 if total else 0.0
    return summary.reset_index(drop=True)


def portfolio_status_breakdown(loans: pd.DataFrame) -> pd.DataFrame:
    """Summarise the portfolio by derived lifecycle stage.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``portfolio_status``, ``count`` and ``total_outstanding``.
    """
    if loans.empty or "portfolio_status" not in loans.columns:
        return pd.DataFrame(columns=["portfolio_status", "count", "total_outstanding"])

    return (
        loans.groupby("portfolio_status", as_index=False)
        .agg(count=("portfolio_status", "size"), total_outstanding=("Due to IBRD (US$)", "sum"))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def yearly_activity(loans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lending activity by board approval year.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``approval_year``, ``loans``, ``commitments``, ``disbursed``
        and ``repaid``, ordered by year.
    """
    if loans.empty or "approval_year" not in loans.columns:
        return pd.DataFrame(columns=["approval_year", "loans", "commitments", "disbursed", "repaid"])

    data = loans.copy()
    data["approval_year"] = pd.to_numeric(data["approval_year"], errors="coerce")
    data = data.dropna(subset=["approval_year"])

    summary = (
        data.groupby("approval_year", as_index=False)
        .agg(
            loans=("approval_year", "size"),
            commitments=("Original Principal Amount (US$)", "sum"),
            disbursed=("Disbursed Amount (US$)", "sum"),
            repaid=("Repaid to IBRD (US$)", "sum"),
        )
        .sort_values("approval_year")
    )
    summary["approval_year"] = summary["approval_year"].astype(int)
    return summary.reset_index(drop=True)


def loan_size_breakdown(loans: pd.DataFrame) -> pd.DataFrame:
    """Summarise the portfolio by loan size category.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A frame of ``loan_size_category``, ``count``, ``avg_commitment`` and
        ``avg_risk``.
    """
    if loans.empty or "loan_size_category" not in loans.columns:
        return pd.DataFrame(columns=["loan_size_category", "count", "avg_commitment", "avg_risk"])

    order = ["Small", "Medium", "Large", "Mega"]
    summary = loans.groupby("loan_size_category", as_index=False).agg(
        count=("loan_size_category", "size"),
        avg_commitment=("Original Principal Amount (US$)", "mean"),
        avg_risk=("risk_score", "mean"),
    )
    summary["_order"] = summary["loan_size_category"].apply(
        lambda value: order.index(value) if value in order else len(order)
    )
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
