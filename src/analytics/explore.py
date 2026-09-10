"""Filtering helpers shared by the dashboard's interactive views.

These functions are intentionally cheap so they can run outside any cache on
every rerun, leaving the expensive CSV read to the cached loader.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

DISPLAY_COLUMNS: tuple[str, ...] = (
    "Loan Number",
    "Country / Economy",
    "Region",
    "Borrower",
    "Loan Type",
    "Loan Status",
    "Project Name",
    "Interest Rate",
    "Original Principal Amount (US$)",
    "Disbursed Amount (US$)",
    "Undisbursed Amount (US$)",
    "Repaid to IBRD (US$)",
    "Due to IBRD (US$)",
    "loan_age_years",
    "repayment_ratio",
    "risk_score",
    "loan_size_category",
    "portfolio_status",
    "approval_year",
)


def filter_options(loans: pd.DataFrame, column: str) -> list[str]:
    """List the distinct values available for a filter widget.

    Args:
        loans: Loan-level portfolio frame.
        column: Column to enumerate.

    Returns:
        Sorted distinct values as strings, empty when the column is absent.
    """
    if loans.empty or column not in loans.columns:
        return []
    return sorted(loans[column].dropna().astype(str).unique().tolist())


def year_bounds(loans: pd.DataFrame) -> tuple[int, int]:
    """Report the approval-year range present in the portfolio.

    Args:
        loans: Loan-level portfolio frame.

    Returns:
        A ``(minimum, maximum)`` tuple, defaulting to ``(1947, 2026)`` when the
        column is missing or unparseable.
    """
    if loans.empty or "approval_year" not in loans.columns:
        return 1947, 2026
    years = pd.to_numeric(loans["approval_year"], errors="coerce").dropna()
    if years.empty:
        return 1947, 2026
    return int(years.min()), int(years.max())


def apply_filters(
    loans: pd.DataFrame,
    regions: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    loan_types: Sequence[str] | None = None,
    size_categories: Sequence[str] | None = None,
    year_range: tuple[int, int] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """Filter the portfolio down to the requested slice.

    An empty or ``None`` value for any argument leaves that dimension unfiltered.

    Args:
        loans: Loan-level portfolio frame.
        regions: Regions to keep.
        statuses: Loan statuses to keep.
        loan_types: Loan type codes to keep.
        size_categories: Loan size buckets to keep.
        year_range: Inclusive ``(start, end)`` approval-year bounds.
        search: Case-insensitive substring matched against loan number, country,
            borrower and project name.

    Returns:
        A filtered copy of ``loans``.
    """
    if loans.empty:
        return loans.copy()

    mask = pd.Series(True, index=loans.index)

    for column, values in (
        ("Region", regions),
        ("Loan Status", statuses),
        ("Loan Type", loan_types),
        ("loan_size_category", size_categories),
    ):
        if values and column in loans.columns:
            mask &= loans[column].astype(str).isin([str(value) for value in values])

    if year_range and "approval_year" in loans.columns:
        years = pd.to_numeric(loans["approval_year"], errors="coerce")
        mask &= years.between(year_range[0], year_range[1])

    if search:
        needle = search.strip().lower()
        if needle:
            searchable = [
                column
                for column in ("Loan Number", "Country / Economy", "Borrower", "Project Name")
                if column in loans.columns
            ]
            if searchable:
                haystack = (
                    loans[searchable]
                    .astype(str)
                    .apply(lambda row: " ".join(row).lower(), axis=1)
                )
                mask &= haystack.str.contains(needle, regex=False, na=False)

    return loans.loc[mask].copy()


def display_frame(loans: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Project the portfolio onto a readable column subset.

    Args:
        loans: Loan-level portfolio frame.
        columns: Columns to keep. Defaults to :data:`DISPLAY_COLUMNS`.

    Returns:
        A copy containing only the requested columns that actually exist.
    """
    if loans.empty:
        return loans.copy()
    wanted = list(columns) if columns else list(DISPLAY_COLUMNS)
    present = [column for column in wanted if column in loans.columns]
    return loans[present].copy() if present else loans.copy()


def describe_numeric(loans: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Produce summary statistics for the numeric columns of a slice.

    Args:
        loans: Loan-level portfolio frame.
        columns: Numeric columns to describe. Defaults to a curated set.

    Returns:
        A frame of ``metric`` plus one column per described field, empty when
        nothing numeric is available.
    """
    default = (
        "Original Principal Amount (US$)",
        "Disbursed Amount (US$)",
        "Due to IBRD (US$)",
        "repayment_ratio",
        "risk_score",
        "loan_age_years",
        "Interest Rate",
    )
    wanted = list(columns) if columns else list(default)
    present = [column for column in wanted if column in loans.columns]
    if loans.empty or not present:
        return pd.DataFrame()

    described = loans[present].apply(pd.to_numeric, errors="coerce").describe().reset_index()
    return described.rename(columns={"index": "metric"})


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Encode a frame as UTF-8 CSV bytes for download.

    Args:
        frame: Frame to encode.

    Returns:
        The encoded CSV payload.
    """
    return frame.to_csv(index=False).encode("utf-8")


def to_json_bytes(record: dict[str, Any]) -> bytes:
    """Encode a mapping as pretty-printed UTF-8 JSON bytes for download.

    Args:
        record: Mapping to encode. Values are stringified when not natively
            serialisable.

    Returns:
        The encoded JSON payload.
    """
    import json

    return json.dumps(record, indent=2, default=str).encode("utf-8")
