"""Cached readers for the dashboard's on-disk data assets.

Every loader degrades to an empty ``DataFrame`` when its source file is absent so
that a partially provisioned checkout still renders. Callers inspect
``DataFrame.empty`` to decide whether to show a fallback message.

Caching note:
    Loaders are memoised with :func:`functools.lru_cache` and therefore return a
    *shared* frame. Consumers must not mutate the result in place; the analytics
    layer copies defensively at function entry.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

_DATE_COLUMNS: tuple[str, ...] = (
    "Board Approval Date",
    "Agreement Signing Date",
    "Effective Date (Most Recent)",
    "Closed Date (Most Recent)",
    "Last Disbursement Date",
    "First Repayment Date",
    "Last Repayment Date",
    "End of Period",
)


def _read_csv(path: Path, parse_dates: bool = False) -> pd.DataFrame:
    """Read a CSV file, returning an empty frame when it is missing.

    Args:
        path: Location of the CSV file.
        parse_dates: When ``True``, convert the known IBRD date columns to
            datetimes after loading.

    Returns:
        The parsed frame, or an empty frame when the file does not exist or
        cannot be parsed.
    """
    if not path.exists():
        logger.warning("Data asset not found: %s", path)
        return pd.DataFrame()

    try:
        # utf-8-sig transparently strips a byte-order mark if one is present.
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        logger.exception("Failed to read data asset: %s", path)
        return pd.DataFrame()

    if parse_dates:
        for column in _DATE_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

    logger.info("Loaded %s (%d rows, %d columns)", path.name, len(frame), frame.shape[1])
    return frame


@lru_cache(maxsize=1)
def load_loans(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned IBRD loan-level dataset.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``config.LOANS_PATH``.

    Returns:
        The cleaned loan portfolio, or an empty frame when unavailable.
    """
    frame = _read_csv(path or config.LOANS_PATH, parse_dates=True)
    if frame.empty:
        return frame

    # Notebook 08 works from a lowercase alias; expose both so downstream code
    # can rely on a single name regardless of which asset it came from.
    if "interest_rate" not in frame.columns and "Interest Rate" in frame.columns:
        frame["interest_rate"] = pd.to_numeric(frame["Interest Rate"], errors="coerce")
    return frame


@lru_cache(maxsize=1)
def load_country_segments(path: Path | None = None) -> pd.DataFrame:
    """Load the K-Means country segmentation output.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``config.COUNTRY_SEGMENTS_PATH``.

    Returns:
        One row per country with its cluster id and segment name, or an empty
        frame when unavailable.
    """
    return _read_csv(path or config.COUNTRY_SEGMENTS_PATH)


@lru_cache(maxsize=1)
def load_anomaly_watchlist(path: Path | None = None) -> pd.DataFrame:
    """Load the precomputed anomaly watchlist.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``config.ANOMALY_WATCHLIST_PATH``.

    Returns:
        Flagged loans ranked by combined anomaly score, or an empty frame when
        unavailable.
    """
    return _read_csv(path or config.ANOMALY_WATCHLIST_PATH)


@lru_cache(maxsize=1)
def load_high_risk_loans(path: Path | None = None) -> pd.DataFrame:
    """Load the high-risk loan extract.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``config.HIGH_RISK_LOANS_PATH``.

    Returns:
        Loans identified as high risk during notebook 04, or an empty frame when
        unavailable.
    """
    return _read_csv(path or config.HIGH_RISK_LOANS_PATH, parse_dates=True)


@lru_cache(maxsize=1)
def load_forecasts(path: Path | None = None) -> pd.DataFrame:
    """Load the annual commitment and outstanding forecast table.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``config.FORECASTS_PATH``.

    Returns:
        Combined historical and forecast rows keyed by year, or an empty frame
        when unavailable.
    """
    return _read_csv(path or config.FORECASTS_PATH)


def asset_status() -> dict[str, bool]:
    """Report which dashboard data assets are present on disk.

    Returns:
        A mapping of human-readable asset name to existence flag, suitable for
        rendering a provisioning checklist.
    """
    return {
        "Cleaned loans": config.LOANS_PATH.exists(),
        "Country segments": config.COUNTRY_SEGMENTS_PATH.exists(),
        "Anomaly watchlist": config.ANOMALY_WATCHLIST_PATH.exists(),
        "High-risk loans": config.HIGH_RISK_LOANS_PATH.exists(),
        "Forecasts": config.FORECASTS_PATH.exists(),
        "Repayment model": config.REPAYMENT_MODEL_PATH.exists(),
        "Repayment features": config.REPAYMENT_FEATURES_PATH.exists(),
        "Country K-Means": config.COUNTRY_KMEANS_PATH.exists(),
        "Country scaler": config.COUNTRY_SCALER_PATH.exists(),
    }


def clear_caches() -> None:
    """Drop every memoised frame so the next call re-reads from disk."""
    for loader in (
        load_loans,
        load_country_segments,
        load_anomaly_watchlist,
        load_high_risk_loans,
        load_forecasts,
    ):
        loader.cache_clear()
    logger.info("Cleared all data loader caches")
