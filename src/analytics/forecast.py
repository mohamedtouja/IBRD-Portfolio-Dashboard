"""Readers for the annual commitment and outstanding forecast.

``data/features/forecasts.csv`` (notebook 07) stacks history and projection into
one table. Historical rows carry ``commitments_billions`` and
``outstanding_billions``; forecast rows carry ``base_commitment_billions``,
``base_outstanding_billions`` and the three scenario columns. The split is
therefore inferred from which columns are populated rather than from a flag.

Values are already expressed in US$ billions.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_HISTORY_COLUMNS: tuple[str, ...] = ("commitments_billions", "outstanding_billions")
_FORECAST_COLUMNS: tuple[str, ...] = ("base_commitment_billions", "base_outstanding_billions")
_SCENARIO_COLUMNS: tuple[str, ...] = (
    "base_case_commitments",
    "best_case_commitments",
    "worst_case_commitments",
)


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce the given columns to numeric in place on a copy.

    Args:
        frame: Source frame.
        columns: Columns to coerce where present.

    Returns:
        A copy with the requested columns coerced.
    """
    data = frame.copy()
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def split_history_forecast(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the combined forecast table into history and projection.

    Args:
        forecasts: Contents of ``forecasts.csv``.

    Returns:
        A ``(history, projection)`` tuple. Both frames are empty when the input
        is empty or lacks a ``year`` column.
    """
    if forecasts.empty or "year" not in forecasts.columns:
        return pd.DataFrame(), pd.DataFrame()

    data = _numeric(forecasts, ("year",) + _HISTORY_COLUMNS + _FORECAST_COLUMNS + _SCENARIO_COLUMNS)
    data = data.dropna(subset=["year"]).sort_values("year")
    data["year"] = data["year"].astype(int)

    history_available = [column for column in _HISTORY_COLUMNS if column in data.columns]
    forecast_available = [column for column in _FORECAST_COLUMNS if column in data.columns]

    history = (
        data[data[history_available].notna().any(axis=1)] if history_available else data.iloc[0:0]
    )
    projection = (
        data[data[forecast_available].notna().any(axis=1)] if forecast_available else data.iloc[0:0]
    )
    return history.reset_index(drop=True), projection.reset_index(drop=True)


def commitment_series(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Build a single commitments series spanning history and forecast.

    Args:
        forecasts: Contents of ``forecasts.csv``.

    Returns:
        A frame of ``year``, ``commitments_billions`` and ``kind`` where ``kind``
        is ``"Historical"`` or ``"Forecast"``.
    """
    history, projection = split_history_forecast(forecasts)
    frames: list[pd.DataFrame] = []

    if not history.empty and "commitments_billions" in history.columns:
        actual = history[["year", "commitments_billions"]].dropna()
        frames.append(actual.assign(kind="Historical"))

    if not projection.empty and "base_commitment_billions" in projection.columns:
        predicted = projection[["year", "base_commitment_billions"]].dropna()
        predicted = predicted.rename(columns={"base_commitment_billions": "commitments_billions"})
        frames.append(predicted.assign(kind="Forecast"))

    if not frames:
        return pd.DataFrame(columns=["year", "commitments_billions", "kind"])
    return pd.concat(frames, ignore_index=True).sort_values("year").reset_index(drop=True)


def outstanding_series(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Build a single outstanding-balance series spanning history and forecast.

    Args:
        forecasts: Contents of ``forecasts.csv``.

    Returns:
        A frame of ``year``, ``outstanding_billions`` and ``kind``.
    """
    history, projection = split_history_forecast(forecasts)
    frames: list[pd.DataFrame] = []

    if not history.empty and "outstanding_billions" in history.columns:
        actual = history[["year", "outstanding_billions"]].dropna()
        frames.append(actual.assign(kind="Historical"))

    if not projection.empty and "base_outstanding_billions" in projection.columns:
        predicted = projection[["year", "base_outstanding_billions"]].dropna()
        predicted = predicted.rename(columns={"base_outstanding_billions": "outstanding_billions"})
        frames.append(predicted.assign(kind="Forecast"))

    if not frames:
        return pd.DataFrame(columns=["year", "outstanding_billions", "kind"])
    return pd.concat(frames, ignore_index=True).sort_values("year").reset_index(drop=True)


def scenario_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Extract the base, best and worst case commitment scenarios.

    Args:
        forecasts: Contents of ``forecasts.csv``.

    Returns:
        A long frame of ``year``, ``scenario`` and ``commitments_billions``.
    """
    _, projection = split_history_forecast(forecasts)
    available = [column for column in _SCENARIO_COLUMNS if column in projection.columns]
    if projection.empty or not available:
        return pd.DataFrame(columns=["year", "scenario", "commitments_billions"])

    labels = {
        "base_case_commitments": "Base case",
        "best_case_commitments": "Best case (+10% YoY)",
        "worst_case_commitments": "Worst case (-15% YoY)",
    }
    long = projection[["year"] + available].melt(
        id_vars="year", var_name="scenario", value_name="commitments_billions"
    )
    long["scenario"] = long["scenario"].map(labels).fillna(long["scenario"])
    return long.dropna(subset=["commitments_billions"]).reset_index(drop=True)


def forecast_kpis(forecasts: pd.DataFrame) -> dict[str, float | int]:
    """Summarise the forecast horizon and its endpoints.

    Args:
        forecasts: Contents of ``forecasts.csv``.

    Returns:
        A mapping with the horizon length, first and last forecast years, and the
        projected commitment and outstanding values for the final year.
    """
    _, projection = split_history_forecast(forecasts)
    if projection.empty:
        return {
            "horizon_years": 0,
            "first_year": 0,
            "final_year": 0,
            "final_commitments": 0.0,
            "final_outstanding": 0.0,
        }

    final = projection.iloc[-1]
    return {
        "horizon_years": int(len(projection)),
        "first_year": int(projection["year"].iloc[0]),
        "final_year": int(final["year"]),
        "final_commitments": float(final.get("base_commitment_billions", float("nan")) or 0.0),
        "final_outstanding": float(final.get("base_outstanding_billions", float("nan")) or 0.0),
    }
