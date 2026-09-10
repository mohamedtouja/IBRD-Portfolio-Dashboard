"""Display formatters for currency, percentages and counts.

Pure string helpers with no framework dependency. Every function tolerates
``None`` and non-finite input so that a single missing aggregate never breaks a
rendered layout.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_MISSING = "n/a"


def _coerce(value: Any) -> float | None:
    """Convert a value to a finite float.

    Args:
        value: Any value that might represent a number.

    Returns:
        The finite float value, or ``None`` when the input is missing,
        non-numeric or non-finite.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.debug("Could not coerce %r to float", value)
        return None
    return number if math.isfinite(number) else None


def format_currency(value: Any, decimals: int = 0) -> str:
    """Format a value as a full-precision US dollar amount.

    Args:
        value: Numeric value in US dollars.
        decimals: Number of decimal places to render.

    Returns:
        A string such as ``"$1,234,567"``, or ``"n/a"`` when the value is
        missing or non-finite.
    """
    number = _coerce(value)
    if number is None:
        return _MISSING
    return f"${number:,.{decimals}f}"


def format_compact_currency(value: Any) -> str:
    """Format a value as an abbreviated US dollar amount.

    Args:
        value: Numeric value in US dollars.

    Returns:
        A string such as ``"$1.2B"`` or ``"-$45.0M"``, or ``"n/a"`` when the
        value is missing or non-finite.
    """
    number = _coerce(value)
    if number is None:
        return _MISSING

    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{sign}${magnitude / threshold:,.1f}{suffix}"
    return f"{sign}${magnitude:,.0f}"


def format_percent(value: Any, decimals: int = 1, already_scaled: bool = True) -> str:
    """Format a value as a percentage.

    Args:
        value: The percentage value.
        decimals: Number of decimal places to render.
        already_scaled: ``True`` when the input is already expressed on a 0-100
            scale; ``False`` when it is a 0-1 ratio that needs multiplying.

    Returns:
        A string such as ``"82.4%"``, or ``"n/a"`` when the value is missing or
        non-finite.
    """
    number = _coerce(value)
    if number is None:
        return _MISSING
    if not already_scaled:
        number *= 100.0
    return f"{number:,.{decimals}f}%"


def format_number(value: Any, decimals: int = 0) -> str:
    """Format a value as a thousands-separated number.

    Args:
        value: Numeric value to render.
        decimals: Number of decimal places to render.

    Returns:
        A string such as ``"9,518"``, or ``"n/a"`` when the value is missing or
        non-finite.
    """
    number = _coerce(value)
    if number is None:
        return _MISSING
    return f"{number:,.{decimals}f}"
