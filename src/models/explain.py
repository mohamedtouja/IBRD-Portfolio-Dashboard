"""SHAP explanations for the repayment pipeline.

SHAP is treated as an optional dependency throughout: every entry point returns
``None`` when the package is absent or the explainer fails, so the dashboard can
skip the explanation panel instead of erroring.

Contributions are computed on the *transformed* feature space, so names carry the
``ColumnTransformer`` prefixes (``num__``, ``cat__``). :func:`prettify_feature_name`
strips them for display.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def shap_available() -> bool:
    """Report whether the SHAP package can be imported.

    Returns:
        ``True`` when SHAP is installed and importable.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


def prettify_feature_name(name: str) -> str:
    """Strip ``ColumnTransformer`` prefixes from a transformed feature name.

    Args:
        name: A transformed feature name such as ``"cat__Loan Type_FSL"``.

    Returns:
        A display-friendly name such as ``"Loan Type_FSL"``.
    """
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _transformed_matrix(pipeline: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]] | None:
    """Run a frame through the pipeline's preprocessor.

    Args:
        pipeline: The fitted repayment pipeline.
        frame: Rows to transform.

    Returns:
        The transformed matrix and its feature names, or ``None`` on failure.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        matrix = preprocessor.transform(frame)
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except (AttributeError, KeyError, ValueError):
        logger.exception("Could not transform features for explanation")
        return None

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix), names


def top_contributions(
    pipeline: Any,
    frame: pd.DataFrame,
    top_n: int = 3,
) -> list[dict[str, Any]] | None:
    """Compute the largest SHAP contributions for a single scored row.

    Args:
        pipeline: The fitted repayment pipeline.
        frame: A one-row frame already ordered to the pipeline's feature list.
        top_n: Number of contributions to return.

    Returns:
        A list of ``{"feature", "contribution", "direction"}`` mappings ordered by
        absolute contribution, or ``None`` when SHAP is unavailable or fails.
    """
    if frame.empty:
        return None

    try:
        import shap
    except ImportError:
        logger.info("SHAP not installed; skipping explanation")
        return None

    transformed = _transformed_matrix(pipeline, frame)
    if transformed is None:
        return None
    matrix, names = transformed

    try:
        model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(matrix))
    except (AttributeError, KeyError, ValueError, TypeError, IndexError):
        logger.exception("SHAP explanation failed")
        return None

    # Binary classifiers may return (n, features) or (n, features, classes).
    if values.ndim == 3:
        values = values[..., -1]
    row = values[0].ravel()
    if row.size != len(names):
        logger.error("SHAP value count %d does not match %d feature names", row.size, len(names))
        return None

    order = np.argsort(-np.abs(row))[: max(1, top_n)]
    return [
        {
            "feature": prettify_feature_name(names[index]),
            "contribution": float(row[index]),
            "direction": "increases" if row[index] > 0 else "decreases",
        }
        for index in order
    ]


def global_importance(pipeline: Any, top_n: int = 15) -> pd.DataFrame:
    """Read gain-based feature importances from the fitted booster.

    This needs no background data and is therefore always available, unlike SHAP.

    Args:
        pipeline: The fitted repayment pipeline.
        top_n: Number of features to return.

    Returns:
        A frame of ``feature`` and ``importance`` ordered descending, or an empty
        frame when importances cannot be read.
    """
    transformed_names: list[str] = []
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
        importances = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    except (AttributeError, KeyError, ValueError):
        logger.exception("Could not read model feature importances")
        return pd.DataFrame(columns=["feature", "importance"])

    if len(importances) != len(transformed_names):
        logger.error("Importance count does not match feature name count")
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame(
        {
            "feature": [prettify_feature_name(name) for name in transformed_names],
            "importance": importances,
        }
    )
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def sample_shap_summary(
    pipeline: Any,
    loans: pd.DataFrame,
    feature_names: list[str],
    sample_size: int = 300,
    top_n: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values across a portfolio sample.

    Args:
        pipeline: The fitted repayment pipeline.
        loans: Loan-level frame containing the model's feature columns.
        feature_names: Ordered feature columns the pipeline expects.
        sample_size: Maximum number of rows to explain.
        top_n: Number of features to return.
        random_state: Seed for the row sample.

    Returns:
        A frame of ``feature`` and ``mean_abs_shap`` ordered descending, or an
        empty frame when SHAP is unavailable or fails.
    """
    if loans.empty or not set(feature_names).issubset(loans.columns):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    try:
        import shap
    except ImportError:
        logger.info("SHAP not installed; skipping summary")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    sample = loans[feature_names]
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=random_state)

    transformed = _transformed_matrix(pipeline, sample)
    if transformed is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    matrix, names = transformed

    try:
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        values = np.asarray(explainer.shap_values(matrix))
    except (AttributeError, KeyError, ValueError, TypeError, IndexError):
        logger.exception("SHAP summary failed")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    if values.ndim == 3:
        values = values[..., -1]
    mean_abs = np.abs(values).mean(axis=0).ravel()
    if mean_abs.size != len(names):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    frame = pd.DataFrame(
        {
            "feature": [prettify_feature_name(name) for name in names],
            "mean_abs_shap": mean_abs,
        }
    )
    return frame.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
