"""Inference wrapper around the trained repayment classifier.

The persisted estimator (``models_pickle/repayment_best_model.pkl``) is an
``imblearn`` pipeline of the shape::

    ColumnTransformer(passthrough[5 numeric] + OneHotEncoder[3 categorical])
      -> SMOTE
      -> XGBClassifier

Because the one-hot encoder lives *inside* the pipeline, callers must supply raw
categorical values. Encoding them by hand produces a feature-count mismatch and
the call fails.

The positive class is ``repayment_ratio > 0.95`` (notebook 05), so the predicted
probability reads as "likelihood this loan is at least 95% repaid" rather than a
forward-looking default probability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def derive_loan_size_category(principal: float) -> str:
    """Bucket a principal amount using notebook 02's bin edges.

    Deriving this rather than accepting it as user input guarantees the category
    can never contradict the amount that sits alongside it in the feature row.

    Bins are left-closed, matching notebook 02's ``pd.cut(..., right=False)``.
    The distinction is not academic: principals land on the round bin edges
    often enough that using closed-right bins reclassifies about 6% of the
    portfolio.

    Args:
        principal: Original principal amount in US dollars.

    Returns:
        One of ``"Small"``, ``"Medium"``, ``"Large"`` or ``"Mega"``.
    """
    edges = config.LOAN_SIZE_BINS
    labels = config.LOAN_SIZE_LABELS
    for label, upper in zip(labels, edges[1:]):
        if principal < upper:
            return label
    return labels[-1]


def risk_band(probability: float) -> str:
    """Map a repayment probability onto a risk band.

    Args:
        probability: Predicted probability that the loan is >95% repaid.

    Returns:
        ``"Low"``, ``"Medium"`` or ``"High"`` risk. A high repayment probability
        maps to low risk.
    """
    if probability >= config.RISK_BAND_LOW_MIN:
        return "Low"
    if probability >= config.RISK_BAND_MEDIUM_MIN:
        return "Medium"
    return "High"


@dataclass(frozen=True)
class RepaymentPrediction:
    """A single scored loan.

    Attributes:
        probability: Predicted probability of being >95% repaid.
        risk_band: Discretised risk label derived from ``probability``.
        features: The exact feature row submitted to the model.
    """

    probability: float
    risk_band: str
    features: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Flatten the prediction into a single serialisable mapping.

        Returns:
            A mapping of feature values plus the prediction outputs, suitable for
            CSV or JSON export.
        """
        record: dict[str, Any] = dict(self.features)
        record["repayment_probability"] = round(self.probability, 6)
        record["repayment_probability_pct"] = round(self.probability * 100.0, 2)
        record["risk_band"] = self.risk_band
        return record


class RepaymentPredictor:
    """Loads the repayment pipeline and scores loans against it.

    A class is used rather than free functions because three pieces of state must
    stay consistent for the lifetime of a prediction: the fitted pipeline, the
    ordered feature list saved alongside it, and the category vocabularies read
    back out of the fitted one-hot encoder. Binding them together lets the caller
    populate a form from ``categories()`` and be certain the resulting row
    satisfies the pipeline's contract.
    """

    def __init__(self, pipeline: Any, feature_names: list[str]) -> None:
        """Initialise the predictor.

        Args:
            pipeline: The fitted estimator exposing ``predict_proba``.
            feature_names: Ordered feature column names the pipeline expects.
        """
        self._pipeline = pipeline
        self._feature_names = list(feature_names)

    @classmethod
    def load(
        cls,
        model_path: Path | None = None,
        features_path: Path | None = None,
    ) -> "RepaymentPredictor | None":
        """Load the pipeline and its feature list from disk.

        Args:
            model_path: Override for the model location. Defaults to
                ``config.REPAYMENT_MODEL_PATH``.
            features_path: Override for the feature-list location. Defaults to
                ``config.REPAYMENT_FEATURES_PATH``.

        Returns:
            A ready predictor, or ``None`` when either artifact is missing or
            cannot be deserialised.
        """
        model_path = model_path or config.REPAYMENT_MODEL_PATH
        features_path = features_path or config.REPAYMENT_FEATURES_PATH

        if not model_path.exists() or not features_path.exists():
            logger.warning(
                "Repayment artifacts missing (model=%s, features=%s)",
                model_path.exists(),
                features_path.exists(),
            )
            return None

        try:
            import joblib

            pipeline = joblib.load(model_path)
            feature_names = joblib.load(features_path)
        except Exception:  # noqa: BLE001
            # Deliberately broad: unpickling arbitrary binary can raise almost
            # anything -- EOFError and UnpicklingError on a truncated or
            # half-written file, ModuleNotFoundError when imbalanced-learn or
            # xgboost is absent from the serving environment. The caller only
            # needs to know the model is unusable, and the traceback is logged.
            logger.exception("Could not deserialise the repayment model")
            return None

        logger.info("Loaded repayment pipeline with %d features", len(feature_names))
        return cls(pipeline, list(feature_names))

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature columns the pipeline expects.

        Returns:
            A copy of the feature name list.
        """
        return list(self._feature_names)

    @property
    def pipeline(self) -> Any:
        """The underlying fitted pipeline.

        Returns:
            The estimator, exposed so the explainability layer can reach the
            preprocessor and booster separately.
        """
        return self._pipeline

    def categories(self, column: str) -> list[str]:
        """Return the category vocabulary the encoder learned for a column.

        Populating form widgets from this guarantees no unseen category reaches
        the model, which matters because the encoder was fitted with
        ``handle_unknown="ignore"`` and would silently emit an all-zero block for
        anything unrecognised.

        Args:
            column: Name of a categorical feature.

        Returns:
            Sorted category values, or an empty list when they cannot be read.
        """
        try:
            encoder = self._pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            for name, values in zip(config.REPAYMENT_CATEGORICAL_FEATURES, encoder.categories_):
                if name == column:
                    return [str(value) for value in values]
        except (AttributeError, KeyError):
            logger.exception("Could not read encoder categories for %s", column)
        return []

    def build_feature_frame(
        self,
        loan_age_years: float,
        principal: float,
        disbursed: float,
        cancelled_percent: float,
        undisbursed_percent: float,
        region: str,
        loan_type: str,
        loan_size_category: str | None = None,
    ) -> pd.DataFrame:
        """Assemble a single-row frame in the exact order the pipeline expects.

        Args:
            loan_age_years: Age of the loan in years.
            principal: Original principal amount in US dollars.
            disbursed: Disbursed amount in US dollars.
            cancelled_percent: Cancelled share of principal, on a 0-100 scale.
            undisbursed_percent: Undisbursed share of principal, on a 0-100 scale.
            region: Region name, from :meth:`categories`.
            loan_type: Loan type code, from :meth:`categories`.
            loan_size_category: Size bucket. Derived from ``principal`` when
                omitted.

        Returns:
            A one-row frame whose columns match ``feature_names``.
        """
        values: dict[str, Any] = {
            "loan_age_years": float(loan_age_years),
            "Original Principal Amount (US$)": float(principal),
            "Disbursed Amount (US$)": float(disbursed),
            "cancelled_percent": float(cancelled_percent),
            "undisbursed_percent": float(undisbursed_percent),
            "Region": region,
            "Loan Type": loan_type,
            "loan_size_category": loan_size_category or derive_loan_size_category(principal),
        }
        return pd.DataFrame([[values[name] for name in self._feature_names]], columns=self._feature_names)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Score a frame of loans.

        Args:
            frame: Rows containing at least every column in ``feature_names``.

        Returns:
            A 1-D array of positive-class probabilities. Returns an array of
            ``nan`` the same length as ``frame`` when scoring fails.
        """
        missing = [name for name in self._feature_names if name not in frame.columns]
        if missing:
            logger.error("Cannot score: missing feature columns %s", missing)
            return np.full(len(frame), np.nan)

        ordered = frame[self._feature_names]
        # XGBoost rejects infinities; the pipeline passes numerics through
        # untouched, so they are neutralised here instead.
        numeric = list(config.REPAYMENT_NUMERIC_FEATURES)
        ordered = ordered.copy()
        ordered[numeric] = (
            ordered[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        )

        try:
            return self._pipeline.predict_proba(ordered)[:, 1]
        except (ValueError, KeyError, AttributeError):
            logger.exception("Repayment scoring failed")
            return np.full(len(frame), np.nan)

    def predict_one(self, **kwargs: Any) -> RepaymentPrediction | None:
        """Build a feature row and score it in one step.

        Args:
            **kwargs: Forwarded verbatim to :meth:`build_feature_frame`.

        Returns:
            The scored prediction, or ``None`` when scoring failed.
        """
        frame = self.build_feature_frame(**kwargs)
        probabilities = self.predict_proba(frame)
        probability = float(probabilities[0])
        if not np.isfinite(probability):
            return None
        return RepaymentPrediction(
            probability=probability,
            risk_band=risk_band(probability),
            features=frame.iloc[0].to_dict(),
        )

    def score_portfolio(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Score every loan in a portfolio frame.

        Args:
            loans: Loan-level frame containing the model's feature columns.

        Returns:
            A copy of ``loans`` with ``repayment_probability`` and
            ``predicted_risk_band`` appended. Returns the input unchanged when
            required columns are absent.
        """
        if loans.empty:
            return loans.copy()

        scored = loans.copy()
        probabilities = self.predict_proba(scored)
        scored["repayment_probability"] = probabilities
        scored["predicted_risk_band"] = [
            risk_band(value) if np.isfinite(value) else "Unknown" for value in probabilities
        ]
        return scored
