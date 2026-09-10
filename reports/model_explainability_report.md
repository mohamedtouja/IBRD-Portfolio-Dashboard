# Model Explainability Report

This report accompanies `notebooks/09_model_explainability.ipynb` and summarizes model cards, key explanations, and business recommendations derived from the trained models in `models_pickle/` and the cleaned dataset `data/processed/ibrd_clean.csv`.

## Data snapshot

- Data file: data/processed/ibrd_clean.csv
- Rows: TBD (computed in notebook)
- Approval Date range: TBD (computed in notebook)

## Model Cards

### repayment_xgboost

- Purpose: Predict loan repayment likelihood.
- Training data rows: TBD
- Training date range: TBD
- Features used: TBD
- Performance metrics: TBD
- Limitations: Model may not generalize to sudden macro shocks.
- Ethical considerations: Avoid automated exclusion based purely on model output.
- Intended use: Portfolio monitoring and prioritization.

### country_kmeans

- Purpose: Cluster countries for engagement strategy segmentation.
- Training data rows: TBD
- Features used: Commitments, repayments, repayment ratios.
- Limitations: KMeans clusters are sensitive to feature scaling.
- Intended use: Inform regional engagement and monitoring approaches.

### anomaly_isolation_forest

- Purpose: Detect unusual loan records for further review.
- Training data rows: TBD
- Features used: Monetary and timing fields.
- Limitations: IsolationForest can flag rare-but-valid cases as anomalies.
- Intended use: Generate watchlist for compliance and audit teams.

## Business Recommendations (top 10, condensed)

1. Increase monitoring for clusters with low repayment_ratio (see cluster_summary).
2. Prioritize loan reviews for top 5 anomalous loans detected by IsolationForest.
3. Use SHAP global importances to reduce feature set and simplify audits.
4. Flag Region-level patterns where models signal elevated default risk for deeper review.
5. Incorporate cluster membership into country engagement strategies.
6. Automate monthly dashboard updates of top risk loans for portfolio managers.
7. Run counterfactual checks on high-impact features before enforcement actions.
8. Calibrate monitoring thresholds per-cluster to reduce false positives.
9. Use local SHAP explanations for any loan escalated to compliance for transparency.
10. Schedule periodic re-training when portfolio composition shifts >10% year-over-year.

## Executive Summary

(Executive summary content will be populated when the notebook runs and computes portfolio-level metrics.)
