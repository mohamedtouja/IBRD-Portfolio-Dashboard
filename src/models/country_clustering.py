from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def country_risk_clustering(df: pd.DataFrame):
    country_summary = (
        df.groupby('Country / Economy', as_index=False)
        .agg(
            total_commitment=('Original Principal Amount (US$)', 'sum'),
            total_outstanding=('Due to IBRD (US$)', 'sum'),
            avg_repayment_ratio=('repayment_ratio', 'mean'),
            loan_count=('Loan Number', 'count'),
        )
    )

    features = ['total_commitment', 'total_outstanding', 'avg_repayment_ratio', 'loan_count']
    X = country_summary[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(3, max(1, len(X_scaled) - 1))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    country_summary['cluster'] = labels
    return country_summary, model
