from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    from lifelines import CoxPHFitter
except Exception:  # pragma: no cover
    CoxPHFitter = None

try:
    import umap
except Exception:  # pragma: no cover
    umap = None


def _prepare_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    for date_col in ['Board Approval Date', 'Agreement Signing Date', 'Effective Date (Most Recent)', 'Last Disbursement Date']:
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

    if 'Board Approval Date' in data.columns:
        data['loan_age_years'] = (
            pd.Timestamp.now() - pd.to_datetime(data['Board Approval Date'], errors='coerce')
        ).dt.days / 365.25
    elif 'loan_age_years' not in data.columns:
        data['loan_age_years'] = 0.0

    if 'Disbursed Amount (US$)' in data.columns:
        data['Disbursed Amount (US$)'] = pd.to_numeric(data['Disbursed Amount (US$)'], errors='coerce').fillna(0.0)
    if 'Due to IBRD (US$)' in data.columns:
        data['Due to IBRD (US$)'] = pd.to_numeric(data['Due to IBRD (US$)'], errors='coerce').fillna(0.0)
    if 'Original Principal Amount (US$)' in data.columns:
        data['Original Principal Amount (US$)'] = pd.to_numeric(data['Original Principal Amount (US$)'], errors='coerce').fillna(0.0)
    if 'Cancelled Amount (US$)' in data.columns:
        data['Cancelled Amount (US$)'] = pd.to_numeric(data['Cancelled Amount (US$)'], errors='coerce').fillna(0.0)
    if 'Repaid to IBRD (US$)' in data.columns:
        data['Repaid to IBRD (US$)'] = pd.to_numeric(data['Repaid to IBRD (US$)'], errors='coerce').fillna(0.0)

    original = data.get('Original Principal Amount (US$)', pd.Series(0.0, index=data.index)).fillna(0.0)
    disbursed = data.get('Disbursed Amount (US$)', pd.Series(0.0, index=data.index)).fillna(0.0)
    repaid = data.get('Repaid to IBRD (US$)', pd.Series(0.0, index=data.index)).fillna(0.0)
    data['repayment_ratio'] = np.divide(
        repaid,
        disbursed.replace(0, np.nan),
        out=np.zeros(len(data), dtype=float),
        where=disbursed.ne(0),
    )
    data['repayment_ratio'] = data['repayment_ratio'].clip(0, 1)
    data['disbursed_percent'] = np.divide(
        disbursed,
        original.replace(0, np.nan),
        out=np.zeros(len(data), dtype=float),
        where=original.ne(0),
    ) * 100
    data['risk_score'] = np.clip(1 - data['repayment_ratio'], 0, 1)

    status_values = data.get('Loan Status', pd.Series('', index=data.index)).fillna('')
    data['is_active'] = status_values.isin(['Disbursing', 'Disbursing&Repaying', 'Repaying', 'Effective'])
    data['is_cancelled'] = status_values.isin(['Cancelled', 'Fully Cancelled'])
    data['is_fully_repaid'] = status_values.eq('Fully Repaid')

    def _flag_high_risk(row: pd.Series) -> int:
        status = str(row.get('Loan Status', '')).strip()
        if status in {'Cancelled', 'Fully Cancelled', 'At-Risk', 'At Risk', 'Delayed', 'Late'}:
            return 1
        if status == 'Fully Repaid':
            return 0
        if row.get('risk_score', 0.0) >= 0.7:
            return 1
        if row.get('repayment_ratio', 0.0) <= 0.35:
            return 1
        if row.get('Due to IBRD (US$)', 0.0) > 0.6 * row.get('Original Principal Amount (US$)', 0.0):
            return 1
        return 0

    data['high_risk'] = data.apply(_flag_high_risk, axis=1).astype(int)
    return data


def _build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = _prepare_loan_features(df)
    feature_cols = [
        'Original Principal Amount (US$)',
        'Disbursed Amount (US$)',
        'Due to IBRD (US$)',
        'Cancelled Amount (US$)',
        'Repaid to IBRD (US$)',
        'loan_age_years',
        'repayment_ratio',
        'disbursed_percent',
        'risk_score',
        'Region',
        'Loan Type',
        'Country / Economy',
    ]
    model_df = data[feature_cols + ['high_risk', 'is_active', 'Loan Number']].copy()
    return model_df


def train_default_watchlist_model(df: pd.DataFrame, threshold: float = 0.7, test_size: float = 0.25, random_state: int = 42):
    model_df = _build_model_frame(df)
    feature_cols = [
        'Original Principal Amount (US$)',
        'Disbursed Amount (US$)',
        'Due to IBRD (US$)',
        'Cancelled Amount (US$)',
        'Repaid to IBRD (US$)',
        'loan_age_years',
        'repayment_ratio',
        'disbursed_percent',
        'risk_score',
        'Region',
        'Loan Type',
        'Country / Economy',
    ]
    data = model_df.dropna(subset=feature_cols + ['high_risk'])
    if data.empty:
        raise ValueError('No rows available to train the default watchlist model.')

    X = data[feature_cols]
    y = data['high_risk'].astype(int)

    numeric_cols = [
        'Original Principal Amount (US$)',
        'Disbursed Amount (US$)',
        'Due to IBRD (US$)',
        'Cancelled Amount (US$)',
        'Repaid to IBRD (US$)',
        'loan_age_years',
        'repayment_ratio',
        'disbursed_percent',
        'risk_score',
    ]
    categorical_cols = ['Region', 'Loan Type', 'Country / Economy']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_cols),
        ],
        remainder='drop',
    )

    if xgb is not None:
        estimator = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        estimator = RandomForestClassifier(n_estimators=300, random_state=random_state)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', estimator),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    pipeline.fit(X_train, y_train)
    test_predictions = pipeline.predict_proba(X_test)[:, 1]
    test_labels = y_test.to_numpy()

    metrics = {
        'accuracy': float(accuracy_score(y_test, pipeline.predict(X_test))),
        'roc_auc': float(roc_auc_score(y_test, test_predictions)),
        'average_precision': float(average_precision_score(y_test, test_predictions)),
        'precision': float(precision_score(y_test, pipeline.predict(X_test), zero_division=0)),
        'recall': float(recall_score(y_test, pipeline.predict(X_test), zero_division=0)),
        'f1': float(f1_score(y_test, pipeline.predict(X_test), zero_division=0)),
    }

    active_loans = data[data['is_active']].copy()
    if active_loans.empty:
        watchlist = pd.DataFrame(columns=['loan_number', 'predicted_probability', 'risk_band'])
    else:
        active_loans['predicted_probability'] = pipeline.predict_proba(active_loans[feature_cols])[:, 1]
        active_loans['risk_band'] = np.where(active_loans['predicted_probability'] >= threshold, 'High', 'Moderate')
        watchlist = active_loans[['Loan Number', 'predicted_probability', 'risk_band']].rename(columns={'Loan Number': 'loan_number'}).copy()
        watchlist = watchlist.sort_values('predicted_probability', ascending=False)
        watchlist = watchlist[watchlist['predicted_probability'] >= threshold].reset_index(drop=True)

    feature_importance = []
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        transformed_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = pipeline.named_steps['model'].feature_importances_
        feature_importance = [
            {'feature': name, 'importance': float(score)}
            for name, score in zip(transformed_names, importances)
        ]
        feature_importance = sorted(feature_importance, key=lambda item: item['importance'], reverse=True)[:10]

    return {
        'model': pipeline,
        'metrics': metrics,
        'watchlist': watchlist,
        'feature_importance': feature_importance,
    }


def detect_anomalies(df: pd.DataFrame, n_outliers: int = 10) -> pd.DataFrame:
    data = _prepare_loan_features(df).copy()
    numeric_cols = [
        'Original Principal Amount (US$)',
        'Disbursed Amount (US$)',
        'Due to IBRD (US$)',
        'Cancelled Amount (US$)',
        'Repaid to IBRD (US$)',
        'loan_age_years',
        'repayment_ratio',
        'disbursed_percent',
        'risk_score',
    ]
    X = data[numeric_cols].fillna(0).astype(float)
    if X.empty:
        return data.head(0).copy()

    contamination = min(max(n_outliers / len(X), 0.01), 0.2)
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=300)
    labels = model.fit_predict(X)
    scores = -model.score_samples(X)
    anomalies = data.copy()
    anomalies['anomaly_score'] = scores
    anomalies['is_anomaly'] = labels == -1
    anomalies = anomalies[anomalies['is_anomaly']].sort_values('anomaly_score', ascending=False)
    return anomalies.head(n_outliers).reset_index(drop=True)


def estimate_repayment_survival(df: pd.DataFrame):
    data = _prepare_loan_features(df).copy()
    data['time_to_repayment'] = data['loan_age_years'].fillna(0.0)
    data['event_observed'] = data['is_fully_repaid'].astype(int)
    data['expected_repayment_years'] = data['loan_age_years'] + np.clip((1 - data['repayment_ratio']) * 5.0, 0, 5.0) + 1.0
    data['hazard_score'] = np.clip((1 - data['repayment_ratio']) * (1 + data['loan_age_years'].fillna(0) / 10.0), 0, 1)

    if CoxPHFitter is not None and len(data) > 10:
        candidate_cols = [
            'Original Principal Amount (US$)',
            'Disbursed Amount (US$)',
            'Due to IBRD (US$)',
            'Cancelled Amount (US$)',
            'loan_age_years',
            'repayment_ratio',
            'disbursed_percent',
            'risk_score',
            'Region',
            'Loan Type',
        ]
        survival_df = data[candidate_cols + ['time_to_repayment', 'event_observed']].copy()
        survival_df = pd.get_dummies(survival_df, columns=['Region', 'Loan Type'], drop_first=True)
        survival_df['time'] = survival_df['time_to_repayment'].fillna(0.0)
        survival_df['event'] = survival_df['event_observed'].astype(int)
        feature_cols = [col for col in survival_df.columns if col not in {'time', 'event'}]

        try:
            model = CoxPHFitter()
            model.fit(survival_df[['time', 'event'] + feature_cols], duration_col='time', event_col='event')
            hazard_predictions = model.predict_partial_hazard(survival_df[feature_cols], risk_matrix=None)
            data['hazard_score'] = np.clip(hazard_predictions.to_numpy().ravel() / max(hazard_predictions.max(), 1e-6), 0, 1)
            return {'model': model, 'survival_summary': data[['Loan Number', 'Loan Status', 'loan_age_years', 'repayment_ratio', 'hazard_score', 'expected_repayment_years', 'event_observed']].copy()}
        except Exception:
            pass

    return {'model': None, 'survival_summary': data[['Loan Number', 'Loan Status', 'loan_age_years', 'repayment_ratio', 'hazard_score', 'expected_repayment_years', 'event_observed']].copy()}


def forecast_portfolio_trends(df: pd.DataFrame, years_ahead: int = 5):
    data = _prepare_loan_features(df).copy()
    if 'Board Approval Date' in data.columns:
        data['approval_year'] = pd.to_datetime(data['Board Approval Date'], errors='coerce').dt.year
    elif 'approval_year' in data.columns:
        data['approval_year'] = pd.to_numeric(data['approval_year'], errors='coerce')
    else:
        data['approval_year'] = 2000

    yearly = (
        data.groupby('approval_year', as_index=False)
        .agg(
            total_commitment=('Original Principal Amount (US$)', 'sum'),
            total_outstanding=('Due to IBRD (US$)', 'sum'),
        )
    )
    yearly = yearly.sort_values('approval_year').reset_index(drop=True)
    if yearly.empty:
        return pd.DataFrame(columns=['Year', 'Forecasted Outstanding'])

    x = yearly['approval_year'].to_numpy(dtype=float)
    y = yearly['total_outstanding'].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    last_year = int(yearly['approval_year'].max())
    future_years = pd.RangeIndex(start=last_year + 1, stop=last_year + 1 + years_ahead, step=1)
    predictions = slope * future_years.to_numpy(dtype=float) + intercept
    forecast = pd.DataFrame({
        'Year': future_years,
        'Forecasted Outstanding': predictions,
    })
    return forecast


def build_country_umap(df: pd.DataFrame):
    data = _prepare_loan_features(df)
    country_summary = (
        data.groupby('Country / Economy', as_index=False)
        .agg(
            total_commitment=('Original Principal Amount (US$)', 'sum'),
            total_outstanding=('Due to IBRD (US$)', 'sum'),
            avg_repayment_ratio=('repayment_ratio', 'mean'),
            avg_risk=('risk_score', 'mean'),
            loan_count=('Loan Number', 'count'),
        )
        .fillna(0)
    )
    feature_cols = ['total_commitment', 'total_outstanding', 'avg_repayment_ratio', 'avg_risk', 'loan_count']
    X = country_summary[feature_cols].astype(float)
    if len(X) < 2:
        return country_summary.assign(umap_x=0.0, umap_y=0.0)

    if umap is not None:
        mapper = umap.UMAP(n_neighbors=min(15, max(2, len(X) - 1)), min_dist=0.1, random_state=42)
        embedding = mapper.fit_transform(X)
    else:
        from sklearn.decomposition import PCA

        embedding = PCA(n_components=2, random_state=42).fit_transform(X)

    country_summary['umap_x'] = embedding[:, 0]
    country_summary['umap_y'] = embedding[:, 1]
    return country_summary
