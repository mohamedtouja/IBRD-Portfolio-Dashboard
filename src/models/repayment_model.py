from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_repayment_model(df: pd.DataFrame):
    model_df = df.copy()
    features = [
        'Original Principal Amount (US$)',
        'Disbursed Amount (US$)',
        'Due to IBRD (US$)',
        'loan_age_years',
        'repayment_ratio',
        'disbursed_percent',
        'Region',
        'Loan Type',
    ]

    model_df = model_df[features + ['is_active']].dropna()
    model_df = pd.get_dummies(model_df, columns=['Region', 'Loan Type'], drop_first=True)

    X = model_df.drop(columns=['is_active'])
    y = model_df['is_active'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, predictions),
        'classification_report': classification_report(y_test, predictions, output_dict=True),
    }
    return model, results
