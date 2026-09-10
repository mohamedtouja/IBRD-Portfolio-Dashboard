import pandas as pd

from src.models.ml_risk import detect_anomalies, train_default_watchlist_model


def make_demo_df():
    rows = []
    for i in range(60):
        status = 'Disbursing' if i % 3 == 0 else 'Repaying' if i % 3 == 1 else 'Cancelled'
        amount = 1_000_000 + i * 50_000
        disbursed = amount * (0.7 if i % 3 == 0 else 0.9)
        due = amount * (0.5 if i % 3 == 2 else 0.2)
        rows.append(
            {
                'Loan Number': f'LOAN-{i:03d}',
                'Country / Economy': 'Country A' if i < 20 else 'Country B',
                'Region': 'Africa' if i < 20 else 'Asia',
                'Loan Type': 'Infrastructure' if i % 2 == 0 else 'Health',
                'Loan Status': status,
                'Board Approval Date': pd.Timestamp('2010-01-01') + pd.Timedelta(days=i * 30),
                'Original Principal Amount (US$)': amount,
                'Cancelled Amount (US$)': amount * 0.6 if status == 'Cancelled' else 0.0,
                'Disbursed Amount (US$)': disbursed,
                'Repaid to IBRD (US$)': disbursed * (0.15 if i % 3 == 0 else 0.85 if i % 3 == 1 else 0.0),
                'Due to IBRD (US$)': due,
                'Effective Date (Most Recent)': pd.Timestamp('2015-01-01') + pd.Timedelta(days=i * 10),
                'Last Disbursement Date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i * 5),
            }
        )
    return pd.DataFrame(rows)


def test_train_default_watchlist_model_returns_watchlist():
    df = make_demo_df()

    result = train_default_watchlist_model(df, threshold=0.55)

    assert 'model' in result
    assert 'metrics' in result
    assert 'watchlist' in result
    assert not result['watchlist'].empty
    assert {'loan_number', 'predicted_probability', 'risk_band'}.issubset(result['watchlist'].columns)
    assert result['watchlist']['predicted_probability'].between(0, 1).all()


def test_detect_anomalies_returns_unusual_loans():
    df = make_demo_df()
    df.loc[0, 'Due to IBRD (US$)'] = 10_000_000_000
    df.loc[0, 'Original Principal Amount (US$)'] = 20_000_000_000

    anomalies = detect_anomalies(df, n_outliers=5)

    assert not anomalies.empty
    assert 'anomaly_score' in anomalies.columns
    assert 'Loan Number' in anomalies.columns
