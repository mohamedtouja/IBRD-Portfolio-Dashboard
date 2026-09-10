from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd


def _generate_synthetic_portfolio(n_rows: int = 1200) -> pd.DataFrame:
    countries = [
        'India', 'Brazil', 'Kenya', 'Indonesia', 'Morocco', 'Mexico',
        'Pakistan', 'Philippines', 'Nigeria', 'Nepal', 'Vietnam', 'Ghana',
        'Bangladesh', 'Colombia', 'Egypt', 'Tanzania', 'Peru', 'Jordan'
    ]
    regions = ['South Asia', 'Latin America', 'Africa', 'East Asia', 'Europe & Central Asia']
    loan_type = ['Investment', 'Agriculture', 'Infrastructure', 'Education', 'Health', 'Climate']
    statuses = ['Disbursing', 'Repaying', 'Effective', 'Fully Repaid', 'Cancelled', 'Disbursing&Repaying']

    rng = random.Random(42)
    rows = []
    for i in range(n_rows):
        country = rng.choice(countries)
        region = rng.choice(regions)
        amount = rng.uniform(5_000_000, 600_000_000)
        disbursed = min(amount, rng.uniform(0.4, 1.2) * amount)
        repaid = rng.uniform(0, disbursed * 0.9)
        due = max(0.0, amount * rng.uniform(0.05, 0.8))
        status = rng.choice(statuses)
        approval_date = pd.Timestamp('2000-01-01') + pd.Timedelta(days=rng.randint(0, 9000))
        loan_number = f'LOAN-{i+1000}'

        rows.append({
            'Loan Number': loan_number,
            'Country / Economy': country,
            'Region': region,
            'Loan Type': rng.choice(loan_type),
            'Loan Status': status,
            'Board Approval Date': approval_date,
            'Original Principal Amount (US$)': round(amount, 2),
            'Cancelled Amount (US$)': 0.0 if status != 'Cancelled' else round(amount * rng.uniform(0.2, 0.8), 2),
            'Disbursed Amount (US$)': round(disbursed, 2),
            'Repaid to IBRD (US$)': round(repaid, 2),
            'Due to IBRD (US$)': round(due, 2),
            'Agreement Signing Date': approval_date - pd.Timedelta(days=rng.randint(10, 200)),
            'Effective Date (Most Recent)': approval_date + pd.Timedelta(days=rng.randint(30, 400)),
            'Last Disbursement Date': approval_date + pd.Timedelta(days=rng.randint(300, 5000)),
        })

    return pd.DataFrame(rows)


def clean_ibrd_data(raw_path: str | Path | None = None, silver_path: str | Path | None = None):
    project_root = Path(__file__).resolve().parents[2]

    if raw_path is None:
        raw_dir = project_root / 'data' / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / 'ibrd_synthetic.csv'
    raw_path = Path(raw_path)

    if silver_path is None:
        silver_path = project_root / 'data' / 'processed' / 'ibrd_clean.csv'
    silver_path = Path(silver_path)
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        df = _generate_synthetic_portfolio()
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)

    if 'Board Approval Date' in df.columns:
        df['Board Approval Date'] = pd.to_datetime(df['Board Approval Date'], errors='coerce')
    if 'Agreement Signing Date' in df.columns:
        df['Agreement Signing Date'] = pd.to_datetime(df['Agreement Signing Date'], errors='coerce')
    if 'Effective Date (Most Recent)' in df.columns:
        df['Effective Date (Most Recent)'] = pd.to_datetime(df['Effective Date (Most Recent)'], errors='coerce')
    if 'Last Disbursement Date' in df.columns:
        df['Last Disbursement Date'] = pd.to_datetime(df['Last Disbursement Date'], errors='coerce')

    numeric_cols = [
        'Original Principal Amount (US$)',
        'Cancelled Amount (US$)',
        'Disbursed Amount (US$)',
        'Repaid to IBRD (US$)',
        'Due to IBRD (US$)',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df['loan_age_days'] = (pd.Timestamp.now() - df['Board Approval Date']).dt.days
    df['loan_age_years'] = df['loan_age_days'] / 365.25

    df['is_active'] = df.get('Loan Status', pd.Series('')).isin(['Disbursing', 'Disbursing&Repaying', 'Repaying', 'Effective'])
    df['is_fully_repaid'] = df.get('Loan Status', pd.Series('')) == 'Fully Repaid'
    df['is_cancelled'] = df.get('Loan Status', pd.Series('')).isin(['Fully Cancelled', 'Cancelled'])

    disbursed = df['Disbursed Amount (US$)'].fillna(0)
    repaid = df['Repaid to IBRD (US$)'].fillna(0)
    df['repayment_ratio'] = np.divide(repaid, disbursed.replace(0, np.nan), out=np.zeros(len(df), dtype=float), where=disbursed.ne(0))
    df['repayment_ratio'] = df['repayment_ratio'].clip(0, 1)

    original = df['Original Principal Amount (US$)'].fillna(0)
    df['disbursed_percent'] = np.divide(disbursed, original.replace(0, np.nan), out=np.zeros(len(df), dtype=float), where=original.ne(0)) * 100

    def categorize_loan(row):
        if row['is_cancelled']:
            return 'Cancelled'
        if row['is_fully_repaid']:
            return 'Fully Repaid'
        if row['repayment_ratio'] > 0.9:
            return 'Near Completion'
        if row['repayment_ratio'] > 0.5:
            return 'Partially Repaid'
        return 'Early Stage'

    df['portfolio_status'] = df.apply(categorize_loan, axis=1)
    df['risk_score'] = np.clip((1 - df['repayment_ratio']).fillna(0), 0, 1)

    df.to_csv(silver_path, index=False)
    print(f'Cleaned data saved to: {silver_path}')
    return df


if __name__ == '__main__':
    clean_ibrd_data()
