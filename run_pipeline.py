from pathlib import Path

from src.data_pipeline.silver_layer import clean_ibrd_data
from src.models.country_clustering import country_risk_clustering
from src.models.ml_risk import (
    build_country_umap,
    detect_anomalies,
    estimate_repayment_survival,
    forecast_portfolio_trends,
    train_default_watchlist_model,
)
from src.models.repayment_model import train_repayment_model


def generate_summary_report(df):
    report = {
        'Total Loans': len(df),
        'Total Countries': df['Country / Economy'].nunique(),
        'Total Regions': df['Region'].nunique(),
        'Total Commitments': df['Original Principal Amount (US$)'].sum(),
        'Total Disbursed': df['Disbursed Amount (US$)'].sum(),
        'Total Repaid': df['Repaid to IBRD (US$)'].sum(),
        'Total Outstanding': df['Due to IBRD (US$)'].sum(),
        'Average Loan Size': df['Original Principal Amount (US$)'].mean(),
        'Repayment Rate': df['Repaid to IBRD (US$)'].sum() / df['Disbursed Amount (US$)'].sum() * 100,
        'Top Borrower': df.groupby('Country / Economy')['Original Principal Amount (US$)'].sum().idxmax(),
        'Top Region': df.groupby('Region')['Original Principal Amount (US$)'].sum().idxmax(),
    }

    out_dir = Path('data/features')
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'portfolio_summary.csv'
    import pandas as pd
    pd.DataFrame([report]).to_csv(summary_path, index=False)

    print('Portfolio summary saved to:', summary_path)
    for key, value in report.items():
        if isinstance(value, float):
            if 'Rate' in key:
                print(f'{key}: {value:.2f}%')
            else:
                print(f'{key}: {value:,.0f}')
        else:
            print(f'{key}: {value}')


def save_ml_export(frame, file_name):
    out_dir = Path('data/features')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    frame.to_csv(out_path, index=False)
    print(f'Saved ML export: {out_path}')


def main():
    print('Cleaning and preparing portfolio data...')
    df = clean_ibrd_data()
    print(f'Data shape: {df.shape}')

    print('Training repayment model...')
    model, results = train_repayment_model(df)
    print('Model accuracy:', round(results['accuracy'], 4))

    print('Training default watchlist model...')
    risk_result = train_default_watchlist_model(df, threshold=0.7)
    watchlist = risk_result['watchlist']
    print('Watchlist loans above 70% risk:', len(watchlist))
    save_ml_export(watchlist, 'loan_watchlist.csv')

    print('Running anomaly detection...')
    anomalies = detect_anomalies(df, n_outliers=10)
    print('Anomalous loans flagged:', len(anomalies))
    save_ml_export(anomalies, 'loan_anomalies.csv')

    print('Estimating repayment survival profile...')
    survival_summary = estimate_repayment_survival(df)['survival_summary']
    print('Survival records generated:', len(survival_summary))
    save_ml_export(survival_summary, 'loan_survival_profiles.csv')

    print('Forecasting next 5 years of exposure...')
    forecast = forecast_portfolio_trends(df, years_ahead=5)
    save_ml_export(forecast, 'portfolio_5y_forecast.csv')

    print('Building country UMAP projection...')
    country_projection = build_country_umap(df)
    save_ml_export(country_projection, 'country_umap_projection.csv')

    print('Running country risk clustering...')
    clusters, kmeans = country_risk_clustering(df)
    print('Clustered countries:', clusters.shape[0])

    print('Generating portfolio summary...')
    generate_summary_report(df)

    print('Pipeline complete.')


if __name__ == '__main__':
    main()
