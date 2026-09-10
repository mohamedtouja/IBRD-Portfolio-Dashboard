Dashboard data directory: data/dashboard/

This folder contains precomputed CSVs used by the Streamlit dashboard. Run `notebooks/10_dashboard_data_prep.ipynb` to regenerate.

Files and schemas:

- kpi_summary.csv
  - total_commitments: numeric
  - total_disbursed: numeric
  - total_repaid: numeric
  - total_outstanding: numeric
  - num_loans: int
  - num_countries: int
  - num_regions: int
  - repayment_rate: float
  - cancellation_rate: float
  - avg_loan_size: numeric
  - median_loan_size: numeric

- regional_summary.csv
  - Region: string
  - commitments: numeric
  - loans: int
  - outstanding: numeric
  - avg_risk: float
  - repayment_rate: float

- country_summary.csv
  - Country / Economy: string
  - commitments: numeric
  - loans: int
  - outstanding: numeric
  - avg_risk: float
  - cluster: optional int/string

- yearly_summary.csv
  - approval_year: int
  - loans: int
  - commitments: numeric
  - disbursed: numeric
  - repaid: numeric

- status_summary.csv
  - Loan Status: string
  - count: int
  - total_amount: numeric
  - pct_portfolio: float

- loan_size_summary.csv
  - loan_size_category: string
  - count: int
  - avg_commitment: numeric
  - avg_risk: float

- age_summary.csv
  - age_category: string
  - count: int
  - outstanding: numeric
  - avg_repayment: numeric

- top_countries.csv
  - Country / Economy: string
  - commitments: numeric

- high_risk_summary.csv
  - Region/Country: grouping columns
  - count: int
  - total_outstanding: numeric

- forecast_data.csv
  - year: int
  - commitments: numeric

Notes:

- The notebook tries to map common column name variants; if a required source column is missing, the corresponding output will be skipped.
- The files are intended to be read directly by the Streamlit app without heavy computation.
