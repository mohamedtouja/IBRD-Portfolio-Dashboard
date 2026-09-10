# IBRD Portfolio Analytics

This project is a clean rebuild of a simple portfolio analytics application for IBRD loan data.

## Structure

- src/data_pipeline: cleaning and feature engineering
- src/models: repayment model and country risk clustering
- dashboard: Streamlit app
- data/raw: raw CSV inputs
- data/processed: cleaned CSV outputs
- notebooks: analysis notebooks

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
```

## Notes

If no raw data file is present, the pipeline creates a synthetic portfolio dataset so the project remains runnable out of the box.
