"""IBRD portfolio analytics dashboard.

Entry point for ``streamlit run app.py``. All UI lives here; every computation,
model call and figure is imported from :mod:`src`, which stays framework-free.

Caching strategy: expensive source reads are cached (``st.cache_data`` for
frames, ``st.cache_resource`` for fitted models) and the cheap interactive
filters run uncached on each rerun, so changing a filter never re-reads a CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import config
from src.analytics import anomalies as anomaly_analytics
from src.analytics import explore, forecast, kpis, regional, risk, segments
from src.charts import figures
from src.dashboard_io import loaders
from src.models.anomaly import AnomalyScorer
from src.models.explain import global_importance, sample_shap_summary, shap_available, top_contributions
from src.models.predictor import RepaymentPredictor, derive_loan_size_category
from src.models.segmenter import CountrySegmenter
from src.utils.formatting import format_compact_currency, format_currency, format_number, format_percent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="IBRD Portfolio Dashboard", layout="wide")

CACHE_TTL = 3600


# --------------------------------------------------------------------------
# Cached bridges into src/
# --------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, max_entries=2, show_spinner="Loading portfolio…")
def get_loans() -> pd.DataFrame:
    """Load the cleaned loan portfolio.

    Returns:
        The loan-level frame, empty when the asset is missing.
    """
    return loaders.load_loans()


@st.cache_data(ttl=CACHE_TTL, max_entries=2)
def get_country_segments() -> pd.DataFrame:
    """Load the country segmentation output.

    Returns:
        One row per country, empty when the asset is missing.
    """
    return loaders.load_country_segments()


@st.cache_data(ttl=CACHE_TTL, max_entries=2)
def get_anomaly_watchlist() -> pd.DataFrame:
    """Load the anomaly watchlist.

    Returns:
        Flagged loans, empty when the asset is missing.
    """
    return loaders.load_anomaly_watchlist()


@st.cache_data(ttl=CACHE_TTL, max_entries=2)
def get_high_risk_loans() -> pd.DataFrame:
    """Load the high-risk loan extract.

    Returns:
        High-risk loans, empty when the asset is missing.
    """
    return loaders.load_high_risk_loans()


@st.cache_data(ttl=CACHE_TTL, max_entries=2)
def get_forecasts() -> pd.DataFrame:
    """Load the forecast table.

    Returns:
        Combined history and projection rows, empty when the asset is missing.
    """
    return loaders.load_forecasts()


@st.cache_resource(show_spinner="Loading repayment model…")
def _load_predictor() -> RepaymentPredictor | None:
    """Load the repayment classifier, memoised for the server's lifetime.

    Returns:
        A ready predictor, or ``None`` when the artifacts are unavailable.
    """
    return RepaymentPredictor.load()


@st.cache_resource
def _load_segmenter() -> CountrySegmenter | None:
    """Load the country segmentation models, memoised.

    Returns:
        A ready segmenter, or ``None`` when the artifacts are unavailable.
    """
    return CountrySegmenter.load(segments=loaders.load_country_segments())


@st.cache_resource(show_spinner="Fitting anomaly detector…")
def _load_anomaly_scorer() -> AnomalyScorer | None:
    """Fit the Isolation Forest used for live anomaly scoring, memoised.

    Notebook 08 never persisted its estimator, so it is refitted here from the
    cleaned portfolio using the notebook's parameters.

    Returns:
        A fitted scorer, or ``None`` when fitting was not possible.
    """
    return AnomalyScorer.fit(loaders.load_loans())


def _without_caching_failure(loader: Any) -> Any:
    """Call a cached loader, refusing to let a failure stick.

    ``st.cache_resource`` memoises whatever it is given, ``None`` included, for
    the lifetime of the server. A load that fails once for a transient reason --
    a notebook rewriting a pickle while the app reads it, for instance -- would
    otherwise keep reporting the artifact as unavailable until the app is
    restarted. Evicting the entry on failure lets the next rerun retry.

    Args:
        loader: A ``st.cache_resource``-decorated zero-argument loader.

    Returns:
        Whatever the loader returned.
    """
    result = loader()
    if result is None:
        loader.clear()
    return result


def get_predictor() -> RepaymentPredictor | None:
    """Return the repayment classifier, retrying after a failed load.

    Returns:
        A ready predictor, or ``None`` when the artifacts are unavailable.
    """
    return _without_caching_failure(_load_predictor)


def get_segmenter() -> CountrySegmenter | None:
    """Return the country segmentation models, retrying after a failed load.

    Returns:
        A ready segmenter, or ``None`` when the artifacts are unavailable.
    """
    return _without_caching_failure(_load_segmenter)


def get_anomaly_scorer() -> AnomalyScorer | None:
    """Return the anomaly scorer, retrying after a failed fit.

    Returns:
        A fitted scorer, or ``None`` when fitting was not possible.
    """
    return _without_caching_failure(_load_anomaly_scorer)


def _display_path(path: Path) -> str:
    """Render an artifact path for display, shortened when it sits in the project.

    Falls back to the absolute path, since the location can be redirected outside
    the project root by the ``IBRD_*`` environment overrides.

    Args:
        path: The path to render.

    Returns:
        A project-relative path when possible, otherwise the absolute path.
    """
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def missing_artifacts(*paths: Path) -> list[Path]:
    """List which of the given artifact paths are absent from disk.

    Args:
        *paths: Artifact locations to test.

    Returns:
        The subset that does not exist.
    """
    return [path for path in paths if not path.exists()]


def artifact_problem(*paths: Path) -> str:
    """Describe why an artifact-backed feature is unavailable.

    Distinguishes a genuinely missing file from one that is present but failed
    to load, because the remedy differs: regenerate versus inspect the logs.

    Args:
        *paths: Artifact locations the feature depends on.

    Returns:
        A sentence naming the specific problem.
    """
    absent = missing_artifacts(*paths)
    if absent:
        names = ", ".join(f"`{_display_path(path)}`" for path in absent)
        return f"Missing file(s): {names}."
    return (
        "The files are present but could not be loaded. This usually means a notebook was "
        "rewriting them as the app read them, or that `imbalanced-learn` or `xgboost` is "
        "missing from the environment running Streamlit. Rerun to retry; see the logs for the error."
    )


def metric_row(items: list[tuple[str, str]], columns: int = 4) -> None:
    """Render a row of metrics.

    Args:
        items: ``(label, value)`` pairs to display.
        columns: Number of columns per row.
    """
    for start in range(0, len(items), columns):
        chunk = items[start : start + columns]
        for column, (label, value) in zip(st.columns(len(chunk)), chunk):
            column.metric(label, value)


def show_table(frame: pd.DataFrame, caption: str | None = None, key: str | None = None) -> None:
    """Render a dataframe with an optional caption.

    Charts on this page use colours that need a text or table fallback, so every
    chart section pairs with one of these.

    Args:
        frame: Frame to display.
        caption: Optional caption rendered beneath the table.
        key: Stable element key, required when the same frame renders in more
            than one place.
    """
    st.dataframe(frame, width="stretch", hide_index=True, key=key)
    if caption:
        st.caption(caption)


# --------------------------------------------------------------------------
# Data + sidebar
# --------------------------------------------------------------------------
st.title("IBRD Portfolio Dashboard")

try:
    loans = get_loans()
except Exception:  # noqa: BLE001 - surface any load failure without crashing
    logger.exception("Portfolio load failed")
    st.error("Could not load the portfolio dataset. Check the logs for details.")
    loans = pd.DataFrame()

if loans.empty:
    st.info(
        "No portfolio data found. Expected `data/processed/ibrd_clean.csv`. "
        "Run the notebooks in `notebooks/` (01 → 10) to generate it.",
        icon=":material/info:",
    )

with st.sidebar:
    st.header("Filters")
    st.caption("Applies to Portfolio, Overview, Regional, Risk and Data explorer.")

    region_options = explore.filter_options(loans, "Region")
    status_options = explore.filter_options(loans, "Loan Status")
    type_options = explore.filter_options(loans, "Loan Type")
    size_options = [
        value for value in config.LOAN_SIZE_LABELS if value in explore.filter_options(loans, "loan_size_category")
    ]
    min_year, max_year = explore.year_bounds(loans)

    selected_regions = st.multiselect("Region", region_options, placeholder="All regions")
    selected_statuses = st.multiselect("Loan status", status_options, placeholder="All statuses")
    selected_types = st.multiselect("Loan type", type_options, placeholder="All types")
    selected_sizes = st.pills("Loan size", size_options, selection_mode="multi", default=None)
    selected_years = st.slider(
        "Approval year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    search_term = st.text_input("Search", placeholder="Loan, country, borrower, project")

    st.divider()
    with st.expander("Data assets", icon=":material/inventory_2:"):
        for name, present in loaders.asset_status().items():
            st.write(f"{':material/check_circle:' if present else ':material/cancel:'} {name}")

filtered = explore.apply_filters(
    loans,
    regions=selected_regions,
    statuses=selected_statuses,
    loan_types=selected_types,
    size_categories=selected_sizes,
    year_range=selected_years,
    search=search_term,
)

if not loans.empty:
    st.caption(
        f"Showing {format_number(len(filtered))} of {format_number(len(loans))} loans "
        f"· approval years {selected_years[0]}–{selected_years[1]}"
    )

tab_portfolio, tab_overview, tab_regional, tab_risk, tab_segments, tab_anomalies, tab_forecast, tab_predict, tab_explain, tab_explorer = st.tabs(
    [
        ":material/dashboard: Portfolio",
        ":material/insights: IBRD overview",
        ":material/public: Regional",
        ":material/warning: Risk",
        ":material/scatter_plot: Segments",
        ":material/troubleshoot: Anomalies",
        ":material/trending_up: Forecast",
        ":material/online_prediction: Predict",
        ":material/psychology: Explainability",
        ":material/table_view: Data explorer",
    ]
)


# --------------------------------------------------------------------------
# Portfolio - preserves the original view
# --------------------------------------------------------------------------
with tab_portfolio:
    if filtered.empty:
        st.info("No loans match the current filters.", icon=":material/filter_alt_off:")
    else:
        try:
            summary = kpis.portfolio_kpis(filtered)
            metric_row(
                [
                    ("Total commitments", format_currency(summary["total_commitments"])),
                    ("Total disbursed", format_currency(summary["total_disbursed"])),
                    ("Total repaid", format_currency(summary["total_repaid"])),
                    ("Total outstanding", format_currency(summary["total_outstanding"])),
                ]
            )
            region_summary = regional.regional_summary(filtered)
            st.plotly_chart(figures.region_commitments_bar(region_summary), width="stretch", key="fig_1")
            status_summary = kpis.portfolio_status_breakdown(filtered)
            st.plotly_chart(figures.portfolio_status_pie(status_summary), width="stretch", key="fig_2")
            with st.expander("View underlying data", icon=":material/table_rows:"):
                show_table(region_summary, key="tbl_2")
                show_table(status_summary, key="tbl_3")
        except Exception:  # noqa: BLE001
            logger.exception("Portfolio tab failed")
            st.error("Could not render the portfolio view.")


# --------------------------------------------------------------------------
# IBRD overview
# --------------------------------------------------------------------------
with tab_overview:
    if filtered.empty:
        st.info("No loans match the current filters.", icon=":material/filter_alt_off:")
    else:
        try:
            summary = kpis.portfolio_kpis(filtered)
            metric_row(
                [
                    ("Loans", format_number(summary["num_loans"])),
                    ("Countries", format_number(summary["num_countries"])),
                    ("Regions", format_number(summary["num_regions"])),
                    ("Active loans", format_number(summary["active_loans"])),
                    ("Commitments", format_compact_currency(summary["total_commitments"])),
                    ("Outstanding", format_compact_currency(summary["total_outstanding"])),
                    ("Repayment rate", format_percent(summary["repayment_rate"])),
                    ("Cancellation rate", format_percent(summary["cancellation_rate"])),
                ]
            )
            st.divider()

            left, right = st.columns(2)
            with left:
                st.plotly_chart(figures.status_bar(kpis.status_breakdown(filtered)), width="stretch", key="fig_3")
            with right:
                st.plotly_chart(figures.loan_size_bar(kpis.loan_size_breakdown(filtered)), width="stretch", key="fig_4")

            yearly = kpis.yearly_activity(filtered)
            st.plotly_chart(figures.yearly_activity_line(yearly), width="stretch", key="fig_5")

            with st.container(border=True):
                st.subheader("Portfolio averages")
                metric_row(
                    [
                        ("Average loan size", format_compact_currency(summary["avg_loan_size"])),
                        ("Median loan size", format_compact_currency(summary["median_loan_size"])),
                        ("Undisbursed", format_compact_currency(summary["total_undisbursed"])),
                        ("Average risk score", f"{summary['avg_risk_score']:.3f}"),
                    ]
                )

            with st.expander("View underlying data", icon=":material/table_rows:"):
                show_table(yearly, key="tbl_4")
        except Exception:  # noqa: BLE001
            logger.exception("Overview tab failed")
            st.error("Could not render the overview.")


# --------------------------------------------------------------------------
# Regional
# --------------------------------------------------------------------------
with tab_regional:
    if filtered.empty:
        st.info("No loans match the current filters.", icon=":material/filter_alt_off:")
    else:
        try:
            region_summary = regional.regional_summary(filtered)
            st.plotly_chart(figures.region_commitments_bar(region_summary), width="stretch", key="fig_6")
            show_table(region_summary, "Regional totals for the current filter selection.", key="tbl_5")

            st.divider()
            st.plotly_chart(
                figures.regional_trend_area(regional.regional_yearly_trend(filtered)), width="stretch"
            , key="fig_7")

            st.divider()
            top_n = st.slider("Countries to show", min_value=10, max_value=120, value=50, step=10)
            st.plotly_chart(
                figures.region_treemap(regional.region_country_treemap_data(filtered, top_n=top_n)),
                width="stretch", key="fig_8")
            show_table(
                regional.country_summary(filtered, top_n=top_n),
                "Country-level totals, ranked by commitments.", key="tbl_6")
        except Exception:  # noqa: BLE001
            logger.exception("Regional tab failed")
            st.error("Could not render the regional view.")


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------
with tab_risk:
    if filtered.empty:
        st.info("No loans match the current filters.", icon=":material/filter_alt_off:")
    else:
        try:
            distribution = risk.risk_distribution(filtered)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(figures.risk_distribution_bar(distribution), width="stretch", key="fig_9")
            with right:
                st.plotly_chart(figures.risk_vs_age_bar(risk.risk_vs_age(filtered)), width="stretch", key="fig_10")
            show_table(distribution, "Loan counts and outstanding exposure per risk band.", key="tbl_7")

            st.divider()
            dimension = st.selectbox(
                "Group average risk by",
                [
                    value
                    for value in ("Region", "Loan Type", "loan_size_category", "portfolio_status", "Loan Status")
                    if value in filtered.columns
                ],
            )
            by_dimension = risk.risk_by_dimension(filtered, dimension)
            st.plotly_chart(
                figures.risk_by_dimension_bar(by_dimension, dimension, f"Average risk by {dimension}"),
                width="stretch", key="fig_11")
            show_table(by_dimension, key="tbl_8")

            st.divider()
            st.plotly_chart(
                figures.interest_vs_risk_scatter(risk.interest_rate_vs_risk(filtered)), width="stretch"
            , key="fig_12")

            st.divider()
            st.subheader("Largest outstanding exposures")
            show_table(risk.top_exposures(filtered, top_n=20), key="tbl_9")

            high_risk = get_high_risk_loans()
            if high_risk.empty:
                st.info(
                    "High-risk extract not found. Expected `data/features/high_risk_loans.csv`.",
                    icon=":material/info:",
                )
            else:
                st.divider()
                st.subheader("High-risk extract")
                st.caption("Precomputed by notebook 04 across the full portfolio; sidebar filters do not apply.")
                show_table(risk.high_risk_summary(high_risk), key="tbl_10")
        except Exception:  # noqa: BLE001
            logger.exception("Risk tab failed")
            st.error("Could not render the risk view.")


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------
with tab_segments:
    country_segments = get_country_segments()
    if country_segments.empty:
        st.info(
            "Country segments not found. Expected `data/features/country_segments.csv`. "
            "Run `notebooks/06_country_segmentation.ipynb` to generate it.",
            icon=":material/info:",
        )
    else:
        try:
            st.caption("K-Means segmentation across all countries; sidebar filters do not apply.")
            overview = segments.segment_overview(country_segments)
            metric_row(
                [
                    ("Countries", format_number(len(country_segments))),
                    ("Segments", format_number(overview["segment_name"].nunique())),
                    ("Commitments", format_compact_currency(overview["total_commitments"].sum())),
                    ("Outstanding", format_compact_currency(overview["total_outstanding"].sum())),
                ]
            )
            st.divider()

            left, right = st.columns(2)
            with left:
                st.plotly_chart(figures.segment_overview_bar(overview), width="stretch", key="fig_13")
            with right:
                st.plotly_chart(
                    figures.segment_profile_radar(segments.segment_profiles(country_segments)),
                    width="stretch", key="fig_14")
            show_table(overview, "One row per segment.", key="tbl_11")

            st.divider()
            st.plotly_chart(
                figures.segment_scatter(segments.segment_scatter_data(country_segments)), width="stretch"
            , key="fig_15")

            st.divider()
            names = segments.segment_names(country_segments)
            chosen = st.selectbox("Inspect a segment", names) if names else None
            if chosen:
                show_table(
                    segments.countries_in_segment(country_segments, chosen, top_n=30),
                    f"Largest countries in {chosen}.", key="tbl_12")

            segmenter = get_segmenter()
            if segmenter is None:
                st.caption(
                    "K-Means artifacts unavailable, so the stored assignments are shown without live scoring."
                )
        except Exception:  # noqa: BLE001
            logger.exception("Segments tab failed")
            st.error("Could not render the segments view.")


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------
with tab_anomalies:
    watchlist = get_anomaly_watchlist()
    if watchlist.empty:
        st.info(
            "Anomaly watchlist not found. Expected `data/features/anomaly_watchlist.csv`. "
            "Run `notebooks/08_anomaly_detection_loans.ipynb` to generate it.",
            icon=":material/info:",
        )
    else:
        try:
            st.caption(
                "Isolation Forest and Local Outlier Factor consensus over the full portfolio; "
                "sidebar filters do not apply."
            )
            watchlist_kpis = anomaly_analytics.watchlist_kpis(watchlist)
            metric_row(
                [
                    ("Flagged loans", format_number(watchlist_kpis["flagged_loans"])),
                    ("Exposure at risk", format_compact_currency(watchlist_kpis["exposure"])),
                    ("Mean score", f"{watchlist_kpis['mean_score']:.3f}"),
                    ("Countries", format_number(watchlist_kpis["countries"])),
                ]
            )
            st.divider()

            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    figures.anomaly_reason_bar(anomaly_analytics.reason_breakdown(watchlist)), width="stretch"
                , key="fig_16")
            with right:
                st.plotly_chart(
                    figures.anomaly_region_bar(anomaly_analytics.region_breakdown(watchlist)), width="stretch"
                , key="fig_17")

            st.plotly_chart(
                figures.anomaly_score_histogram(anomaly_analytics.score_distribution(watchlist)),
                width="stretch", key="fig_18")

            st.divider()
            st.subheader("Watchlist")
            top_rows = st.slider("Rows to show", min_value=10, max_value=100, value=25, step=5)
            flagged = anomaly_analytics.top_anomalies(watchlist, top_n=top_rows)
            show_table(flagged, key="tbl_13")
            st.download_button(
                "Download watchlist",
                data=explore.to_csv_bytes(flagged),
                file_name="anomaly_watchlist.csv",
                mime="text/csv",
                icon=":material/download:",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Anomalies tab failed")
            st.error("Could not render the anomalies view.")


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------
with tab_forecast:
    forecasts = get_forecasts()
    if forecasts.empty:
        st.info(
            "Forecast data not found. Expected `data/features/forecasts.csv`. "
            "Run `notebooks/07_time_series_forecasting.ipynb` to generate it.",
            icon=":material/info:",
        )
    else:
        try:
            horizon = forecast.forecast_kpis(forecasts)
            metric_row(
                [
                    ("Forecast horizon", f"{horizon['horizon_years']} years"),
                    ("Through", format_number(horizon["final_year"])),
                    ("Final commitments", f"US${horizon['final_commitments']:,.1f}B"),
                    ("Final outstanding", f"US${horizon['final_outstanding']:,.1f}B"),
                ]
            )
            st.divider()

            st.plotly_chart(
                figures.forecast_line(
                    forecast.commitment_series(forecasts), "commitments_billions", "Commitments outlook"
                ),
                width="stretch", key="fig_19")
            st.plotly_chart(
                figures.forecast_line(
                    forecast.outstanding_series(forecasts), "outstanding_billions", "Outstanding balance outlook"
                ),
                width="stretch", key="fig_20")

            st.divider()
            scenarios = forecast.scenario_table(forecasts)
            st.plotly_chart(figures.scenario_line(scenarios), width="stretch", key="fig_21")
            show_table(scenarios, "Scenario values in US$ billions.", key="tbl_14")
        except Exception:  # noqa: BLE001
            logger.exception("Forecast tab failed")
            st.error("Could not render the forecast view.")


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
with tab_predict:
    predictor = get_predictor()
    if predictor is None:
        problem = artifact_problem(config.REPAYMENT_MODEL_PATH, config.REPAYMENT_FEATURES_PATH)
        st.info(
            f"Repayment model unavailable. {problem} "
            "Regenerate it with `notebooks/05_repayment_prediction_model.ipynb`.",
            icon=":material/info:",
        )
        if st.button("Retry loading the model", icon=":material/refresh:", key="retry_predictor"):
            st.rerun()
    else:
        st.caption(
            "Scores the probability that a loan reaches at least 95% repayment "
            "(the target used in notebook 05). Categorical values are encoded inside the pipeline."
        )
        regions = predictor.categories("Region")
        loan_types = predictor.categories("Loan Type")

        with st.form("prediction_form"):
            row_one = st.columns(3)
            principal = row_one[0].number_input(
                "Original principal (US$)", min_value=0.0, value=150_000_000.0, step=1_000_000.0, format="%.2f"
            )
            disbursed = row_one[1].number_input(
                "Disbursed amount (US$)", min_value=0.0, value=137_905_999.0, step=1_000_000.0, format="%.2f"
            )
            loan_age = row_one[2].number_input(
                "Loan age (years)", min_value=0.0, max_value=90.0, value=9.5, step=0.5
            )

            row_two = st.columns(3)
            cancelled_pct = row_two[0].number_input(
                "Cancelled (% of principal)", min_value=0.0, max_value=100.0, value=0.0, step=0.5
            )
            undisbursed_pct = row_two[1].number_input(
                "Undisbursed (% of principal)", min_value=0.0, max_value=100.0, value=8.06, step=0.5
            )
            size_category = row_two[2].selectbox(
                "Loan size category",
                list(config.LOAN_SIZE_LABELS),
                index=list(config.LOAN_SIZE_LABELS).index(derive_loan_size_category(principal)),
                help="Derived from the principal using the notebook 02 bin edges. Override only to explore.",
            )

            row_three = st.columns(2)
            region = row_three[0].selectbox("Region", regions) if regions else None
            loan_type = row_three[1].selectbox("Loan type", loan_types) if loan_types else None

            submitted = st.form_submit_button("Score loan", icon=":material/play_arrow:", type="primary")

        if submitted:
            if not region or not loan_type:
                st.error("Region and loan type vocabularies could not be read from the model.")
            else:
                try:
                    prediction = predictor.predict_one(
                        loan_age_years=loan_age,
                        principal=principal,
                        disbursed=disbursed,
                        cancelled_percent=cancelled_pct,
                        undisbursed_percent=undisbursed_pct,
                        region=region,
                        loan_type=loan_type,
                        loan_size_category=size_category,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Prediction failed")
                    prediction = None
                    st.error("Scoring failed. Check the logs for details.")

                if prediction is None:
                    st.session_state.pop("last_prediction", None)
                else:
                    record = prediction.to_record()
                    scorer = get_anomaly_scorer()
                    if scorer is not None:
                        outcome = scorer.score_one(
                            **{
                                "Original Principal Amount (US$)": principal,
                                "Disbursed Amount (US$)": disbursed,
                                "Undisbursed Amount (US$)": principal * undisbursed_pct / 100.0,
                                "Cancelled Amount (US$)": principal * cancelled_pct / 100.0,
                                "Repaid to IBRD (US$)": 0.0,
                                "Due to IBRD (US$)": max(disbursed, 0.0),
                                "loan_age_years": loan_age,
                                "repayment_ratio": 0.0,
                                "interest_rate": 0.0,
                            }
                        )
                        if outcome:
                            record["anomaly_score"] = round(outcome["anomaly_score"], 6)
                            record["is_anomaly"] = outcome["is_anomaly"]

                    contributions = top_contributions(
                        predictor.pipeline,
                        predictor.build_feature_frame(
                            loan_age_years=loan_age,
                            principal=principal,
                            disbursed=disbursed,
                            cancelled_percent=cancelled_pct,
                            undisbursed_percent=undisbursed_pct,
                            region=region,
                            loan_type=loan_type,
                            loan_size_category=size_category,
                        ),
                        top_n=3,
                    )
                    st.session_state["last_prediction"] = {
                        "record": record,
                        "contributions": contributions,
                    }

        stored = st.session_state.get("last_prediction")
        if stored:
            record = stored["record"]
            band = str(record["risk_band"])
            st.divider()
            columns = st.columns(3)
            columns[0].metric("Repayment probability", format_percent(record["repayment_probability_pct"]))
            columns[1].metric("Risk category", band)
            columns[1].markdown(f"{config.RISK_BAND_ICONS.get(band, '')} **{band} risk**")
            if "is_anomaly" in record:
                flagged = bool(record["is_anomaly"])
                columns[2].metric("Anomaly flag", "Flagged" if flagged else "Normal")
                columns[2].markdown(
                    f"{':material/error:' if flagged else ':material/check_circle:'} "
                    f"score {record['anomaly_score']:.3f}"
                )
            else:
                columns[2].metric("Anomaly flag", "Unavailable")
                columns[2].caption("Isolation Forest could not be fitted.")

            contributions = stored.get("contributions")
            if contributions:
                st.plotly_chart(figures.shap_contribution_bar(contributions), width="stretch", key="fig_22")
                for item in contributions:
                    st.markdown(
                        f"- **{item['feature']}** {item['direction']} the score "
                        f"({item['contribution']:+.4f})"
                    )
            elif not shap_available():
                st.caption("Install `shap` to see per-feature contributions.")
            else:
                st.caption("A per-feature explanation was not available for this input.")

            st.divider()
            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Download JSON",
                data=explore.to_json_bytes(record),
                file_name="ibrd_prediction.json",
                mime="application/json",
                icon=":material/download:",
            )
            download_columns[1].download_button(
                "Download CSV",
                data=explore.to_csv_bytes(pd.DataFrame([record])),
                file_name="ibrd_prediction.csv",
                mime="text/csv",
                icon=":material/download:",
            )


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
with tab_explain:
    predictor = get_predictor()
    if predictor is None:
        st.info(
            "Repayment model unavailable, so explanations cannot be produced. "
            f"{artifact_problem(config.REPAYMENT_MODEL_PATH, config.REPAYMENT_FEATURES_PATH)}",
            icon=":material/info:",
        )
    else:
        try:
            st.subheader("Global feature importance")
            st.caption("Gain-based importances read directly from the trained booster.")
            importance = global_importance(predictor.pipeline, top_n=15)
            st.plotly_chart(figures.feature_importance_bar(importance), width="stretch", key="fig_23")
            show_table(importance, key="tbl_15")

            st.divider()
            st.subheader("SHAP summary")
            if not shap_available():
                st.info(
                    "SHAP is not installed, so the summary is unavailable. Install it with `pip install shap`.",
                    icon=":material/info:",
                )
            elif loans.empty:
                st.info("Portfolio data is required to compute a SHAP summary.", icon=":material/info:")
            else:
                st.caption("Computed on a random sample; run it on demand because it is the slowest view here.")
                sample_size = st.slider("Sample size", min_value=100, max_value=1000, value=300, step=100)
                if st.button("Compute SHAP summary", icon=":material/calculate:"):
                    with st.spinner("Explaining a portfolio sample…"):
                        summary_frame = sample_shap_summary(
                            predictor.pipeline, loans, predictor.feature_names, sample_size=sample_size
                        )
                    if summary_frame.empty:
                        st.warning("SHAP returned no values for this sample.")
                    else:
                        st.plotly_chart(figures.shap_summary_bar(summary_frame), width="stretch", key="fig_24")
                        show_table(summary_frame, key="tbl_16")

            st.divider()
            with st.expander("Model contract", icon=":material/description:"):
                st.write("**Expected features, in order:**")
                st.write(predictor.feature_names)
                st.caption(
                    "The pipeline one-hot encodes the categorical columns internally, so callers pass raw "
                    "values. The positive class is `repayment_ratio > 0.95`."
                )
        except Exception:  # noqa: BLE001
            logger.exception("Explainability tab failed")
            st.error("Could not render the explainability view.")


# --------------------------------------------------------------------------
# Data explorer
# --------------------------------------------------------------------------
with tab_explorer:
    if loans.empty:
        st.info("No portfolio data to explore.", icon=":material/info:")
    elif filtered.empty:
        st.info("No loans match the current filters.", icon=":material/filter_alt_off:")
    else:
        try:
            st.caption(f"{format_number(len(filtered))} loans match the sidebar filters.")
            all_columns = list(filtered.columns)
            default_columns = [column for column in explore.DISPLAY_COLUMNS if column in all_columns]
            chosen_columns = st.multiselect("Columns", all_columns, default=default_columns)

            view = explore.display_frame(filtered, chosen_columns)
            row_limit = st.slider("Rows to display", min_value=50, max_value=2000, value=200, step=50)
            st.dataframe(view.head(row_limit), width="stretch", hide_index=True)

            st.download_button(
                "Download filtered data",
                data=explore.to_csv_bytes(view),
                file_name="ibrd_filtered.csv",
                mime="text/csv",
                icon=":material/download:",
            )

            with st.expander("Summary statistics", icon=":material/functions:"):
                described = explore.describe_numeric(filtered)
                if described.empty:
                    st.caption("No numeric columns available to summarise.")
                else:
                    st.dataframe(described, width="stretch", hide_index=True)
        except Exception:  # noqa: BLE001
            logger.exception("Data explorer tab failed")
            st.error("Could not render the data explorer.")
