"""Plotly figure builders for the dashboard.

Pure functions: each takes a prepared frame and returns a ``go.Figure`` with no
knowledge of Streamlit. Colour assignment follows the roles in
:mod:`src.config` -- categorical hues in fixed slot order, a single-hue ramp for
magnitude, a two-pole ramp for polarity, and reserved status colours that are
never reused as series hues.

Backgrounds are left transparent so the host application's theme supplies the
surface, and grids are kept recessive so the marks carry the chart.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import config

logger = logging.getLogger(__name__)

_GRID_COLOR = "rgba(128,128,128,0.18)"
_EMPTY_MESSAGE = "No data available for the current selection"


def _style(figure: go.Figure, height: int = 380, show_legend: bool = False) -> go.Figure:
    """Apply the shared layout treatment to a figure.

    Args:
        figure: The figure to restyle.
        height: Rendered height in pixels.
        show_legend: Whether the legend should be visible. A legend is required
            whenever two or more series share an axis.

    Returns:
        The same figure, restyled.
    """
    figure.update_layout(
        height=height,
        showlegend=show_legend,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor=_GRID_COLOR)
    figure.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False, linecolor=_GRID_COLOR)
    return figure


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` colour to an ``rgba()`` string.

    Args:
        hex_color: Six-digit hex colour, with or without a leading ``#``.
        alpha: Opacity between 0 and 1.

    Returns:
        An ``rgba(r, g, b, a)`` string. Falls back to transparent black when the
        input is not a six-digit hex colour.
    """
    value = hex_color.lstrip("#")
    if len(value) != 6:
        logger.warning("Unexpected colour format: %r", hex_color)
        return f"rgba(0, 0, 0, {alpha})"
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def empty_figure(message: str = _EMPTY_MESSAGE) -> go.Figure:
    """Build a placeholder figure carrying an explanatory message.

    Args:
        message: Text to display in the centre of the plot area.

    Returns:
        An axis-free figure showing only the message.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="rgba(128,128,128,0.9)"),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _style(figure, height=240)


def region_commitments_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot total commitments by region.

    Args:
        summary: Output of :func:`src.analytics.regional.regional_summary`.

    Returns:
        A single-series bar chart. Being one series, it carries no legend.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary.sort_values("commitments"),
        x="commitments",
        y="Region",
        orientation="h",
        title="Total commitments by region",
        labels={"commitments": "Commitments (US$)", "Region": ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[0],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Commitments: $%{x:,.0f}<extra></extra>",
    )
    return _style(figure, height=380)


def portfolio_status_pie(summary: pd.DataFrame) -> go.Figure:
    """Plot the share of loans in each lifecycle stage.

    Args:
        summary: Output of
            :func:`src.analytics.kpis.portfolio_status_breakdown`.

    Returns:
        A donut chart with direct labels on every slice.
    """
    if summary.empty:
        return empty_figure()

    figure = px.pie(
        summary,
        names="portfolio_status",
        values="count",
        title="Portfolio status distribution",
        hole=0.55,
        color_discrete_sequence=list(config.CATEGORICAL_COLORS),
    )
    figure.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(line=dict(color="rgba(255,255,255,0.9)", width=2)),
        hovertemplate="<b>%{label}</b><br>Loans: %{value:,}<br>Share: %{percent}<extra></extra>",
    )
    return _style(figure, height=400)


def status_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot loan counts by reported loan status.

    Args:
        summary: Output of :func:`src.analytics.kpis.status_breakdown`.

    Returns:
        A single-series horizontal bar chart.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary.sort_values("count"),
        x="count",
        y="Loan Status",
        orientation="h",
        title="Loans by status",
        labels={"count": "Loans", "Loan Status": ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[0],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Loans: %{x:,}<extra></extra>",
    )
    return _style(figure, height=400)


def yearly_activity_line(summary: pd.DataFrame) -> go.Figure:
    """Plot commitments, disbursements and repayments by approval year.

    All three measures are US dollar amounts, so they share one axis.

    Args:
        summary: Output of :func:`src.analytics.kpis.yearly_activity`.

    Returns:
        A three-series line chart with a legend.
    """
    if summary.empty:
        return empty_figure()

    long = summary.melt(
        id_vars="approval_year",
        value_vars=[column for column in ("commitments", "disbursed", "repaid") if column in summary.columns],
        var_name="measure",
        value_name="amount",
    )
    labels = {"commitments": "Commitments", "disbursed": "Disbursed", "repaid": "Repaid"}
    long["measure"] = long["measure"].map(labels).fillna(long["measure"])

    figure = px.line(
        long,
        x="approval_year",
        y="amount",
        color="measure",
        title="Annual lending activity",
        labels={"approval_year": "Approval year", "amount": "US$"},
        color_discrete_sequence=list(config.CATEGORICAL_COLORS),
    )
    figure.update_traces(line_width=2, hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>")
    figure.update_layout(hovermode="x unified")
    return _style(figure, height=400, show_legend=True)


def region_treemap(data: pd.DataFrame) -> go.Figure:
    """Plot commitments as a region and country hierarchy.

    Colour encodes average risk on a single-hue ramp, so the treemap reads as
    magnitude rather than as eight competing categories.

    Args:
        data: Output of
            :func:`src.analytics.regional.region_country_treemap_data`.

    Returns:
        A treemap keyed by region then country.
    """
    if data.empty:
        return empty_figure()

    figure = px.treemap(
        data,
        path=[px.Constant("Portfolio"), "Region", "Country / Economy"],
        values="commitments",
        color="avg_risk",
        title="Commitments by region and country",
        color_continuous_scale=list(config.SEQUENTIAL_BLUE),
    )
    figure.update_traces(
        marker=dict(line=dict(color="rgba(255,255,255,0.9)", width=2)),
        hovertemplate="<b>%{label}</b><br>Commitments: $%{value:,.0f}<br>Avg risk: %{color:.2f}<extra></extra>",
    )
    figure.update_layout(coloraxis_colorbar=dict(title="Avg risk"))
    return _style(figure, height=520)


def regional_trend_area(summary: pd.DataFrame) -> go.Figure:
    """Plot commitments over time stacked by region.

    Args:
        summary: Output of
            :func:`src.analytics.regional.regional_yearly_trend`.

    Returns:
        A stacked area chart with a legend.
    """
    if summary.empty:
        return empty_figure()

    figure = px.area(
        summary,
        x="approval_year",
        y="commitments",
        color="Region",
        title="Commitments over time by region",
        labels={"approval_year": "Approval year", "commitments": "Commitments (US$)"},
        color_discrete_sequence=list(config.CATEGORICAL_COLORS),
    )
    figure.update_traces(line_width=0, hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>")
    figure.update_layout(hovermode="x unified")
    return _style(figure, height=420, show_legend=True)


def risk_distribution_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot loan counts across ordered risk bands.

    Bands are ordered, so they take an ordinal single-hue ramp rather than
    unrelated categorical hues.

    Args:
        summary: Output of :func:`src.analytics.risk.risk_distribution`.

    Returns:
        A single-series bar chart.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary,
        x="risk_band",
        y="count",
        title="Loans by risk band",
        labels={"risk_band": "", "count": "Loans"},
    )
    figure.update_traces(
        marker_color=list(config.ORDINAL_BLUE)[: len(summary)],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Loans: %{y:,}<extra></extra>",
    )
    return _style(figure, height=340)


def risk_by_dimension_bar(summary: pd.DataFrame, dimension: str, title: str) -> go.Figure:
    """Plot average risk score grouped by an arbitrary dimension.

    Args:
        summary: Output of :func:`src.analytics.risk.risk_by_dimension`.
        dimension: Name of the grouping column.
        title: Chart title.

    Returns:
        A single-series horizontal bar chart.
    """
    if summary.empty or dimension not in summary.columns:
        return empty_figure()

    figure = px.bar(
        summary.sort_values("avg_risk"),
        x="avg_risk",
        y=dimension,
        orientation="h",
        title=title,
        labels={"avg_risk": "Average risk score", dimension: ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[0],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Average risk: %{x:.3f}<extra></extra>",
    )
    return _style(figure, height=380)


def risk_vs_age_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot average risk across ordered loan-age buckets.

    Args:
        summary: Output of :func:`src.analytics.risk.risk_vs_age`.

    Returns:
        A single-series bar chart in age order.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary,
        x="age_category",
        y="avg_risk",
        title="Average risk by loan age",
        labels={"age_category": "Loan age (years)", "avg_risk": "Average risk score"},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[0],
        marker_line_width=0,
        hovertemplate="<b>%{x} years</b><br>Average risk: %{y:.3f}<extra></extra>",
    )
    return _style(figure, height=340)


def interest_vs_risk_scatter(data: pd.DataFrame) -> go.Figure:
    """Plot interest rate against risk score.

    Magnitude is carried by a single-hue ramp on principal amount rather than by
    a categorical colour, so every pair of points stays distinguishable.

    Args:
        data: Output of :func:`src.analytics.risk.interest_rate_vs_risk`.

    Returns:
        A scatter plot.
    """
    if data.empty:
        return empty_figure()

    figure = px.scatter(
        data,
        x="Interest Rate",
        y="risk_score",
        color="Original Principal Amount (US$)",
        title="Interest rate against risk score",
        labels={"Interest Rate": "Interest rate (%)", "risk_score": "Risk score"},
        color_continuous_scale=list(config.SEQUENTIAL_BLUE),
        opacity=0.65,
    )
    figure.update_traces(
        marker=dict(size=8, line=dict(width=0)),
        hovertemplate="Rate: %{x:.2f}%<br>Risk: %{y:.3f}<br>Principal: $%{marker.color:,.0f}<extra></extra>",
    )
    figure.update_layout(coloraxis_colorbar=dict(title="Principal"))
    return _style(figure, height=420)


def segment_scatter(data: pd.DataFrame) -> go.Figure:
    """Plot countries by repayment ratio against risk, coloured by segment.

    Args:
        data: Output of
            :func:`src.analytics.segments.segment_scatter_data`.

    Returns:
        A bubble chart sized by commitments, with a legend.
    """
    if data.empty:
        return empty_figure()

    figure = px.scatter(
        data,
        x="avg_repayment_ratio",
        y="avg_risk_score",
        color="segment_name",
        size="total_commitments" if "total_commitments" in data.columns else None,
        hover_name="Country / Economy" if "Country / Economy" in data.columns else None,
        title="Country segments by repayment and risk",
        labels={
            "avg_repayment_ratio": "Average repayment ratio",
            "avg_risk_score": "Average risk score",
            "segment_name": "Segment",
        },
        color_discrete_map=dict(config.SEGMENT_COLORS),
        color_discrete_sequence=list(config.CATEGORICAL_COLORS[:3]),
        size_max=42,
    )
    figure.update_traces(marker=dict(line=dict(width=2, color="rgba(255,255,255,0.9)")))
    return _style(figure, height=460, show_legend=True)


def segment_profile_radar(profiles: pd.DataFrame) -> go.Figure:
    """Plot normalised segment centroids as overlapping radar traces.

    Args:
        profiles: Output of
            :func:`src.analytics.segments.segment_profiles`.

    Returns:
        A radar chart with one closed trace per segment and a legend.
    """
    if profiles.empty:
        return empty_figure()

    labels = {
        "avg_repayment_ratio": "Repayment",
        "avg_risk_score": "Risk",
        "avg_loan_age": "Loan age",
        "cancellation_rate": "Cancellation",
    }
    figure = go.Figure()
    for index, (name, group) in enumerate(profiles.groupby("segment_name")):
        axes = [labels.get(value, value) for value in group["feature"]]
        color = config.SEGMENT_COLORS.get(
            str(name), config.CATEGORICAL_COLORS[index % len(config.CATEGORICAL_COLORS)]
        )
        figure.add_trace(
            go.Scatterpolar(
                r=group["normalised"].tolist() + group["normalised"].tolist()[:1],
                theta=axes + axes[:1],
                name=str(name),
                line=dict(color=color, width=2),
                fill="toself",
                fillcolor=_rgba(color, 0.18),
                hovertemplate="<b>%{theta}</b><br>Normalised: %{r:.2f}<extra>" + str(name) + "</extra>",
            )
        )
    figure.update_layout(
        title="Segment profiles (min-max normalised)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor=_GRID_COLOR)),
    )
    return _style(figure, height=440, show_legend=True)


def segment_overview_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot total commitments per segment.

    Args:
        summary: Output of
            :func:`src.analytics.segments.segment_overview`.

    Returns:
        A single-series bar chart coloured by segment identity.
    """
    if summary.empty:
        return empty_figure()

    colors = [
        config.SEGMENT_COLORS.get(str(name), config.CATEGORICAL_COLORS[index % 8])
        for index, name in enumerate(summary["segment_name"])
    ]
    figure = px.bar(
        summary,
        x="segment_name",
        y="total_commitments",
        title="Commitments by segment",
        labels={"segment_name": "", "total_commitments": "Commitments (US$)"},
    )
    figure.update_traces(
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Commitments: $%{y:,.0f}<extra></extra>",
    )
    return _style(figure, height=360)


def anomaly_reason_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot flagged loan counts by driving feature.

    Args:
        summary: Output of
            :func:`src.analytics.anomalies.reason_breakdown`.

    Returns:
        A single-series horizontal bar chart.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary.sort_values("count"),
        x="count",
        y="reason",
        orientation="h",
        title="Anomalies by driving feature",
        labels={"count": "Flagged loans", "reason": ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[1],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Flagged loans: %{x:,}<extra></extra>",
    )
    return _style(figure, height=380)


def anomaly_score_histogram(scores: pd.DataFrame) -> go.Figure:
    """Plot the distribution of combined anomaly scores.

    Args:
        scores: Output of
            :func:`src.analytics.anomalies.score_distribution`.

    Returns:
        A histogram.
    """
    if scores.empty:
        return empty_figure()

    figure = px.histogram(
        scores,
        x="combined_score",
        nbins=30,
        title="Combined anomaly score distribution",
        labels={"combined_score": "Combined score"},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[1],
        marker_line=dict(color="rgba(255,255,255,0.9)", width=2),
        hovertemplate="Score %{x:.3f}<br>Loans: %{y:,}<extra></extra>",
    )
    figure.update_yaxes(title="Loans")
    return _style(figure, height=340)


def anomaly_region_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot flagged loan counts by region.

    Args:
        summary: Output of
            :func:`src.analytics.anomalies.region_breakdown`.

    Returns:
        A single-series horizontal bar chart.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary.sort_values("count"),
        x="count",
        y="Region",
        orientation="h",
        title="Anomalies by region",
        labels={"count": "Flagged loans", "Region": ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[1],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Flagged loans: %{x:,}<extra></extra>",
    )
    return _style(figure, height=340)


def forecast_line(series: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    """Plot a measure over time, splitting history from projection.

    Args:
        series: Output of :func:`src.analytics.forecast.commitment_series` or
            :func:`src.analytics.forecast.outstanding_series`.
        value_column: Name of the value column to plot.
        title: Chart title.

    Returns:
        A two-series line chart, with the forecast arm dashed and a legend.
    """
    if series.empty or value_column not in series.columns:
        return empty_figure()

    figure = go.Figure()
    for index, kind in enumerate(("Historical", "Forecast")):
        subset = series[series["kind"] == kind]
        if subset.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=subset["year"],
                y=subset[value_column],
                name=kind,
                mode="lines+markers",
                line=dict(
                    color=config.CATEGORICAL_COLORS[index],
                    width=2,
                    dash="dash" if kind == "Forecast" else "solid",
                ),
                marker=dict(size=8, line=dict(width=0)),
                hovertemplate="%{x}<br>US$%{y:,.1f}B<extra>" + kind + "</extra>",
            )
        )
    figure.update_layout(title=title, xaxis_title="Year", yaxis_title="US$ billions")
    return _style(figure, height=420, show_legend=True)


def scenario_line(scenarios: pd.DataFrame) -> go.Figure:
    """Plot the base, best and worst case commitment scenarios.

    Args:
        scenarios: Output of
            :func:`src.analytics.forecast.scenario_table`.

    Returns:
        A three-series line chart with a legend.
    """
    if scenarios.empty:
        return empty_figure()

    figure = px.line(
        scenarios,
        x="year",
        y="commitments_billions",
        color="scenario",
        markers=True,
        title="Commitment scenarios",
        labels={"year": "Year", "commitments_billions": "US$ billions", "scenario": "Scenario"},
        color_discrete_sequence=list(config.CATEGORICAL_COLORS[:3]),
    )
    figure.update_traces(line_width=2, marker=dict(size=8, line=dict(width=0)))
    figure.update_layout(hovermode="x unified")
    return _style(figure, height=420, show_legend=True)


def feature_importance_bar(importance: pd.DataFrame) -> go.Figure:
    """Plot gain-based feature importances from the trained booster.

    Args:
        importance: Output of
            :func:`src.models.explain.global_importance`.

    Returns:
        A single-series horizontal bar chart.
    """
    if importance.empty:
        return empty_figure("Feature importances are unavailable for this model")

    figure = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Model feature importance (gain)",
        labels={"importance": "Importance", "feature": ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[0],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    )
    return _style(figure, height=max(340, 26 * len(importance)))


def shap_summary_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot mean absolute SHAP values across a portfolio sample.

    Args:
        summary: Output of
            :func:`src.models.explain.sample_shap_summary`.

    Returns:
        A single-series horizontal bar chart.
    """
    if summary.empty:
        return empty_figure("SHAP summary is unavailable")

    figure = px.bar(
        summary.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Mean absolute SHAP value",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
    )
    figure.update_traces(
        marker_color=config.CATEGORICAL_COLORS[2],
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    )
    return _style(figure, height=max(340, 26 * len(summary)))


def shap_contribution_bar(contributions: list[dict[str, object]]) -> go.Figure:
    """Plot signed SHAP contributions for a single prediction.

    Sign is polarity, so the two directions take opposite poles of a diverging
    pair rather than two unrelated categorical hues.

    Args:
        contributions: Output of
            :func:`src.models.explain.top_contributions`.

    Returns:
        A diverging horizontal bar chart centred on zero.
    """
    if not contributions:
        return empty_figure("No explanation available")

    frame = pd.DataFrame(contributions).sort_values("contribution")
    colors = [
        config.DIVERGING_HIGH if float(value) > 0 else config.DIVERGING_LOW
        for value in frame["contribution"]
    ]
    figure = px.bar(
        frame,
        x="contribution",
        y="feature",
        orientation="h",
        title="Top feature contributions",
        labels={"contribution": "SHAP contribution", "feature": ""},
    )
    figure.update_traces(
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.4f}<extra></extra>",
    )
    figure.add_vline(x=0, line_width=1, line_color=_GRID_COLOR)
    return _style(figure, height=max(260, 60 * len(frame)))


def loan_size_bar(summary: pd.DataFrame) -> go.Figure:
    """Plot loan counts across ordered size categories.

    Args:
        summary: Output of :func:`src.analytics.kpis.loan_size_breakdown`.

    Returns:
        A single-series bar chart in size order.
    """
    if summary.empty:
        return empty_figure()

    figure = px.bar(
        summary,
        x="loan_size_category",
        y="count",
        title="Loans by size category",
        labels={"loan_size_category": "", "count": "Loans"},
    )
    figure.update_traces(
        marker_color=list(config.ORDINAL_BLUE)[: len(summary)],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Loans: %{y:,}<extra></extra>",
    )
    return _style(figure, height=340)
