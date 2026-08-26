"""
plots.py
Plotly figure builders matching the notebook style:
  - Teal #17B1BF for forecast intervals
  - Alpha: 0.5*(1-interval_range) + 0.1
  - Intervals: (0.01,0.99), (0.05,0.95), (0.25,0.75)
  - Observed: black line+dots; post-forecast obs: open circles
  - WIS: 2×2 subplot grid by horizon, teal boxes, model names as text inside
  - Coverage: PI numbers (10–98) on x-axis, Set2-like palette
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design constants ───────────────────────────────────────────────────────────

TEAL = "#17B1BF"

# Fan chart intervals: (q_lo, q_hi) — widest first so bands layer correctly
FAN_INTERVALS = [
    (0.01, 0.99),
    (0.05, 0.95),
    (0.25, 0.75),
]

def _fan_alpha(q_lo: float, q_hi: float) -> float:
    """Notebook formula: 0.5*(1-interval_range) + 0.1"""
    interval_range = q_hi - q_lo
    return 0.5 * (1 - interval_range) + 0.1

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"

# Set2-like palette for coverage plot (8 colors + extra)
SET2_PALETTE = [
    "#66c2a5",  # teal-green
    "#fc8d62",  # orange
    "#8da0cb",  # blue
    "#e78ac3",  # pink
    "#a6d854",  # yellow-green
    "#ffd92f",  # yellow
    "#e5c494",  # tan
    "#b3b3b3",  # gray
    "#8B5A7C",  # purple
    "#e41a1c",  # red
    "#377eb8",  # dark blue
    "#4daf4a",  # green
]

_BASE_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif",
              color="#333333", size=13),
    title_font=dict(family="'Playfair Display', Georgia, serif", size=17, color="#111111"),
)


# ── Tab 1: Fan Chart ───────────────────────────────────────────────────────────

def build_fan_chart(
    forecasts: pd.DataFrame,
    observed: pd.DataFrame,
    location: str,
    location_name: str,
    selected_models: list[str],
    ref_date: str,
    obs_weeks: int = 13,
    y_label: str = "Weekly Admissions",
) -> go.Figure:
    """
    Single-location fan chart for one reference date.
    Shows 98%, 90%, 50% PI bands + median (teal).
    Observed data: black line+dots up to ref_date, dashed line+open circles after.
    obs_weeks: how many weeks of history to show before ref_date (default 13 = ~3 months).
    """
    fig = go.Figure()

    ref_ts = pd.Timestamp(ref_date)

    obs_start = max(ref_ts - pd.Timedelta(weeks=obs_weeks), pd.Timestamp("2022-10-01"))
    obs_end   = (
        forecasts["target_end_date"].max() + pd.Timedelta(weeks=1)
        if not forecasts.empty and "target_end_date" in forecasts.columns
        else ref_ts + pd.Timedelta(weeks=5)
    )

    obs_loc = (
        observed[
            (observed["location"] == location) &
            (observed["date"] >= obs_start) &
            (observed["date"] <= obs_end)
        ]
        .sort_values("date")
    )

    obs_hist = obs_loc[obs_loc["date"] < ref_ts]
    obs_post = obs_loc[obs_loc["date"] >= ref_ts]

    # Solid line + filled markers for historical portion only
    if not obs_hist.empty:
        fig.add_trace(go.Scatter(
            x=obs_hist["date"], y=obs_hist["value"],
            mode="lines+markers",
            name="Observed",
            line=dict(color="#111111", width=1.8),
            marker=dict(size=4, color="#111111"),
            legendgroup="observed",
            hovertemplate="<b>Observed</b><br>%{x|%b %d, %Y}: %{y:,.3g}<extra></extra>",
        ))

    # Dashed line anchored to last historical point + open circle markers for post-forecast
    if not obs_post.empty:
        # Bridge point: last historical obs so dashed line starts from solid line end
        bridge = obs_hist.iloc[[-1]] if not obs_hist.empty else pd.DataFrame()
        dashed_x = pd.concat([bridge, obs_post])["date"].tolist()
        dashed_y = pd.concat([bridge, obs_post])["value"].tolist()
        # Line only (no markers) — creates the dashed segment from last solid point
        fig.add_trace(go.Scatter(
            x=dashed_x, y=dashed_y,
            mode="lines",
            name="Observed (post-forecast)",
            line=dict(color="#111111", width=1.8, dash="dash"),
            showlegend=True,
            legendgroup="observed_post",
            hovertemplate="<b>Observed (post-forecast)</b><br>%{x|%b %d, %Y}: %{y:,.3g}<extra></extra>",
        ))
        # Open circle markers only on actual post-forecast points (not the junction)
        fig.add_trace(go.Scatter(
            x=obs_post["date"].tolist(), y=obs_post["value"].tolist(),
            mode="markers",
            marker=dict(size=6, color="white",
                        line=dict(color="#111111", width=1.8),
                        symbol="circle"),
            showlegend=False,
            legendgroup="observed_post",
            hoverinfo="skip",
        ))

    # Legend entries for PI bands (neutral gray, shown once)
    for q_lo, q_hi in FAN_INTERVALS:
        interval_pct = int(round((q_hi - q_lo) * 100))
        alpha = _fan_alpha(q_lo, q_hi)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, symbol="square", color=f"rgba(100,100,100,{alpha:.2f})"),
            name=f"{interval_pct}% PI",
            legendgroup=f"pi_legend_{interval_pct}",
            showlegend=True,
        ))

    # Forecast fan per model — each model gets its own color
    fc_loc = forecasts[forecasts["location"] == location]
    for m_idx, model in enumerate(selected_models):
        color = SET2_PALETTE[m_idx % len(SET2_PALETTE)]
        m_fc = fc_loc[
            (fc_loc["model"] == model) &
            (fc_loc["reference_date"] == ref_ts) &
            (fc_loc["horizon"].between(0, 3))
        ].sort_values("target_end_date")

        if m_fc.empty:
            continue

        wide = m_fc.pivot_table(
            index="target_end_date",
            columns="output_type_id",
            values="value",
            aggfunc="first",
        )
        dates_x = wide.index.tolist()

        for q_lo, q_hi in FAN_INTERVALS:
            if q_lo not in wide.columns or q_hi not in wide.columns:
                continue
            lo = wide[q_lo].values
            hi = wide[q_hi].values
            alpha = _fan_alpha(q_lo, q_hi)
            interval_pct = int(round((q_hi - q_lo) * 100))
            fig.add_trace(go.Scatter(
                x=dates_x + dates_x[::-1],
                y=list(hi) + list(lo[::-1]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, alpha),
                line=dict(width=0),
                mode="lines",
                showlegend=False,
                name=f"{model} {interval_pct}% PI",
                legendgroup=model,
                hoverinfo="skip",
            ))

        if 0.5 in wide.columns:
            lo90 = wide[0.05].values if 0.05 in wide.columns else [float("nan")] * len(dates_x)
            hi90 = wide[0.95].values if 0.95 in wide.columns else [float("nan")] * len(dates_x)
            customdata = list(zip(lo90, hi90))
            fig.add_trace(go.Scatter(
                x=dates_x,
                y=wide[0.5].values,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                name=model,
                showlegend=True,
                legendgroup=model,
                customdata=customdata,
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "%{x|%b %d}<br>"
                    "Median: %{y:,.3g}<br>"
                    "90% PI: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"{y_label} — {location_name}", x=0.0, xanchor="left"),
        xaxis=dict(title="Date", showgrid=False, linecolor="#cccccc", ticks="outside"),
        yaxis=dict(title=y_label, showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd", borderwidth=1),
        margin=dict(l=65, r=30, t=70, b=55),
        height=480,
    )
    return fig


def build_observed_chart(
    observed: pd.DataFrame,
    location: str,
    location_name: str,
    obs_weeks: int = 13,
    y_label: str = "Weekly Admissions",
) -> go.Figure:
    """Observed data only — no forecasts."""
    fig = go.Figure()
    # Window back from the most recent observation rather than from today, so the
    # view always lands on the latest available data. Anchoring to today meant a
    # target whose reporting lags — flu and metrocast both trail by over a month
    # out of season — rendered an empty chart on the shorter windows.
    obs_all = observed[observed["location"] == location]
    anchor  = obs_all["date"].max() if not obs_all.empty else pd.Timestamp.now()
    obs_start = max(anchor - pd.Timedelta(weeks=obs_weeks), pd.Timestamp("2022-10-01"))
    obs_loc = obs_all[obs_all["date"] >= obs_start].sort_values("date")
    if not obs_loc.empty:
        fig.add_trace(go.Scatter(
            x=obs_loc["date"], y=obs_loc["value"],
            mode="lines+markers",
            name="Observed",
            line=dict(color="#111111", width=1.8),
            marker=dict(size=4, color="#111111"),
            hovertemplate="<b>Observed</b><br>%{x|%b %d, %Y}: %{y:,.3g}<extra></extra>",
        ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"{y_label} — {location_name}", x=0.0, xanchor="left"),
        xaxis=dict(title="Date", showgrid=False, linecolor="#cccccc", ticks="outside"),
        yaxis=dict(title=y_label, showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd", borderwidth=1),
        margin=dict(l=65, r=30, t=70, b=55),
        height=480,
    )
    return fig


def build_all_states_observed(
    observed: pd.DataFrame,
    locations_df: pd.DataFrame,
    obs_weeks: int = 13,
) -> go.Figure:
    """Small-multiple observed-only charts for all locations."""
    us_row     = locations_df[locations_df["location"] == "US"]
    non_us     = locations_df[locations_df["location"] != "US"]
    if "state_abb" in non_us.columns:
        state_rows = non_us.sort_values(["state_abb", "location_name"])
    elif "abbreviation" in non_us.columns:
        state_rows = non_us.sort_values("abbreviation")
    else:
        state_rows = non_us.sort_values("location_name")
    all_rows   = pd.concat([us_row, state_rows], ignore_index=True)
    locs       = list(zip(all_rows["location"], all_rows["location_name"]))

    ncols = 6
    nrows = math.ceil(len(locs) / ncols)
    # Anchored to the latest observation, as in build_observed_chart. Deliberately
    # a single global anchor rather than per-panel, so every small multiple shares
    # the same x window and stays comparable.
    anchor = observed["date"].max() if not observed.empty else pd.Timestamp.now()
    obs_start = max(anchor - pd.Timedelta(weeks=obs_weeks), pd.Timestamp("2022-10-01"))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[name for _, name in locs],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.025,
        vertical_spacing=0.06,
    )

    for idx, (fips, _) in enumerate(locs):
        row = idx // ncols + 1
        col = idx % ncols + 1
        obs_loc = (
            observed[
                (observed["location"] == fips) &
                (observed["date"] >= obs_start)
            ]
            .sort_values("date")
        )
        if not obs_loc.empty:
            fig.add_trace(go.Scatter(
                x=obs_loc["date"], y=obs_loc["value"],
                mode="lines+markers",
                name="Observed",
                line=dict(color="#111111", width=1.2),
                marker=dict(size=2, color="#111111"),
                showlegend=False,
                hoverinfo="skip",
            ), row=row, col=col)

    fig.update_xaxes(showticklabels=False, showgrid=False, linecolor="#dddddd", showline=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, linecolor="#dddddd", showline=True)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="All Locations", x=0.0, xanchor="left"),
        margin=dict(l=10, r=30, t=55, b=10),
        height=max(160 * nrows, 600),
        showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#333333")
    return fig


def build_all_states_panel(
    forecasts: pd.DataFrame,
    observed: pd.DataFrame,
    locations_df: pd.DataFrame,
    selected_models: list[str],
    ref_date: str,
    obs_weeks: int = 13,
) -> go.Figure:
    """
    Small-multiple fan charts for all locations. US first, then locations sorted by state/name.
    6-column grid. Returns the figure.
    """
    _label_col = "abbreviation" if "abbreviation" in locations_df.columns else "location_name"
    us_row     = locations_df[locations_df["location"] == "US"]
    non_us     = locations_df[locations_df["location"] != "US"]
    if "state_abb" in non_us.columns:
        state_rows = non_us.sort_values(["state_abb", "location_name"])
    elif "abbreviation" in non_us.columns:
        state_rows = non_us.sort_values("abbreviation")
    else:
        state_rows = non_us.sort_values("location_name")
    all_rows   = pd.concat([us_row, state_rows], ignore_index=True)
    locs       = list(zip(all_rows["location"], all_rows[_label_col], all_rows["location_name"]))

    ncols = 6
    nrows = math.ceil(len(locs) / ncols)

    ref_ts    = pd.Timestamp(ref_date)
    obs_start = max(ref_ts - pd.Timedelta(weeks=obs_weeks), pd.Timestamp("2022-10-01"))
    obs_end   = (
        forecasts["target_end_date"].max() + pd.Timedelta(weeks=1)
        if not forecasts.empty and "target_end_date" in forecasts.columns
        else ref_ts + pd.Timedelta(weeks=5)
    )

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[name for _, _, name in locs],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.025,
        vertical_spacing=0.06,
    )

    legend_added: set[str] = set()

    for idx, (fips, abbr, _) in enumerate(locs):
        row = idx // ncols + 1
        col = idx % ncols + 1
        rc  = dict(row=row, col=col)

        obs_loc = (
            observed[
                (observed["location"] == fips) &
                (observed["date"] >= obs_start) &
                (observed["date"] <= obs_end)
            ]
            .sort_values("date")
        )

        obs_hist = obs_loc[obs_loc["date"] < ref_ts]
        obs_post = obs_loc[obs_loc["date"] >= ref_ts]

        if not obs_hist.empty:
            fig.add_trace(go.Scatter(
                x=obs_hist["date"], y=obs_hist["value"],
                mode="lines+markers",
                name="Observed",
                line=dict(color="#111111", width=1.2),
                marker=dict(size=2, color="#111111"),
                showlegend=False,
                legendgroup="observed",
            ), **rc)

        if not obs_post.empty:
            bridge = obs_hist.iloc[[-1]] if not obs_hist.empty else pd.DataFrame()
            dashed_x = pd.concat([bridge, obs_post])["date"].tolist()
            dashed_y = pd.concat([bridge, obs_post])["value"].tolist()
            fig.add_trace(go.Scatter(
                x=dashed_x, y=dashed_y,
                mode="lines",
                line=dict(color="#111111", width=1.2, dash="dash"),
                showlegend=False,
                name="Observed (post)",
                legendgroup="observed_post",
            ), **rc)
            fig.add_trace(go.Scatter(
                x=obs_post["date"].tolist(), y=obs_post["value"].tolist(),
                mode="markers",
                marker=dict(size=3, color="white",
                            line=dict(color="#111111", width=1),
                            symbol="circle"),
                showlegend=False,
                legendgroup="observed_post",
                hoverinfo="skip",
            ), **rc)

        fc_loc = forecasts[
            (forecasts["location"] == fips) &
            (forecasts["reference_date"] == ref_ts)
        ]
        for m_idx, model in enumerate(selected_models):
            color = SET2_PALETTE[m_idx % len(SET2_PALETTE)]
            m_fc = fc_loc[(fc_loc["model"] == model) & (fc_loc["horizon"].between(0, 3))].sort_values("target_end_date")
            if m_fc.empty:
                continue

            wide = m_fc.pivot_table(
                index="target_end_date",
                columns="output_type_id",
                values="value",
                aggfunc="first",
            )
            dates_x = wide.index.tolist()

            for q_lo, q_hi in FAN_INTERVALS:
                if q_lo not in wide.columns or q_hi not in wide.columns:
                    continue
                alpha = _fan_alpha(q_lo, q_hi)
                interval_pct = int(round((q_hi - q_lo) * 100))
                fig.add_trace(go.Scatter(
                    x=dates_x + dates_x[::-1],
                    y=list(wide[q_hi].values) + list(wide[q_lo].values[::-1]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, alpha),
                    line=dict(width=0),
                    mode="lines",
                    showlegend=False,
                    name=f"{model} {interval_pct}% PI",
                    legendgroup=model,
                    hoverinfo="skip",
                ), **rc)

            if 0.5 in wide.columns:
                show_leg = model not in legend_added
                if show_leg:
                    legend_added.add(model)
                fig.add_trace(go.Scatter(
                    x=dates_x, y=wide[0.5].values,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=model,
                    showlegend=show_leg,
                    legendgroup=model,
                    hoverinfo="skip",
                ), **rc)

    fig.update_xaxes(showticklabels=False, showgrid=False, linecolor="#dddddd", showline=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, linecolor="#dddddd", showline=True)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="All Locations", x=0.0, xanchor="left"),
        margin=dict(l=10, r=140, t=55, b=10),
        legend=dict(x=1.01, y=1.0, xanchor="left",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd", borderwidth=1,
                    font=dict(size=11)),
        height=max(160 * nrows, 600),
        showlegend=True,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#333333")

    return fig


# ── Tab 2: WIS Box Plots ───────────────────────────────────────────────────────

def build_wis_boxplots(
    wis_df: pd.DataFrame,
    highlight_models: list[str] | None = None,
    log_scale: bool = False,
    by_horizon: bool = True,
    baseline_label: str = "baseline",
) -> go.Figure:
    """
    When by_horizon=True: 2×2 subplot grid, one panel per horizon.
    When by_horizon=False: single panel aggregated across all horizons.
    Matches notebook: teal boxes at 0.1 alpha, model names as y-axis labels,
    no y-axis labels, dashed darksalmon reference line at x=1.
    All hub models shown; highlight_models are bolded.
    """
    if wis_df.empty:
        return go.Figure()

    highlight_models = set(highlight_models or [])

    def _make_ticktext(model_order):
        return [
            f"<b><i><span style='color:{TEAL}'>{m}</span></i></b>" if m in highlight_models
            else f"<span style='color:#aaaaaa'>{m}</span>"
            for m in model_order
        ]

    def _add_boxes(fig, df, row, col):
        model_order = (
            df.groupby("model")["wis_ratio"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        for model in model_order:
            m_vals = df[df["model"] == model]["wis_ratio"].dropna()
            if m_vals.empty:
                continue
            median_val = float(m_vals.median())
            fig.add_trace(go.Box(
                x=m_vals, y=[model] * len(m_vals),
                orientation="h", name=model,
                marker_color=TEAL, fillcolor=_hex_to_rgba(TEAL, 0.1),
                line=dict(color=TEAL, width=1.2),
                showlegend=False, boxmean=False, boxpoints=False,
                hoverinfo="skip",
            ), row=row, col=col)
            # Invisible point at median for clean custom hover
            fig.add_trace(go.Scatter(
                x=[median_val], y=[model],
                mode="markers",
                marker=dict(size=8, color="rgba(0,0,0,0)"),
                showlegend=False,
                hovertemplate=f"<b>{model}</b><br>Median WIS ratio: {median_val:.2f}<extra></extra>",
            ), row=row, col=col)
        fig.add_vline(x=1.0, line=dict(color="#aaaaaa", width=1, dash="dot"), row=row, col=col)
        return model_order

    def _set_yaxis(fig, axis_key, model_order):
        fig.layout[axis_key].update(
            tickmode="array", tickvals=model_order,
            ticktext=_make_ticktext(model_order),
            tickfont=dict(size=11, color="dimgray"),
            showticklabels=True, showgrid=False, linecolor="#cccccc",
        )

    n_models = wis_df["model"].nunique()

    # Vertical room per model name. The 2x2 layout stacks every model twice —
    # once per panel row — so the figure has to be tall enough for 2 * n_models
    # labels, not n_models. Budgeting for one pass is what squashed the names.
    per_model_px = 22

    if by_horizon:
        panel_rows = 2
        height = max(per_model_px * n_models * panel_rows + 220, 600)
        # vertical_spacing is a fraction of total height, so a fixed 0.18 turns
        # into a several-hundred-pixel gap once the figure is tall. Hold the gap
        # near 90px instead, and never exceed the original fraction.
        v_spacing = min(0.18, 90 / height)

        horizons = [0, 1, 2, 3]
        subplot_positions = {0: (1, 1), 1: (1, 2), 2: (2, 1), 3: (2, 2)}
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"Horizon: {h} week{'s' if h != 1 else ''}" for h in horizons],
            horizontal_spacing=0.10, vertical_spacing=v_spacing,
        )
        for horizon in horizons:
            row, col = subplot_positions[horizon]
            h_df = wis_df[wis_df["horizon"] == horizon]
            if h_df.empty:
                continue
            model_order = _add_boxes(fig, h_df, row, col)
            axis_key = "yaxis" if (row == 1 and col == 1) else f"yaxis{(row - 1) * 2 + col}"
            _set_yaxis(fig, axis_key, model_order)
    else:
        height = max(per_model_px * n_models + 180, 400)
        fig = make_subplots(rows=1, cols=1)
        model_order = _add_boxes(fig, wis_df, 1, 1)
        _set_yaxis(fig, "yaxis", model_order)

    fig.update_xaxes(
        title_text=f"WIS Ratio (Model / {baseline_label})",
        title_standoff=12,
        showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc",
        type="log" if log_scale else "linear",
        rangemode="tozero" if not log_scale else None,
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        # _BASE_LAYOUT sets title_font, which leaves a title object with no text —
        # plotly then renders the string "undefined". Blank it explicitly.
        title_text="",
        height=height,
        # No in-figure title: it is rendered as a page heading instead, so the
        # teal accent rule applies. Top margin trimmed to reclaim its space,
        # while leaving room for the per-horizon subplot titles.
        margin=dict(l=220, r=30, t=50, b=70),
        boxmode="overlay",
    )
    return fig


# ── Tab 3: Coverage Calibration ────────────────────────────────────────────────

_PI_LEVELS   = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
_PI_IDEAL    = [p / 100 for p in _PI_LEVELS]
_COV_COLS    = [f"{p}_cov" for p in _PI_LEVELS]


def build_coverage_calibration(
    cov_df: pd.DataFrame,
    selected_models: list[str],
    by_horizon: bool = False,
) -> go.Figure:
    """
    Coverage calibration plot.
    by_horizon=False: single panel averaged across all horizons.
    by_horizon=True: 2×2 grid, one panel per horizon (0–3).
    """
    if cov_df.empty or not selected_models:
        return go.Figure()

    df_filt = cov_df[cov_df["model"].isin(selected_models)]

    # PI levels that have actual data across the full filtered df
    _df_mean_all = df_filt.groupby("model")[_COV_COLS].mean().reset_index()
    _valid_levels = [pi for pi, c in zip(_PI_LEVELS, _COV_COLS) if _df_mean_all[c].notna().any()]

    def _add_lines(fig, df, row, col, show_legend):
        rc = dict(row=row, col=col)
        df_mean = df.groupby("model")[_COV_COLS].mean().reset_index()

        # Only show PI levels where at least one selected model has valid data
        valid_levels = [
            pi for pi, col in zip(_PI_LEVELS, _COV_COLS)
            if df_mean[col].notna().any()
        ]
        if not valid_levels:
            return
        valid_ideal = [pi / 100 for pi in valid_levels]
        valid_cols  = [f"{pi}_cov" for pi in valid_levels]

        # Diagonal
        fig.add_trace(go.Scatter(
            x=valid_levels, y=valid_ideal, mode="lines",
            line=dict(color="#111111", width=2, dash="dash"),
            name="y = x", showlegend=show_legend, legendgroup="diag",
        ), **rc)
        for i, model in enumerate(selected_models):
            mrow = df_mean[df_mean["model"] == model]
            if mrow.empty:
                continue
            covs = [float(mrow[c].iloc[0]) for c in valid_cols]
            # Drop any remaining NaN points for this model
            xs, ys, custom = [], [], []
            for pi, cov in zip(valid_levels, covs):
                if not pd.isna(cov):
                    ideal = pi / 100
                    diff  = cov - ideal
                    if abs(diff) < 0.05:
                        label = "Well calibrated"
                    elif diff > 0:
                        label = f"Underconfident (+{diff:.1%})"
                    else:
                        label = f"Overconfident ({diff:.1%})"
                    xs.append(pi)
                    ys.append(cov)
                    custom.append([ideal, diff, label])
            if not xs:
                continue
            color = SET2_PALETTE[i % len(SET2_PALETTE)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=model, showlegend=show_legend, legendgroup=model,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                customdata=custom,
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "PI: %{x}%<br>"
                    "Coverage: %{y:.1%}<br>"
                    "%{customdata[2]}"
                    "<extra></extra>"
                ),
            ), **rc)

    axis_style = dict(
        showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc",
    )

    if by_horizon:
        horizons = [0, 1, 2, 3]
        positions = {0: (1, 1), 1: (1, 2), 2: (2, 1), 3: (2, 2)}
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"Horizon: {h} week{'s' if h != 1 else ''}" for h in horizons],
            horizontal_spacing=0.1, vertical_spacing=0.12,
        )
        for h in horizons:
            row, col = positions[h]
            h_df = df_filt[df_filt["horizon"] == h]
            _add_lines(fig, h_df, row, col, show_legend=(h == 0))
        fig.update_xaxes(title_text="PI (%)", tickvals=_valid_levels, **axis_style)
        fig.update_yaxes(title_text="Coverage", tickformat=".0%", range=[-0.02, 1.08], **axis_style)
        height, width = 800, 850
    else:
        fig = make_subplots(rows=1, cols=1)
        _add_lines(fig, df_filt, 1, 1, show_legend=True)
        fig.update_xaxes(title_text="Prediction Interval (%)", tickvals=_valid_levels, **axis_style)
        fig.update_yaxes(title_text="Coverage", tickformat=".0%", range=[-0.02, 1.08], **axis_style)
        height, width = 800, 850

    fig.update_layout(
        **_BASE_LAYOUT,
        title_text="",   # see build_wis_boxplots
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd", borderwidth=1, font=dict(size=11)),
        # Title rendered as a page heading — see build_wis_boxplots.
        margin=dict(l=65, r=30, t=50, b=60),
        height=height,
        width=width,
    )
    return fig
