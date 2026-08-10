"""
app.py
Forecast Evaluation Dashboard

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from data_loader import (
    HUB_CONFIGS,
    HUB_LABELS,
    check_github_rate_limit,
    filter_scores,
    get_all_available_dates,
    get_model_list,
    load_forecasts_for_selection,
    load_locations,
    load_precomputed,
    load_truth_data,
)
from scoring import compute_coverage, compute_wis
from plots import (
    build_all_states_observed,
    build_all_states_panel,
    build_coverage_calibration,
    build_fan_chart,
    build_observed_chart,
    build_wis_boxplots,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecast Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Source+Sans+Pro:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif;
    color: #333333;
}
h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 600 !important;
    color: #111111 !important;
    font-size: 2rem !important;
    border-bottom: 2px solid #111111;
    padding-bottom: 0.3rem;
    margin-bottom: 0.2rem;
}
h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #111111 !important;
}
/* Tab lists */
[data-baseweb="tab-list"] {
    gap: 0px !important;
    border-bottom: 2px solid #dddddd !important;
    margin-top: 0.5rem !important;
}
/* Outer hub tabs — large bold */
[data-baseweb="tab"] {
    font-family: 'Source Sans Pro', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #666666 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 12px 32px !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
}
[data-baseweb="tab"] p,
[data-baseweb="tab"] span {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    text-transform: none !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #111111 !important;
    border-bottom: 3px solid #17B1BF !important;
    background: transparent !important;
}
/* Inner content tabs — small uppercase */
[role="tabpanel"] [data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 8px 22px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}
[role="tabpanel"] [data-baseweb="tab"] p,
[role="tabpanel"] [data-baseweb="tab"] span {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Forecast Evaluation Dashboard")

# ── Hub descriptions ───────────────────────────────────────────────────────────
_hub_descriptions = {
    "Flu Hospitalizations": (
        "This target tracks weekly incident influenza hospitalizations as reported through NHSN, "
        "evaluated at the national and state level with forecast horizons of 0–3 weeks ahead. "
        "Forecasts are submitted by modeling teams to the "
        "<a href='https://github.com/cdcepi/FluSight-forecast-hub' "
        "target='_blank' style='color:#17B1BF;'>CDC FluSight Forecast Hub</a>."
    ),
    "Flu ED Visits": (
        "This target tracks the weekly proportion of emergency department visits attributed to influenza, "
        "evaluated at the national and state level with forecast horizons of 0–3 weeks ahead. "
        "Forecasts are submitted by modeling teams to the "
        "<a href='https://github.com/cdcepi/FluSight-forecast-hub' "
        "target='_blank' style='color:#17B1BF;'>CDC FluSight Forecast Hub</a>."
    ),
    "COVID Hospitalizations": (
        "This target tracks weekly incident COVID-19 hospitalizations as reported through NHSN, "
        "evaluated at the national and state level with forecast horizons of 0–3 weeks ahead. "
        "Forecasts are submitted by modeling teams to the "
        "<a href='https://github.com/CDCgov/covid19-forecast-hub' "
        "target='_blank' style='color:#17B1BF;'>CDC COVID-19 Forecast Hub</a>."
    ),
    "Metrocast": (
        "This target tracks the percentage of emergency department visits attributed to influenza "
        "at the Health Service Area (HSA) level, providing sub-state geographic resolution "
        "with forecast horizons of 0–3 weeks ahead. "
        "Forecasts are submitted by modeling teams to the "
        "<a href='https://github.com/reichlab/flu-metrocast' "
        "target='_blank' style='color:#17B1BF;'>Reich Lab Flu Metrocast Hub</a>."
    ),
}


# ── Per-hub dashboard ──────────────────────────────────────────────────────────

def render_hub(selected_hub_label: str) -> None:
    hub = HUB_CONFIGS[selected_hub_label]

    st.markdown(
        f"<p style='color:#666; font-size:0.88rem; margin-top:4px; margin-bottom:20px;'>"
        f"{_hub_descriptions.get(selected_hub_label, '')}</p>",
        unsafe_allow_html=True,
    )

    # ── Load reference data ────────────────────────────────────────────────────
    with st.spinner("Loading reference data…"):
        locations_df = load_locations(selected_hub_label)
        all_models   = get_model_list(selected_hub_label)
        truth_df     = load_truth_data(selected_hub_label)

    loc_df       = locations_df.sort_values("location_name")
    loc_names    = loc_df["location_name"].tolist()
    loc_fips     = loc_df["location"].tolist()
    name_to_fips = dict(zip(loc_names, loc_fips))

    with st.spinner(""):
        eval_dates_pool = get_all_available_dates(selected_hub_label, all_models)

    _min_date = getattr(hub, "min_forecast_date", None)
    if _min_date:
        eval_dates_pool = [d for d in eval_dates_pool if d >= _min_date]

    def _season_label(d: str) -> str:
        ts = pd.Timestamp(d)
        start_yr = ts.year if ts.month >= 10 else ts.year - 1
        return f"{start_yr}-{str(start_yr + 1)[-2:]} season"

    season_dates: dict[str, list[str]] = {}
    for d in eval_dates_pool:
        ts = pd.Timestamp(d)
        if ts.month >= 10 or ts.month <= 5:
            label = _season_label(d)
            season_dates.setdefault(label, []).append(d)

    season_options = sorted(season_dates.keys(), reverse=True)

    # Reserve a fixed slot for the rate-limit warning *before* st.tabs below.
    # st.tabs tracks its active tab by element position on the client, so if the
    # number of elements preceding it changes between reruns, the tab widget is
    # remounted and the selection snaps back to the first tab. This container is
    # always created — whether or not the warning ends up inside it — which keeps
    # that position stable.
    rate_limit_slot = st.container()
    rl = check_github_rate_limit()
    if rl and rl["remaining"] < 10:
        rate_limit_slot.warning(
            f"GitHub API: **{rl['remaining']}** / {rl['limit']} calls remaining. "
            f"Resets {rl['reset_at'].strftime('%H:%M')}. Set `GITHUB_TOKEN` to raise limit."
        )

    # ── Inner tabs ─────────────────────────────────────────────────────────────
    tab_forecast, tab_wis, tab_coverage = st.tabs([
        "Forecast Plots",
        "WIS Evaluation",
        "Coverage Calibration",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 1 — Forecast Fan Chart
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_forecast:
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 2, 3, 2, 2])

        with ctrl1:
            default_loc_name = "US" if "US" in loc_names else loc_names[0]
            selected_loc_name = st.selectbox(
                "Location",
                options=loc_names,
                index=loc_names.index(default_loc_name) if default_loc_name in loc_names else 0,
                key=f"fc_location_{selected_hub_label}",
            )
            selected_location = name_to_fips[selected_loc_name]

        with ctrl2:
            available_dates = get_all_available_dates(selected_hub_label, all_models)
            if _min_date:
                available_dates = [d for d in available_dates if d >= _min_date]
            if available_dates:
                selected_forecast_date = st.selectbox(
                    "Forecast date",
                    options=list(reversed(available_dates)),
                    index=0,
                    key=f"fc_date_{selected_hub_label}",
                )
            else:
                selected_forecast_date = None
                st.caption("No dates found.")

        with ctrl3:
            obs_window_label = st.radio(
                "Historical window",
                options=["1 mo", "3 mo", "6 mo", "12 mo", "All"],
                index=1,
                horizontal=True,
                key=f"fc_obs_window_{selected_hub_label}",
            )
            obs_weeks = {"1 mo": 4, "3 mo": 13, "6 mo": 26, "12 mo": 52, "All": 520}[obs_window_label]

        with ctrl4:
            view_mode = st.radio(
                "View",
                options=["Single location", "All locations"],
                horizontal=False,
                key=f"fc_view_{selected_hub_label}",
            )

        with ctrl5:
            show_mode = st.radio(
                "Show",
                options=["Forecasts + data", "Data only"],
                index=0,
                horizontal=False,
                key=f"fc_show_mode_{selected_hub_label}",
            )

        has_forecast_dates = selected_forecast_date is not None

        fc_default_models = [m for m in ["FluSight-ensemble", "CovidHub-ensemble", "epiENGAGE-ensemble_mean"] if m in all_models] or all_models[:1]
        with st.expander("Models to include", expanded=False):
            fc_select_all = st.checkbox("Select all / none", value=False, key=f"fc_select_all_{selected_hub_label}")
            ncols_fc = 3
            fc_chk_cols = st.columns(ncols_fc)
            selected_models = [
                m for i, m in enumerate(all_models)
                if fc_chk_cols[i % ncols_fc].checkbox(
                    m,
                    value=True if (not fc_select_all and m in fc_default_models) else fc_select_all,
                    key=f"fc_chk_{selected_hub_label}_{m}",
                )
            ]
        if not selected_models:
            selected_models = fc_default_models

        if show_mode == "Data only":
            if view_mode == "Single location":
                obs_fig = build_observed_chart(
                    observed=truth_df,
                    location=selected_location,
                    location_name=selected_loc_name,
                    obs_weeks=obs_weeks,
                    y_label=hub.y_label,
                )
                st.plotly_chart(obs_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                with st.spinner("Building all-locations panel…"):
                    obs_panel_fig = build_all_states_observed(
                        observed=truth_df,
                        locations_df=locations_df,
                        obs_weeks=obs_weeks,
                    )
                st.plotly_chart(obs_panel_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            if not has_forecast_dates:
                st.info("No forecast dates found — showing data only.")
                obs_fig = build_observed_chart(
                    observed=truth_df,
                    location=selected_location,
                    location_name=selected_loc_name,
                    obs_weeks=obs_weeks,
                    y_label=hub.y_label,
                )
                st.plotly_chart(obs_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                prog1 = st.empty()
                with st.spinner(""):
                    forecasts_df = load_forecasts_for_selection(
                        hub_label=selected_hub_label,
                        models=selected_models,
                        ref_dates=[selected_forecast_date],
                        progress_placeholder=prog1,
                    )

                if forecasts_df.empty:
                    st.warning("No forecast data found — showing data only.")
                    obs_fig = build_observed_chart(
                        observed=truth_df,
                        location=selected_location,
                        location_name=selected_loc_name,
                        obs_weeks=obs_weeks,
                        y_label=hub.y_label,
                    )
                    st.plotly_chart(obs_fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    if view_mode == "Single location":
                        fan_fig = build_fan_chart(
                            forecasts=forecasts_df,
                            observed=truth_df,
                            location=selected_location,
                            location_name=selected_loc_name,
                            selected_models=selected_models,
                            ref_date=selected_forecast_date,
                            obs_weeks=obs_weeks,
                            y_label=hub.y_label,
                        )
                        st.plotly_chart(fan_fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        with st.spinner("Building all-locations panel…"):
                            panel_fig = build_all_states_panel(
                                forecasts=forecasts_df,
                                observed=truth_df,
                                locations_df=locations_df,
                                selected_models=selected_models,
                                ref_date=selected_forecast_date,
                                obs_weeks=obs_weeks,
                            )
                        st.plotly_chart(panel_fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("About this chart"):
            st.markdown(f"""
**Fan chart** shows **98%, 90%, and 50% prediction intervals** plus the **median line** for each
selected model on the chosen reference date.

Observed data is shown as a black line with filled dots up to the forecast date,
and open circles for subsequent weeks (data available after the forecast was submitted).

Target: `{hub.target}`
""")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 2 — WIS Evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_wis:
        wis_c1, wis_c2, wis_c3 = st.columns([2, 2, 1])

        with wis_c1:
            wis_loc_options = ["All locations"] + loc_names
            wis_loc_name = st.selectbox(
                "Location",
                options=wis_loc_options,
                index=0,
                key=f"wis_location_{selected_hub_label}",
            )
            wis_all_locs = wis_loc_name == "All locations"
            wis_location = None if wis_all_locs else name_to_fips[wis_loc_name]

        with wis_c2:
            scope_options = ["Selected forecast date"] + season_options
            default_idx = 1 if len(scope_options) > 1 else 0
            eval_scope = st.radio(
                "Evaluation scope",
                options=scope_options,
                index=default_idx,
                key=f"wis_eval_scope_{selected_hub_label}",
            )
            if eval_scope == "Selected forecast date":
                _wis_dates = get_all_available_dates(selected_hub_label, all_models)
                if _min_date:
                    _wis_dates = [d for d in _wis_dates if d >= _min_date]
                if _wis_dates:
                    wis_single_date = st.selectbox(
                        "Forecast date",
                        options=list(reversed(_wis_dates)),
                        index=0,
                        key=f"wis_single_date_{selected_hub_label}",
                    )
                    eval_ref_dates = [wis_single_date]
                else:
                    eval_ref_dates = []
            else:
                eval_ref_dates = season_dates.get(eval_scope, [])

        with wis_c3:
            wis_by_horizon = st.checkbox("Split by horizon", value=True, key=f"wis_by_horizon_{selected_hub_label}")
            wis_log = st.checkbox("Log scale", value=False, key=f"wis_log_{selected_hub_label}")

        wis_models_options = [m for m in all_models if m != hub.baseline]
        with st.expander("Models to include", expanded=False):
            select_all = st.checkbox("Select all / none", value=True, key=f"wis_select_all_{selected_hub_label}")
            ncols = 3
            chk_cols = st.columns(ncols)
            wis_models_shown = [
                m for i, m in enumerate(wis_models_options)
                if chk_cols[i % ncols].checkbox(m, value=select_all, key=f"wis_chk_{selected_hub_label}_{m}")
            ]
        if not wis_models_shown:
            wis_models_shown = wis_models_options

        wis_highlight = [m for m in hub.default_models if m in all_models]

        if not eval_ref_dates:
            st.info("No forecast dates in the selected evaluation range.")
        else:
            # Precomputed scores turn this into a filter. Scoring at request time
            # means fetching every model's forecasts for the whole range and
            # pivoting them in one DataFrame, which needs gigabytes for a full
            # season and is what used to exhaust the memory limit when deployed.
            precomputed_wis = load_precomputed(selected_hub_label, "wis")

            if not precomputed_wis.empty:
                wis_results = filter_scores(
                    precomputed_wis,
                    ref_dates=eval_ref_dates,
                    location=wis_location,
                )
            else:
                wis_load_models = sorted(set(all_models) | {hub.baseline})
                wis_cache_key = (selected_hub_label, tuple(wis_load_models), tuple(eval_ref_dates), wis_location)

                if st.session_state.get(f"_wis_cache_key_{selected_hub_label}") != wis_cache_key:
                    prog2 = st.empty()
                    with st.spinner(""):
                        fc_wis_raw = load_forecasts_for_selection(
                            hub_label=selected_hub_label,
                            models=wis_load_models,
                            ref_dates=eval_ref_dates,
                            progress_placeholder=prog2,
                        )
                    if not fc_wis_raw.empty:
                        fc_wis = fc_wis_raw.copy()
                        fc_wis["horizon"] = fc_wis["horizon"].astype(int)
                        if not wis_all_locs:
                            fc_wis = fc_wis[fc_wis["location"] == wis_location]
                        with st.spinner("Computing WIS scores…"):
                            wis_results = compute_wis(fc_wis, truth_df, baseline_model=hub.baseline)
                        st.session_state[f"_wis_results_{selected_hub_label}"] = wis_results
                    else:
                        st.session_state[f"_wis_results_{selected_hub_label}"] = pd.DataFrame()
                    st.session_state[f"_wis_cache_key_{selected_hub_label}"] = wis_cache_key

                wis_results = st.session_state.get(f"_wis_results_{selected_hub_label}", pd.DataFrame())

            if wis_results.empty:
                st.warning("No forecast data or no overlapping observations for the evaluation range.")
            else:
                plot_wis = wis_results[
                    (wis_results["model"] != hub.baseline) &
                    (wis_results["model"].isin(wis_models_shown))
                ].copy()

                st.caption(
                    f"{plot_wis['model'].nunique()} models · "
                    f"{'all locations' if wis_all_locs else wis_loc_name}"
                )
                wis_fig = build_wis_boxplots(
                    plot_wis,
                    highlight_models=wis_highlight,
                    log_scale=wis_log,
                    by_horizon=wis_by_horizon,
                    baseline_label=hub.baseline,
                )
                st.plotly_chart(wis_fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("About WIS ratio"):
            st.markdown(f"""
**Weighted Interval Score (WIS)** measures forecast accuracy combining sharpness and calibration.

**WIS ratio** = model WIS ÷ {hub.baseline} WIS.
- Ratio **< 1** → better than baseline
- Ratio **> 1** → worse than baseline
- Dotted line marks ratio = 1.0

Each panel shows one forecast horizon (0–3 weeks ahead).
Models are sorted by median WIS ratio (best at top). Highlighted models are shown in teal bold.
""")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 3 — Coverage Calibration
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_coverage:
        cov_c1, cov_c2, cov_c3 = st.columns([2, 2, 1])

        with cov_c1:
            cov_loc_options = ["All locations"] + loc_names
            cov_loc_name = st.selectbox(
                "Location",
                options=cov_loc_options,
                index=0,
                key=f"cov_location_{selected_hub_label}",
            )
            cov_all_locs = cov_loc_name == "All locations"
            cov_location = None if cov_all_locs else name_to_fips[cov_loc_name]

        with cov_c2:
            cov_scope_options = ["Selected forecast date"] + season_options
            cov_default_idx = 1 if len(cov_scope_options) > 1 else 0
            cov_eval_scope = st.radio(
                "Evaluation scope",
                options=cov_scope_options,
                index=cov_default_idx,
                key=f"cov_eval_scope_{selected_hub_label}",
            )
            if cov_eval_scope == "Selected forecast date":
                _cov_dates = get_all_available_dates(selected_hub_label, all_models)
                if _min_date:
                    _cov_dates = [d for d in _cov_dates if d >= _min_date]
                if _cov_dates:
                    cov_single_date = st.selectbox(
                        "Forecast date",
                        options=list(reversed(_cov_dates)),
                        index=0,
                        key=f"cov_single_date_{selected_hub_label}",
                    )
                    cov_eval_ref_dates = [cov_single_date]
                else:
                    cov_eval_ref_dates = []
            else:
                cov_eval_ref_dates = season_dates.get(cov_eval_scope, [])

        with cov_c3:
            cov_by_horizon = st.checkbox("Split by horizon", value=False, key=f"cov_by_horizon_{selected_hub_label}")

        cov_models_options = [m for m in all_models]
        cov_default = [m for m in hub.default_models if m in all_models]
        with st.expander("Models to include", expanded=False):
            cov_select_all = st.checkbox("Select all / none", value=False, key=f"cov_select_all_{selected_hub_label}")
            ncols_cov = 3
            cov_chk_cols = st.columns(ncols_cov)
            cov_models = [
                m for i, m in enumerate(cov_models_options)
                if cov_chk_cols[i % ncols_cov].checkbox(
                    m,
                    value=True if (not cov_select_all and m in cov_default) else cov_select_all,
                    key=f"cov_chk_{selected_hub_label}_{m}",
                )
            ]
        if not cov_models:
            cov_models = cov_default

        if not cov_eval_ref_dates:
            st.info("No forecast dates in the selected evaluation range.")
        else:
            precomputed_cov = load_precomputed(selected_hub_label, "coverage")

            if not precomputed_cov.empty:
                cov_results = filter_scores(
                    precomputed_cov,
                    ref_dates=cov_eval_ref_dates,
                    location=cov_location,
                    models=cov_models,
                )
            else:
                prog3 = st.empty()
                with st.spinner(""):
                    fc_cov_raw = load_forecasts_for_selection(
                        hub_label=selected_hub_label,
                        models=cov_models,
                        ref_dates=cov_eval_ref_dates,
                        progress_placeholder=prog3,
                    )

                if fc_cov_raw.empty:
                    cov_results = pd.DataFrame()
                else:
                    fc_cov = fc_cov_raw.copy()
                    fc_cov["horizon"] = fc_cov["horizon"].astype(int)
                    if not cov_all_locs:
                        fc_cov = fc_cov[fc_cov["location"] == cov_location]

                    with st.spinner("Computing coverage…"):
                        cov_results = compute_coverage(fc_cov, truth_df)

            if cov_results.empty:
                st.warning("Could not compute coverage — no overlapping forecasts and observations.")
            else:
                cov_fig = build_coverage_calibration(cov_results, cov_models, by_horizon=cov_by_horizon)
                st.plotly_chart(cov_fig, use_container_width=False, config={"displayModeBar": False})

        with st.expander("About coverage calibration"):
            st.markdown("""
A **perfectly calibrated** model's empirical coverage follows the diagonal **y = x** line.

- Points **above** the diagonal → intervals too wide (underconfident)
- Points **below** the diagonal → intervals too narrow (overconfident)

Coverage is averaged across all horizons and locations in the selected evaluation range.
""")


# ── Top-level hub tabs ─────────────────────────────────────────────────────────
st.markdown('<div id="hub-tabs-anchor"></div>', unsafe_allow_html=True)
hub_tabs = st.tabs(HUB_LABELS)
for hub_tab, hub_label in zip(hub_tabs, HUB_LABELS):
    with hub_tab:
        render_hub(hub_label)
