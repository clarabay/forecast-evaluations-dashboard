"""
app.py
Forecast Evaluation Dashboard

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd

from data_loader import (
    HUB_CONFIGS,
    HUB_LABELS,
    check_github_rate_limit,
    fetch_forecast,
    filter_scores,
    get_all_available_dates,
    get_model_list,
    load_forecasts_for_selection,
    load_locations,
    load_precomputed,
    load_truth_data,
)
from scoring import compute_coverage, compute_wis, summarize_coverage, summarize_wis
from plots import (
    TEAL,
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
h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #111111 !important;
}
/* Section headings above tables get a teal tick to tie them to the accent */
h5 {
    font-size: 1.05rem !important;
    border-left: 3px solid #17B1BF;
    padding-left: 10px !important;
    margin-bottom: 0.4rem !important;
}
/* No styling on the plotly or dataframe containers. Both measure their own
   container and resize to it, so borders, padding or overflow rules there feed
   back into the resize and the figures either misalign or grow without bound. */

/* Expanders — "Models to include" and the About panels */
[data-testid="stExpander"] {
    border: 1px solid #e4e4e0 !important;
    border-radius: 6px !important;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}
[data-testid="stExpander"] summary:hover {
    background: #f8f8f6;
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
/* Summary note box — used above the WIS and coverage plots */
.note-box {
    background: #f8f8f6;
    border: 1px solid #e4e4e0;
    border-left: 3px solid #17B1BF;
    border-radius: 4px;
    padding: 14px 18px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #444444;
}
.note-box code {
    background: #ecece9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 0.95em;
}
.note-box .note-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #111111;
    margin-bottom: 8px;
}
.note-box p { margin: 0 !important; }
/* Second paragraph is the evaluation results — split off from the description
   with a hairline so the two are not read as one block of prose. */
.note-box p + p {
    margin-top: 11px !important;
    padding-top: 11px;
    border-top: 1px solid #e0e0dc;
}
/* Acknowledgements footer */
.ack-rule {
    border: none;
    border-top: 1px solid #e4e4e0;
    margin: 40px 0 20px;
}
.ack {
    font-size: 0.82rem;
    line-height: 1.65;
    color: #6a6a6a;
}
.ack b {
    display: block;
    margin-bottom: 5px;
    font-size: 0.85rem;
    color: #333333;
    letter-spacing: 0.02em;
}
.ack a { color: #17B1BF; text-decoration: none; }
.ack a:hover { text-decoration: underline; }
.ack-logos { margin-top: 12px; }
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


# ── Epistorm model filter ──────────────────────────────────────────────────────

def _epistorm_available(hub, all_models: list[str]) -> list[str]:
    """
    Epistorm models for this hub that actually appear in its model list.

    The lists are per-target and are intersected rather than used directly: the
    flu names do not exist in the COVID hub, and only a couple of them submit
    the flu ED-visits target at all.
    """
    available = set(all_models)
    return [m for m in hub.epistorm_models if m in available]


def _apply_epistorm(prefix: str, hub_label: str, options: list[str], epistorm: list[str]) -> None:
    """
    Tick exactly the Epistorm models when checked, clear everything when not.

    Runs as a checkbox callback, which Streamlit executes before the script
    reruns, so writing the widget keys here means the per-model checkboxes pick
    the new state up when they are created. Driving the real checkboxes rather
    than overriding the resulting list keeps the expander an honest view of what
    is plotted, and leaves the selection editable afterwards.
    """
    on = st.session_state.get(f"{prefix}_epistorm_{hub_label}", False)
    wanted = set(epistorm) if on else set()
    for model in options:
        st.session_state[f"{prefix}_chk_{hub_label}_{model}"] = model in wanted
    # The selection is no longer "all", so don't leave that box claiming it is.
    st.session_state[f"{prefix}_select_all_{hub_label}"] = False


def _apply_select_all(prefix: str, hub_label: str, options: list[str]) -> None:
    """
    Tick or clear every per-model checkbox to match 'Select all / none'.

    Needed because passing value=select_all to the per-model checkboxes does not
    work: Streamlit ignores a widget's value once its key holds state, so the
    control silently stopped doing anything after the first interaction. Writing
    the keys from a callback is what actually moves them.
    """
    on = st.session_state.get(f"{prefix}_select_all_{hub_label}", False)
    for model in options:
        st.session_state[f"{prefix}_chk_{hub_label}_{model}"] = on
    # Selecting all (or none) is not the Epistorm subset either.
    st.session_state[f"{prefix}_epistorm_{hub_label}"] = False


def _select_all_checkbox(prefix: str, hub_label: str, options: list[str], default: bool) -> bool:
    """'Select all / none' that actually drives the per-model checkboxes."""
    return st.checkbox(
        "Select all / none",
        value=default,
        key=f"{prefix}_select_all_{hub_label}",
        on_change=_apply_select_all,
        args=(prefix, hub_label, options),
    )


def _epistorm_checkbox(prefix: str, hub_label: str, options: list[str], epistorm: list[str]) -> None:
    """Checkbox restricting the selection to this hub's Epistorm models."""
    st.checkbox(
        "Select Epistorm models only" if epistorm
        else "Select Epistorm models only (none for this target)",
        value=False,
        key=f"{prefix}_epistorm_{hub_label}",
        on_change=_apply_epistorm,
        args=(prefix, hub_label, options, epistorm),
        disabled=not epistorm,
    )


# ── Findings sentences for the note boxes ──────────────────────────────────────

def _note_box(description: str, findings: str = "") -> str:
    """The bordered note box: title, description, then the evaluation results."""
    body = f"<div class='note-title'>Summary</div><p>{description}</p>"
    if findings:
        body += f"<p>{findings}</p>"
    return f"<div class='note-box'>{body}</div>"


def _selection_label(scope: str, ref_dates: list[str], all_locations: bool, loc_name: str) -> str:
    """
    'the 2025-26 season (all locations)' — what the findings below actually cover.

    Named explicitly rather than left as "this selection" so a screenshot of the
    box still says which period and geography the numbers came from.
    """
    if scope == "Selected forecast date":
        when = f"forecast date {ref_dates[0]}" if ref_dates else "the selected forecast date"
    else:
        when = f"the {scope}"
    where = "all locations" if all_locations else loc_name
    return f"{when} ({where})"


def _fmt_value(v: float) -> str:
    """Counts and proportions share these tables, so pick decimals by magnitude."""
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:,.3g}"


def _forecast_findings(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    location: str,
    loc_name: str,
    models: list[str],
    ensemble_model: Optional[str],
    ref_date: Optional[str],
    unit_noun: str = "",
) -> str:
    """
    Prose describing the data and one representative forecast on the fan chart.

    One model at a time by design. With a single model selected it describes that
    model; with several it falls back to the hub's ensemble, rather than listing
    every model or aggregating them — the chart already shows the spread, and a
    per-model rundown would grow without bound as models are added.

    The anchor is the last observation from *before* the reference date — strictly
    before, because horizon 0 targets the reference date itself, so that week's
    value is something the forecast is predicting rather than something it could
    have seen. Anchoring on the newest observation in the data instead would
    compare a projection against values that postdate it entirely.
    """
    bits: list[str] = []
    obs = truth[truth["location"] == location].sort_values("date") if not truth.empty else truth
    ref_ts = pd.Timestamp(ref_date) if ref_date else None

    anchor_value = anchor_date = None
    if not obs.empty and ref_ts is not None:
        prior = obs[obs["date"] < ref_ts]
        if not prior.empty:
            anchor_value = float(prior["value"].iloc[-1])
            anchor_date = prior["date"].iloc[-1]

    if forecasts.empty:
        return ""

    single = len(models) == 1
    target = models[0] if single else ensemble_model
    if not target:
        return ""

    df = forecasts[(forecasts["location"] == location) &
                   (forecasts["model"].astype(str) == target)]
    if df.empty:
        if not single:
            bits.append(f"<b>{target}</b> has no forecast for this date.")
        return " ".join(bits)

    wide = df.pivot_table(index="horizon", columns="output_type_id", values="value",
                          aggfunc="first", observed=True)
    if 0.5 not in wide.columns or 3 not in wide.index:
        bits.append(f"<b>{target}</b> does not forecast three weeks ahead for this date.")
        return " ".join(bits)

    made_on = f" made on <b>{ref_date}</b>" if ref_date else ""
    bits.append(f"Showing the <b>{target}</b> forecast{made_on} for <b>{loc_name}</b>.")

    median = float(wide.loc[3, 0.5])
    unit = f" {unit_noun}" if unit_noun else ""
    sentence = f"It projects <b>{_fmt_value(median)}</b>{unit} in three weeks"

    if 0.025 in wide.columns and 0.975 in wide.columns:
        lo, hi = float(wide.loc[3, 0.025]), float(wide.loc[3, 0.975])
        sentence += f" (95% interval {_fmt_value(lo)} to {_fmt_value(hi)})"
    if anchor_value:
        pct = (median - anchor_value) / anchor_value * 100
        direction = "an increase" if pct >= 0 else "a decrease"
        sentence += (f", {direction} of {abs(pct):.0f}% from the last observation "
                     f"({_fmt_value(anchor_value)})")
    bits.append(sentence + ".")

    return " ".join(bits)


def _wis_findings(table: pd.DataFrame, selection: str) -> str:
    """One sentence describing what the WIS results for this selection actually show."""
    if table.empty:
        return ""

    n = len(table)
    better = int((table["Median WIS ratio"] < 1).sum())
    best = table.iloc[0]                      # table arrives sorted by median ratio
    bits = [
        f"For {selection}, <b>{better} of {n}</b> "
        f"{'model has' if n == 1 else 'models have'} a median WIS ratio below 1."
    ]

    # Three decimals throughout, matching the table — rounding to two would show
    # a value as 1.00 while the text describes it as below the baseline.
    lead = (f"<b>{best['Model']}</b> has the lowest median ratio "
            f"({best['Median WIS ratio']:.3f})")
    pw = table["Pairwise rel. WIS"]
    if pw.notna().any():
        best_pw = table.loc[pw.idxmin()]
        if best_pw["Model"] == best["Model"]:
            lead += (f" and also the best pairwise relative WIS "
                     f"({best_pw['Pairwise rel. WIS']:.3f}).")
        else:
            lead += (f", while <b>{best_pw['Model']}</b> leads on pairwise relative WIS "
                     f"({best_pw['Pairwise rel. WIS']:.3f}).")
    else:
        lead += "."
    bits.append(lead)

    return " ".join(bits)


def _coverage_findings(table: pd.DataFrame, selection: str) -> str:
    """One sentence describing what the coverage results for this selection actually show."""
    if table.empty or "95% Coverage" not in table.columns:
        return ""

    n = len(table)
    best = table.iloc[0]                      # table arrives sorted by calibration error
    # Three decimals, as in the table: at two, a 0.9455 reads as 0.95 and appears
    # to contradict the sentence counting it as below nominal.
    bits = [
        f"For {selection}, <b>{best['Model']}</b> is the best calibrated overall "
        f"({best['50% Coverage']:.3f} at the 50% level, {best['95% Coverage']:.3f} at 95%)."
    ]

    # Counted against a 10-point tolerance rather than any deviation at all: a
    # model landing on 0.945 is a rounding hair from nominal, not miscalibrated.
    deviation = table["95% Coverage"] - 0.95
    off = deviation.abs() > 0.10
    n_off = int(off.sum())

    if not n_off:
        bits.append(f"All {n} are within 10 percentage points of nominal at the 95% level.")
    else:
        bits.append(f"<b>{n_off} of {n}</b> {'is' if n_off == 1 else 'are'} more than 10 "
                    f"percentage points off nominal at the 95% level.")

    return " ".join(bits)


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
        # Note box on the right, as on the WIS and coverage tabs. The five controls
        # sit in two rows inside the left column: on one row they squeezed the date
        # select into an ellipsis and wrapped the radio labels onto three lines.
        fc_controls, fc_note = st.columns([6, 5])
        with fc_controls:
            ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.3, 2.0])

            with ctrl1:
                # Keyed on the location code, not the display name — metrocast's
                # names are long and set upstream. Without a configured default,
                # metrocast landed on whatever sorted first alphabetically.
                _code_to_name = {code: name for name, code in name_to_fips.items()}
                default_loc_name = (
                    _code_to_name.get(hub.default_location)
                    or ("US" if "US" in loc_names else loc_names[0])
                )
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

            ctrl4, ctrl5 = st.columns([1, 1])
            with ctrl4:
                # "Geographic scope" rather than "View": the options are about how
                # many locations are drawn, and it echoes "Evaluation scope" on the
                # other two tabs. Plain "Locations" would collide with the
                # Location select sitting right beside it.
                view_mode = st.radio(
                    "Geographic scope",
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

        fc_note_intro = (
            "The solid line is observed data up to the forecast date and the dashed line after "
            "it. Shaded bands are each model's 50%, 90% and 98% prediction intervals around its "
            "median."
        )
        with fc_note:
            fc_note_slot = st.empty()
        fc_note_slot.markdown(_note_box(fc_note_intro), unsafe_allow_html=True)

        has_forecast_dates = selected_forecast_date is not None

        # Taken from the hub config rather than matched against a list of known
        # ensemble names: the metrocast repo also carries a FluSight-ensemble
        # directory, so name-matching picked the wrong one there.
        fc_ensemble = hub.ensemble_model if hub.ensemble_model in all_models else None
        fc_default_models = [fc_ensemble] if fc_ensemble else all_models[:1]
        fc_epistorm = _epistorm_available(hub, all_models)
        with st.expander("Models to include", expanded=False):
            _epistorm_checkbox("fc", selected_hub_label, all_models, fc_epistorm)
            fc_select_all = _select_all_checkbox("fc", selected_hub_label, all_models, False)
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

                if not forecasts_df.empty:
                    # The note reports the ensemble whenever more than one model is
                    # plotted, so fetch it even when it is not itself selected —
                    # one extra cached file read rather than no summary at all.
                    _fc_for_note = forecasts_df
                    if (fc_ensemble and len(selected_models) != 1
                            and fc_ensemble not in selected_models):
                        _ens = fetch_forecast(selected_hub_label, fc_ensemble,
                                              selected_forecast_date)
                        if not _ens.empty:
                            _fc_for_note = pd.concat([forecasts_df, _ens], ignore_index=True)

                    fc_note_slot.markdown(
                        _note_box(fc_note_intro, _forecast_findings(
                            _fc_for_note, truth_df, selected_location, selected_loc_name,
                            selected_models, fc_ensemble, selected_forecast_date,
                            hub.unit_noun,
                        )),
                        unsafe_allow_html=True,
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
        # Selectors compressed into the left-hand columns, with the explanatory
        # text taking the freed space on the right so it sits above the boxplots.
        wis_c1, wis_c2, wis_c3, wis_note = st.columns([1.3, 1.5, 0.9, 3.3])

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

        # Written through a placeholder so the findings sentence can be appended
        # once the table below has been computed, without moving the box.
        wis_note_intro = (
            f"Models are evaluated by their WIS relative to the {hub.baseline}. A lower WIS "
            f"is better, and a WIS ratio below 1 indicates performance better than the baseline."
        )
        with wis_note:
            wis_note_slot = st.empty()
        wis_note_slot.markdown(_note_box(wis_note_intro), unsafe_allow_html=True)

        wis_models_options = [m for m in all_models if m != hub.baseline]
        # The baseline is excluded from the plot, so drop it from the Epistorm set
        # too — its WIS ratio is 1 by definition and it would just be a flat row.
        wis_epistorm = [m for m in _epistorm_available(hub, all_models) if m != hub.baseline]
        with st.expander("Models to include", expanded=False):
            _epistorm_checkbox("wis", selected_hub_label, wis_models_options, wis_epistorm)
            select_all = _select_all_checkbox("wis", selected_hub_label, wis_models_options, True)
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
                st.markdown("##### Forecast WIS ratio")
                wis_fig = build_wis_boxplots(
                    plot_wis,
                    highlight_models=wis_highlight,
                    log_scale=wis_log,
                    by_horizon=wis_by_horizon,
                    baseline_label=hub.baseline,
                )
                st.plotly_chart(wis_fig, use_container_width=True, config={"displayModeBar": False})

                wis_table = summarize_wis(plot_wis)
                if not wis_table.empty:
                    _findings = _wis_findings(
                        wis_table,
                        _selection_label(eval_scope, eval_ref_dates, wis_all_locs, wis_loc_name),
                    )
                    wis_note_slot.markdown(_note_box(wis_note_intro, _findings),
                                           unsafe_allow_html=True)
                    st.markdown("##### WIS Overview Table")
                    # Same models the boxplot picks out, styled to match. Number
                    # formatting moves into the Styler because column_config
                    # formats are not applied to a styled frame.
                    _hl = set(wis_highlight)

                    def _style_highlight(row):
                        css = f"color: {TEAL}; font-weight: 700" if row["Model"] in _hl else ""
                        return [css] * len(row)

                    # Constrained to the left two thirds — six narrow columns
                    # stretched across the full page left a lot of dead space.
                    wis_table_col, _ = st.columns([2, 1])
                    with wis_table_col:
                        st.dataframe(
                            wis_table.style
                            .apply(_style_highlight, axis=1)
                            .format({
                                "Median WIS ratio":  "{:.3f}",
                                "Pairwise rel. WIS": "{:.3f}",
                                "Beats baseline":    "{:.1f}%",
                                "Forecasts":         "{:,.0f}",
                            }, na_rep="—"),
                            use_container_width=True,
                            hide_index=True,
                        )

        with st.expander("About WIS ratio"):
            st.markdown(f"""
**Weighted Interval Score (WIS)** measures forecast accuracy combining sharpness and calibration.

**WIS ratio** = model WIS ÷ {hub.baseline} WIS.
- Ratio **< 1** → better than baseline
- Ratio **> 1** → worse than baseline
- Dotted line marks ratio = 1.0

Each panel shows one forecast horizon (0–3 weeks ahead).
Models are sorted by median WIS ratio (best at top). Highlighted models are shown in teal bold,
in both the plot and the summary table.

##### Summary table columns

**Median WIS ratio** — the middle WIS ratio across every forecast the model made in the
selected range. Robust to outliers, and what the rows are sorted by.

**Pairwise rel. WIS** — a like-for-like comparison against the other selected models, with
`{hub.baseline}` fixed at 1.0. For each pair of models it takes only the forecast tasks
*both* submitted, divides their mean WIS, then takes the geometric mean of those ratios
across all opponents and rescales so the baseline reads 1.0.

This matters because the other columns compare each model only to the baseline, over
whatever forecasts that model happened to make — and models cover very different amounts of
ground. In this hub no single forecast task was submitted by every model, so a plain average
ranks a model scored on a few hundred forecasts against one scored on thousands. Pairwise
comparison avoids that: every pair is judged on its own shared tasks, so a model with sparse
coverage no longer gets credit for having picked an easier slice. Expect it to disagree with
the median ratio — when it does, the pairwise value is the more trustworthy of the two.

Two caveats. It is **relative to the current selection**, so adding or removing models can
shift the numbers; and it is unstable for very small selections, since there are few opponents
to average over.

**Beats baseline** — the share of that model's individual forecasts that scored better than
the baseline. A model can post a strong median yet win only slightly more than half its
forecasts.

**Forecasts** — how many model-location-horizon-date combinations went into the row. Worth
checking before trusting any ranking: a low count means the model joined late, submits
intermittently, or covers only a few locations.
""")

    # ═══════════════════════════════════════════════════════════════════════════
    # Tab 3 — Coverage Calibration
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_coverage:
        # Same shape as the WIS tab: selectors squeezed left, explanation on the
        # right so it sits above the plot.
        cov_c1, cov_c2, cov_c3, cov_note = st.columns([1.3, 1.5, 0.9, 3.3])

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

        # Placeholder, as on the WIS tab, so the findings sentence can be appended
        # after the table is computed.
        cov_note_intro = (
            "Prediction interval coverage is the share of observations that fell inside a "
            "model's prediction interval. A well calibrated model tracks the diagonal "
            "(about 0.50 for the 50% coverage and 0.95 for the 95% coverage). <b>Above</b> "
            "the diagonal means intervals are too wide (underconfident); <b>below</b> means "
            "too narrow (overconfident)."
        )
        with cov_note:
            cov_note_slot = st.empty()
        cov_note_slot.markdown(_note_box(cov_note_intro), unsafe_allow_html=True)

        cov_models_options = [m for m in all_models]
        cov_default = [m for m in hub.default_models if m in all_models]
        cov_epistorm = _epistorm_available(hub, all_models)
        with st.expander("Models to include", expanded=False):
            _epistorm_checkbox("cov", selected_hub_label, cov_models_options, cov_epistorm)
            cov_select_all = _select_all_checkbox("cov", selected_hub_label, cov_models_options, False)
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
                cov_table = summarize_coverage(cov_results)
                if not cov_table.empty:
                    _cov_findings = _coverage_findings(
                        cov_table,
                        _selection_label(cov_eval_scope, cov_eval_ref_dates,
                                         cov_all_locs, cov_loc_name),
                    )
                    cov_note_slot.markdown(_note_box(cov_note_intro, _cov_findings),
                                           unsafe_allow_html=True)
                cov_plot_col, cov_table_col = st.columns([1.55, 1])

                with cov_plot_col:
                    st.markdown("##### Forecast Coverage")
                    cov_fig = build_coverage_calibration(cov_results, cov_models, by_horizon=cov_by_horizon)
                    st.plotly_chart(cov_fig, use_container_width=False, config={"displayModeBar": False})

                with cov_table_col:
                    if not cov_table.empty:
                        st.markdown("##### Coverage Overview Table")
                        # Highlighting only earns its keep when there is something to
                        # contrast against: if the table holds nothing but the default
                        # models, colouring every row teal says nothing.
                        _cov_defaults = set(cov_default)
                        _cov_shown = set(cov_table["Model"].astype(str))
                        _cov_hl = _cov_defaults if (_cov_shown - _cov_defaults) else set()

                        def _style_cov_highlight(row):
                            css = f"color: {TEAL}; font-weight: 700" if row["Model"] in _cov_hl else ""
                            return [css] * len(row)

                        st.dataframe(
                            cov_table.style
                            .apply(_style_cov_highlight, axis=1)
                            .format({
                                "50% Coverage": "{:.3f}",
                                "95% Coverage": "{:.3f}",
                                "Forecasts":    "{:,.0f}",
                            }, na_rep="—"),
                            use_container_width=True,
                            hide_index=True,
                        )

        with st.expander("About coverage calibration"):
            st.markdown("""
A **perfectly calibrated** model's empirical coverage follows the diagonal **y = x** line.

- Points **above** the diagonal → intervals too wide (underconfident)
- Points **below** the diagonal → intervals too narrow (overconfident)

Coverage is averaged across all horizons and locations in the selected evaluation range.

##### Summary table

**Ranking.** Rows are ordered by **mean absolute calibration error**: for each of the eleven
prediction intervals, the gap between the model's empirical coverage and the nominal level is
measured, the absolute values are averaged, and the smallest average ranks first. So rank 1 is
the best calibrated model overall, not the model with the highest coverage.

Two consequences worth keeping in mind. The ranking uses **all eleven intervals**, not just the
50% and 95% shown as columns — a model can be ranked above another whose two visible numbers
look better. And it is **direction-blind**: being 10 points too wide and 10 points too narrow
score identically, even though overconfidence is usually the more costly error. Read the plot
alongside the ranking to see which way a model misses.

**50% Coverage / 95% Coverage** — empirical coverage at those two prediction intervals.
Compare against 0.50 and 0.95.

**Forecasts** — how many model-location-horizon-date combinations went into the row. A low
count means sparse coverage, so treat that row's position with caution.
""")


# ── Top-level hub tabs ─────────────────────────────────────────────────────────
st.markdown('<div id="hub-tabs-anchor"></div>', unsafe_allow_html=True)
hub_tabs = st.tabs(HUB_LABELS)
for hub_tab, hub_label in zip(hub_tabs, HUB_LABELS):
    with hub_tab:
        render_hub(hub_label)


# ── Acknowledgements ───────────────────────────────────────────────────────────
# At module level, after the tab loop, so it renders once at the foot of the page
# rather than repeating inside all four hub tabs.

ASSET_DIR = Path(__file__).resolve().parent / "assets"
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}


def _logo_html(stem: str, height: int = 38) -> str:
    """
    Inline <img> for a logo in assets/, or "" when the file is not there.

    Embedded as a data URI rather than st.image so the logos sit inside the
    acknowledgement line at an exact height, and so a missing file degrades to
    nothing instead of a broken-image placeholder.
    """
    # Prefix match, so "northeastern-logo.png" or "northeastern_2024.svg" both work.
    for path in sorted(ASSET_DIR.glob(f"{stem}*")):
        mime = _MIME.get(path.suffix.lower())
        if not mime:
            continue
        data = base64.b64encode(path.read_bytes()).decode()
        return (f"<img src='data:{mime};base64,{data}' alt='{stem}' "
                f"style='height:{height}px; width:auto; margin-right:22px; "
                f"vertical-align:middle;'>")
    return ""


_logos = _logo_html("epistorm") + _logo_html("northeastern")

st.markdown(
    "<hr class='ack-rule'>"
    "<div class='ack'>"
    "This work is supported by CDC-RFA-FT-23-0069 from the CDC's Center for Forecasting and "
    "Outbreak Analytics. Contact: "
    "<a href='mailto:c.bay@northeastern.edu'>c.bay@northeastern.edu</a>."
    + (f"<div class='ack-logos'>{_logos}</div>" if _logos else "")
    + "</div>",
    unsafe_allow_html=True,
)
