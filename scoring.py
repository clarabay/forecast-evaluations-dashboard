"""
scoring.py
Fully vectorized WIS and coverage calculations.

Strategy: pivot forecasts wide (one row per forecast key, one column per quantile),
then compute all interval scores in a single numpy pass — no Python loops over rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Symmetric PI pairs: (q_lo, q_hi, alpha) — all 11 FluSight quantile pairs
_PI_PAIRS = [
    (0.01,  0.99,  0.02),
    (0.025, 0.975, 0.05),  # FluSight submits at 0.025/0.975
    (0.05,  0.95,  0.10),
    (0.10,  0.90,  0.20),
    (0.15,  0.85,  0.30),
    (0.20,  0.80,  0.40),
    (0.25,  0.75,  0.50),
    (0.30,  0.70,  0.60),
    (0.35,  0.65,  0.70),
    (0.40,  0.60,  0.80),
    (0.45,  0.55,  0.90),
]

_PI_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]

_GROUP_KEYS = ["model", "location", "horizon", "reference_date", "target_end_date"]


def _merge_with_truth(forecasts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_slim = (
        truth[["date", "location", "value"]]
        .rename(columns={"date": "target_end_date", "value": "observed"})
        .copy()
    )
    truth_slim["target_end_date"] = pd.to_datetime(truth_slim["target_end_date"])
    return forecasts.merge(truth_slim, on=["location", "target_end_date"], how="inner")


def _pivot_wide(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per (model, location, horizon, reference_date, target_end_date).
    Columns: group keys + quantile levels + 'observed'.
    """
    obs = merged[_GROUP_KEYS + ["observed"]].drop_duplicates(subset=_GROUP_KEYS)

    wide = merged.pivot_table(
        index=_GROUP_KEYS,
        columns="output_type_id",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide = wide.merge(obs, on=_GROUP_KEYS, how="left")
    return wide


_RATIO_KEYS = ["location", "horizon", "reference_date", "target_end_date"]


def _log_shift(x: np.ndarray, offset: float) -> np.ndarray:
    """log(offset + x) with negatives clipped to zero.

    Quantiles and counts should never be negative, but a model occasionally
    submits a small negative lower quantile, and the log of anything at or
    below zero is a nan or -inf that would silently void the whole row.

    The offset sets where the transform stops being logarithmic and goes
    roughly linear, so it belongs near the floor of the target's own scale:
    1 for admission counts, far smaller for a proportion or a percentage.
    """
    return np.log(np.clip(x, 0.0, None) + offset)


def _wis_terms(wide: pd.DataFrame, obs: np.ndarray, transform=None):
    """
    The three WIS components for one scale, each already divided by 11.5.

    They are accumulated separately rather than summed in place: weighted by
    alpha/2, the (2/alpha) cancels, so each interval contributes its raw
    shortfall to under/over and half its width times alpha to dispersion. All
    three carry the same 1/11.5, so they add back to the total exactly.

    `transform` applies to the quantiles and the observation alike before any
    arithmetic; it must be monotone, or the intervals would stop being ordered.
    """
    f = transform if transform is not None else (lambda x: x)
    o = f(obs)

    disp  = np.zeros(len(wide))
    under = np.zeros(len(wide))
    over  = np.zeros(len(wide))

    for q_lo, q_hi, alpha in _PI_PAIRS:
        if q_lo not in wide.columns or q_hi not in wide.columns:
            continue
        lo = f(wide[q_lo].values.astype(float))
        hi = f(wide[q_hi].values.astype(float))
        valid = ~(np.isnan(lo) | np.isnan(hi))
        disp  += np.where(valid, (hi - lo) * (alpha / 2.0), 0.0)
        under += np.where(valid, np.maximum(lo - o, 0), 0.0)
        over  += np.where(valid, np.maximum(o - hi, 0), 0.0)

    if 0.5 in wide.columns:
        median = f(wide[0.5].values.astype(float))
        # The median term is a pure miss, so it joins the side it missed on:
        # an observation below the median means the forecast ran high.
        med_gap = 0.5 * np.abs(median - o)
        over  += np.where(o < median, med_gap, 0.0)
        under += np.where(o > median, med_gap, 0.0)

    return disp / 11.5, under / 11.5, over / 11.5


def compute_wis_scores(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    log_offset: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorized WIS for every (model, location, horizon, reference_date, target_end_date).

    No baseline ratio, so this is safe to run on one model's file at a time —
    which is how the precomputed score cache is built without holding every
    model's forecasts in memory at once.

    Returns DataFrame with columns: group keys + wis.
    """
    if forecasts.empty or truth.empty:
        return pd.DataFrame()

    # Ensure horizons 0-3 only
    forecasts = forecasts[forecasts["horizon"].astype(int).between(0, 3)]

    if forecasts.empty:
        return pd.DataFrame()

    merged = _merge_with_truth(forecasts, truth)
    if merged.empty:
        return pd.DataFrame()

    wide = _pivot_wide(merged)
    obs = wide["observed"].values.astype(float)

    disp, under, over = _wis_terms(wide, obs)
    wide["wis_dispersion"]      = disp
    wide["wis_underprediction"] = under
    wide["wis_overprediction"]  = over
    wide["wis"] = disp + under + over

    # The same score on the log scale, with an offset matched to the target's
    # own units. Transforming the quantiles and the
    # observation before scoring (Bosse et al. 2023) turns WIS from an absolute
    # into a relative error measure, so a miss of 50 admissions counts the same
    # at a peak as in the off-season, and one blown-up location stops dominating
    # a model's average. Kept alongside the natural-scale score rather than
    # replacing it: they answer different questions and can rank differently.
    log_disp, log_under, log_over = _wis_terms(
        wide, obs, transform=lambda x: _log_shift(x, log_offset))
    wide["wis_log"] = log_disp + log_under + log_over

    return wide[_GROUP_KEYS + ["wis", "wis_log", "wis_dispersion",
                               "wis_underprediction", "wis_overprediction"]].copy()


def add_wis_ratio(
    scores: pd.DataFrame,
    baseline_model: str = "FluSight-baseline",
) -> pd.DataFrame:
    """
    Attach wis_baseline and wis_ratio to already-scored rows.

    Separate from compute_wis_scores because the ratio is the only part that
    needs more than one model: it is wis / wis_baseline over shared keys, so it
    can be derived from the small scored table rather than from raw forecasts.
    Rows with no matching baseline row are dropped, matching the original
    inner-join behaviour.
    """
    if scores.empty:
        return pd.DataFrame()

    # The log-scale score rides along so the pairwise comparison can be run on
    # either scale from the same table; scores built before it exists simply
    # do not carry the column.
    val_cols = ["wis"] + (["wis_log"] if "wis_log" in scores.columns else [])
    baseline = (scores[scores["model"] == baseline_model][_RATIO_KEYS + val_cols]
                .rename(columns={c: f"{c}_baseline" for c in val_cols}))
    df_ratio = scores.merge(baseline, on=_RATIO_KEYS, how="inner")
    df_ratio["wis_ratio"] = df_ratio["wis"] / df_ratio["wis_baseline"]
    return df_ratio


def compute_wis(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    baseline_model: str = "FluSight-baseline",
    log_offset: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorized WIS and WIS ratio for every (model, location, horizon, reference_date, target_end_date).
    Returns DataFrame with columns: model, location, horizon, reference_date, target_end_date, wis, wis_ratio.
    """
    return add_wis_ratio(compute_wis_scores(forecasts, truth, log_offset),
                         baseline_model)


def pairwise_relative_wis(scores: pd.DataFrame, value_col: str = "wis") -> pd.Series:
    """
    Pairwise relative WIS, one value per model, with the baseline at 1.0.

    `value_col` selects the scale: "wis" for the natural-scale score, "wis_log"
    for the same comparison run on log-transformed forecasts and observations.

    Averaging each model's own baseline ratios compares models on different sets
    of forecasts — no task at all is shared by every model in the flu hub, so
    rankings built that way are not like-for-like. This is the standard fix used
    by the forecast hubs: for every pair of models, restrict to the tasks they
    both submitted and take the ratio of their mean WIS; then take the geometric
    mean of those ratios across opponents, and divide by the baseline's value so
    the baseline reads 1.0. Each pair only needs its own overlap, so ragged
    coverage is tolerated.

    The baseline is reconstructed from the wis_baseline column, which lets this
    work on a selection that has the baseline's own row filtered out.

    Values below 1 beat the baseline. Returns an empty Series if the input has no
    usable overlap.
    """
    base_col = f"{value_col}_baseline"
    if scores.empty or value_col not in scores.columns or base_col not in scores.columns:
        return pd.Series(dtype=float)

    task_keys = _RATIO_KEYS
    wide = scores.pivot_table(index=task_keys, columns="model",
                              values=value_col, aggfunc="mean", observed=True)
    wide.columns = [str(c) for c in wide.columns]

    # The baseline's score per task, taken from any row for that task.
    base = (scores[task_keys + [base_col]]
            .drop_duplicates(subset=task_keys)
            .set_index(task_keys)[base_col])
    BASE = "__baseline__"
    wide[BASE] = base.reindex(wide.index)

    models = [c for c in wide.columns]
    mat = wide.to_numpy(dtype=float)
    col_of = {m: i for i, m in enumerate(models)}

    thetas: dict[str, float] = {}
    for m in models:
        i = col_of[m]
        logs = [0.0]          # theta against itself is 1; log(1) = 0
        for other in models:
            if other == m:
                continue
            j = col_of[other]
            both = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
            if not both.any():
                continue      # no shared tasks — this pair cannot be compared
            mean_i = mat[both, i].mean()
            mean_j = mat[both, j].mean()
            if mean_i <= 0 or mean_j <= 0:
                continue      # a zero mean has no logarithm
            logs.append(np.log(mean_i / mean_j))
        thetas[m] = float(np.exp(np.mean(logs))) if logs else float("nan")

    baseline_theta = thetas.get(BASE, float("nan"))
    if not baseline_theta or not np.isfinite(baseline_theta):
        return pd.Series(dtype=float)

    return pd.Series({m: t / baseline_theta for m, t in thetas.items()
                      if m != BASE}, dtype=float)


def summarize_wis(scores: pd.DataFrame) -> pd.DataFrame:
    """
    One row per model summarising the WIS scores currently in view.

    Expects the output of compute_wis / the precomputed table, already filtered
    to whatever the user selected. Sorted best-first by median WIS ratio.
    """
    if scores.empty:
        return pd.DataFrame()

    # observed=True matters: model is a categorical in the precomputed tables,
    # and without it every unused category comes back as an all-NaN row.
    g = scores.groupby("model", observed=True)
    out = pd.DataFrame({
        "Median WIS ratio": g["wis_ratio"].median(),
        # Percent, not a fraction, so the display format can render it as one.
        "Beats baseline":   g["wis_ratio"].apply(lambda s: (s < 1).mean() * 100),
        "Forecasts":        g["wis"].size(),
    })
    # Mapped onto the existing index rather than passed into the constructor: the
    # pairwise Series is keyed by plain strings while this index is categorical,
    # and letting pandas align the two loses the index name.
    out.insert(1, "Pairwise rel. WIS", out.index.astype(str).map(pairwise_relative_wis(scores)))
    # Only where the scores carry the log-scale column: tables precomputed
    # before it was added have no wis_log, and a column of dashes helps nobody.
    if "wis_log" in scores.columns and "wis_log_baseline" in scores.columns:
        out.insert(2, "Pairwise rel. WIS (log)",
                   out.index.astype(str).map(
                       pairwise_relative_wis(scores, value_col="wis_log")))
    out = out.sort_values("Median WIS ratio").reset_index()
    out = out.rename(columns={"model": "Model"})
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def summarize_coverage(scores: pd.DataFrame) -> pd.DataFrame:
    """
    One row per model giving empirical coverage against nominal levels.

    'Mean |error|' averages |empirical - nominal| over every prediction interval
    present, so it reads as a single calibration number: smaller is better
    calibrated, regardless of direction.
    """
    if scores.empty:
        return pd.DataFrame()

    g = scores.groupby("model", observed=True)
    headline = [50, 95]
    cols = {}
    for level in headline:
        col = f"{level}_cov"
        if col in scores.columns:
            cols[f"{level}% Coverage"] = g[col].mean()

    out = pd.DataFrame(cols)
    out["Forecasts"] = g.size()

    # Mean |empirical - nominal| over every interval. Not shown as a column, but
    # it is the ranking: it orders by overall calibration rather than by whichever
    # single interval happens to be leftmost.
    errors = []
    for level in _PI_LEVELS:
        col = f"{level}_cov"
        if col in scores.columns:
            errors.append((g[col].mean() - level / 100).abs())

    if errors:
        out["_mean_abs_error"] = pd.concat(errors, axis=1).mean(axis=1)
        out = out.sort_values("_mean_abs_error").drop(columns="_mean_abs_error").reset_index()
    else:
        out = out.sort_values(out.columns[0]).reset_index()
    out = out.rename(columns={"model": "Model"})
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def compute_coverage(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorized coverage at each PI level for every forecast key.
    Returns DataFrame with columns: group keys + 10_cov, 20_cov, ..., 98_cov.
    """
    if forecasts.empty or truth.empty:
        return pd.DataFrame()

    # Ensure horizons 0-3 only
    forecasts = forecasts[forecasts["horizon"].astype(int).between(0, 3)]

    if forecasts.empty:
        return pd.DataFrame()

    merged = _merge_with_truth(forecasts, truth)
    if merged.empty:
        return pd.DataFrame()

    wide = _pivot_wide(merged)
    obs = wide["observed"].values

    result = wide[_GROUP_KEYS].copy()

    for pi_level in _PI_LEVELS:
        q_lo = round(0.5 - pi_level / 200, 3)
        q_hi = round(0.5 + pi_level / 200, 3)
        if q_lo not in wide.columns or q_hi not in wide.columns:
            result[f"{pi_level}_cov"] = np.nan
            continue
        lo = wide[q_lo].values.astype(float)
        hi = wide[q_hi].values.astype(float)
        result[f"{pi_level}_cov"] = ((obs >= lo) & (obs <= hi)).astype(float)
        result.loc[np.isnan(lo) | np.isnan(hi), f"{pi_level}_cov"] = np.nan

    return result
