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


def compute_wis_scores(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
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
    obs = wide["observed"].values
    wis = np.zeros(len(wide))

    for q_lo, q_hi, alpha in _PI_PAIRS:
        if q_lo not in wide.columns or q_hi not in wide.columns:
            continue
        lo = wide[q_lo].values.astype(float)
        hi = wide[q_hi].values.astype(float)
        valid = ~(np.isnan(lo) | np.isnan(hi))
        dispersion = hi - lo
        underpred  = (2.0 / alpha) * np.maximum(lo - obs, 0)
        overpred   = (2.0 / alpha) * np.maximum(obs - hi, 0)
        wis += np.where(valid, (dispersion + underpred + overpred) * (alpha / 2.0), 0.0)

    if 0.5 in wide.columns:
        median = wide[0.5].values.astype(float)
        wis += 0.5 * np.abs(median - obs)

    wide["wis"] = wis / 11.5

    return wide[_GROUP_KEYS + ["wis"]].copy()


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

    baseline = scores[scores["model"] == baseline_model][_RATIO_KEYS + ["wis"]].rename(
        columns={"wis": "wis_baseline"}
    )
    df_ratio = scores.merge(baseline, on=_RATIO_KEYS, how="inner")
    df_ratio["wis_ratio"] = df_ratio["wis"] / df_ratio["wis_baseline"]
    return df_ratio


def compute_wis(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    baseline_model: str = "FluSight-baseline",
) -> pd.DataFrame:
    """
    Vectorized WIS and WIS ratio for every (model, location, horizon, reference_date, target_end_date).
    Returns DataFrame with columns: model, location, horizon, reference_date, target_end_date, wis, wis_ratio.
    """
    return add_wis_ratio(compute_wis_scores(forecasts, truth), baseline_model)


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
