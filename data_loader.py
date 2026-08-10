"""
data_loader.py
All data fetching for the forecast evaluation dashboard.

Cache priority:
  1. Local disk cache (~/.flusight_cache/) — survives restarts
  2. Streamlit in-session cache (@st.cache_data) — fast within a session
  3. GitHub raw / API — network fetch, saved to disk after first load
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

DISK_CACHE_DIR = Path.home() / ".flusight_cache"
MAX_WORKERS    = 8

_FLUSIGHT_RAW = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main"
_FLUSIGHT_API = "https://api.github.com/repos/cdcepi/FluSight-forecast-hub/contents"


# ── HubConfig dataclass ────────────────────────────────────────────────────────

@dataclass
class HubConfig:
    label: str              # display name
    raw_base: str           # raw GitHub URL base
    api_base: str           # GitHub API URL base
    target: str             # target string in forecast CSVs
    truth_file: str         # path within repo to truth CSV
    truth_cols: dict        # maps CSV col names → standard {date, location, value}
    baseline: str           # baseline model name for WIS ratio
    locations_source: str   # "flusight" or "metrocast"
    cache_dir: str          # subdirectory under DISK_CACHE_DIR
    y_label: str            # y-axis label for fan chart
    socrata_id: Optional[str] = None   # Socrata dataset ID for prelim truth
    socrata_col: Optional[str] = None  # column name in Socrata dataset
    truth_target_filter: Optional[str] = None  # if truth CSV has multiple targets, filter to this
    default_models: list = field(default_factory=list)  # highlighted/default eval models
    min_forecast_date: Optional[str] = None  # earliest valid forecast date for this target


HUB_CONFIGS: dict[str, HubConfig] = {
    "Flu Hospitalizations": HubConfig(
        label            = "Flu Hospitalizations",
        raw_base         = _FLUSIGHT_RAW,
        api_base         = _FLUSIGHT_API,
        target           = "wk inc flu hosp",
        truth_file       = "target-data/target-hospital-admissions.csv",
        truth_cols       = {"date": "date", "location": "location", "value": "value"},
        baseline         = "FluSight-baseline",
        locations_source = "flusight",
        cache_dir        = "flusight_hosp",
        y_label          = "Weekly Admissions",
        socrata_id       = "mpgq-jmmr",
        socrata_col      = "totalconfflunewadm",
        default_models   = [
            "MOBS-GLEAM_RL_FLUH",
            "MOBS-GLEAM_FLUH",
            "NEU_ISI-AdaptiveEnsemble",
            "MOBS-EpyStrain_Flu",
            "NEU_ISI-FluBcast",
            "FluSight-baseline",
            "FluSight-ensemble",
            "NU-PGF_FLUH",
            "NU_UCSD-GLEAM_AI_FLUH",
            "Epistorm-Ensemble_Flu",
        ],
    ),
    "Flu ED Visits": HubConfig(
        label              = "Flu ED Visits",
        raw_base           = _FLUSIGHT_RAW,
        api_base           = _FLUSIGHT_API,
        target             = "wk inc flu prop ed visits",
        truth_file         = "target-data/target-ed-visits-prop.csv",
        truth_cols         = {"date": "date", "location": "location", "value": "value"},
        baseline           = "FluSight-baseline",
        locations_source   = "flusight",
        cache_dir          = "flusight_ed",
        y_label            = "Proportion ED Visits",
        socrata_id         = None,
        min_forecast_date  = "2025-11-22",
        default_models     = [
            "FluSight-baseline",
            "FluSight-ensemble",
            "MOBS-EpyStrain_Flu",
            "NEU_ISI-FluBcast",
        ],
    ),
    "COVID Hospitalizations": HubConfig(
        label            = "COVID Hospitalizations",
        raw_base         = "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/main",
        api_base         = "https://api.github.com/repos/CDCgov/covid19-forecast-hub/contents",
        target           = "wk inc covid hosp",
        truth_file       = "target-data/covid-hospital-admissions.csv",
        truth_cols       = {"target_end_date": "date", "location": "location", "value": "value"},
        baseline         = "CovidHub-baseline",
        locations_source = "flusight",
        cache_dir        = "covid_hosp",
        y_label          = "Weekly Admissions",
        socrata_id       = "ua7e-t2fy",
        socrata_col      = "totalconfcovidnewadm",
        default_models   = [
            "CovidHub-baseline",
            "CovidHub-ensemble",
            "NEU_ISI-AdaptiveEnsemble",
        ],
    ),
    "Metrocast": HubConfig(
        label            = "Metrocast",
        raw_base         = "https://raw.githubusercontent.com/reichlab/flu-metrocast/main",
        api_base         = "https://api.github.com/repos/reichlab/flu-metrocast/contents",
        target           = "Flu ED visits pct",
        truth_file       = "target-data/latest-data.csv",
        truth_cols       = {"target_end_date": "date", "location": "location", "observation": "value"},
        baseline         = "epiENGAGE-baseline",
        locations_source = "metrocast",
        cache_dir        = "metrocast",
        y_label          = "% ED Visits (Flu)",
        socrata_id       = None,
        truth_target_filter = "Flu ED visits pct",
        default_models   = [
            "epiENGAGE-baseline",
            "epiENGAGE-ensemble_mean",
            "MOBS-EpyStrain_Flu",
        ],
    ),
}

HUB_LABELS = list(HUB_CONFIGS.keys())


# ── Legacy constants (kept for backward compat) ────────────────────────────────

EVAL_DEFAULT_MODELS = HUB_CONFIGS["Flu Hospitalizations"].default_models


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_fips(code: str) -> str:
    return code.zfill(2) if code.isdigit() else code


def _github_token() -> Optional[str]:
    """
    Read the API token from the environment, falling back to Streamlit secrets.

    Streamlit Cloud does surface entries from Secrets as environment variables,
    but reading st.secrets directly means a deployment works either way. Locally
    there is usually no secrets file at all, and touching st.secrets then raises,
    so the lookup is guarded.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("GITHUB_TOKEN")
    except Exception:
        return None


def _github_headers() -> dict:
    token = _github_token()
    h = {"Accept": "application/vnd.github.v3+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def check_github_rate_limit() -> dict | None:
    try:
        r = requests.get("https://api.github.com/rate_limit",
                         headers=_github_headers(), timeout=5)
        if r.status_code == 200:
            data = r.json()["rate"]
            return {
                "remaining": data["remaining"],
                "limit": data["limit"],
                "reset_at": pd.Timestamp(data["reset"], unit="s"),
            }
    except Exception:
        pass
    return None


# ── Locations ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=None, show_spinner=False)
def load_locations(hub_label: str = "Flu Hospitalizations") -> pd.DataFrame:
    hub = HUB_CONFIGS[hub_label]
    if hub.locations_source == "metrocast":
        url = f"{hub.raw_base}/auxiliary-data/locations.csv"
    else:
        url = f"{_FLUSIGHT_RAW}/auxiliary-data/locations.csv"
    df = pd.read_csv(url)
    df["location"] = df["location"].astype(str).apply(_normalize_fips)
    # Normalize column names — metrocast may differ
    df.columns = [c.strip().lower() for c in df.columns]
    # Ensure location_name exists (fall back to location)
    if "location_name" not in df.columns:
        if "location" in df.columns:
            df["location_name"] = df["location"]
    return df


# ── Truth / observed data ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_truth_data(hub_label: str = "Flu Hospitalizations") -> pd.DataFrame:
    """
    Returns observed data for the given hub.
    Schema: date (datetime64), location (str), value (numeric).
    For flu hosp only: checks Socrata prelim on Wednesdays.
    """
    hub = HUB_CONFIGS[hub_label]
    official = _load_official_truth(hub)
    official_max = official["date"].max() if not official.empty else pd.Timestamp.min

    if hub.socrata_id and pd.Timestamp.now().day_of_week == 2:  # Wednesday
        prelim = _load_preliminary_nhsn(hub, silent=True)
        if not prelim.empty and prelim["date"].max() > official_max:
            return prelim

    return official


def _load_official_truth(hub: HubConfig) -> pd.DataFrame:
    url = f"{hub.raw_base}/{hub.truth_file}"
    try:
        df = pd.read_csv(url, dtype={"location": str})
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]

    # Rename columns to standard schema
    rename = {}
    for src, dst in hub.truth_cols.items():
        src_lower = src.lower()
        if src_lower in df.columns and src_lower != dst:
            rename[src_lower] = dst
    if rename:
        df = df.rename(columns=rename)

    # Filter to specific target if needed (e.g. Metrocast time-series.csv has multiple)
    if hub.truth_target_filter and "target" in df.columns:
        df = df[df["target"] == hub.truth_target_filter].copy()

    if not {"date", "location", "value"}.issubset(df.columns):
        return pd.DataFrame()

    df = df[["date", "location", "value"]].copy()
    df["date"]     = pd.to_datetime(df["date"])
    df["location"] = df["location"].astype(str).apply(_normalize_fips)
    df["value"]    = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    return df.sort_values("date").reset_index(drop=True)


def _load_preliminary_nhsn(hub: HubConfig, silent: bool = False) -> pd.DataFrame:
    if not hub.socrata_id or not hub.socrata_col:
        return pd.DataFrame()
    try:
        from sodapy import Socrata
        client = Socrata("data.cdc.gov", None)
        results = client.get(hub.socrata_id, limit=100_000)
        raw = pd.DataFrame.from_records(results)
    except Exception as e:
        if not silent:
            st.warning(f"Could not load preliminary NHSN data: {e}")
        return pd.DataFrame()

    needed_cols = ["weekendingdate", "jurisdiction", hub.socrata_col]
    missing = [c for c in needed_cols if c not in raw.columns]
    if missing:
        return pd.DataFrame()

    raw = raw[needed_cols].copy()
    raw["date"]  = pd.to_datetime(raw["weekendingdate"])
    raw["value"] = pd.to_numeric(raw[hub.socrata_col], errors="coerce").fillna(0).astype(int)
    raw["jurisdiction"] = raw["jurisdiction"].apply(lambda x: "US" if x == "USA" else x)

    locs = pd.read_csv(f"{_FLUSIGHT_RAW}/auxiliary-data/locations.csv")
    locs["location"] = locs["location"].astype(str).apply(_normalize_fips)
    raw = raw.merge(locs[["abbreviation", "location"]], left_on="jurisdiction",
                    right_on="abbreviation", how="left")
    raw = raw[["date", "location", "value"]].dropna(subset=["location"])
    raw["location"] = raw["location"].astype(str).apply(_normalize_fips)
    return raw.sort_values("date").reset_index(drop=True)


# ── Model / date discovery ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _repo_tree(repo: str, branch: str) -> Optional[list[dict]]:
    """
    Every path in a repo, from one recursive trees request.

    Cached on the repo rather than the hub so that two hubs backed by the same
    repository (flu hospitalisations and flu ED visits both live in
    FluSight-forecast-hub) share a single request.

    Returns None when the request fails or the tree came back truncated — a
    partial tree would silently hide models and dates, so it is treated as a
    miss rather than trusted.
    """
    try:
        r = requests.get(f"{repo}/git/trees/{branch}",
                         params={"recursive": "1"},
                         headers=_github_headers(), timeout=30)
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception:
        return None

    if payload.get("truncated"):
        return None
    return payload.get("tree", [])


@st.cache_data(ttl=3600, show_spinner=False)
def _model_output_index(hub_label: str) -> Optional[dict[str, list[str]]]:
    """
    Map {model: [forecast date, ...]} for a hub in a single API request.

    The contents API costs one request per model directory — around 110 across
    the four hubs, which on its own exceeds the 60 req/hour unauthenticated
    limit. One recursive trees call returns every path in the repo instead.

    Returns None when the tree is unavailable or nothing matched, so callers can
    fall back to per-directory discovery.
    """
    hub    = HUB_CONFIGS[hub_label]
    repo   = hub.api_base.rsplit("/contents", 1)[0]
    branch = hub.raw_base.rstrip("/").rsplit("/", 1)[-1]

    tree = _repo_tree(repo, branch)
    if tree is None:
        return None

    index: dict[str, set[str]] = {}
    for item in tree:
        if item.get("type") != "blob":
            continue
        parts = item.get("path", "").split("/")
        if len(parts) != 3 or parts[0] != "model-output":
            continue
        model, name = parts[1], parts[2]
        if name.endswith(".csv") and len(name) >= 10:
            index.setdefault(model, set()).add(name[:10])

    if not index:
        return None
    return {model: sorted(dates) for model, dates in index.items()}


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_list(hub_label: str = "Flu Hospitalizations") -> list[str]:
    hub = HUB_CONFIGS[hub_label]

    index = _model_output_index(hub_label)
    if index is not None:
        return sorted(index)

    try:
        r = requests.get(f"{hub.api_base}/model-output",
                         headers=_github_headers(), timeout=10)
        if r.status_code == 200:
            return sorted(item["name"] for item in r.json() if item["type"] == "dir")
        if r.status_code == 403:
            st.warning("GitHub rate limit reached — showing default model list.")
    except Exception:
        pass
    return sorted(hub.default_models) if hub.default_models else []


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_dates(hub_label: str, model: str) -> list[str]:
    hub = HUB_CONFIGS[hub_label]
    try:
        r = requests.get(f"{hub.api_base}/model-output/{model}",
                         headers=_github_headers(), timeout=10)
        if r.status_code == 200:
            dates = []
            for item in r.json():
                name = item.get("name", "")
                if name.endswith(".csv") and len(name) >= 10:
                    dates.append(name[:10])
            return sorted(set(dates))
    except Exception:
        pass
    return []


def get_all_available_dates(hub_label: str, models: list[str]) -> list[str]:
    index = _model_output_index(hub_label)
    if index is not None:
        wanted = set(models)
        dates: set[str] = set()
        for model, model_dates in index.items():
            if model in wanted:
                dates.update(model_dates)
        return sorted(dates)

    all_dates: set[str] = set()
    for model in models:
        all_dates.update(get_model_dates(hub_label, model))
    return sorted(all_dates)


# ── Forecast fetching ──────────────────────────────────────────────────────────

def _disk_cache_path(hub: HubConfig, model: str, date_str: str) -> Path:
    return DISK_CACHE_DIR / hub.cache_dir / model / f"{date_str}.parquet"


@st.cache_data(ttl=None, show_spinner=False)
def fetch_forecast(hub_label: str, model: str, date_str: str) -> pd.DataFrame:
    """
    Fetch quantile forecasts — disk cache first, then GitHub.
    Returns empty DataFrame on failure.
    """
    hub = HUB_CONFIGS[hub_label]
    cache_path = _disk_cache_path(hub, model, date_str)

    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = f"{hub.raw_base}/model-output/{model}/{date_str}-{model}.csv"
    try:
        r = requests.get(url, headers=_github_headers(), timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text), dtype={"location": str})
    except Exception:
        return pd.DataFrame()

    # Filter to the target and quantile output type for this hub
    df = df[
        (df["output_type"] == "quantile") &
        (df["target"] == hub.target)
    ].copy()

    if df.empty:
        return df

    df["model"]          = model
    df["reference_date"]  = pd.to_datetime(df["reference_date"])
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    df["location"]        = df["location"].astype(str).apply(_normalize_fips)
    df["output_type_id"]  = pd.to_numeric(df["output_type_id"], errors="coerce")
    df["horizon"]         = pd.to_numeric(df["horizon"], errors="coerce")
    df["value"]           = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["output_type_id", "value"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return df


def load_forecasts_for_selection(
    hub_label: str,
    models: list[str],
    ref_dates: list[str],
    progress_placeholder=None,
) -> pd.DataFrame:
    """Parallel fetch for all (model, date) combinations."""
    tasks = [(m, d) for m in models for d in ref_dates]
    if not tasks:
        return pd.DataFrame()

    results: list[pd.DataFrame] = []
    completed = 0
    total = len(tasks)

    if progress_placeholder is not None:
        progress_bar = progress_placeholder.progress(0, text="Loading forecasts…")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(fetch_forecast, hub_label, model, date): (model, date)
            for model, date in tasks
        }
        for future in as_completed(future_to_task):
            df = future.result()
            if not df.empty:
                results.append(df)
            completed += 1
            if progress_placeholder is not None:
                pct = completed / total
                progress_bar.progress(pct, text=f"Loading forecasts… {completed}/{total}")

    if progress_placeholder is not None:
        progress_placeholder.empty()

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
