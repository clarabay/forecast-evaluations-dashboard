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

# Scores precomputed by scripts/build_scores.py and committed to the repo, so the
# app never has to hold a season of raw forecasts in memory to score them.
PRECOMPUTED_DIR = Path(__file__).resolve().parent / "precomputed"

# Delphi Epidata V5. Note this is a different host from the V4 API at
# api.delphi.cmu.edu — V5 lives here and V4 is being retired.
DELPHI_V5_SNAPSHOT = "https://delphi.cmu.edu/epidata/v5/snapshot/"
# V5 keeps NHSN revision history only from this date; earlier forecasts have no
# vintage to show, which is a normal outcome rather than an error. NSSP reaches
# further back (2024-04-18), but the ED-visits hub's first forecast date is
# 2025-11-22, so this single conservative floor excludes nothing.
DELPHI_MIN_SNAPSHOT = "2024-11-19"

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
    # Layout of that feed. The NHSN and NSSP datasets share nothing but the
    # host: different date and geography columns, abbreviations vs full state
    # names, counts vs percents, and NSSP needs a server-side filter because
    # it is county-level. Dispatch on this rather than parameterising all of it.
    socrata_kind: str = "nhsn"         # "nhsn" | "nssp"
    # Delphi V5 signal backing the "data as of the forecast date" overlay.
    # Only hubs with a signal here can offer the toggle.
    delphi_source: Optional[str] = None
    delphi_signal: Optional[str] = None
    # Multiplier taking the signal's units to the hub target's units. NSSP
    # publishes a percent while the ED-visits target is a proportion, so that
    # hub needs 0.01; the NHSN admission counts need none.
    delphi_scale: float = 1.0
    truth_target_filter: Optional[str] = None  # if truth CSV has multiple targets, filter to this
    default_models: list = field(default_factory=list)  # highlighted/default eval models
    min_forecast_date: Optional[str] = None  # earliest valid forecast date for this target
    epistorm_models: list = field(default_factory=list)  # Epistorm/Northeastern submissions
    unit_noun: str = ""    # e.g. "hospitalizations"; blank for proportion targets,
                           # where a bare number reads better than a unit phrase
    ensemble_model: Optional[str] = None  # named explicitly: several hubs carry a
                           # FluSight-ensemble directory regardless of target
    default_location: Optional[str] = None  # location code the fan chart opens on;
                           # falls back to US, then the first location


# Epistorm submissions differ by target: the flu model names carry FLUH/Flu
# suffixes and do not exist in the COVID hub, and metrocast only has the one.
_EPISTORM_FLU = [
    "MOBS-GLEAM_FLUH",
    "MIGHTE-Nsemble",
    "MIGHTE-Joint",
    "NU_UCSD-GLEAM_AI_FLUH",
    "CEPH-Rtrend_fluH",
    "NEU_ISI-FluBcast",
    "NEU_ISI-AdaptiveEnsemble",
    "MOBS-EpyStrain_Flu",
    "MOBS-GLEAM_RL_FLUH",
    "NU-PGF_FLUH",
    "Epistorm-Ensemble_Flu",
    "Gatech-ensemble_prob",
    "Gatech-ensemble_stat",
]

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
        unit_noun        = "hospitalizations",
        ensemble_model   = "FluSight-ensemble",
        socrata_id       = "mpgq-jmmr",
        socrata_col      = "totalconfflunewadm",
        delphi_source    = "nhsn",
        delphi_signal    = "confirmed_admissions_flu_ew",
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
        epistorm_models   = _EPISTORM_FLU,
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
        delphi_source      = "nssp",
        delphi_signal      = "pct_ed_visits_influenza",
        delphi_scale       = 0.01,   # NSSP percent -> hub proportion
        y_label            = "Proportion ED Visits",
        unit_noun          = "proportion ED visits",
        ensemble_model     = "FluSight-ensemble",
        # More current than the hub's target file, which trails by weeks out of
        # season. Same underlying NSSP data, so the two agree where they overlap.
        socrata_id         = "rdmq-nq56",
        socrata_col        = "percent_visits_influenza",
        socrata_kind       = "nssp",
        min_forecast_date  = "2025-11-22",
        default_models     = [
            "FluSight-baseline",
            "FluSight-ensemble",
            "MOBS-EpyStrain_Flu",
            "NEU_ISI-FluBcast",
        ],
        epistorm_models    = _EPISTORM_FLU,
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
        unit_noun        = "hospitalizations",
        ensemble_model   = "CovidHub-ensemble",
        socrata_id       = "ua7e-t2fy",
        socrata_col      = "totalconfcovidnewadm",
        delphi_source    = "nhsn",
        delphi_signal    = "confirmed_admissions_covid_ew",
        default_models   = [
            "CovidHub-baseline",
            "CovidHub-ensemble",
            "NEU_ISI-AdaptiveEnsemble",
        ],
        epistorm_models  = [
            "CEPH-Rtrend_covid",
            "NEU_ISI-AdaptiveEnsemble",
            "MOBS-GLEAM_COVID",
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
        unit_noun        = "percent of ED visits",
        ensemble_model   = "epiENGAGE-ensemble_mean",
        default_location = "boston",
        socrata_id       = None,
        truth_target_filter = "Flu ED visits pct",
        default_models   = [
            "epiENGAGE-baseline",
            "epiENGAGE-ensemble_mean",
            "MOBS-EpyStrain_Flu",
        ],
        epistorm_models  = ["MOBS-EpyStrain_Flu"],
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


# Set once if GitHub rejects the configured token, so the UI can say so.
_TOKEN_REJECTED = False


def token_rejected() -> bool:
    """True if a configured GITHUB_TOKEN was rejected during this run."""
    return _TOKEN_REJECTED


def _github_get(url: str, **kwargs) -> requests.Response:
    """
    GET the GitHub API, retrying without the token if the token is rejected.

    An expired or revoked token is worse than no token at all: GitHub answers
    401 to every request, while anonymous access still allows 60 an hour. Without
    this fallback a stale token silently empties the model and forecast-date
    lists, because every discovery call fails and the callers degrade to their
    defaults.
    """
    global _TOKEN_REJECTED
    headers = kwargs.pop("headers", None) or _github_headers()
    response = requests.get(url, headers=headers, **kwargs)

    if response.status_code == 401 and "Authorization" in headers:
        _TOKEN_REJECTED = True
        anonymous = {k: v for k, v in headers.items() if k != "Authorization"}
        response = requests.get(url, headers=anonymous, **kwargs)

    return response


# ── Delphi Epidata: versioned ("as of") observed data ──────────────────────────

# Why the last vintage fetch failed, so the UI can explain the gap rather than
# leaving the toggle looking broken. Set on every failure path.
_DELPHI_RATE_LIMITED = False
_DELPHI_LAST_ERROR: Optional[str] = None


class _DelphiUnavailable(Exception):
    """Raised inside the cached fetch so failures are never cached.

    st.cache_data stores whatever a function returns, including an empty frame,
    and this cache has ttl=None. Returning empty on a transient 429 would pin
    that emptiness for the life of the process and the toggle would stay dead
    long after the limit reset. Streamlit does not cache exceptions, so raising
    here means a failed fetch is retried on the next interaction.
    """


def delphi_rate_limited() -> bool:
    """True if a Delphi request was rate limited during this run."""
    return _DELPHI_RATE_LIMITED


def delphi_last_error() -> Optional[str]:
    """Human-readable reason the last vintage fetch failed, if it did."""
    return _DELPHI_LAST_ERROR


def _delphi_api_key() -> Optional[str]:
    """
    Delphi API key from the environment, falling back to Streamlit secrets.

    Guarded like _github_token(): touching st.secrets raises when no secrets file
    exists, which is the normal local case. Anonymous access works but is capped
    at 3 requests a minute, so the key matters in practice.
    """
    key = os.environ.get("DELPHI_EPIDATA_KEY")
    if key:
        return key
    try:
        return st.secrets.get("DELPHI_EPIDATA_KEY")
    except Exception:
        return None


# FluSight and the COVID hub both take submissions on the Wednesday before the
# Saturday reference date, so the vintage a forecaster actually saw is the
# Wednesday one. NHSN also publishes a Friday release, and asking the API for
# the reference Saturday silently picks that up: measured on three dates, the
# Saturday snapshot overstated the anchor week by up to 13% against the
# Wednesday snapshot the modeller had.
DELPHI_SUBMISSION_OFFSET_DAYS = 3


def delphi_snapshot_date(reference_date) -> str:
    """The vintage date for a forecast: the Wednesday before its reference Saturday."""
    d = pd.Timestamp(reference_date) - pd.Timedelta(days=DELPHI_SUBMISSION_OFFSET_DAYS)
    return d.strftime("%Y-%m-%d")


def _versioned_cache_path(hub: HubConfig, snapshot_date: str, geo_type: str) -> Path:
    return DISK_CACHE_DIR / hub.cache_dir / "asof" / f"{snapshot_date}_{geo_type}.parquet"


@st.cache_data(ttl=None, show_spinner=False)
def _load_versioned_truth_cached(hub_label: str, reference_date: str,
                                 geo_type: str = "nation") -> pd.DataFrame:
    """
    Observed data as published on the submission Wednesday before reference_date.

    Columns: date, location, value (matching load_truth_data) plus report_time,
    the vintage actually served. That last column matters: if no publication
    exists for the requested date the API resolves to an earlier one, and the
    caller must label the chart with what it got rather than what it asked for.

    ttl=None because a past vintage is immutable, unlike load_truth_data. Results
    are also written to parquet, which is what keeps the 3-requests-per-minute
    anonymous limit survivable across restarts.

    Raises _DelphiUnavailable on a fetch failure so the failure is not cached;
    the public wrapper below turns that into an empty frame for callers.
    """
    global _DELPHI_RATE_LIMITED
    hub = HUB_CONFIGS[hub_label]
    if not hub.delphi_signal or not reference_date:
        return pd.DataFrame()
    # Ask for the Wednesday the forecast was submitted, not its Saturday
    # reference date, which would include NHSN's Friday release.
    snapshot_date = delphi_snapshot_date(reference_date)
    # Before V5's history begins there is simply nothing to show.
    if snapshot_date < DELPHI_MIN_SNAPSHOT:
        return pd.DataFrame()

    cache_path = _versioned_cache_path(hub, snapshot_date, geo_type)
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    params = {
        "source": hub.delphi_source,
        # Singular. "signals" is rejected with HTTP 422.
        "signal": hub.delphi_signal,
        "geo_type": geo_type,
        "snapshot_date": snapshot_date,
    }
    # The key goes in the "token" header. V5 declares exactly two security
    # schemes, APIKeyHeader and APIKeyQuery, both named "token"; "api_key" is
    # not a parameter at all and any value for it fails validation with
    # HTTP 422 ("Extra inputs are not permitted"). The header form is used over
    # the query form so the key never lands in a URL or a proxy log.
    headers = {}
    key = _delphi_api_key()
    if key:
        headers["token"] = key.strip()

    try:
        r = requests.get(DELPHI_V5_SNAPSHOT, params=params,
                         headers=headers, timeout=60)
    except Exception as e:
        raise _DelphiUnavailable(f"could not reach the Delphi API ({type(e).__name__})")

    if r.status_code == 429:
        _DELPHI_RATE_LIMITED = True
        raise _DelphiUnavailable(
            "Delphi rate limit reached — anonymous access allows only 3 requests/minute, "
            "so set DELPHI_EPIDATA_KEY to lift it")
    if r.status_code != 200:
        raise _DelphiUnavailable(f"Delphi returned HTTP {r.status_code}")
    if not r.text.strip():
        raise _DelphiUnavailable("Delphi returned an empty response")

    try:
        raw = pd.read_csv(StringIO(r.text))
    except Exception:
        raise _DelphiUnavailable("Delphi response could not be parsed as CSV")

    needed = {"geo_value", "reference_time", "value", "report_time"}
    if raw.empty or not needed.issubset(raw.columns):
        raise _DelphiUnavailable("Delphi response was missing expected columns")

    # One signal can ship several imputation variants; keep the raw one so rows
    # are not silently duplicated per reference week.
    if "fill_method" in raw.columns:
        raw = raw[raw["fill_method"] == "source"]

    raw = raw.copy()
    raw["report_time"] = pd.to_datetime(raw["report_time"], errors="coerce", utc=True).dt.tz_localize(None)
    raw = raw.dropna(subset=["report_time"])

    # Guard against a silent fallback to current data. If the API ever answers a
    # vintage request with newer rows, plotting them would put the "as of" line
    # exactly on top of the observed line — a wrong chart that looks right.
    asked = pd.Timestamp(snapshot_date)
    raw = raw[raw["report_time"] <= asked]
    if raw.empty:
        raise _DelphiUnavailable(
            f"no vintage published on or before {snapshot_date}")

    # Delphi uses lowercase abbreviations ("ma") and "us"; the dashboard uses FIPS
    # and "US". Reuse the locations table rather than a second hand-rolled mapping.
    locs = load_locations(hub_label)
    if "abbreviation" not in locs.columns:
        raise _DelphiUnavailable("location table unavailable, so vintage rows "
                                 "could not be mapped to FIPS codes")

    out = pd.DataFrame({
        "date": pd.to_datetime(raw["reference_time"], errors="coerce"),
        "abbreviation": raw["geo_value"].astype(str).str.upper(),
        # float64 to match load_truth_data exactly; counts parse as int otherwise.
        "value": (pd.to_numeric(raw["value"], errors="coerce").astype("float64")
                  * hub.delphi_scale),
        "report_time": raw["report_time"],
    })
    out = out.merge(locs[["abbreviation", "location"]], on="abbreviation", how="left")
    # dropna rather than fillna(0): a location missing from a vintage must not be
    # drawn as zero, which would read as a collapse rather than as absent data.
    out = out.dropna(subset=["date", "location", "value"])
    out["location"] = out["location"].astype(str).apply(_normalize_fips)
    out = (out[["date", "location", "value", "report_time"]]
           .sort_values(["date", "location"]).reset_index(drop=True))

    if out.empty:
        raise _DelphiUnavailable(
            f"the {snapshot_date} vintage contained no usable rows")

    # Best effort. A read-only or full cache dir must not break the feature, so
    # the mkdir belongs inside the try alongside the write.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return out


def load_versioned_truth(hub_label: str, reference_date: str,
                         geo_type: str = "nation") -> pd.DataFrame:
    """Public entry point: never raises, and records why it came back empty.

    Deliberately uncached — the cache lives on the inner function so successes
    are kept forever (a past vintage is immutable) while failures are retried.
    """
    global _DELPHI_LAST_ERROR
    _DELPHI_LAST_ERROR = None
    try:
        return _load_versioned_truth_cached(hub_label, reference_date, geo_type)
    except _DelphiUnavailable as e:
        _DELPHI_LAST_ERROR = str(e)
    except Exception as e:
        _DELPHI_LAST_ERROR = f"unexpected error loading the vintage ({type(e).__name__})"
    return pd.DataFrame()


def check_github_rate_limit() -> dict | None:
    try:
        r = _github_get("https://api.github.com/rate_limit", timeout=5)
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

    Where a hub has a preliminary NHSN feed, weeks newer than the hub's own
    target file are appended to it. Two deliberate changes from the original:

    - The preliminary feed is consulted every day, not only on Wednesdays. The
      hub target file can trail it by weeks — out of season it stops updating
      altogether — so gating on the weekday meant the observed series jumped
      back and forth by over a month depending on which day it was opened.
    - Preliminary rows are appended rather than substituted for the whole
      series. The finalised history stays authoritative and only the weeks the
      hub has not published yet come from the preliminary feed, which is
      revised.
    """
    hub = HUB_CONFIGS[hub_label]
    official = _load_official_truth(hub)

    if not hub.socrata_id:
        return official

    prelim = _load_preliminary(hub, silent=True)
    if prelim.empty:
        return official
    if official.empty:
        return prelim

    official_max = official["date"].max()
    newer = prelim[prelim["date"] > official_max]
    if newer.empty:
        return official

    combined = pd.concat([official, newer], ignore_index=True)
    return combined.sort_values(["date", "location"]).reset_index(drop=True)


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


def _load_preliminary(hub: HubConfig, silent: bool = False) -> pd.DataFrame:
    """Latest published observations from CDC's Socrata feed for this hub."""
    if hub.socrata_kind == "nssp":
        return _load_preliminary_nssp(hub, silent=silent)
    return _load_preliminary_nhsn(hub, silent=silent)


def _load_preliminary_nssp(hub: HubConfig, silent: bool = False) -> pd.DataFrame:
    """
    State and national ED-visit percentages from the NSSP trajectories dataset.

    Three things differ from the NHSN feed. The dataset is county-level — 664k
    rows against 11k for the rollups — so county='All' is filtered server-side
    rather than downloaded and discarded. Geography is a full state name
    ("Kansas", "United States") instead of an abbreviation. And the values are
    percentages, while the hub target is a proportion.
    """
    if not hub.socrata_id or not hub.socrata_col:
        return pd.DataFrame()
    try:
        from sodapy import Socrata
        client = Socrata("data.cdc.gov", None)
        results = client.get(
            hub.socrata_id,
            select=f"week_end,geography,{hub.socrata_col}",
            where="county='All'",
            limit=50_000,
        )
        raw = pd.DataFrame.from_records(results)
    except Exception as e:
        if not silent:
            st.warning(f"Could not load preliminary NSSP data: {e}")
        return pd.DataFrame()

    if raw.empty or not {"week_end", "geography", hub.socrata_col}.issubset(raw.columns):
        return pd.DataFrame()

    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["week_end"], errors="coerce")
    # Percent -> proportion, matching the hub target's units.
    raw["value"] = pd.to_numeric(raw[hub.socrata_col], errors="coerce") / 100.0
    # locations.csv calls the national row "US", not "United States".
    raw["location_name"] = raw["geography"].replace({"United States": "US"})

    locs = pd.read_csv(f"{_FLUSIGHT_RAW}/auxiliary-data/locations.csv")
    locs["location"] = locs["location"].astype(str).apply(_normalize_fips)
    raw = raw.merge(locs[["location_name", "location"]], on="location_name", how="left")

    # dropna rather than fillna(0): an unmatched geography must not be drawn as
    # zero, which would read as a collapse rather than as absent data.
    raw = raw[["date", "location", "value"]].dropna()
    raw["location"] = raw["location"].astype(str).apply(_normalize_fips)
    return raw.sort_values(["date", "location"]).reset_index(drop=True)


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
        r = _github_get(f"{repo}/git/trees/{branch}",
                        params={"recursive": "1"}, timeout=30)
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception as e:
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


# ── Precomputed scores ─────────────────────────────────────────────────────────

def _precomputed_path(hub_label: str, kind: str) -> Path:
    return PRECOMPUTED_DIR / f"{HUB_CONFIGS[hub_label].cache_dir}_{kind}.parquet"


@st.cache_resource(show_spinner=False)
def load_precomputed(hub_label: str, kind: str) -> pd.DataFrame:
    """
    Read a precomputed score table ('wis' or 'coverage'), or an empty frame.

    Empty means the caller should fall back to scoring at request time — that
    path still works, it is just far more expensive.

    cache_resource rather than cache_data on purpose. cache_data serializes what
    it stores, which for these tables costs more memory than the tables
    themselves; cache_resource holds the object directly and shares one copy
    across sessions instead of one per user.

    The returned frame is shared, so treat it as read-only — filter_scores
    returns a copy whenever it actually filters, which every caller does.
    """
    path = _precomputed_path(hub_label, kind)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Could not read precomputed {kind} scores: {e}")
        return pd.DataFrame()

    if "reference_date" in df.columns:
        df["reference_date"] = pd.to_datetime(df["reference_date"])
    if "target_end_date" in df.columns:
        df["target_end_date"] = pd.to_datetime(df["target_end_date"])

    # scripts/build_scores.py already writes compact dtypes, so this is a no-op
    # for freshly built files and only fixes up older ones. Each branch checks
    # the dtype first: casting an already-categorical column via astype(str)
    # materialises a Python string per row, which on ~2M rows costs hundreds of
    # megabytes — more than the tables themselves.
    for col in ("model", "location", "target"):
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).astype("category")
    if "horizon" in df.columns and df["horizon"].dtype != "int16":
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("int16")
    # Coverage indicators only — see the note in scripts/build_scores.compact.
    for col in df.columns:
        if col.endswith("_cov") and df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


def filter_scores(
    scores: pd.DataFrame,
    ref_dates: list[str],
    location: Optional[str] = None,
    models: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Slice a precomputed table to the selection the user is looking at."""
    if scores.empty:
        return scores

    out = scores
    filtered = False
    if ref_dates:
        wanted = pd.to_datetime(pd.Series(list(ref_dates))).unique()
        out = out[out["reference_date"].isin(wanted)]
        filtered = True
    if location is not None:
        out = out[out["location"] == location]
        filtered = True
    if models is not None:
        out = out[out["model"].isin(set(models))]
        filtered = True
    # Only copy when this is a view over a subset; copying the whole table when
    # nothing was filtered would double peak memory for no reason.
    return out.copy() if filtered else out


# ── Model / date discovery ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_model_list(hub_label: str = "Flu Hospitalizations") -> list[str]:
    hub = HUB_CONFIGS[hub_label]

    index = _model_output_index(hub_label)
    if index is not None:
        return sorted(index)

    try:
        r = _github_get(f"{hub.api_base}/model-output", timeout=10)
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
        r = _github_get(f"{hub.api_base}/model-output/{model}", timeout=10)
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
        r = _github_get(url, timeout=20)
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
