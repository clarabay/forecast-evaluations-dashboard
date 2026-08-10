#!/usr/bin/env python
"""
Precompute WIS and coverage scores for every hub and write them to precomputed/.

Why this exists
---------------
The dashboard used to fetch every (model, forecast date) CSV and score it at
request time. On a 1 GB host that runs out of memory: the flu hub alone defaults
to ~95 models over a full season, and holding those forecasts in one DataFrame to
pivot costs multiple GB.

Scoring is a row-wise reduction over quantiles, so it does not need every model
in memory at once. This script scores one file at a time, keeps only the small
result, and writes a compact parquet the app can filter. Scored output is roughly
1/1500th the size of the raw forecasts it came from.

The output is committed to the repo, because a host with an ephemeral filesystem
would otherwise rebuild it on every cold start — which is the exact thing that
runs out of memory.

Usage
-----
    python scripts/build_scores.py                  # all hubs
    python scripts/build_scores.py --hub "Metrocast"
    python scripts/build_scores.py --limit-dates 5  # quick smoke run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_loader as dl          # noqa: E402
import scoring                    # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "precomputed"


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Store compact dtypes rather than object strings and float64.

    pyarrow round-trips a pandas categorical as a dictionary column, so writing
    these types here means the app reads them back directly. Converting on load
    instead works, but the object->category temporaries spike peak memory to
    several times the size of the data itself.
    """
    out = df.copy()
    for col in ("model", "location", "target"):
        if col in out.columns:
            out[col] = out[col].astype(str).astype("category")
    if "horizon" in out.columns:
        out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("int16")
    # Only the coverage indicators are narrowed: they hold 0.0, 1.0 and NaN, all
    # of which float32 represents exactly. The WIS columns stay float64 so the
    # published scores match what request-time scoring produced bit for bit —
    # float32 would shift them by ~6e-08 relative, which is harmless for the
    # plots but not worth an unexplainable discrepancy in a research figure.
    for col in out.columns:
        if col.endswith("_cov") and out[col].dtype == "float64":
            out[col] = out[col].astype("float32")
    return out


def build_hub(hub_label: str, limit_dates: int | None = None) -> dict:
    hub = dl.HUB_CONFIGS[hub_label]
    truth = dl.load_truth_data(hub_label)
    if truth.empty:
        return {"hub": hub_label, "status": "no truth data"}

    index = dl._model_output_index(hub_label)
    if not index:
        return {"hub": hub_label, "status": "no model index (rate limited?)"}

    # Always include the baseline: without it every wis_ratio is dropped.
    models = sorted(index)
    if hub.baseline not in models:
        print(f"  ! baseline {hub.baseline} missing from {hub_label}")

    tasks: list[tuple[str, str]] = []
    for model in models:
        dates = index[model]
        if hub.min_forecast_date:
            dates = [d for d in dates if d >= hub.min_forecast_date]
        if limit_dates:
            dates = sorted(dates)[-limit_dates:]
        tasks.extend((model, d) for d in dates)

    print(f"  {len(models)} models, {len(tasks)} (model, date) pairs")

    wis_parts: list[pd.DataFrame] = []
    cov_parts: list[pd.DataFrame] = []
    missing = 0
    t0 = time.time()

    for i, (model, date) in enumerate(tasks, 1):
        fc = dl.fetch_forecast(hub_label, model, date)
        if fc.empty:
            missing += 1
        else:
            fc = fc.copy()
            fc["horizon"] = fc["horizon"].astype(int)
            w = scoring.compute_wis_scores(fc, truth)
            if not w.empty:
                wis_parts.append(w)
            c = scoring.compute_coverage(fc, truth)
            if not c.empty:
                cov_parts.append(c)
        del fc

        if i % 250 == 0 or i == len(tasks):
            print(f"    {i}/{len(tasks)}  ({time.time() - t0:.0f}s, {missing} missing)")

    if not wis_parts:
        return {"hub": hub_label, "status": "nothing scored"}

    wis = pd.concat(wis_parts, ignore_index=True)
    del wis_parts
    # Ratio is derived here, from the small scored table rather than raw data.
    wis = scoring.add_wis_ratio(wis, baseline_model=hub.baseline)
    cov = pd.concat(cov_parts, ignore_index=True)
    del cov_parts

    OUT_DIR.mkdir(exist_ok=True)
    wis_path = OUT_DIR / f"{hub.cache_dir}_wis.parquet"
    cov_path = OUT_DIR / f"{hub.cache_dir}_coverage.parquet"
    compact(wis).to_parquet(wis_path, compression="zstd", index=False)
    compact(cov).to_parquet(cov_path, compression="zstd", index=False)

    return {
        "hub": hub_label,
        "status": "ok",
        "pairs": len(tasks),
        "missing": missing,
        "wis_rows": len(wis),
        "cov_rows": len(cov),
        "wis_mb": round(wis_path.stat().st_size / 1024 / 1024, 2),
        "cov_mb": round(cov_path.stat().st_size / 1024 / 1024, 2),
        "seconds": round(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", action="append", help="hub label; repeatable (default: all)")
    ap.add_argument("--limit-dates", type=int, default=None,
                    help="only the N most recent dates per model (for smoke runs)")
    args = ap.parse_args()

    hubs = args.hub or list(dl.HUB_CONFIGS)
    unknown = [h for h in hubs if h not in dl.HUB_CONFIGS]
    if unknown:
        print(f"unknown hub(s): {unknown}\navailable: {list(dl.HUB_CONFIGS)}")
        return 2

    results = []
    for hub_label in hubs:
        print(f"\n=== {hub_label} ===")
        results.append(build_hub(hub_label, args.limit_dates))

    print("\n=== summary ===")
    for r in results:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
