#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_openmeteo.py — climate / water-stress drivers from the Open-Meteo ERA5
archive (free, no API key). Two modes:

  A) --sites_csv  : STANDALONE per-site monthly climate for any site list
                    (resumable, appends each site). Use for the big batch.
  B) --base       : enrich an existing GECO CSV (the 5 Red Sea sites) by merging
                    monthly climate onto it. Produces mangrove_enriched.csv.

Variables: air temp mean/max/min, VPD, ET0, soil moisture, shortwave radiation,
precipitation, and derived water_balance (precip - ET0).

Examples:
    python data/fetch_openmeteo.py --base data/processed/mangrove_all.csv \
        --out data/processed/mangrove_enriched.csv
    python data/fetch_openmeteo.py --sites_csv data/mangrove_sites_global.csv \
        --start 2014-01 --end 2024-12 --out data/processed/climate_global.csv
"""

import argparse
import os
import time
import json
import urllib.parse
import urllib.request

import pandas as pd

DAILY_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "vapour_pressure_deficit_max", "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean", "shortwave_radiation_sum", "precipitation_sum",
]
AGG = {
    "temperature_2m_mean": "mean", "temperature_2m_max": "mean",
    "temperature_2m_min": "mean", "vapour_pressure_deficit_max": "mean",
    "et0_fao_evapotranspiration": "sum", "soil_moisture_0_to_7cm_mean": "mean",
    "shortwave_radiation_sum": "sum", "precipitation_sum": "sum",
}
RENAME = {
    "temperature_2m_mean": "air_temp_mean", "temperature_2m_max": "air_temp_max",
    "temperature_2m_min": "air_temp_min", "vapour_pressure_deficit_max": "vpd",
    "et0_fao_evapotranspiration": "et0", "soil_moisture_0_to_7cm_mean": "soil_moist_era5",
    "shortwave_radiation_sum": "radiation", "precipitation_sum": "precip_era5",
}

DEFAULT_SITES = [  # (site_id, lat, lon)
    ("Al_Shabaan", 24.801613, 37.180562), ("Al_Shoaiba", 20.766092, 39.473378),
    ("Al_Wajh", 26.025006, 36.705946), ("Duba_Lake", 27.433590, 35.601723),
    ("Juzur_Janabiat", 19.787375, 40.647287),
]


def load_sites(sites_csv):
    if not sites_csv:
        return [(s, la, lo, "", "") for s, la, lo in DEFAULT_SITES]
    df = pd.read_csv(sites_csv)
    reg = df["region"] if "region" in df.columns else [""] * len(df)
    ctry = df["country"] if "country" in df.columns else [""] * len(df)
    return list(zip(df["site_id"], df["latitude"], df["longitude"], reg, ctry))


def fetch_site(lat, lon, start, end, retries=5):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "UTC"})
    url = "https://archive-api.open-meteo.com/v1/archive?" + q
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)["daily"]
        except Exception as e:
            print(f"    retry {k+1}: {e}")
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed for {lat},{lon}")


def monthly_climate(lat, lon, start, end):
    d = fetch_site(lat, lon, start, end)
    df = pd.DataFrame(d)
    df["ym"] = pd.to_datetime(df["time"]).dt.to_period("M")
    agg = df.groupby("ym").agg({v: AGG[v] for v in DAILY_VARS}).reset_index()
    agg = agg.rename(columns=RENAME)
    agg["water_balance"] = agg["precip_era5"] - agg["et0"]
    return agg


def run_standalone(args):
    sites = load_sites(args.sites_csv)
    start = args.start + "-01"
    end = (pd.Period(args.end, "M").to_timestamp("M")).strftime("%Y-%m-%d")
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["site_id"].unique())
        print(f"[INFO] resume: {len(done)} sites already in {args.out}")
    for sid, lat, lon, region, country in sites:
        if sid in done:
            print(f"[SKIP] {sid}"); continue
        print(f"[INFO] {sid} ({lat},{lon}) {region}")
        try:
            m = monthly_climate(lat, lon, start, end)
        except Exception as e:
            print(f"[FAIL] {sid}: {e}"); continue
        m["site_id"] = sid; m["latitude"] = lat; m["longitude"] = lon
        m["region"] = region; m["country"] = country
        m["date"] = m["ym"].dt.to_timestamp().dt.strftime("%-m/1/%Y")
        front = ["site_id", "date", "latitude", "longitude", "region", "country"]
        m = m[front + [c for c in m.columns if c not in front + ["ym"]]]
        m.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"    -> appended {len(m)} months")
        time.sleep(1)
    print(f"[DONE] {args.out}")


def run_base(args):
    base = pd.read_csv(args.base)
    base["ym"] = pd.to_datetime(base["date"], format="%m/%d/%Y", errors="coerce").dt.to_period("M")
    start = pd.to_datetime(base["date"], format="%m/%d/%Y").min().strftime("%Y-%m-%d")
    end = (pd.to_datetime(base["date"], format="%m/%d/%Y").max() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    parts = []
    for sid, g in base.groupby("site_id"):
        lat, lon = g["latitude"].iloc[0], g["longitude"].iloc[0]
        print(f"[INFO] {sid}")
        m = monthly_climate(lat, lon, start, end); m["site_id"] = sid
        parts.append(m); time.sleep(1)
    met = pd.concat(parts, ignore_index=True)
    merged = base.merge(met, on=["site_id", "ym"], how="left").drop(columns=["ym"])
    merged.to_csv(args.out, index=False)
    print(f"[INFO] wrote {merged.shape} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_csv", default=None)
    ap.add_argument("--base", default=None, help="existing GECO CSV to enrich")
    ap.add_argument("--start", default="2014-01", help="YYYY-MM (standalone mode)")
    ap.add_argument("--end", default="2024-12", help="YYYY-MM (standalone mode)")
    ap.add_argument("--out", default="data/processed/climate_global.csv")
    args = ap.parse_args()
    if args.base:
        run_base(args)
    else:
        run_standalone(args)


if __name__ == "__main__":
    main()
