#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_modis_ndvi.py — pull REAL MODIS time series WITHOUT Google Earth Engine,
via the ORNL DAAC MODIS/VIIRS Web Service (REST, no authentication).

    API: https://modis.ornl.gov/rst/api/v1/
    MOD13Q1 (250 m, 16-day): NDVI, EVI, pixel reliability
    MOD11A2 (1 km,  8-day):  LST day/night (satellite thermal driver)

Site-list driven + resumable: reads sites from a CSV (site_id, latitude,
longitude[, region, country]) and APPENDS each finished site to --out, so a big
multi-region batch can be interrupted and re-run (already-done sites are skipped).
ORNL rate-limits bursts (intermittent 500s) → chunked /dates + long backoff +
polite pacing.

Examples:
    # default 5 Red Sea sites
    python data/fetch_modis_ndvi.py --with_lst --out data/processed/modis_ndvi.csv
    # large multi-region batch
    python data/fetch_modis_ndvi.py --sites_csv data/mangrove_sites_global.csv \
        --with_lst --out data/processed/modis_global.csv
"""

import argparse
import os
import time
import json
import urllib.request

import numpy as np
import pandas as pd

API = "https://modis.ornl.gov/rst/api/v1"

DEFAULT_SITES = [
    ("Al_Shabaan", 24.801613, 37.180562),
    ("Al_Shoaiba", 20.766092, 39.473378),
    ("Al_Wajh", 26.025006, 36.705946),
    ("Duba_Lake", 27.433590, 35.601723),
    ("Juzur_Janabiat", 19.787375, 40.647287),
]


def load_sites(sites_csv):
    if not sites_csv:
        return [(s, la, lo, "", "") for s, la, lo in DEFAULT_SITES]
    df = pd.read_csv(sites_csv)
    reg = df["region"] if "region" in df.columns else [""] * len(df)
    ctry = df["country"] if "country" in df.columns else [""] * len(df)
    return list(zip(df["site_id"], df["latitude"], df["longitude"], reg, ctry))


# ORNL 500s the default "Python-urllib/x.y" User-Agent — send a browser UA.
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _get_once(url, retries=6):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            wait = min(60, 6 * (2 ** k))
            print(f"    retry {k+1}/{retries} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    return None


def get(url):
    # survive multi-hour ORNL outages: on persistent failure, wait for the
    # service to recover, then try once more before giving up on this call.
    d = _get_once(url)
    if d is not None:
        return d
    print("    persistent failure — waiting for ORNL to recover...", flush=True)
    wait_for_ornl()
    d = _get_once(url)
    if d is not None:
        return d
    raise RuntimeError(f"failed: {url}")


def wait_for_ornl(max_wait=7200, every=120):
    """Block until ORNL responds (it goes down for stretches). For unattended runs."""
    import time as _t
    waited = 0
    while True:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{API}/products", headers=_HEADERS), timeout=60)
            print("[INFO] ORNL is up.", flush=True)
            return
        except Exception as e:
            if waited >= max_wait:
                print(f"[WARN] ORNL still down after {waited}s, proceeding anyway.", flush=True)
                return
            print(f"[INFO] ORNL down, waiting {every}s ({waited}s so far): {e}", flush=True)
            _t.sleep(every); waited += every


def snap_series(ndvi_rows, rel_rows, lo=0.15, hi=0.90, min_valid=0.4):
    """
    'Snap' a windowed subset to mangrove-like pixels WITHOUT GMW: keep pixels
    whose long-term QA-masked mean NDVI sits in [lo, hi] (vegetated, mangrove-
    plausible) and that are valid often enough, then average those per date.
    Filters out water/bare/cloud pixels around a mis-placed coordinate.
    """
    rel_map = {d: v for d, v, _ in rel_rows}
    dates = [d for d, _, _ in ndvi_rows]
    P = len(ndvi_rows[0][1])
    M = np.full((len(ndvi_rows), P), np.nan)
    for ti, (d, arr, sc) in enumerate(ndvi_rows):
        q = rel_map.get(d)
        good = np.ones(P, bool) if q is None else ((q >= 0) & (q <= 1))
        vals = arr.astype(float) * sc
        vals = np.where(good & np.isfinite(vals), vals, np.nan)
        if len(vals) == P:
            M[ti] = vals
    valid_frac = np.mean(np.isfinite(M), axis=0)
    with np.errstate(all="ignore"):
        pmean = np.nanmean(M, axis=0)
    sel = (valid_frac >= min_valid) & (pmean >= lo) & (pmean <= hi)
    if sel.sum() == 0:
        sel = valid_frac >= min_valid          # fallback: any decently-valid pixel
    if sel.sum() == 0:
        sel = np.ones(P, bool)
    rec = {}
    for ti, d in enumerate(dates):
        v = M[ti, sel]
        if np.isfinite(v).any():
            rec[d] = float(np.nanmean(v))
    return rec, int(sel.sum()), P


def valid_dates(product, lat, lon, start_year, end_year):
    d = get(f"{API}/{product}/dates?latitude={lat}&longitude={lon}")
    return [e["modis_date"] for e in d["dates"]
            if start_year <= int(e["calendar_date"][:4]) <= end_year]


def fetch_band(product, lat, lon, band, dates, chunk=10, pause=0.6, km=0):
    rows = []
    for i in range(0, len(dates), chunk):
        c = dates[i:i + chunk]
        url = (f"{API}/{product}/subset?latitude={lat}&longitude={lon}"
               f"&startDate={c[0]}&endDate={c[-1]}&band={band}"
               f"&kmAboveBelow={km}&kmLeftRight={km}")
        d = get(url)
        try:
            scale = float(d.get("scale"))       # reliability band has scale='Not Available'
        except (TypeError, ValueError):
            scale = 1.0
        for s in d["subset"]:
            # ORNL returns 'Not Available' / fill for missing pixels -> NaN
            arr = pd.to_numeric(pd.Series(s["data"]), errors="coerce").to_numpy(dtype=float)
            rows.append((s["calendar_date"], arr, scale))
        time.sleep(pause)
    return rows


def to_series(rows, mask_rows=None, lo=None, hi=None):
    qmap = {d: v for d, v, _ in mask_rows} if mask_rows else None
    rec = {}
    for date, arr, sc in rows:
        if qmap is not None:
            q = qmap.get(date)
            m = (q >= lo) & (q <= hi) if q is not None else np.ones_like(arr, bool)
        else:
            m = np.isfinite(arr)
        if m.sum() == 0:
            continue
        rec[date] = float(np.nanmean(arr[m]) * sc)
    return rec


def monthly(rec, name):
    s = pd.Series(rec)
    df = pd.DataFrame({"calendar_date": s.index, name: s.values})
    df["dt"] = pd.to_datetime(df["calendar_date"])
    df["ym"] = df["dt"].dt.to_period("M")
    return df.groupby("ym")[name].mean()


def fetch_one_site(sid, lat, lon, region, country, start, end, with_lst,
                   with_evi=False, km=0, snap=False):
    vd = valid_dates("MOD13Q1", lat, lon, start, end)
    print(f"    MOD13Q1 dates: {len(vd)}", flush=True)
    ndvi = fetch_band("MOD13Q1", lat, lon, "250m_16_days_NDVI", vd, km=km)
    rel = fetch_band("MOD13Q1", lat, lon, "250m_16_days_pixel_reliability", vd, km=km)
    n_px = None
    if snap and km > 0:
        rec, n_sel, n_tot = snap_series(ndvi, rel)
        print(f"    snap: kept {n_sel}/{n_tot} mangrove-like pixels", flush=True)
        cols = {"ndvi_modis": monthly(rec, "ndvi_modis")}
        n_px = n_sel
    else:
        cols = {"ndvi_modis": monthly(to_series(ndvi, rel, 0, 1), "ndvi_modis")}
    if with_evi:
        evi = fetch_band("MOD13Q1", lat, lon, "250m_16_days_EVI", vd, km=km)
        cols["evi_modis"] = monthly(to_series(evi, rel, 0, 1), "evi_modis")
    if with_lst:
        vl = valid_dates("MOD11A2", lat, lon, start, end)
        print(f"    MOD11A2 dates: {len(vl)}")
        day = fetch_band("MOD11A2", lat, lon, "LST_Day_1km", vl)
        night = fetch_band("MOD11A2", lat, lon, "LST_Night_1km", vl)
        cols["lst_day"] = monthly(to_series(day), "lst_day") - 273.15
        cols["lst_night"] = monthly(to_series(night), "lst_night") - 273.15
    m = pd.DataFrame(cols).reset_index()
    m["site_id"] = sid
    m["latitude"] = lat
    m["longitude"] = lon
    m["region"] = region
    m["country"] = country
    m["date"] = m["ym"].dt.to_timestamp().dt.strftime("%-m/1/%Y")
    front = ["site_id", "date", "latitude", "longitude", "region", "country"]
    return m[front + [c for c in m.columns if c not in front + ["ym"]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_csv", default=None,
                    help="CSV with site_id,latitude,longitude[,region,country]; omit for 5 Red Sea sites")
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--with_lst", action="store_true")
    ap.add_argument("--with_evi", action="store_true")
    ap.add_argument("--km", type=int, default=0, help="half-window in whole km (snap needs >0)")
    ap.add_argument("--snap", action="store_true",
                    help="select mangrove-like pixels in the window (no GMW needed)")
    ap.add_argument("--wait_ornl", action="store_true",
                    help="wait for ORNL to come up before starting (unattended runs)")
    ap.add_argument("--cooldown", type=int, default=0)
    ap.add_argument("--out", default="data/processed/modis_ndvi.csv")
    args = ap.parse_args()

    if args.cooldown:
        print(f"[INFO] cooldown {args.cooldown}s...", flush=True)
        time.sleep(args.cooldown)
    if args.wait_ornl:
        wait_for_ornl()

    sites = load_sites(args.sites_csv)

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["site_id"].unique())
        print(f"[INFO] resume: {len(done)} sites already in {args.out}")

    for sid, lat, lon, region, country in sites:
        if sid in done:
            print(f"[SKIP] {sid} (already done)")
            continue
        print(f"[INFO] {sid} ({lat},{lon}) {region}")
        try:
            m = fetch_one_site(sid, lat, lon, region, country, args.start, args.end,
                               args.with_lst, args.with_evi, km=args.km, snap=args.snap)
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
            continue
        header = not os.path.exists(args.out)
        m.to_csv(args.out, mode="a", header=header, index=False)
        print(f"    -> appended {len(m)} months (NDVI {m.ndvi_modis.min():.3f}..{m.ndvi_modis.max():.3f})")

    print(f"[DONE] {args.out}")


if __name__ == "__main__":
    main()
