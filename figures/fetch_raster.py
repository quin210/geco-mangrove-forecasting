#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fetch a MODIS NDVI PIXEL GRID stack over time for a site (nadir canopy view)."""

import sys, os, time, json, urllib.request
import numpy as np
import pandas as pd

API = "https://modis.ornl.gov/rst/api/v1"
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# iconic sites: (id, lat, lon, km half-window)
TARGETS = {
    "Sundarbans": (21.95, 88.90, 4),
    "Carpentaria": (-16.00, 137.00, 4),
    "MekongDelta": (9.55, 106.30, 4),
}


def get(url):
    for k in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=90) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(min(60, 6 * 2 ** k))
    raise RuntimeError(url)


def dates(lat, lon, y0, y1):
    d = get(f"{API}/MOD13Q1/dates?latitude={lat}&longitude={lon}")
    return [e["modis_date"] for e in d["dates"] if y0 <= int(e["calendar_date"][:4]) <= y1]


def band(lat, lon, b, ds, km, chunk=10):
    rows, meta = [], None
    for i in range(0, len(ds), chunk):
        c = ds[i:i + chunk]
        d = get(f"{API}/MOD13Q1/subset?latitude={lat}&longitude={lon}"
                f"&startDate={c[0]}&endDate={c[-1]}&band={b}&kmAboveBelow={km}&kmLeftRight={km}")
        meta = (d["nrows"], d["ncols"])
        try: sc = float(d.get("scale"))
        except (TypeError, ValueError): sc = 1.0
        for s in d["subset"]:
            arr = pd.to_numeric(pd.Series(s["data"]), errors="coerce").to_numpy(float) * sc
            rows.append((s["calendar_date"], arr))
        time.sleep(0.5)
    return rows, meta


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "Sundarbans"
    y0, y1 = 2014, 2026
    lat, lon, km = TARGETS[which]
    print(f"[{which}] {lat},{lon} km={km} {y0}-{y1}")
    ds = dates(lat, lon, y0, y1)
    nd, meta = band(lat, lon, "250m_16_days_NDVI", ds, km)
    rel, _ = band(lat, lon, "250m_16_days_pixel_reliability", ds, km)
    nr, nc = meta
    rmap = {d: v for d, v in rel}

    recs = {}
    for date, arr in nd:
        q = rmap.get(date)
        good = np.ones_like(arr, bool) if q is None else ((q >= 0) & (q <= 1))
        v = np.where(good & np.isfinite(arr), arr, np.nan)
        recs[date] = v
    df = pd.DataFrame({"date": list(recs)})
    df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M")
    # monthly mean per pixel
    months, grids = [], []
    stack = {d: recs[d] for d in recs}
    tmp = pd.Series(list(recs.keys()))
    ym = pd.to_datetime(tmp).dt.to_period("M")
    for period, idx in pd.Series(range(len(tmp))).groupby(ym):
        arrs = np.stack([recs[tmp[i]] for i in idx.values])
        with np.errstate(all="ignore"):
            g = np.nanmean(arrs, axis=0).reshape(nr, nc)
        months.append(str(period)); grids.append(g)
    G = np.stack(grids)
    os.makedirs("figures/raster", exist_ok=True)
    out = f"figures/raster/{which}.npz"
    np.savez_compressed(out, grids=G, months=np.array(months), nrows=nr, ncols=nc,
                        lat=lat, lon=lon, km=km)
    print(f"saved {out}  stack={G.shape}  NDVI {np.nanmin(G):.2f}..{np.nanmax(G):.2f}")


if __name__ == "__main__":
    main()
