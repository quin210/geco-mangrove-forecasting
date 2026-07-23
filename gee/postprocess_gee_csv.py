#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
postprocess_gee_csv.py — turn the raw GEE export into a GECO-ready CSV.

Steps:
  - parse dates, sort by (site_id, date)
  - drop months with missing NDVI (clouds / no MODIS obs)
  - rebuild a contiguous per-site time_idx (1..T)
  - light interpolation of small gaps in driver columns
  - report per-site temporal coverage

Usage:
  python gee/postprocess_gee_csv.py --in mangrove_gee.csv --out data/processed/mangrove_gee_all.csv
"""

import argparse
import pandas as pd


DRIVER_COLS = [
    "lst_day", "lst_night", "precipitation", "soil_moisture", "sst",
    "wind_speed", "wind_direction_sin", "wind_direction_cos",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--max_gap", type=int, default=2,
                    help="max consecutive months to interpolate in driver columns")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    if df["dt"].isna().any():  # fallback for other date formats
        df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["site_id", "dt"]).reset_index(drop=True)

    before = len(df)
    df = df[df["ndvi"].notna()].copy()
    print(f"[INFO] dropped {before - len(df)} rows with missing NDVI")

    # interpolate short gaps in drivers, per site
    have = [c for c in DRIVER_COLS if c in df.columns]
    df[have] = (df.groupby("site_id")[have]
                  .apply(lambda g: g.interpolate(limit=args.max_gap, limit_area="inside"))
                  .reset_index(drop=True))

    # rebuild contiguous time_idx per site
    df["time_idx"] = df.groupby("site_id").cumcount() + 1

    print("[INFO] per-site coverage (months):")
    for sid, g in df.groupby("site_id"):
        print(f"    {sid:16s} {len(g):4d}  {g['dt'].min().date()} -> {g['dt'].max().date()}")

    out_cols = ["site_id", "date", "time_idx", "latitude", "longitude", "area",
                "ndvi"] + have
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(args.out, index=False)
    print(f"[INFO] wrote {len(df)} rows x {len(out_cols)} cols -> {args.out}")


if __name__ == "__main__":
    main()
