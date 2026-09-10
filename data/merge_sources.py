#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_sources.py — combine all self-collected (GEE-free) sources into one CSV.

  original  (data/processed/mangrove_all.csv)      salinity, SSH, currents, wind, area, ndvi
  + ERA5    (data/processed/mangrove_enriched.csv)  air temp, VPD, ET0, soil moisture, radiation, water_balance
  + MODIS   (data/processed/modis_ndvi.csv)         ndvi_modis, evi_modis, lst_day, lst_night

Output: data/processed/mangrove_multisource.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", default="data/processed/mangrove_enriched.csv")
    ap.add_argument("--modis", default="data/processed/modis_ndvi.csv")
    ap.add_argument("--out", default="data/processed/mangrove_multisource.csv")
    args = ap.parse_args()

    base = pd.read_csv(args.enriched)  # already original + ERA5 climate
    try:
        modis = pd.read_csv(args.modis)
        merged = base.merge(modis, on=["site_id", "date"], how="left")
        added = [c for c in modis.columns if c not in ("site_id", "date")]
        print(f"[INFO] merged MODIS cols: {added}")
        miss = merged[added].isna().sum()
        print("[INFO] missing after merge:\n", miss.to_string())
    except FileNotFoundError:
        print(f"[WARN] {args.modis} not found — writing ERA5-only merge.")
        merged = base

    merged.to_csv(args.out, index=False)
    print(f"[INFO] wrote {merged.shape[0]} rows x {merged.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
