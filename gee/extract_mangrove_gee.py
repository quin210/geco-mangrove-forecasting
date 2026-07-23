#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_mangrove_gee.py
=======================
Extract a monthly, per-site environmental time series for mangrove sites using
Google Earth Engine, matching (and extending) the GECO `mangrove_all.csv`
schema. Adds the currently-missing thermal-stress driver **LST** (land surface
temperature, day & night) needed for the physics-informed forecasting paper.

Output columns (CSV, one row per site-month):
    site_id, date, time_idx, latitude, longitude, area,
    ndvi, lst_day, lst_night,            # <- LST is the new thermal driver
    precipitation, soil_moisture, sst,
    wind_speed, wind_direction_sin, wind_direction_cos

Sites can be supplied three ways (see --site_mode):
    points : a CSV/inline list of (site_id, lon, lat)   [default; reproduces the 5 Red Sea sites]
    gmw    : auto-generate sites from Global Mangrove Watch extent within a bbox
    asset  : an Earth Engine FeatureCollection asset of mangrove polygons

Earth Engine data sources (all free):
    NDVI              MODIS/061/MOD13Q1     (16-day, 250 m)
    LST day/night     MODIS/061/MOD11A2     (8-day, 1 km)
    Precipitation     UCSB-CHG/CHIRPS/DAILY (daily, ~5 km)
    Soil moisture     ECMWF/ERA5_LAND/MONTHLY_AGGR (volumetric soil water L1)
    SST               NOAA/CDR/OISST/V2_1   (daily, 0.25 deg)
    Wind (u/v 10m)    ECMWF/ERA5_LAND/MONTHLY_AGGR

Setup:
    pip install earthengine-api
    earthengine authenticate           # one-time browser login
    python gee/extract_mangrove_gee.py --project YOUR_GEE_CLOUD_PROJECT \
        --start 2014-01 --end 2024-12 --site_mode points

The export is written to your Google Drive (folder set by --drive_folder). Poll
the task in the GEE Code Editor 'Tasks' tab or with the printed task id.
"""

import argparse
import ee

# ----------------------------------------------------------------------
# Default sites = the 5 existing Red Sea mangrove sites (for continuity)
# (site_id, longitude, latitude)
# ----------------------------------------------------------------------
DEFAULT_SITES = [
    ("Al_Shabaan",     37.180562, 24.801613),
    ("Al_Shoaiba",     39.473378, 20.766092),
    ("Al_Wajh",        36.705946, 26.025006),
    ("Duba_Lake",      35.601723, 27.433590),
    ("Juzur_Janabiat", 40.647287, 19.787375),
]
# Coordinates taken from data/processed/mangrove_all.csv (real site centroids).


def build_sites(args):
    """Return an ee.FeatureCollection of site polygons with 'site_id' + 'area'."""
    if args.site_mode == "points":
        feats = []
        for sid, lon, lat in DEFAULT_SITES:
            geom = ee.Geometry.Point([lon, lat]).buffer(args.buffer_m)
            feats.append(ee.Feature(geom, {"site_id": sid}))
        fc = ee.FeatureCollection(feats)

    elif args.site_mode == "asset":
        fc = ee.FeatureCollection(args.sites_asset)
        # ensure a 'site_id' property exists
        fc = fc.map(lambda f: f.set(
            "site_id", ee.Algorithms.If(
                f.propertyNames().contains(args.id_field),
                f.get(args.id_field), f.get("system:index"))))

    elif args.site_mode == "gmw":
        fc = gmw_sites(args)

    else:
        raise ValueError(args.site_mode)

    # attach polygon area (m^2), matching the 'area' column semantics
    return fc.map(lambda f: f.set("area", f.geometry().area(maxError=30)))


def gmw_sites(args):
    """
    Auto-generate mangrove sites from Global Mangrove Watch extent within a bbox.

    Strategy: take the GMW mangrove-extent image, keep mangrove pixels inside the
    bbox, sample stratified points on those pixels, and buffer each to a site.
    Adjust GMW_ASSET to the extent asset available to your account (the
    awesome-gee-community-catalog hosts GMW v3 yearly extents).
    """
    bbox = ee.Geometry.Rectangle(args.bbox)  # [west, south, east, north]

    # GMW v3 yearly extent (community catalog). Update if your asset id differs.
    GMW_ASSET = "projects/sat-io/open-datasets/GMW/extent/GMW_v3"
    gmw = (ee.ImageCollection(GMW_ASSET)
           .filterDate(f"{args.gmw_year}-01-01", f"{args.gmw_year}-12-31")
           .mosaic()
           .gt(0)               # 1 where mangrove
           .selfMask())

    # stratified sample of mangrove pixels -> candidate site centers
    pts = gmw.rename("mangrove").stratifiedSample(
        numPoints=args.n_sites, classBand="mangrove", region=bbox,
        scale=args.gmw_scale, geometries=True, seed=args.seed)

    pts = pts.toList(args.n_sites)
    n = pts.size()

    def make(i):
        f = ee.Feature(pts.get(i))
        geom = f.geometry().buffer(args.buffer_m)
        return ee.Feature(geom, {"site_id": ee.String("GMW_").cat(ee.Number(i).int().format())})

    return ee.FeatureCollection(ee.List.sequence(0, n.subtract(1)).map(make))


def monthly_image(start):
    """Build a single multi-band monthly-composite image for date `start`."""
    end = start.advance(1, "month")

    ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(start, end).select("NDVI").mean()
            .multiply(0.0001).rename("ndvi"))

    lst = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start, end)
    lst_day = lst.select("LST_Day_1km").mean().multiply(0.02).subtract(273.15).rename("lst_day")
    lst_night = lst.select("LST_Night_1km").mean().multiply(0.02).subtract(273.15).rename("lst_night")

    precip = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(start, end).select("precipitation").sum()
              .rename("precipitation"))

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(start, end).first()
    soil = era5.select("volumetric_soil_water_layer_1").rename("soil_moisture")
    u = era5.select("u_component_of_wind_10m")
    v = era5.select("v_component_of_wind_10m")
    wind_speed = u.hypot(v).rename("wind_speed")
    wind_dir = v.atan2(u)                                   # radians
    wdir_sin = wind_dir.sin().rename("wind_direction_sin")
    wdir_cos = wind_dir.cos().rename("wind_direction_cos")

    sst = (ee.ImageCollection("NOAA/CDR/OISST/V2_1")
           .filterDate(start, end).select("sst").mean()
           .multiply(0.01).rename("sst"))

    return (ndvi.addBands([lst_day, lst_night, precip, soil,
                           sst, wind_speed, wdir_sin, wdir_cos]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, required=True,
                    help="your Google Cloud project registered with Earth Engine")
    ap.add_argument("--start", type=str, default="2014-01", help="YYYY-MM inclusive")
    ap.add_argument("--end", type=str, default="2024-12", help="YYYY-MM inclusive")
    ap.add_argument("--site_mode", choices=["points", "gmw", "asset"], default="points")
    ap.add_argument("--buffer_m", type=float, default=500.0, help="site buffer radius (m)")
    ap.add_argument("--scale", type=float, default=250.0, help="reduceRegions scale (m)")
    # gmw mode
    ap.add_argument("--bbox", type=float, nargs=4, default=[34.5, 12.0, 43.5, 28.0],
                    metavar=("W", "S", "E", "N"), help="Red Sea default bbox")
    ap.add_argument("--n_sites", type=int, default=50)
    ap.add_argument("--gmw_year", type=int, default=2020)
    ap.add_argument("--gmw_scale", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    # asset mode
    ap.add_argument("--sites_asset", type=str, default="")
    ap.add_argument("--id_field", type=str, default="site_id")
    # export
    ap.add_argument("--drive_folder", type=str, default="GECO_mangrove")
    ap.add_argument("--out_name", type=str, default="mangrove_gee")
    args = ap.parse_args()

    ee.Initialize(project=args.project)

    sites = build_sites(args)
    print("[INFO] site_mode:", args.site_mode)

    start = ee.Date(args.start + "-01")
    end = ee.Date(args.end + "-01").advance(1, "month")
    n_months = end.difference(start, "month").round()

    bands = ["ndvi", "lst_day", "lst_night", "precipitation", "soil_moisture",
             "sst", "wind_speed", "wind_direction_sin", "wind_direction_cos"]

    def per_month(m):
        m = ee.Number(m)
        s = start.advance(m, "month")
        img = monthly_image(s)
        # site means for this month
        fc = img.reduceRegions(collection=sites, reducer=ee.Reducer.mean(),
                               scale=args.scale)
        # add centroid lat/lon, date, time_idx, carry site_id + area
        def deco(f):
            c = f.geometry().centroid(maxError=30).coordinates()
            return f.set({
                "longitude": c.get(0),
                "latitude": c.get(1),
                "date": s.format("M/1/YYYY"),
                "time_idx": m.add(1),
            })
        return fc.map(deco)

    all_fc = ee.FeatureCollection(
        ee.List.sequence(0, n_months.subtract(1)).map(per_month)).flatten()

    selectors = (["site_id", "date", "time_idx", "latitude", "longitude", "area"]
                 + bands)

    task = ee.batch.Export.table.toDrive(
        collection=all_fc,
        description=args.out_name,
        folder=args.drive_folder,
        fileNamePrefix=args.out_name,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    print(f"[INFO] Export started: task id = {task.id}")
    print(f"[INFO] Watch progress in the GEE Code Editor 'Tasks' tab.")
    print(f"[INFO] Output -> Google Drive/{args.drive_folder}/{args.out_name}.csv")


if __name__ == "__main__":
    main()
