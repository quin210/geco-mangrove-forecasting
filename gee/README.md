# Google Earth Engine extraction for GECO

Extracts a monthly, per-site environmental time series for mangrove sites,
adding the currently-missing thermal driver **LST** (land surface temperature),
in the same schema as `data/processed/mangrove_all.csv`.

## 1. One-time setup

```bash
pip install earthengine-api

# Authenticate (opens a browser; needs a Google account with Earth Engine access).
# If you don't have EE access yet, sign up at https://earthengine.google.com/signup/
earthengine authenticate
```

You also need a **Google Cloud project** registered for Earth Engine — pass it
with `--project`. Find/create one at https://console.cloud.google.com/ and
enable the Earth Engine API.

> If a login command must run interactively in this session, type it yourself
> with a leading `!`, e.g. `! earthengine authenticate`.

## 2. Run

**Reproduce / extend the 5 Red Sea sites** (default point mode):

```bash
python gee/extract_mangrove_gee.py --project YOUR_GEE_PROJECT \
    --start 2014-01 --end 2024-12 --site_mode points
```

**Auto-generate many sites from Global Mangrove Watch** within a bounding box:

```bash
python gee/extract_mangrove_gee.py --project YOUR_GEE_PROJECT \
    --site_mode gmw --bbox 34.5 12.0 43.5 28.0 --n_sites 50 --gmw_year 2020
```

**Use your own uploaded mangrove polygons** (EE FeatureCollection asset):

```bash
python gee/extract_mangrove_gee.py --project YOUR_GEE_PROJECT \
    --site_mode asset --sites_asset users/you/mangrove_polygons --id_field name
```

The script starts an **Export.table.toDrive** task. Watch it in the EE Code
Editor *Tasks* tab; the CSV lands in `Google Drive/GECO_mangrove/mangrove_gee.csv`.

## 3. Variables produced

| Column | Source | Notes |
|---|---|---|
| `ndvi` | MODIS/061/MOD13Q1 | 16-day 250 m, scaled ×1e-4 |
| `lst_day`, `lst_night` | MODIS/061/MOD11A2 | °C (×0.02 − 273.15) — **new thermal driver** |
| `precipitation` | UCSB-CHG/CHIRPS/DAILY | monthly sum (mm) |
| `soil_moisture` | ECMWF/ERA5_LAND/MONTHLY_AGGR | volumetric soil water L1 |
| `sst` | NOAA/CDR/OISST/V2_1 | °C |
| `wind_speed`, `wind_direction_sin/cos` | ERA5-Land 10 m u/v | speed = hypot(u,v) |

> **Not on GEE:** `ocean_current_speed`, `ocean_cur_dir_*` (from Copernicus
> Marine, a separate download). The GECO code auto-detects available columns, so
> a CSV without them still trains.

## 4. After download — merge to the project schema

```bash
python gee/postprocess_gee_csv.py \
    --in ~/Downloads/mangrove_gee.csv \
    --out data/processed/mangrove_gee_all.csv
```

This sorts by site/time, rebuilds a contiguous `time_idx`, drops months with
missing NDVI, and reports coverage per site. Then train exactly as before,
pointing `--csv_path` at the new file — `lst_day`/`lst_night` become available
for a **thermal-stress physics constraint**.

## 5. Notes / gotchas

- **GMW asset id** (`projects/sat-io/open-datasets/GMW/extent/GMW_v3`) is from
  the awesome-gee-community-catalog; if your account can't read it, upload the
  GMW v3 shapefile as your own asset and use `--site_mode asset`.
- **Buffer radius** (`--buffer_m`, default 500 m) sets each site's footprint; a
  larger buffer smooths noise but can mix in non-mangrove pixels.
- **Cloud/quality masking** is intentionally minimal here for clarity — for the
  final paper, add MOD13Q1 `SummaryQA` and MOD11A2 QC masking before `.mean()`.
- Replace the placeholder coordinates in `DEFAULT_SITES` (only `Al_Shabaan` is
  taken from the real data) with the true lat/lon from `data/raw/*.csv`.
