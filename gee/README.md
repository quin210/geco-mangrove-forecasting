# Google Earth Engine extraction for GECO-EWS

Extracts a monthly, per-site environmental time series for mangrove sites,
adding the thermal driver **LST** (land surface temperature), in the same schema
as `data/processed/mangrove_all.csv`.

---

## 0. Do you actually need GEE?  (read this first)

**The default pipeline does NOT need GEE.** `data/fetch_modis_ndvi.py` (ORNL) and
`data/fetch_openmeteo.py` (ERA5) already pull NDVI/EVI/LST + climate for any site
list, no account required. Use this `gee/` path **only** when you want something
the point APIs can't do well:

| Reach for GEE when you need… | Why the free path struggles |
|---|---|
| **Polygon-averaged** NDVI over the true GMW mangrove extent | ORNL only samples a point/grid window; our `--snap` is an approximation |
| **Auto-generating many sites** from GMW polygons in a bbox | point APIs need you to supply coordinates |
| **Strict cloud/QA masking at scale** (humid tropics) | doable point-wise but slow/rate-limited over 100s of sites |
| **LST / SST / tidal for hundreds of sites, fast** | ORNL is slow (~5 s/request) and rate-limits |

> **Workflow if you decide to use GEE:** you (with a GEE account) run the script
> below, download the export, and **drop the CSV in `gee/outputs/`** (see
> [§4](#4-where-gee-outputs-go--how-to-plug-in)). Everything downstream then works
> exactly like the GEE-free path. If GEE is **not** needed, ignore this folder.

---

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

## 3. Variable catalogue — exactly what to pull from GEE

All variables are reduced to a **monthly value per site** (polygon/buffer mean,
`reduceRegions`), `date = M/1/YYYY`. Priority ranks value for the **dieback
early-warning** goal (★★★ = pull first).

### 3A. Core — already in `extract_mangrove_gee.py`

| Column | GEE asset | Band(s) | Scale / unit | Monthly agg | Priority — why |
|---|---|---|---|---|---|
| `ndvi` | `MODIS/061/MOD13Q1` | `NDVI` | ×1e-4 (ratio) | mean | ★★★ target (canopy greenness) |
| `lst_day`, `lst_night` | `MODIS/061/MOD11A2` | `LST_Day_1km`, `LST_Night_1km` | ×0.02 K − 273.15 → °C | mean | ★★★ thermal stress / heatwave |
| `precipitation` | `UCSB-CHG/CHIRPS/DAILY` | `precipitation` | mm | **sum** | ★★ water supply |
| `soil_moisture` | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `volumetric_soil_water_layer_1` | m³/m³ | mean | ★★ drought |
| `sst` | `NOAA/CDR/OISST/V2_1` | `sst` | ×0.01 → °C | mean | ★★★ marine heatwave (free path can't get this at coast) |
| `wind_speed`, `wind_direction_sin/cos` | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `u_/v_component_of_wind_10m` | m/s; dir=atan2(v,u) | mean | ★ storm exposure |

### 3B. Recommended additions — extend the script to unlock the early-warning story

| Column | GEE asset | Band(s) | Scale / unit | Agg | Priority — why |
|---|---|---|---|---|---|
| `evi` | `MODIS/061/MOD13Q1` | `EVI` | ×1e-4 | mean | ★★ less saturating than NDVI in dense canopy |
| `mangrove_extent`, `area_m2` | GMW (e.g. `projects/sat-io/open-datasets/GMW/extent/GMW_v3`) | yearly extent | mask | area | ★★★ define sites + track **loss/dieback area** directly |
| `inundation` | `JRC/GSW1_4/MonthlyHistory` | `water` | fraction inundated | mean | ★★★ hydroperiod; the Carpentaria dieback tracked sea-level/inundation |
| `chlor_a` | `NASA/OCEANDATA/MODIS-Aqua/L3SMI` | `chlor_a` | mg/m³ | mean | ★ coastal productivity / turbidity |
| `canopy_height` | `NASA/GEDI/...` L4A or Potapov 2021 canopy height | height | m | mean/first | ★★ structure / blue-carbon; baseline biomass |
| `nightlights` | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | `avg_rad` | nW·cm⁻²·sr⁻¹ | mean | ★ human pressure (aquaculture/urban) |
| `air_temp`, `vpd` | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `temperature_2m`, `dewpoint_2m` | °C; VPD from T,Td | mean | ★★ (also available via Open-Meteo non-GEE) |

> **QA matters:** for MOD13Q1 mask by `SummaryQA ≤ 1`; for MOD11A2 mask by
> `QC_Day`/`QC_Night` good-quality bits — before the monthly mean. This is a key
> reason to prefer GEE over point sampling in cloudy tropics.

### 3C. Not on GEE — pull elsewhere (non-GEE)

| Column | Source | How |
|---|---|---|
| `air_temp_mean/max/min`, `vpd`, `et0`, `soil_moist_era5`, `radiation`, `precip_era5`, `water_balance` | ERA5 via **Open-Meteo** | `data/fetch_openmeteo.py` (no account) |
| `salinity`, `sea_surface_height`, `ocean_current_speed`, `ocean_cur_dir_*` | **Copernicus Marine (CMEMS)** | free account + `copernicusmarine` client (already in `mangrove_all.csv`) |

## 3.5 GEE vs non-GEE — division of labour

- **Pull from GEE** the things point-APIs can't do well: **GMW-polygon-averaged**
  NDVI/EVI with **strict cloud masking**, **SST at the coast**, **inundation**
  (JRC water), **canopy structure** (GEDI), **mangrove-loss area** (GMW change),
  and auto-generated sites from GMW.
- **Pull non-GEE** (already automated, no account): ERA5 climate/water-stress via
  Open-Meteo, and MODIS NDVI/EVI/LST via ORNL as a fallback.
- **Merge** on `(site_id, date)` — the trainers auto-detect whatever columns are
  present, so GEE and non-GEE variables combine into one richer dataset.

## 4. Where GEE outputs go & how to plug in

**Landing spot:** put the raw Drive export in **`gee/outputs/`** (that folder
exists and is tracked; see `gee/outputs/README.md` for the expected schema). This
is the folder to push GEE results to.

**Then convert to the training schema:**

```bash
python gee/postprocess_gee_csv.py \
    --in  gee/outputs/mangrove_gee.csv \
    --out data/processed/mangrove_gee_all.csv
```

This sorts by site/time, rebuilds a contiguous `time_idx`, drops months with
missing NDVI, and reports coverage per site.

**Then train** — the GEE output is a drop-in replacement for the GEE-free CSVs:

```bash
# single-region / single combined CSV
python train_geco_epi.py --csv_path data/processed/mangrove_gee_all.csv --arch v2 --add_season_feats --physics

# multi-region: use the GEE NDVI as --modis and climate as --climate
python train_geco_multiregion.py --modis data/processed/mangrove_gee_all.csv \
    --climate data/processed/climate_global.csv --physics
```

**Column contract** (what the trainers expect). At minimum: `site_id`, `date`
(`M/D/YYYY`), `latitude`, `longitude`, `ndvi` **or** `ndvi_modis` (target).
Optional drivers are auto-detected: `lst_day`, `lst_night` (unlock a
thermal-stress physics term), `precipitation`/`precip_era5`, `soil_moisture`,
`sst`, `wind_*`. Any missing column is simply skipped — a GEE CSV without ocean
currents still trains.

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
