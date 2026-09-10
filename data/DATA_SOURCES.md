# GECO data sources — provenance & acquisition (GEE-free)

All satellite/reanalysis drivers below can be pulled **without Google Earth
Engine**, most **without any authentication**, using plain HTTP APIs. This makes
the dataset fully reproducible for publication.

## Working now (no authentication)

| Variable(s) | Product | Source / API | Script | Status |
|---|---|---|---|---|
| NDVI, EVI (250 m, 16-day, 2000–present) | MOD13Q1 | ORNL DAAC MODIS Web Service `modis.ornl.gov/rst/api/v1` | `data/fetch_modis_ndvi.py` | ✅ tested |
| LST day/night (1 km, 8-day) | MOD11A2 | ORNL DAAC MODIS Web Service | `data/fetch_modis_ndvi.py --with_lst` | ✅ tested |
| Air temp mean/max/min, VPD, ET0, soil moisture, shortwave radiation | ERA5 reanalysis | Open-Meteo Archive `archive-api.open-meteo.com` | `data/fetch_openmeteo.py` | ✅ tested |
| Water balance (precip − ET0) | derived | — | `data/fetch_openmeteo.py` | ✅ |

Both APIs take a lat/lon and a date range and return JSON. The ORNL point-subset
endpoint limits composites per request, so `fetch_modis_ndvi.py` reads valid
dates from `/dates` and requests in small chunks; MOD13Q1 pixel reliability
(0/1) is used to drop cloud/poor pixels.

## Needs a free account / token

| Variable(s) | Product | Source | Notes |
|---|---|---|---|
| NDVI/EVI/LST/Landsat, point & area samples | MODIS/VIIRS/Landsat | **NASA AppEEARS** REST API | free NASA Earthdata login → bearer token |
| SST (0.25°, daily) | NOAA OISST v2.1 | NOAA ERDDAP / PolarWatch | ERDDAP griddap; some servers need no auth but rate-limited |
| Sentinel-2 / Landsat surface reflectance | — | **Microsoft Planetary Computer** STAC | no key to read; heavier (raster + rasterio) |
| SST, chlorophyll, currents | Copernicus Marine | **CMEMS** | free account + `copernicusmarine` client |
| Tidal amplitude / inundation | FES2014 / GTSM | AVISO / Copernicus | registration |

## Site definition (for spatial breadth)

- **Global Mangrove Watch (GMW v3)** polygons — Zenodo / `data.unep-wcmc.org`.
  Use to derive many site coordinates. Large vector; for a quick start use the
  curated `data/mangrove_sites_global.csv` (multi-region representative points)
  and expand from GMW later.
- Point APIs (ORNL, Open-Meteo) only sample coordinates — they do **not** rasterize
  polygons like GEE, so you supply the site list yourself.

## Gaps vs the original `mangrove_all.csv`

`salinity`, `sea_surface_height`, `ocean_current_speed`, `ocean_cur_dir_*` are
**not** covered by the no-auth APIs above (they come from Copernicus Marine /
altimetry). The GECO code auto-detects available columns, so a dataset without
them still trains; add them from CMEMS when an account is available.

## Reproduce the fully self-collected dataset (5 Red Sea sites)

```bash
python data/fetch_modis_ndvi.py --start 2014 --end 2024 --with_lst \
    --out data/processed/modis_ndvi.csv
python data/fetch_openmeteo.py  --base data/processed/mangrove_all.csv \
    --out data/processed/mangrove_enriched.csv
python data/merge_sources.py     # -> data/processed/mangrove_multisource.csv
```
