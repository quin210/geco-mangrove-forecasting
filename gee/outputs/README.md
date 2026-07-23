# gee/outputs/ — landing spot for Google Earth Engine exports

**Push your GEE export CSVs here.** When you (with a GEE account) run
`gee/extract_mangrove_gee.py`, it writes to your Google Drive; download that CSV
and drop it in this folder, then run `gee/postprocess_gee_csv.py` (see
`../README.md` §4) to convert it into the training schema under
`data/processed/`.

## Expected columns from `extract_mangrove_gee.py`

One row per **site-month**:

```
site_id, date, time_idx, latitude, longitude, area,
ndvi, lst_day, lst_night,            # ndvi = target; LST = thermal driver (°C)
precipitation, soil_moisture, sst,
wind_speed, wind_direction_sin, wind_direction_cos
```

- `date` format `M/1/YYYY` (monthly, day fixed to 1), matching the rest of the project.
- `ndvi` may instead be named `ndvi_modis`; the trainers accept either.
- Any column the free pipeline can't provide (e.g. ocean currents) can simply be
  absent — training auto-detects available features.

## Notes

- CSVs here are **git-tracked on purpose** so a reproducible GEE dataset can be
  committed alongside the code. If a file is very large, consider Git LFS or
  keeping only a compressed copy.
- Keep filenames descriptive, e.g. `mangrove_gee_global40_2014_2024.csv`.
