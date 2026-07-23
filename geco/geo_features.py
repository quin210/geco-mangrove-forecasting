"""
geo_features.py — geographic conditioning for MULTI-REGION mangrove modelling.

When sites span different climate zones and both hemispheres, a raw calendar
month and a raw driver value are ambiguous. This module adds the geographic
context the model needs to interpret them:

  1. Hemisphere-aware growing-season phase (fixes the N-hemisphere assumption
     baked into naive month_sin/cos): Southern-hemisphere sites are phase-shifted
     by 6 months so "phase peak" means the same physiological season everywhere.
  2. Static geographic descriptors per site: signed latitude (thermal regime /
     hemisphere), |latitude| (seasonality amplitude / distance from equator),
     longitude sin/cos, and an aridity index (precip / ET0) when available.

These feed two places in GECO:
  - temporal drivers: `season_sin`, `season_cos` (hemisphere-aware)
  - static node descriptors (graph / StaticVSN context): `abs_lat`, `lat_signed`,
    `lon_sin`, `lon_cos`, `aridity`  -> pass via `static_cols`.
"""

import numpy as np
import pandas as pd


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # month (from date if present, else from time_idx)
    if "date" in df.columns:
        month = pd.to_datetime(df["date"], errors="coerce").dt.month
    else:
        month = (df["time_idx"] % 12) + 1
    df["month"] = month

    # --- hemisphere-aware growing-season phase ---
    # Northern hemisphere: phase = month. Southern: shift by 6 months so the
    # physiological "peak season" aligns across hemispheres.
    south = df["latitude"] < 0
    phase_month = df["month"].where(~south, (df["month"] + 6 - 1) % 12 + 1)
    df["season_sin"] = np.sin(2 * np.pi * phase_month / 12.0)
    df["season_cos"] = np.cos(2 * np.pi * phase_month / 12.0)

    # --- static geographic descriptors ---
    df["lat_signed"] = df["latitude"]
    df["abs_lat"] = df["latitude"].abs()
    df["lon_sin"] = np.sin(np.deg2rad(df["longitude"]))
    df["lon_cos"] = np.cos(np.deg2rad(df["longitude"]))

    # aridity index (dimensionless): mean precip / mean ET0 per site.
    # <0.2 hyper-arid ... >1 humid. Uses whatever precip/ET0 columns exist.
    precip_col = next((c for c in ["precip_era5", "precipitation"] if c in df.columns), None)
    if precip_col and "et0" in df.columns:
        pe = (df.groupby("site_id")
                .apply(lambda g: g[precip_col].mean() / (g["et0"].mean() + 1e-6))
                .rename("aridity"))
        df = df.merge(pe, on="site_id", how="left")

    return df


# columns this module produces, grouped by how GECO should consume them
SEASONAL_TEMPORAL_COLS = ["season_sin", "season_cos"]
STATIC_GEO_COLS = ["lat_signed", "abs_lat", "lon_sin", "lon_cos", "aridity"]


def climatic_similarity_adjacency(
    df: pd.DataFrame,
    static_geo_cols=None,
    driver_cols=None,
    k_within=6,
    k_cross=2,
) -> np.ndarray:
    """
    Block/hierarchical adjacency for multi-region graphs.

    Geodesic distance is meaningless across oceans, so:
      - WITHIN a region: connect k_within nearest neighbours by geodesic distance
        (local hydrological / dispersal connectivity).
      - ACROSS regions: connect k_cross nearest neighbours by CLIMATIC-NICHE
        similarity (standardized climate/geography descriptors), so analogous
        environments (e.g. all arid mangroves) share statistical strength.

    Returns a symmetric weighted adjacency [N, N] over the unique sites (sorted).
    """
    if static_geo_cols is None:
        static_geo_cols = [c for c in STATIC_GEO_COLS if c in df.columns]
    if driver_cols is None:
        driver_cols = [c for c in ["air_temp_mean", "vpd", "aridity", "salinity"]
                       if c in df.columns]

    sites = sorted(df["site_id"].unique())
    N = len(sites)
    site_idx = {s: i for i, s in enumerate(sites)}

    # per-site static vectors
    agg = df.groupby("site_id").agg(
        {**{c: "first" for c in static_geo_cols + ["latitude", "longitude", "region"]
            if c in df.columns},
         **{c: "mean" for c in driver_cols}}
    ).reindex(sites)

    coords = agg[["latitude", "longitude"]].to_numpy(float)
    clim_cols = [c for c in static_geo_cols + driver_cols if c in agg.columns]
    clim = agg[clim_cols].to_numpy(float)
    clim = (clim - np.nanmean(clim, 0)) / (np.nanstd(clim, 0) + 1e-6)
    clim = np.nan_to_num(clim)

    region = agg["region"].to_numpy() if "region" in agg.columns else np.array([""] * N)

    A = np.zeros((N, N), dtype=np.float32)

    # geodesic distance (great-circle approx via haversine)
    def hav(a, b):
        la1, lo1, la2, lo2 = map(np.deg2rad, [a[0], a[1], b[0], b[1]])
        d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 2 * np.arcsin(np.sqrt(d))

    for i in range(N):
        # within-region geodesic neighbours
        same = [j for j in range(N) if j != i and region[j] == region[i]]
        if same:
            dg = np.array([hav(coords[i], coords[j]) for j in same])
            for j in np.array(same)[np.argsort(dg)[:k_within]]:
                w = np.exp(-hav(coords[i], coords[j]) ** 2)
                A[i, j] = max(A[i, j], w)
        # cross-region climatic-similarity neighbours
        other = [j for j in range(N) if region[j] != region[i]]
        if other:
            dc = np.array([np.linalg.norm(clim[i] - clim[j]) for j in other])
            for j in np.array(other)[np.argsort(dc)[:k_cross]]:
                w = np.exp(-np.linalg.norm(clim[i] - clim[j]) ** 2)
                A[i, j] = max(A[i, j], w)

    A = np.maximum(A, A.T)  # symmetric
    return A
