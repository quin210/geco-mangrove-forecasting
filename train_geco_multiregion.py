#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_geco_multiregion.py — GECO on a MULTI-REGION, GEE-free dataset.

Data = MODIS NDVI (target, from ORNL) + ERA5 climate (Open-Meteo) + geographic
conditioning (geco.geo_features). Uses:
  - hemisphere-aware seasonality (season_sin/cos) as temporal drivers
  - static geographic descriptors (lat, |lat|, lon sin/cos, aridity) as node
    features feeding the spatial encoder / static context
  - a CLIMATIC-SIMILARITY block graph (not geodesic) so cross-ocean sites are
    linked by climate niche, not meaningless distance
  - the fixed v2 temporal architecture + physics-informed calibration

Example:
    python train_geco_multiregion.py --modis data/processed/modis_subset.csv \
        --climate data/processed/climate_global.csv --epochs 80
"""

import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from geco.dataset import MangroveWindowDataset
from geco.graph_builder import build_seasonal_adjacencies
from geco.geo_features import (
    add_geo_features, SEASONAL_TEMPORAL_COLS, STATIC_GEO_COLS,
    climatic_similarity_adjacency,
)
from geco.model_geco_v2 import GECOFullV2
from geco.losses import quantile_loss, laplacian_smoothness_loss, QUANTILES
from geco.physics_losses import quantile_crossing_loss, bounds_loss, consistency_metrics


def build_dataframe(modis_path, climate_path):
    """Merge MODIS NDVI (target) + ERA5 climate + geo features; keep sites with both."""
    modis = pd.read_csv(modis_path)
    clim = pd.read_csv(climate_path)
    drivers = ["air_temp_mean", "air_temp_max", "air_temp_min", "vpd", "et0",
               "soil_moist_era5", "radiation", "precip_era5", "water_balance"]
    keep_clim = ["site_id", "date"] + [c for c in drivers if c in clim.columns]
    df = modis.merge(clim[keep_clim], on=["site_id", "date"], how="inner")
    df = df.rename(columns={"ndvi_modis": "ndvi"}).dropna(subset=["ndvi"])
    df = add_geo_features(df)                       # season_sin/cos + static geo + aridity
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("site_id").cumcount() + 1
    return df, [c for c in drivers if c in df.columns]


def normalized_adj(A):
    N = A.shape[0]
    A = A + np.eye(N, dtype=np.float32)
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(d + 1e-6)
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", default="data/processed/modis_subset.csv")
    ap.add_argument("--climate", default="data/processed/climate_global.csv")
    ap.add_argument("--input_length", type=int, default=12)
    ap.add_argument("--forecast_horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_seasons", type=int, default=4)
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    df, driver_cols = build_dataframe(args.modis, args.climate)
    sites = sorted(df["site_id"].unique())
    site_id_to_idx = {s: i for i, s in enumerate(sites)}
    N = len(sites)
    print(f"[INFO] {N} sites: {sites}")
    print(f"[INFO] {len(df)} site-months | drivers: {driver_cols}")

    # temporal feature columns = drivers + hemisphere-aware season + target
    feat_cols = driver_cols + SEASONAL_TEMPORAL_COLS
    # exclude everything non-temporal from the dataset's dynamic features
    static_present = [c for c in STATIC_GEO_COLS if c in df.columns]
    exclude = ["time_idx", "latitude", "longitude"] + static_present + ["evi_modis", "n_comp", "month"]

    # ---- static node features: standardized [geo descriptors | driver means] ----
    S = df.groupby("site_id")[static_present].first().reindex(sites).to_numpy(np.float32)
    Mu = df.groupby("site_id")[driver_cols].mean().reindex(sites).to_numpy(np.float32)
    def z(a): return (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-6)
    static_init = torch.from_numpy(np.concatenate([z(S), z(Mu)], 1).astype(np.float32))

    # ---- climatic-similarity block graph ----
    A = climatic_similarity_adjacency(df, static_geo_cols=static_present, driver_cols=driver_cols,
                                      k_within=min(4, N - 1), k_cross=min(2, N - 1))
    base_adj_norm = torch.from_numpy(normalized_adj(A))
    seasonal_adjs = build_seasonal_adjacencies(df, site_id_to_idx, driver_cols,
                                               args.num_seasons, sigma_dyn=1.0,
                                               k_neighbors=min(4, N - 1))

    # ---- windows + time-based split ----
    L, H = args.input_length, args.forecast_horizon
    cutoff = {s: np.sort(g["time_idx"].to_numpy())[int(round((1 - args.val_frac) * len(g))) - 1]
              for s, g in df.groupby("site_id")}
    probe = MangroveWindowDataset(df, L, H, site_id_to_idx, target_col="ndvi", exclude_cols=exclude)
    dyn_cols = probe.dynamic_cols
    train_rows = df[df.apply(lambda r: r["time_idx"] <= cutoff[r["site_id"]], axis=1)]
    norm_stats = {c: (float(train_rows[c].mean()), float(train_rows[c].std() + 1e-6)) for c in dyn_cols}
    ds = MangroveWindowDataset(df, L, H, site_id_to_idx, target_col="ndvi",
                               exclude_cols=exclude, norm_stats=norm_stats)
    tgt_mean, tgt_std = ds.target_mean, ds.target_std
    print(f"[INFO] {len(dyn_cols)} temporal features: {dyn_cols}")

    tr, va = [], []
    for gi, (ss, start) in enumerate(ds.index):
        sidx, arr, times = ds.series[ss]
        (tr if times[start + L] <= cutoff[sites[sidx]] else va).append(gi)
    print(f"[INFO] train windows={len(tr)} val windows={len(va)}")

    train_loader = DataLoader(Subset(ds, tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=64, shuffle=False)

    model = GECOFullV2(
        static_init=static_init.to(device), base_adj_norm=base_adj_norm.to(device),
        seasonal_adjs=[a.to(device) for a in seasonal_adjs], dyn_input_dim=len(dyn_cols),
        gnn_hidden_dim=64, gnn_out_dim=64, temporal_hidden_dim=128,
        forecast_horizon=H, quantiles=QUANTILES, num_heads=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    med = int(np.argmin([abs(q - 0.5) for q in QUANTILES]))
    lo = (0.0 - tgt_mean) / tgt_std; hi = (1.0 - tgt_mean) / tgt_std
    # per-site TRAIN-mean NDVI (original scale) for the fair within-site metric
    site_mean_ndvi = train_rows.groupby("site_id")["ndvi"].mean().to_dict()
    best = float("inf"); best_rep = None

    for ep in range(1, args.epochs + 1):
        model.train()
        for b in train_loader:
            x = b["x"].to(device); y = b["y"].to(device); si = b["site_idx"].to(device)
            opt.zero_grad()
            yq, zz = model(x, si)
            loss = quantile_loss(y, yq, QUANTILES) + 1e-3 * laplacian_smoothness_loss(zz, model.base_adj_norm)
            if args.physics:
                loss = loss + 1.0 * quantile_crossing_loss(yq) + 1.0 * bounds_loss(yq, lo, hi)
            loss.backward(); opt.step()

        model.eval(); Y, P, SI = [], [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device); y = b["y"].to(device); si = b["site_idx"].to(device)
                yq, _ = model(x, si)
                Y.append(y.cpu().numpy().reshape(-1))
                P.append(yq.cpu().numpy().reshape(-1, yq.shape[-1]))
                SI.append(np.repeat(b["site_idx"].numpy(), H))   # align site to each horizon step
        if not Y:
            print("[WARN] no val windows"); break
        yv = np.concatenate(Y) * tgt_std + tgt_mean
        pq = np.concatenate(P) * tgt_std + tgt_mean
        pv = pq[:, med]
        si_all = np.concatenate(SI)
        r2 = r2_score(yv, pv); mae = mean_absolute_error(yv, pv)
        rmse = math.sqrt(mean_squared_error(yv, pv))
        cm = consistency_metrics(pq, yv, med, 0.0, 1.0, 0.27)

        # ---- FAIR temporal skill: subtract each site's train-mean (anomalies) ----
        smean = np.array([site_mean_ndvi[sites[s]] for s in si_all])
        within_r2 = r2_score(yv - smean, pv - smean)

        if ep % 20 == 0 or ep == 1:
            print(f"[Ep {ep:03d}] R2={r2:.4f} within_R2={within_r2:.4f} "
                  f"RMSE={rmse:.4f} cov80={cm['coverage_80']:.3f}")
        if rmse < best:
            best = rmse
            best_rep = dict(epoch=ep, R2=r2, within_R2=within_r2, MAE=mae, RMSE=rmse, **cm)
            # per-site R² breakdown at the best epoch
            per_site = {}
            for s in np.unique(si_all):
                m = si_all == s
                if m.sum() >= 3 and np.var(yv[m]) > 1e-8:
                    per_site[sites[s]] = round(float(r2_score(yv[m], pv[m])), 3)
            best_rep["_per_site_r2"] = per_site

    ps = best_rep.pop("_per_site_r2", {})
    print("[BEST] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in best_rep.items()))
    print("[PER-SITE R2] " + " ".join(f"{k}={v}" for k, v in ps.items()))


if __name__ == "__main__":
    main()
