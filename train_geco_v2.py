#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_geco_v2.py — corrected training pipeline for GECO.

Fixes over train_geco.py:
  1. TIME-BASED split (no leakage). Windows are assigned to train/val by the
     time_idx at which their *target* starts, using a per-site cutoff.
  2. FEATURE STANDARDIZATION fit on TRAIN rows only, applied to all splits.
  3. DATA-DRIVEN eco-constraint threshold (salinity is expressed in normalized
     units here, so the fixed 35 PSU threshold no longer applies).
  4. Validation metrics reported on the ORIGINAL NDVI scale (de-normalized).

Usage mirrors train_geco.py, e.g.:
    python train_geco_v2.py --csv_path data/processed/mangrove_all.csv \
        --input_length 12 --forecast_horizon 3 --epochs 50 --device cuda
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from geco.dataset import MangroveWindowDataset
from geco.graph_builder import (
    build_static_and_dynamic_stats,
    build_base_ecological_adjacency,
    build_seasonal_adjacencies,
)
from geco.model_geco_full import GECOFull
from geco.model_geco_v2 import GECOFullV2
from geco.losses import quantile_loss, eco_salinity_loss, laplacian_smoothness_loss, QUANTILES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="data/processed/mangrove_all.csv")
    p.add_argument("--input_length", type=int, default=12)
    p.add_argument("--forecast_horizon", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--val_frac", type=float, default=0.2, help="fraction of the LAST timesteps used for validation")
    # graph
    p.add_argument("--lambda_geo", type=float, default=1.0)
    p.add_argument("--lambda_env", type=float, default=1.0)
    p.add_argument("--sigma_geo", type=float, default=1.0)
    p.add_argument("--sigma_env", type=float, default=1.0)
    p.add_argument("--sigma_dyn", type=float, default=1.0)
    p.add_argument("--k_neighbors_base", type=int, default=4)
    p.add_argument("--k_neighbors_seasonal", type=int, default=4)
    p.add_argument("--num_seasons", type=int, default=4)
    # model
    p.add_argument("--gnn_hidden_dim", type=int, default=64)
    p.add_argument("--gnn_out_dim", type=int, default=64)
    p.add_argument("--temporal_hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    # reg
    p.add_argument("--lambda_lap", type=float, default=1e-3)
    p.add_argument("--lambda_eco", type=float, default=1e-3)
    p.add_argument("--sal_thr_norm", type=float, default=0.5,
                   help="salinity threshold in NORMALIZED units for the eco constraint")
    p.add_argument("--arch", type=str, default="v2", choices=["v1", "v2"],
                   help="v1 = original TemporalVSN (scalar bottleneck), v2 = fixed per-feature embedding")
    p.add_argument("--add_season_feats", action="store_true",
                   help="add month_sin / month_cos as dynamic features")
    p.add_argument("--checkpoint", type=str, default="checkpoints/geco_v2.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    df = pd.read_csv(args.csv_path)
    for c in ["site_id", "time_idx", "ndvi", "latitude", "longitude"]:
        if c not in df.columns:
            raise ValueError(f"missing column {c}")

    # explicit seasonality features (dominant cycle in the data)
    if args.add_season_feats and "date" in df.columns:
        m = pd.to_datetime(df["date"]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)
        print("[INFO] added month_sin / month_cos features")

    site_ids = sorted(df["site_id"].unique().tolist())
    site_id_to_idx = {s: i for i, s in enumerate(site_ids)}
    print(f"[INFO] sites={len(site_ids)}")

    # ---- graph inputs (unchanged) ----
    static_cols = [c for c in ["area"] if c in df.columns]
    dyn_for_graph = [c for c in [
        "precipitation", "salinity", "sea_surface_height", "soil_moisture",
        "ocean_current_speed", "ocean_cur_dir_sin", "ocean_cur_dir_cos",
        "wind_speed", "wind_direction_sin", "wind_direction_cos"] if c in df.columns]

    static_init, d_geo, d_env = build_static_and_dynamic_stats(
        df, site_id_to_idx, static_cols=static_cols, dynamic_cols=dyn_for_graph)
    base_adj_norm = build_base_ecological_adjacency(
        d_geo, d_env, args.lambda_geo, args.lambda_env,
        args.sigma_geo, args.sigma_env, args.k_neighbors_base)
    seasonal_adjs = build_seasonal_adjacencies(
        df, site_id_to_idx, dyn_for_graph, args.num_seasons,
        args.sigma_dyn, args.k_neighbors_seasonal)

    # ---- per-site time cutoff for the split ----
    cutoff = {}
    for sid, g in df.groupby("site_id"):
        t = np.sort(g["time_idx"].to_numpy())
        cutoff[sid] = t[int(round((1 - args.val_frac) * len(t))) - 1]  # last train time_idx
    print(f"[INFO] per-site train cutoff time_idx: {cutoff}")

    # ---- fit normalization on TRAIN rows only ----
    L, H = args.input_length, args.forecast_horizon
    probe = MangroveWindowDataset(df, L, H, site_id_to_idx, target_col="ndvi")
    dyn_cols = probe.dynamic_cols
    train_mask = df.apply(lambda r: r["time_idx"] <= cutoff[r["site_id"]], axis=1)
    train_rows = df[train_mask]
    norm_stats = {c: (float(train_rows[c].mean()),
                      float(train_rows[c].std() + 1e-6)) for c in dyn_cols}
    print(f"[INFO] {len(dyn_cols)} dynamic features (lat/lon dropped): {dyn_cols}")

    ds = MangroveWindowDataset(df, L, H, site_id_to_idx, target_col="ndvi",
                               norm_stats=norm_stats)
    tgt_mean, tgt_std = ds.target_mean, ds.target_std

    # ---- assign windows to train/val by target-start time ----
    train_idx, val_idx = [], []
    for gi, (s_series, start) in enumerate(ds.index):
        site_idx, arr, times = ds.series[s_series]
        sid = site_ids[site_idx]
        target_start_time = times[start + L]
        (train_idx if target_start_time <= cutoff[sid] else val_idx).append(gi)
    print(f"[INFO] train windows={len(train_idx)}, val windows={len(val_idx)}")

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    ModelCls = GECOFullV2 if args.arch == "v2" else GECOFull
    print(f"[INFO] architecture: {args.arch} ({ModelCls.__name__})")
    model = ModelCls(
        static_init=static_init.to(device),
        base_adj_norm=base_adj_norm.to(device),
        seasonal_adjs=[A.to(device) for A in seasonal_adjs],
        dyn_input_dim=len(dyn_cols),
        gnn_hidden_dim=args.gnn_hidden_dim, gnn_out_dim=args.gnn_out_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        forecast_horizon=H, quantiles=QUANTILES, num_heads=args.num_heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    med_idx = int(np.argmin([abs(q - 0.5) for q in QUANTILES]))
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tot, n = 0.0, 0
        for b in train_loader:
            x = b["x"].to(device); y = b["y"].to(device)
            xl = b["x_last"].to(device); si = b["site_idx"].to(device)
            opt.zero_grad()
            yq, z = model(x, si)
            ql = quantile_loss(y, yq, QUANTILES)
            lap = laplacian_smoothness_loss(z, model.base_adj_norm)
            eco = eco_salinity_loss(yq, xl, dyn_cols, target_col="ndvi",
                                    sal_col="salinity", sal_thr=args.sal_thr_norm)
            loss = ql + args.lambda_lap * lap + args.lambda_eco * eco
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        train_loss = tot / max(1, n)

        # ---- eval on ORIGINAL ndvi scale ----
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device); y = b["y"].to(device); si = b["site_idx"].to(device)
                yq, _ = model(x, si)
                ys.append(y.cpu().numpy().reshape(-1))
                ps.append(yq[:, :, med_idx].cpu().numpy().reshape(-1))
        if ys:
            yv = np.concatenate(ys) * tgt_std + tgt_mean
            pv = np.concatenate(ps) * tgt_std + tgt_mean
            r2 = r2_score(yv, pv); mae = mean_absolute_error(yv, pv)
            rmse = math.sqrt(mean_squared_error(yv, pv))
        else:
            r2 = mae = rmse = float("nan")

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_R2={r2:.4f} | val_MAE={mae:.4f} | val_RMSE={rmse:.4f}")

        if rmse < best:
            best = rmse
            torch.save({"model_state_dict": model.state_dict(),
                        "norm_stats": norm_stats, "dyn_cols": dyn_cols,
                        "site_id_to_idx": site_id_to_idx}, args.checkpoint)
    print(f"[INFO] done. best val_RMSE(original scale)={best:.4f}")


if __name__ == "__main__":
    main()
