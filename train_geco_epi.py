#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_geco_epi.py — GECO v2 + Eco-Physically Informed (EPI) constraints.

Adds physics-informed penalties (quantile non-crossing, canopy-inertia rate
limit, physical bounds, multi-driver water stress) on top of the corrected v2
architecture, and reports BOTH predictive skill and physical-consistency
metrics. Designed as the reproducible experiment script for the paper's
physics-informed ablation.

Example (full physics):
    python train_geco_epi.py --arch v2 --add_season_feats --physics --epochs 120
Ablation baseline (no physics):
    python train_geco_epi.py --arch v2 --add_season_feats --epochs 120
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
from geco.losses import quantile_loss, laplacian_smoothness_loss, QUANTILES
from geco.physics_losses import (
    quantile_crossing_loss, temporal_rate_loss, bounds_loss,
    water_stress_loss, consistency_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="data/processed/mangrove_all.csv")
    p.add_argument("--input_length", type=int, default=12)
    p.add_argument("--forecast_horizon", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    # graph
    p.add_argument("--num_seasons", type=int, default=4)
    p.add_argument("--k_neighbors_base", type=int, default=4)
    p.add_argument("--k_neighbors_seasonal", type=int, default=4)
    p.add_argument("--sigma_geo", type=float, default=1.0)
    p.add_argument("--sigma_env", type=float, default=1.0)
    p.add_argument("--sigma_dyn", type=float, default=1.0)
    p.add_argument("--lambda_geo", type=float, default=1.0)
    p.add_argument("--lambda_env", type=float, default=1.0)
    # model
    p.add_argument("--arch", type=str, default="v2", choices=["v1", "v2"])
    p.add_argument("--add_season_feats", action="store_true")
    p.add_argument("--gnn_hidden_dim", type=int, default=64)
    p.add_argument("--gnn_out_dim", type=int, default=64)
    p.add_argument("--temporal_hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    # regularizers
    p.add_argument("--lambda_lap", type=float, default=1e-3)
    p.add_argument("--physics", action="store_true", help="enable EPI defaults")
    p.add_argument("--lambda_cross", type=float, default=None)
    p.add_argument("--lambda_rate", type=float, default=None)
    p.add_argument("--lambda_bounds", type=float, default=None)
    p.add_argument("--lambda_water", type=float, default=None)
    p.add_argument("--rate_pctl", type=float, default=99.0,
                   help="percentile of |monthly dNDVI| used as the rate limit")
    p.add_argument("--checkpoint", type=str, default="checkpoints/geco_epi.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # EPI default weights
    if args.physics:
        if args.lambda_cross is None: args.lambda_cross = 1.0
        if args.lambda_rate is None: args.lambda_rate = 0.1
        if args.lambda_bounds is None: args.lambda_bounds = 1.0
        if args.lambda_water is None: args.lambda_water = 0.05
    for k in ["lambda_cross", "lambda_rate", "lambda_bounds", "lambda_water"]:
        if getattr(args, k) is None:
            setattr(args, k, 0.0)

    df = pd.read_csv(args.csv_path)
    if args.add_season_feats and "date" in df.columns:
        m = pd.to_datetime(df["date"]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    site_ids = sorted(df["site_id"].unique().tolist())
    site_id_to_idx = {s: i for i, s in enumerate(site_ids)}

    static_cols = [c for c in ["area"] if c in df.columns]
    dyn_for_graph = [c for c in [
        "precipitation", "salinity", "sea_surface_height", "soil_moisture",
        "ocean_current_speed", "ocean_cur_dir_sin", "ocean_cur_dir_cos",
        "wind_speed", "wind_direction_sin", "wind_direction_cos"] if c in df.columns]
    static_init, d_geo, d_env = build_static_and_dynamic_stats(
        df, site_id_to_idx, static_cols, dyn_for_graph)
    base_adj_norm = build_base_ecological_adjacency(
        d_geo, d_env, args.lambda_geo, args.lambda_env,
        args.sigma_geo, args.sigma_env, args.k_neighbors_base)
    seasonal_adjs = build_seasonal_adjacencies(
        df, site_id_to_idx, dyn_for_graph, args.num_seasons,
        args.sigma_dyn, args.k_neighbors_seasonal)

    L, H = args.input_length, args.forecast_horizon
    cutoff = {}
    for sid, g in df.groupby("site_id"):
        t = np.sort(g["time_idx"].to_numpy())
        cutoff[sid] = t[int(round((1 - args.val_frac) * len(t))) - 1]

    probe = MangroveWindowDataset(df, L, H, site_id_to_idx, target_col="ndvi")
    dyn_cols = probe.dynamic_cols
    train_rows = df[df.apply(lambda r: r["time_idx"] <= cutoff[r["site_id"]], axis=1)]
    norm_stats = {c: (float(train_rows[c].mean()),
                      float(train_rows[c].std() + 1e-6)) for c in dyn_cols}

    ds = MangroveWindowDataset(df, L, H, site_id_to_idx, target_col="ndvi",
                               norm_stats=norm_stats)
    tgt_mean, tgt_std = ds.target_mean, ds.target_std

    # normalized physical thresholds (raw -> normalized ndvi space)
    d_raw = df.groupby("site_id")["ndvi"].diff().abs().dropna()
    max_delta_raw = float(np.percentile(d_raw, args.rate_pctl))
    max_delta_norm = max_delta_raw / tgt_std
    lo_norm = (0.0 - tgt_mean) / tgt_std
    hi_norm = (1.0 - tgt_mean) / tgt_std

    # water-stress index from x_last: high when soil moisture AND precip are low
    sm_i = dyn_cols.index("soil_moisture") if "soil_moisture" in dyn_cols else None
    pr_i = dyn_cols.index("precipitation") if "precipitation" in dyn_cols else None

    train_idx, val_idx = [], []
    for gi, (s_series, start) in enumerate(ds.index):
        site_idx, arr, times = ds.series[s_series]
        sid = site_ids[site_idx]
        (train_idx if times[start + L] <= cutoff[sid] else val_idx).append(gi)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    ModelCls = GECOFullV2 if args.arch == "v2" else GECOFull
    model = ModelCls(
        static_init=static_init.to(device), base_adj_norm=base_adj_norm.to(device),
        seasonal_adjs=[A.to(device) for A in seasonal_adjs], dyn_input_dim=len(dyn_cols),
        gnn_hidden_dim=args.gnn_hidden_dim, gnn_out_dim=args.gnn_out_dim,
        temporal_hidden_dim=args.temporal_hidden_dim, forecast_horizon=H,
        quantiles=QUANTILES, num_heads=args.num_heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    med_idx = int(np.argmin([abs(q - 0.5) for q in QUANTILES]))
    tgt_i = dyn_cols.index("ndvi")

    print(f"[INFO] arch={args.arch} season={args.add_season_feats} "
          f"physics(cross={args.lambda_cross},rate={args.lambda_rate},"
          f"bounds={args.lambda_bounds},water={args.lambda_water}) "
          f"rate_limit_raw={max_delta_raw:.3f}")

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    best = float("inf"); best_report = None

    def water_index(x_last):
        if sm_i is None or pr_i is None:
            return torch.zeros(x_last.size(0), device=x_last.device)
        return -(x_last[:, sm_i] + x_last[:, pr_i]) / 2.0  # high => dry

    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in train_loader:
            x = b["x"].to(device); y = b["y"].to(device)
            xl = b["x_last"].to(device); si = b["site_idx"].to(device)
            opt.zero_grad()
            yq, z = model(x, si)
            loss = quantile_loss(y, yq, QUANTILES)
            loss = loss + args.lambda_lap * laplacian_smoothness_loss(z, model.base_adj_norm)
            if args.lambda_cross: loss = loss + args.lambda_cross * quantile_crossing_loss(yq)
            if args.lambda_rate:  loss = loss + args.lambda_rate * temporal_rate_loss(yq, xl[:, tgt_i], max_delta_norm, med_idx)
            if args.lambda_bounds: loss = loss + args.lambda_bounds * bounds_loss(yq, lo_norm, hi_norm)
            if args.lambda_water: loss = loss + args.lambda_water * water_stress_loss(yq, water_index(xl), xl[:, tgt_i], 0.5, 1.0, med_idx)
            loss.backward(); opt.step()

        # ---- eval: accuracy + physical consistency (original scale) ----
        model.eval(); Y, P = [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device); y = b["y"].to(device); si = b["site_idx"].to(device)
                yq, _ = model(x, si)
                Y.append(y.cpu().numpy().reshape(-1))
                P.append(yq.cpu().numpy().reshape(-1, yq.shape[-1]))
        yv = np.concatenate(Y) * tgt_std + tgt_mean
        pq = np.concatenate(P) * tgt_std + tgt_mean
        pv = pq[:, med_idx]
        r2 = r2_score(yv, pv); mae = mean_absolute_error(yv, pv)
        rmse = math.sqrt(mean_squared_error(yv, pv))
        cm = consistency_metrics(pq, yv, med_idx, 0.0, 1.0, max_delta_raw)

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Ep {epoch:03d}] R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | "
                  f"cross={cm['quantile_crossing_rate']:.3f} oob={cm['out_of_bounds_rate']:.3f} "
                  f"cov80={cm['coverage_80']:.3f}")
        if rmse < best:
            best = rmse
            best_report = dict(epoch=epoch, R2=r2, MAE=mae, RMSE=rmse, **cm)
            torch.save({"model_state_dict": model.state_dict(), "norm_stats": norm_stats,
                        "dyn_cols": dyn_cols, "args": vars(args)}, args.checkpoint)

    print("[BEST] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in best_report.items()))


if __name__ == "__main__":
    main()
