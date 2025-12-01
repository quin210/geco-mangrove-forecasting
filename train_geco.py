#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train GECOFull + TFT on mangrove_all.csv

Example:
    python train_geco.py \
      --csv_path data/mangrove_all.csv \
      --input_length 12 \
      --forecast_horizon 3 \
      --batch_size 64 \
      --epochs 50 \
      --lr 1e-3 \
      --checkpoint checkpoints/geco_full_tft.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from geco.dataset import MangroveWindowDataset
from geco.graph_builder import (
    build_static_and_dynamic_stats,
    build_base_ecological_adjacency,
    build_seasonal_adjacencies,
)
from geco.model_geco_full import GECOFull
from geco.utils import train_epoch, eval_epoch
from geco.losses import QUANTILES


def parse_args():
    parser = argparse.ArgumentParser()

    # data & basic model setup
    parser.add_argument("--csv_path", type=str, default="data/mangrove_all.csv")
    parser.add_argument("--input_length", type=int, default=12,
                        help="Lookback window L (months)")
    parser.add_argument("--forecast_horizon", type=int, default=3,
                        help="Forecast horizon H (steps ahead)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)

    # graph hyperparameters
    parser.add_argument("--lambda_geo", type=float, default=1.0)
    parser.add_argument("--lambda_env", type=float, default=1.0)
    parser.add_argument("--sigma_geo", type=float, default=1.0)
    parser.add_argument("--sigma_env", type=float, default=1.0)
    parser.add_argument("--sigma_dyn", type=float, default=1.0)
    parser.add_argument("--k_neighbors_base", type=int, default=5)
    parser.add_argument("--k_neighbors_seasonal", type=int, default=5)
    parser.add_argument("--num_seasons", type=int, default=4,
                        help="Number of seasonal graphs (e.g. 4 = DJF/MAM/JJA/SON)")

    # model dims
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_out_dim", type=int, default=64)
    parser.add_argument("--temporal_hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)

    # regularization
    parser.add_argument("--lambda_lap", type=float, default=1e-3)
    parser.add_argument("--lambda_eco", type=float, default=1e-3)

    # checkpoint
    parser.add_argument("--checkpoint", type=str, default="checkpoints/geco_full_tft.pt")

    return parser.parse_args()


def main():
    args = parse_args()

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------
    print(f"[INFO] Loading data from {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    required_cols = ["site_id", "time_idx", "ndvi", "latitude", "longitude"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # encode site ids
    site_ids = sorted(df["site_id"].unique().tolist())
    site_id_to_idx = {sid: i for i, sid in enumerate(site_ids)}
    num_sites = len(site_ids)
    print(f"[INFO] Number of sites: {num_sites}")

    # ------------------------------------------------------
    # 2. Define which columns are used for graph stats
    # ------------------------------------------------------
    # static descriptors for graph (besides lat/lon which are used internally)
    default_static_cols = ["area"]
    static_cols = [c for c in default_static_cols if c in df.columns]

    # dynamic drivers for ecological similarity (for Mu & seasonal anomalies)
    default_dyn_for_graph = [
        "precipitation",
        "salinity",
        "sea_surface_height",
        "soil_moisture",
        "ocean_current_speed",
        "ocean_cur_dir_sin",
        "ocean_cur_dir_cos",
        "wind_speed",
        "wind_direction_sin",
        "wind_direction_cos",
    ]
    dyn_for_graph = [c for c in default_dyn_for_graph if c in df.columns]
    print(f"[INFO] Static cols for graph: {static_cols}")
    print(f"[INFO] Dynamic cols for graph: {dyn_for_graph}")

    # ------------------------------------------------------
    # 3. Build node initialization + base & seasonal graphs
    # ------------------------------------------------------
    print("[INFO] Building static initialization and distance matrices...")
    static_init, d_geo, d_env = build_static_and_dynamic_stats(
        df,
        site_id_to_idx,
        static_cols=static_cols,
        dynamic_cols=dyn_for_graph,
    )

    print("[INFO] Building base ecological adjacency...")
    base_adj_norm = build_base_ecological_adjacency(
        d_geo=d_geo,
        d_env=d_env,
        lambda_geo=args.lambda_geo,
        lambda_env=args.lambda_env,
        sigma_geo=args.sigma_geo,
        sigma_env=args.sigma_env,
        k_neighbors=args.k_neighbors_base,
    )

    print("[INFO] Building seasonal adjacencies...")
    seasonal_adjs = build_seasonal_adjacencies(
        df,
        site_id_to_idx,
        dynamic_cols=dyn_for_graph,
        num_seasons=args.num_seasons,
        sigma_dyn=args.sigma_dyn,
        k_neighbors=args.k_neighbors_seasonal,
    )
    print(f"[INFO] Built {len(seasonal_adjs)} seasonal graphs.")

    # move graph stuff to device later via model; static_init/base_adj stay on CPU here

    # ------------------------------------------------------
    # 4. Build dataset (sliding windows)
    # ------------------------------------------------------
    print("[INFO] Building window dataset...")
    ds = MangroveWindowDataset(
        df=df,
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        site_id_to_idx=site_id_to_idx,
        target_col="ndvi",
    )

    dyn_cols_dataset = ds.dynamic_cols
    print(f"[INFO] Dynamic feature columns used in temporal module ({len(dyn_cols_dataset)}):")
    print(dyn_cols_dataset)

    # simple random split train/val over windows
    n_total = len(ds)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    print(f"[INFO] Train samples: {n_train}, Val samples: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    dyn_input_dim = len(dyn_cols_dataset)

    # ------------------------------------------------------
    # 5. Build model
    # ------------------------------------------------------
    print("[INFO] Initializing GECOFull model...")
    model = GECOFull(
        static_init=static_init.to(device),
        base_adj_norm=base_adj_norm.to(device),
        seasonal_adjs=[A.to(device) for A in seasonal_adjs],
        dyn_input_dim=dyn_input_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_out_dim=args.gnn_out_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        forecast_horizon=args.forecast_horizon,
        quantiles=QUANTILES,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    best_val_rmse = float("inf")

    print("[INFO] Start training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            device=device,
            dyn_cols=dyn_cols_dataset,
            lambda_lap=args.lambda_lap,
            lambda_eco=args.lambda_eco,
            target_col="ndvi",
            optimizer=optimizer,
            quantiles=QUANTILES,
        )

        r2, mae, rmse = eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            quantiles=QUANTILES,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_R2={r2:.4f} | val_MAE={mae:.4f} | val_RMSE={rmse:.4f}"
        )

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_length": args.input_length,
                    "forecast_horizon": args.forecast_horizon,
                    "dyn_input_dim": dyn_input_dim,
                    "gnn_hidden_dim": args.gnn_hidden_dim,
                    "gnn_out_dim": args.gnn_out_dim,
                    "temporal_hidden_dim": args.temporal_hidden_dim,
                    "num_heads": args.num_heads,
                    "num_seasons": args.num_seasons,
                    "lambda_geo": args.lambda_geo,
                    "lambda_env": args.lambda_env,
                    "sigma_geo": args.sigma_geo,
                    "sigma_env": args.sigma_env,
                    "sigma_dyn": args.sigma_dyn,
                    "k_neighbors_base": args.k_neighbors_base,
                    "k_neighbors_seasonal": args.k_neighbors_seasonal,
                    "lambda_lap": args.lambda_lap,
                    "lambda_eco": args.lambda_eco,
                    "target_col": "ndvi",
                    "quantiles": QUANTILES,
                },
                "site_id_to_idx": site_id_to_idx,
                "dyn_cols": dyn_cols_dataset,
            }
            torch.save(ckpt, args.checkpoint)
            print(f"[INFO] Saved best checkpoint to {args.checkpoint}")

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()