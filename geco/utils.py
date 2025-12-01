import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List

from .losses import quantile_loss, eco_salinity_loss, laplacian_smoothness_loss, QUANTILES


def train_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    dyn_cols: List[str],
    lambda_lap: float,
    lambda_eco: float,
    target_col: str,
    optimizer: torch.optim.Optimizer,
    quantiles: List[float] = None,
) -> float:
    """
    Train one epoch of GECOFull.

    Returns: average total loss over samples.
    """
    if quantiles is None:
        quantiles = QUANTILES

    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        x = batch["x"].to(device)         # [B,L,D]
        y = batch["y"].to(device)         # [B,H]
        x_last = batch["x_last"].to(device)
        site_idx = batch["site_idx"].to(device)

        optimizer.zero_grad()
        y_hat_q, z_all = model(x, site_idx)    # [B,H,Q], [N,d_z]

        ql = quantile_loss(y, y_hat_q, quantiles)
        lap = laplacian_smoothness_loss(z_all, model.base_adj_norm)
        eco = eco_salinity_loss(
            y_pred_q=y_hat_q,
            x_last=x_last,
            dyn_cols=dyn_cols,
            target_col=target_col,
            sal_col="salinity",
        )

        loss = ql + lambda_lap * lap + lambda_eco * eco
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    return total_loss / max(1, n_samples)


@torch.no_grad()
def eval_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    quantiles: List[float] = None,
):
    """
    Evaluate model using median quantile (τ ~ 0.5) predictions.

    Returns:
      R^2, MAE, RMSE
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    if quantiles is None:
        quantiles = QUANTILES

    model.eval()
    all_y = []
    all_pred = []

    # index of median quantile
    med_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        site_idx = batch["site_idx"].to(device)

        y_hat_q, _ = model(x, site_idx)        # [B,H,Q]
        y_med = y_hat_q[:, :, med_idx]         # [B,H]

        all_y.append(y.cpu().numpy().reshape(-1))
        all_pred.append(y_med.cpu().numpy().reshape(-1))

    if not all_y:
        return float("nan"), float("nan"), float("nan")

    all_y = np.concatenate(all_y, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    r2 = r2_score(all_y, all_pred)
    mae = mean_absolute_error(all_y, all_pred)
    rmse = math.sqrt(mean_squared_error(all_y, all_pred))
    return r2, mae, rmse