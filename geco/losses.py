import torch
from typing import List

# Default quantiles used across the project
QUANTILES: List[float] = [0.1, 0.5, 0.9]


def quantile_loss(
    y_true: torch.Tensor,    # [B,H]
    y_pred_q: torch.Tensor,  # [B,H,Q]
    quantiles: List[float],
) -> torch.Tensor:
    """
    Standard quantile regression loss, averaged over (batch, horizon, quantile).
    """
    B, H = y_true.shape
    Q = len(quantiles)

    y_true_exp = y_true.unsqueeze(-1).expand(B, H, Q)
    u = y_true_exp - y_pred_q  # [B,H,Q]

    losses = []
    for qi, tau in enumerate(quantiles):
        u_tau = u[:, :, qi]
        loss_tau = torch.max(tau * u_tau, (tau - 1) * u_tau)
        losses.append(loss_tau)
    loss_all = torch.stack(losses, dim=-1)  # [B,H,Q]
    return loss_all.mean()


def eco_salinity_loss(
    y_pred_q: torch.Tensor,     # [B,H,Q]
    x_last: torch.Tensor,       # [B,D]
    dyn_cols: List[str],
    target_col: str = "ndvi",
    sal_col: str = "salinity",
    sal_thr: float = 35.0,
    alpha_sal: float = 0.1,
) -> torch.Tensor:
    """
    Eco-constrained loss:
      penalizes cases where, under very high salinity,
      NDVI forecast increases compared to last observed NDVI.

      phi_sal = max(0, alpha * (s - s_thr) * (y_hat - y_prev))
    """
    if sal_col not in dyn_cols or target_col not in dyn_cols:
        return torch.tensor(0.0, device=y_pred_q.device)

    target_idx = dyn_cols.index(target_col)
    sal_idx = dyn_cols.index(sal_col)

    y_prev = x_last[:, target_idx]  # [B]
    s = x_last[:, sal_idx]          # [B]

    Q = y_pred_q.size(-1)
    med_idx = Q // 2
    y_med = y_pred_q[:, 0, med_idx]  # [B] (use horizon=1 median)

    cond = s - sal_thr
    violation = alpha_sal * cond * (y_med - y_prev)
    penalty = torch.clamp(violation, min=0.0)
    return penalty.mean()


def laplacian_smoothness_loss(z: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    """
    Laplacian smoothness:
        L_lap = sum_i || z_i - sum_j A_ij z_j ||^2

    Encourages node embeddings to be close to their graph-smoothed neighbors.
    """
    z_neigh = torch.matmul(adj_norm, z)  # [N,d]
    diff = z - z_neigh
    return (diff ** 2).sum()