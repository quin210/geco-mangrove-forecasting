"""
physics_losses.py — Eco-Physically Informed (EPI) constraints for GECO.

Each constraint is a differentiable soft penalty. All operate in the model's
(normalized) output space; thresholds must be supplied already expressed in that
same normalized space (see train_geco_v2.py for the conversion from raw units).

Design principle: only impose constraints supported by mangrove eco-physiology
AND consistent with the observed data. The original `eco_salinity_loss`
(high salinity -> NDVI must not rise) is intentionally NOT used here because in
this Red Sea dataset salinity varies little (37-40 PSU), the species (Avicennia
marina) is highly salt-tolerant, and corr(salinity anomaly, next-month dNDVI)
is slightly positive — so that constraint would fight the data.

Constraints:
  1. quantile_crossing_loss   — enforce q_0.1 <= q_0.5 <= q_0.9 (probabilistic consistency)
  2. temporal_rate_loss       — canopy inertia: |NDVI_{t+1}-NDVI_t| bounded
  3. bounds_loss              — NDVI stays in a physical interval
  4. water_stress_loss        — under sustained low water availability, NDVI should not rise
"""

import torch


def quantile_crossing_loss(y_pred_q: torch.Tensor) -> torch.Tensor:
    """
    y_pred_q: [B, H, Q] with quantiles assumed sorted ascending (e.g. 0.1,0.5,0.9).
    Penalize any violation q_{k} > q_{k+1}.
    """
    diffs = y_pred_q[..., 1:] - y_pred_q[..., :-1]      # should be >= 0
    return torch.clamp(-diffs, min=0.0).mean()


def temporal_rate_loss(
    y_pred_q: torch.Tensor,   # [B, H, Q]
    y_last: torch.Tensor,     # [B]  last observed (normalized) NDVI
    max_delta: float,         # max plausible monthly change (normalized units)
    med_idx: int = 1,
) -> torch.Tensor:
    """
    Canopy-inertia prior on the MEDIAN trajectory.
    Builds [y_last, y_1, ..., y_H] and penalizes step changes beyond max_delta.
    """
    med = y_pred_q[:, :, med_idx]                        # [B, H]
    traj = torch.cat([y_last.unsqueeze(1), med], dim=1)  # [B, H+1]
    step = (traj[:, 1:] - traj[:, :-1]).abs()            # [B, H]
    excess = torch.clamp(step - max_delta, min=0.0)
    return (excess ** 2).mean()


def bounds_loss(
    y_pred_q: torch.Tensor,   # [B, H, Q]
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Hinge penalty for predictions outside the physical interval [lo, hi]."""
    below = torch.clamp(lo - y_pred_q, min=0.0)
    above = torch.clamp(y_pred_q - hi, min=0.0)
    return (below + above).mean()


def water_stress_loss(
    y_pred_q: torch.Tensor,   # [B, H, Q]
    stress_index: torch.Tensor,  # [B]  high = strong water stress (normalized, >0)
    y_last: torch.Tensor,     # [B]  last observed NDVI (normalized)
    stress_thr: float = 0.5,
    alpha: float = 1.0,
    med_idx: int = 1,
) -> torch.Tensor:
    """
    Multi-driver water-stress constraint (replaces the salinity-only version).
    Under strong water stress (stress_index > stress_thr), a forecast that
    INCREASES NDVI vs the last observation is penalized.

      phi = max(0, alpha * (stress - stress_thr) * (y_hat - y_prev))

    `stress_index` should combine drought signals, e.g. negative soil-moisture
    and negative precipitation anomalies (built in the training script).
    """
    y_med = y_pred_q[:, 0, med_idx]                      # horizon-1 median
    cond = stress_index - stress_thr
    violation = alpha * cond * (y_med - y_last)
    return torch.clamp(violation, min=0.0).mean()


# ----------------------------------------------------------------------
# Physical-consistency diagnostics (for evaluation tables, not training).
# Operate on ORIGINAL-scale predictions. Return fractions in [0,1].
# ----------------------------------------------------------------------

def consistency_metrics(y_pred_q_np, y_true_np, med_idx, lo, hi, max_delta_raw):
    """
    y_pred_q_np: [M, Q] predictions (original scale) for flattened (b,h) pairs,
                 already ordered by quantile.
    y_true_np:   [M] true values (original scale).
    Returns dict of violation/coverage rates.
    """
    import numpy as np
    q = y_pred_q_np
    med = q[:, med_idx]
    cross = np.mean(np.any(np.diff(q, axis=1) < 0, axis=1))
    oob = np.mean((med < lo) | (med > hi))
    # 80% coverage between outermost quantiles
    cover = np.mean((y_true_np >= q[:, 0]) & (y_true_np <= q[:, -1]))
    return {
        "quantile_crossing_rate": float(cross),
        "out_of_bounds_rate": float(oob),
        "coverage_80": float(cover),
    }
