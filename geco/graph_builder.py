import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple


def gaussian_kernel(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(- (d ** 2) / (sigma ** 2 + 1e-8))


def build_static_and_dynamic_stats(
    df: pd.DataFrame,
    site_id_to_idx: Dict,
    static_cols: List[str],
    dynamic_cols: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build initial node features h_i^(0) = [s_i || mu(x_i,.)]
    and distance matrices for geo and env:

      - S_i: static descriptors for each site (mean over time)
      - mu_i: mean dynamic drivers for each site

    Returns:
      static_init: [N,d_init]
      d_geo: [N,N] (geo distance)
      d_env: [N,N] (env distance on S)
    """

    sites = sorted(site_id_to_idx.keys())
    N = len(sites)

    S_list = []
    Mu_list = []
    lats = []
    lons = []

    for sid in sites:
        sub = df[df["site_id"] == sid]
        lat = sub["latitude"].mean()
        lon = sub["longitude"].mean()
        lats.append(lat)
        lons.append(lon)

        if static_cols:
            s_vec = sub[static_cols].mean(axis=0).to_numpy(dtype=np.float32)
        else:
            s_vec = np.zeros(1, dtype=np.float32)

        if dynamic_cols:
            mu_vec = sub[dynamic_cols].mean(axis=0).to_numpy(dtype=np.float32)
        else:
            mu_vec = np.zeros(1, dtype=np.float32)

        S_list.append(s_vec)
        Mu_list.append(mu_vec)

    S = np.stack(S_list, axis=0)   # [N,d_s]
    Mu = np.stack(Mu_list, axis=0) # [N,d_x_mu]

    # Standardize S and Mu
    S = (S - S.mean(axis=0, keepdims=True)) / (S.std(axis=0, keepdims=True) + 1e-6)
    Mu = (Mu - Mu.mean(axis=0, keepdims=True)) / (Mu.std(axis=0, keepdims=True) + 1e-6)

    h0 = np.concatenate([S, Mu], axis=1)  # [N,d_init]

    # Geo distance in normalized lat/lon space
    coords = np.vstack([np.array(lats), np.array(lons)]).T  # [N,2]
    coords_norm = (coords - coords.mean(axis=0, keepdims=True)) / (
        coords.std(axis=0, keepdims=True) + 1e-6
    )
    d_geo = np.linalg.norm(coords_norm[None, :, :] - coords_norm[:, None, :], axis=-1)

    # Env distance on S (static descriptors)
    d_env = np.linalg.norm(S[None, :, :] - S[:, None, :], axis=-1)

    return (
        torch.from_numpy(h0.astype(np.float32)),
        torch.from_numpy(d_geo.astype(np.float32)),
        torch.from_numpy(d_env.astype(np.float32)),
    )


def build_base_ecological_adjacency(
    d_geo: torch.Tensor,
    d_env: torch.Tensor,
    lambda_geo: float,
    lambda_env: float,
    sigma_geo: float,
    sigma_env: float,
    k_neighbors: int,
) -> torch.Tensor:
    """
    Composite ecological adjacency (single base graph),
    then symmetric k-NN sparsification and normalized:

      A_tilde = D^{-1/2}(A + I)D^{-1/2}
    """

    d_geo_np = d_geo.cpu().numpy()
    d_env_np = d_env.cpu().numpy()

    k_geo = gaussian_kernel(d_geo_np, sigma_geo)
    k_env = gaussian_kernel(d_env_np, sigma_env)

    W = lambda_geo * k_geo + lambda_env * k_env
    np.fill_diagonal(W, 0.0)

    A = np.zeros_like(W)
    N = A.shape[0]
    for i in range(N):
        idxs = np.argsort(-W[i])  # descending by weight
        idxs = [j for j in idxs if j != i]
        idxs = idxs[:k_neighbors]
        A[i, idxs] = W[i, idxs]
    A = np.maximum(A, A.T)

    A_with_self = A + np.eye(N, dtype=np.float32)
    deg = A_with_self.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-6)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_with_self @ D_inv_sqrt

    return torch.from_numpy(A_norm.astype(np.float32))


def build_seasonal_adjacencies(
    df: pd.DataFrame,
    site_id_to_idx: Dict,
    dynamic_cols: List[str],
    num_seasons: int,
    sigma_dyn: float = 1.0,
    k_neighbors: int = 5,
) -> List[torch.Tensor]:
    """
    Build seasonal graphs as in seasonal similarity eq.:

      - Split months into num_seasons bins (DJF / MAM / JJA / SON if 4).
      - For each season:
          * compute anomalies x_tilde = x_{i,t} - mu_i
          * average over season window => seasonal anomaly vector
          * compute pairwise distances, Gaussian kernel -> k_tau
          * k-NN adjacency + symmetric + normalized.

    Returns:
      list of normalized adjacency matrices [N,N].
    """

    sites = sorted(site_id_to_idx.keys())
    N = len(sites)

    # Month-of-year
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        months = dates.dt.month.to_numpy()
    else:
        # Fallback: derive pseudo-month from time_idx
        months = (df["time_idx"].to_numpy() % 12) + 1

    df = df.copy()
    df["month"] = months

    # Mean dynamic drivers per site (for anomalies)
    mu_per_site = {}
    for sid in sites:
        sub = df[df["site_id"] == sid]
        if len(dynamic_cols) > 0:
            mu_per_site[sid] = sub[dynamic_cols].to_numpy(dtype=np.float32).mean(axis=0)
        else:
            mu_per_site[sid] = np.zeros(1, dtype=np.float32)

    # Seasonal bins: split months 1..12 into num_seasons groups
    season_bins = np.array_split(np.arange(1, 13), num_seasons)
    seasonal_adjs: List[torch.Tensor] = []

    for season_idx in range(num_seasons):
        months_in_season = set(season_bins[season_idx].tolist())

        # seasonal anomaly vector per site
        S_tau = []
        for sid in sites:
            sub = df[(df["site_id"] == sid) & (df["month"].isin(months_in_season))]
            if len(sub) == 0 or len(dynamic_cols) == 0:
                S_tau.append(np.zeros_like(mu_per_site[sid]))
            else:
                x = sub[dynamic_cols].to_numpy(dtype=np.float32)
                mu = mu_per_site[sid]
                x_tilde = x - mu
                S_tau.append(x_tilde.mean(axis=0))

        S_tau = np.stack(S_tau, axis=0)  # [N,d_dyn]
        d_tau = np.linalg.norm(S_tau[None, :, :] - S_tau[:, None, :], axis=-1)
        k_tau = gaussian_kernel(d_tau, sigma_dyn)
        np.fill_diagonal(k_tau, 0.0)

        A_tau = np.zeros_like(k_tau)
        for i in range(N):
            idxs = np.argsort(-k_tau[i])
            idxs = [j for j in idxs if j != i]
            idxs = idxs[:k_neighbors]
            A_tau[i, idxs] = k_tau[i, idxs]
        A_tau = np.maximum(A_tau, A_tau.T)

        A_tau_self = A_tau + np.eye(N, dtype=np.float32)
        deg = A_tau_self.sum(axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-6)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        A_tau_norm = D_inv_sqrt @ A_tau_self @ D_inv_sqrt

        seasonal_adjs.append(torch.from_numpy(A_tau_norm.astype(np.float32)))

    return seasonal_adjs