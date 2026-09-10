"""
model_geco_v2.py — architecture fix for GECO's temporal branch.

The original TemporalVSN collapses all D features into a single scalar per
timestep (x_tilde: [B, L, 1]) before the GRU, discarding almost all
multivariate information. This module implements proper TFT-style variable
selection: every feature is projected into an embedding space and the selection
weights combine those embeddings *in that space* (output: [B, L, hidden_dim]).

Reuses the seasonal GAT encoder and helper blocks from model_geco_full.
"""

from typing import List

import torch
from torch import nn

from .losses import QUANTILES as DEFAULT_QUANTILES
from .model_geco_full import (
    SeasonalGATEncoder,
    GRN,
    StaticVSN,
    TemporalBackboneTFT,
)


class TemporalVSNv2(nn.Module):
    """
    Proper TFT-style temporal variable selection.

      - Each feature k -> its own embedding e_{t,k} in R^{hidden}
      - Selection weights v_{t,k} = softmax_k(GRN(x_t, context))
      - Combined signal xi_t = sum_k v_{t,k} * e_{t,k}   (in embedding space)

    Output: [B, L, hidden]  (NOT collapsed to a scalar).
    """

    def __init__(self, num_features: int, hidden_dim: int, context_dim: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # one linear embedding per feature (1 -> hidden)
        self.feat_proj = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(num_features)]
        )
        # variable-selection weights from the full feature vector + static context
        self.vs_grn = GRN(num_features, hidden_dim, context_dim=context_dim)
        self.weight_proj = nn.Linear(num_features, num_features)

    def forward(self, x: torch.Tensor, c_stat: torch.Tensor):
        """
        x: [B, L, D], c_stat: [B, C]
        return: xi [B, L, hidden], weights [B, L, D]
        """
        B, L, D = x.shape

        c_exp = c_stat.unsqueeze(1).expand(B, L, c_stat.size(-1)).reshape(B * L, -1)
        flat = x.reshape(B * L, D)
        h = self.vs_grn(flat, c_exp)                 # [B*L, D]
        logits = self.weight_proj(h)                 # [B*L, D]
        w = torch.softmax(logits, dim=-1).view(B, L, D, 1)   # [B, L, D, 1]

        # per-feature embeddings -> [B, L, D, hidden]
        embs = torch.stack(
            [self.feat_proj[k](x[..., k:k + 1]) for k in range(D)], dim=2
        )
        xi = (w * embs).sum(dim=2)                   # [B, L, hidden]
        return xi, w.squeeze(-1)


class TFTTemporalModuleV2(nn.Module):
    def __init__(
        self,
        dyn_input_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        forecast_horizon: int = 3,
        quantiles: List[float] = None,
        num_heads: int = 4,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = DEFAULT_QUANTILES
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.static_vsn = StaticVSN(z_dim, hidden_dim, hidden_dim)
        self.temporal_vsn = TemporalVSNv2(dyn_input_dim, hidden_dim, hidden_dim)
        self.backbone = TemporalBackboneTFT(
            input_dim=hidden_dim,           # <-- now embedding-width, not 1
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            forecast_horizon=forecast_horizon,
            num_quantiles=self.num_quantiles,
        )

    def forward(self, x_seq: torch.Tensor, z_batch: torch.Tensor) -> torch.Tensor:
        c_stat = self.static_vsn(z_batch)
        xi, _ = self.temporal_vsn(x_seq, c_stat)     # [B, L, hidden]
        return self.backbone(xi)                     # [B, H, Q]


class GECOFullV2(nn.Module):
    """Same as GECOFull but with the fixed temporal module."""

    def __init__(
        self,
        static_init: torch.Tensor,
        base_adj_norm: torch.Tensor,
        seasonal_adjs: List[torch.Tensor],
        dyn_input_dim: int,
        gnn_hidden_dim: int = 64,
        gnn_out_dim: int = 64,
        temporal_hidden_dim: int = 128,
        forecast_horizon: int = 3,
        quantiles: List[float] = None,
        num_heads: int = 4,
    ):
        super().__init__()
        self.register_buffer("static_init", static_init)
        self.register_buffer("base_adj_norm", base_adj_norm)
        self.seasonal_adjs = nn.ParameterList(
            [nn.Parameter(adj, requires_grad=False) for adj in seasonal_adjs]
        )
        self.num_sites = static_init.size(0)
        self.num_seasons = len(seasonal_adjs)

        self.spatial_encoder = SeasonalGATEncoder(
            in_dim=static_init.size(-1),
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_layers=2,
            num_seasons=self.num_seasons,
        )
        self.temporal_module = TFTTemporalModuleV2(
            dyn_input_dim=dyn_input_dim,
            z_dim=gnn_out_dim,
            hidden_dim=temporal_hidden_dim,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            num_heads=num_heads,
        )

    def forward(self, x_seq: torch.Tensor, site_idx: torch.Tensor):
        adjs = [adj for adj in self.seasonal_adjs]
        z_all = self.spatial_encoder(self.static_init, adjs)
        z_batch = z_all[site_idx]
        y_hat_q = self.temporal_module(x_seq, z_batch)
        return y_hat_q, z_all
