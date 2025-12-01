import torch
from torch import nn
from typing import List

from .losses import QUANTILES as DEFAULT_QUANTILES


# ============================================================
# GAT layers & seasonal encoder
# ============================================================

class GATLayer(nn.Module):
    """
    Single-head GAT layer (simplified, using dense adjacency).
    """

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h: [N, F_in]
        adj: [N, N] (normalized adjacency; >0 means edge)
        """
        Wh = self.W(h)                      # [N, F_out]
        N = Wh.size(0)
        device = h.device

        # attention logits e_ij, init -inf to mask non-edges
        e = torch.empty(N, N, device=device)
        e.fill_(float("-inf"))

        idx_i, idx_j = (adj > 0).nonzero(as_tuple=True)
        for i, j in zip(idx_i.tolist(), idx_j.tolist()):
            a_input = torch.cat([Wh[i], Wh[j]], dim=-1)  # [2F]
            e_ij = self.leaky_relu(self.a(a_input))      # scalar
            e[i, j] = e_ij

        alpha = torch.softmax(e, dim=1)   # [N,N]
        h_prime = torch.matmul(alpha, Wh) # [N,F_out]
        return torch.relu(h_prime)


class SeasonalGATEncoder(nn.Module):
    """
    Seasonal GAT encoder:
      - One GAT stack per seasonal graph.
      - Node-conditioned attention to fuse seasonal embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_seasons: int = 4,
    ):
        super().__init__()
        self.num_seasons = num_seasons

        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GATLayer(
                    in_dim if l == 0 else hidden_dim,
                    out_dim if l == num_layers - 1 else hidden_dim,
                )
                for l in range(num_layers)
            ])
            for _ in range(num_seasons)
        ])

        # seasonal fusion query vector q
        self.q = nn.Parameter(torch.randn(out_dim))

    def forward(self, h0: torch.Tensor, seasonal_adjs: List[torch.Tensor]) -> torch.Tensor:
        """
        h0: [N, d_init]
        seasonal_adjs: list of [N,N], len = num_seasons
        return: z: [N, d_out]
        """
        N = h0.size(0)
        seasonal_embeddings = []

        for tau in range(self.num_seasons):
            A_tau = seasonal_adjs[tau].to(h0.device)
            h_tau = h0
            for layer in self.gat_layers[tau]:
                h_tau = layer(h_tau, A_tau)     # [N,d]
            seasonal_embeddings.append(h_tau)

        # [N, T_s, d_out]
        H = torch.stack(seasonal_embeddings, dim=1)
        # scores: [N,T_s]
        scores = torch.einsum("d,ntd->nt", self.q, H)
        beta = torch.softmax(scores, dim=1)      # [N,T_s]
        beta_exp = beta.unsqueeze(-1)            # [N,T_s,1]

        z = (beta_exp * H).sum(dim=1)           # [N,d_out]
        return z


# ============================================================
# TFT-style temporal module
# ============================================================

class GRN(nn.Module):
    """
    Gated Residual Network (used in VSNs and post-attention blocks).
    """

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(input_dim)

        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        else:
            self.context_proj = None

        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.fc2(h)

        g = self.gate(h)
        x_gated = g * h
        return self.layer_norm(residual + x_gated)


class StaticVSN(nn.Module):
    """
    Static Variable Selection Network on spatial embedding z_i.
    Produces a context vector for temporal processing.
    """

    def __init__(self, z_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.grn = GRN(z_dim, hidden_dim)
        self.proj = nn.Linear(z_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B,d_z] -> c_stat: [B,out_dim]
        """
        z_grn = self.grn(z)
        c_stat = self.proj(z_grn)
        return c_stat


class TemporalVSN(nn.Module):
    """
    Temporal Variable Selection Network.
    Simplified version that collapses features -> 1-d per timestep.

    For each (t, feature k), runs GRN(x_{t,k}, context),
    then softmax over features to get feature weights.
    """

    def __init__(self, num_features: int, hidden_dim: int, context_dim: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.feature_grn = GRN(1, hidden_dim, context_dim=context_dim)
        self.weight_proj = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, c_stat: torch.Tensor):
        """
        x: [B,L,D]
        c_stat: [B,C]
        return:
          x_tilde: [B,L,1]
          weights: [B,L,D]
        """
        B, L, D = x.shape
        x_flat = x.view(B * L * D, 1)  # [B*L*D,1]

        # broadcast static context
        c_exp = c_stat.unsqueeze(1).unsqueeze(2).expand(B, L, D, c_stat.size(-1))
        c_flat = c_exp.reshape(B * L * D, -1)

        h = self.feature_grn(x_flat, c_flat)   # [B*L*D,1]
        logits = self.weight_proj(h)           # [B*L*D,1]
        logits = logits.view(B, L, D)

        weights = torch.softmax(logits, dim=-1)   # [B,L,D]
        x_tilde = (weights * x).sum(dim=-1, keepdim=True)  # [B,L,1]

        return x_tilde, weights


class TemporalBackboneTFT(nn.Module):
    """
    GRU + Multi-head attention + projection to multi-horizon / multi-quantile.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        forecast_horizon: int,
        num_quantiles: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_quantiles = num_quantiles

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.grn_post_attn = GRN(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, forecast_horizon * num_quantiles)

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        x_tilde: [B,L,d_in]
        """
        enc_out, _ = self.gru(x_tilde)          # [B,L,H]
        attn_out, _ = self.attn(enc_out, enc_out, enc_out)  # [B,L,H]
        attn_out = self.grn_post_attn(attn_out)

        h_last = attn_out[:, -1, :]             # [B,H]
        out = self.proj(h_last)                 # [B,H_out*Q]
        B = out.size(0)
        out = out.view(B, self.forecast_horizon, self.num_quantiles)
        return out  # [B,H,Q]


class TFTTemporalModule(nn.Module):
    """
    Full TFT-style temporal module used inside GECO:
      - Static VSN on spatial embedding z_i
      - Temporal VSN on x_{i,t} with context
      - GRU + Multi-head attention
      - Multi-horizon, multi-quantile output
    """

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
        self.temporal_vsn = TemporalVSN(dyn_input_dim, hidden_dim, hidden_dim)
        self.backbone = TemporalBackboneTFT(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            forecast_horizon=forecast_horizon,
            num_quantiles=self.num_quantiles,
        )

    def forward(self, x_seq: torch.Tensor, z_batch: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B,L,D]
        z_batch: [B,d_z]
        return: y_hat_q [B,H,Q]
        """
        c_stat = self.static_vsn(z_batch)          # [B,Hd]
        x_tilde, _ = self.temporal_vsn(x_seq, c_stat)  # [B,L,1]
        y_hat_q = self.backbone(x_tilde)           # [B,H,Q]
        return y_hat_q


# ============================================================
# GECOFull model = Seasonal GAT + TFT Temporal Module
# ============================================================

class GECOFull(nn.Module):
    """
    Full GECO model:
      - static_init: node init states h_i^(0) = [s_i || mu(x_i,.)]
      - base_adj_norm: normalized ecological adjacency (for Laplacian loss)
      - seasonal_adjs: list of normalized seasonal adjacency matrices
      - dyn_input_dim: number of dynamic features per timestep
    """

    def __init__(
        self,
        static_init: torch.Tensor,         # [N,d_init]
        base_adj_norm: torch.Tensor,       # [N,N]
        seasonal_adjs: List[torch.Tensor], # list of [N,N]
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

        # store seasonal adjacencies as non-trainable parameters
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

        self.temporal_module = TFTTemporalModule(
            dyn_input_dim=dyn_input_dim,
            z_dim=gnn_out_dim,
            hidden_dim=temporal_hidden_dim,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            num_heads=num_heads,
        )

    def forward(self, x_seq: torch.Tensor, site_idx: torch.Tensor):
        """
        x_seq: [B,L,D]
        site_idx: [B]
        """
        adjs = [adj for adj in self.seasonal_adjs]
        z_all = self.spatial_encoder(self.static_init, adjs)    # [N,d_z]
        z_batch = z_all[site_idx]                               # [B,d_z]
        y_hat_q = self.temporal_module(x_seq, z_batch)          # [B,H,Q]
        return y_hat_q, z_all