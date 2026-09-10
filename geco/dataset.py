import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class MangroveWindowDataset(Dataset):
    """
    Sliding-window multivariate time-series dataset grouped by site_id.

    For each site i:
        Input:  x_{i, t-L+1:t}  (L = input_length)  -> [L, D]
        Target: y_{i, t+1:t+H} (H = forecast_horizon) -> [H] (scalar NDVI)

    Expected columns in dataframe:
      - 'site_id'
      - 'time_idx' (integer, increasing)
      - 'ndvi' (target)
      - numeric covariates (float) as dynamic drivers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_length: int,
        forecast_horizon: int,
        site_id_to_idx: Dict,
        target_col: str = "ndvi",
        exclude_cols: Optional[List[str]] = None,
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col

        # Columns that are numeric but should NOT be temporal drivers
        # (constant per site => no temporal signal, only add noise).
        if exclude_cols is None:
            exclude_cols = ["time_idx", "latitude", "longitude"]

        # Sort by site/time
        df = df.sort_values(["site_id", "time_idx"]).reset_index(drop=True)
        self.df = df

        # Numeric columns for dynamic features (pandas-native check so it also
        # handles the modern StringDtype used for object columns like site_id/date).
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
        ]
        if target_col not in numeric_cols:
            raise ValueError(f"Target column '{target_col}' must be numeric and present.")
        self.dynamic_cols = numeric_cols

        self.site_id_to_idx = site_id_to_idx

        # Optional per-feature standardization (stats must be fit on TRAIN only).
        self.norm_stats = norm_stats
        if norm_stats is not None:
            self.feat_mean = np.array(
                [norm_stats[c][0] for c in self.dynamic_cols], dtype=np.float32
            )
            self.feat_std = np.array(
                [norm_stats[c][1] for c in self.dynamic_cols], dtype=np.float32
            )
            tgt_i = self.dynamic_cols.index(target_col)
            self.target_mean = float(self.feat_mean[tgt_i])
            self.target_std = float(self.feat_std[tgt_i])
        else:
            self.feat_mean = None
            self.feat_std = None
            self.target_mean = 0.0
            self.target_std = 1.0

        # Build per-site time series: (site_idx, arr_dyn, time_idx)
        self.series = []
        for sid, g in df.groupby("site_id"):
            arr = g[self.dynamic_cols].to_numpy(dtype=np.float32)
            if self.feat_mean is not None:
                arr = (arr - self.feat_mean) / self.feat_std
            times = g["time_idx"].to_numpy()
            s_idx = site_id_to_idx[sid]
            self.series.append((s_idx, arr, times))

        # Global index: (series_id, start_pos)
        self.index = []
        L = input_length
        H = forecast_horizon
        for s_id, (_, arr, times) in enumerate(self.series):
            T = arr.shape[0]
            max_start = T - (L + H)
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                self.index.append((s_id, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_series_idx, start = self.index[idx]
        site_idx, arr, times = self.series[s_series_idx]
        L = self.input_length
        H = self.forecast_horizon

        x_seq = arr[start:start + L]  # [L,D]
        y_seq = arr[start + L:start + L + H,
                    self.dynamic_cols.index(self.target_col)]  # [H]
        x_last = x_seq[-1]  # [D]

        return {
            "x": torch.from_numpy(x_seq),           # [L,D]
            "y": torch.from_numpy(y_seq),           # [H]
            "x_last": torch.from_numpy(x_last),     # [D]
            "site_idx": torch.tensor(site_idx, dtype=torch.long),
            "t0": torch.tensor(int(times[start + L]), dtype=torch.long),  # first target time_idx
        }