import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict


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
    ):
        super().__init__()
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col

        # Sort by site/time
        df = df.sort_values(["site_id", "time_idx"]).reset_index(drop=True)
        self.df = df

        # Numeric columns for dynamic features
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if "time_idx" in numeric_cols:
            numeric_cols.remove("time_idx")
        if target_col not in numeric_cols:
            raise ValueError(f"Target column '{target_col}' must be numeric and present.")
        self.dynamic_cols = numeric_cols

        self.site_id_to_idx = site_id_to_idx

        # Build per-site time series: (site_idx, arr_dyn, time_idx)
        self.series = []
        for sid, g in df.groupby("site_id"):
            arr = g[self.dynamic_cols].to_numpy(dtype=np.float32)
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
        }