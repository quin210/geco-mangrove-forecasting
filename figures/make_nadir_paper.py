#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper composite: nadir canopy of 3 sites over time + a same-period
ground-truth-vs-forecast column for a held-out 2025 month.

Note: the model forecasts the SITE-MEAN NDVI (one value), not per pixel — so the
'forecast' cell is a uniform tile in the same colour scale; the observed cell
keeps its full pixel pattern. Both annotate their spatial/scalar mean.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#1b2733", "#5b6b7a"
CTX_MONTHS = ["2016-03", "2020-03", "2024-03"]      # observed context (low-cloud month)
HELD = "2025-03"                                     # held-out target month
SITES = [("Sundarbans", "SouthAsia_Sundarbans_IN", "Sundarbans (IN/BD)"),
         ("Carpentaria", "Australia_Carpentaria", "Gulf of Carpentaria (AU)"),
         ("MekongDelta", "SEAsia_MekongDelta", "Mekong delta (VN)")]
VMAX = 0.8


def main():
    preds = pd.read_csv("data/processed/future_preds.csv")
    preds["ym"] = pd.to_datetime(preds["date"], format="%m/%d/%Y", errors="coerce").dt.to_period("M").astype(str)
    cols = CTX_MONTHS + [HELD, HELD]
    ncol = len(cols)
    fig, axes = plt.subplots(len(SITES), ncol, figsize=(2.05 * ncol, 2.2 * len(SITES) + 0.6), dpi=200)

    for ri, (rk, mid, rlabel) in enumerate(SITES):
        z = np.load(f"figures/raster/{rk}.npz", allow_pickle=True)
        G = z["grids"]; months = [str(m) for m in z["months"]]

        def grid_at(ym):
            return G[months.index(ym)] if ym in months else np.full_like(G[0], np.nan)

        # context + held-out observed
        for ci, ym in enumerate(CTX_MONTHS + [HELD]):
            ax = axes[ri, ci]
            ax.imshow(grid_at(ym), cmap="YlGn", vmin=0, vmax=VMAX, interpolation="nearest")
            m = np.nanmean(grid_at(ym))
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(("observed  " + ym) if ci < len(CTX_MONTHS) else f"GROUND TRUTH\n{ym}",
                             fontsize=8.5, color=INK)
            ax.text(0.5, -0.08, f"mean {m:.2f}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=7.5, color=MUTED)
            for s in ax.spines.values():
                s.set_edgecolor("#bbb")

        # forecast cell (uniform tile at predicted site-mean)
        ax = axes[ri, ncol - 1]
        p = preds[(preds.site_id == mid) & (preds.ym == HELD)]   # any horizon
        pred = float(p["q50"].mean()) if len(p) else np.nan
        obs = float(p["y_true"].mean()) if len(p) else np.nan
        ax.imshow(np.full((10, 10), pred), cmap="YlGn", vmin=0, vmax=VMAX)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(f"FORECAST\n{HELD}", fontsize=8.5, color="#B2182B")
        ax.text(0.5, -0.08, f"pred {pred:.2f}\n(obs {obs:.2f})", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=MUTED)
        for s in ax.spines.values():
            s.set_edgecolor("#B2182B"); s.set_linewidth(1.4)

        axes[ri, 0].set_ylabel(rlabel, fontsize=9.5, color=INK)

    sm = plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, VMAX))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.015, location="right")
    cb.set_label("NDVI (canopy greenness)", fontsize=9)
    fig.suptitle("Nadir mangrove canopy (MODIS 250 m, GEE-free): spatiotemporal change and a held-out forecast",
                 fontsize=12, color=INK, y=1.0)
    fig.text(0.5, 0.005, "Forecast cell = model site-mean NDVI (uniform tile); observed cells keep the pixel pattern.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    fig.savefig("figures/fig_nadir_paper.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_nadir_paper.png")


if __name__ == "__main__":
    main()
