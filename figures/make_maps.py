#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication maps: (A) study sites by region, (B) sites by held-out-future skill."""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SITES = "data/mangrove_sites_global.csv"
FUT = "/tmp/claude-1026/-mnt-ssd5-quynh-mangrove/ddf5b72f-09d4-414a-b68e-3a03ba21f7b6/tasks/b6b31dddm.output"

# 10-region qualitative palette (distinct, print-friendly)
REGION_COLORS = {
    "RedSea": "#E69F00", "PersianGulf": "#D55E00", "SouthAsia": "#009E73",
    "SEAsia": "#0072B2", "Australia": "#CC79A7", "Pacific": "#56B4E9",
    "EastAfrica": "#117733", "WestAfrica": "#882255", "NorthAmerica": "#332288",
    "SouthAmerica": "#AA4499", "CentralAmerica": "#999933",
}


def basemap(ax):
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#EAF2F8")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F4F1EC")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="#9AA7B2")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.2, edgecolor="#CBD3DA")
    ax.set_extent([-120, 160, -40, 40], crs=ccrs.PlateCarree())


def load_future_r2():
    line = [l for l in open(FUT) if l.startswith("[PER-SITE")][0]
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"([A-Za-z_]+)=([-\d.]+)", line)}


def main():
    sites = pd.read_csv(SITES)
    r2 = load_future_r2()

    # ---- Figure A: sites by region ----
    fig = plt.figure(figsize=(11, 5.2), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    basemap(ax)
    for reg, g in sites.groupby("region"):
        ax.scatter(g.longitude, g.latitude, s=70, transform=ccrs.PlateCarree(),
                   color=REGION_COLORS.get(reg, "#666"), edgecolor="white", lw=0.8,
                   label=f"{reg} (n={len(g)})", zorder=5)
    ax.set_title("GECO-EWS study network: 40 mangrove sites across 10 biogeographic regions",
                 fontsize=12, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=8, ncol=6,
              frameon=False, markerscale=1.0, columnspacing=1.2, handletextpad=0.4)
    fig.savefig("figures/fig_map_regions.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_map_regions.png")

    # ---- Figure B: sites by held-out-future per-site R^2 ----
    sites["r2"] = sites["site_id"].map(r2)
    d = sites.dropna(subset=["r2"]).copy()
    d["r2c"] = d["r2"].clip(-1, 1)   # clip degenerate low-variance sites for colour
    fig = plt.figure(figsize=(11, 5.2), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    basemap(ax)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sc = ax.scatter(d.longitude, d.latitude, c=d.r2c, cmap="RdYlGn", norm=norm,
                    s=95, transform=ccrs.PlateCarree(), edgecolor="#333", lw=0.7, zorder=5)
    ax.set_title("Held-out future forecast (2025–2026): per-site skill (R²)\n"
                 "green = skilful · red = fails (mostly arid, near-flat NDVI)",
                 fontsize=12, pad=8)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, extend="min")
    cb.set_label("per-site R²  (clipped to [-1, 1])", fontsize=9)
    fig.savefig("figures/fig_map_skill.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_map_skill.png")


if __name__ == "__main__":
    main()
