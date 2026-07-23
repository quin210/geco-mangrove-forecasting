#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global mangrove NDVI time-lapse (2014-2026) as MP4 + GIF."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

INK, MUTED = "#1b2733", "#5b6b7a"


def main():
    df = pd.read_csv("data/processed/modis_full.csv")
    df["dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    coords = df.groupby("site_id")[["latitude", "longitude"]].first()
    # month x site NDVI matrix
    piv = (df.pivot_table(index="dt", columns="site_id", values="ndvi_modis")
             .sort_index())
    months = piv.index
    sites = piv.columns
    lat = coords.loc[sites, "latitude"].to_numpy()
    lon = coords.loc[sites, "longitude"].to_numpy()

    fig = plt.figure(figsize=(13, 4.4), dpi=140)
    ax = plt.axes([0.02, 0.08, 0.9, 0.82], projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#EAF2F8")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F4F1EC")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="#9AA7B2")
    ax.set_extent([-120, 160, -40, 40], crs=ccrs.PlateCarree())

    vals0 = piv.iloc[0].to_numpy()
    sc = ax.scatter(lon, lat, c=vals0, cmap="YlGn", vmin=0.0, vmax=0.8,
                    s=150, edgecolor="#333", lw=0.6, transform=ccrs.PlateCarree(), zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("NDVI (canopy greenness)", fontsize=10)
    title = ax.set_title("", fontsize=14, color=INK, pad=8)
    fig.text(0.5, 0.02, "GECO-EWS · 32 mangrove sites · MODIS MOD13Q1 (ORNL, GEE-free)",
             ha="center", fontsize=8.5, color=MUTED, style="italic")

    def update(i):
        v = piv.iloc[i].to_numpy()
        sc.set_array(v)
        title.set_text(f"Global mangrove canopy NDVI — {months[i]:%b %Y}")
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(months), interval=140, blit=False)

    mp4 = "figures/ndvi_timelapse.mp4"
    anim.save(mp4, writer=FFMpegWriter(fps=7, bitrate=2400))
    print("saved", mp4)
    # lighter GIF (every 2nd month)
    anim2 = FuncAnimation(fig, update, frames=range(0, len(months), 2), interval=180, blit=False)
    gif = "figures/ndvi_timelapse.gif"
    anim2.save(gif, writer=PillowWriter(fps=6))
    print("saved", gif)


if __name__ == "__main__":
    main()
