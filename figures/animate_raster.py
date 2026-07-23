#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nadir canopy animation: MODIS NDVI pixel grid of a mangrove site changing over time."""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

INK, MUTED = "#1b2733", "#5b6b7a"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "Sundarbans"
    z = np.load(f"figures/raster/{which}.npz", allow_pickle=True)
    G, months = z["grids"], z["months"]
    lat, lon, km = float(z["lat"]), float(z["lon"]), float(z["km"])

    # approx geographic extent (deg) for axis ticks
    dlat = km / 111.0
    dlon = km / (111.0 * np.cos(np.deg2rad(lat)))
    extent = [lon - dlon, lon + dlon, lat - dlat, lat + dlat]

    vmax = float(np.nanpercentile(G, 98))
    fig, ax = plt.subplots(figsize=(6.6, 6.4), dpi=150)
    im = ax.imshow(G[0], cmap="YlGn", vmin=0.0, vmax=max(0.6, vmax),
                   extent=extent, origin="upper", interpolation="nearest")
    ax.set_xlabel("Longitude", fontsize=9, color=MUTED)
    ax.set_ylabel("Latitude", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("NDVI (canopy greenness)", fontsize=10)
    title = ax.set_title("", fontsize=13, color=INK)
    fig.text(0.5, 0.02, f"{which} · nadir view ≈ {int(2*km)}×{int(2*km)} km · MODIS 250 m pixels (GEE-free)",
             ha="center", fontsize=8.5, color=MUTED, style="italic")

    def update(i):
        im.set_data(G[i])
        title.set_text(f"{which} mangrove canopy — {months[i]}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(months), interval=150, blit=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    anim.save(f"figures/raster/{which}_nadir.mp4", writer=FFMpegWriter(fps=7, bitrate=2600))
    anim2 = FuncAnimation(fig, update, frames=range(0, len(months), 2), interval=200, blit=False)
    anim2.save(f"figures/raster/{which}_nadir.gif", writer=PillowWriter(fps=6))
    print(f"saved figures/raster/{which}_nadir.mp4 + .gif  ({len(months)} frames)")


if __name__ == "__main__":
    main()
