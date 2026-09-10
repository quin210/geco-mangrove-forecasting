#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Side-by-side nadir canopy comparison video for 3 mangrove sites over a common timeline."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

SITES = [("Sundarbans", "Sundarbans (IN/BD) — dense delta"),
         ("Carpentaria", "Gulf of Carpentaria (AU) — 2015–16 dieback"),
         ("MekongDelta", "Mekong delta (VN)")]
INK, MUTED = "#1b2733", "#5b6b7a"


def main():
    data = {}
    for k, _ in SITES:
        z = np.load(f"figures/raster/{k}.npz", allow_pickle=True)
        data[k] = (z["grids"], [str(m) for m in z["months"]])
    # common month set (intersection)
    common = sorted(set.intersection(*[set(v[1]) for v in data.values()]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), dpi=140)
    ims = []
    for ax, (k, title) in zip(axes, SITES):
        g, months = data[k]
        i0 = months.index(common[0])
        im = ax.imshow(g[i0], cmap="YlGn", vmin=0, vmax=0.8, interpolation="nearest")
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#ccc")
        ims.append((im, k))
    cb = fig.colorbar(ims[0][0], ax=axes, shrink=0.7, pad=0.01, location="right")
    cb.set_label("NDVI (canopy greenness)", fontsize=10)
    sup = fig.suptitle("", fontsize=15, color=INK, y=0.98)
    fig.text(0.5, 0.03, "Nadir MODIS 250 m NDVI · ≈ 8×8 km per site · GEE-free (ORNL)",
             ha="center", fontsize=9, color=MUTED, style="italic")

    def update(mi):
        ym = common[mi]
        for im, k in ims:
            g, months = data[k]
            im.set_data(g[months.index(ym)])
        sup.set_text(f"Mangrove canopy from space — {ym}")
        return [im for im, _ in ims] + [sup]

    anim = FuncAnimation(fig, update, frames=len(common), interval=150, blit=False)
    anim.save("figures/nadir_compare.mp4", writer=FFMpegWriter(fps=7, bitrate=3000))
    anim2 = FuncAnimation(fig, update, frames=range(0, len(common), 2), interval=200, blit=False)
    anim2.save("figures/nadir_compare.gif", writer=PillowWriter(fps=6))
    print(f"saved figures/nadir_compare.mp4 + .gif ({len(common)} frames)")


if __name__ == "__main__":
    main()
