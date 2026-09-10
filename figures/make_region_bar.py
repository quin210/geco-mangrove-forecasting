#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-region held-out-future skill bar chart."""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FUT = "/tmp/claude-1026/-mnt-ssd5-quynh-mangrove/ddf5b72f-09d4-414a-b68e-3a03ba21f7b6/tasks/b6b31dddm.output"
GOOD, BAD, INK, MUTED, GRID = "#1A9850", "#D73027", "#1b2733", "#5b6b7a", "#DDDDDD"


def main():
    line = [l for l in open(FUT) if l.startswith("[PER-SITE")][0]
    r2 = {m.group(1): float(m.group(2)) for m in re.finditer(r"([A-Za-z_]+)=([-\d.]+)", line)}
    reg = pd.read_csv("data/mangrove_sites_global.csv").set_index("site_id")["region"].to_dict()
    d = {}
    for s, v in r2.items():
        d.setdefault(reg.get(s, "?"), []).append(v)
    rows = [(k, np.median([x for x in v if x > -2]), sum(x > 0 for x in v), len(v))
            for k, v in d.items()]
    rows.sort(key=lambda x: x[1])
    names = [f"{k}  ({p}/{n})" for k, m, p, n in rows]
    med = [m for k, m, p, n in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    colors = [GOOD if m > 0 else BAD for m in med]
    ax.barh(names, med, color=colors, height=0.66, zorder=3,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=2)
    for i, m in enumerate(med):
        ax.text(m + (0.02 if m >= 0 else -0.02), i, f"{m:+.2f}",
                va="center", ha="left" if m >= 0 else "right", fontsize=8.5, color=INK)

    ax.set_xlabel("Median per-site R²  (held-out future 2025–2026)", fontsize=10, color=INK)
    ax.set_title("Where the model forecasts the future — by region\n"
                 "(humid deltas skilful · arid Red Sea / Gulf fail; n = sites, x/y positive)",
                 fontsize=11, color=INK, pad=8)
    ax.set_xlim(-1.05, 0.9)
    ax.grid(True, axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig_region_bar.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_region_bar.png")


if __name__ == "__main__":
    main()
