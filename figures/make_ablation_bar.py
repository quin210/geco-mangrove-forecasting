#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature-ablation bar: within-site R^2 (held-out future) when each group is removed."""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, BASE, DROP = "#1b2733", "#5b6b7a", "#DDDDDD", "#0072B2", "#D55E00"
LABELS = {"full": "Full model", "no_thermal": "− Thermal (temp/VPD/ET0/rad)",
          "no_water": "− Water (precip/soil/balance)", "no_season": "− Seasonality (sin/cos)",
          "no_geostatic": "− Geographic static (aridity/lat)"}
ORDER = ["full", "no_thermal", "no_water", "no_season", "no_geostatic"]


def main():
    d = {}
    for r in open("results/ablation.txt"):
        m = re.match(r"(\w+) seed=\d+ .*within_R2=([-\d.]+)", r)
        if m and m.group(1) in LABELS:
            d.setdefault(m.group(1), []).append(float(m.group(2)))
    keys = [k for k in ORDER if k in d]
    mean = np.array([np.mean(d[k]) for k in keys])
    sd = np.array([np.std(d[k]) for k in keys])
    labels = [LABELS[k] for k in keys]
    colors = [BASE if k == "full" else DROP for k in keys]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    y = np.arange(len(keys))[::-1]
    ax.barh(y, mean, xerr=sd, height=0.62, color=colors, edgecolor="white",
            error_kw=dict(ecolor=MUTED, lw=1), zorder=3)
    full_mean = d["full"] and np.mean(d["full"])
    ax.axvline(full_mean, color=BASE, ls=(0, (4, 3)), lw=1.2, zorder=2)
    for yi, m in zip(y, mean):
        ax.text(m + 0.006, yi, f"{m:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Within-site R²  (held-out future 2025–2026; ↓ = feature helped)", fontsize=10, color=INK)
    ax.set_title("Feature-group ablation: contribution to genuine temporal skill", fontsize=11.5, color=INK, pad=8)
    ax.grid(True, axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig_ablation.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_ablation.png")
    for k in keys:
        print(f"  {k:14s} within_R2={np.mean(d[k]):+.3f} ± {np.std(d[k]):.3f}")


if __name__ == "__main__":
    main()
