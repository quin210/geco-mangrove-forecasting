#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Held-out future forecast panels: observed vs predicted NDVI (q10-q90 band), 2025-2026."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBS, PRED, BAND, INK, MUTED, GRID = "#111111", "#0072B2", "#0072B2", "#1b2733", "#5b6b7a", "#E4E4E4"

# representative sites: skilful (incl. dieback zone) + failures
SEL = [("Australia_Carpentaria", "Gulf of Carpentaria (AU) — 2015–16 dieback zone"),
       ("EastAfrica_Zambezi", "Zambezi delta (MZ)"),
       ("SAmerica_AmazonMarajo", "Amazon / Marajó (BR)"),
       ("SouthAsia_Bhitarkanika", "Bhitarkanika (IN)"),
       ("RedSea_AlShabaan", "Al Shabaan, Red Sea (SA) — arid"),
       ("Gulf_Qeshm", "Qeshm, Persian Gulf (IR) — arid")]


def main():
    df = pd.read_csv("data/processed/future_preds.csv")
    df = df[df["horizon"] == 1].copy()
    df["dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values(["site_id", "dt"])

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), dpi=200, sharex=False)
    for ax, (sid, title) in zip(axes.ravel(), SEL):
        g = df[df["site_id"] == sid]
        if len(g) == 0:
            ax.set_visible(False); continue
        ax.fill_between(g.dt, g.q10, g.q90, color=BAND, alpha=0.18, lw=0, label="80% interval")
        ax.plot(g.dt, g.q50, color=PRED, lw=2, label="forecast (median)", zorder=3)
        ax.plot(g.dt, g.y_true, color=OBS, lw=0, marker="o", ms=4.5, label="observed (MODIS)", zorder=4)
        from sklearn.metrics import r2_score
        r2 = r2_score(g.y_true, g.q50) if g.y_true.var() > 1e-8 else float("nan")
        ax.set_title(f"{title}\nR²={r2:+.2f}", fontsize=9.5, color=INK)
        ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylabel("NDVI", fontsize=9, color=MUTED)

    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Held-out future forecast (train ≤ 2024) vs. satellite ground truth, 2025–2026",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_forecast_panels.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_forecast_panels.png")


if __name__ == "__main__":
    main()
