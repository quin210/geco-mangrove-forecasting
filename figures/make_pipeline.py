#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication schematic of the GECO-EWS pipeline (GEE-free data -> model -> early warning)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#1b2733", "#5b6b7a"
COL = {"data": "#0072B2", "proc": "#009E73", "model": "#CC79A7",
       "phys": "#E69F00", "out": "#117733", "warn": "#B2182B"}


def box(ax, x, y, w, h, text, color, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=0, facecolor=color, alpha=0.95, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color=MUTED, zorder=1,
                                 connectionstyle="arc3,rad=0"))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")

    ax.text(6, 6.35, "GECO-EWS  —  a GEE-free, physics-informed early-warning pipeline for mangrove canopy dynamics",
            ha="center", fontsize=12.5, color=INK, weight="bold")

    # --- Column 1: data sources (no GEE) ---
    ax.text(1.4, 5.75, "1 · Open data (no Google Earth Engine)", ha="center", fontsize=9.5, color=MUTED, weight="bold")
    box(ax, 0.3, 4.75, 2.2, 0.7, "MODIS NDVI/EVI/LST\n(ORNL DAAC)", COL["data"], 8.5)
    box(ax, 0.3, 3.85, 2.2, 0.7, "ERA5 climate\n(Open-Meteo)", COL["data"], 8.5)
    box(ax, 0.3, 2.95, 2.2, 0.7, "GMW extent · optional\n(site polygons)", COL["data"], 8.5)

    # --- Column 2: processing ---
    ax.text(4.3, 5.75, "2 · Harmonise", ha="center", fontsize=9.5, color=MUTED, weight="bold")
    box(ax, 3.3, 4.3, 2.0, 0.7, "GMW-free pixel\nsnapping + QA mask", COL["proc"], 8.5)
    box(ax, 3.3, 3.4, 2.0, 0.7, "Monthly · time-split\n(train ≤2024)", COL["proc"], 8.5)
    box(ax, 3.3, 2.5, 2.0, 0.7, "Geographic conditioning\n(season · aridity · lat)", COL["proc"], 8.2)

    # --- Column 3: model ---
    ax.text(7.6, 5.75, "3 · GECO-EWS model", ha="center", fontsize=9.5, color=MUTED, weight="bold")
    box(ax, 6.3, 4.55, 2.6, 0.8, "Seasonal GAT encoder\n(climatic-similarity graph)", COL["model"], 8.5)
    box(ax, 6.3, 3.55, 2.6, 0.8, "TFT temporal module\n(per-feature VSN + attention)", COL["model"], 8.5)
    box(ax, 6.3, 2.55, 2.6, 0.8, "Physics-informed losses\nbounds · rate · water · non-cross", COL["phys"], 8.0, "white")

    # --- Column 4: output / early warning ---
    ax.text(10.9, 5.75, "4 · Early warning", ha="center", fontsize=9.5, color=MUTED, weight="bold")
    box(ax, 9.7, 4.3, 2.0, 0.9, "Probabilistic NDVI\nforecast (τ=.1/.5/.9)\nmulti-horizon", COL["out"], 8.2)
    box(ax, 9.7, 3.0, 2.0, 0.9, "Held-out future test\n2025–2026 vs satellite", COL["out"], 8.2)
    box(ax, 9.7, 1.7, 2.0, 0.8, "Dieback risk flag\n(calibrated uncertainty)", COL["warn"], 8.5)

    # arrows
    for y in (5.1, 4.2, 3.3):
        arrow(ax, 2.5, y, 3.3, 4.65 if y > 4.5 else (3.75 if y > 3.6 else 2.85))
    arrow(ax, 5.3, 3.75, 6.3, 4.0)
    arrow(ax, 5.3, 2.85, 6.3, 3.2)
    arrow(ax, 8.9, 4.0, 9.7, 4.5)
    arrow(ax, 8.9, 3.0, 9.7, 3.4)
    arrow(ax, 10.7, 3.0, 10.7, 2.5)

    # honesty footer
    ax.text(6, 0.5, "Reproducible · strictly temporal splits · evaluated with within-site R² (not inflated pooled R²)",
            ha="center", fontsize=8.5, color=MUTED, style="italic")

    fig.savefig("figures/fig_pipeline.png", bbox_inches="tight", facecolor="white")
    print("saved figures/fig_pipeline.png")


if __name__ == "__main__":
    main()
