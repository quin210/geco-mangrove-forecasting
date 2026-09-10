#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the breadth-scaling curve: within-site & pooled R^2 vs number of sites."""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe pair
C_WITHIN, C_POOLED = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def load(path="results/scaling_fixed.txt"):
    d = {}
    for r in open(path):
        if not r.startswith("N="):
            continue
        N = int(re.search(r"N=(\d+)", r).group(1))
        w = float(re.search(r"within_R2=([-\d.]+)", r).group(1))
        p = float(re.search(r" R2=([-\d.]+)", r).group(1))
        d.setdefault(N, []).append((w, p))
    Ns = sorted(d)
    W = np.array([[np.mean([x[0] for x in d[N]]), np.std([x[0] for x in d[N]])] for N in Ns])
    P = np.array([[np.mean([x[1] for x in d[N]]), np.std([x[1] for x in d[N]])] for N in Ns])
    return np.array(Ns), W, P


def main():
    Ns, W, P = load()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)

    for mu, c, lab in [(P, C_POOLED, "Pooled R²  (dominated by between-site level)"),
                       (W, C_WITHIN, "Within-site R²  (genuine temporal skill)")]:
        ax.fill_between(Ns, mu[:, 0] - mu[:, 1], mu[:, 0] + mu[:, 1], color=c, alpha=0.14, lw=0)
        ax.plot(Ns, mu[:, 0], color=c, lw=2.0, marker="o", ms=7,
                mec="white", mew=1.2, label=lab, zorder=3)

    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
    # annotate the operational dataset size
    ax.axvline(32, color=GRID, lw=1, zorder=0)
    ax.text(32, ax.get_ylim()[0], " full set (32)", color=MUTED, fontsize=8, va="bottom", ha="left")

    ax.set_xlabel("Number of training sites (N)", color=INK, fontsize=11)
    ax.set_ylabel("R²  (held-out, time-split)", color=INK, fontsize=11)
    ax.set_title("Breadth vs. skill: within-site skill stabilises with more sites",
                 color=INK, fontsize=12, pad=10)
    ax.set_xticks(Ns)
    ax.set_ylim(-0.15, 1.0)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    fig.tight_layout()
    out = "results/breadth_scaling.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[INFO] saved {out}")
    for N, w, p in zip(Ns, W, P):
        print(f"  N={N:2d}  within={w[0]:+.3f}±{w[1]:.3f}  pooled={p[0]:.3f}±{p[1]:.3f}")


if __name__ == "__main__":
    main()
