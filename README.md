# GECO-EWS: A Physics-Informed Graph Early-Warning Framework for Global Mangrove Canopy Dieback under Climate Stress

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **GECO-EWS** is a hybrid **spatio-temporal** deep-learning framework toward a
> **global early-warning system** for mangrove canopy decline and dieback under
> climate stress. It couples an **ecological / climatic graph neural network**
> with a **TFT-style temporal forecaster**, under **physics-informed** constraints
> and explicit **geographic conditioning** across biogeographic regions and both
> hemispheres, to produce multi-horizon **probabilistic** NDVI forecasts — all on
> a **fully reproducible, Google-Earth-Engine-free** data pipeline.

*Naming.* **GECO-EWS** is the framework (Graph-Ecological Coastal Forecaster →
Early-Warning System); `GECO` / `GECOFullV2` remain the core model classes.

---

## Table of Contents

- [Abstract](#abstract)
- [What is in this repository](#what-is-in-this-repository)
- [Architecture](#architecture)
- [Data pipeline (GEE-free)](#data-pipeline-gee-free)
- [Geographic conditioning for multi-region modelling](#geographic-conditioning-for-multi-region-modelling)
- [Physics-informed constraints](#physics-informed-constraints)
- [Evaluation methodology](#evaluation-methodology)
- [Results so far](#results-so-far)
- [Installation](#installation)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Roadmap](#roadmap)

---

## Abstract

Mangroves are critical blue-carbon sinks and coastal defences, and they are
increasingly exposed to climate stress — marine heatwaves, drought, and abrupt
sea-level fluctuations have already driven **mass dieback** events (e.g. the Gulf
of Carpentaria, 2015–16). Anticipating canopy decline is hard because it couples
**spatial ecological connectivity** with **temporal environmental drivers** that
differ sharply across climate regimes and hemispheres. **GECO-EWS** targets this
as an **early-warning** problem: a multi-seasonal Graph Attention encoder produces
a spatial embedding per site, which conditions a Temporal Fusion Transformer–style
module emitting **multi-horizon, multi-quantile** NDVI forecasts, so that
elevated risk can be flagged with calibrated uncertainty. Learning is regularised
by **eco-physically-informed** constraints and made comparable across
biogeographic provinces through explicit **geographic conditioning**.

This repository accompanies ongoing work and emphasises **methodological
honesty**: every satellite/reanalysis input is obtainable without Google Earth
Engine, splits are strictly temporal, and skill is reported with a **within-site
R²** that separates genuine temporal forecasting from trivial between-site level
differences. We present GECO-EWS as a **framework and reproducible benchmark
toward** operational early warning, not a finished operational system — current
within-site skill is modest, and closing that gap is the explicit research goal.

---

## What is in this repository

| Component | File | Purpose |
|---|---|---|
| Fixed temporal architecture | `geco/model_geco_v2.py` | `GECOFullV2` with a per-feature TFT variable-selection network |
| Physics-informed losses | `geco/physics_losses.py` | quantile non-crossing, canopy-inertia rate limit, physical bounds, water stress + consistency metrics |
| Geographic conditioning | `geco/geo_features.py` | hemisphere-aware seasonality, static geo descriptors, climatic-similarity graph |
| Baseline architecture | `geco/model_geco_full.py` | original seasonal-GAT + TFT (kept for ablation) |
| Data / windows / losses | `geco/dataset.py`, `geco/graph_builder.py`, `geco/losses.py` | sliding windows, graph construction, quantile/Laplacian/eco losses |
| Single-region training | `train_geco_v2.py`, `train_geco_epi.py` | corrected pipeline + physics ablation |
| Multi-region training | `train_geco_multiregion.py` | geo-conditioned model + pooled / within-site / per-site R² |
| GEE-free data fetchers | `data/fetch_modis_ndvi.py`, `data/fetch_openmeteo.py` | MODIS (ORNL) + ERA5 (Open-Meteo) |
| Data provenance | `data/DATA_SOURCES.md`, `data/merge_sources.py` | sources, licences, merge |
| Optional GEE path | `gee/` | Earth Engine extraction if preferred |

---

## Architecture

### 1. Ecological graph encoder
Static site descriptors and dynamic driver statistics initialise node features
`h_i^{(0)}`. A base **ecological / climatic** adjacency plus per-season
adjacencies feed a **multi-season GAT** with node-conditioned attention fusion,
yielding a spatial embedding `z_i ∈ ℝ^{d_z}`.

### 2. Temporal module (TFT-style, corrected)
The original variable-selection network collapsed all `D` drivers to a **single
scalar per timestep** before the GRU, discarding most multivariate information.
`geco/model_geco_v2.py` replaces it with a **per-feature variable selection**
that projects each driver into an embedding space and combines the selection
weights *in that space* (output `[B, L, hidden]`), followed by GRU +
multi-head self-attention and a multi-quantile head.

```
x_{i,t-L:t}, z_i
   │
   ├── Static VSN(z_i) ─────────────► context c_stat
   │
   └── Per-feature Temporal VSN(x, c_stat) ─► ξ_t ∈ ℝ^{hidden}
                                          │
                            GRU → Multi-Head Attention → GRN
                                          │
                       multi-horizon, multi-quantile  ŷ^{(τ)}_{i,t+1:t+H}
```

**Training objective**

$$
\mathcal{L} = \mathcal{L}_{\text{QL}} + \lambda_{\text{lap}}\,\mathcal{L}_{\text{lap}}
            + \sum_c \lambda_c\, \mathcal{L}^{\text{phys}}_c
$$

quantile (pinball) loss + graph-Laplacian smoothness + physics-informed terms.

---

## Data pipeline (GEE-free)

Every driver is retrievable over plain HTTP; **most need no account**. See
`data/DATA_SOURCES.md` for the full table.

| Variable(s) | Product | Source | Auth |
|---|---|---|---|
| NDVI, EVI (250 m, 16-day, 2000–present) | MOD13Q1 | ORNL DAAC MODIS Web Service | none |
| LST day/night (1 km, 8-day) | MOD11A2 | ORNL DAAC MODIS Web Service | none |
| Air temp, VPD, ET₀, soil moisture, radiation, precipitation | ERA5 | Open-Meteo Archive API | none |

Point APIs sample coordinates rather than rasterising polygons, so a **GMW-free
"snap"** (`--km 1 --snap`) samples a ~2 km window and keeps the pixels whose
long-term, QA-masked mean NDVI is mangrove-plausible (∈ [0.15, 0.9]) — dropping
water/cloud pixels around an imperfectly placed coordinate. Fetchers are
**resumable** and survive ORNL's intermittent outages.

A curated multi-region site list is provided in
`data/mangrove_sites_global.csv` (40 sites across 9 biogeographic regions,
including the 2015–16 Gulf of Carpentaria dieback zone).

To enrich further with variables the point APIs can't do well — **GMW-polygon
NDVI/EVI with strict cloud masking, coastal SST, inundation (JRC water), canopy
structure (GEDI), mangrove-loss area** — an optional Google Earth Engine path is
provided in `gee/`. See **`gee/README.md` §3** for the exact variable catalogue
(asset IDs, bands, scaling, aggregation) and drop exports into `gee/outputs/`;
GEE and non-GEE variables merge on `(site_id, date)`.

---

## Geographic conditioning for multi-region modelling

Pooling sites from different climate zones and **both hemispheres** makes a raw
month or a raw driver value ambiguous. `geco/geo_features.py` supplies the
context the model needs:

- **Hemisphere-aware seasonality** — Southern-hemisphere phase is shifted by six
  months so a phase peak means the same physiological season everywhere.
- **Static geographic descriptors** — signed latitude, `|latitude|`
  (seasonality amplitude), longitude sin/cos, and an **aridity index**
  (precip / ET₀) that cleanly orders regimes (Persian Gulf ≈ 0.02 → SE Asia ≈ 1.9).
- **Climatic-similarity block graph** — within-region geodesic neighbours plus
  cross-region **climatic-niche** neighbours, because geodesic distance is
  meaningless across oceans.

---

## Physics-informed constraints

`geco/physics_losses.py` implements differentiable soft penalties, applied only
where they are consistent with mangrove eco-physiology **and** the data:

- **Quantile non-crossing** — enforce q₀.₁ ≤ q₀.₅ ≤ q₀.₉.
- **Canopy-inertia rate limit** — bound month-to-month NDVI change (calibrated to
  the empirical 99th percentile).
- **Physical bounds** — keep NDVI in a physical interval.
- **Multi-driver water stress** — under sustained low water availability, penalise
  forecast NDVI increases.

> The original salinity constraint is intentionally **not** used: in the Red Sea
> data salinity varies little (37–40 PSU), *Avicennia marina* is salt-tolerant,
> and the salinity–NDVI relationship is not data-supported there.

---

## Evaluation methodology

Splits are **strictly temporal** (past → train, future → validation) per site;
no random window shuffling. Skill is reported at three levels:

- **Pooled R²** — over all sites/horizons. Across regions this is dominated by
  *between-site level differences* and can look deceptively high.
- **Within-site R²** — each site's train-mean is removed first, isolating genuine
  temporal forecasting skill (the metric that matters).
- **Per-site R²** — a per-site breakdown.

Probabilistic quality is tracked with 80% **coverage**, quantile-crossing rate,
and out-of-bounds rate.

---

## Results so far

Representative validation numbers from the corrected pipeline (single seed unless
noted; the point is the *methodology*, not a leaderboard):

**Single region (5 Red Sea sites)**

| Model | val R² | note |
|---|---|---|
| Seasonal-naive baseline | 0.19 | value 12 months prior |
| GECO (original scalar-VSN architecture) | ≈ 0.0 | loses to seasonal-naive |
| **GECO v2** (per-feature VSN + seasonality) | **≈ 0.64** | clears every naive baseline |
| GECO v2 + physics | ≈ 0.64 | ~equal accuracy, **better calibration** (coverage 0.76 → 0.80) |

**Multi-region (demo subset, geo-conditioned)**

| Metric | value | reading |
|---|---|---|
| Pooled R² | ≈ 0.88 | mostly between-site separation |
| Predict-site-mean baseline | ≈ 0.86 | shows how much pooled R² is "free" |
| **Within-site R²** | ≈ 0.1 | **genuine temporal skill is still low** |

**Honest takeaway.** Fixing the architecture and adding geographic conditioning
lets a single model span NDVI regimes from ≈ 0.15 (arid) to ≈ 0.70 (humid)
without collapsing — geography works. But **within-site temporal skill is still
near zero**, and for a credible *early-warning* system it is precisely this
within-site, ahead-of-time skill (especially around anomalous decline) that must
improve. We therefore report it openly: it defines the open research problem
GECO-EWS targets, rather than hiding it behind an inflated pooled R².

---

## Installation

```bash
git clone https://github.com/quin210/geco-mangrove-forecasting.git
cd geco-mangrove-forecasting
conda env create -f environment.yml && conda activate geco   # or: pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # GPU optional
```

## Usage

```bash
# 1) Fetch data (no Google Earth Engine, no account needed)
python data/fetch_modis_ndvi.py --sites_csv data/mangrove_sites_global.csv \
    --km 1 --snap --wait_ornl --out data/processed/modis_global.csv
python data/fetch_openmeteo.py --sites_csv data/mangrove_sites_global.csv \
    --start 2014-01 --end 2024-12 --out data/processed/climate_global.csv

# 2a) Single-region training + physics ablation
python train_geco_epi.py --arch v2 --add_season_feats --physics --epochs 120

# 2b) Multi-region training with geographic conditioning
python train_geco_multiregion.py --modis data/processed/modis_global.csv \
    --climate data/processed/climate_global.csv --physics --epochs 80
```

---

## Repository structure

```
geco-mangrove-forecasting/
├── geco/
│   ├── dataset.py               # sliding-window dataset (+ standardization)
│   ├── graph_builder.py         # ecological / seasonal graph construction
│   ├── model_geco_full.py       # original GECO (baseline / ablation)
│   ├── model_geco_v2.py         # GECOFullV2: per-feature TFT VSN
│   ├── geo_features.py          # geographic conditioning + climatic graph
│   ├── losses.py                # quantile / Laplacian / eco losses
│   └── physics_losses.py        # physics-informed constraints + metrics
├── train_geco_v2.py             # corrected single-region pipeline
├── train_geco_epi.py            # + physics ablation
├── train_geco_multiregion.py    # multi-region, geo-conditioned
├── data/
│   ├── fetch_modis_ndvi.py      # MODIS via ORNL (GEE-free)
│   ├── fetch_openmeteo.py       # ERA5 via Open-Meteo (GEE-free)
│   ├── merge_sources.py         # combine sources
│   ├── DATA_SOURCES.md          # provenance & licences
│   ├── mangrove_sites_global.csv# 40 multi-region sites
│   └── processed/mangrove_all.csv
└── gee/                         # optional Earth Engine extraction path
```

---

## Roadmap

Toward an operational early-warning framework:

- [ ] Complete the 40-site global multi-region dataset (GMW-free snapping) and
      report **within-site R²** across regions.
- [ ] Lift within-site / anomaly skill — the core requirement for early warning.
- [ ] Thermal-stress physics term using MODIS LST (marine-heatwave signal).
- [ ] Ocean-current propagule-dispersal graph edges.
- [ ] Transfer learning from data-rich to data-poor regions.
- [ ] **Dieback early-warning case study** — hindcast the Gulf of Carpentaria
      2015–16 event; evaluate lead time and false-alarm rate, not just R².

---

*This repository documents research in progress; results and interfaces may change.*
