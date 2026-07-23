# The physics-informed formulation for GECO-EWS

A precise articulation of what "physics-informed" means in this problem, the
constraints, their ecological justification, and how they enter the objective.
Written so it can be lifted into the paper's Methods/Discussion.

---

## 1. What "physics-informed" means *here* (state this precisely)

Classical **physics-informed neural networks (PINNs)** enforce a known governing
**PDE** in the loss. Mangrove canopy greenness (NDVI) has **no clean governing
PDE**, so GECO-EWS does **not** claim PDE residual minimisation. Instead it uses
**eco-physically-informed soft constraints**: differentiable penalties that
encode domain knowledge which a purely data-driven loss ignores. Four kinds:

1. **Measurement / physical-range** constraints on the target,
2. **Biophysical-inertia** constraints on its rate of change,
3. **Ecophysiological driver–response** constraints (stress monotonicity),
4. **Probabilistic-consistency** constraints on the predictive distribution.

> **Recommended framing for the paper.** Call it *eco-physically-informed* (or
> *knowledge-guided*) regularisation and say so explicitly; keep "physics-informed"
> in the title as the broad umbrella (standard usage in eco/remote-sensing ML) but
> define it precisely in §Methods to pre-empt the "this isn't a PINN" review. The
> value proposition is **regularisation under data scarcity** (few sites, short
> series) — exactly this study's regime.

---

## 2. Notation

For site *i*, horizon *h ∈ {1..H}*, quantile *τ ∈ {0.1, 0.5, 0.9}*:
`ŷ_{i,h}^{(τ)}` is the predicted NDVI, `y_{i}^{prev}` the last observed NDVI,
`ẑ_i` the median trajectory `[y_i^{prev}, ŷ_{i,1}^{(0.5)}, …, ŷ_{i,H}^{(0.5)}]`.
All quantities are in the model's normalised space; thresholds are converted from
raw units (see `train_geco_*`).

---

## 3. The constraints

### C1 — Physical bounds (measurement physics)
NDVI is a bounded ratio; a forecast outside its physical interval `[lo, hi]`
(here `[0,1]`) is unphysical.

$$\mathcal{L}_{\text{bound}}=\frac{1}{BHQ}\sum \big[\max(0,\,lo-\hat y)+\max(0,\,\hat y-hi)\big]$$

*Justification:* definitional. *Effect (observed):* drives out-of-bounds rate to
**0** on the multi-region set (0.043 → 0).

### C2 — Canopy-inertia rate limit (biophysical)
Canopy biomass/LAI change gradually; NDVI cannot jump arbitrarily month-to-month.
Penalise median-trajectory steps beyond a plausible bound `δ` (calibrated to the
empirical 99th percentile of |ΔNDVI/month| ≈ 0.27).

$$\mathcal{L}_{\text{rate}}=\frac{1}{BH}\sum \big[\max(0,\,|\hat z_{h}-\hat z_{h-1}|-\delta)\big]^2$$

*Justification:* phenological inertia of woody canopies. *Effect:* suppresses
implausible overshoot in multi-horizon forecasts.

### C3 — Multi-driver water-stress monotonicity (ecophysiology)
Under sustained water deficit, greening should **not increase**. A stress index
`s_i` combines standardised drought signals (low soil moisture + low
precipitation / negative water balance). When `s_i` exceeds a threshold, penalise
a forecast NDVI *rise* over the last observation:

$$\mathcal{L}_{\text{water}}=\frac{1}{B}\sum \max\!\big(0,\ \alpha\,(s_i-s_{\text{thr}})\,(\hat y_{i,1}^{(0.5)}-y_i^{prev})\big)$$

*Justification:* water availability limits mangrove productivity. **This replaces
the original salinity constraint** — see §4.

### C4 — Quantile non-crossing (probabilistic consistency)
A valid predictive CDF requires `q_{0.1} ≤ q_{0.5} ≤ q_{0.9}`. Crossing quantiles
are physically meaningless.

$$\mathcal{L}_{\text{cross}}=\frac{1}{BH}\sum \max\!\big(0,\ \hat y^{(\tau_k)}-\hat y^{(\tau_{k+1})}\big)$$

*Justification:* consistency of the forecast distribution. *Effect (observed):*
better **calibration** — 80% coverage moves toward nominal (0.76 → 0.80 single-region).

### (Spatial prior, already in the base model)
**Graph-Laplacian smoothness** on the site embeddings is itself a physics-style
**diffusion/connectivity prior**: ecologically connected sites should have similar
latent states. `L_lap = Σ_i ‖z_i − Σ_j A_{ij} z_j‖²`.

---

## 4. What we deliberately do NOT impose (honesty point)

The original code included a **salinity constraint** ("under high salinity, NDVI
must not rise"). We **remove** it because the data contradict it in this domain:
- salinity varies only 37–40 PSU (near-constant), so the hinge is always active;
- *Avicennia marina* (the dominant species) is highly salt-tolerant;
- empirically `corr(salinity anomaly, next-month ΔNDVI) = +0.07` — the assumed
  negative relationship is **not present**.

Imposing a constraint the data reject would fight the likelihood and bias
forecasts. **Principle:** only encode priors that are consistent with both
eco-physiology *and* the observed data. State this explicitly — reviewers reward
it.

---

## 5. Total objective

$$\mathcal{L}=\underbrace{\mathcal{L}_{\text{QL}}}_{\text{pinball (data)}}
 +\lambda_{\text{lap}}\mathcal{L}_{\text{lap}}
 +\lambda_{\text{bound}}\mathcal{L}_{\text{bound}}
 +\lambda_{\text{rate}}\mathcal{L}_{\text{rate}}
 +\lambda_{\text{water}}\mathcal{L}_{\text{water}}
 +\lambda_{\text{cross}}\mathcal{L}_{\text{cross}}$$

Implemented in `geco/physics_losses.py`; weights and threshold calibration in
`train_geco_epi.py` / `train_geco_multiregion.py`.

---

## 6. How to evaluate it (so the claim is defensible)

Report **both** predictive skill **and physical-consistency** metrics, since the
point of the constraints is plausibility, not only accuracy:
- accuracy: within-site R², per-site/per-region R² (not just pooled R²);
- consistency: out-of-bounds rate, quantile-crossing rate, 80% coverage;
- **ablation**: drop each constraint / driver group and report the change on the
  **held-out-future (2025–26)** split.

Empirically so far: physics gives a **small accuracy change (within noise)** but a
**clear calibration gain** and **zero unphysical predictions** — the honest story
is *plausibility & reliability*, not a leaderboard jump.

---

## 7. Extensions that would strengthen the physics story (roadmap)

1. **Thermal-stress term (now feasible with MODIS LST).** Penalise greening when
   LST exceeds a species/site thermal optimum, or above a marine-/land-heatwave
   percentile — directly ties to the dieback early-warning goal.
2. **Energy–water balance bound.** Cap achievable greening by available water
   (precip − ET₀) and radiation (a light-use-efficiency-style soft ceiling).
3. **Hemisphere-aware phenological prior.** Penalise deviation of the predicted
   annual trajectory from a smooth seasonal cycle (already have season features).
4. **Propagule-dispersal connectivity graph.** Replace/augment geodesic edges with
   ocean-current connectivity — a mechanistic spatial prior.
