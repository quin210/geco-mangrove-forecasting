# GECO: Graph-Ecological Coastal Forecaster for Mangrove Canopy Dynamics

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Official implementation** of GECO: A hybrid spatio-temporal deep learning framework that combines **ecological graph neural networks** with **transformer-based temporal forecasting** for multi-horizon mangrove canopy dynamics prediction.

---

## Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training Objectives](#training-objectives)
- [Repository Structure](#repository-structure)

---

## Abstract

Mangrove ecosystems are critical blue carbon sinks and coastal defense systems, yet accurately forecasting their health dynamics remains challenging due to complex interactions between spatial connectivity and temporal environmental drivers. We present **GECO** (Graph-Ecological Coastal Forecaster), a novel spatio-temporal deep learning framework that explicitly models ecological connectivity through multi-seasonal graph neural networks combined with temporal feature transformers (TFT).

**Key innovations:**
- **Ecological graph construction** combining geodesic distance, environmental similarity, and seasonal anomaly patterns
- **Multi-seasonal GAT encoder** with node-conditioned attention for spatial embedding
- **TFT-inspired temporal module** with static and temporal variable selection networks
- **Physics-informed constraints** via eco-constrained loss functions
- **Probabilistic forecasting** with multi-quantile predictions

**Application domain:** Red Sea mangrove sites with 5+ years of multi-variate satellite and reanalysis data (NDVI, SST, salinity, precipitation, ocean currents, wind patterns, soil moisture).

---

## Key Features

- **Hybrid Spatio-Temporal Architecture**
  - Multi-seasonal Graph Attention Networks (GAT) for ecological connectivity
  - Temporal Feature Transformer (TFT) with variable selection
  
- **Probabilistic Forecasting**
  - Multi-quantile predictions (τ ∈ {0.1, 0.5, 0.9})
  - Uncertainty quantification for decision support

- **Physics-Informed Learning**
  - Laplacian smoothness regularization
  - Eco-constrained loss (e.g., salinity-NDVI relationships)

- **Production-Ready**
  - Clean, modular PyTorch implementation
  - Comprehensive data preprocessing pipeline
  - Checkpoint-based training with validation

---

## Architecture

GECO integrates two tightly coupled components:

### 1. **Ecological Graph Encoder**

```
Input: Static features (s_i) + Dynamic statistics (μ_i)
       ↓
Multi-Seasonal Graph Construction
  • Base ecological adjacency: A_base = f(d_geo, d_env)
  • Seasonal adjacencies: A_τ = f(seasonal anomalies), τ ∈ {DJF, MAM, JJA, SON}
       ↓
Multi-Season GAT Layers
  • Parallel GAT processing per season
  • Node-conditioned attention fusion
  • Laplacian smoothness regularization
       ↓
Output: Spatial embedding z_i ∈ ℝ^d_z per site
```

### 2. **Temporal Forecasting Module (TFT-style)**

```
Input: Time series x_{i,t-L:t} ∈ ℝ^{L×D}, Spatial embedding z_i
       ↓
Static Variable Selection Network (VSN)
  • Context vector c_stat from z_i
       ↓
Temporal Variable Selection Network
  • Feature-wise selection conditioned on c_stat
  • GRN-based attention over dynamic features
       ↓
GRU + Multi-Head Self-Attention
  • Sequence encoding with temporal dependencies
       ↓
Multi-Quantile Projection
  • Output: ŷ_{i,t+1:t+H}^{(τ)} ∈ ℝ^{H×Q}
```

**Training Objective:**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{QL}}(\mathbf{y}, \hat{\mathbf{y}}) + \lambda_{\text{lap}} \mathcal{L}_{\text{lap}}(\mathbf{Z}, \mathbf{A}) + \lambda_{\text{eco}} \mathcal{L}_{\text{eco}}(\hat{\mathbf{y}}, \mathbf{x})
$$

Where:
- **$\mathcal{L}_{\text{QL}}$**: Quantile loss for probabilistic forecasting
- **$\mathcal{L}_{\text{lap}}$**: Laplacian smoothness on graph embeddings
- **$\mathcal{L}_{\text{eco}}$**: Eco-constrained loss (salinity-NDVI relationship)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional)
- Git

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/quin210/geco-mangrove-forecasting.git
cd geco-mangrove-forecasting

# Create conda environment
conda env create -f environment.yml
conda activate geco

# Install PyTorch with CUDA support (if available)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: pip Installation

```bash
# Clone the repository
git clone https://github.com/quin210/geco-mangrove-forecasting.git
cd geco-mangrove-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset

### Data Structure

The preprocessed dataset (`data/processed/mangrove_all.csv`) contains multi-year observations from 5 Red Sea mangrove sites:

**Sites:** Al Shabaan, Al Shoaiba, Al Wajh, Duba Lake, Juzur Janabiat

**Features (per timestep):**
- **Target:** `ndvi` (Normalized Difference Vegetation Index)
- **Spatial:** `latitude`, `longitude`, `area`
- **Environmental drivers:**
  - `precipitation`, `salinity`, `sea_surface_height`
  - `soil_moisture`, `ocean_current_speed`
  - `ocean_cur_dir_sin`, `ocean_cur_dir_cos`
  - `wind_speed`, `wind_direction_sin`, `wind_direction_cos`

**Temporal coverage:** January 2014 – Present (~140+ monthly observations per site)

### Data Format

```csv
site_id,date,time_idx,latitude,longitude,area,ndvi,precipitation,salinity,...
Al_Shabaan,1/1/2014,1,24.80,37.18,490467,0.333,0,39.49,...
```

### Raw Data

Raw per-site CSV files are available in `data/raw/`:
- `Al_Shabaan.csv`
- `Al_Shoaiba.csv`
- `Al_Wajh.csv`
- `Duba_Lake.csv`
- `Juzur_Janabiat.csv`

---

## Usage

### Quick Start: Training

```bash
python train_geco.py \
  --csv_path data/processed/mangrove_all.csv \
  --input_length 12 \
  --forecast_horizon 3 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --checkpoint checkpoints/geco_full_tft.pt
```

### Training Options

#### Data & Model Configuration:
```bash
--csv_path              # Path to processed CSV
--input_length 12       # Lookback window (months)
--forecast_horizon 3    # Forecast horizon (months)
--batch_size 64         # Training batch size
--epochs 50             # Number of training epochs
--lr 1e-3               # Learning rate
--device cuda           # Device: cuda, cpu, or auto-detect
```

#### Graph Hyperparameters:
```bash
--num_seasons 4               # Number of seasonal graphs (4 = DJF/MAM/JJA/SON)
--lambda_geo 1.0              # Weight for geodesic distance
--lambda_env 1.0              # Weight for environmental similarity
--sigma_geo 1.0               # Gaussian kernel width (geodesic)
--sigma_env 1.0               # Gaussian kernel width (environment)
--sigma_dyn 1.0               # Gaussian kernel width (seasonal dynamics)
--k_neighbors_base 5          # k-NN for base ecological graph
--k_neighbors_seasonal 5      # k-NN for seasonal graphs
```

#### Model Architecture:
```bash
--gnn_hidden_dim 64           # GAT hidden dimension
--gnn_out_dim 64              # Spatial embedding dimension
--temporal_hidden_dim 128     # TFT hidden dimension
--num_heads 4                 # Multi-head attention heads
```

#### Regularization:
```bash
--lambda_lap 1e-3             # Laplacian smoothness weight
--lambda_eco 1e-3             # Eco-constraint weight
```

### Example: Extended Training

```bash
python train_geco.py \
  --csv_path data/processed/mangrove_all.csv \
  --input_length 18 \
  --forecast_horizon 6 \
  --batch_size 32 \
  --epochs 100 \
  --lr 5e-4 \
  --num_seasons 4 \
  --gnn_hidden_dim 128 \
  --gnn_out_dim 128 \
  --temporal_hidden_dim 256 \
  --num_heads 8 \
  --lambda_lap 5e-4 \
  --lambda_eco 1e-3 \
  --checkpoint checkpoints/geco_extended.pt
```

### Model Inference

```python
import torch
import pandas as pd
from geco.model_geco_full import GECOFull

# Load checkpoint
ckpt = torch.load("checkpoints/geco_full_tft.pt")
model = GECOFull(
    static_init=ckpt['static_init'],
    base_adj_norm=ckpt['base_adj_norm'],
    seasonal_adjs=ckpt['seasonal_adjs'],
    **ckpt['config']
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Prepare input data
# x_seq: [batch_size, input_length, num_features]
# site_idx: [batch_size] - integer site indices

# Inference
with torch.no_grad():
    predictions, spatial_embeddings = model(x_seq, site_idx)
    # predictions: [B, forecast_horizon, num_quantiles]
    # spatial_embeddings: [num_sites, gnn_out_dim]
    
    # Extract median forecast (quantile 0.5)
    median_forecast = predictions[:, :, 1]  # [B, H]
```

---

## Model Components

### 1. Dataset Module (`geco/dataset.py`)

**`MangroveWindowDataset`**: Sliding-window time series dataset
- Constructs input sequences: $x_{i,t-L:t}$
- Extracts target horizons: $y_{i,t+1:t+H}$
- Automatic feature detection and normalization
- Site-aware batching

### 2. Graph Builder (`geco/graph_builder.py`)

**Core Functions:**
- `build_static_and_dynamic_stats()`: Construct node initialization vectors
- `build_base_ecological_adjacency()`: Base graph with geo + env similarity
- `build_seasonal_adjacencies()`: Multi-seasonal graphs from anomaly patterns

**Graph Construction:**

Base ecological adjacency:
```python
A_base = λ_geo · K_geo(d_geo, σ_geo) + λ_env · K_env(d_env, σ_env)
```

Seasonal graphs:
```python
A_τ = K_dyn(seasonal_anomalies, σ_dyn)
```

Where $K(\cdot)$ is Gaussian RBF kernel with k-NN sparsification.

### 3. Model Architecture (`geco/model_geco_full.py`)

**Components:**
- `GATLayer`: Single-head Graph Attention layer with LeakyReLU activation
- `SeasonalGATEncoder`: Multi-seasonal GAT with fusion attention
  - Parallel GAT processing per season
  - Node-conditioned attention fusion via learnable query vector
  - Output: spatial embeddings $\mathbf{z}_i \in \mathbb{R}^{d_z}$
  
- `StaticVSN`: Static Variable Selection Network
  - GRN-based processing of spatial embeddings
  - Produces context vector for temporal module
  
- `TemporalVSN`: Temporal Variable Selection Network
  - Feature-wise selection at each timestep
  - Conditioned on static context
  
- `TemporalBackboneTFT`: GRU + Multi-head Attention + Quantile projection
  - Sequence encoding with GRU
  - Self-attention for long-range dependencies
  - Multi-quantile output layer
  
- `GECOFull`: Complete end-to-end model
  - Integrates spatial encoder + temporal module
  - Forward pass returns predictions + spatial embeddings

### 4. Loss Functions (`geco/losses.py`)

- **`quantile_loss()`**: Standard quantile regression loss
  ```python
  ρ_τ(u) = max(τ·u, (τ-1)·u)
  ```
  
- **`laplacian_smoothness_loss()`**: Graph regularization
  ```python
  L_lap = Σ_i || z_i - Σ_j A_ij z_j ||²
  ```
  
- **`eco_salinity_loss()`**: Physics-informed constraint
  - Penalizes NDVI increase under extreme salinity stress
  ```python
  φ_sal = max(0, α·(s - s_thr)·(ŷ - y_prev))
  ```

### 5. Training Utils (`geco/utils.py`)

- `train_epoch()`: Training loop with composite loss
  - Forward pass through model
  - Compute quantile loss + regularization terms
  - Backpropagation and optimization
  
- `eval_epoch()`: Validation with R², MAE, RMSE metrics
  - Uses median quantile (τ=0.5) for point predictions
  - Computes standard regression metrics

---

## Training Objectives

The total loss combines three components:

### 1. Quantile Loss ($\mathcal{L}_{\text{QL}}$)

Enables probabilistic forecasting with uncertainty quantification:

$$
\mathcal{L}_{\text{QL}} = \frac{1}{BHQ} \sum_{b=1}^B \sum_{h=1}^H \sum_{q=1}^Q \rho_{\tau_q}(y_{b,h} - \hat{y}_{b,h,q})
$$

Where: $\rho_{\tau}(u) = \max(\tau u, (\tau-1)u)$ is the pinball loss.

**Default quantiles:** τ ∈ {0.1, 0.5, 0.9} (10th, 50th, 90th percentiles)

### 2. Laplacian Smoothness Loss ($\mathcal{L}_{\text{lap}}$)

Enforces spatial consistency over the ecological graph:

$$
\mathcal{L}_{\text{lap}} = \sum_{i=1}^N \left\| \mathbf{z}_i - \sum_{j=1}^N A_{ij} \mathbf{z}_j \right\|^2
$$

**Purpose:** Encourages neighboring sites (in the ecological graph) to have similar embeddings.

### 3. Eco-Constrained Loss ($\mathcal{L}_{\text{eco}}$)

Penalizes biophysically implausible forecasts:

$$
\mathcal{L}_{\text{eco}} = \frac{1}{B} \sum_{b=1}^B \max\left(0, \alpha_{\text{sal}} (s_b - s_{\text{thr}}) (\hat{y}_b - y_{b,\text{prev}})\right)
$$

**Constraint:** Prevents NDVI increase when salinity exceeds threshold (default: 35 PSU).

**Total Objective:**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{QL}} + \lambda_{\text{lap}} \mathcal{L}_{\text{lap}} + \lambda_{\text{eco}} \mathcal{L}_{\text{eco}}
$$

**Default weights:** $\lambda_{\text{lap}} = 10^{-3}$, $\lambda_{\text{eco}} = 10^{-3}$


## Repository Structure

```
geco-mangrove-forecasting/
├── README.md                    # This file
├── LICENSE                      # Apache-2.0 License
├── requirements.txt             # Python dependencies (pip)
├── environment.yml              # Conda environment specification
├── train_geco.py               # Main training script
│
├── data/                        # Dataset directory
│   ├── raw/                     # Raw per-site CSV files
│   │   ├── Al_Shabaan.csv
│   │   ├── Al_Shoaiba.csv
│   │   ├── Al_Wajh.csv
│   │   ├── Duba_Lake.csv
│   │   └── Juzur_Janabiat.csv
│   │
│   └── processed/               # Preprocessed combined dataset
│       └── mangrove_all.csv
│
├── geco/                        # Core model package
│   ├── __init__.py
│   ├── dataset.py              # MangroveWindowDataset class
│   ├── graph_builder.py        # Graph construction utilities
│   ├── model_geco_full.py      # GECOFull model architecture
│   ├── losses.py               # Loss functions (QL, Laplacian, Eco)
│   └── utils.py                # Training/evaluation utilities
│
└── checkpoints/                 # Model checkpoints (created during training)
    └── .gitkeep
```