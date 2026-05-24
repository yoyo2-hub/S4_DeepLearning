# S4 Pathfinder Replication

Replication of the **Pathfinder** experiment from:
> *Efficiently Modeling Long Sequences with Structured State Spaces*
> Gu et al., 2021 — [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

---

## What This Project Demonstrates

The **Pathfinder task** is the canonical benchmark for long-range sequence modeling:
- Input: 32×32 (or 128×128) image flattened to a 1D sequence
- Task: binary classification — does the dotted path connect the two endpoint circles?
- Challenge: requires tracking dependencies across **1,024–16,384 steps**

| Model       | Pathfinder-32 | Pathfinder-128 |
|-------------|:-------------:|:--------------:|
| LSTM        | ~57%          | ~57% (random)  |
| Transformer | ~60%          | fails (OOM)    |
| **S4**      | **~88%**      | **~94%**       |

---

## Project Structure

```
s4_pathfinder/
├── models/
│   └── s4_layer.py          # S4 implementation (HiPPO, FFT kernel, dual mode)
├── data/
│   └── pathfinder_dataset.py # Synthetic + real LRA Pathfinder loader
├── utils/
│   ├── baselines.py          # LSTM and Transformer baselines
│   └── metrics.py            # Training curves, kernel viz, hidden state viz
├── notebooks/
│   └── s4_pathfinder_colab.py  # Full Colab notebook (16 cells)
├── train.py                  # Training script with CLI
└── README.md
```

---

## Quick Start — Google Colab

### Step 1: Open a new Colab notebook

```
Runtime → Change runtime type → GPU (T4)
```

### Step 2: Upload the project

```python
from google.colab import files
files.upload()   # upload s4_pathfinder.zip
!unzip s4_pathfinder.zip
```

### Step 3: Install dependencies

```bash
!pip install einops torch torchvision matplotlib numpy -q
```

### Step 4: Run the notebook

Open `notebooks/s4_pathfinder_colab.py` and run each cell in order,
**or** use the CLI:

```bash
# Train S4 only
!python train.py --model s4 --epochs 20

# Compare all models
!python train.py --model compare_all --epochs 20

# Larger image size (harder task)
!python train.py --model s4 --img_size 64 --epochs 40
```

---

## Key Implementation Details

### HiPPO Initialization
```python
# models/s4_layer.py → make_HiPPO(N)
# Creates structured A matrix for stable long-range memory
A[n,k] = -sqrt((2n+1)(2k+1))  for k < n
          (n+1)                 for k = n
```

### Dual Mode (Convolutional ↔ Recurrent)
```python
model(x, mode="conv")       # training  — O(N log N) via FFT
model(x, mode="recurrent")  # inference — O(1) per step
```

### Separate Learning Rates for SSM Parameters
```python
# train.py → build_optimizer()
# A, B, C, Δ use a lower LR than the rest of the network
optimizer = build_optimizer(model, base_lr=1e-3, ssm_lr=1e-3)
```

---

## Visualizations Produced

| Visualization | Description |
|---|---|
| `pathfinder_samples.png` | Sample images from the dataset |
| `training_curves.png` | S4 vs LSTM vs Transformer accuracy/loss |
| `hidden_states.png` | Hidden state x(t) evolution over 1024 steps |
| `kernel.png` | Learned convolution kernel (spatial + frequency domain) |

---

## Extending the Project

### 1. Use the Real LRA Dataset
```bash
# Download
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar -xzf lra_release.gz

# Train
python train.py --model s4 --data_mode lra --data_dir ./lra_release/lra_release/pathfinder32
```

### 2. Pathfinder-128 (16,384 steps)
```bash
python train.py --model s4 --img_size 128 --d_state 64 --epochs 50
# Expected: ~94% accuracy — the result that made S4 famous
```

### 3. Add Mamba's Selective Scan
Modify `S4Layer` so B, C, Δ are functions of the input:
```python
# Instead of fixed parameters:
self.B = nn.Parameter(...)
# Make them input-dependent:
self.B_proj = nn.Linear(d_model, d_state)
# Then in forward: B = self.B_proj(u)
```

---

## References

- [S4 Paper](https://arxiv.org/abs/2111.00396) — Gu et al., 2021
- [HiPPO Paper](https://arxiv.org/abs/2008.07669) — Gu et al., 2020
- [Long Range Arena](https://arxiv.org/abs/2011.04006) — Tay et al., 2020
- [Mamba](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023
- [Official S4 Code](https://github.com/state-spaces/s4)
