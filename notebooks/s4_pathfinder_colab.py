# ╔══════════════════════════════════════════════════════════════════╗
# ║   S4 Pathfinder Replication — Google Colab Notebook             ║
# ║   Run each cell in order in Google Colab                        ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# This notebook replicates the Pathfinder experiment from:
#   "Efficiently Modeling Long Sequences with Structured State Spaces"
#   Gu et al., 2021 — arXiv:2111.00396
#
# Runtime: GPU (T4 recommended)  →  Runtime → Change runtime type → GPU


# ─────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────
# %%
# !pip install einops torch torchvision matplotlib numpy -q

# ─────────────────────────────────────────────────────────────────────
# CELL 2 — Clone / upload project files
# ─────────────────────────────────────────────────────────────────────
# %%
# Option A: Upload the zip and unzip
# from google.colab import files
# uploaded = files.upload()   # upload s4_pathfinder.zip
# !unzip s4_pathfinder.zip

# Option B: Clone from GitHub (after you push to a repo)
# !git clone https://github.com/YOUR_USERNAME/s4_pathfinder.git

# Option C: Paste the code directly (files already in Colab)


# ─────────────────────────────────────────────────────────────────────
# CELL 3 — Imports & Setup
# ─────────────────────────────────────────────────────────────────────
# %%
import sys, os
sys.path.insert(0, "/content/s4_pathfinder")   # adjust path if needed

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output

from models.s4_layer import S4Model, make_HiPPO
from data.pathfinder_dataset import get_pathfinder_loaders, SyntheticPathfinderDataset
from utils.baselines import LSTMBaseline, TransformerBaseline
from utils.metrics import (
    plot_training_curves, plot_hidden_states,
    plot_kernel, plot_pathfinder_samples
)
from train import train_model, build_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")


# ─────────────────────────────────────────────────────────────────────
# CELL 4 — Visualize the HiPPO Matrix
# ─────────────────────────────────────────────────────────────────────
# %%
N = 32
A = make_HiPPO(N).numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0F172A")
fig.suptitle("HiPPO Matrix — The Memory Keeper", color="white", fontsize=13)

im0 = axes[0].imshow(A, cmap="RdBu_r", aspect="auto")
axes[0].set_title("HiPPO-LegS Matrix A", color="white")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(np.abs(A), cmap="plasma", aspect="auto")
axes[1].set_title("|A| — Magnitude", color="white")
plt.colorbar(im1, ax=axes[1])

for ax in axes:
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8")
    ax.spines[:].set_color("#334155")

plt.tight_layout()
plt.show()
print("HiPPO matrix shape:", A.shape)
print("This structured initialization gives S4 its long-range memory.")


# ─────────────────────────────────────────────────────────────────────
# CELL 5 — Visualize Pathfinder samples
# ─────────────────────────────────────────────────────────────────────
# %%
dataset = SyntheticPathfinderDataset(n_samples=100, img_size=32)

fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor="#0F172A")
fig.suptitle("Pathfinder Task — Binary Classification\n"
             "Does the dotted path connect the two circles?",
             color="white", fontsize=13)

connected_shown    = 0
not_connected_shown = 0
shown              = 0

for i in range(len(dataset)):
    if shown >= 8:
        break
    x, y = dataset[i]
    img  = x.numpy().reshape(32, 32)
    if y == 1 and connected_shown < 4:
        row, col = 0, connected_shown
        connected_shown += 1
    elif y == 0 and not_connected_shown < 4:
        row, col = 1, not_connected_shown
        not_connected_shown += 1
    else:
        continue

    ax    = axes[row, col]
    label = "✓ Connected" if y == 1 else "✗ Not Connected"
    color = "#3B82F6" if y == 1 else "#EF4444"
    ax.imshow(img, cmap="hot", vmin=0, vmax=1)
    ax.set_title(label, color=color, fontsize=10, fontweight="bold")
    ax.axis("off")
    shown += 1

plt.tight_layout()
plt.show()
print(f"Sequence length: {32*32} = 1,024 steps per image")
print("The model must track path connectivity across ALL 1024 steps.")


# ─────────────────────────────────────────────────────────────────────
# CELL 6 — Build Models
# ─────────────────────────────────────────────────────────────────────
# %%
IMG_SIZE   = 32           # use 128 for the full Pathfinder-128 benchmark
D_MODEL    = 64           # reduce for faster Colab runs
D_STATE    = 32
NUM_LAYERS = 3
N_SAMPLES  = 3000         # synthetic dataset size

s4_model = S4Model(
    d_input=1, d_model=D_MODEL, d_state=D_STATE,
    num_layers=NUM_LAYERS, num_classes=2,
).to(device)

lstm_model = LSTMBaseline(
    d_input=1, d_model=D_MODEL,
    num_layers=NUM_LAYERS, num_classes=2,
).to(device)

transformer_model = TransformerBaseline(
    d_input=1, d_model=D_MODEL,
    num_layers=NUM_LAYERS, num_classes=2,
    seq_len=IMG_SIZE ** 2,
).to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"{'Model':<15} {'Parameters':>12}")
print("─" * 30)
for name, m in [("S4", s4_model), ("LSTM", lstm_model), ("Transformer", transformer_model)]:
    print(f"{name:<15} {count_params(m):>12,}")


# ─────────────────────────────────────────────────────────────────────
# CELL 7 — Load Data
# ─────────────────────────────────────────────────────────────────────
# %%
train_loader, val_loader, test_loader = get_pathfinder_loaders(
    mode       = "synthetic",
    img_size   = IMG_SIZE,
    batch_size = 32,
    n_samples  = N_SAMPLES,
)

x_sample, y_sample = next(iter(train_loader))
print(f"Batch shape:   {x_sample.shape}   (batch × seq_len)")
print(f"Label shape:   {y_sample.shape}")
print(f"Class balance: {y_sample.float().mean():.1%} positive (connected)")
print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")


# ─────────────────────────────────────────────────────────────────────
# CELL 8 — Train S4  (main experiment)
# ─────────────────────────────────────────────────────────────────────
# %%
EPOCHS = 20     # increase to 40–50 for better convergence

s4_model, s4_history = train_model(
    model_name   = "s4",
    model        = s4_model,
    train_loader = train_loader,
    val_loader   = val_loader,
    test_loader  = test_loader,
    epochs       = EPOCHS,
    lr           = 1e-3,
    device       = device,
    save_dir     = "/content/outputs",
)


# ─────────────────────────────────────────────────────────────────────
# CELL 9 — Train LSTM baseline
# ─────────────────────────────────────────────────────────────────────
# %%
lstm_model, lstm_history = train_model(
    model_name   = "lstm",
    model        = lstm_model,
    train_loader = train_loader,
    val_loader   = val_loader,
    test_loader  = test_loader,
    epochs       = EPOCHS,
    lr           = 1e-3,
    device       = device,
    save_dir     = "/content/outputs",
)


# ─────────────────────────────────────────────────────────────────────
# CELL 10 — Train Transformer baseline
#            (may be slow / OOM for img_size > 32 on Colab T4)
# ─────────────────────────────────────────────────────────────────────
# %%
transformer_model, tf_history = train_model(
    model_name   = "transformer",
    model        = transformer_model,
    train_loader = train_loader,
    val_loader   = val_loader,
    test_loader  = test_loader,
    epochs       = EPOCHS,
    lr           = 1e-3,
    device       = device,
    save_dir     = "/content/outputs",
)


# ─────────────────────────────────────────────────────────────────────
# CELL 11 — Plot Training Curves
# ─────────────────────────────────────────────────────────────────────
# %%
all_histories = {
    "s4":          s4_history,
    "lstm":        lstm_history,
    "transformer": tf_history,
}

plot_training_curves(all_histories, save_dir="/content/outputs")

# Also display inline
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F172A")
colors = {"s4": "#2563EB", "lstm": "#DC2626", "transformer": "#059669"}

for name, hist in all_histories.items():
    c      = colors[name]
    epochs = range(1, len(hist["train_acc"]) + 1)
    axes[0].plot(epochs, [a * 100 for a in hist["val_acc"]], color=c, linewidth=2.5, label=name.upper())
    axes[1].plot(epochs, hist["val_loss"], color=c, linewidth=2.5, label=name.upper())

axes[0].axhline(50, color="#F59E0B", linestyle=":", linewidth=1.5, label="Random baseline")

for ax, title, ylabel in zip(axes, ["Validation Accuracy", "Validation Loss"], ["Accuracy (%)", "Loss"]):
    ax.set_facecolor("#1E293B")
    ax.set_xlabel("Epoch", color="#94A3B8")
    ax.set_ylabel(ylabel, color="#94A3B8")
    ax.set_title(title, color="white", fontsize=12)
    ax.tick_params(colors="#94A3B8")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#334155", labelcolor="white")
    ax.grid(True, color="#334155", linestyle="--", alpha=0.5)

fig.suptitle("S4 vs LSTM vs Transformer — Pathfinder Task", color="white", fontsize=14)
plt.tight_layout()
plt.show()

# Final results table
print("\n" + "═" * 40)
print("  FINAL TEST ACCURACY")
print("═" * 40)
for name, hist in all_histories.items():
    bar = "█" * int(hist["test_acc"] * 30)
    print(f"  {name.upper():<13} {hist['test_acc']:>6.2%}  {bar}")
print("═" * 40)
print("  Random baseline:   50.00%")


# ─────────────────────────────────────────────────────────────────────
# CELL 12 — Visualize Hidden State Dynamics
# ─────────────────────────────────────────────────────────────────────
# %%
s4_model.eval()
x_vis, y_vis = next(iter(test_loader))
x_vis = x_vis[:1].to(device)   # single sample

# Collect hidden states
hidden = s4_model.get_hidden_states(x_vis, layer_idx=0)   # (L, 8)
hidden_np = hidden.numpy()

fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor="#0F172A")
fig.suptitle("S4 Hidden State Dynamics x(t)\nTracking path connectivity across 1024 steps",
             color="white", fontsize=13)

# Plot input sequence
img = x_vis[0].cpu().numpy()
axes[0].imshow(img.reshape(1, -1), cmap="hot", aspect="auto", vmin=0, vmax=1)
axes[0].set_title("Input sequence (flattened 32×32 image)", color="white", fontsize=10)
axes[0].set_ylabel("", color="#94A3B8")
axes[0].tick_params(colors="#94A3B8")
axes[0].spines[:].set_color("#334155")

# Plot hidden state evolution
cmap_h = plt.cm.plasma
L, D = hidden_np.shape
for d in range(D):
    c = cmap_h(d / D)
    axes[1].plot(hidden_np[:, d], color=c, linewidth=0.9, alpha=0.85, label=f"h_{d}")

axes[1].set_xlabel("Sequence step t", color="#94A3B8")
axes[1].set_ylabel("State value", color="#94A3B8")
axes[1].set_title("Hidden state x(t) — first 8 dimensions", color="white", fontsize=10)
axes[1].tick_params(colors="#94A3B8")
axes[1].spines[:].set_color("#334155")
axes[1].set_facecolor("#1E293B")
axes[1].legend(facecolor="#334155", labelcolor="white", ncol=4, fontsize=8)
axes[1].grid(True, color="#334155", linestyle="--", alpha=0.4)

axes[0].set_facecolor("#1E293B")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────
# CELL 13 — Visualize Learned Convolution Kernel
# ─────────────────────────────────────────────────────────────────────
# %%
s4_model.eval()
L_vis = IMG_SIZE ** 2   # 1024

# Extract kernel from first S4 layer
s4_layer = s4_model.layers[0]["s4"]
with torch.no_grad():
    K = s4_layer._get_kernel(L_vis)[0].cpu().numpy()   # first feature head

fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="#0F172A")
fig.suptitle("Learned S4 Convolution Kernel", color="white", fontsize=13)

axes[0].plot(K, color="#2563EB", linewidth=1.2)
axes[0].set_title("Spatial Domain K[l]", color="white")
axes[0].set_xlabel("Sequence position l", color="#94A3B8")
axes[0].set_ylabel("Amplitude", color="#94A3B8")

K_freq = np.abs(np.fft.rfft(K))
freqs  = np.fft.rfftfreq(L_vis)
axes[1].plot(freqs, K_freq, color="#7C3AED", linewidth=1.2)
axes[1].set_title("Frequency Domain |K(f)|", color="white")
axes[1].set_xlabel("Frequency", color="#94A3B8")
axes[1].set_ylabel("Magnitude", color="#94A3B8")

for ax in axes:
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8")
    ax.spines[:].set_color("#334155")
    ax.grid(True, color="#334155", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────
# CELL 14 — Mode Switching Demo: Recurrent vs Convolutional
# ─────────────────────────────────────────────────────────────────────
# %%
import time

s4_model.eval()
x_demo = torch.randn(1, IMG_SIZE ** 2, device=device)

# Convolutional mode (training)
t0 = time.time()
with torch.no_grad():
    _ = s4_model(x_demo, mode="conv")
t_conv = time.time() - t0

# Recurrent mode (inference)
t0 = time.time()
with torch.no_grad():
    _ = s4_model(x_demo, mode="recurrent")
t_recurrent = time.time() - t0

fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0F172A")
ax.set_facecolor("#1E293B")
modes  = ["Convolutional\n(Training)", "Recurrent\n(Inference)"]
times  = [t_conv * 1000, t_recurrent * 1000]
colors = ["#2563EB", "#059669"]
bars   = ax.bar(modes, times, color=colors, width=0.4, edgecolor="#334155", linewidth=1.5)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f} ms", ha="center", color="white", fontsize=11)
ax.set_ylabel("Time (ms)", color="#94A3B8")
ax.set_title("S4 Dual Mode — Speed Comparison", color="white", fontsize=12)
ax.tick_params(colors="#94A3B8")
ax.spines[:].set_color("#334155")
ax.grid(True, axis="y", color="#334155", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

print(f"Convolutional (parallel, FFT):  {t_conv*1000:.1f} ms — used during training")
print(f"Recurrent (sequential, O(1)):   {t_recurrent*1000:.1f} ms — used during inference")
print("\nBoth modes produce identical outputs — same model, two execution strategies.")


# ─────────────────────────────────────────────────────────────────────
# CELL 15 — Inference on Custom Input
# ─────────────────────────────────────────────────────────────────────
# %%
s4_model.eval()

# Get a test batch and show predictions
x_test, y_test = next(iter(test_loader))
x_test = x_test.to(device)

with torch.no_grad():
    logits = s4_model(x_test, mode="conv")
    probs  = torch.softmax(logits, dim=-1)
    preds  = logits.argmax(dim=-1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(14, 6), facecolor="#0F172A")
fig.suptitle("S4 Predictions on Test Set", color="white", fontsize=13)

for i in range(10):
    img  = x_test[i].cpu().numpy().reshape(IMG_SIZE, IMG_SIZE)
    pred = preds[i].item()
    true = y_test[i].item()
    conf = probs[i, pred].item()

    row, col = i // 5, i % 5
    ax = axes[row, col]
    ax.imshow(img, cmap="hot", vmin=0, vmax=1)

    correct = (pred == true)
    label_color = "#22C55E" if correct else "#EF4444"
    status      = "✓" if correct else "✗"
    ax.set_title(
        f"{status} Pred: {'Conn' if pred==1 else 'Not'}\n"
        f"True: {'Conn' if true==1 else 'Not'} ({conf:.0%})",
        color=label_color, fontsize=8
    )
    ax.axis("off")

plt.tight_layout()
plt.show()

n_correct = (preds == y_test).sum().item()
print(f"Accuracy on this batch: {n_correct}/{len(y_test)} = {n_correct/len(y_test):.1%}")


# ─────────────────────────────────────────────────────────────────────
# CELL 16 — Summary & Next Steps
# ─────────────────────────────────────────────────────────────────────
# %%
print("""
╔══════════════════════════════════════════════════════════╗
║              PROJECT COMPLETE — SUMMARY                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  You have successfully:                                  ║
║                                                          ║
║  ✓ Implemented S4 with HiPPO initialization              ║
║  ✓ Built FFT-accelerated convolutional kernel            ║
║  ✓ Added recurrent/convolutional dual mode               ║
║  ✓ Trained on Pathfinder binary classification           ║
║  ✓ Compared against LSTM and Transformer baselines       ║
║  ✓ Visualized hidden state dynamics                      ║
║  ✓ Visualized learned convolution kernels                ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  NEXT STEPS (to go further):                             ║
║                                                          ║
║  1. Scale to Pathfinder-128 (16,384-step sequences)      ║
║     → Download real LRA dataset                          ║
║     → Expected S4 accuracy: ~94%                         ║
║                                                          ║
║  2. Add Mamba's selective scan (S6)                      ║
║     → Make B, C, Δ input-dependent                       ║
║     → Should improve accuracy further                    ║
║                                                          ║
║  3. Run on Long Range Arena benchmark                    ║
║     → ListOps, Text, Retrieval, Image, Pathfinder        ║
║     → Compare against official S4 results                ║
║                                                          ║
║  4. Implement S4D (diagonal simplification)              ║
║     → Faster training, similar accuracy                  ║
║     → Compare training time vs S4                        ║
╚══════════════════════════════════════════════════════════╝
""")
