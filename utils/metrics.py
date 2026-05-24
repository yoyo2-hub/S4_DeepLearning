"""
Metrics and Visualization Utilities
====================================
Plotting training curves, confusion matrices, kernel visualization,
and hidden state dynamics.
"""

import json
import numpy as np
from pathlib import Path


def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, F1 for binary classification."""
    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec   = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1     = 2 * prec * recall / (prec + recall + 1e-8)
    return {"acc": acc, "precision": prec, "recall": recall, "f1": f1}


def plot_training_curves(histories: dict, save_dir: str = "outputs"):
    """
    Plot training/validation accuracy curves for all models.
    histories: {model_name: {train_acc, val_acc, ...}}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    colors = {"s4": "#2563EB", "lstm": "#DC2626", "transformer": "#059669"}
    styles = {"s4": "-", "lstm": "--", "transformer": "-."}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F172A")
    fig.suptitle("Pathfinder Task — Model Comparison", color="white", fontsize=14, y=1.02)

    for name, hist in histories.items():
        c = colors.get(name, "white")
        s = styles.get(name, "-")
        epochs = range(1, len(hist["train_acc"]) + 1)

        axes[0].plot(epochs, [a * 100 for a in hist["train_acc"]],
                     color=c, linestyle=s, linewidth=2, alpha=0.7, label=f"{name.upper()} train")
        axes[0].plot(epochs, [a * 100 for a in hist["val_acc"]],
                     color=c, linestyle=s, linewidth=2, label=f"{name.upper()} val")

        axes[1].plot(epochs, hist["train_loss"],
                     color=c, linestyle=s, linewidth=2, alpha=0.7)
        axes[1].plot(epochs, hist["val_loss"],
                     color=c, linestyle=s, linewidth=2)

    for ax, title, ylabel in zip(axes, ["Accuracy (%)", "Loss"], ["Accuracy (%)", "Loss"]):
        ax.set_facecolor("#1E293B")
        ax.set_xlabel("Epoch", color="#94A3B8")
        ax.set_ylabel(ylabel, color="#94A3B8")
        ax.set_title(title, color="white")
        ax.tick_params(colors="#94A3B8")
        ax.spines[:].set_color("#334155")
        ax.legend(facecolor="#334155", labelcolor="white", framealpha=0.8)
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)

    # Add random baseline reference
    axes[0].axhline(50, color="#F59E0B", linestyle=":", linewidth=1.5, alpha=0.7, label="Random")
    axes[0].legend(facecolor="#334155", labelcolor="white", framealpha=0.8)

    plt.tight_layout()
    out_path = save_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_hidden_states(hidden_states, save_dir: str = "outputs"):
    """
    Visualize the evolution of S4 hidden state dimensions over time.
    hidden_states: (L, D) numpy array
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    L, D = hidden_states.shape
    fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")

    cmap   = plt.cm.plasma
    colors = [cmap(i / D) for i in range(D)]

    for d in range(min(D, 8)):
        ax.plot(hidden_states[:, d], color=colors[d], linewidth=0.8,
                alpha=0.85, label=f"h_{d}")

    ax.set_xlabel("Sequence step (t)", color="#94A3B8")
    ax.set_ylabel("Hidden state value", color="#94A3B8")
    ax.set_title("S4 Hidden State Dynamics x(t)", color="white", fontsize=13)
    ax.tick_params(colors="#94A3B8")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#334155", labelcolor="white", ncol=4, fontsize=8)
    ax.grid(True, color="#334155", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = save_dir / "hidden_states.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"  Saved: {out}")


def plot_kernel(kernel, save_dir: str = "outputs"):
    """
    Visualize the learned S4 convolution kernel in spatial and frequency domains.
    kernel: (L,) numpy array
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    save_dir = Path(save_dir)
    L = len(kernel)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="#0F172A")
    fig.suptitle("Learned S4 Convolution Kernel", color="white", fontsize=13)

    # Spatial domain
    axes[0].plot(kernel, color="#2563EB", linewidth=1.2)
    axes[0].set_title("Spatial Domain K[l]", color="white")
    axes[0].set_xlabel("Sequence position l", color="#94A3B8")
    axes[0].set_ylabel("Amplitude", color="#94A3B8")

    # Frequency domain
    K_freq = np.abs(np.fft.rfft(kernel))
    freqs  = np.fft.rfftfreq(L)
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
    out = save_dir / "kernel.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"  Saved: {out}")


def plot_pathfinder_samples(dataset, n: int = 6, save_dir: str = "outputs"):
    """Visualize sample Pathfinder images from dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    save_dir = Path(save_dir)
    img_size = int(len(dataset[0][0]) ** 0.5)

    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 5), facecolor="#0F172A")
    fig.suptitle("Pathfinder Samples  (blue=connected, red=not)", color="white", fontsize=12)

    shown = {"0": 0, "1": 0}
    idx   = 0
    plotted = 0

    while plotted < n and idx < len(dataset):
        x, y = dataset[idx]
        img  = x.numpy().reshape(img_size, img_size)
        col  = plotted % (n // 2)
        row  = plotted // (n // 2)
        ax   = axes[row, col]

        ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
        label = "Connected" if y == 1 else "Not Connected"
        color = "#3B82F6" if y == 1 else "#EF4444"
        ax.set_title(label, color=color, fontsize=9)
        ax.axis("off")
        plotted += 1
        idx += 1

    plt.tight_layout()
    out = save_dir / "pathfinder_samples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"  Saved: {out}")
