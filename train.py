"""
Training Script — S4 Pathfinder
================================
Trains S4 on the Pathfinder task and compares against LSTM and Transformer baselines.

Usage:
    python train.py --model s4 --img_size 32 --epochs 20
    python train.py --model lstm
    python train.py --model transformer
    python train.py --compare_all

Run on Google Colab:
    !python train.py --model s4 --epochs 30
"""

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.s4_layer import S4Model
from data.pathfinder_dataset import get_pathfinder_loaders
from utils.baselines import LSTMBaseline, TransformerBaseline
from utils.metrics import compute_metrics, plot_training_curves


# ─────────────────────────────────────────────
# Optimizer with separate LR for SSM params
# ─────────────────────────────────────────────

def build_optimizer(model, base_lr: float = 1e-3, ssm_lr: float = 1e-3):
    """
    S4 needs a separate (often lower) learning rate for SSM parameters
    (A, B, C, dt) vs. the rest of the network.
    """
    ssm_params   = []
    other_params = []

    for name, param in model.named_parameters():
        if hasattr(param, "_ssm_lr"):
            ssm_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": base_lr},
        {"params": ssm_params,   "lr": ssm_lr, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=base_lr, weight_decay=1e-2)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, mode="conv"):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if hasattr(model, "forward") and isinstance(model, S4Model):
            logits = model(x, mode=mode)
        else:
            logits = model(x)

        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping — important for SSMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        preds         = logits.argmax(dim=-1)
        total_loss   += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, mode="conv"):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if isinstance(model, S4Model):
            logits = model(x, mode=mode)
        else:
            logits = model(x)

        loss          = criterion(logits, y)
        preds         = logits.argmax(dim=-1)
        total_loss   += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train_model(
    model_name:  str,
    model:       nn.Module,
    train_loader,
    val_loader,
    test_loader,
    epochs:      int   = 20,
    lr:          float = 1e-3,
    device:      torch.device = None,
    save_dir:    str   = "outputs",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if isinstance(model, S4Model):
        optimizer = build_optimizer(model, base_lr=lr, ssm_lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "epoch_time": [],
    }

    best_val_acc  = 0.0
    save_path     = Path(save_dir) / f"{model_name}_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Training: {model_name.upper()}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")
    print(f"{'─'*60}")
    print(f"  {'Epoch':>5}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  {'V-Acc':>7}  {'Time':>6}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        marker = " ←" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"  {epoch:>5}  {train_loss:>8.4f}  {train_acc:>6.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.1f}s{marker}")

    # ── Final test evaluation ──────────────────────────
    print(f"\n  Loading best model (val_acc={best_val_acc:.2%}) …")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"  Test accuracy: {test_acc:.2%}")
    print(f"{'─'*60}\n")

    history["test_acc"]  = test_acc
    history["test_loss"] = test_loss
    history["model_name"] = model_name

    # Save history
    hist_path = Path(save_dir) / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train S4 on Pathfinder")
    parser.add_argument("--model",       type=str,   default="s4",
                        choices=["s4", "lstm", "transformer", "compare_all"])
    parser.add_argument("--img_size",    type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--d_state",     type=int,   default=64)
    parser.add_argument("--num_layers",  type=int,   default=4)
    parser.add_argument("--n_samples",   type=int,   default=4000)
    parser.add_argument("--data_mode",   type=str,   default="synthetic",
                        choices=["synthetic", "lra"])
    parser.add_argument("--data_dir",    type=str,   default=None)
    parser.add_argument("--save_dir",    type=str,   default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = args.img_size ** 2

    # ── Data ──────────────────────────────────────────
    print(f"Loading Pathfinder-{args.img_size} ({args.data_mode} mode) …")
    train_loader, val_loader, test_loader = get_pathfinder_loaders(
        mode       = args.data_mode,
        data_dir   = args.data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        n_samples  = args.n_samples,
    )
    print(f"  Train: {len(train_loader.dataset)}  "
          f"Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    # ── Model factory ─────────────────────────────────
    def make_s4():
        return S4Model(
            d_input=1, d_model=args.d_model, d_state=args.d_state,
            num_layers=args.num_layers, num_classes=2,
        )

    def make_lstm():
        return LSTMBaseline(
            d_input=1, d_model=args.d_model,
            num_layers=args.num_layers, num_classes=2,
        )

    def make_transformer():
        return TransformerBaseline(
            d_input=1, d_model=args.d_model,
            num_layers=args.num_layers, num_classes=2,
            seq_len=seq_len,
        )

    models_to_train = []
    if args.model == "compare_all":
        models_to_train = [("s4", make_s4()), ("lstm", make_lstm()), ("transformer", make_transformer())]
    elif args.model == "s4":
        models_to_train = [("s4", make_s4())]
    elif args.model == "lstm":
        models_to_train = [("lstm", make_lstm())]
    elif args.model == "transformer":
        models_to_train = [("transformer", make_transformer())]

    # ── Train ─────────────────────────────────────────
    all_histories = {}
    for name, model in models_to_train:
        _, history = train_model(
            model_name   = name,
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            epochs       = args.epochs,
            lr           = args.lr,
            device       = device,
            save_dir     = args.save_dir,
        )
        all_histories[name] = history

    # ── Summary table ─────────────────────────────────
    if len(all_histories) > 1:
        print("\n" + "═" * 45)
        print("  FINAL COMPARISON")
        print("═" * 45)
        print(f"  {'Model':<14}  {'Test Acc':>9}  {'Params':>10}")
        print("─" * 45)
        for name, hist in all_histories.items():
            print(f"  {name.upper():<14}  {hist['test_acc']:>8.2%}  {'—':>10}")
        print("═" * 45)

    # Plot
    try:
        plot_training_curves(all_histories, save_dir=args.save_dir)
        print(f"\n  Plots saved to {args.save_dir}/")
    except Exception as e:
        print(f"  (Plotting skipped: {e})")


if __name__ == "__main__":
    main()
