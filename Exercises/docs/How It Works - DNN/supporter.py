"""Non-DNN helpers: dataset loading, tensor conversion, plotting.

Imported in main.py as:
    import supporter as Sapo
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed numpy + torch so runs are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
def load_moons(n_samples: int, noise: float, seed: int):
    """Generate the two-moons dataset and split it train / test (75 / 25)."""
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=seed)
    return X_tr, X_te, y_tr, y_te


def to_tensors(X_tr, X_te, y_tr, y_te):
    """numpy -> float32 torch tensors. Labels are shape (N, 1) to match the
    sigmoid output of DNN."""
    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_te, dtype=torch.float32).unsqueeze(1),
    )


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def plot_dataset(X_tr, X_te, y_tr, y_te, title: str = "Two-moons dataset"):
    """Side-by-side scatter of training and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, X, y, name in [
        (axes[0], X_tr, y_tr, "train"),
        (axes[1], X_te, y_te, "test"),
    ]:
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="black", s=15, label="class 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="red",   s=15, label="class 1")
        ax.set_title(f"{title} — {name}")
        ax.legend(loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()


def plot_history(history: dict, title: str = "Training curves"):
    """Loss and accuracy vs epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["tr_loss"], label="train")
    axes[0].plot(history["epoch"], history["te_loss"], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["epoch"], history["tr_acc"], label="train")
    axes[1].plot(history["epoch"], history["te_acc"], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend()

    fig.suptitle(title); plt.tight_layout(); plt.show()


def plot_decision_boundary(model, X_all, y_all, title: str = "Decision boundary"):
    """Contour plot of P(y = 1 | x) over a regular grid, with data overlaid."""
    xs = np.linspace(X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5, 300)
    ys = np.linspace(X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5, 300)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        probs = model(grid).numpy().reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
    plt.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1)
    plt.scatter(X_all[y_all == 0, 0], X_all[y_all == 0, 1], c="black", s=15, label="class 0")
    plt.scatter(X_all[y_all == 1, 0], X_all[y_all == 1, 1], c="red",   s=15, label="class 1")
    plt.title(title); plt.xticks([]); plt.yticks([]); plt.legend()
    plt.tight_layout(); plt.show()


def stack_all(X_tr, X_te, y_tr, y_te):
    """Small convenience: combine train/test into a single (X, y) pair for plots."""
    return np.vstack([X_tr, X_te]), np.concatenate([y_tr, y_te])
