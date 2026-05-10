"""DNN-related building blocks.

Everything that touches torch.nn lives here:
    - The DNN model class.
    - Factory helpers for the optimizer and the loss.
    - The training loop.
    - The evaluation helper.

main.py imports these as flat names (`from my_dnn import ...`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
class DNN(nn.Module):
    """A simple feed-forward binary classifier.

    `layer_list` gives the width of each layer (the last entry is the output
    and must be 1 for this binary-classification setup). The activation is
    applied between every hidden layer but not after the final one; the raw
    final logit is squashed through a sigmoid so the output can be read as
    P(y = 1 | x).
    """

    def __init__(self, input_dim: int, layer_list: list[int], activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for n in layer_list:
            self.layers.append(nn.Linear(prev, n))
            prev = n
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return torch.sigmoid(x)


def build_model(input_dim: int, layer_list: list[int], activation: type[nn.Module] = nn.ReLU) -> DNN:
    """Thin wrapper so main.py never imports torch directly."""
    return DNN(input_dim=input_dim, layer_list=layer_list, activation=activation)

def make_activation(name: str) -> type[nn.Module]:
    """Return a torch activation by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    raise ValueError(f"unknown activation: {name}")


# ---------------------------------------------------------------------------
# 2. Optimizer / loss factories
# ---------------------------------------------------------------------------
def make_optimizer(params, name: str, lr: float):
    """Return a torch optimizer by name."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def make_loss(name: str):
    """Return a torch loss by name. Only BCE/MSE here — we have a sigmoid output."""
    name = name.lower()
    if name == "bce":
        return nn.BCELoss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"unknown loss: {name}")


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
def train_dnn(
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    *,
    optimizer,
    loss_fn,
    epochs: int,
    record_every: int = 1,
    verbose: bool = True,
):
    """Full-batch training. Returns a history dict with per-record-step metrics.

    history = {
        "epoch":    [...],
        "tr_loss":  [...],
        "te_loss":  [...],
        "tr_acc":   [...],
        "te_acc":   [...],
    }
    """
    history = {"epoch": [], "tr_loss": [], "te_loss": [], "tr_acc": [], "te_acc": []}

    for ep in range(1, epochs + 1):
        # --- one training step (full batch) ---
        model.train()
        optimizer.zero_grad()          # (a) reset gradients
        out_tr = model(Xtr)            # (b) forward pass
        loss = loss_fn(out_tr, ytr)    # (c) compute loss
        loss.backward()                # (d) backward pass: autograd fills .grad
        optimizer.step()               # (e) parameter update

        # --- record metrics every `record_every` epochs ---
        if ep % record_every == 0 or ep == epochs:
            tr_loss, tr_acc = evaluate(model, Xtr, ytr, loss_fn)
            te_loss, te_acc = evaluate(model, Xte, yte, loss_fn)
            history["epoch"].append(ep)
            history["tr_loss"].append(tr_loss)
            history["te_loss"].append(te_loss)
            history["tr_acc"].append(tr_acc)
            history["te_acc"].append(te_acc)
            if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs):
                print(
                    f"  epoch {ep:4d}  "
                    f"train loss {tr_loss:.4f} acc {tr_acc:.3f}  |  "
                    f"test loss {te_loss:.4f} acc {te_acc:.3f}"
                )

    return history


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn) -> tuple[float, float]:
    """Return (loss, accuracy) on a batch. No gradient tracking."""
    model.eval()
    with torch.no_grad():
        out = model(X)
        loss = loss_fn(out, y).item()
        acc = ((out >= 0.5).float() == y).float().mean().item()
    return loss, acc


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Return hard 0/1 predictions for X."""
    model.eval()
    with torch.no_grad():
        return (model(X) >= 0.5).float()
