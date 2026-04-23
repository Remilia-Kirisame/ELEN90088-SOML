---
tags:
  - course/ELEN90088
  - deep-learning
  - pytorch
aliases:
  - DNN Workflow
  - How It Works — DNN
---

# How It Works — DNN Workflow

> [!abstract]
> End-to-end walkthrough of the minimal DNN pipeline in this folder. The code is split three ways so each file has one job: **`my_dnn.py`** defines the model and the training loop, **`supporter.py`** handles everything that is not a neural network (data, tensors, plots), and **`main.py`** is a thin script that glues them together in block order.

---

## Quick Reference

> [!info] Workflow Chart
>
> ```
> ┌───────────────────────────────────────────────────────────────┐
> │ BLOCK 0  hyperparameters                                      │
> │     LAYERS, ACTIVATION, OPTIMIZER, LEARNING_RATE, LOSS, ...   │
> ├───────────────────────────────────────────────────────────────┤
> │ BLOCK 1  Sapo.set_seed            → reproducibility           │
> │ BLOCK 2  Sapo.load_moons          → (X_tr, X_te, y_tr, y_te)  │
> │          Sapo.plot_dataset        → sanity check              │
> │ BLOCK 3  Sapo.to_tensors          → float32 tensors, y ∈ ℝ^N×1│
> │ BLOCK 4  build_model              → DNN(2, [16,8,1], ReLU)    │
> │          make_optimizer           → Adam(lr = 0.01)           │
> │          make_loss                → BCELoss                   │
> │ BLOCK 5  train_dnn       ╭──────────────────────────────────╮ │
> │                          │  for ep in 1..EPOCHS:            │ │
> │                          │    zero_grad → forward → loss    │ │
> │                          │    backward  → step              │ │
> │                          │    (record metrics periodically) │ │
> │                          ╰──────────────────────────────────╯ │
> │ BLOCK 6  evaluate                 → final train / test stats │
> │ BLOCK 7  Sapo.plot_history        → loss + accuracy curves   │
> │          Sapo.plot_decision_boundary → contour of P(y=1|x)   │
> └───────────────────────────────────────────────────────────────┘
> ```

> [!tip] File Map
> - **`my_dnn.py`** — `DNN`, `build_model`, `make_optimizer`, `make_loss`, `train_dnn`, `evaluate`, `predict`
> - **`supporter.py`** (aliased `Sapo`) — `set_seed`, `load_moons`, `to_tensors`, `plot_dataset`, `plot_history`, `plot_decision_boundary`, `stack_all`
> - **`main.py`** — one block per numbered step above

> [!example] Default hyperparameters
> `LAYERS = [16, 8, 1]`, `ACTIVATION = nn.ReLU`, `OPTIMIZER = "adam"`, `LEARNING_RATE = 0.01`, `LOSS = "bce"`, `EPOCHS = 300`. Expected result on `noise = 0.2`: train ≈ 98 %, test ≈ 94–96 %.

---

## The Pipeline, Step by Step

### 1. Hyperparameters Up Front
*→ `main.py` BLOCK 0*

Every knob — network shape, activation, optimiser, learning rate, loss, epoch count — is a module-level constant at the top of `main.py`. Change one, re-run the file, and the entire pipeline below uses the new value.

> [!success] Why front-loaded?
> Nothing about the *workflow* changes when you tune a hyperparameter. Keeping them in one visible block makes it obvious what is a *choice* (tunable) vs. what is *machinery* (fixed).

### 2. Reproducibility
*→ `Sapo.set_seed` in `supporter.py`*

Sets the numpy and torch RNGs from a single seed. Called once at the start of `main.py`. This pins the dataset sample, the train/test split, and the random initialisation of the network, so two runs with the same hyperparameters give identical curves.

### 3. Data Loading and Inspection
*→ `Sapo.load_moons`, `Sapo.plot_dataset`*

`load_moons` wraps `sklearn.datasets.make_moons` and `train_test_split` so `main.py` never imports sklearn directly. The default split is 75 / 25.

> [!note] Shape convention
> `X` is `(N, 2)`, `y` is `(N,)`. Both are still numpy arrays at this stage — conversion to torch happens in the next block.

### 4. Tensor Conversion
*→ `Sapo.to_tensors`*

Casts to `float32` and **reshapes labels to `(N, 1)`**. That matters: `BCELoss` needs the predictions and targets to have the same shape, and our model's last `Linear` outputs shape `(N, 1)`. If you leave `y` as `(N,)` you get a silent broadcasting bug with the wrong loss.

### 5. Model Definition
*→ `DNN` / `build_model` in `my_dnn.py`*

```python
class DNN(nn.Module):
    def __init__(self, input_dim, layer_list, activation=nn.ReLU):
        self.layers = nn.ModuleList([nn.Linear(prev, n) for ...])
        self.activation = activation()
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return torch.sigmoid(x)
```

Two points worth calling out:

> [!note] Why `nn.ModuleList` and not a plain Python list?
> PyTorch only recognises sub-modules that live in attributes it can walk (`nn.Module`, `nn.ModuleList`, `nn.Sequential`). A plain `[]` would hide the linear layers from `model.parameters()`, and the optimiser would update nothing. You would train silently for 300 epochs and wonder why the loss never moved.

> [!note] Why no activation after the last layer?
> The last `Linear` produces a **logit** — the real-valued score we want to classify. The `torch.sigmoid` at the bottom of `forward` is the only non-linearity on the output path, and it maps the logit to $(0, 1)$ so it can be read as $P(y = 1 \mid x)$ and fed to `BCELoss`.

### 6. Optimiser and Loss
*→ `make_optimizer`, `make_loss` in `my_dnn.py`*

Factories that return a torch object by name. Keeping them as strings in `main.py` means the hyperparameter block stays human-readable; the factory does the `torch.optim.Adam` / `nn.BCELoss` lookup.

> [!info] BCELoss vs. BCEWithLogitsLoss
> We use `BCELoss` because our model already applies `sigmoid`. For numerically-stable training in production you would drop the sigmoid from the model and use `BCEWithLogitsLoss` instead — the two are mathematically equivalent but the logits version avoids underflow.

### 7. The Training Loop
*→ `train_dnn` in `my_dnn.py`*

The core five-line dance, repeated `EPOCHS` times:

```python
optimizer.zero_grad()          # (a) clear stale gradients
out = model(Xtr)               # (b) forward pass
loss = loss_fn(out, ytr)       # (c) scalar loss
loss.backward()                # (d) autograd populates .grad on every parameter
optimizer.step()               # (e) apply the update rule (Adam / SGD / ...)
```

> [!warning] Forgetting `zero_grad` is the classic mistake
> PyTorch *accumulates* gradients by default (because RNN / graph-style models need it). Omit `zero_grad` and your parameter update at epoch 10 is driven by the sum of gradients from epochs 1–10 — training goes wildly off.

> [!note] Full-batch vs. mini-batch
> With only 300 training points we feed the entire training set as one batch per epoch. For larger datasets you would wrap `Xtr` in a `DataLoader` and run an inner loop over mini-batches per epoch; the five-line dance is the same.

Every `RECORD_EVERY` epochs the loop switches the model to eval mode, runs a no-grad evaluation on both splits, and appends to a `history` dict. That history is what the plots consume.

### 8. Evaluation
*→ `evaluate` in `my_dnn.py`*

Two non-negotiable steps:

1. `model.eval()` — toggles layers that behave differently at inference (dropout, batch-norm). On this pure-`Linear + ReLU` network there is nothing to toggle, but it costs nothing and it is the correct habit.
2. `with torch.no_grad():` — disables the autograd tape. Saves memory and time, and prevents a subtle bug where the evaluation graph accidentally contributes gradients on the next `.backward()`.

Accuracy is computed by thresholding the sigmoid at 0.5.

### 9. Plots
*→ `Sapo.plot_history`, `Sapo.plot_decision_boundary`*

- **`plot_history`** — train/test loss and accuracy vs. epoch. This is the curve that tells you *when* to stop. For the defaults you will see the test-loss curve start to rise around epoch ~100 while train-loss keeps falling: the early signs of over-fitting that Q3.3 explores in depth.
- **`plot_decision_boundary`** — forward-pass the model on a regular grid, read out $P(y = 1 \mid x)$, and contour it. The $0.5$ contour is the decision boundary; the colour strength shows the model's confidence.

---

## Mapping to the Notebook's Q3

> [!tip] What carries over
> The `DNN` class, the training loop, and the evaluation helper in `my_dnn.py` are the same ones used in the Q3.2 baseline cell of `SOML2026_Exercise_2.ipynb`. The difference is purely packaging — this folder strips away the sweep/epoch/advanced-variant experiments so the underlying pipeline is the only thing on the page.

| Concept in Q3 | Where it lives here |
|---|---|
| `DNN` class (Q3.1) | `my_dnn.py::DNN` |
| Best combo (Q3.2) | `main.py` BLOCK 0 defaults |
| Training loop | `my_dnn.py::train_dnn` |
| Train/test gap story (Q3.3) | `history` + `Sapo.plot_history` |
| Advanced variants (Q3.4) | *not included* — kept out for clarity |

---

## Common Gotchas

> [!warning] Silent bugs that cost hours
> - Labels with shape `(N,)` instead of `(N, 1)` → broadcasting explodes `BCELoss`.
> - A Python list of `nn.Linear` instead of `nn.ModuleList` → optimiser gets zero parameters, nothing updates.
> - Forgotten `optimizer.zero_grad()` → gradients accumulate across epochs.
> - Evaluating without `torch.no_grad()` → VRAM creep and accidental graph retention.
> - Integer tensors fed into a float network → opaque dtype errors. Always cast inputs to `torch.float32`.
