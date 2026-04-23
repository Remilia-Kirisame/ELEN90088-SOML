---
tags:
  - course/ELEN90088
  - deep-learning
  - pytorch
aliases:
  - DNN Internals
  - How It Works — DNN in Detail
---

# How It Works — DNN in Detail

> [!abstract]
> Companion note to [[How It Works - DNN Workflow]]. Where the workflow note answers *what the pipeline does*, this one answers *what happens inside the model*: how the class is wired, what a forward / backward / update step is doing at the math level, and why `model(x)` runs your `forward` method without you calling it.

---

## 1. Anatomy of the `DNN` Class

*→ `my_dnn.py::DNN`*

```python
class DNN(nn.Module):
    def __init__(self, input_dim, layer_list, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for n in layer_list:
            self.layers.append(nn.Linear(prev, n))
            prev = n
        self.activation = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return torch.sigmoid(x)
```

### 1.1 `class DNN(nn.Module)` and `super().__init__()`

`nn.Module` is the base class for every PyTorch model. It implements the plumbing that makes `.parameters()`, `.to(device)`, `.state_dict()`, `.train()` / `.eval()`, hooks, and (importantly) the `__call__` hook in §3 all work.

> [!warning] You must call `super().__init__()`
> Skip it and `nn.Module`'s internal bookkeeping dictionaries (`_parameters`, `_modules`, `_buffers`) are never created. Any `self.whatever = nn.Linear(...)` assignment after that point will **silently** land in `__dict__` instead of being registered — and the optimiser will never see the parameters.

### 1.2 `nn.ModuleList` — the parameter registry

`self.layers = nn.ModuleList([...])` stores the linear layers in a container that `nn.Module` recognises and recurses into.

> [!note] Why not a plain Python list?
> A plain `[]` is a non-module attribute. `model.parameters()` walks only `_parameters` and `_modules`, so a list of `nn.Linear` is invisible to it. `optimizer = Adam(model.parameters())` then gets an empty iterator — the optimiser steps do nothing and the loss stays flat. Use `nn.ModuleList` whenever the layers do not need a fixed ordering in attributes (`nn.Sequential` works too, and adds an auto-generated forward pass).

### 1.3 `nn.Linear(in_features, out_features)` — one affine layer

Each `nn.Linear` owns two learnable tensors:

- a weight matrix $W \in \mathbb{R}^{\text{out} \times \text{in}}$ (attribute `.weight`),
- a bias vector $b \in \mathbb{R}^{\text{out}}$ (attribute `.bias`).

Given a batch $X \in \mathbb{R}^{N \times \text{in}}$ the layer computes

$$
Y = X W^{\top} + b
$$

(PyTorch stores $W$ as `(out, in)` and transposes during the matmul, which is why the `.weight` shape looks "backwards" from the math). These tensors are created with `requires_grad=True` so autograd tracks them.

> [!info] Default initialisation
> `nn.Linear` uses Kaiming-uniform for the weights and a small uniform bias. With ReLU hidden units this keeps activation variances roughly stable through depth, so gradients neither explode nor vanish from the very first step. Change it with `nn.init.*` on `layer.weight` / `layer.bias` if you want something else.

### 1.4 The activation — where non-linearity enters

`self.activation = activation()` stores one instance (e.g. `nn.ReLU()`). It is **stateless** — no learnable parameters, just a function the forward pass applies pointwise. Without it the entire network collapses to a single affine map:

$$
x \mapsto W_L (W_{L-1} (\cdots W_1 x + b_1) + \cdots) + b_L \;=\; W_{\text{eq}} x + b_{\text{eq}}
$$

which can only draw a straight-line boundary. The activation is what lets stacked layers carve curved boundaries like the two moons.

### 1.5 `forward` — how the pieces compose

The loop threads the input through every linear layer, applying the activation after every hidden layer **except the last**. The final call `torch.sigmoid(x)` squashes the last layer's logit into $(0, 1)$ so the output reads as $P(y = 1 \mid x)$.

> [!tip] Why activation ≠ after the last layer
> The final layer produces a **score** (logit); the sigmoid maps it to a probability. Applying the hidden-layer activation here too would either destroy the output range (ReLU caps scores at $[0, \infty)$) or double up non-linearities unnecessarily.

> [!note] What `forward` does *not* do
> It does not "run the network" by itself — it only describes how to compute the output **once the graph is hooked into autograd**. That hooking happens in `__call__` (§3).

---

## 2. The Inner Process — Train and Evaluate

*→ `my_dnn.py::train_dnn` and `my_dnn.py::evaluate`*

The training step is the classic five-line dance. Conceptually each step corresponds to a well-defined math operation.

```python
optimizer.zero_grad()
out_tr = model(Xtr)            # forward
loss   = loss_fn(out_tr, ytr)  # loss
loss.backward()                # backward
optimizer.step()               # update
```

### 2.1 The Forward Pass — compute the prediction

For a depth-$L$ network (in our case $L = 3$) the forward pass is the composition

$$
\hat{y} \;=\; \sigma\!\bigl(W_L\, a_{L-1} + b_L\bigr), \qquad
a_\ell \;=\; \phi\!\bigl(W_\ell\, a_{\ell-1} + b_\ell\bigr),\quad a_0 = x,
$$

where $\phi$ is the hidden activation (ReLU) and $\sigma$ is the sigmoid. Each intermediate $a_\ell$ is cached by autograd because it will be needed by the backward pass.

At the same time, calling `model(x)` builds a **computation graph** — a DAG where every tensor operation records its inputs and the local gradient function (`grad_fn`). This graph is the scaffold that makes `.backward()` possible.

> [!info] Tensor shapes through the network
> With `LAYERS = [16, 8, 1]` and a batch of $N$ points:
>
> `(N, 2) → Linear → (N, 16) → ReLU → (N, 16) → Linear → (N, 8) → ReLU → (N, 8) → Linear → (N, 1) → sigmoid → (N, 1)`

### 2.2 The Loss — summarise the error as a scalar

For binary classification with sigmoid outputs we use **binary cross-entropy**:

$$
\mathcal{L}(\hat{y}, y) \;=\; -\frac{1}{N}\sum_{i=1}^{N} \Bigl[\, y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\,\Bigr].
$$

This is just the negative log-likelihood of a Bernoulli model with parameter $\hat{y}_i$. Importantly, $\mathcal{L}$ is a **scalar**, which is what `.backward()` needs.

> [!note] Why BCE and not MSE?
> Both work, but BCE has a much cleaner gradient through the sigmoid: $\partial \mathcal{L} / \partial z_L = \hat{y} - y$ (where $z_L$ is the pre-sigmoid logit). MSE through a sigmoid gives an extra $\sigma'(z_L) = \hat{y}(1 - \hat{y})$ factor that can nearly kill the gradient whenever the output saturates near 0 or 1.

### 2.3 The Backward Pass — propagate the gradient

`loss.backward()` applies the chain rule backwards through the graph built during the forward pass. For every parameter $\theta$ (every `.weight` and `.bias`) autograd computes

$$
\frac{\partial \mathcal{L}}{\partial \theta}
$$

and stores it in `\theta.grad`. Concretely, the chain rule unrolls as follows — letting $z_\ell = W_\ell a_{\ell-1} + b_\ell$ be the pre-activation of layer $\ell$, and $\delta_\ell = \partial \mathcal{L} / \partial z_\ell$:

1. **Output layer.** $\delta_L = \hat{y} - y$ (BCE + sigmoid simplification).
2. **Recursion.** For $\ell = L - 1, \ldots, 1$,

$$
\delta_\ell \;=\; \bigl(W_{\ell+1}^{\top}\, \delta_{\ell+1}\bigr) \odot \phi'(z_\ell).
$$

3. **Gradients w.r.t. parameters.**

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} = \delta_\ell\, a_{\ell-1}^{\top}, \qquad
\frac{\partial \mathcal{L}}{\partial b_\ell} = \delta_\ell.
$$

You do not write any of this by hand — autograd does it for you, using the cached $a_\ell$ and $z_\ell$ from the forward pass. That cache is why PyTorch cannot do backward without first doing forward under `grad` tracking, and why `torch.no_grad()` disables it in `evaluate` (§2.5).

> [!warning] Gradients accumulate by default
> Every `.backward()` call **adds** to `.grad`. If you don't call `optimizer.zero_grad()` first, the update at epoch 10 is driven by the *sum* of gradients from epochs 1–10. Always clear before the new forward pass.

### 2.4 The Optimiser Step — update the parameters

`optimizer.step()` applies an update rule to every parameter using the freshly computed `.grad`. For plain SGD:

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}.
$$

For Adam (our default) each parameter has its own adaptive step size driven by first- and second-moment estimates of the gradient:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2,
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^{t}}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^{t}}, \qquad
\theta \leftarrow \theta - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}.
$$

The per-parameter scaling by $1/\sqrt{\hat{v}_t}$ is what makes Adam robust across wildly different gradient magnitudes and why it is such a strong default on toy problems.

### 2.5 Evaluation — forward without training

*→ `my_dnn.py::evaluate`*

`evaluate` runs the same forward pass but with two switches flipped:

- **`model.eval()`** — toggles layers whose behaviour depends on the mode (dropout turns off, batch-norm uses running statistics). Our pure-linear network has nothing to toggle, but making the call is the correct habit.
- **`with torch.no_grad():`** — tells autograd **not** to build the graph. No cached $a_\ell$, no `grad_fn`, no memory for backward. This makes evaluation faster and keeps the eval forward pass from leaking into the next `.backward()`.

Accuracy is then just a threshold at $\hat{y} \ge 0.5$.

---

## 3. Why `model(Xtr)` Calls `forward`

*→ `my_dnn.py::train_dnn` line `out_tr = model(Xtr)`*

You never write `model.forward(x)` — and you should not. Here is the reason.

### 3.1 `nn.Module.__call__` is the real entry point

`nn.Module` defines a `__call__` method. In Python, writing `model(x)` is exactly equivalent to `type(model).__call__(model, x)` — i.e. it dispatches through the class's `__call__`, *not* straight into `forward`.

Roughly (simplified):

```python
def __call__(self, *args, **kwargs):
    # 1. run pre-forward hooks
    for h in self._forward_pre_hooks.values():
        h(self, args)
    # 2. actually run user-defined forward
    result = self.forward(*args, **kwargs)
    # 3. run post-forward hooks
    for h in self._forward_hooks.values():
        h(self, args, result)
    return result
```

So `model(Xtr)`:

1. fires any registered **forward-pre-hooks** (e.g. quantisation or input validation),
2. calls **your** `forward` method,
3. fires any **forward-hooks** (e.g. activation logging, mixed-precision bookkeeping).

### 3.2 Why it matters

- **Autograd relies on it.** `__call__` is where the graph-building context is set up. Bypassing it can still work for a pure `forward`, but you lose the hook machinery that downstream PyTorch features (profilers, JIT tracing, DataParallel, `torch.compile`) depend on.
- **Training / eval mode dispatch.** Some modules (dropout, batch-norm) read `self.training` inside their forward. `self.training` is flipped by `model.train()` / `model.eval()`, and `__call__` is what guarantees your call site is consistent with those flags.
- **Hooks.** If later you want to debug — "what did layer 2 output?" — a single `layer2.register_forward_hook(lambda m, i, o: print(o.shape))` works precisely because everything goes through `__call__`.

> [!success] The rule
> You **define** `forward`. You **call** `model(x)`. The base class's `__call__` is the glue that makes the two the same operation plus the right side effects.

> [!warning] Do not call `model.forward(x)` directly
> It skips the pre/post hooks, skips the dispatch that some backends rely on, and future PyTorch features may break on it silently. Stick to `model(x)` — it is the documented API.

---

## Cheat Sheet

> [!example] One-glance summary
>
> | Concept | Code | Math |
> |---|---|---|
> | Linear layer | `nn.Linear(in, out)` | $y = Wx + b$ |
> | Activation | `nn.ReLU()` | $\phi(z) = \max(0, z)$ |
> | Output squashing | `torch.sigmoid(x)` | $\sigma(z) = 1 / (1 + e^{-z})$ |
> | Loss | `nn.BCELoss()` | $-\tfrac{1}{N}\sum y\log\hat y + (1-y)\log(1-\hat y)$ |
> | Forward | `out = model(x)` | $\hat y = \sigma(W_L \phi(\cdots) + b_L)$ |
> | Backward | `loss.backward()` | chain rule fills `θ.grad` |
> | Update (Adam) | `opt.step()` | $\theta \leftarrow \theta - \eta\, \hat m_t / (\sqrt{\hat v_t} + \varepsilon)$ |
> | Eval mode | `model.eval()` + `no_grad` | no graph, no dropout |
