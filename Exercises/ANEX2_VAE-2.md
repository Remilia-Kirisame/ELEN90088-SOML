---
tags:
  - ELEN90088
created: 2026-05-10
Author: Claude Opus 4.7
---

# Exercise 2 — Question 5 discussion

## Q: Why is the `ConvVAE` (Part B) configured the way it is?

The architecture is a fairly canonical "small-image conv VAE" template. Each choice is dictated by either spatial-dimension arithmetic or standard generative-model practice.

### 1. Why three conv layers with `kernel=3, stride=2, padding=1`

That exact combination is the standard "halve the spatial size" block. The output size formula gives
$$
H_\text{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1 = \left\lfloor \frac{H + 2 - 3}{2} \right\rfloor + 1 = \frac{H}{2}.
$$
So three of them take $32\!\to\!16\!\to\!8\!\to\!4$. Two would leave you at $8\times 8$ (flatten = 2048, large FC head); four would crush you to $2\times 2$ and throw away too much spatial structure. Three is the sweet spot for $32\times 32$ inputs.

### 2. Why the channels grow 3 → 16 → 32 → 64

Standard CNN bookkeeping: as you shrink spatial resolution, widen channels to preserve representational capacity. Tracking total feature count per stage:

| stage | shape | numel |
|---|---|---|
| input | $3\times 32\times 32$ | 3072 |
| after conv1 | $16\times 16\times 16$ | 4096 |
| after conv2 | $32\times 8\times 8$ | 2048 |
| after conv3 | $64\times 4\times 4$ | 1024 |

It tapers gently, building a hierarchy of features without an abrupt information bottleneck before the latent.

### 3. Why two separate `Linear` heads for `mu` and `log_var`

The encoder outputs the parameters of $q(z|x)=\mathcal{N}(\mu,\text{diag}(\sigma^2))$, not $z$ itself. The two heads are independent because $\mu$ and $\sigma^2$ are independent parameters of that distribution. Predicting `log_var` rather than `var` or `std` is deliberate:

- A `Linear` layer's output is unconstrained in $\mathbb{R}$, which matches $\log\sigma^2\in\mathbb{R}$ but **not** $\sigma^2>0$ or $\sigma>0$. Using `log_var` removes the need for an extra positivity activation.
- $\sigma=\exp(\tfrac12\log\text{var})$ is numerically stable across many orders of magnitude.
- The KL term you derived in C.1 is naturally written in $\log\sigma^2$, so it plugs straight in.

### 4. Why the decoder mirrors the encoder

`ConvTranspose2d(k=3, s=2, p=1, output_padding=1)` is the inverse-arithmetic counterpart of the encoder's strided conv: it **doubles** spatial size each layer ($4\!\to\!8\!\to\!16\!\to\!32$). Symmetric encoder/decoder is convention rather than necessity, but it makes input/output shapes line up by construction and gives the decoder roughly matching capacity to invert what the encoder did.

### 5. Why `ReLU` everywhere except a final `Sigmoid`

ReLU is the default cheap non-linearity for hidden conv stacks. The final `Sigmoid` is **required** here because of the data pipeline choice in Part C.4: `transforms.ToTensor()` puts CIFAR-10 pixels in $[0,1]$, and the MSE reconstruction loss compares the output directly to those normalised pixels. Without `Sigmoid`, $\hat{x}$ could land anywhere in $\mathbb{R}$, which both makes the MSE harder to optimise and produces unphysical "pixels" outside $[0,1]$.

### 6. Why no BatchNorm/Dropout/etc.

The exercise intentionally specifies a minimal skeleton — the goal is to see ELBO + reparameterization working end-to-end, not to chase SOTA reconstruction. Adding normalization layers would change the discussion of the compression-vs-quality trade-off in Part D.

So the configuration is essentially: *the smallest fully-convolutional encoder/decoder pair that takes $3\times 32\times 32 \leftrightarrow$ a flat latent code while keeping spatial-dim arithmetic clean.*

---

## Q: In Part C.1, why $\mathbb{E}_q[(z-\mu)^2] = \sigma^2$ and $\mathbb{E}_q[z^2] = \sigma^2 + \mu^2$?

Two pieces — *why the identities are true*, and *why those particular expectations show up*.

### Why the identities hold

When $z\sim q=\mathcal{N}(\mu,\sigma^2)$:

- $\mathbb{E}_q[(z-\mu)^2]=\sigma^2$ is **literally the definition of variance**: $\text{Var}(z)\equiv\mathbb{E}[(z-\mathbb{E}[z])^2]$, and for this $q$, $\mathbb{E}[z]=\mu$, $\text{Var}(z)=\sigma^2$.
- $\mathbb{E}_q[z^2]=\sigma^2+\mu^2$ comes from rearranging the same definition: $\text{Var}(z)=\mathbb{E}[z^2]-(\mathbb{E}[z])^2$, so $\mathbb{E}[z^2]=\text{Var}(z)+(\mathbb{E}[z])^2=\sigma^2+\mu^2$.

### Why those two expectations

Look at the integrand we built up just before that step:
$$
\log\frac{q(z|x)}{p(z)} = -\tfrac{1}{2}\log\sigma^2 \;-\; \frac{(z-\mu)^2}{2\sigma^2} \;+\; \frac{z^2}{2}.
$$
The KL is $\mathbb{E}_q\!\left[\log\frac{q}{p}\right]$, so we take the expectation under $q$ term-by-term:

1. $\mathbb{E}_q\!\left[-\tfrac12\log\sigma^2\right] = -\tfrac12\log\sigma^2$ (constant in $z$).
2. $\mathbb{E}_q\!\left[-\dfrac{(z-\mu)^2}{2\sigma^2}\right] = -\dfrac{1}{2\sigma^2}\,\mathbb{E}_q[(z-\mu)^2] = -\dfrac{1}{2\sigma^2}\cdot\sigma^2 = -\tfrac12$.
3. $\mathbb{E}_q\!\left[\dfrac{z^2}{2}\right] = \tfrac12\,\mathbb{E}_q[z^2] = \tfrac12(\sigma^2+\mu^2)$.

Adding the three:
$$
D_{KL}(q\|p) = -\tfrac12\log\sigma^2 - \tfrac12 + \tfrac12(\sigma^2+\mu^2) = -\tfrac12\!\left(1+\log\sigma^2-\mu^2-\sigma^2\right).
$$

The two specific expectations show up because terms 2 and 3 of the integrand are *exactly* a centered second moment and a raw second moment of $z$ under $q$ — and for a Gaussian both are known in closed form.

---

## Q: For C.3, How `train` training process works?

> Recap: `How It Works - DNN/`

```python
def train(model, train_loader, lr, epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_samples = len(train_loader.dataset)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(x)
            loss = vae_loss(recon_x, x, mu, log_var)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / n_samples
        print(f"Epoch {epoch:3d}/{epochs} | avg loss per image: {avg_loss:.4f}")
```

### Big picture: stochastic mini-batch gradient descent

Training a NN is just iteratively solving an optimisation problem
$$
\theta^\star = \arg\min_\theta\; \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\!\left(f_\theta(x_i),\,x_i\right),
$$
where $\theta$ are all encoder/decoder weights and $\mathcal{L}$ is the negative ELBO. Computing the full-dataset gradient ($N=50{,}000$) every step is expensive and unnecessary — a random mini-batch gives a noisy but unbiased estimate of that gradient. So the canonical recipe is:

1. **Outer loop over epochs** — one epoch is "look at every training image once".
2. **Inner loop over mini-batches** — for each batch, compute a gradient estimate and take one optimizer step.

That's the bones of every PyTorch training loop. Now the lines:

### Setup (runs once)

- `model.to(device)` — moves all parameter tensors onto the device (GPU/MPS/CPU).
- `optimizer = torch.optim.Adam(model.parameters(), lr=lr)` — instantiates Adam, handing it the iterator of trainable tensors. Adam keeps per-parameter running estimates of the gradient's first and second moments to adapt the effective step size; it's the standard "just works" optimiser for deep nets, especially VAEs where the loss landscape mixes a reconstruction term and a KL term with very different scales.
- `n_samples = len(train_loader.dataset)` — total images in the epoch (50 000 for CIFAR-10 train). 

### Outer loop — one epoch

- `model.train()` — flips the model into training mode. It's a no-op for our `ConvVAE` because we have no `Dropout` / `BatchNorm`, but it's idiomatic and protects you the moment you add such a layer.
- `running_loss = 0.0` — a plain Python float accumulator for **monitoring only**; it never enters the gradient computation.

### Inner loop — one mini-batch step

- `for x, _ in train_loader:` — the `DataLoader` yields `(images, labels)` pairs. We discard the labels with `_` because a VAE is unsupervised; we only need $x$ to reconstruct $x$.
- `x = x.to(device)` — move this batch's tensor to the same device as the model.
- `optimizer.zero_grad()` — **critical**. PyTorch *accumulates* gradients into `param.grad` across `.backward()` calls. Without zeroing, every step's gradient would include a stale sum of all previous batches and training would diverge instantly. Calling it at the top of every iteration says "fresh gradient for this batch."
- `recon_x, mu, log_var = model(x)` — forward pass. `model(x)` runs `ConvVAE.forward`, which under the hood does encoder → reparameterisation → decoder, and PyTorch builds the autograd graph as it goes.
- `loss = vae_loss(recon_x, x, mu, log_var)` — scalar negative ELBO, summed over all pixels and all images in the batch (because both MSE and KL use `reduction='sum'`-style aggregation).
- `loss.backward()` — runs reverse-mode autodiff over the graph just built, depositing $\partial L/\partial \theta$ into each parameter's `.grad` attribute. The reparameterisation trick from Part A is exactly what makes this differentiation possible — the stochastic node is now $\mu+\sigma\odot\epsilon$, a deterministic function of $\theta$ given $\epsilon$.
- `optimizer.step()` — applies the Adam update rule using the gradients now sitting in `.grad`. **This** is the only line that actually changes parameter values.
- `running_loss += loss.item()` — `.item()` pulls a Python `float` out of a 0-D tensor, **detaching** it from the autograd graph. If you accidentally accumulated `loss` (the tensor) instead, every iteration's graph would be kept alive in memory and you'd OOM (Out of Memory) within an epoch.

### After the epoch

- `avg_loss = running_loss / n_samples` — because `vae_loss` returns a *sum* over the batch, summing across batches gives a sum over the whole dataset; dividing by `n_samples` converts that to "average negative ELBO per image", which is comparable across runs with different batch sizes.

### Why this exact shape rather than alternatives

- **Why mini-batches and not full-batch?** A 50 000-image gradient is both expensive and deterministic; mini-batch noise actually helps escape sharp minima and gives many cheap updates per epoch.
- **Why Adam and not SGD?** Adam's per-parameter adaptive scale handles the mixed-magnitude ELBO objective without you having to hand-tune learning rates for the conv weights vs. the latent heads.
- **Why move data inside the loop, not all at once?** CIFAR-10 fits in VRAM, but in general you want streaming so the loop scales to datasets that don't.
- **Why no `.eval()` or `with torch.no_grad()` here?** Because this is *only* the training routine. Evaluation lives in a separate `test()` function (Part D.1) which does set those.

### One sentence to remember

> One PyTorch training step is **always** `zero_grad → forward → loss → backward → step`. Everything else (epochs, device transfer, logging) is bookkeeping around that five-call core.

---

## Q: For Part D. Is a lower compression ratio "good" (saves storage)? And does larger $d$ mean higher compression ratio and theoretically lower MSE?

Yes on both — but the terminology is worth pinning down because "compression ratio" can mean opposite things in different conventions.

### What $\rho$ means in this exercise

The exercise defines
$$
\rho(d) \;=\; \frac{\text{latent size in bits}}{\text{image size in bits}} \;=\; \frac{32\,d}{24{,}576} \;=\; \frac{d}{768}.
$$

That's the *fraction of the original size that the compressed code occupies*. So:

| $d$ | $\rho$ | meaning |
|---|---|---|
| 16 | $\approx 0.021$ | code is 2.1% the size of the image — very aggressive compression |
| 64 | $\approx 0.083$ | 8.3% |
| 128 | $\approx 0.167$ | 16.7% |
| 256 | $\approx 0.333$ | 33.3% — code is a third of the original |

**Lower $\rho$ ⇒ more storage saved.** Yes, in this exercise's convention "low ratio = good for storage".

> Heads up: some textbooks/papers use the **reciprocal** — "compression ratio of 10:1" or just "10" meaning the compressed file is 1/10 the original. In that convention *higher = more compressed*. The plot in Part D follows the *latent-over-original* convention, where lower is more compressed.

### Does larger $d$ → higher $\rho$ → lower MSE?

Yes, **all three statements are equivalent** in this setup, and that's exactly what the Part D.3 plot shows:

- $d\uparrow$ ⇒ $\rho = d/768 \uparrow$ (less aggressive compression)
- $d\uparrow$ ⇒ more capacity at the bottleneck ⇒ MSE $\downarrow$ (better reconstruction)

So as you slide right along the x-axis (larger $\rho$, less compression), the MSE drops. That is the classical **rate–distortion trade-off**:

$$
\underbrace{\text{rate}}_{\text{bits per image} \;\sim\; \rho}\quad \text{vs.}\quad \underbrace{\text{distortion}}_{\text{MSE}}.
$$

You cannot simultaneously make both small — *every additional bit of compression costs you reconstruction quality*, and vice versa.

### Why "wanted" depends on the goal

It's worth being explicit about which axis you're optimising:

- **Goal = storage / transmission cost.** You want $\rho$ as small as possible *subject to* MSE staying below some perceptual threshold.
- **Goal = reconstruction fidelity.** You want $\rho$ large (i.e. big $d$) — but at $d=3072$ there is no compression at all; you've just built an identity-like autoencoder.
- **Goal = a generative model.** You want $d$ small enough that the latent space is *useful* for sampling/interpolation (a tight, well-organised manifold) and big enough that decoded samples look reasonable. The KL term in the ELBO is what makes "small but well-organised" possible — without it, the encoder would just spread codes across $\mathbb{R}^d$ to minimise reconstruction error and you'd get a vanilla AE.

### What the actual run showed (revised)

The textbook "monotone with diminishing returns" picture **did not hold** in our experiment. The measured numbers were:

| $d$ | $\rho$ | test MSE |
|---|---|---|
| 16 | 0.0208 | **60.98** |
| 64 | 0.0833 | **52.75** ← best |
| 128 | 0.1667 | 53.04 |
| 256 | 0.3333 | 53.96 |

So:

- **Sharp drop, $d=16\to 64$** — consistent with the "sub-intrinsic-dimension bottleneck" story.
- **Plateau (and slight rise), $d\geq 64$** — extra latent dimensions are not improving reconstruction; $d=256$ is actually *worse* than $d=64$.

### Why the plateau (not a continued descent)

1. **KL pressure scales with $d$ but reconstruction loss does not.** The KL term is a sum over $d$ latent dimensions; the MSE term is a sum over a fixed $3{,}072$ pixels. As $d$ grows, the relative weight of "match the prior $\mathcal{N}(\mathbf{0},\mathbf{I}_d)$" grows, and the optimiser increasingly *prefers* to leave extra latent dimensions uninformative. This is the classic **posterior-collapse** failure mode — many dimensions of $z$ collapse onto the prior and carry no information about $x$.
2. **Trained on ELBO, evaluated on MSE.** The training objective penalises both reconstruction and KL; the test metric reports only MSE. Once $d$ is large enough that the ELBO optimum trades MSE for KL, the test MSE stops improving.
3. **Fixed 25-epoch budget.** Larger models have more parameters (mostly in `fc_mu`, `fc_log_var`, `fc_dec`) and may simply be undertrained at this schedule — likely explanation for the slight bump from $d=128$ to $d=256$.
4. **Decoder capacity is fixed.** Only the latent and `fc_dec` widen with $d$; the three `ConvTranspose2d` layers are unchanged. Even if the encoder packed more information into the latent, the decoder may lack the capacity to exploit it.

### Practical take-away

For this exact architecture, training schedule, and $\beta=1$ ELBO, **the sweet spot is $d=64$** — it gives both the best MSE and the smallest latent we tried (other than the under-capacity $d=16$). Increasing $d$ past 64 only enlarges the code without buying reconstruction quality.

To recover a continuously decreasing curve you'd need to break one of the constraints above:

- $\beta$-VAE with $\beta<1$ to weaken KL pressure;
- train the larger models for more epochs;
- scale decoder capacity proportionally to $d$.

### Sanity-check intuition

A useful mental picture: $\rho$ tells you "how wide is the pipe between encoder and decoder, as a fraction of the input". A very narrow pipe ($\rho\!\to\!0$) forces the network to throw away information — great for storage, bad for fidelity. A wide pipe should preserve more, but with $\beta=1$ ELBO regularisation the KL term will *cap* the useful pipe width: at some $d^\star$ the model stops opening the pipe further, even if you give it more nominal capacity.
