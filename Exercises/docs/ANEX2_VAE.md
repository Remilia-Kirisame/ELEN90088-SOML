---
tags:
  - ELEN90088
created: 2026-05-06
Author: Claude Opus 4.7
---

> [!tip]- Intuition first — read this before the formal write-up
>
> ### Step 1 — What the autoencoder is *trying* to do
>
> Imagine the following game. You're handed a CIFAR-10 image: a $3 \times 32 \times 32$ tensor, which is $3{,}072$ numbers. Your task is to describe it to a friend over the phone using **only 128 numbers**, and your friend then has to redraw the original image as faithfully as possible.
>
> - You are the **encoder**.
> - Your friend is the **decoder**.
> - The 128 numbers you pick are $z$, the **latent code**.
>
> Neither of you starts out good at this. So you both practice together: you describe, your friend redraws, you compare the redrawing to the original, and you both adjust your strategies. After enough practice (training), you learn to compress *what matters about the image* into 128 numbers, and your friend learns to translate those numbers back into pixels.
>
> That's the entire autoencoder. The "loss" $\|x - \hat{x}\|^2$ is just measuring "how off was the redrawing?" — both of you minimise it together.
>
> The bottleneck (128 numbers vs 3072 pixels) is what *forces* learning. If you were allowed 3072 numbers, you'd just copy pixel-for-pixel, and neither of you would learn anything useful. The narrowness compels you to encode *concepts* — "red car, mountains, sunset" — instead of raw pixels.
>
> ---
>
> ### Step 2 — Why this autoencoder can't *generate* new images
>
> After training, you and your friend are excellent at the round trip: image → 128 numbers → redrawn image. Now I try something different. **I throw away the encoder.** I just pick a random point in 128-dimensional space — say `[0.3, -1.7, 0.0, ..., 2.1]` — and hand it to your friend. "Redraw this."
>
> Your friend produces garbage.
>
> Why? Picture the 128-dim latent space as a vast empty room. During training, the encoder placed each training image at *some specific point* in that room — like sticking 50,000 thumbtacks on a wall. The decoder learned what to draw at each thumbtack location. **It learned nothing about the empty space between thumbtacks.** If I pick a point that isn't near any thumbtack, the decoder has no idea what should be there. Output: noise.
>
> Even worse, the encoder is free to scatter the thumbtacks however it likes. Some clusters will be packed tight, others will be far apart, and most of the room will be empty desert.
>
> So the autoencoder can compress-and-restore, but it *cannot generate*. Picking random codes doesn't work.
>
> ---
>
> ### Step 3 — The VAE's two changes (separately motivated)
>
> The VAE makes **two** changes to fix this. Each one solves a separate problem. Take them one at a time.
>
> #### Change 1 — Each image becomes a fuzzy blob, not a sharp point
>
> Instead of encoding an image to a single point, the encoder outputs a **fuzzy region**: a centre $\mu$ and a width $\sigma$. When the decoder needs a code, it samples a random point from somewhere inside that fuzzy region.
>
> What does this *do*?
>
> During training, the same input image gets encoded to slightly different codes each time (because of the random sampling). The decoder has to reconstruct the image from *all* those slightly-different codes. So the decoder learns: "any code anywhere in this fuzzy blob should redraw the same image." This means **nearby points in latent space now decode to similar images** — the empty desert immediately around each training point fills in with sensible reconstructions.
>
> This is what makes the latent space **smooth**. No more thumbtacks-on-a-wall — now you have *smudges*, each smudge covering a small neighbourhood.
>
> #### Change 2 — All the smudges huddle around the origin
>
> Smudges alone aren't enough. The encoder could still place all the smudges in one corner of the room and leave the rest empty. So we add the **KL term** to the loss, which says:
>
> >> "Your encoded smudge for every input should look like the standard normal blob $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ — centered at the origin, width 1."
>
> This is a *per-image* pressure: every single input's smudge gets pulled toward the same big anchor blob at the origin. Since every training image is being pulled toward the same target, all the smudges end up overlapping in roughly the same region of space — a big fluffy cloud centered at the origin.
>
> Now the latent space looks completely different: a *continuous, overlapping carpet of smudges*, all crowded around the origin, no more deserts.
>
> #### The result
>
> Pick a random point from $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ — i.e., somewhere in that fluffy cloud. Hand it to the decoder. Because the cloud is densely packed with overlapping training smudges, your random point *is* near where the decoder learned to draw something sensible. Out comes a plausible (if blurry) CIFAR-like image.
>
> **That's generation.** You no longer need a real image to start from — just a sample from the standard normal.
>
> ---
>
> ### The two changes are in tension — that's the whole point
>
> - **Change 1** wants the smudges *small* and *separated* (so reconstruction stays sharp — each image gets its own little neighbourhood).
> - **Change 2** wants the smudges *large* and *overlapping with the origin blob* (so the space stays sample-able).
>
> If only Change 1 wins → you get back to the autoencoder problem (sharp thumbtacks, can't sample).
> If only Change 2 wins → every image collapses to the same smudge, reconstruction is meaningless.
>
> Training finds a compromise: smudges large enough to overlap and tile the space, small enough that each image still has a recognisable neighbourhood. *That balance* is the smooth, sample-able latent space.
>
> ---
>
> ### A picture in your head
>
> Hold this image in your head when you read the rest of the file:
>
> ```
> Autoencoder:                       VAE:
>
>    . .                             ░░░░░░░░░░░░
>         .                          ░░ ▓▓ ░░ ▓▓░░
>   .         .                      ░▓▓░░▓▓░░▓░░░
>        .                           ░░▓▓░░░░▓▓░░░
>    .  .   .                        ░░░▓▓▓▓▓░░░░
>                                    ░░░░░░░░░░░░
>    sparse thumbtacks               overlapping smudges
>    in empty space                  filling a soft cloud
> ```
>
> The decoder has only learned what to draw at the dots / inside the smudges. The autoencoder leaves vast emptiness between dots. The VAE fills the cloud, so any random point inside it lands somewhere meaningful.
>
> ---
>
> Once this picture clicks, the math in Part A and the loss in Part C stop feeling arbitrary. Reparameterisation is just *how* we sample from the smudge in a way that lets gradients flow. The KL term is just *how* we measure "is your smudge close enough to the standard-normal anchor blob?" And the reconstruction term is just *how well does the decoder still redraw the input despite the fuzziness?*

## VAE Fundamentals (Q5 Context)

Convolutional Variational Autoencoder for CIFAR-10 image generation and reconstruction.

### What "generative modeling" actually means

A *discriminative* model (the SVM from Q4) learns $p(y \mid x)$ — given an image, predict the label. A **generative** model learns the *data distribution itself*: $p(x)$, the manifold of plausible CIFAR-10 images. Once you have that, you can:

- **Sample** new images that look like CIFAR-10 but never appeared in training.
- **Compress** images into a small code and reconstruct them.
- **Interpolate** smoothly between two images in code-space.

A VAE is one way to do this — others include GANs, diffusion models, and normalizing flows. VAEs are the cleanest pedagogically because every piece has an explicit probabilistic meaning.

---

### Start with the (vanilla) Autoencoder, then upgrade

A plain **autoencoder** is two neural networks chained together:

$$
x \;\xrightarrow{\text{encoder}}\; z \in \mathbb{R}^d \;\xrightarrow{\text{decoder}}\; \hat{x}
$$

Train them jointly to minimise $\|x - \hat{x}\|^2$. The hidden code $z$ is a "bottleneck" — much smaller than $x$, so the network is forced to learn a compressed representation.

**The problem.** The latent space is *unstructured*. Each training image gets mapped to *some* point in $\mathbb{R}^d$, but there's no reason nearby points should decode to plausible images, and there are huge "holes" between training points where the decoder produces garbage. So you cannot *generate* — you can only reconstruct things you've already seen.

**The VAE fix.** Force the encoder to output not a single point, but a *distribution* over $z$, and pull all those distributions toward the same anchor: the standard normal $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$. Then sampling from $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ and feeding it to the decoder produces a valid image — the latent space has been *regularised* into a continuous, sample-able manifold.

That's the entire intuition. Everything that follows is just making it mathematically honest.

---

### The encoder outputs a Gaussian, not a point

For input $x$, the encoder outputs two vectors:

- $\mu(x) \in \mathbb{R}^d$ — the mean of the latent distribution
- $\sigma^2(x) \in \mathbb{R}^d$ — the (diagonal) variance

Together they define $q(z \mid x) = \mathcal{N}(\mu(x),\, \mathrm{diag}(\sigma^2(x)))$. To get a code, you **sample** $z \sim q(z \mid x)$.

> **Practical note** — the encoder actually outputs $\log\sigma^2$ (called `log_var` in the code). Why? Variance must be positive, but a `Linear` layer can output any real number. Exponentiating $\log\sigma^2$ gives a guaranteed-positive variance with no awkward clamping. You'll see this in `reparameterize`: `std = torch.exp(0.5 * log_var)`.

---

### The reparameterisation trick (Part A)

Sampling is non-differentiable: `z = torch.normal(mu, sigma)` produces a random number, and there is no gradient $\partial z / \partial \mu$ to back-propagate through. The encoder would never learn.

**Trick:** push the randomness *outside* the differentiable path:

$$
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)
$$

Now $\epsilon$ is just a constant (drawn fresh each forward pass), and $z$ is a deterministic, differentiable function of $\mu$ and $\sigma$:

$$
\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon.
$$

Same distribution for $z$, but gradients now flow. **This is the single trick that makes VAEs trainable.** Part A.2 is just plugging numbers into this.

---

### The ELBO loss — two terms in tension

The training objective has two pieces:

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} \;+\; \underbrace{D_{KL}\!\left(q(z \mid x)\,\|\,\mathcal{N}(\mathbf{0}, \mathbf{I}_d)\right)}_{\text{regulariser}}
$$

- **Reconstruction term** — the decoder must rebuild the input faithfully. Pulls each $\mu(x)$ toward "wherever decodes back to $x$."
- **KL term** — the encoded distribution must look like the standard normal. Pulls every $\mu(x)$ toward $\mathbf{0}$ and every $\sigma(x)$ toward $1$.

The two terms **fight each other**. Pure reconstruction wants distinct, sharp codes for every image (variance → 0, means scattered far apart). Pure KL wants every input to map to the same $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ blob, which destroys reconstruction. Training balances the two — and *that balance is what creates a smooth, sample-able latent space*.

For diagonal Gaussians, the KL term has a closed form:

$$
D_{KL} = -\tfrac{1}{2}\sum_{i=1}^{d}\!\left(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2\right)
$$

— which is the one-liner inside `vae_loss`:

```python
kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

> **Why "ELBO"?** The "true" objective for a generative model is $\log p(x)$ (likelihood of the data), which is intractable. The ELBO (Evidence Lower BOund) is a tractable *lower bound* on $\log p(x)$, and it decomposes exactly into "reconstruction + KL." Maximising the ELBO is equivalent to minimising the loss above. Full derivation is graduate-level and not required here — but if you ever wonder why the loss looks the way it does, that's where it comes from.

---

### Convolutional encoder/decoder (Part B)

CIFAR-10 images are $3\times 32\times 32$. A purely fully-connected VAE would work but ignore spatial structure. So we use:

- **Encoder:** three `Conv2d` layers with stride 2 — each halves the spatial size: $32 \to 16 \to 8 \to 4$, while channels grow $3 \to 16 \to 32 \to 64$. Final feature map: $64 \times 4 \times 4 = 1024$ values, then two `Linear` heads produce $\mu$ and $\log\sigma^2$.
- **Decoder:** mirror image. A `Linear` layer expands $z$ back to $64 \times 4 \times 4$, then three `ConvTranspose2d` layers double the spatial size each time: $4 \to 8 \to 16 \to 32$. Final `Sigmoid` clamps pixels to $[0, 1]$.

Shape trace for one image:

```
(B, 3, 32, 32)
  → Conv2d → ReLU → (B, 16, 16, 16)
  → Conv2d → ReLU → (B, 32,  8,  8)
  → Conv2d → ReLU → (B, 64,  4,  4)
  → flatten        → (B, 1024)
  → fc_mu, fc_log_var → (B, d), (B, d)
  → reparameterize → z: (B, d)
  → fc_dec → ReLU → (B, 1024) → reshape → (B, 64, 4, 4)
  → ConvTranspose2d → ReLU → (B, 32,  8,  8)
  → ConvTranspose2d → ReLU → (B, 16, 16, 16)
  → ConvTranspose2d → Sigmoid → (B,  3, 32, 32)
```

The `Sigmoid` matters: the data is also in $[0, 1]$ (because `transforms.ToTensor()` divides by 255), so MSE compares like-for-like. If you used `Tanh` or no activation, the loss landscape would be off.

---

### The compression trade-off (Part D)

Smaller `latent_dim` = harsher bottleneck = blurrier reconstructions but stronger compression. Larger `latent_dim` = closer to the original but the code is bigger. Part D asks you to plot test MSE against compression ratio across $d \in \{16, 64, 128, 256\}$ and explain the curve.

Compression ratio derivation:
- One CIFAR-10 image: $32 \times 32 \times 3 \times 8 = 24{,}576$ bits
- Latent code: $32\,d$ bits (32-bit floats)
- Ratio: $\rho(d) = d / 768$

The expected story: reconstruction error **drops sharply** from $d = 16$ to $d = 64$, then **plateaus** — at some point the bottleneck stops being the limiting factor and the model is constrained by decoder capacity, KL pressure, and the stochastic sampling itself, not by the size of $z$.

---

### How to approach the question

The `.py` file already has full, correct solutions for every sub-part. So the work at this point is **understanding**, not coding:

1. **Re-read Part A** with the reparameterisation trick picture firmly in mind: stochastic node bad, deterministic-function-of-noise good.
2. **Read Part B's `ConvVAE` line by line** alongside the shape trace above.
3. **Re-derive the KL formula** yourself on paper. It's the one piece of pure math in the assignment, and it's instructive.
4. **Run the training script** once with a tiny `epochs=2` to confirm the loss decreases, before committing to the full 25-epoch × 4-model run.

---

### Quick glossary

| Term | Meaning |
|------|---------|
| Latent space | The low-dimensional code-space $\mathbb{R}^d$ where $z$ lives |
| Posterior $q(z \mid x)$ | Encoder's distribution over codes, given an input image |
| Prior $p(z)$ | Anchor distribution we pull the posterior toward — here $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ |
| ELBO | Evidence Lower BOund — tractable lower bound on $\log p(x)$ |
| KL divergence | "Distance" between two distributions; closed-form for two Gaussians |
| Reparameterisation | Trick that moves randomness out of the gradient path so back-prop works |
| `log_var` | Encoder outputs $\log\sigma^2$ to keep variance positive without clamping |
