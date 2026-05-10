---
tags:
  - course/ELEN90088
  - assignment
aliases:
  - Exercise 2 Notes
---

# Additional Notes — Exercise 2

> [!abstract]
> Instructive guidance for SOML 2026 Exercise 2. Key concepts, common pitfalls, and mathematical insights to help navigate each question.

---



## Question 1: Bayesian Inference, MLE and MAP

### Selecting the Prior Hyperparameters

> [!note] Key Concept — Beta Prior Mode
> For $\theta \sim \text{Beta}(a, b)$ with $a > 1$ and $b > 1$, the mode is
>
> $$
> \text{mode} = \frac{a - 1}{a + b - 2}
> $$
>
> Setting this equal to a target value $p$ and solving for $b$ gives $b = \frac{a - 1}{p} - a + 2$.

- The constraint $\text{mode} = p$ defines a **one-parameter family** of Beta distributions — you have one free degree of freedom after fixing the mode.
- Larger $a, b$ values $\Rightarrow$ sharper (more confident) prior; smaller values $\Rightarrow$ flatter (weaker) prior.

> [!tip] Choosing the Concentration
> A good rule of thumb: $a + b$ controls the "effective sample size" of your prior. Pick it to reflect how confident you are *before* seeing any data.

### Sequential Bayesian Updating Derivation

> [!warning] Common Pitfall — Conditional vs. Unconditional Independence
> The coin tosses are **conditionally independent given $\theta$**, i.e. $p(x_1, x_2 \mid \theta) = p(x_1 \mid \theta)\, p(x_2 \mid \theta)$.
>
> However, **marginally** $x_1$ and $x_2$ are **not** independent: $p(x_1, x_2) \neq p(x_1)\, p(x_2)$.
>
> This is because both observations carry information about the shared latent parameter $\theta$, which induces marginal dependence.

**Goal:** Show that $p(\theta \mid x_1, x_2) \propto p(x_2 \mid \theta)\, p(\theta \mid x_1)$ — i.e. we can update sequentially.

**1. Apply Bayes' rule to the joint data:**

$$
p(\theta \mid x_1, x_2) = \frac{p(x_1, x_2 \mid \theta)\, p(\theta)}{p(x_1, x_2)}
$$

**2. Expand the numerator using conditional independence:**

$$
p(\theta \mid x_1, x_2) = \frac{p(x_2 \mid \theta)\, p(x_1 \mid \theta)\, p(\theta)}{p(x_1, x_2)}
$$

**3. Substitute the first posterior:**

Since $p(\theta \mid x_1) = \frac{p(x_1 \mid \theta)\, p(\theta)}{p(x_1)}$, we can rewrite $p(x_1 \mid \theta)\, p(\theta) = p(\theta \mid x_1)\, p(x_1)$:

$$
p(\theta \mid x_1, x_2) = \frac{p(x_2 \mid \theta)\, p(\theta \mid x_1)\, p(x_1)}{p(x_1, x_2)}
$$

**4. Factor the denominator using the chain rule and cancel:**

By the chain rule (not marginal independence!):

$$
p(x_1, x_2) = p(x_2 \mid x_1)\, p(x_1)
$$

Substituting and cancelling $p(x_1)$:

$$
p(\theta \mid x_1, x_2) = \frac{p(x_2 \mid \theta)\, p(\theta \mid x_1)}{p(x_2 \mid x_1)}
$$

**5. Convert to proportionality:**

Since $p(x_2 \mid x_1)$ is a normalizing constant (independent of $\theta$):

$$
p(\theta \mid x_1, x_2) \propto p(x_2 \mid \theta)\, p(\theta \mid x_1)
$$

> [!success] Takeaway
> This result means we can **update beliefs sequentially**: treat the previous posterior as the new prior, multiply by the new likelihood, and renormalize. No need to reprocess all past data from scratch.

---

### Part A: Beta-Bernoulli Conjugacy

> [!note] Key Concept — Conjugate Priors
> A prior is **conjugate** to a likelihood if the posterior belongs to the same distribution family as the prior. The Beta distribution is conjugate to the Bernoulli/Binomial likelihood.

- The proof strategy is: multiply the Beta prior kernel $\theta^{a-1}(1-\theta)^{b-1}$ by the Binomial likelihood kernel $\theta^{n_H}(1-\theta)^{n-n_H}$, then **collect exponents**.
- The result $\text{Beta}(n_H + a,\; n - n_H + b)$ falls out immediately — no integration required, just pattern matching.
- The proportionality symbol $\propto$ is your best friend: it lets you ignore all normalizing constants and focus on the $\theta$-dependent terms.

> [!tip] Pseudo-Count Interpretation
> The prior parameters $a$ and $b$ act as **pseudo-counts**: $a - 1$ imaginary heads and $b - 1$ imaginary tails observed before any real data. The posterior simply adds real counts on top.

### Part B: MAP vs. MLE Estimators

> [!note] Key Concept — Log Trick
> Both MAP and MLE are optimization problems. Since $\log$ is monotonically increasing, maximizing a function is equivalent to maximizing its log. This converts products into sums, making differentiation straightforward.

- **MLE** maximizes the likelihood only: set $\frac{d}{d\theta}\log L(\theta) = 0$ and solve. Result: $\hat{\theta}_{\text{MLE}} = n_H / n$ (the sample proportion).
- **MAP** maximizes the posterior (likelihood $\times$ prior): the prior contributes extra terms $(a-1)\log\theta + (b-1)\log(1-\theta)$ to the objective. Result: $\hat{\theta}_{\text{MAP}} = (n_H + a - 1)/(n + a + b - 2)$.

> [!warning] Common Pitfall — Forgetting the Log
> Don't try to differentiate the raw product $\theta^{n_H}(1-\theta)^{n-n_H}$ directly — take the log first, then differentiate. The algebra is much cleaner.

> [!success] Takeaway — MAP as Regularised MLE
> MAP estimation is MLE with a **regularisation term** from the prior. The prior "pulls" the estimate toward its mode, especially when $n$ is small. As $n \to \infty$, the data overwhelms the prior and MAP $\to$ MLE.

### Part C: Plotting MLE vs. MAP Over Time

- The key observation is **convergence**: both MLE and MAP converge to the true $\theta$ as $n$ grows, but they behave differently for small $n$.
- **MAP starts closer** to the true value when the prior is well-chosen (here $a=2, b=6$ encodes a bias toward tails), while MLE can be erratic early on (e.g. $\hat{\theta}_{\text{MLE}} = 1$ after a single head).
- As $n \to \infty$, the prior's influence vanishes and both estimates merge — this is the visual proof that MAP $\to$ MLE asymptotically.

> [!tip] Implementation Detail
> Use `np.cumsum` on the toss sequence to get a running count of heads $n_H$ at each step. This avoids an explicit loop and vectorises the MLE/MAP formulas cleanly.

### Part D: Online Learning and Regret

> [!note] Key Concept — Online vs. Batch Prediction
> This is **not** a batch problem. At step $i$, you may only use outcomes $x_1, \ldots, x_{i-1}$ to make your prediction. You cannot look ahead.

- **MLE strategy:** Estimate $\hat{\theta}$ from past data, predict $\hat{x}_i = 1$ if $\hat{\theta} \geq 0.5$, else predict $0$. First prediction is $1$ (no data yet).
- **Optimal strategy:** Since $\theta = 0.3555 < 0.5$, the oracle always predicts $0$ (tails). This maximises expected reward because tails is more likely.

> [!warning] Common Pitfall — Early MLE Instability
> At the start, the MLE may be $\geq 0.5$ (e.g. if the first toss is heads, $\hat{\theta}_{\text{MLE}} = 1$). This causes the strategy to predict heads when it shouldn't, accumulating regret in the early phase.

> [!success] Takeaway — Regret Vanishes Asymptotically
> The normalised regret $R_n \to 0$ as $n \to \infty$. This means the MLE strategy eventually "learns" the true bias and performs as well as the oracle. The regret is concentrated in the early exploration phase — this is the fundamental **exploration cost** in online learning.

---



## Question 2: K-means Clustering

### Part A: K-means on Two-Moons

> [!note] Key Concept — K-means Produces a Voronoi Partition
> K-means minimises the within-cluster sum of squares
>
> $$
> J(\mu, z) = \sum_{i=1}^{n} \sum_{k=1}^{K} z_{ik}\, \|x_i - \mu_k\|^2
> $$
>
> The resulting decision boundaries are perpendicular bisectors between centroids — i.e. **straight lines**. Every K-means clustering is a convex Voronoi tessellation of $\mathbb{R}^d$.

- **Failure on moons:** the two moons are non-convex and interleaved, so no straight-line partition separates them. K-means with $k=2$ produces a left/right split that cuts *across* both moons.
- **Initialisation sensitivity:** Lloyd's algorithm only finds **local** minima of a non-convex objective. `init='random'` with `n_init=1` gives different clusterings for different seeds; `k-means++` + `n_init=10` (sklearn default) mitigates this.
- **Increasing $k$ doesn't help:** more centres fragment each moon into convex sub-pieces. Inertia always decreases with $k$ (monotone), but structure is never recovered.

> [!tip] Model–Data Match
> K-means is equivalent to hard-EM for a **spherical equal-covariance Gaussian mixture**. Clusters that are elongated, curved, or density-connected violate the inductive bias — use GMM, spectral clustering, or DBSCAN instead.

> [!warning] Inertia ≠ Clustering Quality
> Don't pick $k$ by minimising inertia alone — it decreases monotonically in $k$. Use the elbow heuristic, silhouette, or BIC/AIC (for GMM).

### NumPy Broadcasting in a From-Scratch K-means

> **From-Scratch code:**
>
> ```python
> X, y = noisy_moons = datasets.make_moons(n_samples=200, noise=0.1)
> 
> # K-means clustering
> # Inital Centers
> # mu = [X[0, :], X[1, :]]
> # Convert the list to a NumPy array immediately
> mu = np.array([X[0, :], X[1, :]])
> # Initialize before the loop to satisfy the linter
> closest_center = np.zeros(X.shape[0], dtype=int)
> 
> print(f"Size of X: {X.shape}, size of mu: {len(mu)}, size of expanded X: {X[:, np.newaxis].shape}")
> 
> i = 0
> j = 0
> while i < 100:  # Maximum number of iterations
>     # Assign each point to the nearest center
>     distances = np.linalg.norm(X[:, np.newaxis] - mu, axis=2)
>     closest_center = np.argmin(distances, axis=1)
>     if j == 0:
>         print(f"size of distances: {distances.shape}, size of closest_center: {closest_center.shape}")
>         j = 1
> 
>     # Update centers
>     new_mu = np.array([X[closest_center == i].mean(axis=0) for i in range(len(mu))])
> 
>     # Check for convergence (if centers do not change)
>     if np.all(mu == new_mu):
>         break
> 
>     mu = new_mu
>     i += 1
> 
> # Plot the clusters and centers
> plt.figure()
> plt.scatter(X[closest_center == 0, 0], X[closest_center == 0, 1], color='black', label='Cluster 0')
> plt.scatter(X[closest_center == 1, 0], X[closest_center == 1, 1], color='red', label='Cluster 1')
> plt.scatter(mu[:, 0], mu[:, 1], color='blue', marker='X', s=200, label='Centers')
> plt.legend()
> plt.title('K-means Clustering')
> plt.show()
> ```

> [!example] Vectorised Distance Matrix
> Given data $X \in \mathbb{R}^{n \times d}$ and centres $\mu \in \mathbb{R}^{K \times d}$, compute all pairwise distances without a Python loop:
>
> ```python
> distances = np.linalg.norm(X[:, np.newaxis] - mu, axis=2)  # shape (n, K)
> closest   = np.argmin(distances, axis=1)                    # shape (n,)
> new_mu    = np.array([X[closest == k].mean(axis=0)
>                       for k in range(K)])                   # shape (K, d)
> ```

**Shape trace (with $n=200,\ K=2,\ d=2$):**

| Expression | Shape | What it holds |
|---|---|---|
| `X` | `(200, 2)` | data points |
| `mu` | `(2, 2)` | cluster centres |
| `X[:, np.newaxis]` | `(200, 1, 2)` | data with an empty centre-axis |
| `X[:, np.newaxis] - mu` | `(200, 2, 2)` | `diff[i, k, :]` = $x_i - \mu_k$ |
| `np.linalg.norm(..., axis=2)` | `(200, 2)` | distance from point $i$ to centre $k$ |
| `np.argmin(..., axis=1)` | `(200,)` | assigned cluster for each point |

> [!info] Broadcasting Rule (the "golden rule")
> When shapes differ in rank, NumPy aligns them **from the right** and **left-pads** the shorter one with 1s. Then every dimension of size 1 is stretched to match.
>
> ```
> X[:, np.newaxis]:  (200, 1, 2)
> mu               : (  1, 2, 2)   ← left-padded
> result           : (200, 2, 2)   ← size-1 dims stretched
> ```

> [!warning] `np.newaxis` Position Gotcha
> Commas separate *existing* dimensions; `np.newaxis` inserts a new one at that slot.
>
> - `X[:, np.newaxis]` is shorthand for `X[:, np.newaxis, :]` → `(200, 1, 2)`
> - `X[:, :, np.newaxis]` (or `X[..., np.newaxis]`) → `(200, 2, 1)`
>
> The new axis goes *where you put the comma*, not at the end.

> [!tip] Mental Model
>
> In data science the leftmst axes are usually assumed to be batch related dimensions.
>
> - **Leftmost axes** = batch / collection (how many items).
> - **Rightmost axes** = the "atomic" features of one item (coordinates, channels).
> - Broadcasting aligns the atoms first, then stretches batches — this is why left-padding is the rule.

### Review of the From-Scratch Implementation (`test.py`)

> [!success] Algorithmically Correct
> The two-step loop (assign → update, break on no-change) is a valid implementation of Lloyd's algorithm and produces the right answer on the two-moons data.

> [!warning] Subtle Issues Worth Knowing
>
> 1. **Loop-variable shadowing:**
>    ```python
>    i = 0
>    while i < 100:
>        ...
>        new_mu = np.array([X[closest == i].mean(axis=0) for i in range(len(mu))])
>        i += 1   # `i` is now len(mu), not the outer counter!
>    ```
>    The list comprehension rebinds `i` to `len(mu)-1`. The outer iteration cap (`< 100`) is effectively broken — the code only terminates via the convergence `break`. Use a different variable name inside the comprehension (e.g. `k`).
>
> 2. **Fragile convergence check:** `np.all(mu == new_mu)` tests *exact* float equality. It happens to work once assignments stabilise (the same subset produces the same mean bit-for-bit), but `np.allclose(mu, new_mu)` is the safer idiom.
>
> 3. **Naive seeding:** `mu = np.array([X[0], X[1]])` uses the first two data points. Fine as a demo, but sensitive to data ordering. `k-means++` or random sampling from $X$ is more principled.

---

### Part B: GMM as a Density Estimator

> [!note] Key Concept — Gaussian Mixture Model
> A GMM models the density as
>$$
> p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \pi_k \geq 0,\ \sum_k \pi_k = 1
> $$
> 
>Fit by EM; each component $k$ has its own **mean $\mu_k$** and **covariance $\Sigma_k$**. With `covariance_type='full'`, components can be elongated and rotated ellipses — not just isotropic balls.

> [!info] Soft vs. Hard Assignment
> EM returns **responsibilities** $\gamma_{ik} = p(z_i = k \mid x_i)$ — the posterior probability that point $i$ was generated by component $k$. In sklearn these are `gmm.predict_proba(X)`. Hard labels (`gmm.predict`) are just $\arg\max_k \gamma_{ik}$.
>
> K-means is the limiting case: spherical equal covariances $\Sigma_k = \sigma^2 I$, equal weights $\pi_k = 1/K$, and a hard E-step.

- **Quadratic boundaries:** because $\log \mathcal{N}$ is a quadratic form in $x$, the boundary $\{x : \gamma_{i1}(x) = \gamma_{i2}(x)\}$ is a conic — not a straight line. This is why GMM fits the moons better than K-means with the same $K=2$.
- **Uncertainty is a feature:** soft probabilities near $0.5/0.5$ flag points the model *cannot confidently assign* — useful diagnostic information that K-means throws away.

> [!warning] A Gaussian is still unimodal and convex
> Each component is a single ellipse, so a single Gaussian cannot capture a curved moon exactly. Increasing $K$ *tiles* the curve with many small Gaussians. This is why BIC typically picks $K > 2$ on this dataset — the extra components are needed to approximate the curvature, even though the ground truth has two classes.

#### Model Selection: BIC vs. AIC

> [!info] Information Criteria
> Both penalise the maximum log-likelihood $\ell(\hat\theta)$ by model complexity $p$ (number of free parameters):
>
> $$
> \text{BIC} = -2\,\ell(\hat\theta) + p\, \log n, \qquad \text{AIC} = -2\,\ell(\hat\theta) + 2p
> $$
>
> **Lower is better.** Pick $K^\star = \arg\min_K \text{BIC}(K)$.

- For a full-covariance GMM in $d$ dimensions with $K$ components, $p = \underbrace{K d}_{\text{means}} + \underbrace{K\,\tfrac{d(d+1)}{2}}_{\text{covariances}} + \underbrace{(K-1)}_{\text{weights}}$.
- **BIC is stricter than AIC** whenever $\log n > 2$ (i.e. $n > 7$), so BIC tends to pick smaller $K$. AIC often keeps growing with $K$ on small datasets — if your AIC curve never bottoms out, that is a red flag.

> [!tip] Practical Heuristics
>
> - Fit GMMs for a **range** of $K$ (say $1$ to $10$) and plot BIC/AIC vs. $K$. Look for the minimum *or* the "elbow".
> - BIC's minimum is the principled choice for *density estimation* or when you want a parsimonious model.
> - If the BIC curve is very flat near its minimum, several values of $K$ are defensible — report the range, not just the argmin.

A common trap worth flagging explicitly:

> [!warning] BIC Selects Density Fit, Not Ground-Truth Clusters
> BIC answers *"what $K$ best describes the density?"*, not *"what $K$ matches the ground-truth classes?"*. On moons, those answers differ (density needs $K \approx 6$; ground truth has $2$). If your goal is recovering the true classes of a non-convex dataset, switch models (DBSCAN, spectral clustering) rather than tuning $K$.

Pulling the whole part together:

> [!success] Takeaway
> GMM beats K-means on two-moons because (i) full covariance captures elongation and (ii) soft responsibilities quantify uncertainty. But a mixture of Gaussians is still a mixture of *convex* components — increasing $K$ to minimise BIC fits the density better but splits each moon into multiple clusters, so it doesn't recover the two-class ground truth either.

---

## Question 3: Neural Networks

### Part 1: Implementing the `DNN` Class

> [!note] Key Concept — `nn.Module` Contract
> Two methods are mandatory:
>
> - `__init__` registers all **learnable** sub-modules as attributes (or inside an `nn.ModuleList`/`nn.Sequential`). PyTorch traverses these to collect `parameters()` for the optimizer.
> - `forward` runs one forward pass. *Never* call it directly — use `model(x)` so hooks and autograd bookkeeping fire properly.

- Storing linear layers in an **`nn.ModuleList`** (not a plain Python list) is what makes them visible to `.parameters()`, `.to(device)`, `.state_dict()`, etc. Plain lists silently hide sub-modules.
- Apply the activation **after every layer except the last**. The final `Linear` produces the logit; then a `sigmoid` maps it into $(0, 1)$ so the output can be read as $P(y = 1 \mid x)$.
- Keeping the activation class as a constructor argument (`activation=nn.ReLU`) is how you hand an easy tuning knob to the rest of the notebook without editing the class.

> [!tip] Sigmoid at the Output
> Because we want `BCELoss` on a scalar probability, squeeze the output through `torch.sigmoid`. If you instead use `BCEWithLogitsLoss`, drop the sigmoid — that loss folds the sigmoid into the loss for better numerical stability.

### Part 2: Picking a Good Combination

> [!info] A Sane Search Order
>
> 1. **Structure first.** Start at the suggested `[8, 4, 1]` and widen / deepen until train accuracy is comfortably above 95%.
> 2. **Optimizer + learning rate together.** `Adam(lr=1e-2)` is a strong default for 2D toy data; `SGD(lr=5e-2, momentum=0.9)` is competitive but LR-sensitive.
> 3. **Activation.** `ReLU` and `Tanh` are both reasonable here; `Tanh` gives smoother boundaries.
> 4. **Loss.** `BCELoss` pairs naturally with the sigmoid output.
> 5. **Epochs.** Pick last (see Q3.3).

> [!warning] Compare Fairly
> A single run's numbers are noisy — particularly with only 300 training points. Always **average over several seeds** (resample the moons *and* reseed `torch`) before declaring one combination "better" than another. Differences of $\leq 1\%$ are usually within the seed-to-seed variance.

> [!success] What Worked Best
> On noise $= 0.2$, the structure `[16, 8, 1]` with **ReLU + Adam (lr $= 10^{-2}$) + BCE** reaches ≈ 96% seed-averaged test accuracy at ~200–300 epochs — essentially tying larger networks while keeping the generalisation gap small.

### Part 3: Epoch Effect and Over-Fitting

> [!note] Key Concept — The U-shape of Test Loss
> With enough capacity and long training, the test loss curve is typically **U-shaped** (or at least L+rise shaped): it drops during the fitting phase, bottoms out near the best-generalising epoch, and then rises as the model starts to memorise the training set.

- The **training loss** keeps falling towards zero.
- The **test loss** has a clear minimum — that minimum epoch is what you would pick with **early stopping** on a held-out set.
- Train accuracy $\gg$ test accuracy is the operational signature of over-fitting. The size of the gap is a rough measure of how much effective capacity is "wasted" fitting noise.

> [!tip] When the Gap Is Invisible
> If train and test accuracy both plateau near 100% on `noise = 0.2`, the signal is drowned out. The question hint — *raise the noise to 0.3 or above* — is the right move: it injects enough label ambiguity that memorising the training set actually hurts.

> [!warning] A Small Gap Is Not Automatically Good
> If training accuracy is mediocre *and* test accuracy matches it, you are **under-fitting**, not generalising well. Check train loss first: it should be low before you congratulate the model on its small generalisation gap.

### Part 4: Advanced Variants

> [!info] Two Easy Upgrades
>
> - **Dropout** (`nn.Dropout(p)` after each hidden activation) randomly zeros a fraction of units each step. This forces redundancy across units and is a cheap, effective regulariser — it typically **delays** the over-fitting phase seen in Q3.3.
> - **Residual connections** (`x ← x + f(x)`) let gradients skip past blocks, which is critical for deep networks but mostly cosmetic on 2D toy data.

> [!tip] Remember the `.train()` / `.eval()` Switch
> Dropout and batch-norm behave differently during training and inference. Always call `model.train()` before the optimisation step and `model.eval()` (inside a `torch.no_grad()` block) when measuring loss/accuracy — otherwise your evaluation numbers are still stochastic.

> [!success] Takeaway
> Choosing structure/activation/optimiser/loss gets you to a good model; **regularisation** (dropout, weight decay, early stopping) is what closes the remaining train/test gap. The two are complementary, not competing.
