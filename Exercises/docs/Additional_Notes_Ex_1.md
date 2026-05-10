[TOC]



## Question 5

### Python Patterns Used in Q5.2–Q5.3

#### 1. Higher-order functions and `*args` / `**kwargs`

In Python, functions are **first-class objects** — they can be passed as arguments. A function that receives another function as a parameter is called a **higher-order function**. In `fit_logistic`, we pass `update_weight=update_weight_gd` so the optimisation loop doesn't need to know which update rule it's running — you can swap in a different one without changing `fit_logistic`.

- `**kwargs` collects all unrecognised keyword arguments into a dictionary and forwards them. This lets `fit_logistic` pass `lr=1.0` through to `update_weight_gd` without explicitly knowing about `lr`.
- `*args` does the same for **positional** arguments — collects them into a tuple.

> **Examples of `*args` and `**kwargs`:**
>
> ```python
> # *args collects positional arguments into a tuple
> def add_all(*args):
>     return sum(args)
> add_all(1, 2, 3)        # args = (1, 2, 3), returns 6
>
> # **kwargs collects keyword arguments into a dict
> def print_info(**kwargs):
>     for k, v in kwargs.items():
>         print(f"{k} = {v}")
> print_info(name="Alice", age=25)   # kwargs = {'name': 'Alice', 'age': 25}
>
> # Forwarding: the outer function doesn't need to know the inner function's parameters
> def outer(func, **kwargs):
>     return func(**kwargs)
>
> def greet(name, greeting="Hello"):
>     return f"{greeting}, {name}!"
>
> outer(greet, name="Bob", greeting="Hi")  # "Hi, Bob!"
> # `outer` never mentions `name` or `greeting` — it just passes them through
> ```

#### 2. Mutable references and `.copy()` — why numpy differs from C++

In Python, variables holding numpy arrays (and lists, dicts, etc.) are **references** (pointers), not value copies. Assignment `b = a` makes `b` point to the **same object** — this is like C++ pointer assignment, not value assignment.

```python
a = np.array([1, 2, 3])
b = a           # b and a point to the SAME array object
b[0] = 999
print(a)        # [999, 2, 3] — a is changed too!
```

In C++ terms: Python's `b = a` for mutable objects behaves like `int* b = a` (pointer copy), not like `int b = *a` (value copy). There is no implicit deep copy.

This is why we use `.copy()` in three places:

- `w = w_init.copy()` — don't mutate the caller's input
- `w_history = [w.copy()]` — snapshot the initial state
- `w_history.append(w.copy())` — snapshot each iteration

Without `.copy()`, every entry in `w_history` would be a reference to the same array, and they'd all end up showing the final value of `w`.

> **Note:** This applies to mutable objects (arrays, lists, dicts). For immutable types (int, float, str, tuple), reassignment creates a new object anyway, so the distinction doesn't arise.

#### 3. Regularisation and `C=np.inf`

**Regularisation** adds a penalty to the loss to discourage large weights and prevent overfitting:

$$
\text{Total loss} = \hat{R}(\mathbf{w}) + \frac{1}{2C}\|\mathbf{w}\|^2
$$

The $\frac{1}{2C}\|\mathbf{w}\|^2$ term is **L2 regularisation** (also called ridge penalty). In sklearn's `LogisticRegression`, the parameter `C` controls the inverse regularisation strength:

- Small `C` → strong penalty → weights shrink toward zero
- Large `C` → weak penalty → closer to unregularised
- `C = np.inf` → penalty term is $0$ → **all regularisation disabled**

This is equivalent to the older `penalty=None` (deprecated in newer sklearn versions). We need this for a fair comparison since our implementation has no regularisation.

#### 4. sklearn's `.fit()` API

sklearn separates **configuration** from **computation**:

- `LogisticRegression(C=np.inf)` — creates the model object, sets hyperparameters (no training yet)
- `.fit(X, y)` — runs the actual optimisation, populates `.coef_` and `.intercept_`
- `.predict(X)` / `.predict_proba(X)` — uses learned weights for inference

#### 5. Shape of `coef_` and `intercept_` — 1D arrays and broadcasting

- `clf.coef_` has shape `(1, n_features)` — a 2D row vector (one row per class)
- `clf.intercept_` has shape `(1,)` — a 1D array

After `clf.coef_[0]` we get a 1D array of shape `(n_features,)`. In numpy, **1D arrays are neither row nor column vectors** — they have shape `(n,)`, not `(1, n)` or `(n, 1)`. They adapt via broadcasting, which can produce different results depending on context:

> **When 1D vs 2D matters:**
>
> ```python
> v = np.array([1, 2, 3])          # shape (3,)
> row = np.array([[1, 2, 3]])      # shape (1, 3)
> col = np.array([[1], [2], [3]])  # shape (3, 1)
>
> M = np.ones((3, 3))
>
> # 1D: broadcasts as a row — adds [1,2,3] to every row
> M + v        # shape (3, 3)
>
> # Explicit row: same result
> M + row      # shape (3, 3)
>
> # Explicit column: adds 1 to row 0, 2 to row 1, 3 to row 2
> M + col      # shape (3, 3), different result!
>
> # Matrix multiplication also differs:
> v @ v        # dot product, scalar: 14
> row @ col    # matrix product, shape (1, 1): [[14]]
> col @ row    # outer product, shape (3, 3)
> ```
>
> Rule of thumb: numpy treats 1D arrays as rows for broadcasting (aligns on the last axis), but `@` treats them flexibly — as a row on the left, column on the right.

---




## Question 1

**Subquestion 5**

**Explicit coefficient conditions:**

For the general degree-4 polynomial, the Hessian $H(x)$ is a $2\times 2$ matrix of degree-2 polynomials in $x$:

$$
H_{11}(x) = 12c_{40}x_1^2 + 6c_{31}x_1x_2 + 2c_{22}x_2^2 + 6c_{30}x_1 + 2c_{21}x_2 + 2c_{20}
$$

$$
H_{12}(x) = 3c_{31}x_1^2 + 4c_{22}x_1x_2 + 3c_{13}x_2^2 + 2c_{21}x_1 + 2c_{12}x_2 + c_{11}
$$

$$
H_{22}(x) = 2c_{22}x_1^2 + 6c_{13}x_1x_2 + 12c_{04}x_2^2 + 2c_{12}x_1 + 6c_{03}x_2 + 2c_{02}
$$

By Sylvester's criterion, $H(x)\succeq 0$ for all $x$ iff:

$$
\text{(i)}\quad H_{11}(x) \geq 0 \quad \forall x \in \mathbb{R}^2
$$

$$
\text{(ii)}\quad H_{11}(x)\,H_{22}(x) - H_{12}(x)^2 \geq 0 \quad \forall x \in \mathbb{R}^2
$$

**Why condition (i) is tractable:** $H_{11}(x)$ is a degree-2 polynomial in $x$. Checking whether a quadratic polynomial is globally non-negative is equivalent to an SDP (semidefinite program): it requires the quadratic part to be PSD and the minimum of the resulting quadratic to be $\geq 0$. Concretely, $H_{11} \geq 0$ everywhere iff
$$
c_{40} \geq 0, \quad c_{22} \geq 0, \quad 36c_{31}^2 \leq 96 c_{40}c_{22}
$$
and the shifted minimum $2c_{20} - \frac{(6c_{30},\,2c_{21})\,A^{-1}(6c_{30},\,2c_{21})^\top}{4} \geq 0$ (where $A$ is the matrix of the degree-2 part), with similar conditions for $H_{22}$.

**Why condition (ii) is NP-hard:** $H_{11}H_{22} - H_{12}^2$ is a degree-4 polynomial in $x$. Certifying that an arbitrary degree-4 polynomial is globally non-negative is NP-hard, and no finite set of algebraic inequalities on the $c_{ij}$ fully characterises it.

**Contrast with $d=2$:** For a quadratic, the Hessian is a constant matrix — conditions (i) and (ii) collapse to two scalar inequalities checked once:
$$
2c_{20} \geq 0 \quad \text{and} \quad 4c_{20}c_{02} - c_{11}^2 \geq 0
$$
The jump from $d=2$ to $d=4$ replaces a single constant-matrix PSD check with a global polynomial non-negativity problem.

---



## Question 3

> [!IMPORTANT]
>
> TODO:
>
> Refresh the memory of KKT, refer 【ELEN90026】Intro2Opt Lecture 3 Slides.

### KKT Conditions — A Detailed Walkthrough

In Q3.2 we solved $\max_p S(p)$ subject to $0 \leq p \leq 1$ using direct comparison of candidates. The KKT (Karush–Kuhn–Tucker) conditions provide a more systematic alternative for constrained optimisation. Here we unpack every piece.

#### Setting up: converting to standard form

KKT is stated for **minimisation** with **inequality constraints of the form $g_i(x) \leq 0$**. So we rewrite:

$$
\min_p \; -S(p) \quad \text{subject to} \quad \underbrace{-p}_{g_1(p)} \leq 0, \quad \underbrace{p - 1}_{g_2(p)} \leq 0
$$

- $g_1(p) = -p \leq 0$ encodes $p \geq 0$.
- $g_2(p) = p - 1 \leq 0$ encodes $p \leq 1$.

The **Lagrangian function** is given by,
$$
\bold{L} (x, \lambda) = f(x) - \sum_{i\in \mathcal{E\cup I}} \lambda_i g_i (x)
$$

#### What are the Lagrange multipliers $\lambda_i$?

Each inequality constraint $g_i(p) \leq 0$ gets a **Lagrange multiplier** $\lambda_i \geq 0$. Intuitively:

- $\lambda_i$ measures how much the objective would improve if constraint $g_i$ were relaxed slightly. It's the "price" or "shadow cost" of the constraint.
- If a constraint is **not active** at the optimum (i.e., $g_i(p^*) < 0$, the inequality holds strictly), then relaxing it doesn't help — so $\lambda_i = 0$.
- If a constraint **is active** (i.e., $g_i(p^*) = 0$, the solution sits right on the boundary), then $\lambda_i$ may be positive.

This relationship is formalised as **complementary slackness**: $\lambda_i \cdot g_i(p^*) = 0$ for each $i$.

#### What is $p^*$?

$p^*$ denotes the **optimal solution** — the value of $p$ that achieves the minimum of $-S$ (equivalently, the maximum of $S$). It's what we're solving for.

#### What does "interior point" mean?

The **feasible set** here is the closed interval $[0, 1]$.

- Its **interior** is the open interval $(0, 1)$ — all points strictly between the boundaries.
- Its **boundary** consists of just the two endpoints $\{0, 1\}$.

Since $p^* = 1/N$ and $N \geq 2$, we have $0 < 1/N < 1$, so $p^*$ lies in the interior — it doesn't touch either wall of the feasible region.

#### Why interior point $\Rightarrow$ inactive constraints $\Rightarrow$ $\lambda_i = 0$

This is the key chain of logic:

1. **$p^* = 1/N$ is interior**, meaning $p^* > 0$ and $p^* < 1$.

2. **Both constraints are inactive** (satisfied with strict inequality):

   

3. **Complementary slackness forces both multipliers to zero:**
   
4. **Stationarity simplifies to the unconstrained condition.** The full stationarity condition is:

$$
-S'(p^*) + \lambda_1 \cdot (-1) + \lambda_2 \cdot (1) = 0
$$

   With $\lambda_1 = \lambda_2 = 0$ this reduces to $S'(p^*) = 0$ — the constraints exist but don't "bite" at the optimum.

#### When would the multipliers be nonzero?

If the problem had a tighter constraint, say $p \leq 1/(2N)$, the unconstrained optimum $1/N$ would violate it. The solution gets pushed to $p^* = 1/(2N)$, that constraint becomes **active** ($g(p^*) = 0$), and its multiplier $\lambda > 0$ quantifies how much the objective suffers from being constrained.



#### Using KKT to *find* $p^*$ (without knowing it in advance)

The walkthrough above assumed we already knew $p^* = 1/N$ and verified it satisfies KKT. But KKT can also work the other way: solve the KKT system to **discover** $p^*$ from scratch.

The idea: complementary slackness ($\lambda_i \cdot g_i = 0$) means for each constraint, either $\lambda_i = 0$ or $g_i = 0$. This creates a finite number of **cases** — enumerate them all, solve each, and compare. With $m$ inequality constraints there are $2^m$ cases (here $m = 2$, so 4 cases).

Recall the stationarity condition for $\min_p\, {-S(p)}$ with Lagrangian $L = -S(p) + \lambda_1(-p) + \lambda_2(p-1)$:

$$
-S'(p^*) - \lambda_1 + \lambda_2 = 0
$$

**Case 1: Both constraints active** ($g_1 = 0$ and $g_2 = 0$, i.e., $p = 0$ and $p = 1$). Contradiction — impossible.

**Case 2: Only $g_1$ active** ($p^* = 0$, $\lambda_2 = 0$). Stationarity gives $-S'(0) - \lambda_1 = 0$. Since $S'(0) = N$, we get $\lambda_1 = -N < 0$. This violates dual feasibility ($\lambda_1 \geq 0$), so **not a valid KKT point**.

**Case 3: Only $g_2$ active** ($p^* = 1$, $\lambda_1 = 0$). Stationarity gives $-S'(1) + \lambda_2 = 0$. Since $S'(1) = 0$, we get $\lambda_2 = 0 \geq 0$ ✓. Valid KKT point, but $S(1) = 0$.

**Case 4: Neither constraint active** ($\lambda_1 = \lambda_2 = 0$, $p^* \in (0,1)$). Stationarity becomes $S'(p^*) = 0$, giving $p^* = 1/N$. Check: $0 < 1/N < 1$ ✓. This gives $S = \left(\frac{N-1}{N}\right)^{N-1} > 0$.

**Comparing valid KKT points:** Case 3 gives $S = 0$; Case 4 gives $S > 0$. The global optimum is Case 4: $p^* = 1/N$.

This case-enumeration approach is how KKT works as a **solving tool**, not just a verification tool.

#### Summary table

| Concept | Meaning in this problem |
|---------|------------------------|
| $p^*$ | The optimal transmission probability ($= 1/N$) |
| $\lambda_i$ | Shadow price of constraint $i$; how much the objective improves per unit of relaxation |
| Interior point | $p^*$ lies strictly inside $(0,1)$, not on the boundary |
| Inactive constraint | $g_i(p^*) < 0$: the constraint holds with room to spare |
| Complementary slackness | Inactive constraint $\Rightarrow$ $\lambda_i = 0$; the constraint doesn't influence the solution |

---



## Question 4

### Physical Meaning of the Nonlinear SINR (Q4.4)

The **original SINR** models **additive interference** — each unwanted transmitter contributes independently to the noise floor:

$$
S_i = \frac{G_{ii}P_i}{\sigma_i + \sum_{k \neq i} G_{ik}P_k}
$$

The **modified SINR** (Q4.4) models **cross-interference** or **intermodulation** — interference arises from the *product* of signals from pairs of other transmitters:

$$
S_i = \frac{G_{ii}P_i}{0.5\sum_{j \neq i}\left(G_{ij}P_j \sum_{k \neq i,\, k \neq j} G_{ik}P_k\right) + \sigma_i}
$$

- In real RF systems, nonlinear components (amplifiers, mixers) can cause two interfering signals to combine multiplicatively, producing spurious emissions at new frequencies — **intermodulation products**.
- For $N=3$, the double sum collapses neatly: the interference at receiver $i$ is simply the product of the received interference powers from the other two transmitters (e.g., $G_{12}G_{13}P_2P_3$ for receiver 1).

### Why the Original SINR Reduces to an LP but the New One Cannot

This is the central insight the question is after. Both problems are valid GPs, but they differ in what happens when you rearrange the SINR constraint.

**Original SINR — rearranges to a linear constraint:**

Starting from $\dfrac{G_{ii}P_i}{\sigma + \sum_{k \neq i} G_{ik}P_k} \geq S^{\min}$, multiply both sides by the (positive) denominator:

$$
G_{ii}P_i \geq S^{\min}\sigma + S^{\min}\sum_{k \neq i} G_{ik}P_k
$$

$$
G_{ii}P_i - S^{\min}\sum_{k \neq i} G_{ik}P_k \geq S^{\min}\sigma
$$

This is an **affine inequality** in $P$. Combined with the linear objective $\sum P_i$ and box constraints, the entire problem is a **Linear Program**.

**New SINR — inherently nonlinear:**

Starting from $\dfrac{G_{ii}P_i}{G_{ij}G_{ik}P_jP_k + \sigma} \geq S^{\min}$ (for $N=3$), the same rearrangement gives:

$$
G_{ii}P_i - S^{\min}G_{ij}G_{ik}P_jP_k \geq S^{\min}\sigma
$$

The term $P_jP_k$ is a **product of two decision variables** — this is not affine. No substitution or rearrangement can eliminate this nonlinearity. Under the GP log-transform $y_i = \log P_i$, the term becomes $e^{y_j + y_k}$, which contributes to a log-sum-exp structure — **convex but not affine**. So the problem is a GP (and therefore convex), but **cannot** be reduced to an LP.

### Converting the Original Problem to LP Form

> **Recall:**
> - A **Linear Program (LP)** has a linear objective and linear inequality/equality constraints: $\min \mathbf{c}^\top \mathbf{x}$ s.t. $A\mathbf{x} \leq \mathbf{b}$.
> - A **Geometric Program (GP)** in standard form has: minimise a posynomial, subject to posynomial $\leq 1$ and monomial $= 1$. A **posynomial** is a sum of monomials $\sum_k c_k x_1^{a_{1k}} \cdots x_n^{a_{nk}}$ with $c_k > 0$. Every LP (with positive variables) is a special case of a GP, but not vice versa.

For completeness, here is the full LP formulation of the original problem (Q4.1–Q4.3):

$$
\begin{array}{ll}
\min_{\mathbf{P}} & \sum_{i=1}^N P_i \\[6pt]
\text{subject to} & G_{ii}P_i - S^{\min}\sum_{k \neq i} G_{ik}P_k \geq S^{\min}\sigma, \quad \forall\, i \\[6pt]
& P^{\min} \leq P_i \leq P^{\max}, \quad \forall\, i
\end{array}
$$

All constraints and the objective are linear in $P$. This is a standard LP solvable by simplex or interior-point methods. The GP formulation is more general (it handles both the original and modified SINR), but recognising the LP structure is useful: LPs are cheaper to solve, have well-understood sensitivity analysis, and the dual provides direct economic interpretation of the SINR constraints (each dual variable $\lambda_i$ is the marginal cost of tightening the SINR floor at receiver $i$).

