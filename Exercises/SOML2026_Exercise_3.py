# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv_soml
#     language: python
#     name: python3
# ---

# %% [markdown] id="DEu7yIxCSCtc"
# # ELEN90088 System Optimisation and Machine Learning, 2026
#
# # Exercise 3
# ## Due date: <u> 23:59, Monday the 8 June, 2026 </u>
#
# ## Submission guideline:
#
# * One submission per group by the due date on LMS.
# * Answer the exercise questions in this Python notebook itself.
# * Export your **executed** notebook file (.ipynb) as a PDF file, on which we give marks and comments. This means that each group should submit two versions of the exercise report (.ipynb file and PDF).
# * You could either submit your photocopied handwritten solutions (making sure they are legible) or typed solutions (e.g. with Latex).
# * **WARNING: DO NOT SUBMIT A COMPRESSED ZIP FILE. FAILURE TO FOLLOW THESE INSTRUCTIONS WILL RESULT IN A 10-POINT PENALTY!**
# * Demonstrators will conduct a brief oral assessment for selected groups in subsequent workshop. Details will be announced on LMS.
# * Regarding the use of LLM and other generative AI tools: refer to information in the introductory slides.
#
#

# %% [markdown] id="7gRM95wmvV0o"
# ## Question 1: A problem with a positive duality gap (Marks: 9 + 6 + 5 = 20 points)
#
# Consider the problem
# \begin{align*}
#     &\text{minimize } e^{-x}\\
#     &\text{subject to } x^2/y\leq 0
# \end{align*}
# over the domain $\{(x,y)|y>0\}$.
#
# * (a) Is this a convex optimization problem? Find the optimal value for the objective $p^*$ and the optimizing values $x^*, y^*$.
# * (b) Derive the dual problem. Find the optimal value for the objective $d^*$ of the dual problem and the optimal dual variable $\lambda^*$. Determine the duality gap $p^*-d^*$.
# * (c) Derive the KKT condition for this problem. Do all $\lambda^*, x^*, y^*$ satisfy the KKT condition?

# %% [markdown]
# ### Solution
#
# **(a) Convexity and the primal optimum.**
#
# The objective $f_0(x,y)=e^{-x}$ is convex (convex in $x$, independent of $y$). The constraint function $f_1(x,y)=x^2/y$ is the quadratic-over-linear function, which is jointly convex on the domain $\{y>0\}$; hence the feasible set $\{f_1\le 0\}$ and the domain are convex. **So this is a convex optimization problem.**
#
# Because $y>0$, the constraint collapses to a single point set:
# $$
# \frac{x^2}{y}\le 0 \iff x^2\le 0 \iff x=0 .
# $$
# The feasible set is therefore the ray $\{(0,y):y>0\}$, on which the objective equals the constant $e^{0}=1$. Hence
# $$
# p^\star = 1, \qquad x^\star = 0, \qquad y^\star>0\ \text{(arbitrary).}
# $$

# %% [markdown]
# **(b) Dual problem, $d^\star$, $\lambda^\star$, and the duality gap.**
#
# For $\lambda\ge 0$ and $y>0$ the Lagrangian is $L(x,y,\lambda)=e^{-x}+\lambda\,x^2/y$, and the dual function is $g(\lambda)=\inf_{x\in\mathbb R,\,y>0}L$.
#
# * **$\lambda\ge 0$:** both terms are $\ge 0$, so $L\ge 0$; taking $x\to+\infty$ along $y=x^3$ gives $e^{-x}\to 0$ and $\lambda x^2/y=\lambda/x\to 0$, so the infimum is $0$ (approached, never attained). Thus $g(\lambda)=0$.
# * **$\lambda<0$:** fix any $x\ne 0$ and let $y\to 0^+$; then $x^2/y\to+\infty$ and $\lambda x^2/y\to-\infty$, so $g(\lambda)=-\infty$.
#
# Maximizing over $\lambda\ge 0$ gives
# $$
# d^\star = 0, \qquad \lambda^\star\ge 0\ \text{(every non-negative $\lambda$ is dual-optimal),}
# $$
# and the **duality gap** is
# $$
# p^\star-d^\star = 1-0 = 1 > 0 .
# $$
# The gap is strictly positive *despite convexity* because **Slater's condition fails**: there is no strictly feasible point ($x^2/y<0$ is impossible for $y>0$), so strong duality is not guaranteed.

# %% [markdown]
# **(c) KKT conditions.**
#
# The KKT system reads
# $$
# \text{(stationarity)}\quad \partial_x L=-e^{-x}+\frac{2\lambda x}{y}=0,\qquad \partial_y L=-\frac{\lambda x^2}{y^2}=0;
# $$
# $$
# \text{(primal feas.)}\ \ \frac{x^2}{y}\le 0\ (\Rightarrow x=0),\ y>0;\qquad \text{(dual feas.)}\ \ \lambda\ge 0;\qquad \text{(compl. slack.)}\ \ \lambda\,\frac{x^2}{y}=0 .
# $$
# At the optimum $x^\star=0,\ y^\star>0$ with any $\lambda^\star\ge 0$, primal/dual feasibility, complementary slackness ($\lambda\cdot 0=0$) and $\partial_y L=-\lambda\cdot 0=0$ all hold — **but the $x$-stationarity fails**:
# $$
# \partial_x L\big|_{x=0}=-e^{0}+\frac{2\lambda\cdot 0}{y}=-1\neq 0 \qquad\text{for every }\lambda\ge 0 .
# $$
# Hence **no triple $(\lambda^\star,x^\star,y^\star)$ satisfies all the KKT conditions** — in particular the optimum does not. This is consistent with part (b): KKT is necessary for optimality only under a constraint qualification (here Slater's condition), which fails, so this convex problem has a positive duality gap and admits no KKT point.

# %%
# Numerical sanity checks for Q1
import numpy as np

# (a) feasibility forces x = 0  ->  p* = e^0 = 1
p_star = np.exp(0.0)
print(f"(a) feasible set = {{(0, y): y > 0}}  ->  p* = {p_star:.6f}")

# (b) g(lambda) = inf L; for lambda >= 0 the infimum is 0, approached along x = t, y = t^3:
for lam in (0.0, 0.5, 2.0):
    t = np.array([1e0, 1e1, 1e2, 1e3])
    L = np.exp(-t) + lam * t**2 / t**3            # = e^{-t} + lam/t  ->  0
    print(f"(b) lambda={lam:>3}:  L(x=t, y=t^3) = {np.array2string(L, precision=2)}  -> inf -> 0")
print(f"(b) d* = 0  ->  duality gap p* - d* = {p_star - 0.0:.6f}")

# (c) x-stationarity residual at the optimum x* = 0 equals -1 for every lambda:
for lam in (0.0, 1.0, 5.0):
    dLdx = -np.exp(-0.0) + 2 * lam * 0.0 / 1.0
    print(f"(c) x*=0, lambda={lam}:  dL/dx = {dLdx:+.1f}  (!= 0  =>  KKT stationarity fails)")

# %% [markdown] id="dtaI1z-DwAFC"
# ## Question 2: KKT condition for non-convex problems (Marks: 7 + 9 + 4 = 20 points)
#
# Consider the problem
# \begin{align*}
#     &\text{minimize }-3x_1^2+x_2^2+2x_3^2+2(x_1+x_2+x_3)\\
#     &\text{subject to } x_1^2+x_2^2+x_3^2=1
# \end{align*}
#
# It can be shown that strong duality holds for this problem.  
# * (a) Is this a convex optimization problem? Derive the KKT condition for this problem.
# * (b) Find all solutions $x, \nu$ that satisfy the KKT conditions. (You need to solve this numerically)
# * (c) Which pair corresponds to the optimum? Give explanation from different angles.

# %% [markdown]
# ### Solution
#
# **(a) Convexity and the KKT conditions.**
#
# Let $f_0(\mathbf x)=-3x_1^2+x_2^2+2x_3^2+2(x_1+x_2+x_3)$ and $h(\mathbf x)=x_1^2+x_2^2+x_3^2-1$. The objective Hessian is $\nabla^2 f_0=\operatorname{diag}(-6,2,4)$, which is **indefinite** (it has the eigenvalue $-6<0$), and the equality set $\|\mathbf x\|_2^2=1$ (a sphere) is **not convex**. **So this is not a convex problem.**
#
# With one equality constraint and multiplier $\nu\in\mathbb R$ (unrestricted in sign, no complementary slackness), $L=f_0+\nu h$ and the KKT conditions are stationarity $\nabla_{\mathbf x}L=\mathbf 0$,
# $$
# -6x_1+2+2\nu x_1=0,\qquad 2x_2+2+2\nu x_2=0,\qquad 4x_3+2+2\nu x_3=0,
# $$
# together with primal feasibility $x_1^2+x_2^2+x_3^2=1$. Solving each stationarity equation for its own coordinate gives
# $$
# x_1=\frac{1}{3-\nu},\qquad x_2=\frac{-1}{1+\nu},\qquad x_3=\frac{-1}{2+\nu},
# $$
# and substituting into the constraint leaves a single equation in $\nu$:
# $$
# \frac{1}{(3-\nu)^2}+\frac{1}{(1+\nu)^2}+\frac{1}{(2+\nu)^2}=1 .
# $$

# %% [markdown]
# **(b) All KKT solutions (solved numerically).**
#
# Clearing denominators turns the $\nu$-equation into a degree-6 polynomial (with poles at $\nu=-2,-1,3$). We compute all its roots, keep the real ones, and recover $\mathbf x(\nu)$ for each — also recording the sign of $\nabla^2_{\mathbf{xx}}L=\operatorname{diag}(2\nu-6,\,2\nu+2,\,2\nu+4)$, which part (c) uses.

# %%
import numpy as np
import sympy as sp

nu = sp.symbols('nu', real=True)
x1, x2, x3 = 1/(3 - nu), -1/(1 + nu), -1/(2 + nu)
numer, _ = sp.fraction(sp.together(x1**2 + x2**2 + x3**2 - 1))
P = sp.Poly(sp.expand(numer), nu)

real_nu = sorted(float(sp.re(r)) for r in sp.nroots(P, n=20) if abs(sp.im(r)) < 1e-9)

def f0(x):
    a, b, c = x
    return -3*a**2 + b**2 + 2*c**2 + 2*(a + b + c)

print(f"degree-{P.degree()} polynomial  ->  {len(real_nu)} real KKT multipliers\n")
header = f"{'nu':>10}{'x1':>10}{'x2':>10}{'x3':>10}{'||x||^2':>10}{'f0':>11}{'d2L>=0':>9}"
print(header); print("-" * len(header))
solutions = []
for v in real_nu:
    x = np.array([1/(3 - v), -1/(1 + v), -1/(2 + v)])
    psd = bool(np.all(np.array([2*v - 6, 2*v + 2, 2*v + 4]) >= 0))   # is d^2L/dx^2 PSD?
    solutions.append((v, x, f0(x), psd))
    print(f"{v:>10.5f}{x[0]:>10.5f}{x[1]:>10.5f}{x[2]:>10.5f}{x @ x:>10.5f}{f0(x):>11.5f}{str(psd):>9}")

v_opt, x_opt, f_opt, _ = min(solutions, key=lambda s: s[2])
print(f"\nOptimum (global min):  nu* = {v_opt:.5f},  p* = f0* = {f_opt:.5f},  x* = {np.round(x_opt, 5)}")

# %% [markdown]
# **(c) Which KKT pair is the optimum?**
#
# Among the four KKT points the optimum is the one with the smallest objective:
# $$
# \nu^\star\approx 4.035,\qquad \mathbf x^\star\approx(-0.966,\,-0.199,\,-0.166),\qquad p^\star\approx -5.365 .
# $$
# Three independent ways to confirm it:
#
# 1. **Direct comparison + existence.** The unit sphere is compact and $f_0$ is continuous, so a global minimum exists and is attained; since $\nabla h=2\mathbf x\neq\mathbf 0$ everywhere on the sphere, LICQ (Linear Independence Constraint Qualification, ensures the uniqueness of Lagrange multipliers and the stability of KKT conditions) holds, so every minimizer must be a KKT point. The minimum is therefore the KKT point of least objective — the $\nu\approx 4.035$ row of the table above.
# 2. **Second-order / global condition for a quadratic on a sphere.** $\nabla^2_{\mathbf{xx}}L=\nabla^2 f_0+2\nu I=\operatorname{diag}(2\nu-6,\,2\nu+2,\,2\nu+4)$. A global minimizer requires this to be $\succeq 0$, i.e. $2\nu-6\ge 0\Rightarrow \nu\ge 3$. **Only $\nu\approx 4.035$ satisfies $\nu\ge 3$** (the `d2L>=0` column), so it is the unique global minimizer; symmetrically $\nu\approx-3.149$ (where $\nu\le-2$ and $\nabla^2_{\mathbf{xx}}L\preceq 0$) is the global **maximizer** ($f_0\approx+4.647$), and the two middle points are saddles on the sphere.
# 3. **Strong duality.** Strong duality holds (given), so $p^\star=\max_\nu g(\nu)$ with $g(\nu)=\inf_{\mathbf x}L(\mathbf x,\nu)$. The inner infimum is finite only on the branch where $\nabla^2_{\mathbf{xx}}L\succeq 0$ (i.e. $\nu\ge 3$) — exactly the branch containing $\nu^\star\approx 4.035$ — and maximizing $g$ there reproduces $p^\star\approx-5.365$, equal to the primal optimum.

# %% [markdown] id="dMhiKP_xcImK"
# ## Question 3: Binary SVM Classification (Marks: 10 + 8 + 2 = 20 Points)
# In this question, the SVM algorithm will be used in a classification task based on a noisy moon data set. This dataset can be directly available from the sklearn library, which is shown as follows.

# %% colab={"base_uri": "https://localhost:8080/", "height": 907} executionInfo={"elapsed": 718, "status": "ok", "timestamp": 1777436698875, "user": {"displayName": "Sugar He", "userId": "09823635978394101361"}, "user_tz": -600} id="wkJCpSqWe2y1" outputId="21981e73-0d40-4bd9-b929-adec267d577c"
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = noisy_moons = datasets.make_moons(n_samples=400, noise=0.2)

X_train, X_test, y_train, y_test =  train_test_split(X, y)

# Visualize the training data set
order_ind = np.argsort(y_train)
Xm1_train = X_train[order_ind[0:150]]
Xm2_train = X_train[order_ind[151:300]]

plt.figure()
plt.scatter(Xm1_train[:,0], Xm1_train[:,1], color='black')
plt.scatter(Xm2_train[:,0], Xm2_train[:,1], color='red')
plt.title("Training Data Set")
plt.show()

# Visualize the test data set
order_ind = np.argsort(y_test)
Xm1_test = X_test[order_ind[0:50]]
Xm2_test = X_test[order_ind[51:100]]

plt.figure()
plt.scatter(Xm1_test[:,0], Xm1_test[:,1], color='black')
plt.scatter(Xm2_test[:,0], Xm2_test[:,1], color='red')
plt.title("Test Data Set")
plt.show()

# %% [markdown] id="_BSjElQ1cPqB"
# **Questions:** SVM classifiers
#
# 1. Try an [SVM classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) with a linear kernel and different C parameters. Plot the test output and boundary<a name="cite_ref-1"></a>[<sup>[1]</sup>](#cite_note-1). Discuss your observations and comment on linear separability of this data. Provide the precision, recall, and F-score metrics<a name="cite_ref-2"></a>[<sup>[2]</sup>](#cite_note-2).
# 2. Next, use an [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) and repeat the first part for different C and gamma parameters. Do you observe an improvement compared to the linear version (both visually and in terms of scores)? Discuss your results.
# 3. Compare DNN results (from Exercise 2 Question 3) with SVM and illustrate your findings.
#
# *Some hints which may be relevant and helpful:*
#
# <a name="cite_note-1"></a> [<sup>[1]</sup>](#cite_ref-1) For ease of visualization, the official [`DecisionBoundaryDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html#sklearn.inspection.DecisionBoundaryDisplay) function is efficient and simple to implement. You can learn how to use it via an [official example](https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py).
#
# <a name="cite_note-2"></a> [<sup>[2]</sup>](#cite_ref-2) See [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics), especially [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
#

# %% [markdown]
# ### Solution
#
# We re-create the moons with a **fixed seed** (the question's intro cell above draws unseeded data, for visualisation only), split it 75/25, and reuse one helper that returns macro precision / recall / F1 and accuracy. All three parts below share this split, so the comparisons are direct.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.inspection import DecisionBoundaryDisplay

SEED_Q3 = 42
X_q3, y_q3 = datasets.make_moons(n_samples=400, noise=0.2, random_state=SEED_Q3)
X_tr, X_te, y_tr, y_te = train_test_split(X_q3, y_q3, test_size=0.25, random_state=SEED_Q3)

def scores(y_true, y_pred):
    """Return (macro precision, macro recall, macro F1, accuracy)."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    m = rep["macro avg"]
    return m["precision"], m["recall"], m["f1-score"], rep["accuracy"]

# %%
# 1. Linear-kernel SVM over a log-spaced sweep of C; boundary + test points + metrics.
LIN_C = [0.01, 0.1, 1.0, 10.0, 100.0]
fig, axes = plt.subplots(1, len(LIN_C), figsize=(4 * len(LIN_C), 4))
print(f"{'C':>8} {'precision':>10} {'recall':>8} {'f1':>7} {'accuracy':>9}")
for ax, C in zip(axes, LIN_C):
    clf = SVC(kernel='linear', C=C).fit(X_tr, y_tr)
    p, r, f1, acc = scores(y_te, clf.predict(X_te))
    print(f"{C:>8.2g} {p:>10.3f} {r:>8.3f} {f1:>7.3f} {acc:>9.3f}")
    DecisionBoundaryDisplay.from_estimator(
        clf, X_q3, ax=ax, cmap='RdBu_r', alpha=0.5,
        response_method='predict', plot_method='contourf', grid_resolution=200)
    ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], c='black', s=22, edgecolors='w', linewidths=0.5)
    ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], c='red',   s=22, edgecolors='w', linewidths=0.5)
    ax.set_title(f"linear, C={C:g}"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

# %% [markdown]
# **1. Linear SVM — observations.**
#
# * **The boundary is a straight line for every $C$** — the defining property of a linear kernel, no matter how hard violations are penalised.
# * **The data is not linearly separable.** The two moons interleave, so no straight line separates them: macro-F1 plateaus at **≈ 0.81** for $C\ge 1$ (rising only from ≈ 0.77 at $C=0.01$ as the soft margin tightens). The residual ~19 % error sits in the curling tips that lie on the wrong side of *any* line.
# * **Effect of $C$:** small $C$ → wide soft margin, mild under-fit; large $C$ → narrow margin. Past $C\approx 1$ the metrics are flat — performance is capped by the linear hypothesis class, not by the regularisation strength.

# %%
# 2. RBF-kernel SVM, sweeping (C, gamma); pick the best by macro-F1 (reused in part 3).
RBF_PARAMS = [(1, 0.1), (1, 1), (1, 10), (10, 1), (100, 1), (10, 10)]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
print(f"{'C':>6} {'gamma':>6} {'precision':>10} {'recall':>8} {'f1':>7} {'accuracy':>9}")
best = (-1.0, None)
for ax, (C, g) in zip(axes.ravel(), RBF_PARAMS):
    clf = SVC(kernel='rbf', C=C, gamma=g).fit(X_tr, y_tr)
    p, r, f1, acc = scores(y_te, clf.predict(X_te))
    print(f"{C:>6.2g} {g:>6.2g} {p:>10.3f} {r:>8.3f} {f1:>7.3f} {acc:>9.3f}")
    if f1 > best[0]:
        best = (f1, (C, g))
    DecisionBoundaryDisplay.from_estimator(
        clf, X_q3, ax=ax, cmap='RdBu_r', alpha=0.5,
        response_method='predict', plot_method='contourf', grid_resolution=200)
    ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], c='black', s=22, edgecolors='w', linewidths=0.5)
    ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], c='red',   s=22, edgecolors='w', linewidths=0.5)
    ax.set_title(f"rbf, C={C:g}, γ={g:g}"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
best_f1, (best_C, best_g) = best
print(f"\nbest RBF by macro-F1: C={best_C}, gamma={best_g}  (F1={best_f1:.3f})")

# %% [markdown]
# **2. RBF SVM — observations.**
#
# * **Curved boundaries that trace the moons.** Macro-F1 jumps from the linear ceiling of ≈ 0.81 to **≈ 0.96** — a clear, real improvement both visually and on every metric. (It is not the near-perfect score one gets on cleaner noise=0.1 data: at noise=0.2 the classes genuinely overlap, so some error is irreducible.)
# * **Effect of $\gamma$:** too small ($\gamma=0.1$) makes the Gaussian bumps so wide that the boundary is essentially linear again (F1 ≈ 0.81); $\gamma\approx 1$ matches the moon scale; very large $\gamma$ would carve islands around individual points (over-fit).
# * **Effect of $C$:** mild here — for $\gamma\approx 1$, any $C\in[1,100]$ lands at ≈ 0.95–0.96. The sweet spot is $\gamma\approx 1$ with a moderate $C$.

# %%
# 3. DNN (the Exercise 2 Q3 baseline) vs. the best RBF SVM, on the same split.
import torch
import torch.nn as nn

class DNN(nn.Module):
    """Small MLP matching the Exercise 2 Q3 baseline: hidden layers [16, 8] (ReLU) + sigmoid head."""
    def __init__(self, dims=(2, 16, 8, 1)):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

torch.manual_seed(0); np.random.seed(0)
Xtr_t = torch.tensor(X_tr, dtype=torch.float32); ytr_t = torch.tensor(y_tr, dtype=torch.float32)
Xte_t = torch.tensor(X_te, dtype=torch.float32)

dnn = DNN(); opt = torch.optim.Adam(dnn.parameters(), lr=0.01); bce = nn.BCELoss()
for _ in range(300):
    opt.zero_grad(); bce(dnn(Xtr_t), ytr_t).backward(); opt.step()
dnn.eval()
with torch.no_grad():
    y_dnn = (dnn(Xte_t).numpy() >= 0.5).astype(int)

svm = SVC(kernel='rbf', C=best_C, gamma=best_g).fit(X_tr, y_tr)
print("DNN [16, 8, 1] + ReLU + Adam(1e-2) + BCE, 300 epochs:")
print(classification_report(y_te, y_dnn, digits=3, zero_division=0))
print(f"RBF SVM (C={best_C}, gamma={best_g}):")
print(classification_report(y_te, svm.predict(X_te), digits=3, zero_division=0))

def plot_boundary_torch(model, X_all, y_all, ax, title):
    """Decision boundary for a torch model via a prediction grid (DecisionBoundaryDisplay expects an sklearn estimator)."""
    g0 = np.linspace(X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5, 300)
    g1 = np.linspace(X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5, 300)
    xx0, xx1 = np.meshgrid(g0, g1)
    grid = np.c_[xx0.ravel(), xx1.ravel()].astype(np.float32)
    with torch.no_grad():
        zz = (model(torch.tensor(grid)).numpy() >= 0.5).astype(int).reshape(xx0.shape)
    ax.contourf(xx0, xx1, zz, cmap='RdBu_r', alpha=0.5)
    ax.scatter(X_all[y_all == 0, 0], X_all[y_all == 0, 1], c='black', s=15)
    ax.scatter(X_all[y_all == 1, 0], X_all[y_all == 1, 1], c='red',   s=15)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_boundary_torch(dnn, X_q3, y_q3, axes[0], "DNN [16, 8, 1] + ReLU")
DecisionBoundaryDisplay.from_estimator(
    svm, X_q3, ax=axes[1], cmap='RdBu_r', alpha=0.5,
    response_method='predict', plot_method='contourf', grid_resolution=200)
axes[1].scatter(X_q3[y_q3 == 0, 0], X_q3[y_q3 == 0, 1], c='black', s=15)
axes[1].scatter(X_q3[y_q3 == 1, 0], X_q3[y_q3 == 1, 1], c='red',   s=15)
axes[1].set_title(f"RBF SVM (C={best_C}, γ={best_g})"); axes[1].set_xticks([]); axes[1].set_yticks([])
plt.tight_layout(); plt.show()

# %% [markdown]
# **3. DNN vs. SVM — observations.**
#
# * **Both reach ≈ 0.96–0.97 macro-F1** at noise=0.2 — essentially tied, with the DNN a hair ahead here (≈ 0.97 vs ≈ 0.96). Both decisively beat the linear SVM (≈ 0.81).
# * **Boundary geometry differs.** The DNN boundary is piecewise-linear (a composition of ReLUs carves space into polygons); the RBF SVM boundary is smooth (a level set of summed Gaussians). Both follow the gap between the moons.
# * **Practical trade-offs.** The SVM has just two hyperparameters ($C,\gamma$), trains in milliseconds with no GPU, and solves a convex objective (global optimum). The DNN is more flexible and scales to high-dimensional / unstructured data, but adds many knobs (architecture, optimiser, learning rate, epochs) and trains via non-convex SGD.
# * **Bottom line.** On a 2-D toy problem the SVM is simpler, faster, and ties on accuracy; the DNN's extra flexibility only pays off on harder data where no fixed kernel captures the structure.

# %% [markdown] id="0_eFMCEMV6E4"
# ## Question 4: Importance weighted SVM (Marks: 15 points)
#
# Consider a binary classification problem. You are given a data set $\mathcal D$ where each data point comes with an _importance weight_. Specifically, each data point consists of a triplet $(\mathbf x_i, y_i, p_i)$ where $p_i\in[0,1]$ indicates the importance of the data. The larger $p_i$ is, more important the pair $(\mathbf x_i, y_i)$ is. We still assume $\mathbf x_i\in\mathbb R^d$ and $y_i\in\{-1,1\}$, and the data is in general non-separable.
#
# Now, try to develop a version of SVM which incorporates the importance weight information. First, derive the primal SVM constrained optimization problem, and then find the corresponding dual problem formulation. How is it different from standard SVM?
#
# (_Hint: there might be more than one way to incorporate the importance weight into SVM. One idea is to scale the penalty of mislabeling the point $\mathbf x_i$ by $p_i$. But you are free to explore other ideas._)

# %% [markdown]
# ### Solution
#
# **Primal problem.**
# Recall the standard soft-margin ($C$-)SVM, which for non-separable data introduces a slack $\xi_i\ge 0$ measuring the margin violation of point $i$ and penalizes the total slack uniformly:
# $$
# \min_{\mathbf w,b,\boldsymbol\xi}\ \tfrac12\|\mathbf w\|_2^2+C\sum_{i=1}^{n}\xi_i\quad\text{s.t.}\quad y_i(\mathbf w^\top\mathbf x_i+b)\ge 1-\xi_i,\ \ \xi_i\ge 0 .
# $$
# To make the classifier care more about high-importance points, scale each slack penalty by its weight $p_i\in[0,1]$ (the "scale the misclassification penalty" idea from the hint):
# $$
# \boxed{\ \min_{\mathbf w,b,\boldsymbol\xi}\ \tfrac12\|\mathbf w\|_2^2+C\sum_{i=1}^{n}p_i\,\xi_i\quad\text{s.t.}\quad y_i(\mathbf w^\top\mathbf x_i+b)\ge 1-\xi_i,\ \ \xi_i\ge 0 .\ }
# $$
# A violation on an important point (large $p_i$) now costs more, so the optimizer fits those points first; a point with $p_i=0$ is effectively ignored.

# %% [markdown]
# **Dual problem.**
# Introduce multipliers $\alpha_i\ge 0$ for the margin constraints and $\mu_i\ge 0$ for $\xi_i\ge 0$:
# $$
# \mathcal L=\tfrac12\|\mathbf w\|^2+C\sum_i p_i\xi_i-\sum_i\alpha_i\big[y_i(\mathbf w^\top\mathbf x_i+b)-1+\xi_i\big]-\sum_i\mu_i\xi_i .
# $$
# Stationarity gives the usual relations plus a modified $\xi$-condition:
# $$
# \frac{\partial\mathcal L}{\partial\mathbf w}=0\Rightarrow \mathbf w=\sum_i\alpha_i y_i\mathbf x_i,\qquad
# \frac{\partial\mathcal L}{\partial b}=0\Rightarrow \sum_i\alpha_i y_i=0,\qquad
# \frac{\partial\mathcal L}{\partial\xi_i}=0\Rightarrow C p_i-\alpha_i-\mu_i=0 .
# $$
# Because $\mu_i\ge 0$, the last identity forces $0\le\alpha_i\le C p_i$. Substituting back to eliminate $\mathbf w,b,\boldsymbol\xi$ yields the dual
# $$
# \boxed{\ \max_{\boldsymbol\alpha}\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j\,y_i y_j\,\mathbf x_i^\top\mathbf x_j\quad\text{s.t.}\quad \sum_i\alpha_i y_i=0,\ \ 0\le\alpha_i\le C p_i .\ }
# $$
# (Kernelizing simply replaces $\mathbf x_i^\top\mathbf x_j$ by $k(\mathbf x_i,\mathbf x_j)$.)

# %% [markdown]
# **How it differs from the standard SVM.**
# The objective is *identical* to the standard dual; the only change is the **per-point box constraint** — the upper bound on each dual variable is $C p_i$ instead of a uniform $C$.
#
# * Important points (large $p_i$) get a wider box, so their $\alpha_i$ can grow large and exert more pull on $\mathbf w=\sum_i\alpha_i y_i\mathbf x_i$ — they are allowed to become "stronger" support vectors.
# * Low-importance points are capped at a small $C p_i$; if $p_i=0$ the point is forced to $\alpha_i=0$ and drops out of the model entirely.
# * Standard SVM is recovered as the special case $p_i\equiv 1$.
#
# *Alternative weightings.* One could instead weight the margin itself (require $y_i(\mathbf w^\top\mathbf x_i+b)\ge p_i-\xi_i$), use a class-dependent $C^{\pm}$, or duplicate/resample points in proportion to $p_i$. The penalty-scaling form above is the cleanest, because it changes only the dual box bounds and leaves the QP structure — hence any standard SVM solver — intact.

# %% [markdown] id="BGT0vnqkPR9R"
# ## Question 5: Backpropagation (Marks: 4 + 4 + 12 + 5 = 25 points)
#
#
# **Background**
#
# In very deep neural networks, gradients can become vanishingly small during backpropagation, making earlier layers difficult to train. Residual Networks (ResNets) mitigate this by introducing "skip connections" (or identity mappings) that bypass one or more layers. Instead of forcing a layer to learn an entire underlying mapping, it only has to learn the *residual* (the difference) from the identity.
#
# Consider a simplified ResNet block used for a regression task. The network processes an input vector $\mathbf{x} \in \mathbb{R}^d$ through a hidden layer with Sigmoid activation, applies a second linear transformation that outputs $f(\mathbf{x})$, adds a skip connection, and maps the output $\mathbf{x}+f(\mathbf{x})$ to a scalar prediction $\hat{y}$.
#
# To handle potential outliers in the dataset, we optimize a robust logarithmic loss combined with $L_2$ (Ridge) regularization on all weights.
#
# The forward pass can be specified as:
# *   **Hidden Layer:** $\mathbf{z}_1 = \mathbf{W}_1\mathbf{x}$
# *   **Activation:** $\mathbf{h}_1 = \sigma(\mathbf{z}_1)$ *(where $\sigma$ is the element-wise logistic sigmoid function)*
# *   **Residual Connection:** $\mathbf{h}_2 = \mathbf{W}_2\mathbf{h}_1 + \mathbf{x}$
# *   **Output:** $\hat{y} = \mathbf{v}^T\mathbf{h}_2$
#
# where $\mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{d \times d}$, and $\mathbf{v} \in \mathbb{R}^d$.
#
# We utilize a regularized loss function.
# For a single data point $(\mathbf{x}, y)$, the loss function is defined as:
# $$L = \log\left(1 + \frac{1}{2}(\hat{y} - y)^2\right) + \frac{\lambda_1}{2} \|\mathbf{W}_1\|_F^2 + \frac{\lambda_2}{2} \|\mathbf{W}_2\|_F^2,$$
# where $\|\cdot\|_F$ is the Frobenius norm.
#
# **Question**
#
# Given the following initialized values for $d=2$:
# *   $\mathbf{x} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$, $y = 1$
# *   $\mathbf{W}_1 = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$, $\mathbf{W}_2 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$
# *   $\mathbf{v} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$
# *   $\lambda_1 = 0.1$, $\lambda_2 = 0.1$
#
#
# 1. Forward Pass Calculation: Compute the exact numerical values for $\mathbf{z}_1$, $\mathbf{h}_1$, $\mathbf{h}_2$, and $\hat{y}$.
# 2. Loss Gradient Formulation: Derive the analytical expression for the derivative of the data-loss term with respect to the prediction $\hat{y}$. Let $\delta = \frac{\partial L}{\partial \hat{y}}$. Calculate the exact numerical value of $\delta$.
# 3. Backpropagation: Derive the analytical gradients of the *total loss* with respect to the learned parameters, and then compute their exact numerical matrices/vectors: $\nabla_{\mathbf{v}} L$, $\nabla_{\mathbf{W}_2} L$, and $\nabla_{\mathbf{W}_1} L$.
# 4. Architectural Analysis: Derive the analytical gradient of the data-loss term with respect to the input vector $\mathbf{x}$ (i.e., $\nabla_{\mathbf{x}} L_{data}$). Using this expression, explicitly explain how the skip connection helps mitigate the vanishing gradient problem for layers preceding this block.
#
# Note: Provide your final numerical answers as exact fractions.
#
#
#

# %% [markdown] id="t7ywIB4KPDZd"
# Demonstrator's image replaced by this simple mermaid flowchart:
#
# ```mermaid
# flowchart TD
#     x["x"]
#     add(("+"))
#     out["x + F(x)"]
#
#     subgraph Fx["F(x)"]
#         L1["layer"]
#         L2["layer"]
#         L1 --> L2
#     end
#
#     x --> L1
#     L2 --> add
#     x -->|identity| add
#     add --> out
# ```

# %% [markdown]
# ### Solution
#
# Every sigmoid here is evaluated at $z=0$, where $\sigma(0)=\tfrac12$ and $\sigma'(0)=\sigma(0)\big(1-\sigma(0)\big)=\tfrac14$, so all quantities below are exact rationals.
#
# **1. Forward pass.**
# $$
# \mathbf z_1=\mathbf W_1\mathbf x=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}\begin{bmatrix}2\\2\end{bmatrix}=\begin{bmatrix}0\\0\end{bmatrix},\qquad
# \mathbf h_1=\sigma(\mathbf z_1)=\begin{bmatrix}\tfrac12\\[2pt]\tfrac12\end{bmatrix},
# $$
# $$
# \mathbf h_2=\mathbf W_2\mathbf h_1+\mathbf x=\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}\tfrac12\\[2pt]\tfrac12\end{bmatrix}+\begin{bmatrix}2\\2\end{bmatrix}=\begin{bmatrix}\tfrac32\\[2pt]\tfrac72\end{bmatrix}+\begin{bmatrix}2\\2\end{bmatrix}=\begin{bmatrix}\tfrac72\\[2pt]\tfrac{11}{2}\end{bmatrix},
# $$
# $$
# \hat y=\mathbf v^\top\mathbf h_2=\begin{bmatrix}1&-1\end{bmatrix}\begin{bmatrix}\tfrac72\\[2pt]\tfrac{11}{2}\end{bmatrix}=\tfrac72-\tfrac{11}{2}=-2 .
# $$

# %% [markdown]
# **2. Loss gradient with respect to the prediction.**
# The data-loss term is $L_{\text{data}}=\log\!\big(1+\tfrac12(\hat y-y)^2\big)$, so by the chain rule
# $$
# \delta=\frac{\partial L}{\partial\hat y}=\frac{\hat y-y}{1+\tfrac12(\hat y-y)^2}.
# $$
# With $\hat y=-2,\ y=1$: $\hat y-y=-3$ and $1+\tfrac12(9)=\tfrac{11}{2}$, hence
# $$
# \delta=\frac{-3}{\,11/2\,}=-\frac{6}{11}.
# $$

# %% [markdown]
# **3. Backpropagation — gradients of the total loss.**
# The Ridge term adds $\lambda_k\mathbf W_k$ to each weight matrix's data-loss gradient; $\mathbf v$ is unregularized. With $\odot$ the elementwise product and $\sigma'(\mathbf z_1)=\big[\tfrac14,\tfrac14\big]^\top$:
#
# $$
# \nabla_{\mathbf v}L=\delta\,\mathbf h_2=-\frac{6}{11}\begin{bmatrix}\tfrac72\\[2pt]\tfrac{11}{2}\end{bmatrix}=\begin{bmatrix}-\tfrac{21}{11}\\[4pt]-3\end{bmatrix}.
# $$
#
# $$
# \nabla_{\mathbf W_2}L=\delta\,\mathbf v\,\mathbf h_1^\top+\lambda_2\mathbf W_2
# =-\frac{6}{11}\begin{bmatrix}\tfrac12&\tfrac12\\[2pt]-\tfrac12&-\tfrac12\end{bmatrix}+\frac{1}{10}\begin{bmatrix}1&2\\3&4\end{bmatrix}
# =\begin{bmatrix}-\tfrac{19}{110}&-\tfrac{4}{55}\\[4pt]\tfrac{63}{110}&\tfrac{37}{55}\end{bmatrix}.
# $$
#
# For $\mathbf W_1$, first backpropagate to the hidden pre-activation, $\boldsymbol\delta_1=\big(\mathbf W_2^\top\mathbf v\big)\odot\sigma'(\mathbf z_1)=\begin{bmatrix}-2\\-2\end{bmatrix}\odot\begin{bmatrix}\tfrac14\\[2pt]\tfrac14\end{bmatrix}=\begin{bmatrix}-\tfrac12\\[2pt]-\tfrac12\end{bmatrix}$, then
# $$
# \nabla_{\mathbf W_1}L=\delta\,\boldsymbol\delta_1\,\mathbf x^\top+\lambda_1\mathbf W_1
# =-\frac{6}{11}\begin{bmatrix}-1&-1\\-1&-1\end{bmatrix}+\frac{1}{10}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
# =\begin{bmatrix}\tfrac{71}{110}&\tfrac{49}{110}\\[4pt]\tfrac{49}{110}&\tfrac{71}{110}\end{bmatrix}.
# $$

# %% [markdown]
# **4. Gradient with respect to the input, and the role of the skip connection.**
# Differentiating $\hat y=\mathbf v^\top\big(\mathbf W_2\,\sigma(\mathbf W_1\mathbf x)+\mathbf x\big)$ through *both* paths,
# $$
# \nabla_{\mathbf x}L_{\text{data}}=\delta\Big(\underbrace{\mathbf W_1^\top\big(\sigma'(\mathbf z_1)\odot(\mathbf W_2^\top\mathbf v)\big)}_{\text{through the block}}+\underbrace{\mathbf v}_{\text{skip}}\Big).
# $$
# Numerically the through-block term vanishes here: $\sigma'(\mathbf z_1)\odot(\mathbf W_2^\top\mathbf v)=\big[-\tfrac12,-\tfrac12\big]^\top$ and $\mathbf W_1^\top\big[-\tfrac12,-\tfrac12\big]^\top=[0,0]^\top$, so
# $$
# \nabla_{\mathbf x}L_{\text{data}}=\delta\,(\mathbf 0+\mathbf v)=\delta\,\mathbf v=-\frac{6}{11}\begin{bmatrix}1\\-1\end{bmatrix}=\begin{bmatrix}-\tfrac{6}{11}\\[4pt]\tfrac{6}{11}\end{bmatrix}.
# $$
# **Why this mitigates vanishing gradients.** The Jacobian of the block output $\mathbf x+f(\mathbf x)$ with respect to $\mathbf x$ is $I+\partial f/\partial\mathbf x$: the identity term routes the upstream gradient *straight through*, so what reaches earlier layers is $\delta(\mathbf v+\text{block term})$ rather than $\delta\cdot(\text{block term})$ alone. This example is the extreme case — the block term is *exactly* $\mathbf 0$ (the sigmoids sit at $\mathbf z_1=\mathbf 0$ and $\mathbf W_1$ cancels the back-propagated signal), yet $\nabla_{\mathbf x}L_{\text{data}}=\delta\mathbf v\neq\mathbf 0$ survives entirely via the skip. Stacking $L$ such blocks multiplies Jacobians of the form $(I+\partial f/\partial\mathbf x)$, which stay near $I$ instead of shrinking toward $\mathbf 0$ the way a product of saturated-sigmoid Jacobians would — exactly how residual connections keep early-layer gradients from vanishing.

# %%
# Q5 — exact-fraction verification by symbolic autodiff
import sympy as sp

A = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"a{i}{j}"))    # W1
B = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"b{i}{j}"))    # W2
vv = sp.Matrix(2, 1, lambda i, j: sp.Symbol(f"v{i}"))      # v
xx = sp.Matrix(2, 1, lambda i, j: sp.Symbol(f"x{i}"))      # x
sig = lambda z: 1 / (1 + sp.exp(-z))

z1 = A * xx
h1 = z1.applyfunc(sig)
h2 = B * h1 + xx
yhat = (vv.T * h2)[0]
y, lam = sp.Integer(1), sp.Rational(1, 10)
L_data = sp.log(1 + sp.Rational(1, 2) * (yhat - y) ** 2)
L = L_data + lam / 2 * sum(e**2 for e in A) + lam / 2 * sum(e**2 for e in B)

val = {A[0, 0]: 1, A[0, 1]: -1, A[1, 0]: -1, A[1, 1]: 1,
       B[0, 0]: 1, B[0, 1]: 2,  B[1, 0]: 3,  B[1, 1]: 4,
       vv[0]: 1, vv[1]: -1, xx[0]: 2, xx[1]: 2}

print("z1, h1, h2, yhat =", list(z1.subs(val)), list(h1.subs(val)), list(h2.subs(val)), yhat.subs(val))
Y = sp.Symbol("Y")
print("delta           =", sp.diff(sp.log(1 + sp.Rational(1, 2) * (Y - y) ** 2), Y).subs(Y, yhat.subs(val)))
print("grad_v          =", list(sp.Matrix([sp.diff(L, vv[i]) for i in range(2)]).subs(val)))
print("grad_W2         =", sp.Matrix(2, 2, lambda i, j: sp.diff(L, B[i, j])).subs(val).tolist())
print("grad_W1         =", sp.Matrix(2, 2, lambda i, j: sp.diff(L, A[i, j])).subs(val).tolist())
print("grad_x (L_data) =", list(sp.Matrix([sp.diff(L_data, xx[i]) for i in range(2)]).subs(val)))
