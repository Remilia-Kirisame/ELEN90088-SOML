---
tags:
  - ELEN90088
  - SVM
  - classification
  - metrics
  - exercise-2
created: 2026-05-05
---

## SVM Fundamentals (Q4 Context)

Binary SVM classification on the `sklearn.datasets.make_moons` (noise=0.1) dataset.

### What is an SVM?

Support Vector Machine finds the separating hyperplane that **maximises the margin** — the distance from the boundary to the nearest points of each class. Wider margin → better generalisation.

> **"support"** refers to the specific, critical data points from the training set—known as *support vectors*—that are closest to the decision boundary (hyperplane). It's not a verb.

### Key Concepts

- **Decision boundary**: $w \cdot x + b = 0$. Sign of LHS gives predicted class.
- **Support vectors**: Training points lying exactly on the margin edge. Only these determine the boundary; all other points are irrelevant.
- **$C$ (regularisation)**: Trade-off between margin width and misclassification tolerance.
  - Small $C$ → wide margin, tolerates violations → underfitting risk
  - Large $C$ → narrow margin, penalises violations hard → overfitting risk
- **Linear kernel**: Only draws straight lines. The moons are **not linearly separable** — structural limitation, not fixable by tuning $C$.
- **RBF kernel** ($k(x, x') = \exp(-\gamma \|x - x'\|^2)$): Implicitly maps data to high-dimensional space where it becomes separable → curved boundaries.
  - $\gamma$ controls Gaussian bump width: small → smooth; large → wiggly, overfits.
- **SVM vs DNN**: Both near-perfect on this 2D problem. SVM: 2 hyperparams, convex (global optimum), fast, poor scaling to high-dim. DNN: many knobs, non-convex, scales to images/text/audio.

### Exercise Flow

1. **Q4.1 (Linear)**: Demonstrates linear non-separability; $C$ sweep shows limited effect.
2. **Q4.2 (RBF)**: Kernel trick solves the problem; $C$/$\gamma$ sweeps show regularisation and kernel scale effects.
3. **Q4.3 (Comparison)**: SVM and DNN achieve similar performance with different trade-offs.

---

## Q4.1 — Linear Kernel

### Q: What are precision, recall, F1, and test accuracy?

All metrics are built from the **confusion matrix** — four counts on the test set:

> [!caution]
>
> "Positive" and "Negative" only asserts on "belongs to or not" for ONE class. For binary SVM, negative on "belongs to class 0" and positive on "belongs to class 1" are effectively the same but different in report. That's why we have two classes of report and use `macro avg` later.

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | TP (True Positive)  | FN (False Negative) / Miss |
| **Actual Negative** | FP (False Positive) / False Alarm | TN (True Negative) |

#### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Fraction of all predictions that are correct. Simple, but misleading when classes are imbalanced (e.g., 95% class A → a model that always says "A" gets 95% accuracy while being useless).

#### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Of all points the model **called positive**, what fraction actually are positive? High precision → *low false alarm* rate. When the model says "this is class 1", you can trust it.

#### Recall (a.k.a. Sensitivity, True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Of all points that **truly are positive**, what fraction did the model find? High recall → the model catches most positives, doesn't miss them (*low miss* rate).

#### F1-score

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\cdot TP}{2\cdot TP + FP + FN}
$$

The **harmonic mean** of precision and recall. Why harmonic mean and not arithmetic? Because it penalises extreme imbalance. If precision = 1.0 and recall = 0.0, the arithmetic mean is 0.5 (misleading) but $F_1 = 0$ (correct — the model is terrible). $F_1 \in [0, 1]$, higher is better.

#### The trade-off

Precision and recall pull in opposite directions. A conservative model that only calls "positive" when very sure → high precision, low recall. A liberal model that calls "positive" often → high recall, low precision. $F_1$ balances both.

---

### Q: Where does `macro avg` come from?

In **binary** classification, each metric (precision, recall, F1) is reported **per class**:

```
              precision    recall  f1-score   support
class 0          0.862      0.880     0.871        50
class 1          0.878      0.860     0.869        50
```

The question is: how do we combine these into one number? There are two common strategies:

#### Macro average (`macro avg`)

$$
\text{Macro-Precision} = \frac{P_{\text{class 0}} + P_{\text{class 1}}}{2}
$$

Compute the metric **independently for each class**, then take the arithmetic mean. Every class is weighted **equally**, regardless of how many samples it has. This is what the exercise uses because:

- The moons dataset is balanced (200/200 each class), so macro ≈ weighted anyway.
- It treats both classes symmetrically — a model that nails class 0 but fails class 1 gets a poor macro average, which is exactly what you want to see.

#### Weighted average (`weighted avg`)

$$
\text{Weighted-Precision} = \frac{N_0 \cdot P_0 + N_1 \cdot P_1}{N_0 + N_1}
$$

Same idea, but each class's metric is weighted by its **support** (number of true samples). This matters for imbalanced datasets — a rare class with 5 samples won't dominate the average.

#### Why the code uses `macro avg`

```python
report = classification_report(y_te_q4, y_pred, output_dict=True, zero_division=0)
macro = report['macro avg']
```

- `classification_report` with `output_dict=True` returns a nested dict with keys: `'0'`, `'1'`, `'accuracy'`, `'macro avg'`, `'weighted avg'`.
- `report['macro avg']` is a dict like `{'precision': 0.870, 'recall': 0.870, 'f1-score': 0.870, 'support': 100}`.
- For a **balanced** dataset, `macro avg` = `weighted avg`, so either works. The exercise picks `macro avg` because it's the conceptually cleaner choice: it directly answers "how well does the model perform on an average class?"

