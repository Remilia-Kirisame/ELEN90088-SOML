"""End-to-end DNN workflow on the two-moons dataset.

Each block below is a single step of the training pipeline. No heavy logic
lives in this file — the actual work is in `my_dnn.py` (model + training) and
`supporter.py` (data + plotting).
"""

import supporter as Sapo
from my_dnn import build_model, make_activation, make_optimizer, make_loss, train_dnn, evaluate

# =========================================================================
# BLOCK 0 — Hyperparameters (edit freely)
# =========================================================================
SEED = 0

# Data
N_SAMPLES = 400
NOISE     = 0.2

# Model
LAYERS     = [16, 8, 1]      # last entry = 1 for binary classification
ACTIVATION = "relu"  # "relu", "tanh", or "sigmoid"

# Training
OPTIMIZER     = "adam"
LEARNING_RATE = 0.01
LOSS          = "bce"
EPOCHS        = 300
RECORD_EVERY  = 10


# =========================================================================
# BLOCK 1 — Reproducibility
# =========================================================================
Sapo.set_seed(SEED)


# =========================================================================
# BLOCK 2 — Load dataset and visualise
# =========================================================================
X_tr, X_te, y_tr, y_te = Sapo.load_moons(n_samples=N_SAMPLES, noise=NOISE, seed=SEED)
Sapo.plot_dataset(X_tr, X_te, y_tr, y_te)


# =========================================================================
# BLOCK 3 — Convert numpy arrays to torch tensors
# =========================================================================
Xtr, Xte, ytr, yte = Sapo.to_tensors(X_tr, X_te, y_tr, y_te)


# =========================================================================
# BLOCK 4 — Build the model, optimizer, and loss
# =========================================================================
activation = make_activation(ACTIVATION)
model     = build_model(input_dim=2, layer_list=LAYERS, activation=activation)
optimizer = make_optimizer(model.parameters(), name=OPTIMIZER, lr=LEARNING_RATE)
loss_fn   = make_loss(LOSS)

print(model)


# =========================================================================
# BLOCK 5 — Train
# =========================================================================
history = train_dnn(
    model, Xtr, ytr, Xte, yte,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    record_every=RECORD_EVERY,
)


# =========================================================================
# BLOCK 6 — Evaluate final model
# =========================================================================
tr_loss, tr_acc = evaluate(model, Xtr, ytr, loss_fn)
te_loss, te_acc = evaluate(model, Xte, yte, loss_fn)
print(f"\nFinal  train: loss {tr_loss:.4f}  acc {tr_acc:.3f}")
print(f"Final  test : loss {te_loss:.4f}  acc {te_acc:.3f}")


# =========================================================================
# BLOCK 7 — Plot results (training curves + decision boundary)
# =========================================================================
Sapo.plot_history(history, title=f"DNN {LAYERS}  {ACTIVATION}  {OPTIMIZER} lr={LEARNING_RATE}  {LOSS}")

X_all, y_all = Sapo.stack_all(X_tr, X_te, y_tr, y_te)
Sapo.plot_decision_boundary(model, X_all, y_all, title="Learned decision boundary")
