# train.py
# Simple linear regression training using batch gradient descent
# Saves learned parameters to /mnt/data/model.json

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration (you can change these) ---
CSV_PATH = "data/data.csv"
MODEL_PATH = "data/model.json"
KM_DIVISOR = 1000.0      # we convert km -> thousands of km
ALPHA = 0.0001           # learning rate (start small)
ITERS = 5000             # number of gradient descent iterations
USE_MEAN_NORMALIZE = True  # whether to subtract feature mean (helps training)

# --- Load data ---
df = pd.read_csv(CSV_PATH)
df["km_thousands"] = df["km"] / KM_DIVISOR
x = df["km_thousands"].to_numpy(dtype=float)
y = df["price"].to_numpy(dtype=float)
m = len(x)

# --- Optional mean normalization (center feature) ---
if USE_MEAN_NORMALIZE:
    x_mean = x.mean()
    x_train = x - x_mean
else:
    x_mean = 0.0
    x_train = x

# --- Model helpers ---
def predict(theta0, theta1, x_array):
    return theta0 + theta1 * x_array

def compute_cost(theta0, theta1, x_array, y_array):
    preds = predict(theta0, theta1, x_array)
    errors = preds - y_array
    return float(np.mean(errors**2))  # MSE

def gradient_step(theta0, theta1, x_array, y_array, learning_rate):
    m_local = len(x_array)
    preds = predict(theta0, theta1, x_array)
    errors = preds - y_array
    g0 = np.sum(errors) / m_local
    g1 = np.sum(errors * x_array) / m_local
    tmp0 = learning_rate * g0
    tmp1 = learning_rate * g1
    new_theta0 = theta0 - tmp0
    new_theta1 = theta1 - tmp1
    return new_theta0, new_theta1


def sgd_update(theta0, theta1, x_i, y_i, learning_rate):
    """Perform a single-sample (stochastic) gradient descent update.

    This updates parameters using the gradient computed from one sample (x_i, y_i).
    """
    pred = theta0 + theta1 * x_i
    error = pred - y_i
    g0 = error            # gradient w.r.t theta0 (no averaging)
    g1 = error * x_i      # gradient w.r.t theta1
    new_theta0 = theta0 - learning_rate * g0
    new_theta1 = theta1 - learning_rate * g1
    return new_theta0, new_theta1

# --- Training loop ---
theta0 = 0.0
theta1 = 0.0
print("Start training with alpha =", ALPHA, "iters =", ITERS)
print("m =", m, "feature mean (km thousands) =", x_mean)

# -- Training loop (stochastic / per-sample gradient descent)
for epoch in range(1, ITERS + 1):
    # shuffle data each epoch for stochastic updates
    perm = np.random.permutation(m)
    for idx in perm:
        theta0, theta1 = sgd_update(theta0, theta1, x_train[idx], y[idx], ALPHA)

    if epoch == 1 or epoch % (ITERS // 10) == 0 or epoch == ITERS:
        cost = compute_cost(theta0, theta1, x_train, y)
        print(f"epoch {epoch:5d} | theta0' = {theta0:.6f}, theta1' = {theta1:.6f}, cost = {cost:.2f}")

# If we trained on normalized x (x - mean), we convert parameters back to original scale:
# If x' = x - mean, model y = theta0' + theta1' * x' = (theta0' + theta1' * (-mean)) + theta1' * x
# So original theta0 = theta0' - theta1' * mean  (this is the conversion we used when normalizing earlier)
if USE_MEAN_NORMALIZE:
    theta1_orig = theta1
    theta0_orig = theta0 - theta1 * x_mean
else:
    theta0_orig = theta0
    theta1_orig = theta1

# Save model and scaling info
model = {
    "theta0": float(theta0_orig),
    "theta1": float(theta1_orig),
    "scale": {"km_divisor": KM_DIVISOR, "mean_km_thousands": float(x_mean) if USE_MEAN_NORMALIZE else None}
}
with open(MODEL_PATH, "w") as f:
    json.dump(model, f)

print("\nTraining finished.")
print("Saved model to", MODEL_PATH)
print(f"Final theta0 = {theta0_orig:.6f}, theta1 = {theta1_orig:.6f}")

# Plot data and fitted line (km in thousands)
try:
    plt.figure()
    plt.scatter(x, y, color='blue', label='data')
    x_vals = np.linspace(x.min(), x.max(), 200)
    y_vals = theta0_orig + theta1_orig * x_vals
    plt.plot(x_vals, y_vals, color='red', label='fit')
    plt.xlabel('km (thousands)')
    plt.ylabel('price')
    plt.legend()
    plot_path = "data/fit.png"
    plt.savefig(plot_path)
    # Try to show interactively; if environment doesn't support it, ignore the error
    try:
        plt.show()
    except Exception:
        pass
    print("Saved plot to", plot_path)
except Exception as e:
    print("Could not generate plot:", e)
