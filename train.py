# train.py
# Simple linear regression training using batch gradient descent
# Saves learned parameters to /mnt/data/model.json

import json
import numpy as np
import pandas as pd

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

# --- Training loop ---
theta0 = 0.0
theta1 = 0.0
print("Start training with alpha =", ALPHA, "iters =", ITERS)
print("m =", m, "feature mean (km thousands) =", x_mean)

for i in range(1, ITERS + 1):
    theta0, theta1 = gradient_step(theta0, theta1, x_train, y, ALPHA)
    if i == 1 or i % (ITERS // 10) == 0 or i == ITERS:
        cost = compute_cost(theta0, theta1, x_train, y)
        print(f"iter {i:5d} | theta0' = {theta0:.6f}, theta1' = {theta1:.6f}, cost = {cost:.2f}")

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
