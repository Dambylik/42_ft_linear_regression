import pandas as pd #table reader
import numpy as np #math

# Load data
path = "data.csv"
df = pd.read_csv(path) #turn into data frame (table)
df["km_thousands"] = df["km"] / 1000.0
x = df["km_thousands"].to_numpy(dtype=float)
y = df["price"].to_numpy(dtype=float)
m = len(x) #number of samples

# linear regression formula: prediction = θ₁ × x + θ₀
def predict(theta0, theta1, x_array):
    return theta1 * x_array + theta0

def compute_cost(theta0, theta1, x_array, y_array):
    m = len(x_array)
    predicted_price = predict(theta0, theta1, x_array)
    errors = predicted_price - y_array #Compute the errors
    return float(np.mean(errors**2))  # mean squared error (no 1/2 factor)

# Compute costs
theta_sets = [
    (0.0, 0.0),
    (8500.0, -50.0)
]

results = []
for (t0, t1) in theta_sets:
    cost = compute_cost(t0, t1, x, y)
    predicted_price = predict(t0, t1, x)
    results.append((t0, t1, cost, predicted_price[:5]))  # show first 5 predictions

# Display results
for t0, t1, cost, sample_predicted_price in results:
    print(f"theta0 = {t0}, theta1 = {t1}")
    print("  first 5 predictions:", np.round(sample_predicted_price, 2))
    print("  cost (MSE):", round(cost, 2))
    print()
# We'll compute the cost (mean squared error) for two example parameter pairs:
# 1) theta0 = 0, theta1 = 0  (the starting point)
# 2) theta0 = 8500, theta1 = -50 (corresponds to 8500 - 0.05*km when km in km; since we scaled km to thousands,
#    the slope -0.05 per km becomes -50 per thousand-km)
#
# This will show how cost changes and help you see why we need to learn theta values.