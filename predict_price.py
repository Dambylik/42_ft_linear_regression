import pandas as pd #table reader
import numpy as np #math

path = "data/data.csv"
df = pd.read_csv(path)
df["mileage"] = df["km"]
mileage = df["mileage"].to_numpy(dtype=float)
price = df["price"].to_numpy(dtype=float)
m = len(mileage) # number of samples

# linear regression formula: prediction = θ₀ +θ₁ × x 
def estimate_price(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)

def compute_cost(theta0, theta1, mileage, price):
    m = len(mileage)
    estimated_price = estimate_price(theta0, theta1, mileage)
    error = estimated_price - price
    loss = np.mean(error**2)
    return float(loss) 

def gradient_descent(theta0, theta1, mileage, price, learning_rate):
    compute_cost(theta0, theta1, mileage, price)
    gradient0 = np.sum(error) / m #g0 is the gradient (partial derivative) of the loss with respect to θ0 (the intercept).
    gradient1 = np.sum(error * mileage) / m #g1 is the gradient (partial derivative) of the loss with respect to θ1 (the slope).
    theta0 = theta0 - learning_rate * gradient0
    theta1 = theta1 - learning_rate * gradient1
    return theta0, theta1