# predict.py
# Load saved model and predict price for a mileage entered by the user.

import json

MODEL_PATH = "data/model.json"

def load_model(path):
    with open(path, "r") as f:
        return json.load(f)

def predict_price(model, mileage_km):
    # apply the same scaling used in training
    divisor = model["scale"]["km_divisor"]
    mean_thousands = model["scale"].get("mean_km_thousands", None)
    x = mileage_km / divisor  # convert to thousands
    if mean_thousands is not None:
        x = x - mean_thousands
    theta0 = model["theta0"]
    theta1 = model["theta1"]
    # If model was saved after conversion to original scale (train.py does that),
    # theta0/theta1 are in original scale and scale.mean_km_thousands will be present but ignored.
    # This predict function is defensive: if mean_km_thousands is None, it still works.
    return theta0 + theta1 * (x if mean_thousands is None else x)

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    while True:
        s = input("Enter mileage in km (or 'q' to quit): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            km = float(s)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        price = predict_price(model, km)
        print(f"Estimated price: {price:.2f}")
