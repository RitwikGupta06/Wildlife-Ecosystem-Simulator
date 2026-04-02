"""
train_model.py
Generates synthetic simulation data and trains a Random Forest classifier
to predict ecosystem risk (Stable / At Risk / Collapse Imminent).
"""

import numpy as np
import joblib
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def lotka_volterra(state, t, alpha, beta, delta, gamma):
    prey, pred = state
    prey = max(prey, 0)
    pred = max(pred, 0)
    dprey = alpha * prey - beta * prey * pred
    dpred = delta * prey * pred - gamma * pred
    return [dprey, dpred]


def label_run(prey_hist, pred_hist):
    """Label a simulation run based on final state."""
    prey_final = prey_hist[-1]
    pred_final = pred_hist[-1]
    if prey_final < 3 or pred_final < 0.5:
        return 2  # Collapse Imminent
    if prey_final < 15 or pred_final < 3 or prey_final > 500 or pred_final > 250:
        return 1  # At Risk
    return 0  # Stable


def extract_features(prey_hist, pred_hist, params):
    """Extract features from a simulation run for ML training."""
    alpha, beta, delta, gamma = params
    prey_now = prey_hist[-1]
    pred_now = pred_hist[-1]

    window = min(20, len(prey_hist))
    prey_trend = (prey_hist[-1] - prey_hist[-window]) / (window + 1e-6)
    pred_trend = (pred_hist[-1] - pred_hist[-window]) / (window + 1e-6)

    prey_cv = np.std(prey_hist[-50:]) / (np.mean(prey_hist[-50:]) + 1e-6)
    pred_cv = np.std(pred_hist[-50:]) / (np.mean(pred_hist[-50:]) + 1e-6)

    return [prey_now, pred_now, alpha, beta, delta, gamma,
            prey_trend, pred_trend, prey_cv, pred_cv]


def generate_training_data(n_samples=3000):
    """Run many simulations with random parameters to generate labeled data."""
    np.random.seed(42)
    X, y = [], []

    for _ in range(n_samples):
        # Random parameters
        alpha = np.random.uniform(0.1, 1.5)
        beta  = np.random.uniform(0.005, 0.12)
        delta = np.random.uniform(0.002, 0.06)
        gamma = np.random.uniform(0.05, 1.2)
        prey0 = np.random.uniform(5, 300)
        pred0 = np.random.uniform(1, 80)

        params = (alpha, beta, delta, gamma)
        t = np.linspace(0, 80, 800)

        try:
            sol = odeint(lotka_volterra, [prey0, pred0], t, args=params)
            prey_hist = np.clip(sol[:, 0], 0, None)
            pred_hist = np.clip(sol[:, 1], 0, None)

            # Sample from a mid-run snapshot (not always the end)
            snapshot = np.random.randint(len(t) // 3, len(t))
            features = extract_features(prey_hist[:snapshot], pred_hist[:snapshot], params)
            label = label_run(prey_hist, pred_hist)

            X.append(features)
            y.append(label)
        except Exception:
            continue

    return np.array(X), np.array(y)


def train_and_save_model(path="risk_model.pkl"):
    """Generate data, train model, save to disk."""
    print("Generating training data...")
    X, y = generate_training_data(n_samples=3000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print("\nModel evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred,
          target_names=["Stable", "At Risk", "Collapse Imminent"]))

    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return model

if __name__ == "__main__":
    train_and_save_model()