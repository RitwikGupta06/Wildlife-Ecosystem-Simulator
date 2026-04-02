import numpy as np
from scipy.integrate import odeint


def lotka_volterra(state, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra equations.
    alpha  = prey birth rate
    beta   = predation rate
    delta  = predator efficiency (prey eaten -> predator births)
    gamma  = predator death rate
    """
    prey, pred = state
    prey = max(prey, 0)
    pred = max(pred, 0)
    dprey = alpha * prey - beta * prey * pred
    dpred = delta * prey * pred - gamma * pred
    return [dprey, dpred]


def simulate(prey0, pred0, params, years, intervention="None", intervention_year=None):
    """
    Run the simulation, optionally applying an intervention at a given year.
    Returns prey_hist, pred_hist, time_array (all numpy arrays).
    """
    alpha, beta, delta, gamma = params
    t_all = np.linspace(0, years, years * 10)

    if intervention == "None" or intervention_year is None:
        sol = odeint(lotka_volterra, [prey0, pred0], t_all, args=params)
        prey_hist = np.clip(sol[:, 0], 0, None)
        pred_hist = np.clip(sol[:, 1], 0, None)
        return prey_hist, pred_hist, t_all

    # Split into pre- and post-intervention
    t_pre = t_all[t_all <= intervention_year]
    t_post = t_all[t_all > intervention_year]

    sol_pre = odeint(lotka_volterra, [prey0, pred0], t_pre, args=params)
    prey_end = max(sol_pre[-1, 0], 0)
    pred_end = max(sol_pre[-1, 1], 0)

    # Apply intervention
    prey_mod, pred_mod, alpha_mod = _apply_intervention(
        prey_end, pred_end, alpha, intervention
    )
    params_post = (alpha_mod, beta, delta, gamma)

    if len(t_post) > 0:
        sol_post = odeint(lotka_volterra, [prey_mod, pred_mod],
                          np.concatenate([[t_pre[-1]], t_post]), args=params_post)
        sol_post = sol_post[1:]  # drop duplicate boundary point
        prey_hist = np.clip(np.concatenate([sol_pre[:, 0], sol_post[:, 0]]), 0, None)
        pred_hist = np.clip(np.concatenate([sol_pre[:, 1], sol_post[:, 1]]), 0, None)
    else:
        prey_hist = np.clip(sol_pre[:, 0], 0, None)
        pred_hist = np.clip(sol_pre[:, 1], 0, None)

    return prey_hist, pred_hist, t_all[:len(prey_hist)]


def _apply_intervention(prey, pred, alpha, intervention):
    """Modify populations/parameters based on intervention type."""
    prey_mod, pred_mod, alpha_mod = prey, pred, alpha

    if intervention == "Hunting pressure (halve predators)":
        pred_mod = max(1.0, pred * 0.5)

    elif intervention == "Disease outbreak (reduce prey 60%)":
        prey_mod = max(1.0, prey * 0.4)

    elif intervention == "Habitat loss (reduce birth rate 30%)":
        alpha_mod = max(0.05, alpha * 0.7)

    elif intervention == "Predator reintroduction (+15 predators)":
        pred_mod = pred + 15

    return prey_mod, pred_mod, alpha_mod


def assess_risk_rules(prey_final, pred_final):
    """Simple rule-based fallback risk assessment."""
    if prey_final < 5 or pred_final < 1:
        return "Collapse Imminent"
    if prey_final < 20 or pred_final < 5 or prey_final > 400 or pred_final > 200:
        return "At Risk"
    return "Stable"


def assess_risk_ml(model, prey_hist, pred_hist, params):
    """
    Use trained ML model to assess ecosystem risk.
    Features: current populations, params, trend, variability.
    """
    alpha, beta, delta, gamma = params
    prey_now = prey_hist[-1]
    pred_now = pred_hist[-1]

    # Trend over last 20 steps
    window = min(20, len(prey_hist))
    prey_trend = (prey_hist[-1] - prey_hist[-window]) / (window + 1e-6)
    pred_trend = (pred_hist[-1] - pred_hist[-window]) / (window + 1e-6)

    # Variability (coefficient of variation)
    prey_cv = np.std(prey_hist[-50:]) / (np.mean(prey_hist[-50:]) + 1e-6)
    pred_cv = np.std(pred_hist[-50:]) / (np.mean(pred_hist[-50:]) + 1e-6)

    features = np.array([[
        prey_now, pred_now,
        alpha, beta, delta, gamma,
        prey_trend, pred_trend,
        prey_cv, pred_cv
    ]])

    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    label_map = {0: "Stable", 1: "At Risk", 2: "Collapse Imminent"}
    label = label_map.get(pred_class, "Unknown")

    # Collapse probability = prob of class 2
    collapse_prob = pred_proba[2] if len(pred_proba) > 2 else pred_proba[-1]
    return label, collapse_prob
