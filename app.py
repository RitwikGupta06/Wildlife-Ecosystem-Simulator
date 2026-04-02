import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import pandas as pd
import joblib
import os

from simulator import simulate, assess_risk_ml, assess_risk_rules
from train_model import train_and_save_model

st.set_page_config(page_title="Wildlife Ecosystem Explorer", page_icon="🐺", layout="wide")

st.title("Wildlife Ecosystem Scenario Explorer")
st.markdown("Simulate predator-prey dynamics and trigger real-world interventions to see how ecosystems respond.")

# --- Train model if not already saved ---
MODEL_PATH = "risk_model.pkl"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training AI risk model for the first time..."):
        train_and_save_model(MODEL_PATH)

# --- Sidebar: Parameters ---
st.sidebar.header("Ecosystem Parameters")
alpha = st.sidebar.slider("Prey birth rate (α)", 0.1, 1.5, 0.6, 0.05,
    help="How fast prey reproduce naturally")
beta = st.sidebar.slider("Predation rate (β)", 0.01, 0.10, 0.03, 0.005,
    help="How often predators catch prey")
delta = st.sidebar.slider("Predator efficiency (δ)", 0.005, 0.05, 0.015, 0.002,
    help="How well predators convert food into offspring")
gamma = st.sidebar.slider("Predator death rate (γ)", 0.1, 1.0, 0.4, 0.05,
    help="How fast predators die naturally")

prey0 = st.sidebar.number_input("Initial prey population", 10, 500, 100, 10)
pred0 = st.sidebar.number_input("Initial predator population", 1, 100, 20, 1)
years = st.sidebar.slider("Simulation length (years)", 20, 200, 100, 10)

# --- Interventions ---
st.sidebar.header("Interventions")
st.sidebar.markdown("Apply a disturbance at a chosen year:")

intervention = st.sidebar.selectbox("Intervention type", [
    "None",
    "Hunting pressure (halve predators)",
    "Disease outbreak (reduce prey 60%)",
    "Habitat loss (reduce birth rate 30%)",
    "Predator reintroduction (+15 predators)"
])
intervention_year = st.sidebar.slider("Apply at year", 1, years - 1, years // 3)

# --- Run simulation ---
params = (alpha, beta, delta, gamma)
prey_hist, pred_hist, t = simulate(prey0, pred0, params, years, intervention, intervention_year)

# --- AI Risk Assessment ---
model = joblib.load(MODEL_PATH)
risk_label, risk_prob = assess_risk_ml(model, prey_hist, pred_hist, params)
rule_risk = assess_risk_rules(prey_hist[-1], pred_hist[-1])

# --- Layout ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final prey population", f"{prey_hist[-1]:.0f}")
col2.metric("Final predator population", f"{pred_hist[-1]:.0f}")
col3.metric("Peak prey", f"{max(prey_hist):.0f}")
col4.metric("Peak predator", f"{max(pred_hist):.0f}")

# Risk display
risk_colors = {"Stable": "🟢", "At Risk": "🟡", "Collapse Imminent": "🔴"}
st.markdown(f"### AI Risk Assessment: {risk_colors.get(risk_label, '⚪')} **{risk_label}**")
st.progress(float(risk_prob), text=f"Collapse probability: {risk_prob:.1%}")

# --- Population Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=prey_hist, name="Prey", line=dict(color="#185FA5", width=2), fill="tozeroy", fillcolor="rgba(24,95,165,0.08)"))
fig.add_trace(go.Scatter(x=t, y=pred_hist, name="Predator", line=dict(color="#993C1D", width=2), fill="tozeroy", fillcolor="rgba(153,60,29,0.08)"))

if intervention != "None":
    fig.add_vline(x=intervention_year, line_dash="dash", line_color="gray",
                  annotation_text=f"Intervention: year {intervention_year}", annotation_position="top right")

fig.update_layout(
    title="Population over time",
    xaxis_title="Year",
    yaxis_title="Population",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=380,
    margin=dict(t=60, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# --- Phase Portrait ---
col_phase, col_info = st.columns([1, 1])
with col_phase:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=prey_hist, y=pred_hist,
        mode="lines",
        line=dict(color="#1D9E75", width=1.5),
        name="Phase trajectory"
    ))
    fig2.add_trace(go.Scatter(
        x=[prey_hist[0]], y=[pred_hist[0]],
        mode="markers", marker=dict(size=10, color="#185FA5"),
        name="Start"
    ))
    fig2.add_trace(go.Scatter(
        x=[prey_hist[-1]], y=[pred_hist[-1]],
        mode="markers", marker=dict(size=10, color="#993C1D"),
        name="End"
    ))
    fig2.update_layout(
        title="Phase portrait (prey vs predator)",
        xaxis_title="Prey population",
        yaxis_title="Predator population",
        height=340,
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_info:
    st.markdown("#### How to read this")
    st.markdown("""
- **Population chart**: shows how prey (blue) and predators (red) change over time. Classic oscillating cycles are healthy.
- **Phase portrait**: plots prey vs predator directly. A closed loop = stable cycle. A spiral inward = collapse risk.
- **AI risk model**: trained on thousands of simulated runs to predict collapse probability from current population + parameters.

**Try these scenarios:**
- Set predator death rate high → predators go extinct, prey explode
- Apply "disease outbreak" early → cascade collapse
- Try "predator reintroduction" after hunting pressure
    """)

# --- Data table ---
with st.expander("View raw simulation data"):
    df = pd.DataFrame({"Year": t, "Prey": np.round(prey_hist, 2), "Predator": np.round(pred_hist, 2)})
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "simulation_data.csv", "text/csv")
