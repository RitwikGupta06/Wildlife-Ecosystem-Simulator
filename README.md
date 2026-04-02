# 🌿 Wildlife Ecosystem Scenario Explorer

A predator-prey simulation tool with AI-powered risk assessment.  
Built with Lotka-Volterra equations, Ranga Kutta 4 numerical integration, and a Random Forest classifier.

---

## Setup

**1. Clone / download this project folder**

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Project Structure

```
wildlife_simulator/
├── app.py              # Streamlit UI — main entry point
├── simulator.py        # Simulation engine (Lotka-Volterra + RK4 + interventions)
├── train_model.py      # ML model training (generates data + trains classifier)
├── requirements.txt    # Python dependencies
└── README.md
```

The first time you run the app, it will automatically train the AI risk model
and save it as `risk_model.pkl`. This takes about 10–20 seconds.
To retrain from scratch, delete `risk_model.pkl` and rerun.

---

## How It Works

### Simulation Engine (`simulator.py`)
Uses the classic **Lotka-Volterra** differential equations:

```
dPrey/dt  = α·Prey - β·Prey·Predator
dPred/dt  = δ·Prey·Predator - γ·Predator
```

Solved numerically using `scipy.integrate.odeint`.  
Interventions are applied by splitting the simulation at the chosen year
and modifying populations or parameters before continuing.

### AI Risk Model (`train_model.py`)
1. Runs **3,000 random simulations** with varied parameters
2. Labels each run: `Stable (0)`, `At Risk (1)`, `Collapse Imminent (2)`
3. Trains a **Random Forest classifier** on 10 extracted features:
   - Current prey & predator populations
   - All 4 ecosystem parameters (α, β, δ, γ)
   - Prey & predator population trends (slope over last 20 steps)
   - Prey & predator variability (coefficient of variation)

### Interventions
| Intervention | Effect |
|---|---|
| Hunting pressure | Predator population halved |
| Disease outbreak | Prey population reduced by 60% |
| Habitat loss | Prey birth rate (α) permanently reduced 30% |
| Predator reintroduction | +15 predators added |

---

## Academic Notes

- The phase portrait shows prey vs predator directly — a closed loop means a stable cycle
- The AI risk model can be improved by adding more features or using a neural network
- Real-world dataset to compare against: **Canadian Lynx-Hare** (Hudson Bay Company records)
- Possible extension: multi-species food web (add vegetation, scavengers)


source ~/venv/bin/activate
cd ~/Desktop/"AI PR"
streamlit run app.py