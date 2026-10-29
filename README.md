# 🏎️ F1 Race Strategy Simulator / Pit-Stop Optimizer

A Python-based simulator that uses **real historical Formula 1 data** to model
tyre degradation and safety car probability, then runs a **Monte Carlo
simulation** to determine the optimal pit-stop strategy (0-stop, 1-stop, or
2-stop) for a given race — mirroring the strategy calls made by real F1
race engineers on the pit wall.

> **MCA Mini Project**
> Name: Suraj (26MCA20007)
> Program: MCA (General), University Institute of Computing, Chandigarh University
> Section: MCA-1/A | Semester 1

---

## 📌 Problem Statement

Formula 1 race strategy is one of the highest-leverage decisions a team makes
each weekend: choosing when to pit and which tyre compound to switch to can
be worth several seconds — and often positions — over a race. Teams must
weigh:

- How quickly each tyre compound degrades (loses lap-time performance)
- The probability of a safety car or virtual safety car appearing (which
  drastically cuts the time lost during a pit stop)
- The fixed time cost of a pit stop itself

This project answers, using **data instead of intuition**:
**"How many pit stops should a driver make, and on which lap, to finish
fastest at this specific circuit?"**

---

## ✨ Features

- 📊 **Real data, not synthetic data** — pulls actual lap times, tyre
  compounds, and stint data from past F1 races via the FastF1 library
- 🔧 **Tyre degradation modeling** — fits linear and quadratic regression
  curves per tyre compound to predict lap time as tyres age
- 🚨 **Safety car probability model** — analyzes multiple past editions of a
  circuit to estimate how likely a safety car is, and when it typically appears
- 🎲 **Monte Carlo simulation** — runs 1,000+ randomized race scenarios per
  strategy to measure not just average performance but consistency
- 📈 **Rich visualizations** — degradation curves, strategy timelines,
  win-rate charts, and finish-time distributions
- 🖥️ **Interactive dashboard** — a Streamlit app to pick any race, define
  custom strategies, and see the recommended optimal strategy instantly

---

## 🛠️ Tech Stack

| Component        | Technology            |
|-------------------|------------------------|
| Language          | Python 3.10+           |
| Historical data   | [FastF1](https://docs.fastf1.dev/) |
| Data processing   | pandas, numpy           |
| Visualization     | matplotlib, Plotly      |
| Dashboard         | Streamlit               |
| Exploration       | Jupyter Notebook        |

---

## 📁 Project Structure

```
f1-strategy-simulator/
├── data/                     # FastF1 local cache (auto-created)
├── src/
│   ├── data_loader.py        # Load & clean historical race/lap data
│   ├── degradation_model.py  # Fit tyre degradation curves per compound
│   ├── safety_car_model.py   # Estimate safety car probability per circuit
│   ├── strategy_simulator.py # Core Monte Carlo strategy simulation engine
│   └── visualizer.py         # All chart-generation functions
├── notebooks/
│   └── exploration.ipynb     # Step-by-step walkthrough / demo
├── app.py                    # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd f1-strategy-simulator

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Note:** FastF1 needs an internet connection the first time you load a given
race (to download timing data from the F1 API). After that, it's cached
locally in `data/` so future runs are instant and work offline.

---

## ▶️ Usage

### Option A — Interactive Dashboard (recommended for demo/viva)

```bash
streamlit run app.py
```

Then in the browser:
1. Select a season and Grand Prix from the dropdown
2. View the tyre degradation curves and safety car probability for that circuit
3. Define 2-3 pit strategies (number of stops, compounds, pit laps)
4. Click **Run Simulation** to see the Monte Carlo results and recommended strategy

### Option B — Notebook walkthrough

```bash
jupyter notebook notebooks/exploration.ipynb
```

Runs through the full pipeline step-by-step on a single example race, useful
for understanding or explaining the methodology.

### Option C — Script / library use

```python
from src.data_loader import setup_cache, load_race_session, get_lap_data
from src.degradation_model import compare_compounds
from src.safety_car_model import historical_sc_probability
from src.strategy_simulator import run_monte_carlo, compare_strategies

setup_cache()
session = load_race_session(2023, "Bahrain", "R")
laps = get_lap_data(session)
degradation_models = compare_compounds(laps)
sc_data = historical_sc_probability("Bahrain")

# define strategies, then:
results = run_monte_carlo(strategies, degradation_models, race_length=57,
                           sc_probability_data=sc_data, n_simulations=1000)
```

---

## 🧠 Methodology

1. **Data Collection** — Load real lap-by-lap timing, tyre compound, and
   stint data for a chosen historical race using FastF1.
2. **Degradation Modeling** — Fit a regression curve per tyre compound
   relating lap time to tyre age (laps since fitted), after filtering out
   in/out laps and safety car laps that would distort the fit.
3. **Safety Car Modeling** — Analyze several past editions of the same
   circuit to estimate the probability of a safety car appearing and the
   lap(s) it's historically likely to appear at.
4. **Simulation** — For each candidate strategy, simulate the full race
   lap-by-lap using the degradation curves, adding pit-stop time loss at
   each stop (reduced if the stop coincides with a randomly-sampled safety
   car event).
5. **Monte Carlo Analysis** — Repeat the simulation 1,000+ times per
   strategy with randomized safety car timing to measure both the average
   finishing time and the consistency (win rate) of each strategy.
6. **Recommendation** — Present the strategy with the best combination of
   average pace and consistency, with supporting visual evidence.

---

## 📊 Sample Output

- Degradation curve chart comparing Soft/Medium/Hard tyre wear
- Bar chart comparing predicted total race time: 1-stop vs 2-stop vs 3-stop
- Violin/histogram plot showing finish-time spread per strategy across
  simulations
- Win-rate pie chart showing what % of simulated races each strategy wins
- Stint timeline (Gantt-style) chart per strategy

*(Add screenshots here once the dashboard is built.)*

---

## 🚀 Future Scope

- Extend to multi-driver strategy interaction (undercut/overcut modeling)
- Incorporate weather data (rain probability changing optimal strategy)
- Add fuel-load-corrected lap time modeling for more accuracy
- Live-race mode using OpenF1's real-time API during an actual session
- Track-position-aware simulation (accounting for traffic after a pit stop)

---

## 📚 Data Sources & References

- [FastF1 documentation](https://docs.fastf1.dev/)
- [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1) (Ergast successor)
- [OpenF1 API](https://openf1.org/)

---

## 📄 License

This project was built for academic purposes (MCA mini project). Historical
F1 data is provided by FastF1 under its own terms of use; this project does
not redistribute raw F1 timing data.
