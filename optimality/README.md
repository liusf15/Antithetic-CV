# Optimality Simulations

Code to reproduce the simulation results of *On the optimality of antithetic randomization for cross-validation*.

## Installation

```bash
cd optimality
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the simulation and make plots

```
python scripts/run_experiments.py
python scripts/plot_results.py
```