# Optimality Simulations

This directory is a self-contained reproducibility capsule for the ridge and
hard-thresholded ridge antithetic CV optimality simulations.

Run commands from this directory so generated results and figures are written
under `optimality/results/` and `optimality/figures/`.

## Workflow

```bash
cd optimality
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiments.py
python scripts/plot_results.py
```

Results are saved as tidy Parquet tables under `results/`. Figures are saved under `figures/`.
The default plot format is PDF. Use `--formats pdf,png` to save multiple formats.
The default error band is mean +/- SD. To save a separate 95% percentile-interval figure:

```bash
python scripts/plot_results.py \
  --figure three_panel \
  --errorbar percentile_95 \
  --formats pdf,png
```

The two-panel MSE plot defaults to a structured bootstrap 95% CI over the saved
response/randomization replications and uses a linear y-axis:

```bash
python scripts/plot_results.py --figure mse_two_panel --formats pdf,png
```

To plot mean +/- SD bands instead:

```bash
python scripts/plot_results.py --figure mse_two_panel --errorbar sd --formats pdf,png
```

To run only one experiment:

```bash
python scripts/run_experiments.py --experiment ridge_alpha
```

## Result Schema

Each experiment writes raw risk-estimate rows with these columns:

- `experiment`
- `model`
- `method`
- `tuning_name`
- `tuning_value`
- `alpha`
- `lambda_value`
- `response_rep`
- `randomization_rep`
- `estimated_risk`
- `actual_risk`

Plotting and summary statistics are computed from these raw rows instead of from hidden in-memory arrays.

## Scripts

- `scripts/run_experiments.py`
- `scripts/plot_results.py`

Use `--help` on any script to see configurable simulation parameters.

## Smoke Test

For a quick check that the environment is working:

```bash
python scripts/run_experiments.py \
  --n-response-reps 1 \
  --n-randomization-reps 1 \
  --n-alpha-values 2 \
  --n-lambda-values 2 \
  --n-threshold-risk-train-reps 2 \
  --results-dir /tmp/acv_smoke_results \
  --quiet

python scripts/plot_results.py \
  --results-dir /tmp/acv_smoke_results \
  --figures-dir /tmp/acv_smoke_figures \
  --formats png \
  --quiet
```
