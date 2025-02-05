import numpy as np
import argparse
from prediction_algorithms import mlp
from risk_estimators import antithetic_cv, coupled_bootstrap, cv_split

def run_experiment(alpha, K, rng):
    Y = generate_data(rng)
    risk_anti_cv = antithetic_cv(g, (X, Y), sigma, alpha, K, rng)
    risk_cb = coupled_bootstrap(g, (X, Y), sigma, alpha, K, rng)
    risk_split_cv = cv_split(g, (X, Y), K, rng)
    return {'antithetic_cv': risk_anti_cv, 'coupled_bootstrap': risk_cb, 'cv_split': risk_split_cv}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    
    # setup
    n = 1000
    d = 10
    sigma = 1.
    rng = np.random.default_rng(2025)
    X = rng.random(size=(n, d))
    # Friedman function
    f = lambda x: 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4]
    mu = f(X)
    def generate_data(rng):
        return mu + sigma * rng.normal(size=(n, ))

    # prediction function
    g = mlp

    rng = np.random.default_rng(args.seed)
    risk_estimates = run_experiment(args.alpha, args.K, rng)
    print(risk_estimates)
