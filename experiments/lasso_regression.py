import numpy as np
import argparse
from prediction_algorithms import lasso
from risk_estimators import antithetic_cv, coupled_bootstrap, cv_split, BY, breiman, ye

def run_experiment(g, X, sigma, alpha, K, rng):
    Y = generate_data(rng)
    risk_anti_cv = antithetic_cv(g, (X, Y), sigma, alpha, K, rng)
    risk_cb = coupled_bootstrap(g, (X, Y), sigma, alpha, K, rng)
    risk_split_cv = cv_split(g, (X, Y), K, rng)
    risk_by = BY(g, (X, Y), sigma, alpha, K, rng)
    risk_breiman = breiman(g, (X, Y), sigma, alpha, K, rng)
    risk_ye = ye(g, (X, Y), sigma, alpha, K, rng)
    return {'antithetic_cv': risk_anti_cv, 'coupled_bootstrap': risk_cb, 'cv_split': risk_split_cv, 'BY': risk_by, 'breiman': risk_breiman, 'ye': risk_ye}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    
    # setup
    n = 100
    p = 200
    rng = np.random.default_rng(2025)
    X = rng.normal(size=(n, p))
    s = 5
    signal_idx = rng.choice(p, s, replace=False)
    beta = np.zeros(p)
    beta[signal_idx] = (rng.random(s) - 0.5) * 2
    mu = X @ beta
    snr = 0.4
    sigma = np.sqrt(np.var(mu) / snr)
    def generate_data(rng):
        return mu + sigma * rng.normal(size=(n, ))

    # prediction function
    g = lasso

    rng = np.random.default_rng(args.seed)
    risk_estimates = run_experiment(g, X, sigma, args.alpha, args.K, rng)
    print(risk_estimates)
