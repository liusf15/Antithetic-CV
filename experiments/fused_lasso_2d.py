import numpy as np
import argparse
from PIL import Image

from prediction_algorithms import fused_lasso
from risk_estimators import antithetic_cv, coupled_bootstrap, cv_split

def run_experiment(g, sigma, alpha, K, rng):
    Y = generate_data(rng)
    risk_anti_cv = antithetic_cv(g, (None, Y), sigma, alpha, K, rng)
    risk_cb = coupled_bootstrap(g, (None, Y), sigma, alpha, K, rng)
    return {'antithetic_cv': risk_anti_cv, 'coupled_bootstrap': risk_cb}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lbd', type=float, default=0.5)
    args = parser.parse_args()
    
    # setup
    mu = np.asarray(Image.open('experiments/parrot.png')) / 255.
    rows, cols = mu.shape
    n = rows * cols
    sigma = 0.3
    def generate_data(rng):
        return mu + sigma * rng.normal(size=(rows, cols))

    # prediction function
    g = lambda _, Y: fused_lasso(Y, lbd=args.lbd)

    rng = np.random.default_rng(args.seed)
    risk_estimates = run_experiment(g, sigma, args.alpha, args.K, rng)
    print(risk_estimates)
