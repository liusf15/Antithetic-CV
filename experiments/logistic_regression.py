import numpy as np
from scipy.special import expit
import pandas as pd
from scipy.linalg import sqrtm
import argparse
from prediction_algorithms import logistic_reg_from_score
from risk_estimators import antithetic_cv, coupled_bootstrap, cv_split

def estimate_H(X, s):
    beta_hat = g(X, s)
    mu_hat = expit(X @ beta_hat)
    W = np.diag(mu_hat * (1 - mu_hat))
    return X.T @ W @ X

def run_experiment(g, X, alpha, K, rng):
    Y = generate_data(rng)
    s = X.T @ Y
    H_hat = estimate_H(X, s)
    sqrtH = sqrtm(H_hat)
    risk_anti_cv = antithetic_cv(g, (X, s), sqrtH, alpha, K, rng, loss_fn='logistic')
    risk_cb = coupled_bootstrap(g, (X, s), sqrtH, alpha, K, rng, loss_fn='logistic')
    risk_split_cv = cv_split(g, (X, Y), K, rng, loss_fn='logistic')
    return {'antithetic_cv': risk_anti_cv, 'coupled_bootstrap': risk_cb, 'cv_split': risk_split_cv}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    
    # setup
    rng = np.random.default_rng(0)
    n = 100
    d_cts = 4
    d_cat = 2
    beta = np.array([1, -1, 1., -1., .5, -.5, 0.5, -0.5])
    d = 8
    X = rng.normal(size=(n, d_cts))
    for k in range(d_cat):
        X_cat = pd.get_dummies(rng.choice(3, n, p=[0.1, 0.1, 0.8])).to_numpy() * 1.
        X_cat = X_cat[:, :-1]
        X = np.hstack([X, X_cat])
    mu = expit(X @ beta)
    X /= np.linalg.norm(X, axis=0)
    def generate_data(rng):
        return rng.binomial(1, mu, size=(n, ))

    # prediction function
    g = lambda X, s: logistic_reg_from_score(X, s, l2_penalty=0.01)

    rng = np.random.default_rng(args.seed)
    risk_estimates = run_experiment(g, X, args.alpha, args.K, rng)
    print(risk_estimates)
