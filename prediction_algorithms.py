import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize
from scipy.special import expit

def mlp(X, Y):
    """
    Multi-layer perceptron regressor

    Parameters
    ----------
    Z : tuple
        Tuple of X and Y

    Returns
    -------
    function
        A function that takes in X and returns the predicted values
    """
    model = MLPRegressor(hidden_layer_sizes=(64, 64), batch_size=64, alpha=0.01, max_iter=2000, learning_rate_init=0.001, random_state=42)
    model.fit(X, Y)
    return model.predict

def isotonic(X, Y):
    """
    Isotonic regression

    Parameters
    ----------
    Z : tuple
        Tuple of X and Y

    Returns
    -------
    function
        A function that takes in X and returns the predicted values
    """
    f = IsotonicRegression(out_of_bounds='clip')
    f.fit(X, Y)
    return f.predict

def lasso(X, Y):
    """
    Perform Lasso regression using cross-validation to select the best model.

    Parameters
    ----------
    Z (tuple): A tuple containing two elements:
        - X (array-like): The input data.
        - Y (array-like): The target values.
    
    Returns
    -------
    function
        A function that takes in X and returns the predicted values
    """
    f = LassoCV(fit_intercept=False, alphas=np.logspace(-2, 0, 10), tol=1e-3, n_jobs=-1, random_state=0)
    f.fit(X, Y)
    return f.predict

def logistic_reg_from_score(X, s, l2_penalty=1.):
    """
    Perform logistic regression using the score vector.
    
    Parameters
    ----------
    X : array-like
        design matrix
    s : array-like
        score vector, X'Y
    l2_penalty : float, optional

    Returns
    -------
    array-like
        regression coefficients beta
    """
    def nll(beta, X, score, l2_penalty):
        return np.sum(np.log(1 + np.exp(X @ beta))) - np.dot(score, beta) + (l2_penalty / 2) * np.sum(beta**2) 

    def nll_grad(beta, X, score, l2_penalty):
        return X.T @ expit(X @ beta) - score + l2_penalty * beta

    d = X.shape[1]
    res = minimize(nll, np.zeros(d), jac=nll_grad, args=(X, s, l2_penalty), method='BFGS')
    return res.x

