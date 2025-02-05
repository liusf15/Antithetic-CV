import numpy as np

def antithetic_cv(g, data, sigma, alpha, K, rng):
    """
    Cross-validation based on external antithetic Gaussian noise for estimating the prediction error g:
    E \| g(Y) - Y_new \|_2^2, where Y, Y_new \iid N(\theta, \sigma^2 I_n)

    Parameters
    ----------
    g : callable
        The prediction function
    data : tuple
        Input data of g
    sigma : float
        The noise level
    alpha : float
        Randomization level
    K : int
        Number of folds
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    float
        The estimated prediction error
    """
    X, Y = data
    n = len(Y)
    W = rng.normal(size=(K, n)) * sigma
    W = (W - W.mean(0)) * np.sqrt(K / (K - 1))

    err = 0
    for k in range(K):
        wk = W[k]
        Y_train = Y + np.sqrt(alpha) * wk
        Y_test = Y - wk / np.sqrt(alpha)
        Y_hat = g((X, Y_train))(X)
        err += np.mean((Y_hat - Y_test)**2 - (1 / alpha) * wk**2)
    return err / K

def coupled_bootstrap(g, data, sigma, alpha, B, rng):
    """
    Coupled bootstrap method for risk estimation of g

    Parameters
    ----------
    g : callable
        The prediction function
    data : tuple
        Input data of g
    sigma : float
        The noise level
    alpha : float
        Randomization level
    B : int
        Number of bootstrap samples
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    float
        The estimated prediction error
    """
    X, Y = data
    n = len(Y)
    err = 0
    for _ in range(B):
        wb = sigma * rng.normal(size=(n, ))
        Y_train = Y + np.sqrt(alpha) * wb
        Y_test = Y - wb / np.sqrt(alpha)
        Y_hat = g((X, Y_train))(X)
        err += np.mean((Y_hat - Y_test)**2 - (1 / alpha) * wb**2)
    return err / B

def cv_split(g, data, K, rng):
    """
    standard K-fold cross-validation
    Parameters
    ----------
    g : callable
        The prediction function
    data : tuple
        Input data of g
    K : int
        Number of folds
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    float
        The estimated prediction error
    """
    X, Y = data
    n = len(Y)
    folds = np.array_split(rng.choice(n, n, replace=False), K)
    err = 0
    for k in range(K):
        test_inds = folds[k]
        train_inds = np.hstack(folds[:k] + folds[k+1:])
        X_train = X[train_inds]
        Y_train = Y[train_inds]
        X_test = X[test_inds]
        Y_test = Y[test_inds]

        Y_hat = g((X_train, Y_train))(X_test)
        err += np.mean((Y_hat - Y_test)**2)
    return err / K

def BY(g, data, sigma, alpha, B, rng):
    """
    Breiman-Yu estimator as described in the coupled bootstrap paper

    Parameters
    ----------
    g : callable
        The prediction function
    data : tuple
        Input data of g
    sigma : float
        The noise level
    alpha : float
        Randomization level
    B : int
        Number of bootstrap samples
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    float
        The estimated prediction error
    """
    X, Y = data
    n = len(Y)
    Y_hat = g(data)(X)
    err = np.mean((Y_hat - Y)**2)
    Y_boot = np.zeros((B, n))
    g_boot = np.zeros((B, n))
    for b in range(B):
        Y_b = Y + np.sqrt(alpha) * sigma * rng.normal(size=(n, ))
        g_boot[b] = g((X, Y_b))(X)
        Y_boot[b] = Y_b
    Y_boot -= np.mean(Y_boot, axis=0)
    cov_hat = np.sum(Y_boot * g_boot) / (B-1) / n
    return err + cov_hat * 2 / alpha

def breiman(g, data, sigma, alpha, B, rng):
    """
    Breiman's risk estimator from Breiman (1992)

    Parameters
    ----------
    g : callable
        The prediction function
    data : tuple
        Input data of g
    sigma : float
        The noise level
    alpha : float
        Randomization level
    B : int
        Number of bootstrap samples
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    float
        The estimated prediction error
    """
    X, Y = data
    n = len(Y)
    Y_hat = g(data)(X)
    err = np.mean((Y_hat - Y)**2)
    Y_boot = np.zeros((B, n))
    g_boot = np.zeros((B, n))
    for b in range(B):
        Y_b = Y + np.sqrt(alpha) * sigma * rng.normal(size=(n, ))
        g_boot[b] = g((X, Y_b))(X)
        Y_boot[b] = Y_b
    Y_boot -= Y
    cov_hat = np.sum(Y_boot * g_boot) / B / n
    return err + cov_hat * 2 / alpha

def ye(g, data, sigma, alpha, B, rng):
    """
    Ye's risk estimator from Ye (1998)

    Parameters
    ----------
    g : callable
        The prediction function
    data : tuple
        Input data of g
    sigma : float
        The noise level
    alpha : float
        Randomization level
    B : int
        Number of bootstrap samples
    rng : numpy.random.Generator
        Random number generator

    Returns
    -------
    float
        The estimated prediction error
    """
    X, Y = data
    n = len(Y)
    Y_hat = g(data)(X)
    err = np.mean((Y_hat - Y)**2)
    Y_boot = np.zeros((B, n))
    g_boot = np.zeros((B, n))
    for b in range(B):
        Y_b = Y + np.sqrt(alpha) * sigma * rng.normal(size=(n, ))
        g_boot[b] = g((X, Y_b))(X)
        Y_boot[b] = Y_b
    Y_boot -= np.mean(Y_boot, axis=0)
    _var = np.sum(Y_boot**2, 0) / (B-1)
    _cov = np.sum(Y_boot * g_boot, 0) / (B-1) 
    cov_hat = _cov / _var
    return err + 2 * np.mean(cov_hat)
