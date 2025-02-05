import numpy as np

def antithetic_cv(g, data, sigma, alpha, K, rng, loss_fn='quadratic'):
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
    W = rng.normal(size=(K, n))
    W = (W - W.mean(0)) * np.sqrt(K / (K - 1))
    if np.isscalar(sigma):
        sigma_type = 'scalar'
    elif sigma.ndim == 2:
        sigma_type = 'matrix'
    else:
        raise ValueError('sigma must be a scalar or a matrix')

    err = 0
    for k in range(K):
        if sigma_type == 'scalar':
            wk = sigma * W[k]
        else:
            wk = sigma @ W[k]
        Y_train = Y + np.sqrt(alpha) * wk
        Y_test = Y - wk / np.sqrt(alpha)
        if loss_fn == 'quadratic':
            Y_hat = g(X, Y_train)(X)
            err += np.mean((Y_hat - Y_test)**2 - (1 / alpha) * wk**2)
        elif loss_fn == 'logistic':
            beta_hat = g(X, Y_train)
            err += (np.sum(np.log(1 + np.exp(X @ beta_hat)) ) - np.sum(beta_hat * Y_test)) / X.shape[0]
        else:
            raise NotImplementedError('loss_fn {} not implemented'.format(loss_fn))
    return err / K

def coupled_bootstrap(g, data, sigma, alpha, B, rng, loss_fn='quadratic'):
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
    if np.isscalar(sigma):
        sigma_type = 'scalar'
    elif sigma.ndim == 2:
        sigma_type = 'matrix'
    else:
        raise ValueError('sigma must be a scalar or a matrix')
    
    err = 0
    for _ in range(B):
        if sigma_type == 'scalar':
            wb = sigma * rng.normal(size=(n, ))
        else:
            wb = sigma @ rng.normal(size=(n, ))
        Y_train = Y + np.sqrt(alpha) * wb
        Y_test = Y - wb / np.sqrt(alpha)
        if loss_fn == 'quadratic':
            Y_hat = g(X, Y_train)(X)
            err += np.mean((Y_hat - Y_test)**2 - (1 / alpha) * wb**2)
        elif loss_fn == 'logistic':
            beta_hat = g(X, Y_train)
            err += (np.sum(np.log(1 + np.exp(X @ beta_hat)) ) - np.sum(beta_hat * Y_test)) / X.shape[0]
        else:
            raise NotImplementedError('loss_fn {} not implemented'.format(loss_fn))
    return err / B

def cv_split(g, data, K, rng, loss_fn='quadratic'):
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
        if loss_fn == 'quadratic':
            Y_hat = g(X_train, Y_train)(X_test)
            err += np.mean((Y_hat - Y_test)**2)
        elif loss_fn == 'logistic':
            n_test = len(test_inds)
            s_train = X_train.T @ Y_train
            s_test = X_test.T @ Y_test
            beta_hat = g(X_train, s_train)
            err += np.sum(np.log(1 + np.exp(X_test @ beta_hat)) ) / n_test - np.sum(beta_hat * s_test) / n_test
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
    if np.isscalar(sigma):
        sigma_type = 'scalar'
    elif sigma.ndim == 2:
        sigma_type = 'matrix'
    else:
        raise ValueError('sigma must be a scalar or a matrix')
    
    Y_hat = g(X, Y)(X)
    err = np.mean((Y_hat - Y)**2)
    Y_boot = np.zeros((B, n))
    g_boot = np.zeros((B, n))
    for b in range(B):
        if sigma_type == 'scalar':
            wb = sigma * rng.normal(size=(n, ))
        else:
            wb = sigma @ rng.normal(size=(n, ))
        Y_b = Y + np.sqrt(alpha) * wb
        g_boot[b] = g(X, Y_b)(X)
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
    if np.isscalar(sigma):
        sigma_type = 'scalar'
    elif sigma.ndim == 2:
        sigma_type = 'matrix'
    else:
        raise ValueError('sigma must be a scalar or a matrix')
    
    Y_hat = g(X, Y)(X)
    err = np.mean((Y_hat - Y)**2)
    Y_boot = np.zeros((B, n))
    g_boot = np.zeros((B, n))
    for b in range(B):
        if sigma_type == 'scalar':
            wb = sigma * rng.normal(size=(n, ))
        else:
            wb = sigma @ rng.normal(size=(n, ))
        Y_b = Y + np.sqrt(alpha) * wb
        g_boot[b] = g(X, Y_b)(X)
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
    if np.isscalar(sigma):
        sigma_type = 'scalar'
    elif sigma.ndim == 2:
        sigma_type = 'matrix'
    else:
        raise ValueError('sigma must be a scalar or a matrix')
    
    Y_hat = g(X, Y)(X)
    err = np.mean((Y_hat - Y)**2)
    Y_boot = np.zeros((B, n))
    g_boot = np.zeros((B, n))
    for b in range(B):
        if sigma_type == 'scalar':
            wb = sigma * rng.normal(size=(n, ))
        else:
            wb = sigma @ rng.normal(size=(n, ))
        Y_b = Y + np.sqrt(alpha) * wb
        g_boot[b] = g(X, Y_b)(X)
        Y_boot[b] = Y_b
    Y_boot -= np.mean(Y_boot, axis=0)
    _var = np.sum(Y_boot**2, 0) / (B-1)
    _cov = np.sum(Y_boot * g_boot, 0) / (B-1) 
    cov_hat = _cov / _var
    return err + 2 * np.mean(cov_hat)
