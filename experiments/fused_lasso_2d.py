import numpy as np
import argparse
from PIL import Image
import jax
import jax.numpy as jnp
import optax
from optax import adam

from prediction_algorithms import mlp
from risk_estimators import antithetic_cv, coupled_bootstrap, cv_split

def fused_lasso(noisy_image, lbd, lr=1e-3, max_iter=100):
    """
    2d fused lasso implemented by Jax

    Install jax and optax by
    ```
    pip install jax optax
    ```

    Parameters
    ----------
    noisy_image : ndarray
        The noisy image, shape (rows, cols)
    lbd : float
        The regularization parameter for the fused term
    lr : float
        The learning rate for Adam
    max_iter : int
        Number of iterations for Adam

    Returns
    -------
    ndarray
        The denoised image
    """
    rows, cols = noisy_image.shape
    n = rows * cols
    y = noisy_image.flatten()

    @jax.jit
    def objective(x):
        x = x.reshape(rows, cols)
        diff_x = -jnp.diff(x, axis=0)
        diff_y = -jnp.diff(x, axis=1)
        fused_term = lbd * (jnp.sum(jnp.abs(diff_x)) + jnp.sum(jnp.abs(diff_y)))
        return (0.5 * jnp.sum((x.flatten() - y) ** 2) + fused_term) / n

    @jax.jit
    def gradient(x):
        x = x.reshape(rows, cols)
        grad = x - noisy_image
        diff_x = -jnp.diff(x, axis=0)
        diff_y = -jnp.diff(x, axis=1)
        grad = grad.at[:-1, :].add(lbd * jnp.sign(diff_x))
        grad = grad.at[1:, :].add(-lbd * jnp.sign(diff_x))
        grad = grad.at[:, :-1].add(lbd * jnp.sign(diff_y))
        grad = grad.at[:, 1:].add(-lbd * jnp.sign(diff_y))

        return grad.flatten() / n

    optimizer = adam(lr)
    x0 = noisy_image.flatten()
    opt_state = optimizer.init(x0)

    @jax.jit
    def update_step(carry, _):
        x, opt_state = carry
        grads = gradient(x)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = optax.apply_updates(x, updates)
        return (x, opt_state), objective(x)

    (x_final, final_state), losses = jax.lax.scan(update_step, (x0, opt_state), None, length=max_iter)
    return lambda _: x_final.reshape(rows, cols)

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
