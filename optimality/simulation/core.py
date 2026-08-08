"""Core simulation objects, estimators, and perturbation designs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np


METHOD_NORMAL_ACV = "normal_acv"
METHOD_ADJUSTED_ACV = "adjusted_acv"
METHOD_CONTROL_BENCHMARK = "control_benchmark"
METHOD_POSITIVE_RHO = "positive_rho"
METHOD_RADEMACHER_ACV = "rademacher_acv"
METHOD_ROTATION_ACV = "rotation_acv"
METHOD_TRUE_RISK = "true_risk"

METHOD_LABELS = {
    METHOD_NORMAL_ACV: "ACV (Normal)",
    METHOD_ADJUSTED_ACV: "ACV (Adjusted)",
    METHOD_CONTROL_BENCHMARK: r"$\rho=0$",
    METHOD_POSITIVE_RHO: r"$\rho=1/(K-1)$",
    METHOD_RADEMACHER_ACV: "ACV (Rademacher)",
    METHOD_ROTATION_ACV: "ACV (Rotation)",
    METHOD_TRUE_RISK: "True risk",
}

METHOD_SEED_IDS = {
    METHOD_NORMAL_ACV: 101,
    METHOD_ADJUSTED_ACV: 102,
    METHOD_CONTROL_BENCHMARK: 103,
    METHOD_POSITIVE_RHO: 104,
    METHOD_RADEMACHER_ACV: 105,
    METHOD_ROTATION_ACV: 106,
}

STREAM_DESIGN = 11
STREAM_RIDGE_ALPHA = 21
STREAM_HARD_THRESHOLD_ALPHA = 31
STREAM_HARD_THRESHOLD_LAMBDA = 41
STREAM_RISK_ESTIMATION = 51

MODEL_RIDGE = "ridge"
MODEL_HARD_THRESHOLD_RIDGE = "hard_thresholded_ridge"

EXPERIMENT_RIDGE_ALPHA = "ridge_alpha"
EXPERIMENT_HARD_THRESHOLD_ALPHA = "hard_threshold_alpha"
EXPERIMENT_HARD_THRESHOLD_LAMBDA = "hard_threshold_lambda"

TUNING_ALPHA = "alpha"
TUNING_LAMBDA = "lambda"

SeedPart = Union[int, np.integer]


@dataclass(frozen=True)
class SimulationConfig:
    n_response_reps: int = 100
    n_randomization_reps: int = 10
    n_observations: int = 200
    n_features: int = 50
    n_signal_features: int = 10
    signal_to_noise_ratio: float = 2.0
    seed: int = 2026
    n_alpha_values: int = 10
    alpha_min: float = 0.001
    alpha_max: float = 0.1
    fixed_lambda: float = 10.0
    alpha_for_lambda_grid: float = 0.01
    n_lambda_values: int = 20
    lambda_min_power: float = -1.0
    lambda_max_power: float = 2.5
    threshold_multiplier: float = 1.65
    n_threshold_risk_train_reps: int = 100

    @property
    def alpha_grid(self) -> np.ndarray:
        return np.linspace(self.alpha_max, self.alpha_min, self.n_alpha_values)

    @property
    def lambda_grid(self) -> np.ndarray:
        powers = np.linspace(self.lambda_min_power, self.lambda_max_power, self.n_lambda_values)
        return 10.0**powers

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionProblem:
    design: np.ndarray
    coefficients: np.ndarray
    mean_response: np.ndarray
    noise_sd: float

    @property
    def n_observations(self) -> int:
        return self.design.shape[0]

    @property
    def n_features(self) -> int:
        return self.design.shape[1]


def make_rng(*seed_parts: SeedPart) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(part) for part in seed_parts]))


def generate_problem(config: SimulationConfig, rng: np.random.Generator) -> RegressionProblem:
    design = rng.normal(size=(config.n_observations, config.n_features))
    signal_indices = rng.choice(
        config.n_features,
        size=config.n_signal_features,
        replace=False,
    )

    coefficients = np.zeros(config.n_features)
    coefficients[signal_indices] = rng.uniform(-1.0, 1.0, size=config.n_signal_features)
    mean_response = design @ coefficients

    population_variance = float(np.var(mean_response, ddof=0))
    noise_sd = float(np.sqrt(population_variance / config.signal_to_noise_ratio))
    return RegressionProblem(design, coefficients, mean_response, noise_sd)


def sample_response(problem: RegressionProblem, rng: np.random.Generator) -> np.ndarray:
    noise = problem.noise_sd * rng.normal(size=problem.n_observations)
    return problem.mean_response + noise


def ridge_operator(design: np.ndarray, lambda_value: float) -> np.ndarray:
    gram = design.T @ design
    penalty = lambda_value * np.eye(design.shape[1])
    return np.linalg.solve(gram + penalty, design.T)


def ridge_hat_matrix(design: np.ndarray, lambda_value: float) -> np.ndarray:
    return design @ ridge_operator(design, lambda_value)


def hard_threshold(ridge_coefficients: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.where(np.abs(ridge_coefficients) > threshold, ridge_coefficients, 0.0)


def hard_threshold_threshold(
    ridge_op: np.ndarray,
    noise_sd: float,
    threshold_multiplier: float,
) -> np.ndarray:
    coefficient_sd = noise_sd * np.sqrt(np.sum(ridge_op**2, axis=1))
    return threshold_multiplier * coefficient_sd


def hard_threshold_predictions(
    design: np.ndarray,
    ridge_op: np.ndarray,
    responses: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    response_matrix = np.atleast_2d(responses)
    ridge_coefficients = response_matrix @ ridge_op.T
    thresholded_coefficients = hard_threshold(ridge_coefficients, threshold[None, :])
    predictions = thresholded_coefficients @ design.T
    if responses.ndim == 1:
        return predictions[0]
    return predictions


def ridge_actual_risk(problem: RegressionProblem, hat_matrix: np.ndarray) -> float:
    residual_operator = np.eye(problem.n_observations) - hat_matrix
    bias_squared = float(
        problem.mean_response @ residual_operator.T @ residual_operator @ problem.mean_response
    )
    prediction_variance = float(problem.noise_sd**2 * np.sum(hat_matrix**2))
    irreducible_variance = problem.n_observations * problem.noise_sd**2
    return bias_squared + prediction_variance + irreducible_variance


def conditional_prediction_risk(problem: RegressionProblem, predictions: np.ndarray) -> float:
    squared_bias = np.sum((problem.mean_response - predictions) ** 2)
    irreducible_variance = problem.n_observations * problem.noise_sd**2
    return float(squared_bias + irreducible_variance)


def estimate_hard_threshold_actual_risk(
    problem: RegressionProblem,
    ridge_op: np.ndarray,
    threshold: np.ndarray,
    n_train_reps: int,
    seed_parts: Sequence[int],
) -> float:
    risks = np.empty(n_train_reps)
    for train_rep in range(n_train_reps):
        train_rng = make_rng(*seed_parts, train_rep)
        train_response = sample_response(problem, train_rng)
        predictions = hard_threshold_predictions(
            problem.design,
            ridge_op,
            train_response,
            threshold,
        )
        risks[train_rep] = conditional_prediction_risk(problem, predictions)

    return float(np.mean(risks))


def equicorrelated_normal(
    rng: np.random.Generator,
    n_folds: int,
    n_observations: int,
    rho: float,
) -> np.ndarray:
    lower_bound = -1.0 / (n_folds - 1)
    if rho < lower_bound - 1e-12 or rho > 1.0 + 1e-12:
        raise ValueError(f"rho={rho} outside valid range [{lower_bound}, 1].")

    raw_noise = rng.normal(size=(n_folds, n_observations))
    column_mean = raw_noise.mean(axis=0, keepdims=True)
    centered_noise = raw_noise - column_mean
    shared_scale = math.sqrt(max(0.0, 1.0 + (n_folds - 1) * rho))
    centered_scale = math.sqrt(max(0.0, 1.0 - rho))
    return shared_scale * column_mean + centered_scale * centered_noise


def rademacher_antithetic(
    rng: np.random.Generator,
    n_folds: int,
    n_observations: int,
) -> np.ndarray:
    if n_folds % 2 != 0:
        raise ValueError("Rademacher antithetic perturbations require an even n_folds.")

    base_noise = rng.normal(size=n_observations)
    signs = np.array([1.0] * (n_folds // 2) + [-1.0] * (n_folds // 2))
    signs = rng.permutation(signs)
    return signs[:, None] * base_noise[None, :]


def rotation_antithetic(
    rng: np.random.Generator,
    n_folds: int,
    n_observations: int,
) -> np.ndarray:
    if n_observations % 2 != 0:
        raise ValueError("Rotation perturbations require an even number of observations.")

    base_noise = rng.normal(size=n_observations).reshape(n_observations // 2, 2)
    angles = rng.permutation(np.arange(1, n_folds + 1)) * (2.0 * math.pi / n_folds)

    perturbations = np.empty((n_folds, n_observations))
    for row, angle in enumerate(angles):
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        perturbations[row, :] = (base_noise @ rotation.T).reshape(n_observations)
    return perturbations


def normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / math.sqrt(2.0 * math.pi)


def quadratic_cv_errors(
    predictions: np.ndarray,
    observed_response: np.ndarray,
    perturbations: np.ndarray,
    alpha: float,
) -> np.ndarray:
    test_response = observed_response[None, :] - perturbations / math.sqrt(alpha)
    return np.sum((predictions - test_response) ** 2 - perturbations**2 / alpha, axis=1)


def ridge_cv_estimate(
    hat_matrix: np.ndarray,
    observed_response: np.ndarray,
    noise_sd: float,
    alpha: float,
    standard_perturbations: np.ndarray,
) -> float:
    perturbations = noise_sd * standard_perturbations
    train_response = observed_response[None, :] + math.sqrt(alpha) * perturbations
    predictions = train_response @ hat_matrix.T
    return float(
        np.mean(quadratic_cv_errors(predictions, observed_response, perturbations, alpha))
    )


def hard_threshold_cv_estimate(
    design: np.ndarray,
    ridge_op: np.ndarray,
    observed_response: np.ndarray,
    noise_sd: float,
    alpha: float,
    threshold: np.ndarray,
    standard_perturbations: np.ndarray,
) -> float:
    perturbations = noise_sd * standard_perturbations
    train_response = observed_response[None, :] + math.sqrt(alpha) * perturbations
    predictions = hard_threshold_predictions(design, ridge_op, train_response, threshold)
    errors = quadratic_cv_errors(predictions, observed_response, perturbations, alpha)
    return float(np.mean(errors))


def adjusted_hard_threshold_cv_estimate(
    design: np.ndarray,
    ridge_op: np.ndarray,
    observed_response: np.ndarray,
    noise_sd: float,
    alpha: float,
    threshold: np.ndarray,
    standard_perturbations: np.ndarray,
) -> float:
    perturbations = noise_sd * standard_perturbations
    sqrt_alpha = math.sqrt(alpha)

    ridge_coefficients = ridge_op @ observed_response
    row_norms = np.sqrt(np.sum(ridge_op**2, axis=1))
    ridge_design_diag = np.diag(ridge_op @ design)

    safe_norms = np.where(row_norms == 0.0, np.nan, row_norms)
    phi_plus = normal_pdf((threshold - ridge_coefficients) / (noise_sd * sqrt_alpha * safe_norms))
    phi_minus = normal_pdf((threshold + ridge_coefficients) / (noise_sd * sqrt_alpha * safe_norms))
    analytic_terms = (
        noise_sd
        * np.divide(
            ridge_coefficients * ridge_design_diag,
            safe_norms,
            out=np.zeros_like(ridge_coefficients),
            where=np.isfinite(safe_norms) & (safe_norms != 0.0),
        )
        * (phi_plus - phi_minus)
    )
    analytic_sum = float(np.nansum(analytic_terms))

    train_response = observed_response[None, :] + sqrt_alpha * perturbations
    predictions = hard_threshold_predictions(design, ridge_op, train_response, threshold)
    base_errors = quadratic_cv_errors(predictions, observed_response, perturbations, alpha)

    train_coefficients = train_response @ ridge_op.T
    active_coefficients = np.abs(train_coefficients) > threshold[None, :]
    perturbation_design_products = perturbations @ design
    stochastic_sum = np.sum(
        (ridge_coefficients[None, :] * perturbation_design_products) * active_coefficients,
        axis=1,
    )

    adjusted_errors = base_errors + (2.0 / sqrt_alpha) * (analytic_sum - stochastic_sum)
    return float(np.mean(adjusted_errors))
