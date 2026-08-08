"""Experiment runners that return tidy raw simulation estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .core import (
    EXPERIMENT_HARD_THRESHOLD_ALPHA,
    EXPERIMENT_HARD_THRESHOLD_LAMBDA,
    EXPERIMENT_RIDGE_ALPHA,
    METHOD_ADJUSTED_ACV,
    METHOD_CONTROL_BENCHMARK,
    METHOD_NORMAL_ACV,
    METHOD_POSITIVE_RHO,
    METHOD_RADEMACHER_ACV,
    METHOD_ROTATION_ACV,
    METHOD_SEED_IDS,
    MODEL_HARD_THRESHOLD_RIDGE,
    MODEL_RIDGE,
    STREAM_DESIGN,
    STREAM_HARD_THRESHOLD_ALPHA,
    STREAM_HARD_THRESHOLD_LAMBDA,
    STREAM_RIDGE_ALPHA,
    STREAM_RISK_ESTIMATION,
    TUNING_ALPHA,
    TUNING_LAMBDA,
    RegressionProblem,
    SimulationConfig,
    adjusted_hard_threshold_cv_estimate,
    equicorrelated_normal,
    estimate_hard_threshold_actual_risk,
    generate_problem,
    hard_threshold_cv_estimate,
    hard_threshold_threshold,
    make_rng,
    rademacher_antithetic,
    ridge_actual_risk,
    ridge_cv_estimate,
    ridge_hat_matrix,
    ridge_operator,
    rotation_antithetic,
    sample_response,
)


@dataclass(frozen=True)
class PerturbationSpec:
    method: str
    n_folds: int
    generator: Callable[[np.random.Generator, int, int], np.ndarray]


RiskEstimator = Callable[[PerturbationSpec, int, np.ndarray, float, np.ndarray], float]


def _problem(config: SimulationConfig) -> RegressionProblem:
    return generate_problem(config, make_rng(config.seed, STREAM_DESIGN))


def _equicorrelation_spec(method: str, n_folds: int, rho: float) -> PerturbationSpec:
    return PerturbationSpec(
        method=method,
        n_folds=n_folds,
        generator=lambda rng, k, n: equicorrelated_normal(rng, k, n, rho),
    )


def _ridge_alpha_specs() -> List[PerturbationSpec]:
    return [
        _equicorrelation_spec(METHOD_NORMAL_ACV, 6, -1.0 / 5.0),
        _equicorrelation_spec(METHOD_CONTROL_BENCHMARK, 6, 0.0),
        _equicorrelation_spec(METHOD_POSITIVE_RHO, 6, 1.0 / 5.0),
        PerturbationSpec(METHOD_RADEMACHER_ACV, 6, rademacher_antithetic),
        PerturbationSpec(METHOD_ROTATION_ACV, 6, rotation_antithetic),
    ]


def _hard_threshold_specs(include_positive_rho: bool) -> List[PerturbationSpec]:
    n_folds = 6
    specs = [
        _equicorrelation_spec(METHOD_NORMAL_ACV, n_folds, -1.0 / (n_folds - 1)),
        _equicorrelation_spec(METHOD_ADJUSTED_ACV, n_folds, -1.0 / (n_folds - 1)),
        _equicorrelation_spec(METHOD_CONTROL_BENCHMARK, n_folds, 0.0),
    ]
    if include_positive_rho:
        specs.append(_equicorrelation_spec(METHOD_POSITIVE_RHO, n_folds, 1.0 / (n_folds - 1)))
    return specs


def _method_rng(
    config: SimulationConfig,
    stream_id: int,
    response_rep: int,
    randomization_rep: int,
    method: str,
    tuning_index: int,
) -> np.random.Generator:
    return make_rng(
        config.seed,
        stream_id,
        response_rep,
        randomization_rep,
        METHOD_SEED_IDS[method],
        tuning_index,
    )


def _result_row(
    experiment: str,
    model: str,
    method: str,
    tuning_name: str,
    tuning_value: float,
    alpha: float,
    lambda_value: float,
    response_rep: int,
    randomization_rep: int,
    estimated_risk: float,
    actual_risk: float,
) -> Dict[str, object]:
    return {
        "experiment": experiment,
        "model": model,
        "method": method,
        "tuning_name": tuning_name,
        "tuning_value": tuning_value,
        "alpha": alpha,
        "lambda_value": lambda_value,
        "response_rep": response_rep,
        "randomization_rep": randomization_rep,
        "estimated_risk": estimated_risk,
        "actual_risk": actual_risk,
    }


def _run_grid(
    *,
    config: SimulationConfig,
    problem: RegressionProblem,
    experiment: str,
    model: str,
    tuning_name: str,
    tuning_values: Sequence[float],
    alpha_values: Sequence[float],
    lambda_values: Sequence[float],
    actual_risks: Sequence[float],
    stream_id: int,
    perturbation_specs: Iterable[PerturbationSpec],
    estimate_risk: RiskEstimator,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for response_rep in range(config.n_response_reps):
        response_rng = make_rng(config.seed, stream_id, response_rep)
        observed_response = sample_response(problem, response_rng)

        for randomization_rep in range(config.n_randomization_reps):
            for spec in perturbation_specs:
                for tuning_index, tuning_value in enumerate(tuning_values):
                    alpha = float(alpha_values[tuning_index])
                    lambda_value = float(lambda_values[tuning_index])
                    perturbation_rng = _method_rng(
                        config,
                        stream_id,
                        response_rep,
                        randomization_rep,
                        spec.method,
                        tuning_index,
                    )
                    standard_perturbations = spec.generator(
                        perturbation_rng,
                        spec.n_folds,
                        problem.n_observations,
                    )
                    rows.append(
                        _result_row(
                            experiment,
                            model,
                            spec.method,
                            tuning_name,
                            float(tuning_value),
                            alpha,
                            lambda_value,
                            response_rep,
                            randomization_rep,
                            estimate_risk(
                                spec,
                                tuning_index,
                                observed_response,
                                alpha,
                                standard_perturbations,
                            ),
                            float(actual_risks[tuning_index]),
                        )
                    )

    return pd.DataFrame(rows)


def run_ridge_alpha_experiment(config: SimulationConfig) -> pd.DataFrame:
    problem = _problem(config)
    alpha_grid = tuple(float(value) for value in config.alpha_grid)
    lambda_value = float(config.fixed_lambda)
    hat_matrix = ridge_hat_matrix(problem.design, lambda_value)
    actual_risk = ridge_actual_risk(problem, hat_matrix)

    def estimate(
        spec: PerturbationSpec,
        tuning_index: int,
        observed_response: np.ndarray,
        alpha: float,
        standard_perturbations: np.ndarray,
    ) -> float:
        return ridge_cv_estimate(
            hat_matrix,
            observed_response,
            problem.noise_sd,
            alpha,
            standard_perturbations,
        )

    return _run_grid(
        config=config,
        problem=problem,
        experiment=EXPERIMENT_RIDGE_ALPHA,
        model=MODEL_RIDGE,
        tuning_name=TUNING_ALPHA,
        tuning_values=alpha_grid,
        alpha_values=alpha_grid,
        lambda_values=[lambda_value] * len(alpha_grid),
        actual_risks=[actual_risk] * len(alpha_grid),
        stream_id=STREAM_RIDGE_ALPHA,
        perturbation_specs=_ridge_alpha_specs(),
        estimate_risk=estimate,
    )


def run_hard_threshold_alpha_experiment(config: SimulationConfig) -> pd.DataFrame:
    problem = _problem(config)
    alpha_grid = tuple(float(value) for value in config.alpha_grid)
    lambda_value = float(config.fixed_lambda)
    ridge_op = ridge_operator(problem.design, lambda_value)
    threshold = hard_threshold_threshold(
        ridge_op,
        problem.noise_sd,
        config.threshold_multiplier,
    )
    actual_risk = estimate_hard_threshold_actual_risk(
        problem,
        ridge_op,
        threshold,
        config.n_threshold_risk_train_reps,
        [config.seed, STREAM_HARD_THRESHOLD_ALPHA, STREAM_RISK_ESTIMATION],
    )

    def estimate(
        spec: PerturbationSpec,
        tuning_index: int,
        observed_response: np.ndarray,
        alpha: float,
        standard_perturbations: np.ndarray,
    ) -> float:
        if spec.method == METHOD_ADJUSTED_ACV:
            return adjusted_hard_threshold_cv_estimate(
                problem.design,
                ridge_op,
                observed_response,
                problem.noise_sd,
                alpha,
                threshold,
                standard_perturbations,
            )
        return hard_threshold_cv_estimate(
            problem.design,
            ridge_op,
            observed_response,
            problem.noise_sd,
            alpha,
            threshold,
            standard_perturbations,
        )

    return _run_grid(
        config=config,
        problem=problem,
        experiment=EXPERIMENT_HARD_THRESHOLD_ALPHA,
        model=MODEL_HARD_THRESHOLD_RIDGE,
        tuning_name=TUNING_ALPHA,
        tuning_values=alpha_grid,
        alpha_values=alpha_grid,
        lambda_values=[lambda_value] * len(alpha_grid),
        actual_risks=[actual_risk] * len(alpha_grid),
        stream_id=STREAM_HARD_THRESHOLD_ALPHA,
        perturbation_specs=_hard_threshold_specs(include_positive_rho=True),
        estimate_risk=estimate,
    )


def run_hard_threshold_lambda_experiment(config: SimulationConfig) -> pd.DataFrame:
    problem = _problem(config)
    lambda_grid = tuple(float(value) for value in config.lambda_grid)
    alpha = float(config.alpha_for_lambda_grid)
    ridge_ops = [ridge_operator(problem.design, lambda_value) for lambda_value in lambda_grid]
    thresholds = [
        hard_threshold_threshold(ridge_op, problem.noise_sd, config.threshold_multiplier)
        for ridge_op in ridge_ops
    ]
    actual_risks = [
        estimate_hard_threshold_actual_risk(
            problem,
            ridge_op,
            threshold,
            config.n_threshold_risk_train_reps,
            [config.seed, STREAM_HARD_THRESHOLD_LAMBDA, STREAM_RISK_ESTIMATION, lambda_index],
        )
        for lambda_index, (ridge_op, threshold) in enumerate(zip(ridge_ops, thresholds))
    ]

    def estimate(
        spec: PerturbationSpec,
        tuning_index: int,
        observed_response: np.ndarray,
        alpha_value: float,
        standard_perturbations: np.ndarray,
    ) -> float:
        ridge_op = ridge_ops[tuning_index]
        threshold = thresholds[tuning_index]
        if spec.method == METHOD_ADJUSTED_ACV:
            return adjusted_hard_threshold_cv_estimate(
                problem.design,
                ridge_op,
                observed_response,
                problem.noise_sd,
                alpha_value,
                threshold,
                standard_perturbations,
            )
        return hard_threshold_cv_estimate(
            problem.design,
            ridge_op,
            observed_response,
            problem.noise_sd,
            alpha_value,
            threshold,
            standard_perturbations,
        )

    return _run_grid(
        config=config,
        problem=problem,
        experiment=EXPERIMENT_HARD_THRESHOLD_LAMBDA,
        model=MODEL_HARD_THRESHOLD_RIDGE,
        tuning_name=TUNING_LAMBDA,
        tuning_values=lambda_grid,
        alpha_values=[alpha] * len(lambda_grid),
        lambda_values=lambda_grid,
        actual_risks=actual_risks,
        stream_id=STREAM_HARD_THRESHOLD_LAMBDA,
        perturbation_specs=_hard_threshold_specs(include_positive_rho=False),
        estimate_risk=estimate,
    )


EXPERIMENT_RUNNERS = {
    EXPERIMENT_RIDGE_ALPHA: run_ridge_alpha_experiment,
    EXPERIMENT_HARD_THRESHOLD_ALPHA: run_hard_threshold_alpha_experiment,
    EXPERIMENT_HARD_THRESHOLD_LAMBDA: run_hard_threshold_lambda_experiment,
}
