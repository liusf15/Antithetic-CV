"""Shared command-line helpers for experiment scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .core import SimulationConfig


@dataclass(frozen=True)
class PipelineConfig:
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")
    figure_formats: Tuple[str, ...] = ("pdf",)


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(item.strip().lower().lstrip(".") for item in value.split(",") if item.strip())
    if not formats:
        raise argparse.ArgumentTypeError("At least one format is required.")
    return formats


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-response-reps", type=int, default=SimulationConfig.n_response_reps)
    parser.add_argument(
        "--n-randomization-reps",
        type=int,
        default=SimulationConfig.n_randomization_reps,
    )
    parser.add_argument("--n-observations", type=int, default=SimulationConfig.n_observations)
    parser.add_argument("--n-features", type=int, default=SimulationConfig.n_features)
    parser.add_argument(
        "--n-signal-features",
        type=int,
        default=SimulationConfig.n_signal_features,
    )
    parser.add_argument(
        "--signal-to-noise-ratio",
        type=float,
        default=SimulationConfig.signal_to_noise_ratio,
    )
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    parser.add_argument("--n-alpha-values", type=int, default=SimulationConfig.n_alpha_values)
    parser.add_argument("--alpha-min", type=float, default=SimulationConfig.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=SimulationConfig.alpha_max)
    parser.add_argument("--fixed-lambda", type=float, default=SimulationConfig.fixed_lambda)
    parser.add_argument(
        "--alpha-for-lambda-grid",
        type=float,
        default=SimulationConfig.alpha_for_lambda_grid,
    )
    parser.add_argument("--n-lambda-values", type=int, default=SimulationConfig.n_lambda_values)
    parser.add_argument(
        "--lambda-min-power",
        type=float,
        default=SimulationConfig.lambda_min_power,
    )
    parser.add_argument(
        "--lambda-max-power",
        type=float,
        default=SimulationConfig.lambda_max_power,
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=SimulationConfig.threshold_multiplier,
    )
    parser.add_argument(
        "--n-threshold-risk-train-reps",
        type=int,
        default=SimulationConfig.n_threshold_risk_train_reps,
    )


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=PipelineConfig.results_dir)
    parser.add_argument("--figures-dir", type=Path, default=PipelineConfig.figures_dir)
    parser.add_argument("--formats", type=parse_formats, default=PipelineConfig.figure_formats)
    parser.add_argument("--quiet", action="store_true")


def simulation_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        n_response_reps=args.n_response_reps,
        n_randomization_reps=args.n_randomization_reps,
        n_observations=args.n_observations,
        n_features=args.n_features,
        n_signal_features=args.n_signal_features,
        signal_to_noise_ratio=args.signal_to_noise_ratio,
        seed=args.seed,
        n_alpha_values=args.n_alpha_values,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        fixed_lambda=args.fixed_lambda,
        alpha_for_lambda_grid=args.alpha_for_lambda_grid,
        n_lambda_values=args.n_lambda_values,
        lambda_min_power=args.lambda_min_power,
        lambda_max_power=args.lambda_max_power,
        threshold_multiplier=args.threshold_multiplier,
        n_threshold_risk_train_reps=args.n_threshold_risk_train_reps,
    )
