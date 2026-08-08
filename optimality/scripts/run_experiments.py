#!/usr/bin/env python3
"""Run one or all antithetic CV simulation experiments."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from simulation.cli import add_pipeline_arguments, add_simulation_arguments, simulation_config_from_args
from simulation.experiments import EXPERIMENT_RUNNERS
from simulation.results import result_paths, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all", *EXPERIMENT_RUNNERS),
        default="all",
        help="Experiment to run. Defaults to all experiments.",
    )
    add_simulation_arguments(parser)
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    config = simulation_config_from_args(args)
    paths = result_paths(args.results_dir)
    jobs = (
        EXPERIMENT_RUNNERS.items()
        if args.experiment == "all"
        else [(args.experiment, EXPERIMENT_RUNNERS[args.experiment])]
    )

    for experiment_name, runner in jobs:
        frame = runner(config)
        write_result(frame, paths[experiment_name], experiment_name, config)
        if not args.quiet:
            print(f"Saved {paths[experiment_name]} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
