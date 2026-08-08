#!/usr/bin/env python3
"""Create antithetic CV figures from saved results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import _bootstrap  # noqa: F401
from simulation.cli import add_pipeline_arguments
from simulation.plotting import plot_mse_two_panel_figure, plot_three_panel_figure


def _default_output_name(figure: str, errorbar: str) -> str:
    if figure == "three_panel":
        return (
            "combined_three_panel_percentile_interval"
            if errorbar == "percentile_95"
            else "combined_three_panel"
        )
    return (
        "mse_two_panel_sd"
        if errorbar == "sd"
        else "mse_two_panel_percentile_interval"
        if errorbar == "percentile_95"
        else "mse_two_panel"
    )


def _plot_figure(
    figure: str,
    results_dir: Path,
    figures_dir: Path,
    formats: Tuple[str, ...],
    errorbar: str,
    output_name: Optional[str],
) -> None:
    output_stem = figures_dir / (output_name or _default_output_name(figure, errorbar))
    if figure == "three_panel":
        plot_three_panel_figure(results_dir, output_stem, formats, errorbar=errorbar)
    else:
        plot_mse_two_panel_figure(results_dir, output_stem, formats, errorbar=errorbar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("all", "three_panel", "mse_two_panel"),
        default="all",
        help="Figure to create. Defaults to all figures.",
    )
    parser.add_argument(
        "--errorbar",
        choices=("ci_95", "sd", "percentile_95"),
        default=None,
        help="Error band. Defaults to sd for three_panel and ci_95 for mse_two_panel.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename stem inside --figures-dir. Only valid for one figure.",
    )
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    figures = (
        ("three_panel", "sd"),
        ("mse_two_panel", "ci_95"),
    )
    if args.figure != "all":
        default_errorbar = "sd" if args.figure == "three_panel" else "ci_95"
        figures = ((args.figure, default_errorbar),)
    elif args.output_name is not None:
        parser.error("--output-name can only be used with a single --figure")

    for figure, default_errorbar in figures:
        errorbar = args.errorbar or default_errorbar
        _plot_figure(
            figure,
            args.results_dir,
            args.figures_dir,
            args.formats,
            errorbar,
            args.output_name,
        )
        if not args.quiet:
            output_name = args.output_name or _default_output_name(figure, errorbar)
            print(f"Saved {args.figures_dir / output_name} ({', '.join(args.formats)})")


if __name__ == "__main__":
    main()
