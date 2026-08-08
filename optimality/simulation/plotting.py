"""Seaborn/matplotlib plotting for saved experiment results."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.abspath(".cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.algorithms import bootstrap as seaborn_bootstrap

from .core import (
    EXPERIMENT_HARD_THRESHOLD_ALPHA,
    EXPERIMENT_HARD_THRESHOLD_LAMBDA,
    EXPERIMENT_RIDGE_ALPHA,
    METHOD_ADJUSTED_ACV,
    METHOD_CONTROL_BENCHMARK,
    METHOD_LABELS,
    METHOD_NORMAL_ACV,
    METHOD_POSITIVE_RHO,
    METHOD_RADEMACHER_ACV,
    METHOD_TRUE_RISK,
)
from .results import (
    mse_observations,
    read_result,
    reducible_variance_observations,
    result_paths,
    risk_estimate_observations,
)


PANEL_BACKGROUND = "white"
LINE_WIDTH = 3.8
TRUE_RISK_LINE_WIDTH = 4.1
LEGEND_FONT_SIZE = 23
N_BOOTSTRAPS = 1000

_PALETTE = sns.color_palette("colorblind", 6)

METHOD_COLORS = {
    METHOD_NORMAL_ACV: _PALETTE[0],
    METHOD_ADJUSTED_ACV: "#7F7F7F",
    METHOD_CONTROL_BENCHMARK: _PALETTE[2],
    METHOD_POSITIVE_RHO: _PALETTE[3],
    METHOD_RADEMACHER_ACV: _PALETTE[4],
    METHOD_TRUE_RISK: "#D62728",
}

METHOD_LINESTYLES = {
    METHOD_NORMAL_ACV: "-",
    METHOD_ADJUSTED_ACV: "-.",
    METHOD_CONTROL_BENCHMARK: "--",
    METHOD_POSITIVE_RHO: ":",
    METHOD_RADEMACHER_ACV: "-.",
    METHOD_TRUE_RISK: ":",
}

LEGEND_METHOD_ORDER = [
    METHOD_CONTROL_BENCHMARK,
    METHOD_POSITIVE_RHO,
    METHOD_RADEMACHER_ACV,
    METHOD_NORMAL_ACV,
    METHOD_ADJUSTED_ACV,
    METHOD_TRUE_RISK,
]

MSE_LEGEND_METHOD_ORDER = [
    METHOD_RADEMACHER_ACV,
    METHOD_NORMAL_ACV,
    METHOD_ADJUSTED_ACV,
]


def seaborn_errorbar(errorbar: str):
    if errorbar == "ci_95":
        return ("ci", 95)
    if errorbar == "sd":
        return "sd"
    if errorbar == "percentile_95":
        return ("pi", 95)
    raise ValueError(f"Unknown errorbar mode: {errorbar}")


def blend_with_background(color: str, alpha: float, background: str = PANEL_BACKGROUND) -> Tuple[float, float, float]:
    foreground_rgb = np.array(mcolors.to_rgb(color))
    background_rgb = np.array(mcolors.to_rgb(background))
    return tuple(alpha * foreground_rgb + (1.0 - alpha) * background_rgb)


def _clip_for_log_axis(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    positive_values = []
    for column in columns:
        values = out[column].to_numpy()
        positive_values.extend(values[np.isfinite(values) & (values > 0.0)])

    floor = min(positive_values) * 0.5 if positive_values else np.finfo(float).tiny
    for column in columns:
        out[column] = out[column].clip(lower=floor)
    return out


def _finish_panel(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    xscale: str,
    yscale: str,
) -> None:
    grid_color = blend_with_background("#b0b0b0", alpha=0.28)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("")
    ax.grid(which="major", color=grid_color)
    ax.grid(which="minor", visible=False)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.2)


def _plot_observation_panel(
    ax: plt.Axes,
    observations: pd.DataFrame,
    methods: Sequence[str],
    title: str,
    xlabel: str,
    x_column: str = "tuning_value",
    true_risk: Optional[pd.DataFrame] = None,
    errorbar: str = "sd",
    xscale: str = "log",
    yscale: str = "log",
) -> None:
    plot_frame = (
        _clip_for_log_axis(observations, ["metric_value"])
        if yscale == "log"
        else observations.copy()
    )
    ax.set_facecolor(PANEL_BACKGROUND)

    for method in methods:
        method_frame = plot_frame[plot_frame["method"] == method].sort_values(x_column)
        if method_frame.empty:
            continue

        color = METHOD_COLORS[method]
        sns.lineplot(
            data=method_frame,
            x=x_column,
            y="metric_value",
            ax=ax,
            color=color,
            linestyle=METHOD_LINESTYLES[method],
            linewidth=LINE_WIDTH,
            label=METHOD_LABELS[method],
            estimator="mean",
            errorbar=seaborn_errorbar(errorbar),
            err_style="band",
            err_kws={"alpha": 0.18, "linewidth": 0},
        )

    if true_risk is not None:
        true_frame = (
            _clip_for_log_axis(true_risk, ["metric_value"])
            if yscale == "log"
            else true_risk.copy()
        ).sort_values(x_column)
        sns.lineplot(
            data=true_frame,
            x=x_column,
            y="metric_value",
            ax=ax,
            color=METHOD_COLORS[METHOD_TRUE_RISK],
            linestyle=METHOD_LINESTYLES[METHOD_TRUE_RISK],
            linewidth=TRUE_RISK_LINE_WIDTH,
            label=METHOD_LABELS[METHOD_TRUE_RISK],
            errorbar=None,
        )

    _finish_panel(ax, title, xlabel, xscale, yscale)


def _errorbar_summary(
    observations: pd.DataFrame,
    group_columns: Sequence[str],
    errorbar: str,
    value_column: str = "metric_value",
    unit_column: str = "response_rep",
) -> pd.DataFrame:
    lower_percentile = 2.5
    upper_percentile = 97.5
    rows = []

    for group_index, (group_key, group) in enumerate(
        observations.groupby(list(group_columns), sort=True)
    ):
        values = group[value_column].to_numpy(dtype=float)
        estimate = float(np.mean(values))

        if len(values) <= 1:
            lower = upper = np.nan
        elif errorbar == "ci_95":
            units = group[unit_column].to_numpy() if unit_column in group else None
            bootstrapped_means = seaborn_bootstrap(
                values,
                units=units,
                func="mean",
                n_boot=N_BOOTSTRAPS,
                seed=2026 + group_index,
            )
            lower, upper = np.percentile(
                bootstrapped_means,
                [lower_percentile, upper_percentile],
            )
        elif errorbar == "sd":
            spread = float(np.std(values, ddof=1))
            lower, upper = estimate - spread, estimate + spread
        elif errorbar == "percentile_95":
            lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
        else:
            raise ValueError(f"Unknown errorbar mode: {errorbar}")

        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, group_values))
        row[value_column] = estimate
        row[f"{value_column}min"] = float(lower)
        row[f"{value_column}max"] = float(upper)
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_summary_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: Sequence[str],
    title: str,
    xlabel: str,
    x_column: str = "tuning_value",
    xscale: str = "log",
    yscale: str = "log",
) -> None:
    columns = ["metric_value", "metric_valuemin", "metric_valuemax"]
    plot_frame = _clip_for_log_axis(summary, columns) if yscale == "log" else summary.copy()
    ax.set_facecolor(PANEL_BACKGROUND)

    for method in methods:
        method_frame = plot_frame[plot_frame["method"] == method].sort_values(x_column)
        if method_frame.empty:
            continue

        color = METHOD_COLORS[method]
        sns.lineplot(
            data=method_frame,
            x=x_column,
            y="metric_value",
            ax=ax,
            color=color,
            linestyle=METHOD_LINESTYLES[method],
            linewidth=LINE_WIDTH,
            label=METHOD_LABELS[method],
            estimator=None,
            errorbar=None,
        )
        ax.fill_between(
            method_frame[x_column].to_numpy(dtype=float),
            method_frame["metric_valuemin"].to_numpy(dtype=float),
            method_frame["metric_valuemax"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    _finish_panel(ax, title, xlabel, xscale, yscale)


def save_figure(fig: plt.Figure, output_stem: Path, formats: Sequence[str]) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        fig.savefig(output_stem.with_suffix(f".{file_format}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _add_shared_legend(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    method_order: Sequence[str],
) -> None:
    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)

    ordered_labels = [
        METHOD_LABELS[method]
        for method in method_order
        if METHOD_LABELS[method] in handles_by_label
    ]
    ordered_handles = [handles_by_label[label] for label in ordered_labels]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=len(ordered_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.8,
        columnspacing=1.2,
    )


def plot_three_panel_figure(
    results_dir: Path,
    output_stem: Path,
    formats: Sequence[str],
    errorbar: str = "sd",
) -> None:
    paths = result_paths(results_dir)
    ridge_alpha = read_result(paths[EXPERIMENT_RIDGE_ALPHA])
    hard_threshold_alpha = read_result(paths[EXPERIMENT_HARD_THRESHOLD_ALPHA])
    hard_threshold_lambda = read_result(paths[EXPERIMENT_HARD_THRESHOLD_LAMBDA])

    ridge_methods = [
        METHOD_NORMAL_ACV,
        METHOD_CONTROL_BENCHMARK,
        METHOD_POSITIVE_RHO,
        METHOD_RADEMACHER_ACV,
    ]
    hard_threshold_alpha_methods = [
        METHOD_NORMAL_ACV,
        METHOD_ADJUSTED_ACV,
        METHOD_CONTROL_BENCHMARK,
        METHOD_POSITIVE_RHO,
    ]
    hard_threshold_lambda_methods = [
        METHOD_NORMAL_ACV,
        METHOD_ADJUSTED_ACV,
        METHOD_CONTROL_BENCHMARK,
    ]

    ridge_reducible_variance = reducible_variance_observations(ridge_alpha, ridge_methods)
    hard_threshold_reducible_variance = reducible_variance_observations(
        hard_threshold_alpha,
        hard_threshold_alpha_methods,
    )
    ridge_reducible_variance["inverse_alpha"] = 1.0 / ridge_reducible_variance["tuning_value"]
    hard_threshold_reducible_variance["inverse_alpha"] = (
        1.0 / hard_threshold_reducible_variance["tuning_value"]
    )
    hard_threshold_lambda_estimates = risk_estimate_observations(
        hard_threshold_lambda,
        hard_threshold_lambda_methods,
    )
    true_risk = (
        hard_threshold_lambda.groupby("tuning_value", as_index=False)
        .agg(metric_value=("actual_risk", "first"))
        .assign(method=METHOD_TRUE_RISK)
    )

    sns.set_theme(
        style="ticks",
        context="talk",
        rc={
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 19,
            "axes.linewidth": 1.25,
            "lines.linewidth": LINE_WIDTH,
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), constrained_layout=False)

    _plot_observation_panel(
        axes[0],
        ridge_reducible_variance,
        ridge_methods,
        "(a) Ridge\nreducible variance",
        r"$1/\alpha$",
        x_column="inverse_alpha",
        errorbar=errorbar,
    )
    _plot_observation_panel(
        axes[1],
        hard_threshold_reducible_variance,
        hard_threshold_alpha_methods,
        "(b) Hard-thresholded ridge\nreducible variance",
        r"$1/\alpha$",
        x_column="inverse_alpha",
        errorbar=errorbar,
    )
    _plot_observation_panel(
        axes[2],
        hard_threshold_lambda_estimates,
        hard_threshold_lambda_methods,
        r"(c) Hard-thresholded ridge" "\n" r"estimated risk versus $\lambda$",
        r"$\lambda$",
        true_risk=true_risk,
        errorbar=errorbar,
    )

    _add_shared_legend(fig, axes, LEGEND_METHOD_ORDER)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), w_pad=2.0)
    save_figure(fig, output_stem, formats)


def plot_mse_two_panel_figure(
    results_dir: Path,
    output_stem: Path,
    formats: Sequence[str],
    errorbar: str = "ci_95",
) -> None:
    paths = result_paths(results_dir)
    ridge_alpha = read_result(paths[EXPERIMENT_RIDGE_ALPHA])
    hard_threshold_alpha = read_result(paths[EXPERIMENT_HARD_THRESHOLD_ALPHA])

    ridge_methods = [METHOD_NORMAL_ACV, METHOD_RADEMACHER_ACV]
    hard_threshold_methods = [METHOD_NORMAL_ACV, METHOD_ADJUSTED_ACV]

    ridge_mse = mse_observations(ridge_alpha, ridge_methods)
    hard_threshold_mse = mse_observations(hard_threshold_alpha, hard_threshold_methods)
    ridge_mse["inverse_alpha"] = 1.0 / ridge_mse["tuning_value"]
    hard_threshold_mse["inverse_alpha"] = 1.0 / hard_threshold_mse["tuning_value"]
    ridge_mse_summary = _errorbar_summary(
        ridge_mse,
        ["method", "inverse_alpha"],
        errorbar,
    )
    hard_threshold_mse_summary = _errorbar_summary(
        hard_threshold_mse,
        ["method", "inverse_alpha"],
        errorbar,
    )

    sns.set_theme(
        style="ticks",
        context="talk",
        rc={
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 19,
            "axes.linewidth": 1.25,
            "lines.linewidth": LINE_WIDTH,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), constrained_layout=False)

    _plot_summary_panel(
        axes[0],
        ridge_mse_summary,
        ridge_methods,
        "(a) Ridge\nMSE",
        r"$1/\alpha$",
        x_column="inverse_alpha",
        yscale="linear",
    )
    _plot_summary_panel(
        axes[1],
        hard_threshold_mse_summary,
        hard_threshold_methods,
        "(b) Hard-thresholded ridge\nMSE",
        r"$1/\alpha$",
        x_column="inverse_alpha",
        yscale="linear",
    )

    _add_shared_legend(fig, axes, MSE_LEGEND_METHOD_ORDER)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0), w_pad=2.0)
    save_figure(fig, output_stem, formats)
