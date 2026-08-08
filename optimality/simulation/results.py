"""Result storage and plotting observation helpers."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .core import (
    EXPERIMENT_HARD_THRESHOLD_ALPHA,
    EXPERIMENT_HARD_THRESHOLD_LAMBDA,
    EXPERIMENT_RIDGE_ALPHA,
    SimulationConfig,
)


RESULT_FILENAMES = {
    EXPERIMENT_RIDGE_ALPHA: "ridge_alpha.parquet",
    EXPERIMENT_HARD_THRESHOLD_ALPHA: "hard_threshold_alpha.parquet",
    EXPERIMENT_HARD_THRESHOLD_LAMBDA: "hard_threshold_lambda.parquet",
}


@contextmanager
def _suppress_native_stderr():
    """Suppress native-library stderr noise during Parquet IO."""

    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def result_paths(results_dir: Path) -> Dict[str, Path]:
    return {
        experiment: results_dir / filename
        for experiment, filename in RESULT_FILENAMES.items()
    }


def metadata_path(result_path: Path) -> Path:
    return result_path.with_suffix(".metadata.json")


def write_result(
    frame: pd.DataFrame,
    result_path: Path,
    experiment: str,
    config: SimulationConfig,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with _suppress_native_stderr():
        frame.to_parquet(result_path, index=False)
    metadata = {
        "experiment": experiment,
        "config": config.to_metadata(),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    metadata_path(result_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_result(result_path: Path) -> pd.DataFrame:
    with _suppress_native_stderr():
        return pd.read_parquet(result_path)


def reducible_variance_observations(frame: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    filtered = frame[frame["method"].isin(methods)]
    return (
        filtered.groupby(["method", "tuning_name", "tuning_value", "response_rep"], as_index=False)
        .agg(metric_value=("estimated_risk", lambda values: values.var(ddof=1)))
        .dropna(subset=["metric_value"])
    )


def risk_estimate_observations(frame: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    filtered = frame[frame["method"].isin(methods)].copy()
    filtered["metric_value"] = filtered["estimated_risk"]
    return filtered


def mse_observations(frame: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    filtered = frame[frame["method"].isin(methods)].copy()
    filtered["metric_value"] = (filtered["estimated_risk"] - filtered["actual_risk"]) ** 2
    return filtered
